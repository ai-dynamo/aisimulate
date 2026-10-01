#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute annotated README blocks; keep installation profiles isolated."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import venv
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("source", "macos", "development", "release", "dynamo")


def read_blocks(readme: str, manifest: dict) -> dict[str, str]:
    fences = list(re.finditer(r"^```(bash|yaml)\n(.*?)^```", readme, re.M | re.S))
    blocks = {}
    for fence in fences:
        marker = re.search(r"<!-- readme-check: ([a-z0-9-]+) -->\n$", readme[: fence.start()])
        if not marker:
            raise ValueError("Every bash/yaml block needs a readme-check ID")
        name = marker[1]
        if name in blocks or name not in manifest:
            raise ValueError(f"Duplicate or unregistered block: {name}")
        entry = manifest[name]
        if (fence[1] == "yaml") != ("file" in entry):
            raise ValueError(f"Wrong block type: {name}")
        blocks[name] = fence[2]
    if blocks.keys() != manifest.keys():
        raise ValueError(f"Manifest drift: {manifest.keys() - blocks.keys()}")
    for name, entry in manifest.items():
        if "file" in entry:
            if Path(entry["file"]).name != entry["file"]:
                raise ValueError(f"Unsafe output path: {name}")
            yaml.safe_load(blocks[name])
        else:
            if not entry["profiles"] or set(entry["profiles"]) - set(PROFILES):
                raise ValueError(f"Invalid profiles: {name}")
            for dependency in entry["needs"]:
                if dependency not in list(blocks)[: list(blocks).index(name)] and dependency != "source-install":
                    raise ValueError(f"Dependency must run first: {name}: {dependency}")
    return blocks


def run_shell(command: str, cwd: Path, env: dict, log: Path, timeout: float) -> tuple[int, bool]:
    """Track descendants too: sweeper workers can create their own sessions."""
    descendants: set[psutil.Process] = set()
    timed_out = False
    with log.open("w") as output:
        output.write(command + "\n\n")
        output.flush()
        process = subprocess.Popen(
            ["bash", "-euo", "pipefail", "-c", command],
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        parent = psutil.Process(process.pid)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                try:
                    descendants.update(parent.children(recursive=True))
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(0.1)
        finally:
            # Also remove leaked workers after an otherwise successful command.
            for child in descendants:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            psutil.wait_procs(list(descendants), timeout=5)
    return process.returncode, timed_out


def assert_outputs(entry: dict, cwd: Path, log: Path) -> None:
    if path := entry.get("prediction"):
        report = json.loads((cwd / path).read_text())
        summary = report.get("summary", report)
        if summary.get("completed_requests", 0) <= 0:
            raise ValueError("Prediction completed no requests")
    if path := entry.get("recommendations"):
        candidates = sorted((cwd / path).glob("*.yaml"))
        if not candidates or candidates[0].name != "0001.yaml":
            raise ValueError("Recommendation produced no ranked candidate")
        for candidate in candidates:
            config = yaml.safe_load(candidate.read_text())
            # Validate through the public schema and then predict 0001 in the next block.
            if not isinstance(config, dict) or not isinstance(config.get("engine"), dict):
                raise ValueError(f"Invalid candidate: {candidate.name}")

            def concrete(value):
                if isinstance(value, dict):
                    if set(value) & {"choices", "range", "preset"}:
                        raise ValueError(f"Unresolved search domain: {candidate.name}")
                    for item in value.values():
                        concrete(item)
                elif isinstance(value, list):
                    for item in value:
                        concrete(item)

            concrete(config["engine"])
    for expected in entry.get("stdout_contains", []):
        # The first part of the log is the command; only inspect actual output.
        if expected not in log.read_text().split("\n\n", 1)[1]:
            raise ValueError(f"Missing expected output: {expected}")


def run_profile(profile: str, workspace: Path, output: Path, blocks: dict, manifest: dict) -> dict:
    workspace.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    env["AISIMULATE_REF"] = sha
    # Clone exactly the checked-out tree, including commits on a not-yet-merged PR.
    # The README command itself remains unchanged; no network fetch of mutable main.
    env.update(
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0=f"url.{ROOT}.insteadOf",
        GIT_CONFIG_VALUE_0="https://github.com/ai-dynamo/aisimulate.git",
    )
    source = profile in {"source", "macos", "development"}
    cwd = workspace / "aisimulate" if source else workspace
    environment = cwd / ".venv"
    if not source:
        venv.create(environment, with_pip=True)
    env["PATH"] = str(environment / "bin") + os.pathsep + env["PATH"]
    env["VIRTUAL_ENV"] = str(environment)
    results = []
    selected = [name for name, entry in manifest.items() if profile in entry.get("profiles", [])]
    if source:
        selected.remove("source-install")
        selected.insert(0, "source-install")
    for name in selected:
        entry = manifest[name]
        failed = [
            dependency
            for dependency in entry["needs"]
            if not any(row["id"] == dependency and row["status"] == "passed" for row in results)
        ]
        row = {
            "id": name,
            "status": "blocked",
            "seconds": 0,
            "reason": "prerequisite: " + ", ".join(failed),
        }
        results.append(row)
        if not failed:
            for config, config_entry in manifest.items():
                if "file" in config_entry and cwd.exists():
                    (cwd / config_entry["file"]).write_text(blocks[config])
            env["PIP_REPORT"] = str(output / f"{name}-install.json")
            started = time.monotonic()
            log = output / f"{name}.log"
            try:
                code, timed_out = run_shell(
                    blocks[name],
                    workspace if name == "source-install" else cwd,
                    env,
                    log,
                    entry["timeout"],
                )
                row.update(
                    status="timeout" if timed_out else "failed" if code else "passed",
                    returncode=code,
                    reason="",
                )
                if row["status"] == "passed":
                    assert_outputs(entry, cwd, log)
            except (OSError, ValueError, KeyError) as error:
                row.update(status="failed", reason=str(error))
            row["seconds"] = round(time.monotonic() - started, 2)
        print(f"{profile}/{name}: {row['status']} ({row['seconds']}s)", flush=True)
        report = {
            "profile": profile,
            "sha": sha,
            "checks": results,
            "expected": selected,
            "complete": len(results) == len(selected),
            "platform": sys.platform,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if environment.exists():
        run_shell(
            "python -m pip freeze --all\npython -m pip check",
            cwd,
            env,
            output / "environment.log",
            60,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "scripts/readme_commands.json").read_text())
    blocks = read_blocks((ROOT / "README.md").read_text(), manifest)
    if args.validate:
        print(f"Covered all {len(blocks)} README blocks")
        return 0
    if not args.profile or not args.workspace or not args.output:
        parser.error("--profile, --workspace and --output are required for execution")
    report = run_profile(args.profile, args.workspace.resolve(), args.output.resolve(), blocks, manifest)
    return int(any(row["status"] != "passed" for row in report["checks"]))


if __name__ == "__main__":
    raise SystemExit(main())
