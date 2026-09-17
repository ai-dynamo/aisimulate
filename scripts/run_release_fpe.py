#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify an unchanged release wheel using separately identified CI probe tooling."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("aisimulate/", "aisimulate_core/", "aiconfigurator/", "aiconfigurator_core/")
PROBES = Path("python/aisimulate/tools/support_matrix")
DATA = Path("python/aisimulate/src/aiconfigurator_core/systems/fpe_support_matrix")


def list_releases(root: Path) -> list[dict]:
    """Pin every fetched release/<version> tip before any wheel is built."""
    refs = subprocess.check_output(
        ["git", "for-each-ref", "--format=%(refname)%00%(objectname)", "refs/remotes/origin/release/"],
        cwd=root,
        text=True,
    ).splitlines()
    releases = []
    for ref in refs:
        name, sha = ref.split("\0")
        version = name.removeprefix("refs/remotes/origin/release/")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(f"invalid release branch or commit: {name}")
        releases.append({"version": version, "source_sha": sha})
    if len(releases) > 256:
        raise ValueError("release inventory exceeds GitHub's 256-job matrix limit")
    return releases


def revision(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def identity(source: Path, source_sha: str, tooling_sha: str, branch: str) -> dict:
    if branch != "main" and not re.fullmatch(r"release/[A-Za-z0-9][A-Za-z0-9._-]*", branch):
        raise ValueError("expected main or a release/<version> branch")
    for label, root, expected in [("source", source, source_sha), ("tooling", ROOT, tooling_sha)]:
        if not re.fullmatch(r"[0-9a-f]{40}", expected) or revision(root) != expected:
            raise ValueError(f"{label} checkout differs from its pinned commit")
        subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=root, check=True, stdout=subprocess.DEVNULL)
    return {"schema_version": 1, "source_branch": branch, "source_sha": source_sha, "tooling_sha": tooling_sha}


def wheel_identity(directory: Path) -> tuple[Path, str]:
    wheels = list(directory.glob("aisimulate-*.whl"))
    if len(wheels) != 1:
        raise ValueError("expected exactly one release wheel")
    return wheels[0], hashlib.sha256(wheels[0].read_bytes()).hexdigest()


def verify_installed_wheel(wheel: Path, *, distribution=None) -> None:
    """Verify package bytes and active import locations, including the native runtime."""
    dist = distribution or importlib.metadata.distribution("aisimulate")
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if (
                name.startswith(PACKAGES)
                and not name.endswith("/")
                and Path(dist.locate_file(name)).read_bytes() != archive.read(name)
            ):
                raise ValueError(f"installed file differs from qualified release wheel: {name}")
    for name in ("aisimulate", "aisimulate_core", "aiconfigurator", "aiconfigurator_core", "aisimulate._runtime"):
        spec = importlib.util.find_spec(name)
        owned = {Path(dist.locate_file(p)).resolve() for p in dist.files or ()}
        if spec is None or spec.origin is None or Path(spec.origin).resolve() not in owned:
            raise ValueError(f"{name} is not imported from the qualified installed release wheel")


def prepare_harness(source: Path, destination: Path) -> Path:
    if destination.exists():
        raise ValueError("release probe harness directory must not already exist")
    package = destination / "tools" / "support_matrix"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    for name in ("fpe_support_matrix.py", "generate_fpe_support_matrix.py"):
        shutil.copyfile(ROOT / PROBES / name, package / name)
    # Model inventory and task/topology enumeration belong to the release.
    shutil.copyfile(source / PROBES / "support_matrix.py", package / "support_matrix.py")
    return destination


def output(name: str, value: str) -> None:
    if path := os.environ.get("GITHUB_OUTPUT"):
        with Path(path).open("a") as handle:
            handle.write(f"{name}={value}\n")


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / PROBES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def package_reports(reports: Path, expected: dict, wheel_sha: str, shards: list, destination: Path) -> None:
    qualifier = load_tool("qualify_fpe_support_matrix")
    report = qualifier.qualify(
        reports,
        expected_shards=shards,
        expected_sha=expected["source_sha"],
        expected_wheel_sha256=wheel_sha,
        required_probes=json.loads((ROOT / ".github/fpe-required-probes.json").read_text())["probes"],
    )
    report.update(expected)
    destination.mkdir(parents=True)
    (destination / "fpe-qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    builder = ROOT / PROBES / "build_fpe_support_matrix.py"
    subprocess.run([sys.executable, str(builder), str(reports), "--output-dir", str(destination / DATA)], check=True)
    # Keep copyable reproducers tied to the same release and tooling checkouts.
    import csv

    prefix = "python python/aisimulate/tools/support_matrix/generate_fpe_support_matrix.py"
    command = (
        " ".join(
            f"{name}={shlex.quote(value)}"
            for name, value in [
                ("FPE_SOURCE_SHA", expected["source_sha"]),
                ("FPE_TOOLING_SHA", expected["tooling_sha"]),
                ("FPE_BRANCH", expected["source_branch"]),
            ]
        )
        + " release-source/python/aisimulate/.venv/bin/python scripts/run_release_fpe.py probe"
    )
    for path in (destination / DATA).glob("*.csv"):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            if not row["Command"].startswith(prefix):
                raise ValueError("unexpected FPE reproducer command")
            row["Command"] = command + row["Command"][len(prefix) :]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list-releases", "record-wheel", "discover", "probe", "package"))
    parser.add_argument("--source-root", type=Path, default=ROOT / "release-source")
    parser.add_argument("--wheel-dir", type=Path, default=ROOT / "fpe-release-wheel")
    parser.add_argument("--harness-dir", type=Path, default=ROOT / "release-probe-harness")
    parser.add_argument("--reports", type=Path, default=ROOT / "fpe-release-results")
    parser.add_argument("--shards", type=Path, default=ROOT / "fpe-release-shards.json")
    parser.add_argument("--destination", type=Path, default=ROOT / "fpe-release-web")
    args, probe_args = parser.parse_known_args()
    if probe_args and args.action != "probe":
        parser.error(f"unexpected arguments: {probe_args}")
    if args.action == "list-releases":
        releases = json.dumps(list_releases(ROOT))
        output("releases", releases)
        print(releases)
        return
    expected = identity(
        args.source_root,
        os.environ["FPE_SOURCE_SHA"],
        os.environ["FPE_TOOLING_SHA"],
        os.environ["FPE_BRANCH"].removeprefix("refs/heads/"),
    )
    wheel, wheel_sha = wheel_identity(args.wheel_dir)
    receipt = {**expected, "wheel_sha256": wheel_sha}
    receipt_path = args.wheel_dir / "provenance.json"
    if args.action == "record-wheel":
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        output("wheel_sha256", wheel_sha)
        return
    if json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("release wheel does not match its source/tooling provenance")
    if args.action == "package":
        package_reports(args.reports, expected, wheel_sha, json.loads(args.shards.read_text()), args.destination)
        return
    verify_installed_wheel(wheel)
    sys.path.insert(0, str(prepare_harness(args.source_root, args.harness_dir)))
    if args.action == "discover":
        from tools.support_matrix.support_matrix import SupportMatrix

        shards = sorted({(s, b) for _, s, b, _ in SupportMatrix().generate_combinations()})
        if not shards:
            raise ValueError("release inventory has no FPE shards")
        value = json.dumps([{"system": s, "backend": b} for s, b in shards])
        args.shards.write_text(value + "\n")
        output("shards", value)
    else:
        from tools.support_matrix import generate_fpe_support_matrix as generator

        os.environ["FPE_WHEEL_SHA256"] = wheel_sha
        generator._source_sha = lambda: expected["source_sha"]
        sys.argv = [str(ROOT / "scripts/run_release_fpe.py"), *probe_args]
        generator.main()


if __name__ == "__main__":
    main()
