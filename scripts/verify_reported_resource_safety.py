#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the reported workload through the real Dynamo CLI under a small host budget.

Missing native bindings, wrong exit codes, and missing resource evidence fail the
run. This verifies safe refusal; it never attempts the original eager allocation.
Run separately on macOS and Linux and retain each machine's manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "tests/e2e/configs/resource_safety/reported-mac-sweep.yaml"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_identity(name: str) -> dict:
    package = importlib.metadata.distribution(name)
    return {"name": name, "version": package.version}


def check_refusal(output: Path, runtime: dict) -> dict:
    plan = json.loads((output / "resource-plan.json").read_text())
    if runtime["status"] != "resource_limited" or runtime["exit_code"] != 3:
        raise RuntimeError(f"CLI did not refuse safely: {runtime}")
    if not runtime["termination_complete"]:
        raise RuntimeError("CLI descendants were not reaped")
    estimate = plan["estimate"]
    if plan["status"] != "resource_limited" or estimate["request_count"] != 6_451_200:
        raise RuntimeError("missing largest-candidate resource preflight")
    if estimate["allocation_model"] != "dynamo-generated-u32-v1":
        raise RuntimeError("an unqualified runtime is not regression evidence")
    if runtime["peak_observed_rss_bytes"] > runtime["budget"]["memory_limit_bytes"]:
        raise RuntimeError("preflight failed to stop before the runtime memory budget")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dynamo-wheel", type=Path, required=True)
    parser.add_argument("--dynamo-source-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    # Require a real native module and complete a small native replay first.
    # An import error or placeholder runtime must never count as safe handling.
    from dynamo import _core
    from dynamo.mocker import MockEngineArgs
    from dynamo.replay import run_synthetic_trace_replay
    from aisimulate.config.common import ResourceConfig
    from aisimulate.supervision import run_process

    native_hash = digest(Path(_core.__file__).read_bytes())
    with zipfile.ZipFile(args.dynamo_wheel) as wheel:
        native_members = [
            name
            for name in wheel.namelist()
            if name.startswith("dynamo/_core") and name.endswith((".so", ".pyd"))
        ]
        if (
            len(native_members) != 1
            or digest(wheel.read(native_members[0])) != native_hash
        ):
            raise RuntimeError("loaded native binary does not match the supplied wheel")
    if (
        getattr(_core, "OFFLINE_SYNTHETIC_CONCURRENCY_ALLOCATION_MODEL", None)
        != "generated-u32-v1"
    ):
        raise RuntimeError("the native wheel does not include the lazy-replay change")
    dynamo_source = args.dynamo_source_dir.resolve()
    dynamo_diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=dynamo_source)
    if dynamo_diff:
        raise RuntimeError(
            "Dynamo source must be committed before recording build provenance"
        )
    lock = dynamo_source / "lib/bindings/python/Cargo.lock"
    core = next(
        package
        for package in tomllib.loads(lock.read_text())["package"]
        if package["name"] == "aisimulate-core"
    )

    smoke = run_synthetic_trace_replay(
        input_tokens=16,
        output_tokens=3,
        request_count=8,
        replay_concurrency=2,
        replay_mode="offline",
        router_mode="round_robin",
        num_workers=1,
        extra_engine_args=MockEngineArgs(
            block_size=4, num_gpu_blocks=128, speedup_ratio=1000.0
        ),
    )
    summary = smoke.summary
    if summary["num_requests"] != 8 or summary["completed_requests"] != 8:
        raise RuntimeError(f"native smoke replay did not complete: {summary}")

    original = yaml.safe_load(CASE.read_text())
    identity = {
        "fixture_sha256": digest(CASE.read_bytes()),
        "normalized_fixture_sha256": digest(
            json.dumps(original, sort_keys=True, separators=(",", ":")).encode()
        ),
        "aisimulate_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "aisimulate_diff_sha256": digest(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "native_binary_sha256": native_hash,
        "dynamo_wheel_sha256": digest(args.dynamo_wheel.read_bytes()),
        "dynamo_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=dynamo_source, text=True
        ).strip(),
        "dynamo_binding_lock_sha256": digest(lock.read_bytes()),
        "dynamo_core_dependency": core,
        "provenance_note": "Binary identity is checked against the wheel; source-to-wheel provenance requires the retained build log.",
        "native_allocation_model": getattr(
            _core, "OFFLINE_SYNTHETIC_CONCURRENCY_ALLOCATION_MODEL", "eager"
        ),
        "packages": [
            package_identity("aisimulate"),
            package_identity("ai-dynamo-runtime"),
        ],
        "native_smoke_summary": summary,
        "historical_eager_input_bytes": 264_241_152_000,
        "verdict": "pending",
    }
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(identity, indent=2) + "\n")
    runs = []
    for name, largest_only in [("full-sweep", False), ("largest-candidate", True)]:
        raw = yaml.safe_load(CASE.read_text())
        if largest_only:
            raw["traffic"]["load"]["concurrency"] = 64_512
        # Only the execution budget changes. Model, topology, token lengths,
        # stop rule, and scheduler/search domains remain the reported values.
        raw["execution"] = {"resources": {"memory_limit_gib": 2.0, "cpu_limit": 1}}
        config = args.output_dir / f"{name}.yaml"
        config.write_text(yaml.safe_dump(raw, sort_keys=False))
        output = args.output_dir / name
        command = [
            sys.executable,
            "-m",
            "aisimulate",
            "recommend",
            "--stack",
            "dynamo",
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ]
        runtime = run_process(
            command,
            policy=ResourceConfig(memory_limit_gib=2.0, cpu_limit=1),
            timeout=60,
        )
        plan = check_refusal(output, runtime)
        runs.append(
            {
                "name": name,
                "config_sha256": digest(config.read_bytes()),
                "runtime": runtime,
                "plan": plan,
            }
        )
        identity["runs"] = runs
        manifest.write_text(json.dumps(identity, indent=2) + "\n")
    identity["verdict"] = "safe_refusal"
    identity["limitations"] = [
        "No full-scale workload completion claimed",
        "Original kernel panic root cause and reporter rerun remain unverified",
    ]
    manifest.write_text(json.dumps(identity, indent=2) + "\n")
    print(manifest)


if __name__ == "__main__":
    main()
