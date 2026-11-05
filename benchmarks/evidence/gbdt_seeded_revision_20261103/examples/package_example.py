#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package a completed public example run with explicit producing-source evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    parser.add_argument("--producer-revision")
    args = parser.parse_args()
    root, run, inputs = args.repo.resolve(), args.run.resolve(), args.inputs.resolve()
    receipt = json.loads((run / "receipt.json").read_text())
    manifest = json.loads((inputs / "inputs.json").read_text())
    bundle = run / "exported-bundle"
    if receipt["native"]["sha256"] != args.expected_native_sha256:
        raise ValueError("The run did not use the expected native binary")
    if receipt["dataset_manifest_sha256"] != sha(inputs / "inputs.json"):
        raise ValueError("The input manifest differs from the completed run")
    if receipt["bundle_manifest_sha256"] != sha(bundle / "bundle.json"):
        raise ValueError("The exported bundle differs from the completed run")
    generator = root / "python/aisimulate/src/aisimulate_core/sdk/gbdt_example.py"
    if manifest["generator"]["sha256"] != sha(generator):
        raise ValueError("The installed generator differs from the input producer")
    if not receipt["readiness"]["execution"]["ready"]:
        raise ValueError("The example did not achieve complete decode execution readiness")
    producer_paths = (
        "python/aisimulate/src/aisimulate_core/sdk/fpm_bundle.py",
        "python/aisimulate/src/aisimulate_core/sdk/fpm_learned.py",
        "python/aisimulate/src/aisimulate_core/sdk/rust_engine_step.py",
    )
    source_hashes = {path: sha(root / path) for path in producer_paths}
    revision = None
    if args.producer_revision:
        revision = git(root, "rev-parse", f"{args.producer_revision}^{{commit}}").decode().strip()
        for path in producer_paths:
            if hashlib.sha256(git(root, "show", f"{revision}:{path}")).hexdigest() != source_hashes[path]:
                raise ValueError(f"Producer source differs from requested revision: {path}")
    build = {
        "kind": "aisimulate_synthetic_bundle_reproduction",
        "schema_version": 1,
        "generation_stage": "final_producer_pinned" if revision else "preparation_dirty_worktree",
        "producer_revision": revision,
        "observed_git_head": git(root, "rev-parse", "HEAD").decode().strip(),
        "tracked_worktree_dirty": bool(git(root, "status", "--porcelain", "--untracked-files=no")),
        "producer_source_sha256": source_hashes,
        "generator_source_sha256": sha(generator),
        "native_binary_sha256": receipt["native"]["sha256"],
        "bundle_manifest_sha256": receipt["bundle_manifest_sha256"],
        "input_manifest_sha256": receipt["dataset_manifest_sha256"],
        "python": receipt["python"],
        "dependencies": receipt["dependencies"],
        "offline_recipe": receipt["offline_recipe"],
        "completed_lifecycle_checks": receipt["checks"],
        "coverage_exclusions": receipt["coverage_exclusions"],
        "qualification": "functional_only_synthetic",
        "hardware_measured": False,
        "reproduction": "python -m aisimulate_core.sdk.gbdt_example run --input-dir INPUTS --output-dir NEW",
        "verification": "python -m aisimulate_core.sdk.gbdt_example verify-bundle --output-dir NEW",
        "revision_scope": (
            "producer_revision pins the producer SDK source, independently of the PR4 generator digest. "
            "Native binary bytes are pinned separately. No generated artifact contains its own Git commit hash."
        ),
    }
    target = root / "python/aisimulate/src/aisimulate_core/systems/fpm_learned/examples/synthetic-functional-v1"
    for name, source in (("seed", bundle / "seed"), ("inputs", inputs)):
        shutil.rmtree(target / name)
        shutil.copytree(source, target / name)
    shutil.copy2(bundle / "bundle.json", target / "bundle.json")
    (target / "build-info.json").write_text(json.dumps(build, indent=2, sort_keys=True) + "\n")
    print(json.dumps(build, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
