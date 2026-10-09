# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare ordered scientific content, independently of renamed file identities.

Each reader runs in a fresh process to measure peak memory without retaining two
copies of a campaign. This deliberately permits dirty local migration checkouts;
it is a migration check, not a publishable evaluation.
"""

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

from scripts.fpm_accuracy.hf.dataset import HfDataset
from scripts.fpm_accuracy.hf.overrides import HfOverrides


def dataset(root):
    source = HfDataset(
        Path(root),
        repo_id="nvidia/aisimulate-fpm-dataset",
        revision="0" * 40,
        overrides=HfOverrides(version=1),
        overrides_sha256=None,
    )
    # The legacy catalog has two retired snapshots without a current leaf.
    # Compare their content too; publication hash-chain validation is separate.
    source._configuration_cache[True] = tuple(
        source._load_configuration(path)
        for path in source._index["configuration_manifests"] + source._index["history_manifests"]
    )
    return source


def fingerprint(root, configuration, snapshot):
    started = time.perf_counter()
    case = dataset(root).measurement_case(configuration, snapshot_id=snapshot)
    elapsed = time.perf_counter() - started
    digest = hashlib.sha256()
    for observation in case.observations:
        value = [
            observation.source_file_id,
            observation.source_row,
            observation.actual_ms,
            observation.tuning_payload(),
        ]
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return {
        "count": len(case.observations),
        "ordered_content_sha256": digest.hexdigest(),
        "status": case.status.value,
        "ordering": case.ordering.value,
        "issues": sorted((i.source_file_id, i.state.value, i.reason, i.count) for i in case.issues),
        "read_s": elapsed,
        "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy", type=Path)
    parser.add_argument("migrated", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configuration")
    parser.add_argument("--snapshot")
    parser.add_argument("--reader", choices=("legacy", "migrated"))
    args = parser.parse_args()
    if args.reader:
        print(json.dumps(fingerprint(getattr(args, args.reader), args.configuration, args.snapshot)))
        return
    reports = []
    for config in dataset(args.legacy).configurations(include_history=True):
        if args.configuration and config.configuration_path != args.configuration:
            continue
        results = {}
        for reader in ("legacy", "migrated"):
            command = [
                sys.executable,
                "-m",
                "scripts.fpm_accuracy.verify_storage_migration",
                str(args.legacy),
                str(args.migrated),
                "--output",
                str(args.output),
                "--configuration",
                config.configuration_path,
                "--snapshot",
                config.snapshot_id,
                "--reader",
                reader,
            ]
            results[reader] = json.loads(subprocess.check_output(command, text=True))
        equivalent = all(
            results["legacy"][k] == results["migrated"][k]
            for k in ("count", "ordered_content_sha256", "status", "ordering", "issues")
        )
        reports.append({"configuration": config.configuration_id, "equivalent": equivalent, **results})
        args.output.write_text(json.dumps(reports, indent=2) + "\n")
        print(config.configuration_id, equivalent, results["legacy"]["count"], flush=True)
        if not equivalent:
            raise SystemExit("Scientific content changed; see comparison report")


if __name__ == "__main__":
    main()
