#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate shared measurement visualization assets once per pinned campaign."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Support direct execution as well as package imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.fpm_accuracy.contract import HF_REPO, sha
from scripts.fpm_accuracy.dashboard.visualization import VisualizationWriter
from scripts.fpm_accuracy.hf import HfDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--evaluator-sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-dataset", type=Path)
    args = parser.parse_args()
    sha(args.hf_revision)
    sha(args.evaluator_sha)
    dataset = (
        HfDataset.from_local(args.local_dataset, revision=args.hf_revision)
        if args.local_dataset
        else HfDataset.from_hub(revision=args.hf_revision)
    )
    writer = VisualizationWriter(args.output, repo_id=HF_REPO, revision=dataset.revision)
    for configuration in dataset.configurations():
        print(f"Preparing {configuration.configuration_path}/{configuration.snapshot_id}", flush=True)
        writer.add_case(
            dataset.measurement_case(configuration.configuration_path, snapshot_id=configuration.snapshot_id)
        )
    writer.finish()
    qualification = dict(
        schema_version=1,
        hf_revision=dataset.revision,
        evaluator_sha=args.evaluator_sha,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
        completed_at=datetime.now(UTC).isoformat(),
        manifest_sha256=hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest(),
    )
    (args.output / "qualification.json").write_text(json.dumps(qualification, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
