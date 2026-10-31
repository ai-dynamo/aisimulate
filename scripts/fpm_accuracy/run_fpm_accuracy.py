#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate current HF cases with the installed exact AISim wheel; export aggregates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from scripts.fpm_accuracy.contract import (
    HF_REPO,
    METHODS,
    eligible_branch,
    sha,
    validate_summary,
)
from scripts.fpm_accuracy.dashboard_contract import MAX_FILE, validate_details
from scripts.fpm_accuracy.evaluate import evaluate_case
from scripts.fpm_accuracy.hf import HfDataset
from scripts.fpm_accuracy.scenario_contract import validate_scenarios
from scripts.fpm_accuracy.scenarios import evaluate_scenario_case, scenario_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--evaluator-sha", required=True)
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-dataset", type=Path)
    parser.add_argument("--scenarios", action="store_true", help="Also compare offline and online FPM scenarios")
    parser.add_argument(
        "--comparison-output",
        type=Path,
        help="Separate notification evidence; not a Pages artifact",
    )
    parser.add_argument(
        "--configuration",
        action="append",
        help="Smoke only: selected configuration paths; cannot qualify for publication",
    )
    args = parser.parse_args()
    if not eligible_branch(args.branch):
        parser.error("branch must be main or release >= 0.12.0")
    for revision in (args.commit, args.evaluator_sha, args.hf_revision):
        sha(revision)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output must be empty")
    # Fail the campaign for a broken installation, rather than publishing all-error rows.
    if not hasattr(importlib.import_module("aisimulate_core.sdk"), "RustForwardPassPerfModel"):
        raise RuntimeError("AISim native forward-pass SDK is unavailable")
    distribution = importlib.metadata.distribution("aisimulate")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    installed_hash = direct.get("archive_info", {}).get("hashes", {}).get("sha256")
    wheel_hash = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
    if installed_hash != wheel_hash:
        raise ValueError("installed AISim distribution does not match the supplied wheel")
    dataset = (
        HfDataset.from_local(args.local_dataset, revision=args.hf_revision)
        if args.local_dataset
        else HfDataset.from_hub(revision=args.hf_revision)
    )
    if dataset.revision != args.hf_revision:
        raise ValueError("resolved HF revision differs from the pinned campaign revision")
    configurations = dataset.configurations()
    if args.configuration:
        selected = set(args.configuration)
        configurations = tuple(item for item in configurations if item.configuration_path in selected)
        if {item.configuration_path for item in configurations} != selected:
            raise ValueError("unknown smoke configuration")
    comparison = {} if args.comparison_output else None
    rows = []
    details = []
    scenario_rows = []
    for index, configuration in enumerate(configurations, start=1):
        started = perf_counter()
        print(f"Evaluating {index}/{len(configurations)}: {configuration.configuration_path}", flush=True)
        case = dataset.measurement_case(configuration.configuration_path, snapshot_id=configuration.snapshot_id)
        rows.append(evaluate_case(case, comparison=comparison, details=details))
        if args.scenarios:
            scenario_rows.append(evaluate_scenario_case(case))
        print(f"Completed configuration in {perf_counter() - started:.1f}s", flush=True)
    summary = {
        "schema_version": 2,
        "snapshot": {
            "branch": args.branch,
            "commit_sha": args.commit,
            "hf_repo": HF_REPO,
            "hf_revision": dataset.revision,
            "evaluator_sha": args.evaluator_sha,
            "wheel_sha256": wheel_hash,
            "completed_at": datetime.now(UTC).isoformat(),
            "run_id": args.run_id,
            "run_attempt": args.run_attempt,
            "configuration_count": len(configurations),
            "complete": True,
        },
        "methods": list(METHODS),
        "rows": rows,
    }
    validate_summary(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("output must be empty")
    data = (json.dumps(summary, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    (args.output / "summary.json").write_bytes(data)
    detail_document = {"schema_version": 2, "snapshot": summary["snapshot"], "rows": details}
    validate_details(detail_document, summary)
    detail_data = (json.dumps(detail_document, allow_nan=False, sort_keys=True) + "\n").encode()
    (args.output / "details.json").write_bytes(detail_data)
    scenario_data = None
    if args.scenarios:
        scenarios = scenario_document(scenario_rows, snapshot=summary["snapshot"])
        validate_scenarios(scenarios, summary)
        scenario_data = (json.dumps(scenarios, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if len(scenario_data) > MAX_FILE:
            raise ValueError("scenario artifact exceeds the public asset size limit; campaign cannot qualify")
        (args.output / "scenarios.json").write_bytes(scenario_data)
    if args.configuration:
        (args.output / "SMOKE_ONLY.txt").write_text("Partial dataset smoke; not qualified for publication.\n")
    else:
        qualification = {
            "schema_version": 3 if scenario_data is not None else 2,
            "snapshot": summary["snapshot"],
            "summary_sha256": hashlib.sha256(data).hexdigest(),
            "details_sha256": hashlib.sha256(detail_data).hexdigest(),
        }
        if scenario_data is not None:
            qualification["scenarios_sha256"] = hashlib.sha256(scenario_data).hexdigest()
        (args.output / "qualification.json").write_text(json.dumps(qualification, allow_nan=False, indent=2) + "\n")
        if args.comparison_output:
            args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
            args.comparison_output.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "snapshot": summary["snapshot"],
                        "summary_sha256": hashlib.sha256(data).hexdigest(),
                        "points": comparison,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
    print(f"Completed {len(rows)} configurations; accuracy is advisory.", flush=True)


if __name__ == "__main__":
    main()
