#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt an existing seed using a complete, validated file of later observations.

No telemetry is collected. The caller validates/selects ROLE.online.jsonl and
keeps it separate from offline training and held-out evaluation. This example
starts from the original seed and processes the file once; it is not a live
stream resumer. Applications own admission cursors and must coordinate them
with saved checkpoints to avoid repeated feedback. Failed outputs are partial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path


def fingerprint(path):
    digest, rows = hashlib.sha256(), 0
    with path.open("rb") as source:
        for line in source:
            digest.update(line)
            rows += bool(line.strip())
    return digest.hexdigest(), rows


def compatible_busy(row, role):
    scheduled = row["scheduled_requests"]
    prefill = int(scheduled["sum_prefill_tokens"]) > 0
    decode = int(scheduled["num_decode_requests"]) > 0
    duration = float(row["wall_time"])
    return (
        math.isfinite(duration)
        and duration > 0
        and (prefill or decode)
        and (role != "prefill" or not decode)
        and (role != "decode" or not prefill)
    )


def write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("aggregated", "prefill", "decode"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for this adaptation attempt")
    parser.add_argument("--flush-updates", action="store_true", help="fit a pending partial update batch before saving")
    args = parser.parse_args()

    from aisimulate_core.sdk import RustForwardPassPerfModel
    from aisimulate_core.sdk.fpm_learned import iter_fpm_records

    run, role, output = args.run_dir.resolve(), args.role, args.output_dir.resolve()
    source = run / "selected" / f"{role}.online.jsonl"
    seed = run / "models" / f"{role}.online-seed"
    config = json.loads((run / f"{role}.identity.json").read_text())
    if config.get("worker_type") != role or not seed.is_dir():
        raise ValueError("Require a matching role identity and existing online-seed directory")
    if output.is_relative_to(seed):
        raise ValueError("Keep adaptation outputs outside the original seed directory")
    source_hash, source_rows = fingerprint(source)
    if not source_rows:
        raise ValueError("No feedback observations")
    output.mkdir(parents=True, exist_ok=False)
    config.update(estimation_mode="fpm_learned", fallback_policy="deny")
    config["estimator_config"] = {
        "fpm_learned": {"checkpoint_path": str(seed), "adaptation": "online"},
        "correction": {"enabled": False},
    }
    online = RustForwardPassPerfModel.best_available(config)
    admitted, query = 0, None
    for row in iter_fpm_records([source]):
        if not compatible_busy(row, role):
            raise ValueError(f"Feedback observation {admitted} is not role-compatible busy work")
        query = [{key: value for key, value in row.items() if key != "wall_time"}]
        prediction = online.estimate_forward_pass_time_ms(query)
        if prediction is None or not math.isfinite(prediction) or prediction <= 0:
            raise ValueError(f"No valid prediction for feedback observation {admitted}; inspect seed stores")
        # Predict before the measured seconds-valued label enters fitting state.
        online.tune_with_fpms([[row]])
        admitted += 1
    if admitted != source_rows or fingerprint(source) != (source_hash, source_rows):
        raise ValueError("Feedback changed or the parser skipped records; inspect the validated source")
    if args.flush_updates:
        online.flush_updates()

    checkpoint = output / "checkpoint"
    online.save_online_checkpoint(checkpoint)
    write_json(output / "model.json", online.export_learned_artifact())
    restore_config = deepcopy(online.diagnostics()["provenance"]["config"])
    learned = restore_config["estimator_config"]["fpm_learned"]
    learned.pop("artifact_path", None)
    learned.pop("source_sha256", None)
    learned["checkpoint_path"] = str(checkpoint)
    restored = RustForwardPassPerfModel.best_available(restore_config)
    expected = online.estimate_forward_pass_time_ms(query)
    actual = restored.estimate_forward_pass_time_ms(query)
    if expected is None or actual is None or not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("Restored prediction does not match the saved model")
    write_json(output / "restore-config.json", restored.diagnostics()["provenance"]["config"])
    receipt = {
        "role": role,
        "seed": str(seed),
        "feedback": str(source),
        "feedback_sha256": source_hash,
        "admitted_observations": admitted,
        "last_admitted_observation_index": admitted - 1,
        "flushed_updates": args.flush_updates,
        "restored_prediction_ms": actual,
        "restore_check": "last observation prediction only; not an accuracy qualification",
    }
    write_json(output / "adaptation.json", receipt)
    print(json.dumps({"output_dir": str(output), **receipt}, indent=2))


if __name__ == "__main__":
    main()
