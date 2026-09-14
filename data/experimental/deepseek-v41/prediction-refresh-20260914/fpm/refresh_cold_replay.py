# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run current FPM on shared, hash-closed original GB300 cold request inputs.

The recovery packet proves each complete original SILICON replay specification.
Only the timing provider changes for this FPM prediction. Request tokens,
arrival offsets, scheduler and topology remain unchanged. Missing prefix seed
arrivals and unrecoverable request metadata are explicitly unavailable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

from refresh_reported_native import read, require, sha, write

PACKETS = {
    "off": "ca98d203890a856a7a89e0b3070c2206b67729591dec4467799985437f4f7007",
    "on": "c5446a862e47e1f6dbdf31716e32a14bd00d69f5b7712f82529dc856115bb091",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument(
        "--expected-packet-sha256", help="Pin a regenerated portable packet; original run pin is the default"
    )
    parser.add_argument("--profile", choices=PACKETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    require(Path.cwd() == repo, "run from current repository")
    require(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.expected_head, "head differs"
    )
    packet_sha = args.expected_packet_sha256 or PACKETS[args.profile]
    require(sha(args.packet) == packet_sha, "shared packet differs")
    tools = repo / "data/experimental/deepseek-v41/verification-plan"
    sys.path[:0] = [str(tools), str(repo / "python/aisimulate"), str(repo / "python/aisimulate/src")]
    import json

    from compare_e2e import canonical, comparison_sources, paired_summary, predicted_metrics, timing_provider
    from compare_trace import systems_identity

    import aisimulate._runtime as native

    require(Path(native.__file__).resolve().is_relative_to(repo), "foreign native binary")
    require(sha(native.__file__) == args.expected_native_sha256, "native differs")
    packet = read(args.packet)
    require(
        packet["predicted_timing_as_input"] is False and packet["fresh_raw_admission"] is False, "packet scope differs"
    )
    config_path = (
        repo
        / f"data/experimental/deepseek-v41/gb300-fpm/{args.profile}-union-v3-tracewait2"
        / "reports/core/prediction-config.json"
    )
    config = read(config_path)
    require(config["decoder_replay"] is (args.profile == "on") and config["fpm_fmha_dtype"] == "fp8", "config differs")
    identity, sources = systems_identity(config), comparison_sources()
    results = []
    for original in packet["cohorts"]:
        row = {
            key: original[key]
            for key in (
                "cohort_id",
                "purpose",
                "trial_index",
                "trial_seed",
                "observed",
                "input_ordinal",
                "segment_ordinal",
                "recovery_status",
                "replay_spec_sha256",
            )
            if key in original
        }
        if original["recovery_status"] != "exact_original_spec_sha256":
            row.update(
                status="prediction_unavailable",
                failure_type="IncompleteOriginalInput",
                failure=original.get("recovery_failure", original.get("reason", "original request input incomplete")),
            )
            results.append(row)
            continue
        spec = copy.deepcopy(original["replay_spec"])
        require(
            hashlib.sha256(canonical(spec).encode()).hexdigest() == original["replay_spec_sha256"], "old spec differs"
        )
        require(
            original["cohort_start_ms"] == 0 and original["purpose"] != "prefix-reuse-B", "not a complete cold input"
        )
        old_provider = copy.deepcopy(spec["engine"]["rank"]["timing_model"]["config"])
        provider = timing_provider(config)
        for key in set(old_provider) | set(provider):
            if key not in {"system", "systems_path", "database_mode", "forward_model", "fpm_fmha_dtype", "fmha_dtype"}:
                require(old_provider.get(key) == provider.get(key), "execution identity changed: " + key)
        spec["engine"]["rank"]["timing_model"]["config"] = provider
        row["current_replay_spec_sha256"] = hashlib.sha256(canonical(spec).encode()).hexdigest()
        try:
            raw = json.loads(native.run_replay_json(canonical(spec)))
            prediction, requests = predicted_metrics(raw, original["request_ids"], original["cohort_start_ms"])
            observed = row["observed"]
            require(all(math.isfinite(v) and v > 0 for v in observed.values()), "invalid observed metric")
        except Exception as error:
            row.update(
                status="prediction_unavailable",
                failure_type=type(error).__name__,
                failure=str(error).replace(config["systems_path"], "<frozen systems overlay>"),
            )
        else:
            cached = original["native_initial_cached_tokens"]
            for request in requests:
                request["native_initial_cached_tokens"] = cached[request["request_id"]]
                request["cache_reuse_matches_native"] = request["reused_input_tokens"] == cached[request["request_id"]]
            row.update(
                status="predicted",
                prediction=prediction,
                requests=requests,
                cache_semantics_match=all(r["cache_reuse_matches_native"] for r in requests),
                signed_error_percent={
                    key: 100 * (prediction[key] / value - 1) for key, value in observed.items() if key in prediction
                },
            )
        results.append(row)
    metrics = {}
    for metric in (
        "ttft_ms",
        "average_tpot_ms",
        "output_tokens_per_second",
        "request_latency_ms",
        "last_token_latency_ms",
    ):
        pairs = [r for r in results if r["status"] == "predicted" and metric in r["signed_error_percent"]]
        metrics[metric] = {
            "planned_cohorts": len(results),
            "predicted_cohorts": len(pairs),
            "mape_percent": statistics.mean(abs(r["signed_error_percent"][metric]) for r in pairs) if pairs else None,
            "mean_signed_error_percent": statistics.mean(r["signed_error_percent"][metric] for r in pairs)
            if pairs
            else None,
            "weighting": "equal supported HTTP cohorts; cold subset; no lifecycle pooling for confidence intervals",
        }
    require(sha(args.packet) == packet_sha and sources == comparison_sources(), "inputs/source changed")
    require(identity == systems_identity(config), "calibration changed")
    write(
        args.output,
        {
            "schema": "dsv41.current-source.hash-closed-cold-replay.v1",
            "scope": args.profile + "/core/cold-subset",
            "predictor_commit": args.expected_head,
            "native_extension_sha256": args.expected_native_sha256,
            "packet_sha256": packet_sha,
            "script_sha256": sha(__file__),
            "prediction_sources": sources,
            "prediction_config": config,
            "prediction_config_sha256": sha(config_path),
            "systems_identity": identity,
            "fresh_raw_admission": False,
            "new_observations": False,
            "correction_fitting": False,
            "cohorts": results,
            "metrics": metrics,
            "purpose_descriptive_summaries": paired_summary(results, final=False),
            "failure_counts": dict(Counter(r.get("failure", "") for r in results if r["status"] != "predicted")),
            "confidence_intervals": None,
            "limitations": [
                "Coverage is a recoverable cold subset, not the complete historical HTTP study.",
                "All original cohort denominators and unavailable rows retained.",
                "Old exact replay hashes qualify workload inputs; current FPM timing provider is separately bound.",
                "No predicted duration is fed back into arrivals or any other replay input.",
            ],
        },
    )


if __name__ == "__main__":
    main()
