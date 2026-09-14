# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current strict SILICON replay of hash-closed original cold HTTP workloads."""

import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import aisimulate._runtime as native


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--analysis-tools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    REPO = args.repo.resolve()
    OUT = args.output.resolve()
    args.workloads = args.workloads.resolve()
    expected_helper = "7a2b069783e84cca3d1107e157c43232ce117ddd2a4033e3a199742231ff3802"
    if sha(args.analysis_tools / "compare_e2e.py") != expected_helper:
        raise ValueError("Original comparison/metric source differs")
    sys.path.insert(0, str(args.analysis_tools.resolve()))
    import compare_e2e

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    expected = "f21ed55168074f8d32153a04d3cdfc524484722b"
    changed = subprocess.check_output(
        ["git", "diff", expected, "--", "crates/core", "python/aisimulate/src"],
        cwd=REPO,
    )
    if changed or not Path(native.__file__).resolve().is_relative_to(REPO):
        raise ValueError(
            "Predictor source differs or native extension is outside selected worktree"
        )
    bindings = json.loads(
        (Path(__file__).parent / "original-input-bindings.json").read_text()
    )
    for profile in ("off", "on"):
        binding = bindings[profile]
        root = REPO / binding["prediction_config"]["systems_path"]
        actual = {
            str(p.relative_to(root)): sha(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }
        if actual != binding["systems_files_sha256"]:
            raise ValueError("Frozen operation calibration tables differ")
    os.chdir(REPO)
    OUT.mkdir(parents=True, exist_ok=False)
    identity = {
        "predictor_commit": head,
        "native_extension_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "metric_source_sha256": sha(Path(compare_e2e.__file__)),
    }
    write(OUT / "started.json", identity)
    summaries = []
    for scope in ("off", "on"):
        path = args.workloads / f"{scope}.json.gz"
        source_sha = sha(path)
        original = json.loads(gzip.decompress(path.read_bytes()))
        cohorts = []
        for cohort in original["cohorts"]:
            row = {
                k: copy.deepcopy(v)
                for k, v in cohort.items()
                if k not in ("replay_spec", "request_ids", "cohort_start_ms")
            }
            if cohort["recovery_status"] != "exact_original_spec_sha256":
                row.update(
                    status="prediction_input_unavailable",
                    failure_type="RetainedInputUnavailable",
                    failure=cohort["reason"],
                )
            else:
                spec = cohort["replay_spec"]
                assert (
                    hashlib.sha256(canonical(spec).encode()).hexdigest()
                    == cohort["replay_spec_sha256"]
                )
                provider = spec["engine"]["rank"]["timing_model"]["config"]
                assert (
                    provider["database_mode"] == "SILICON"
                    and provider["forward_model"] == "op_level"
                )
                assert (
                    provider["strict_provenance"]
                    and provider["enable_shared_layer"] is False
                )
                try:
                    actual = json.loads(native.run_replay_json(canonical(spec)))
                    predicted, requests = compare_e2e.predicted_metrics(
                        actual, cohort["request_ids"], cohort["cohort_start_ms"]
                    )
                    for r in requests:
                        cached = cohort.get("native_initial_cached_tokens", {}).get(
                            r["request_id"]
                        )
                        r["native_initial_cached_tokens"] = cached
                        r["cache_reuse_matches_native"] = (
                            None
                            if cached is None
                            else cached == r["reused_input_tokens"]
                        )
                except Exception as error:
                    row.update(
                        status="prediction_unavailable",
                        failure_type=type(error).__name__,
                        failure=str(error),
                    )
                else:
                    row.update(
                        status="predicted",
                        prediction=predicted,
                        requests=requests,
                        current_cache_semantics_match=all(
                            r["cache_reuse_matches_native"] is True for r in requests
                        ),
                        native_result_sha256=hashlib.sha256(
                            canonical(actual).encode()
                        ).hexdigest(),
                    )
            cohorts.append(row)
        metrics = {}
        for metric in (
            "ttft_ms",
            "average_tpot_ms",
            "output_tokens_per_second",
            "exact_itl_ms",
            "request_latency_ms",
            "last_token_latency_ms",
        ):
            paired = [
                (r["observed"][metric], r["prediction"][metric])
                for r in cohorts
                if r["status"] == "predicted"
                and metric in r["observed"]
                and metric in r["prediction"]
            ]
            value = {"observed_cohorts": len(cohorts), "predicted_cohorts": len(paired)}
            if paired:
                value.update(
                    mape_percent=statistics.mean(
                        100 * abs(p / o - 1) for o, p in paired
                    ),
                    wape_percent=100
                    * sum(abs(p - o) for o, p in paired)
                    / sum(o for o, _ in paired),
                )
            metrics[metric] = value
        assert (
            sha(path) == source_sha
            and sha(Path(native.__file__)) == identity["native_extension_sha256"]
        )
        report = identity | {
            "scope": scope,
            "shared_workload_sha256": source_sha,
            "fresh_raw_admission": False,
            "new_measurements": False,
            "correction_fitting": False,
            "metric_scope": "Hash-closed cold cohort subset; finite-cohort throughput, not capacity.",
            "first_output_semantics": "Current SGLang replay emits first output token at final prefill completion.",
            "confidence_intervals": None,
            "summary": metrics,
            "cohorts": cohorts,
        }
        target = OUT / f"{scope}.json.gz"
        write(target, report)
        summaries.append(
            {"scope": scope, "summary": metrics, "report_sha256": sha(target)}
        )
        print(json.dumps(summaries[-1]), flush=True)
    write(OUT / "completion.json", identity | {"complete": True, "scopes": summaries})


if __name__ == "__main__":
    main()
