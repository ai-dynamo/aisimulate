#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run identical captured replay inputs and score hybrid-cache admission accuracy.

Private captures are inputs, never bundled fixtures. The cache score uses actual
native first-prefill observations, not a checkpoint oracle. The optional frozen
forest score is conditional on each replay's predicted admission geometry; it is
same-capture reconstruction, not held-out or end-to-end latency accuracy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    execution = read_json(args.execution)
    rank = execution["spec"]["engine"]["rank"]
    if rank["backend"] != "sglang" or execution["spec"]["engine"]["dp_size"] != 1:
        raise ValueError("this comparison requires an aggregated SGLang attention-DP1 capture")
    if args.mamba_config is not None:
        rank.setdefault("sglang", {})["mamba_cache"] = json.loads(args.mamba_config)
    timing = rank["timing_model"]
    config = timing["config"]
    if config.get("estimation_mode") != "fpm_learned":
        raise ValueError("use the identical frozen same-capture learned timing model")
    learned = config["estimator_config"]["fpm_learned"]
    if learned["adaptation"] != "frozen":
        raise ValueError("online adaptation would confound a cache-only comparison")
    forest = Path(learned["artifact_path"])
    if digest(forest) != learned["source_sha256"]:
        raise ValueError("frozen forest digest differs from execution descriptor")
    # Archived experimental builds exposed this observer-only field; released
    # runtime descriptors reject it. It does not affect model predictions.
    config.pop("experimental_learned_query_capture", None)
    execution["capture_telemetry"] = False
    execution["spec"]["record_per_request"] = True
    write_json(args.output / "execution.json", execution)
    traces = execution.get("traffic", {}).get("trace_paths", [])
    manifest = {
        "runtime_sha256": digest(args.runtime),
        "source_revision": args.revision,
        "input_execution_sha256": digest(args.execution),
        "executed_payload_sha256": digest(args.output / "execution.json"),
        "forest_sha256": digest(forest),
        "trace_sha256": {str(path): digest(Path(path)) for path in traces},
        "mamba_cache": rank.get("sglang", {}).get("mamba_cache"),
        "runner_sha256": digest(Path(__file__)),
        "status": "prepared",
        "execution_api": "run_replay_json",
    }
    write_json(args.output / "manifest.json", manifest)
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    spec = importlib.util.spec_from_file_location("_runtime", args.runtime)
    if spec is None or spec.loader is None:
        raise ValueError("runtime must be a Python extension shared object")
    runtime = importlib.util.module_from_spec(spec)
    sys.modules["_runtime"] = runtime
    spec.loader.exec_module(runtime)
    try:
        report = json.loads(runtime.run_replay_json(json.dumps(execution)))
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(args.output / "manifest.json", manifest)
        raise
    write_json(args.output / "report.json", report)
    manifest.update(
        status="executed",
        completed_requests=report["completed_requests"],
        report_sha256=digest(args.output / "report.json"),
    )
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest))


def first_native_prefills(path: Path) -> dict[str, dict[str, Any]]:
    first: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        witness = row["canonical_witness"]
        roles = witness["request_roles"]
        if "prefill" not in roles:
            continue
        if roles != ["prefill"]:
            raise ValueError("conditional first-step scoring requires singleton prefill batches")
        label = witness["request_labels"][0]
        request_id = label["x_request_id"]
        geometry = row["scheduled_requests"]
        first.setdefault(
            request_id,
            {
                "cached_tokens": geometry["past_kv_lengths"][0],
                "extend_tokens": geometry["extend_lengths"][0],
                "duration_ms": row["wall_time"] * 1000,
                "phase": label["benchmark_phase"],
                "source_trace_id": label["source_trace_id"],
            },
        )
    return first


def validate_prompt_binding(trace_hashes: dict[str, str], prompts_path: Path) -> None:
    expected = digest(prompts_path)
    if not trace_hashes:
        raise ValueError("captured replay must identify its trace inputs")
    for name, expected_trace_hash in trace_hashes.items():
        path = Path(name)
        if digest(path) != expected_trace_hash:
            raise ValueError("trace changed after replay")
        with path.open() as stream:
            header = json.loads(next(stream))
        if header.get("source", {}).get("native_prompt_map_sha256") != expected:
            raise ValueError("replay trace is not bound to the supplied native prompt capture")


def validate_first_geometry(reference: dict[str, Any], prompt_length: int, chunk: int) -> None:
    cached = reference["cached_tokens"]
    if not 0 <= cached < prompt_length or reference["extend_tokens"] != min(prompt_length - cached, chunk):
        raise ValueError("native first-prefill geometry does not satisfy the singleton chunk scoring contract")


def forest_prediction(forest: dict[str, Any], extend: int, past: int) -> float:
    values = {
        "req_batch_size": 1,
        "req_sum_extend": extend,
        "req_max_extend": extend,
        "req_min_extend": extend,
        "req_sum_past": past,
        "req_max_past": past,
        "req_min_past": past,
        "req_sum_extend_x_past": extend * past,
        "req_sum_extend_squared": extend * extend,
        "req_sum_past_squared": past * past,
    }
    features = [values[name] for name in forest["features"]]
    store = forest["stores"]["pure_prefill"]
    log_ms = store["baseline"]
    for tree in store["trees"]:
        node = 0
        while tree["left"][node] != -1:
            direction = "left" if features[tree["feature"][node]] <= tree["threshold"][node] else "right"
            node = tree[direction][node]
        log_ms += tree["value"][node]
    return math.exp(log_ms)


def aggregate(rows: list[dict[str, Any]], side: str) -> dict[str, Any]:
    errors = [abs(row[side]["cached_tokens"] - row["native"]["cached_tokens"]) for row in rows]
    uncached_total = sum(row["prompt_tokens"] - row["native"]["cached_tokens"] for row in rows)
    return {
        "requests": len(rows),
        "cache_exact_matches": sum(error == 0 for error in errors),
        "cache_exact_match_pct": 100 * sum(error == 0 for error in errors) / len(rows),
        "cache_token_mae": statistics.mean(errors),
        "cache_token_max_abs_error": max(errors),
        "uncached_prefill_token_wape_pct": 100 * sum(errors) / uncached_total,
        "conditional_first_prefill_step_mape_pct": statistics.mean(
            row[side]["conditional_first_step_ape_pct"] for row in rows
        ),
    }


def client_timing(rows: list[dict[str, Any]], side: str) -> dict[str, Any]:
    selected = [row for row in rows if "native_client" in row]
    result: dict[str, Any] = {"profiling_requests": len(selected)}
    for name, observed_field in [("ttft", "ttft_ms"), ("tpot", "tpot_visible_span_ms")]:
        measured = []
        predicted = []
        for row in selected:
            truth = row["native_client"][observed_field]
            prediction = row[side][f"replay_{name}_ms"]
            if truth is None or prediction is None or truth <= 0:
                raise ValueError(f"missing or nonpositive {name} timing; do not silently change cohort")
            measured.append(truth)
            predicted.append(prediction)
        result[f"{name}_mape_pct"] = statistics.mean(
            100 * abs(prediction - truth) / truth for prediction, truth in zip(predicted, measured, strict=True)
        )
        result[f"{name}_native_mean_ms"] = statistics.mean(measured)
        result[f"{name}_replay_mean_ms"] = statistics.mean(predicted)
    return result


def compare(args: argparse.Namespace) -> None:
    manifests = {
        side: read_json(path / "manifest.json") for side, path in [("before", args.before), ("after", args.after)]
    }
    for side, directory in [("before", args.before), ("after", args.after)]:
        manifest = manifests[side]
        if manifest.get("status") != "executed":
            raise ValueError(f"run did not complete: {side}")
        for filename, key in [("report.json", "report_sha256"), ("execution.json", "executed_payload_sha256")]:
            if digest(directory / filename) != manifest[key]:
                raise ValueError(f"stored evidence digest differs: {side} {filename}")
    for key in ["input_execution_sha256", "forest_sha256", "trace_sha256", "execution_api"]:
        if manifests["before"][key] != manifests["after"][key]:
            raise ValueError(f"comparison inputs differ: {key}")
    payloads = {
        side: read_json(path / "execution.json") for side, path in [("before", args.before), ("after", args.after)]
    }
    normalized = copy.deepcopy(payloads)
    for payload in normalized.values():
        rank = payload["spec"]["engine"]["rank"]
        rank["sglang"].pop("mamba_cache", None)
        rank["timing_model"]["config"].pop("experimental_learned_query_capture", None)
    if normalized["before"] != normalized["after"]:
        raise ValueError("payloads differ beyond hybrid cache configuration and capture destination")
    reports = {side: read_json(path / "report.json") for side, path in [("before", args.before), ("after", args.after)]}
    records = {side: {row["request_id"]: row for row in report["per_request"]} for side, report in reports.items()}
    for side, report in reports.items():
        if len(records[side]) != len(report["per_request"]):
            raise ValueError(f"duplicate replay request IDs: {side}")
    validate_prompt_binding(manifests["before"]["trace_sha256"], args.native_prompts)
    native = first_native_prefills(args.native_canonical)
    prompt_rows = read_jsonl(args.native_prompts)
    prompts = {row["request_labels"]["x_request_id"]: row for row in prompt_rows}
    if len(prompts) != len(prompt_rows):
        raise ValueError("duplicate native prompt request IDs")
    if any(set(rows) != set(native) for rows in records.values()) or set(prompts) != set(native):
        raise ValueError("all native requests must have exactly one prompt and before/after record")
    forest_path = Path(
        payloads["before"]["spec"]["engine"]["rank"]["timing_model"]["config"]["estimator_config"]["fpm_learned"][
            "artifact_path"
        ]
    )
    if digest(forest_path) != manifests["before"]["forest_sha256"]:
        raise ValueError("frozen forest changed after replay")
    forest = read_json(forest_path)
    if forest["target"] != "log_ms":
        raise ValueError("unexpected frozen forest target")
    chunk = payloads["before"]["spec"]["engine"]["rank"]["sglang"]["chunked_prefill_size"]
    clients = {}
    if args.native_client_metrics is not None:
        clients = {row["request_id"]: row for row in read_json(args.native_client_metrics)["requests"]}
        expected = {key for key, row in native.items() if row["phase"] == "profiling"}
        if set(clients) != expected:
            raise ValueError("client metric cohort must exactly equal native profiling requests")
    rows = []
    for request_id, reference in native.items():
        prompt_length = prompts[request_id]["prompt_length"]
        validate_first_geometry(reference, prompt_length, chunk)
        if len(prompts[request_id]["prompt_token_ids"]) != prompt_length:
            raise ValueError(f"native prompt length disagrees with captured tokens: {request_id}")
        row = {"request_id": request_id, "prompt_tokens": prompt_length, "native": reference}
        if request_id in clients:
            row["native_client"] = clients[request_id]
        for side, replay in records.items():
            record = replay[request_id]
            if record["terminal_status"] != "completed" or record["input_length"] != prompt_length:
                raise ValueError(f"incomplete or different prompt: {side} {request_id}")
            cached = record["first_admission_reused_input_tokens"]
            extend = min(prompt_length - cached, chunk)
            predicted = forest_prediction(forest, extend, cached)
            row[side] = {
                "cached_tokens": cached,
                "uncached_prefill_tokens": prompt_length - cached,
                "first_step_extend_tokens": extend,
                "conditional_first_step_prediction_ms": predicted,
                "conditional_first_step_ape_pct": 100
                * abs(predicted - reference["duration_ms"])
                / reference["duration_ms"],
                "replay_ttft_ms": record["ttft_ms"],
                "replay_tpot_ms": record["itl_ms"],
            }
        rows.append(row)
    result = {
        "schema": "aisimulate.mamba_cache_accuracy.v1",
        "scope": (
            "Same-capture diagnostic: both runtimes run real Replay cache lifecycle; "
            "frozen FPM conditional first-step scoring"
        ),
        "native_canonical_sha256": digest(args.native_canonical),
        "native_prompts_sha256": digest(args.native_prompts),
        "source_trace_count": len({row["native"]["source_trace_id"] for row in rows}),
        "first_baseline_cache_mismatch": next(
            (row for row in rows if row["before"]["cached_tokens"] != row["native"]["cached_tokens"]), None
        ),
        "runs": manifests,
        "all_requests": {side: aggregate(rows, side) for side in ["before", "after"]},
        "profiling_requests": {
            side: aggregate([row for row in rows if row["native"]["phase"] == "profiling"], side)
            for side in ["before", "after"]
        },
        "limitations": [
            "Frozen FPM trained on this capture; independent test rows are zero.",
            "Conditional step prediction changes only cache-derived input geometry; "
            "it does not assert aligned post-divergence whole batches.",
            "Both sides use the same closed-loop diagnostic trace; native arrival times "
            "and client-visible timing boundaries remain different.",
            "Native generated output token IDs and complete live checkpoint inventory were not captured.",
        ],
        "requests": rows,
    }
    if clients:
        result["native_client_metrics_sha256"] = digest(args.native_client_metrics)
        result["client_timing_diagnostic"] = {side: client_timing(rows, side) for side in ["before", "after"]}
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ["all_requests", "profiling_requests"]}, indent=2))


HOST_RUNTIME_FIELDS = {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}


def canonical_report(report: dict[str, Any]) -> dict[str, Any]:
    """Exclude only measured CPU execution cost; preserve all simulated timings."""
    return {key: value for key, value in report.items() if key not in HOST_RUNTIME_FIELDS}


def parity(args: argparse.Namespace) -> None:
    reports = [canonical_report(read_json(directory / "report.json")) for directory in [args.first, args.second]]
    if reports[0] != reports[1]:
        raise ValueError("replay reports differ beyond host runtime fields")
    normalized = json.dumps(reports[0], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result = {
        "status": "pass",
        "excluded_host_runtime_fields": sorted(HOST_RUNTIME_FIELDS),
        "canonical_report_sha256": hashlib.sha256(normalized).hexdigest(),
        "runs": [read_json(directory / "manifest.json") for directory in [args.first, args.second]],
    }
    write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--runtime", type=Path, required=True)
    execute.add_argument("--revision", required=True)
    execute.add_argument("--execution", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--mamba-config")
    score = commands.add_parser("compare")
    score.add_argument("--before", type=Path, required=True)
    score.add_argument("--after", type=Path, required=True)
    score.add_argument("--native-canonical", type=Path, required=True)
    score.add_argument("--native-prompts", type=Path, required=True)
    score.add_argument("--native-client-metrics", type=Path)
    score.add_argument("--output", type=Path, required=True)
    repeat = commands.add_parser("parity")
    repeat.add_argument("--first", type=Path, required=True)
    repeat.add_argument("--second", type=Path, required=True)
    repeat.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"run": run, "compare": compare, "parity": parity}[args.command](args)


if __name__ == "__main__":
    main()
