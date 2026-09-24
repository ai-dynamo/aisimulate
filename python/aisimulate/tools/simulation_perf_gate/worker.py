# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One fresh process, one full replay, using this revision's public runner."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.simulation_perf_gate import PROTOCOL_VERSION, digest
from tools.simulation_perf_gate.contract import MODEL_FIELDS, fields

HOST_FIELDS = {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}


def portable_model_identity(model: dict, systems_root: Path) -> dict:
    paths = model.get("systems_paths", [])
    if not paths or any(Path(path).resolve() != systems_root.resolve() for path in paths):
        raise ValueError("benchmark requires this installation's packaged model data")
    return {**fields(model, MODEL_FIELDS, "/model_identity"), "systems_paths": ["package:aisimulate_core/systems"]}


def run(request: dict) -> dict:
    import aisimulate_core
    from aisimulate import CorePredictionConfig, EngineReplayRunnerFactory, ReplayOutputRequirements
    from aisimulate.compiler import prediction_to_replay_spec

    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("incompatible simulation-performance protocol")
    item = request["case"]
    if item.get("determinism") != "canonical_v1":
        raise ValueError("benchmark requires canonical_v1 determinism")
    config = deepcopy(item["config"])
    availability = request["phase"] == "availability"
    if request["phase"] not in {"availability", "measure"}:
        raise ValueError("unknown benchmark phase")
    if item.get("trace_sha256"):
        trace = Path(__file__).parent / "fixtures/agentx.jsonl"
        if hashlib.sha256(trace.read_bytes()).hexdigest() != item["trace_sha256"]:
            raise ValueError("local trace does not match the controller's input hash")
        config["traffic"]["source"]["paths"] = [str(trace)]
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    identity = {}
    provenance = {}
    for role, field in (
        ("aggregated", "agg_engine_args"),
        ("prefill", "prefill_engine_args"),
        ("decode", "decode_engine_args"),
    ):
        args = getattr(spec.backend_deployment, field)
        if args is not None:
            timing = args["timing_model"]
            model = timing["config"]
            if (
                timing.get("provider") != "aic"
                or model.get("estimation_mode") != "op_level"
                or model.get("fallback_policy") != "deny"
                or model.get("database_mode") != "SILICON"
                or model.get("enable_shared_layer") is not True
            ):
                raise ValueError(f"{role} did not retain the pinned real-model policy")
            identity[role] = portable_model_identity(model, Path(aisimulate_core.__file__).parent / "systems")
            provenance[role] = model
    if not identity:
        raise ValueError("replay has no real forward model")
    runner = EngineReplayRunnerFactory(determinism=item["determinism"]).create(0)
    try:
        result = runner.run(
            spec,
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=availability),
        )
    finally:
        runner.close()
    report = result.metadata["native_report"]
    expected = item["expected_requests"]
    if report.get("num_requests") != expected or report.get("completed_requests") != expected:
        raise ValueError(f"incomplete replay: expected {expected}, got {report.get('completed_requests')}")
    if report.get("total_output_tokens") != item["expected_output_tokens"]:
        raise ValueError("replay did not produce all requested output tokens")
    wall = report.get("wall_time_ms")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
        raise ValueError("replay wall_time_ms must be finite and positive")
    # Keep diagnostics in the artifact; the controller owns the comparison projection.
    normalized = {key: value for key, value in report.items() if key not in HOST_FIELDS}
    evidence = {}
    if availability:
        records = normalized["per_request"]
        if len(records) != expected or any(
            row["terminal_status"] != "completed" or row["output_length"] != row["requested_output_length"]
            for row in records
        ):
            raise ValueError("incomplete per-request results")
        if config["engine"]["mode"] == "disaggregated":
            transferred = sum(row.get("destination_activated_ms") is not None for row in records)
            evidence["pd_activated_requests"] = transferred
            if transferred != expected:
                raise ValueError("P/D case did not activate every destination")
        if item.get("trace_sha256"):
            outcomes = report.get("agentic_play_outcomes", [])
            if len(outcomes) != 1 or outcomes[0]["status"] != "completed" or outcomes[0]["settled_at_ms"] is None:
                raise ValueError("AgentX play did not settle successfully")
        if item.get("require_cache_pressure"):
            reference_config = deepcopy(config)
            cache = reference_config["engine"]["workers"]["aggregated"]["kv_cache"]
            cache["capacity"]["blocks"] = 1_048_576 // cache["block_size"]
            reference_runner = EngineReplayRunnerFactory(determinism="canonical_v1").create(0)
            try:
                reference = reference_runner.run(
                    prediction_to_replay_spec(CorePredictionConfig.model_validate(reference_config))
                ).metrics
            finally:
                reference_runner.close()
            extra = report["committed_prefill_tokens"] - reference["committed_prefill_tokens"]
            evidence["extra_prefill_tokens_under_pressure"] = extra
            if report["prefix_cache_reused_ratio"] <= 0 or extra <= 0:
                raise ValueError("cache case must show reuse and additional prefill versus a large cache")
    return {
        "status": "OK",
        "wall_time_ms": wall,
        "model_identity": identity,
        "model_provenance": provenance,
        "behavior": normalized,
        "coverage": evidence,
    }


def main() -> int:
    request = json.load(sys.stdin)
    item = request.get("case", {})
    response = {
        "protocol_version": PROTOCOL_VERSION,
        "case_id": item.get("case_id"),
        "case_hash": digest(item),
        "revision": request.get("revision"),
        "phase": request.get("phase"),
    }
    try:
        with contextlib.redirect_stdout(sys.stderr):
            response.update(run(request))
    except Exception as error:
        response.update(status="ERROR", error={"type": type(error).__name__, "message": str(error)})
    print(json.dumps(response, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
