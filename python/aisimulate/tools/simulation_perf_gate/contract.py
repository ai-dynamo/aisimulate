# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed protocol-v3 comparison fields, independent of native diagnostic schemas."""

import math
from pathlib import Path

MODEL_FIELDS = (
    "model",
    "system",
    "backend",
    "backend_version",
    "worker_type",
    "tp",
    "pp",
    "attention_dp",
    "moe_tp_size",
    "moe_ep_size",
    "attention_backend",
    "gemm_quant_mode",
    "moe_quant_mode",
    "kvcache_quant_mode",
    "fmha_quant_mode",
    "fpm_fmha_quant_mode",
    "comm_quant_mode",
    "kv_block_size",
)
COUNTS = (
    "num_requests",
    "completed_requests",
    "total_input_tokens",
    "total_output_tokens",
    "committed_prefill_tokens",
    "num_ttft_samples",
    "num_tpot_samples",
    "num_e2e_latency_samples",
)
DISTRIBUTION_STATS = ("mean", "min", "max", "median", "p75", "p90", "p95", "p99", "std")
METRICS = ("duration_ms", "prefix_cache_reused_ratio", "first_admission_prefix_cache_reused_ratio") + tuple(
    f"{stat}_{metric}_ms" for metric in ("ttft", "ttst", "tpot", "itl", "e2e_latency") for stat in DISTRIBUTION_STATS
)
TRAJECTORY_COUNTS = ("total_trajectories", "completed_trajectories", "incomplete_trajectories")
TRAJECTORY_METRICS = tuple(f"{stat}_trajectory_e2e_latency_ms" for stat in (*DISTRIBUTION_STATS, "p50"))
REQUEST_FIELDS = (
    "uuid",
    "session_id",
    "turn_index",
    "terminal_status",
    "arrival_time_ms",
    "dispatched_at_ms",
    "first_admit_ms",
    "terminal_time_ms",
    "first_token_ms",
    "last_token_ms",
    "ttft_ms",
    "ttst_ms",
    "e2e_latency_ms",
    "itl_ms",
    "input_length",
    "requested_output_length",
    "output_length",
    "reused_input_tokens",
    "prefill_worker_idx",
    "decode_worker_idx",
    "prefill_admit_ms",
    "source_held_ms",
    "destination_reserved_ms",
    "destination_activated_ms",
    "decode_admit_ms",
    "source_released_ms",
    "decode_reused_input_tokens",
    "prefill_route_overlap_tokens",
    "decode_route_overlap_tokens",
    "admission_count",
    "readmission_count",
)
ROUTING_COUNTS = (
    "logical_worker_id",
    "scheduler_id",
    "dp_rank",
    "reported_overlap_tokens",
    "selected_overlap_blocks",
    "best_available_overlap_blocks",
    "overlap_regret_blocks",
    "placement_replica_id",
)
ROUTING_FIELDS = ("pool", "outcome", "queue_entered_at_ms", "released_at_ms", "queue_wait_ms") + ROUTING_COUNTS
ADMISSION_FIELDS = (
    "admission_ordinal",
    "pool_admission_ordinal",
    "pool",
    "at_ms",
    "reused_input_tokens",
    "is_readmission",
)
AGENT_FIELDS = ("request_id", "play_id", "conversation_id")
PLAY_FIELDS = ("play_id", "status", "causal_terminal_ms", "settled_at_ms")


def check_finite(value: object) -> None:
    """Check every number, including diagnostics, without encoding another JSON copy."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite number in worker data")
    elif isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, list):
        for item in value:
            check_finite(item)


def check_evidence(response: dict, case: dict, phase: str, summary: dict) -> None:
    """Require role-specific real models and coverage even after records are archived."""
    engine = case["config"]["engine"]
    roles = {"prefill", "decode"} if engine["mode"] == "disaggregated" else {"aggregated"}
    identity = model_identity(response.get("model_identity"))
    provenance = response.get("model_provenance")
    if set(identity) != roles or not isinstance(provenance, dict) or set(provenance) != roles:
        raise ValueError("missing role-specific model provenance or identity")
    for role in roles:
        model = provenance[role]
        projected = fields(model, MODEL_FIELDS, f"/model_provenance/{role}")
        if projected != {key: identity[role][key] for key in MODEL_FIELDS}:
            raise ValueError(f"{role}: model provenance differs from identity")
        worker = engine["workers"][role]
        parallelism = worker["parallelism"]
        expected = {
            "provider": "aic",
            "estimation_mode": "op_level",
            "fallback_policy": "deny",
            "database_mode": "SILICON",
            "enable_shared_layer": True,
            "worker_type": role,
            "tp": parallelism["tensor"],
            "pp": parallelism["pipeline"],
            "attention_dp": parallelism["attention_data"],
            "kv_block_size": worker["kv_cache"]["block_size"],
            "model": engine["model"],
            "system": engine["hardware"],
            "backend": engine["backend"],
            "backend_version": engine["backend_version"],
        }
        if any(type(model.get(key)) is not type(value) or model[key] != value for key, value in expected.items()):
            raise ValueError(f"{role}: invalid real-model provenance")
        paths = model.get("systems_paths")
        if (
            identity[role]["systems_paths"] != ["package:aisimulate_core/systems"]
            or not isinstance(paths, list)
            or not paths
            or any(not isinstance(path, str) or not Path(path).is_absolute() for path in paths)
        ):
            raise ValueError(f"{role}: missing packaged model provenance")
    coverage = response.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("missing coverage evidence")
    if phase != "equivalence":
        return
    if case.get("require_cache_pressure"):
        extra = coverage.get("extra_prefill_tokens_under_pressure")
        if (
            type(extra) not in (int, float)
            or extra <= 0
            or int(extra) != extra
            or summary["prefix_cache_reused_ratio"] <= 0
        ):
            raise ValueError("cache coverage requires reuse and additional prefill under pressure")
    if engine["mode"] == "disaggregated":
        activated = coverage.get("pd_activated_requests")
        if type(activated) is not int or activated != case["expected_requests"]:
            raise ValueError("P/D coverage requires every destination activation")
        if "per_request" in summary and any(row["destination_activated_ms"] is None for row in summary["per_request"]):
            raise ValueError("P/D records are missing a destination activation")


def fields(value: dict, required: tuple[str, ...], path: str, optional: tuple[str, ...] = ()) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    missing = sorted(set(required) - value.keys())
    if missing:
        raise ValueError(f"{path}: missing fields {', '.join(missing)}")
    return {**{key: value[key] for key in required}, **{key: value.get(key) for key in optional}}


def records(value: object, path: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected records")
    return value


def check_counts(value: dict, names: tuple[str, ...], path: str, *, nullable: bool = False) -> None:
    for name in names:
        count = value[name]
        if nullable and count is None:
            continue
        if type(count) is not int or count < 0:
            raise ValueError(f"{path}/{name}: invalid count")


def model_identity(value: dict) -> dict:
    if not isinstance(value, dict) or not value:
        raise ValueError("missing model identity")
    result = {
        role: fields(model, (*MODEL_FIELDS, "systems_paths"), f"/model_identity/{role}")
        for role, model in value.items()
    }
    for role, model in result.items():
        for name in ("model", "system", "backend", "backend_version", "worker_type"):
            if not isinstance(model[name], str) or not model[name]:
                raise ValueError(f"/model_identity/{role}/{name}: invalid identity")
        check_counts(model, ("tp", "pp", "attention_dp", "kv_block_size"), f"/model_identity/{role}")
        check_counts(model, ("moe_tp_size", "moe_ep_size"), f"/model_identity/{role}", nullable=True)
    return result


def request_behavior(row: dict, path: str, *, agentic: bool) -> dict:
    result = fields(row, REQUEST_FIELDS, path, ("request_id", "play_id"))
    check_counts(
        result,
        (
            "input_length",
            "requested_output_length",
            "output_length",
            "reused_input_tokens",
            "admission_count",
            "readmission_count",
        ),
        path,
    )
    check_counts(
        result,
        (
            "turn_index",
            "prefill_worker_idx",
            "decode_worker_idx",
            "decode_reused_input_tokens",
            "prefill_route_overlap_tokens",
            "decode_route_overlap_tokens",
        ),
        path,
        nullable=True,
    )
    for name, required in (("routing_history", ROUTING_FIELDS), ("admission_history", ADMISSION_FIELDS)):
        result[name] = [
            fields(item, required, f"{path}/{name}/{index}")
            for index, item in enumerate(records(row.get(name), f"{path}/{name}"))
        ]
    for index, route in enumerate(result["routing_history"]):
        check_counts(route, ROUTING_COUNTS, f"{path}/routing_history/{index}", nullable=True)
    for index, admission in enumerate(result["admission_history"]):
        check_counts(
            admission,
            ("admission_ordinal", "pool_admission_ordinal", "reused_input_tokens"),
            f"{path}/admission_history/{index}",
        )
    if agentic:
        fields(row, ("request_id", "play_id"), path)
        result["agentic"] = fields(
            row.get("agentic"), AGENT_FIELDS, f"{path}/agentic", ("lane_id", "root_id", "parent_id", "cache_id")
        )
    return result


def behavior(report: dict, *, per_request: bool, agentic: bool = False) -> dict:
    """Project results for comparison only; callers retain the unmodified report."""
    counts = COUNTS + (TRAJECTORY_COUNTS if agentic else ())
    metrics = METRICS + (TRAJECTORY_METRICS if agentic else ())
    result = fields(report, counts + metrics, "/behavior")
    check_counts(result, counts, "/behavior")
    for name in metrics:
        value = result[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"/behavior/{name}: invalid metric")
    if agentic:
        outcomes = records(report.get("agentic_play_outcomes"), "/behavior/agentic_play_outcomes")
        result["agentic_play_outcomes"] = [
            fields(row, PLAY_FIELDS, f"/behavior/agentic_play_outcomes/{index}") for index, row in enumerate(outcomes)
        ]
        if len(outcomes) != 1 or outcomes[0]["status"] != "completed" or outcomes[0]["settled_at_ms"] is None:
            raise ValueError("/behavior/agentic_play_outcomes: incomplete play")
        if result["incomplete_trajectories"] or result["completed_trajectories"] != result["total_trajectories"]:
            raise ValueError("/behavior: incomplete trajectories")
        for name in ("causal_terminal_ms", "settled_at_ms"):
            value = outcomes[0][name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"/behavior/agentic_play_outcomes/0/{name}: invalid time")
    if per_request:
        result["per_request"] = [
            request_behavior(row, f"/per_request/{index}", agentic=agentic)
            for index, row in enumerate(records(report.get("per_request"), "/per_request"))
        ]
    return result
