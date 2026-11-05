# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare recorded benchmark conditions without using latency as evidence.

The consumer contract is Dynamo's measurement_protocol/benchmark_measurement
schema 1 (ai-dynamo/dynamo, commit 92cf45e2e6707ac32b1fee275f949270fdd61200).
Prompt hashes observe admission inputs, not generated tokens or cache tensors.
Effective worker configuration is checked separately by execution validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .native_artifact import NativeCollection, _rank_artifacts, select_native_measurements

_UNOBSERVED = [
    "sampled_continuation_token_ids",
    "kv_cache_tensors",
    "recurrent_state_tensors",
    "execution_history_equivalence",
    "observed_per_call_graph_dispatch",
]
_GRAPH_FIELDS = ("expected_cudagraph_mode", "expected_capture_size", "padding_tokens")
_RUNTIME_FIELDS = ("limits", "cudagraph", "recurrent_state")
_EXECUTION_CONTRACT = {
    "schema_version": 1,
    "sample_field": "benchmark_sample",
    "warmup_field": "warmup_evidence",
    "forward_index_scope": "rank_process",
    "timing_clock": "process_local_monotonic",
    "cudagraph_source": "ModelRunnerOutput.cudagraph_stats",
    "warmup_scope": "request_completion_and_existing_shape_validation",
}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _coordinates(point: dict[str, Any]) -> dict[str, Any]:
    result = {key: point[key] for key in ("batch_size", "total_kv_read_tokens")}
    if point["point_type"] == "prefill":
        result["total_prefill_tokens"] = point["total_prefill_tokens"]
        result.update({key: point[key] for key in ("partition", "rows") if point.get(key) is not None})
    return result


def _content_key(point: dict[str, Any]) -> str:
    # These fields are the producer's schema-1 content identity, distinct from
    # AISimulate's phase-cell-local coordinate key used by repeatability.
    identity = {key: point[key] for key in ("point_type", "batch_size")}
    identity.update({key: point.get(key, 0) for key in ("total_prefill_tokens", "total_kv_read_tokens")})
    identity.update({key: point.get(key) for key in ("partition", "rows")})
    return _hash(identity)


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _regime_gaps(point: dict[str, Any], regime: str | None) -> list[str]:
    reasons = [f"missing {key}" for key in _GRAPH_FIELDS if key not in point]
    if not isinstance(point.get("expected_cudagraph_mode"), str) or not point["expected_cudagraph_mode"]:
        reasons.append("missing expected CUDA graph mode")
    if point.get("expected_cudagraph_mode") not in {None, "NONE"}:
        if not _integer(point.get("expected_capture_size"), 1):
            reasons.append("missing expected CUDA graph capture size")
        if not _integer(point.get("padding_tokens")):
            reasons.append("missing expected CUDA graph padding")
    if not regime:
        reasons.append("missing KV initialization regime")
    return reasons


def _validate_protocol(protocol: Any) -> None:
    if not isinstance(protocol, dict):
        raise ValueError("measurement_protocol must be a mapping")
    fixed = {
        "schema_version": 1,
        "content_identity": "coordinate_rank_slot_v1",
        "prompt_hash_encoding": "uint32_le",
        "independent_repetitions": 1,
        "timing_metric": "scheduler_wall_time",
        "input_evidence_scope": "injected_prompt_token_ids",
    }
    if any(protocol.get(key) != value for key, value in fixed.items()) or not all(
        _integer(protocol.get(key), 1) for key in ("schema_version", "independent_repetitions")
    ):
        raise ValueError("measurement_protocol has unsupported identity, timing, or repetition semantics")
    if not all(isinstance(protocol.get(key), str) for key in ("content_seed", "synthetic_pool_tag")) or protocol.get(
        "synthetic_content"
    ) not in {"random", "sharegpt", "sharegpt_chain", "zeros"}:
        raise ValueError("measurement_protocol has malformed content settings")
    unobserved = protocol.get("unobserved")
    if not isinstance(unobserved, list) or not all(isinstance(item, str) and item for item in unobserved):
        raise ValueError("measurement_protocol must describe unobserved inputs")
    preparation = protocol.get("preparation")
    if (
        not isinstance(preparation, dict)
        or not all(
            _integer(preparation.get(key)) for key in ("warmup_iterations", "giant_kv_threshold", "giant_kv_repeats")
        )
        or not all(type(preparation.get(key)) is bool for key in ("prefill_real_seed", "decode_real_kv_warmup"))
    ):
        raise ValueError("measurement_protocol has malformed preparation settings")


def _validate_prompts(prompts: Any) -> bool:
    if not isinstance(prompts, dict) or prompts.get("status") not in {"recorded", "unavailable"}:
        raise ValueError("benchmark_measurement has malformed prompt evidence")
    requests = prompts.get("requests")
    if prompts["status"] == "unavailable":
        if prompts.get("sha256") is not None or requests != []:
            raise ValueError("unavailable prompt evidence contains observed hashes")
        return False
    if not isinstance(requests, list) or not requests:
        raise ValueError("recorded prompt evidence has no requests")
    for index, request in enumerate(requests):
        if (
            not isinstance(request, dict)
            or not _integer(request.get("slot"))
            or request["slot"] != index
            or not _integer(request.get("num_tokens"))
            or not _digest(request.get("sha256"))
        ):
            raise ValueError("recorded prompt evidence has malformed request hashes")
    if prompts.get("sha256") != _hash(requests):
        raise ValueError("recorded prompt evidence aggregate hash disagrees with requests")
    return True


def _validate_estimate(measurement: dict[str, Any], fpm: dict[str, Any], rank: int, benchmark_id: int) -> None:
    raw = measurement.get("raw_fpms")
    expected = measurement.get("expected_internal_samples")
    estimate = measurement.get("estimate")
    if (
        not _integer(expected, 1)
        or not isinstance(raw, list)
        or not raw
        or len(raw) > expected
        or not isinstance(estimate, dict)
    ):
        raise ValueError("benchmark_measurement has malformed internal sample evidence")
    if not _integer(fpm.get("counter_id")) or fpm["counter_id"] != benchmark_id:
        raise ValueError("benchmark_measurement retained counter does not match its point")
    for sample in raw:
        if (
            not isinstance(sample, dict)
            or type(sample.get("dp_rank")) is not int
            or sample["dp_rank"] != rank
            or not _integer(sample.get("counter_id"))
            or sample["counter_id"] != benchmark_id
            or isinstance(sample.get("wall_time"), bool)
            or not isinstance(sample.get("wall_time"), int | float)
            or not math.isfinite(sample["wall_time"])
            or sample["wall_time"] <= 0
        ):
            raise ValueError("benchmark_measurement has invalid raw counter/rank/timing evidence")
    method = estimate.get("method")
    if method == "single_step" and len(raw) == 1:
        indices = [0]
    elif method == "last_step" and expected > 1 and len(raw) == expected:
        indices = [len(raw) - 1]
    elif method == "adjacent_upper_median" and expected > 2 and len(raw) > 1:
        indices = list(range(1, len(raw)))
    else:
        raise ValueError("benchmark_measurement has inconsistent reduction method")
    if estimate.get("raw_sample_indices") != indices or any(
        type(index) is not int for index in estimate["raw_sample_indices"]
    ):
        raise ValueError("benchmark_measurement has inconsistent raw sample indices")
    walls = sorted(raw[index]["wall_time"] for index in indices)
    if not math.isclose(walls[len(walls) // 2], fpm["wall_time"], rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("benchmark_measurement reduction disagrees with retained timing")


def _read_measurement(
    row: dict[str, Any], *, rank: int, grid_digest: str, position: int, protocol: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, list[str]]:
    fpm = row["fpms"][0]
    if protocol is None:
        if "benchmark_measurement" in fpm:
            raise ValueError("benchmark_measurement has no measurement_protocol")
        return None, ["missing measurement_protocol and benchmark_measurement"]
    measurement = fpm.get("benchmark_measurement")
    if (
        not isinstance(measurement, dict)
        or type(measurement.get("schema_version")) is not int
        or measurement["schema_version"] != 1
        or type(measurement.get("dp_rank")) is not int
        or measurement["dp_rank"] != rank
        or measurement.get("point_key") != _content_key(row["point"])
    ):
        raise ValueError("benchmark_measurement has missing or inconsistent rank/point identity")
    preparation = measurement.get("preparation")
    if (
        not isinstance(preparation, dict)
        or preparation.get("grid_digest") != grid_digest
        or not _integer(preparation.get("completed_points_before"))
        or preparation["completed_points_before"] != position
        or preparation.get("kv_seed_regime") != row.get("kv_seed_regime")
    ):
        raise ValueError("benchmark_measurement has inconsistent grid/order/KV preparation")
    recorded = _validate_prompts(measurement.get("prompts"))
    _validate_estimate(measurement, fpm, rank, row["point"]["benchmark_id"])
    return measurement, [] if recorded else [f"rank {rank}: injected prompt hashes unavailable"]


def _scheduling_observations(raw_root: Path, artifacts, collection) -> tuple[dict[str, Any], list[str]]:
    """Read resolved scheduler/worker state, never derive it from launch defaults."""
    observations = {str(rank): [] for rank, *_ in collection.rank_timings}
    for path, payload in artifacts:
        engine = payload.get("engine")
        if not isinstance(engine, dict) or engine.get("capture_error"):
            continue
        scheduler = engine.get("scheduler")
        if isinstance(scheduler, dict) and "async_scheduling" in scheduler:
            observations[str(payload["dp"]["rank"])].append(
                {"async_scheduling": scheduler["async_scheduling"], "source": str(path)}
            )
    for path in sorted(raw_root.glob("**/fpm-execution-worker-*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or payload.get("status") != "observed":
            continue
        provenance = payload.get("collector_provenance", {})
        rank = str(payload.get("dp_rank"))
        if (
            payload.get("schema_name") != "aisimulate_fpm_runtime_execution"
            or payload.get("schema_version") != 1
            or not isinstance(provenance, dict)
            or provenance.get("attempt_id") != collection.collector_attempt_id
            or payload.get("backend_version") != collection.backend_version
            or rank not in observations
        ):
            raise ValueError("scheduling observation has inconsistent worker provenance")
        config = payload.get("resolved_config")
        scheduler = config.get("scheduler_config") if isinstance(config, dict) else None
        if isinstance(scheduler, dict) and "async_scheduling" in scheduler:
            observations[rank].append({"async_scheduling": scheduler["async_scheduling"], "source": str(path)})
    result, missing = {}, []
    for rank, items in observations.items():
        if any(item["async_scheduling"] is not None and type(item["async_scheduling"]) is not bool for item in items):
            raise ValueError("observed async_scheduling must be Boolean")
        modes = {item["async_scheduling"] for item in items if item["async_scheduling"] is not None}
        if len(modes) > 1:
            raise ValueError(f"rank {rank}: native scheduler and workers disagree on async_scheduling")
        result[rank] = {"async_scheduling": next(iter(modes)) if modes else None, "observations": items}
        if not modes:
            missing.append(f"rank {rank}: effective async_scheduling is unreported")
    if len({item["async_scheduling"] for item in result.values()} - {None}) > 1:
        raise ValueError("native ranks disagree on effective async_scheduling")
    return result, missing


def _warmup_history(payload: dict[str, Any], protocol: dict[str, Any] | None):
    if protocol is None or "execution_evidence" not in protocol:
        return None, []
    if protocol["execution_evidence"] != _EXECUTION_CONTRACT:
        raise ValueError("unsupported measurement execution_evidence contract")
    history = payload.get("warmup_evidence")
    if (
        not isinstance(history, dict)
        or history.get("status") not in {"recorded", "unavailable"}
        or not isinstance(history.get("records"), list)
        or (history["status"] == "unavailable" and history["records"])
    ):
        raise ValueError("invalid warmup_evidence status or records")
    prefixes = [_hash([])]
    digest = hashlib.sha256()
    previous_completed = 0
    previous_start = 0
    for record in history["records"]:
        if (
            not isinstance(record, dict)
            or record.get("kind") not in {"global", "eager_shape", "real_prefix_seed", "real_prefix_shape"}
            or record.get("status") not in {"running", "completed", "failed"}
            or not isinstance(record.get("validation"), dict)
            or record["validation"].get("status") not in {"not_performed", "passed", "failed"}
            or not _integer(record.get("completed_points_before"))
        ):
            raise ValueError("malformed warmup_evidence record")
        start, end = record.get("forward_index_start"), record.get("forward_index_end")
        count = record.get("observed_forward_count")
        if (
            not _integer(count)
            or any(value is not None and not _integer(value) for value in (start, end))
            or (start is not None and end is not None and (end < start or count != end - start))
            or (record["status"] == "running") != (end is None)
            or (start is None and (record["kind"] != "eager_shape" or record["status"] != "failed" or count != 0))
            or (record["validation"]["status"] == "passed" and record["status"] != "completed")
            or (record["validation"]["status"] == "failed" and record["status"] != "failed")
        ):
            raise ValueError("warmup_evidence has contradictory completion or forward indices")
        completed = record["completed_points_before"]
        if (
            not previous_completed <= completed <= len(payload["results"])
            or (record["kind"] == "global" and completed != 0)
            or (start is not None and start < previous_start)
        ):
            raise ValueError("warmup_evidence has impossible preparation order")
        previous_completed = completed
        if start is not None:
            previous_start = start
        digest.update(_hash(record).encode())
        prefixes.append(digest.hexdigest())
    return history, prefixes


def _observed_execution(measurement, fpm, history, prefixes, previous_forward):
    """Validate within one rank/process; absolute clocks are not compared across runs."""
    if "benchmark_sample" in fpm:
        raise ValueError("retained estimate cannot claim a single benchmark_sample")
    observed, reasons = [], []
    preceding_forward = previous_forward
    for index, raw in enumerate(measurement["raw_fpms"]):
        sample = raw.get("benchmark_sample")
        if (
            not isinstance(sample, dict)
            or not _integer(sample.get("sample_index"))
            or sample["sample_index"] != index
            or not _integer(sample.get("forward_index"))
            or sample["forward_index"] <= previous_forward
        ):
            raise ValueError("benchmark_sample has invalid sample or process-local forward index")
        previous_forward = sample["forward_index"]
        timing, graph = sample.get("timing"), sample.get("cudagraph")
        if not isinstance(timing, dict) or timing.get("basis") not in {"schedule_to_output", "inter_output"}:
            raise ValueError("benchmark_sample has invalid timing basis")
        start, end = timing.get("start_monotonic"), timing.get("end_monotonic")
        if (
            type(end) not in {int, float}
            or not math.isfinite(end)
            or (
                start is not None
                and (
                    type(start) not in {int, float}
                    or not math.isfinite(start)
                    or start > end
                    or not math.isclose(end - start, raw["wall_time"], rel_tol=1e-9, abs_tol=1e-9)
                )
            )
        ):
            raise ValueError("benchmark_sample contradicts its local timing interval")
        if start is None:
            reasons.append("sample timing start is unavailable")
        if not isinstance(graph, dict) or graph.get("status") not in {"observed", "unavailable"}:
            raise ValueError("benchmark_sample has invalid graph observation status")
        values = [
            graph.get(name) for name in ("runtime_mode", "num_unpadded_tokens", "num_padded_tokens", "num_paddings")
        ]
        if graph["status"] == "unavailable":
            if any(value is not None for value in values):
                raise ValueError("unavailable graph observation contains observed values")
            reasons.append("per-sample CUDA graph dispatch is unavailable")
        elif (
            values[0] not in {"NONE", "PIECEWISE", "FULL"}
            or not all(_integer(value) for value in values[1:])
            or values[2] - values[1] != values[3]
        ):
            raise ValueError("benchmark_sample has invalid observed graph dispatch or padding")
        observed.append(
            {
                "sample_index": index,
                "forward_index": previous_forward,
                "timing_basis": timing["basis"],
                "cudagraph": graph,
            }
        )
    count = measurement["preparation"].get("warmup_records_before")
    prefix = None
    if history["status"] == "unavailable":
        if count is not None:
            raise ValueError("unavailable warmup history claims an observed record count")
        reasons.append("warmup history is unavailable")
    elif not _integer(count) or count >= len(prefixes):
        raise ValueError("measurement has invalid warmup_records_before")
    else:
        prefix = prefixes[count]
        first_forward = observed[0]["forward_index"]
        completed = measurement["preparation"]["completed_points_before"]
        if count != sum(record["completed_points_before"] <= completed for record in history["records"]):
            raise ValueError("measurement warmup_records_before contradicts preparation order")
        for record in history["records"][:count]:
            start, end = record["forward_index_start"], record["forward_index_end"]
            if (start is not None and start > first_forward) or (end is not None and end > first_forward):
                raise ValueError("recorded preparation ends after its measurement")
            if record["completed_points_before"] == completed and any(
                value is not None and value <= preceding_forward for value in (start, end)
            ):
                raise ValueError("recorded warmup preparation precedes its completed point count")
            if record["status"] == "running":
                prefix = None
                reasons.append("referenced warmup preparation is still running")
    return {"samples": observed, "warmup_prefix_sha256": prefix}, reasons, previous_forward


def extract_measurement_evidence(
    raw_root: Path, collection: NativeCollection, *, evidence_version: int = 2
) -> dict[str, Any]:
    """Read schema-1 evidence for the publication-selected native samples.

    Call after native validation. Missing legacy evidence is explicitly
    unestablished; malformed or contradictory new evidence is rejected.
    Raw adjacent steps are retained as evidence of one launch, never counted as
    independent observations. Duplicate consolidation uses publication's rule.
    """
    if type(evidence_version) is not int or evidence_version not in {1, 2}:
        raise ValueError("unsupported measurement evidence version")
    artifacts = _rank_artifacts(raw_root)
    if not artifacts:
        raise ValueError("measurement evidence has no native rank artifacts")
    protocol = artifacts[0][1].get("measurement_protocol")
    present = "measurement_protocol" in artifacts[0][1]
    if present:
        _validate_protocol(protocol)
    expected_ranks = {rank for rank, *_ in collection.rank_timings}
    measurements = {item.point["benchmark_id"]: item for item in collection.points}
    selected = select_native_measurements(collection, cell_id="measurement evidence")
    selected_ids = {item.point["benchmark_id"] for item in selected}
    points = {
        _hash(_coordinates(item.point)): {
            "point": item.point,
            "kv_seed_regime": item.kv_seed_regime,
            "regime": {key: item.point.get(key) for key in _GRAPH_FIELDS},
            "wall_time_seconds": max(wall for _, wall in item.rank_wall_times),
            "ranks": {},
            "reasons": _regime_gaps(item.point, item.kv_seed_regime),
        }
        for item in selected
    }
    result: dict[str, Any] = {
        "schema_version": evidence_version,
        "backend_version": collection.backend_version,
        "collector_attempt_id": collection.collector_attempt_id,
        "runtime_run_id": collection.runtime_run_id,
        "runtime_grid_digest": collection.runtime_grid_digest,
        "measurement_protocol": protocol,
        "runtime_identity": {},
        "benchmark_config": {},
        "points": points,
        "raw_files": [],
        "reasons": [] if present else ["legacy artifact has no measurement protocol"],
        "unobserved": sorted(set(_UNOBSERVED + (protocol or {}).get("unobserved", []))),
    }
    if evidence_version == 2:
        result["scheduling"], scheduling_gaps = _scheduling_observations(raw_root, artifacts, collection)
        result["reasons"].extend(scheduling_gaps)
        result["warmup_evidence"] = {}
        result["unobserved"] = sorted(
            set(
                result["unobserved"]
                + [
                    "complete_cuda_graph_descriptor",
                    "capture_versus_replay",
                    "gpu_execution_time",
                    "warmup_graph_kernel_and_cache_equivalence",
                    "decode_real_kv_chain_warmup_history",
                ]
            )
        )
    seen_ranks = set()
    execution_order = None
    for path, payload in artifacts:
        if ("measurement_protocol" in payload) != present or payload.get("measurement_protocol") != protocol:
            raise ValueError("native ranks disagree on measurement_protocol (including mixed legacy/new evidence)")
        rank = payload["dp"]["rank"]
        if not _integer(rank) or rank not in expected_ranks or rank in seen_ranks:
            raise ValueError("measurement evidence has inconsistent DP ranks")
        seen_ranks.add(rank)
        history, prefixes = _warmup_history(payload, protocol) if evidence_version == 2 else (None, [])
        previous_forward = -1
        if history is not None:
            result["warmup_evidence"][str(rank)] = history
        elif evidence_version == 2:
            result["reasons"].append(f"rank {rank}: per-sample execution and warmup evidence is unreported")
        if (payload.get("run_id"), payload.get("grid_digest")) != (
            collection.runtime_run_id,
            collection.runtime_grid_digest,
        ):
            raise ValueError("measurement evidence differs from validated native run/grid identity")
        runtime = {key: payload.get(key) for key in _RUNTIME_FIELDS}
        result["runtime_identity"][str(rank)] = runtime
        config = payload.get("config")
        result["benchmark_config"][str(rank)] = (
            {key: value for key, value in config.items() if key != "output_path"} if isinstance(config, dict) else None
        )
        for key, value in runtime.items():
            if not isinstance(value, dict) or not value:
                result["reasons"].append(f"rank {rank}: missing native {key} evidence")
        if not isinstance(config, dict) or not config:
            result["reasons"].append(f"rank {rank}: missing benchmark config")
        rows = payload["results"]
        if len(rows) != len(measurements) or {row["point"]["benchmark_id"] for row in rows} != set(measurements):
            raise ValueError("measurement evidence differs from validated native point set")
        order = [row["point"]["benchmark_id"] for row in rows]
        if execution_order is None:
            execution_order = order
        elif present and order != execution_order:
            raise ValueError("native ranks disagree on recorded measurement order")
        prefix = hashlib.sha256()
        prefix_missing = False
        for position, row in enumerate(rows):
            benchmark_id = row["point"]["benchmark_id"]
            source = measurements[benchmark_id]
            if (
                row["point"] != source.point
                or row.get("kv_seed_regime") != source.kv_seed_regime
                or row["fpms"][0]["wall_time"] != dict(source.rank_wall_times)[rank]
            ):
                raise ValueError("measurement evidence differs from validated native sample")
            measurement, reasons = _read_measurement(
                row, rank=rank, grid_digest=collection.runtime_grid_digest, position=position, protocol=protocol
            )
            prompt_missing = bool(reasons)
            if history is not None:
                execution, gaps, previous_forward = _observed_execution(
                    measurement, row["fpms"][0], history, prefixes, previous_forward
                )
                measurement = {**measurement, "observed_execution": execution}
                reasons.extend(f"rank {rank}: {reason}" for reason in gaps)
            if benchmark_id in selected_ids:
                point = points[_hash(_coordinates(row["point"]))]
                point["ranks"][str(rank)] = (
                    {**measurement, "execution_prefix_sha256": prefix.hexdigest()} if measurement is not None else None
                )
                point["reasons"].extend(reasons)
                if prefix_missing:
                    point["reasons"].append(f"rank {rank}: preceding prompt evidence incomplete")
            prefix_missing |= prompt_missing
            preparation = measurement["preparation"] if measurement else None
            if evidence_version == 2 and preparation is not None:
                preparation = {key: value for key, value in preparation.items() if key != "warmup_records_before"}
            prefix.update(
                _hash(
                    {
                        "point": row["point"],
                        "prompts": measurement["prompts"] if measurement else None,
                        "preparation": preparation,
                    }
                ).encode()
            )
        result["raw_files"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if seen_ranks != expected_ranks:
        raise ValueError("measurement evidence is missing native ranks")
    for point in points.values():
        point["status"] = "unestablished" if point["reasons"] else "recorded"
    if (
        evidence_version == 2
        and protocol is not None
        and "execution_evidence" in protocol
        and all(
            sample["cudagraph"]["status"] == "observed"
            for point in points.values()
            for measurement in point["ranks"].values()
            for sample in measurement["observed_execution"]["samples"]
        )
    ):
        result["unobserved"].remove("observed_per_call_graph_dispatch")
    result["status"] = (
        "unestablished" if result["reasons"] or any(point["reasons"] for point in points.values()) else "recorded"
    )
    return result


def compare_measurements(
    reference: dict[str, Any], candidate: dict[str, Any], *, point_key: str, same_context: bool = True
) -> dict[str, Any]:
    """Assess observed identity first, keeping cross-grid history unestablished.

    Comparable means matching recorded conditions for this observation scope;
    it does not prove identical cache tensors, per-call dispatch, or stability.
    Latency never determines comparability and is deliberately not inspected.
    """
    missing = []
    mismatches = []
    extended = reference.get("schema_version") == 2 and candidate.get("schema_version") == 2
    for label, evidence in (("reference", reference), ("candidate", candidate)):
        missing.extend(f"{label}: {reason}" for reason in evidence.get("reasons", []))
        if evidence.get("schema_version") not in {1, 2} or evidence.get("measurement_protocol") is None:
            missing.append(f"{label}: missing supported measurement evidence")
        if not evidence.get("runtime_run_id"):
            missing.append(f"{label}: missing independent runtime run identity")
    for key in ("backend_version", "measurement_protocol", "runtime_identity"):
        left_value, right_value = reference.get(key), candidate.get(key)
        if extended and key == "measurement_protocol":
            left_value, right_value = (
                {name: value for name, value in item.items() if name not in {"execution_evidence", "unobserved"}}
                if isinstance(item, dict)
                else item
                for item in (left_value, right_value)
            )
        if left_value is not None and right_value is not None and left_value != right_value:
            mismatches.append(f"{key} differs")
    if reference.get("schema_version") == 2 or candidate.get("schema_version") == 2:
        for rank in sorted(set(reference.get("scheduling", {})) | set(candidate.get("scheduling", {}))):
            values = [
                item.get("scheduling", {}).get(rank, {}).get("async_scheduling") for item in (reference, candidate)
            ]
            if any(type(value) is not bool for value in values):
                missing.append(f"rank {rank}: effective async_scheduling comparison is unestablished")
            elif values[0] != values[1]:
                mismatches.append(f"rank {rank}: effective async_scheduling differs")
    if reference.get("runtime_run_id") and reference.get("runtime_run_id") == candidate.get("runtime_run_id"):
        mismatches.append("runtime_run_id is not an independent launch")
    left = reference.get("points", {}).get(point_key)
    right = candidate.get("points", {}).get(point_key)
    if left is None or right is None:
        missing.append("point is missing from reference or candidate")
    else:
        for label, point in (("reference", left), ("candidate", right)):
            missing.extend(f"{label}: {reason}" for reason in point.get("reasons", []))
        if _content_key(left["point"]) != _content_key(right["point"]):
            mismatches.append("point shape/partition differs")
        for key in ("kv_seed_regime", "regime"):
            if left.get(key) != right.get(key):
                mismatches.append(f"{key} differs")
        if set(left["ranks"]) != set(right["ranks"]):
            mismatches.append("DP rank set differs")
        for rank in sorted(set(left["ranks"]) & set(right["ranks"])):
            lhs, rhs = left["ranks"][rank], right["ranks"][rank]
            if lhs is None or rhs is None:
                missing.append(f"rank {rank}: missing measurement evidence")
                continue
            for key in ("point_key", "prompts", "expected_internal_samples", "estimate"):
                if lhs[key] != rhs[key]:
                    mismatches.append(f"rank {rank}: {key} differs")
            observed_left, observed_right = lhs.get("observed_execution"), rhs.get("observed_execution")
            if observed_left is not None and observed_right is not None:
                # Relative forward order belongs to matched-sweep preparation;
                # absolute process-local timestamps never establish equality.
                fields = (
                    ("sample_index", "timing_basis", "forward_index")
                    if same_context
                    else ("sample_index", "timing_basis")
                )
                left_samples, right_samples = observed_left["samples"], observed_right["samples"]
                if len(left_samples) != len(right_samples):
                    mismatches.append(f"rank {rank}: observed sample count differs")
                for a, b in zip(left_samples, right_samples, strict=False):
                    if any(a[name] != b[name] for name in fields):
                        mismatches.append(f"rank {rank}: observed sample order or timing basis differs")
                    if a["cudagraph"]["status"] == b["cudagraph"]["status"] == "observed":
                        if a["cudagraph"] != b["cudagraph"]:
                            mismatches.append(f"rank {rank}: observed graph dispatch differs")
                    else:
                        missing.append(f"rank {rank}: observed graph dispatch comparison is unestablished")
                if same_context:
                    if observed_left["warmup_prefix_sha256"] is None or observed_right["warmup_prefix_sha256"] is None:
                        missing.append(f"rank {rank}: observed warmup comparison is unestablished")
                    elif observed_left["warmup_prefix_sha256"] != observed_right["warmup_prefix_sha256"]:
                        mismatches.append(f"rank {rank}: observed warmup preparation differs")
            elif extended or (observed_left is None) != (observed_right is None):
                missing.append(f"rank {rank}: per-sample execution comparison is unestablished")
            if same_context:
                for key in ("preparation", "execution_prefix_sha256"):
                    left_value, right_value = lhs[key], rhs[key]
                    if extended and key == "preparation":
                        left_value, right_value = (
                            {name: value for name, value in item.items() if name != "warmup_records_before"}
                            for item in (left_value, right_value)
                        )
                    if left_value != right_value:
                        mismatches.append(f"rank {rank}: {key} differs")
    if same_context:
        for key in ("runtime_grid_digest", "benchmark_config"):
            if reference.get(key) is not None and candidate.get(key) is not None and reference[key] != candidate[key]:
                mismatches.append(f"{key} differs")
    identity_status = "mismatch" if mismatches else "unestablished" if missing else "comparable"
    reasons = sorted({*mismatches, *missing})
    if not same_context:
        reasons.append("cross-context execution history equivalence has not been established")
    return {
        "status": "unestablished" if not same_context and identity_status == "comparable" else identity_status,
        "identity_status": identity_status,
        "reasons": reasons,
        "unobserved": sorted(
            set(
                (
                    _UNOBSERVED
                    if reference.get("schema_version") != 2 or candidate.get("schema_version") != 2
                    else _UNOBSERVED[:-1]
                )
                + reference.get("unobserved", [])
                + candidate.get("unobserved", [])
            )
        ),
        "scope": "recorded measurement protocol and injected prompts; worker runtime validation remains required",
        "context": "matched_native_sweep" if same_context else "cross_context_unproven",
    }
