# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest
from collector.fpm_forward.measurement_evidence import compare_measurements, extract_measurement_evidence
from collector.fpm_forward.native_artifact import NativeCollection, NativePointMeasurement

pytestmark = pytest.mark.unit


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _add_execution_protocol(payload, *, clock_offset=0.0):
    # Synthetic observations of the documented Dynamo #15110 contract, not
    # expectations reconstructed from an engine configuration.
    payload["measurement_protocol"]["execution_evidence"] = {
        "schema_version": 1,
        "sample_field": "benchmark_sample",
        "warmup_field": "warmup_evidence",
        "forward_index_scope": "rank_process",
        "timing_clock": "process_local_monotonic",
        "cudagraph_source": "ModelRunnerOutput.cudagraph_stats",
        "warmup_scope": "request_completion_and_existing_shape_validation",
    }
    payload["warmup_evidence"] = {
        "status": "recorded",
        "records": [
            {
                "kind": "global",
                "requested_shape": {"prompt_lengths": [8]},
                "status": "completed",
                "forward_index_start": 0,
                "forward_index_end": 3,
                "completed_points_before": 0,
                "observed_forward_count": 3,
                "first_scheduled_requests": {"num_requests": 1},
                "last_scheduled_requests": {"num_requests": 1},
                "validation": {"status": "not_performed"},
            }
        ],
    }
    forward_index = 3
    for index, row in enumerate(payload["results"]):
        point = row["point"]
        group = next(item for item in payload["iteration_groups"] if item["point"] == point)
        fpms = [*row["fpms"], *(fpm for rank in group["rank_results"] for fpm in rank["fpms"])]
        for fpm in fpms:
            measurement = fpm["benchmark_measurement"]
            measurement["preparation"]["warmup_records_before"] = 1
            for sample_index, raw in enumerate(measurement["raw_fpms"]):
                start = clock_offset + measurement["dp_rank"] * 1000 + index + sample_index
                tokens = point["total_prefill_tokens"] or point["batch_size"]
                padded = point["expected_capture_size"] or tokens
                raw["benchmark_sample"] = {
                    "sample_index": sample_index,
                    "forward_index": forward_index + sample_index,
                    "timing": {
                        "basis": "schedule_to_output" if sample_index == 0 else "inter_output",
                        "start_monotonic": start,
                        "end_monotonic": start + raw["wall_time"],
                    },
                    "cudagraph": {
                        "status": "observed",
                        "runtime_mode": point["expected_cudagraph_mode"],
                        "num_unpadded_tokens": tokens,
                        "num_padded_tokens": padded,
                        "num_paddings": padded - tokens,
                    },
                }
        forward_index += len(row["fpms"][0]["benchmark_measurement"]["raw_fpms"])


def _remove_execution_protocol(payload):
    payload["measurement_protocol"].pop("execution_evidence", None)
    payload.pop("warmup_evidence", None)
    fpms = [fpm for row in payload["results"] for fpm in row["fpms"]]
    fpms.extend(fpm for group in payload["iteration_groups"] for rank in group["rank_results"] for fpm in rank["fpms"])
    for fpm in fpms:
        measurement = fpm["benchmark_measurement"]
        measurement["preparation"].pop("warmup_records_before", None)
        for raw in measurement["raw_fpms"]:
            raw.pop("benchmark_sample", None)


def _add_measurement_protocol(payload, *, seed="0", execution=True):
    """Add the producer's observed metadata to a synthetic native rank artifact."""
    payload["measurement_protocol"] = {
        "schema_version": 1,
        "content_identity": "coordinate_rank_slot_v1",
        "content_seed": seed,
        "synthetic_content": "random",
        "synthetic_pool_tag": "",
        "prompt_hash_encoding": "uint32_le",
        "independent_repetitions": 1,
        "timing_metric": "scheduler_wall_time",
        "input_evidence_scope": "injected_prompt_token_ids",
        "unobserved": [
            "sampled_continuation_token_ids",
            "kv_cache_tensors",
            "recurrent_state_tensors",
            "execution_history_equivalence",
        ],
        "preparation": {
            "warmup_iterations": 5,
            "prefill_real_seed": True,
            "decode_real_kv_warmup": True,
            "giant_kv_threshold": 32768,
            "giant_kv_repeats": 3,
        },
    }
    for position, row in enumerate(payload["results"]):
        point = row["point"]
        identity = {
            "point_type": point["point_type"],
            "batch_size": point["batch_size"],
            "total_prefill_tokens": point.get("total_prefill_tokens", 0),
            "total_kv_read_tokens": point.get("total_kv_read_tokens", 0),
            "partition": point.get("partition"),
            "rows": point.get("rows"),
        }
        for rank in range(payload["dp"]["size"]):
            fpm = next(group for group in payload["iteration_groups"] if group["point"] == point)["rank_results"][rank][
                "fpms"
            ][0]
            raw = {key: value for key, value in fpm.items() if key != "benchmark_measurement"}
            requests = [{"slot": 0, "num_tokens": 4, "sha256": _hash([seed, rank, identity])}]
            fpm["benchmark_measurement"] = {
                "schema_version": 1,
                "point_key": _hash(identity),
                "dp_rank": rank,
                "prompts": {"status": "recorded", "requests": requests, "sha256": _hash(requests)},
                "preparation": {
                    "grid_digest": payload["grid_digest"],
                    "completed_points_before": position,
                    "kv_seed_regime": row["kv_seed_regime"],
                },
                "expected_internal_samples": 1,
                "raw_fpms": [raw],
                "estimate": {"method": "single_step", "raw_sample_indices": [0]},
            }
            if rank == payload["dp"]["rank"]:
                row["fpms"] = [copy.deepcopy(fpm)]

    if execution:
        _add_execution_protocol(payload)
    else:
        _remove_execution_protocol(payload)


def _artifacts(tmp_path, *, run="run-a", seed="0", duplicate=False, order=(1, 2), factor=1.0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    measurements = []
    for index, batch in enumerate((4, 8), 1):
        point = {
            "point_type": "prefill",
            "benchmark_id": index,
            "batch_size": batch,
            "total_prefill_tokens": batch,
            "total_kv_read_tokens": 0,
            "expected_cudagraph_mode": "PIECEWISE",
            "expected_capture_size": batch,
            "padding_tokens": 0,
            "sample_reasons": [],
        }
        measurements.append(
            NativePointMeasurement(
                point, tuple((rank, factor * (0.01 + rank * 0.001)) for rank in range(2)), "not_applicable"
            )
        )
    if duplicate:
        point = {**measurements[0].point, "benchmark_id": 3, "sample_reasons": ["prefill_real_seed"]}
        measurements.append(NativePointMeasurement(point, ((0, 0.02), (1, 0.022)), "real_prefix"))
        order = (*order, 3)
    collection = NativeCollection(
        points=tuple(measurements),
        rank_timings=((0, 1.0, 0.03), (1, 1.0, 0.03)),
        backend_version="0.27.0",
        collector_attempt_id=run,
        runtime_run_id=run,
        runtime_grid_digest="native-grid",
    )
    for rank in range(2):
        results = []
        groups = []
        for index in order:
            measurement = measurements[index - 1]
            point = measurement.point
            ranks = [
                {
                    "dp_rank": dp,
                    "fpms": [{"dp_rank": dp, "counter_id": index, "wall_time": wall, "scheduled_requests": {}}],
                }
                for dp, wall in measurement.rank_wall_times
            ]
            results.append({"point": point, "kv_seed_regime": measurement.kv_seed_regime, "fpms": ranks[rank]["fpms"]})
            groups.append({"point": point, "rank_results": ranks})
        payload = {
            "artifact_type": "rank",
            "dp": {"rank": rank, "size": 2},
            "run_id": run,
            "grid_digest": "native-grid",
            "config": {"mode": "prefill", "output_path": str(tmp_path / "benchmark.json")},
            "limits": {"block_size": 16, "max_model_len": 256000},
            "cudagraph": {"mode": "FULL_AND_PIECEWISE", "capture_sizes": [4, 8]},
            "recurrent_state": {"initialization": "unchanged", "policy": None, "uniform_bound": None},
            "engine": {"scheduler": {"async_scheduling": False}},
            "results": results,
            "iteration_groups": groups,
        }
        _add_measurement_protocol(payload, seed=seed)
        (tmp_path / f"benchmark-dp{rank}.json").write_text(json.dumps(payload))
    return collection


def _edit(root, action, rank=None):
    paths = sorted(root.glob("benchmark-dp*.json")) if rank is None else [root / f"benchmark-dp{rank}.json"]
    for path in paths:
        payload = json.loads(path.read_text())
        action(payload)
        path.write_text(json.dumps(payload))


def _read_pair(tmp_path, **kwargs):
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    left = extract_measurement_evidence(source, _artifacts(source))
    right = extract_measurement_evidence(candidate, _artifacts(candidate, run="run-b", **kwargs))
    return left, right, next(iter(left["points"]))


def test_matching_conditions_remain_comparable_despite_slow_samples(tmp_path):
    left, right, key = _read_pair(tmp_path, factor=6)
    comparison = compare_measurements(left, right, point_key=key)
    assert comparison["status"] == "comparable"
    assert left["points"][key]["wall_time_seconds"] * 6 == right["points"][key]["wall_time_seconds"]
    assert "kv_cache_tensors" in comparison["unobserved"]
    assert "observed_per_call_graph_dispatch" not in comparison["unobserved"]
    assert left["points"][key]["ranks"]["0"]["prompts"] != left["points"][key]["ranks"]["1"]["prompts"]


def test_publication_selection_preserves_ordinary_zero_kv_sample(tmp_path):
    collection = _artifacts(tmp_path, duplicate=True)
    evidence = extract_measurement_evidence(tmp_path, collection)
    assert len(evidence["points"]) == 2
    selected = next(iter(evidence["points"].values()))
    assert selected["point"]["benchmark_id"] == 1
    assert selected["kv_seed_regime"] == "not_applicable"
    assert selected["wall_time_seconds"] == 0.011


def test_legacy_and_missing_comparisons_remain_unestablished(tmp_path):
    collection = _artifacts(tmp_path)

    def legacy(payload):
        payload.pop("measurement_protocol")
        for row in payload["results"]:
            row["fpms"][0].pop("benchmark_measurement")
        for group in payload["iteration_groups"]:
            for rank in group["rank_results"]:
                rank["fpms"][0].pop("benchmark_measurement")

    _edit(tmp_path, legacy)
    evidence = extract_measurement_evidence(tmp_path, collection)
    assert evidence["status"] == "unestablished"
    comparison = compare_measurements(evidence, {}, point_key=next(iter(evidence["points"])))
    assert comparison["status"] == "unestablished"
    assert compare_measurements({}, {}, point_key="absent")["status"] == "unestablished"


@pytest.mark.parametrize("mutation", ["missing", "invalid", "seed"])
def test_mixed_or_conflicting_rank_protocol_rejected(tmp_path, mutation):
    collection = _artifacts(tmp_path)

    def change(payload):
        if mutation == "missing":
            payload.pop("measurement_protocol")
        elif mutation == "invalid":
            payload["measurement_protocol"] = None
        else:
            payload["measurement_protocol"]["content_seed"] = "1"

    _edit(tmp_path, change, rank=1)
    with pytest.raises(ValueError, match="measurement_protocol"):
        extract_measurement_evidence(tmp_path, collection)


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", True), ("content_identity", "unknown"), ("independent_repetitions", 3), ("preparation", {})],
)
def test_invalid_shared_protocol_rejected(tmp_path, field, value):
    collection = _artifacts(tmp_path)
    _edit(tmp_path, lambda payload: payload["measurement_protocol"].update({field: value}))
    with pytest.raises(ValueError, match="measurement_protocol"):
        extract_measurement_evidence(tmp_path, collection)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dp_rank", 4, "rank/point"),
        ("point_key", "0" * 64, "rank/point"),
        ("preparation", {"grid_digest": "wrong"}, "grid/order/KV"),
        ("prompts", {"status": "recorded", "requests": [], "sha256": "a" * 64}, "no requests"),
        ("expected_internal_samples", 0, "internal sample"),
        ("estimate", {"method": "median", "raw_sample_indices": [0]}, "reduction"),
        ("raw_fpms", [{"counter_id": 1, "dp_rank": 0, "wall_time": 99.0}], "retained timing"),
    ],
)
def test_malformed_measurement_evidence_rejected(tmp_path, field, value, message):
    collection = _artifacts(tmp_path)
    _edit(
        tmp_path,
        lambda payload: payload["results"][0]["fpms"][0]["benchmark_measurement"].update({field: value}),
        rank=0,
    )
    with pytest.raises(ValueError, match=message):
        extract_measurement_evidence(tmp_path, collection)


def test_prompt_hash_tampering_rejected(tmp_path):
    collection = _artifacts(tmp_path)
    _edit(
        tmp_path,
        lambda payload: payload["results"][0]["fpms"][0]["benchmark_measurement"]["prompts"].update(sha256="a" * 64),
    )
    with pytest.raises(ValueError, match="aggregate hash"):
        extract_measurement_evidence(tmp_path, collection)


@pytest.mark.parametrize("value", [999, None, True, "missing"])
@pytest.mark.parametrize("raw", [True, False])
def test_sample_counters_must_match_the_measured_point(tmp_path, raw, value):
    collection = _artifacts(tmp_path)

    def change(payload):
        fpms = [payload["results"][0]["fpms"][0]]
        fpms.extend(rank["fpms"][0] for rank in payload["iteration_groups"][0]["rank_results"])
        for fpm in fpms:
            samples = fpm["benchmark_measurement"]["raw_fpms"] if raw else [fpm]
            for sample in samples:
                if value == "missing":
                    sample.pop("counter_id")
                else:
                    sample["counter_id"] = value

    _edit(tmp_path, change)
    with pytest.raises(ValueError, match="counter"):
        extract_measurement_evidence(tmp_path, collection)


def test_unavailable_prompts_and_missing_runtime_remain_unestablished(tmp_path):
    collection = _artifacts(tmp_path)

    def missing(payload):
        payload.pop("recurrent_state")
        payload["results"][0]["fpms"][0]["benchmark_measurement"]["prompts"] = {
            "status": "unavailable",
            "sha256": None,
            "requests": [],
        }

    _edit(tmp_path, missing)
    evidence = extract_measurement_evidence(tmp_path, collection)
    assert evidence["status"] == "unestablished"
    point = next(iter(evidence["points"].values()))
    assert "injected prompt hashes unavailable" in " ".join(point["reasons"])
    assert "recurrent_state" in " ".join(evidence["reasons"])
    assert "preceding prompt evidence incomplete" in " ".join(list(evidence["points"].values())[1]["reasons"])


@pytest.mark.parametrize("difference", ["seed", "prompts", "preparation", "regime", "ranks", "runtime", "estimator"])
def test_different_recorded_conditions_are_mismatches(tmp_path, difference):
    left, right, key = _read_pair(tmp_path)
    point = right["points"][key]
    if difference == "seed":
        right["measurement_protocol"]["content_seed"] = "another"
    elif difference == "prompts":
        point["ranks"]["1"]["prompts"]["sha256"] = "b" * 64
    elif difference == "preparation":
        right["measurement_protocol"]["preparation"]["warmup_iterations"] += 1
    elif difference == "regime":
        point["regime"]["expected_capture_size"] = 16
    elif difference == "ranks":
        point["ranks"].pop("1")
    elif difference == "runtime":
        right["runtime_identity"]["0"]["limits"]["block_size"] = 32
    else:
        point["ranks"]["0"]["estimate"]["method"] = "last_step"
    assert compare_measurements(left, right, point_key=key)["status"] == "mismatch"


def test_cross_grid_matching_prompts_do_not_prove_history_equivalence(tmp_path):
    left, right, key = _read_pair(tmp_path)
    right["runtime_grid_digest"] = "subset-grid"
    right["points"][key]["ranks"]["0"]["preparation"]["grid_digest"] = "subset-grid"
    assert compare_measurements(left, right, point_key=key)["status"] == "mismatch"
    comparison = compare_measurements(left, right, point_key=key, same_context=False)
    assert comparison["identity_status"] == "comparable"
    assert comparison["status"] == "unestablished"
    assert "cross-context" in " ".join(comparison["reasons"])


def test_execution_order_changes_are_not_comparable(tmp_path):
    left, right, key = _read_pair(tmp_path, order=(2, 1))
    result = compare_measurements(left, right, point_key=key)
    assert result["status"] == "mismatch"
    assert "preparation differs" in " ".join(result["reasons"])
    assert "execution_prefix_sha256 differs" in " ".join(result["reasons"])


def test_contradictory_rank_execution_order_is_rejected(tmp_path):
    collection = _artifacts(tmp_path)

    def reorder(payload):
        payload["results"].reverse()
        for position, row in enumerate(payload["results"]):
            row["fpms"][0]["benchmark_measurement"]["preparation"]["completed_points_before"] = position

    _edit(tmp_path, reorder, rank=1)
    with pytest.raises(ValueError, match="measurement order"):
        extract_measurement_evidence(tmp_path, collection)


def test_internal_decode_samples_remain_one_launch_estimate(tmp_path):
    collection = _artifacts(tmp_path)

    def adjacent(payload):
        fpm = payload["results"][0]["fpms"][0]
        measurement = fpm["benchmark_measurement"]
        rank = payload["dp"]["rank"]
        measurement.update(
            expected_internal_samples=4,
            raw_fpms=[
                {"counter_id": fpm["counter_id"], "dp_rank": rank, "wall_time": fpm["wall_time"] * scale}
                for scale in (4, 0.8, 1, 3)
            ],
            estimate={"method": "adjacent_upper_median", "raw_sample_indices": [1, 2, 3]},
        )

    _edit(tmp_path, adjacent)
    _edit(tmp_path, _add_execution_protocol)
    evidence = extract_measurement_evidence(tmp_path, collection)
    point = next(iter(evidence["points"].values()))
    assert evidence["measurement_protocol"]["independent_repetitions"] == 1
    assert point["wall_time_seconds"] == 0.011
    assert len(point["ranks"]["0"]["raw_fpms"]) == 4
    assert point["ranks"]["0"]["raw_fpms"][-1]["wall_time"] == 0.03


def test_duplicate_launch_cannot_count_as_an_independent_repetition(tmp_path):
    left, _, key = _read_pair(tmp_path)
    assert compare_measurements(left, copy.deepcopy(left), point_key=key)["status"] == "mismatch"


def test_extraction_requires_same_native_samples_and_run(tmp_path):
    collection = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="run/grid identity"):
        extract_measurement_evidence(tmp_path, replace(collection, runtime_run_id="different"))


@pytest.mark.parametrize("reasons", [None, []])
def test_zero_kv_duplicates_normalize_optional_reasons(reasons):
    from collector.fpm_forward.native_artifact import _zero_kv_prefill_sample

    point = {"point_type": "prefill", "total_kv_read_tokens": 0}
    ordinary = NativePointMeasurement(point, ((0, 0.01),), "not_applicable")
    duplicate = NativePointMeasurement({**point, "sample_reasons": reasons}, ((0, 0.02),), "real_prefix")
    assert _zero_kv_prefill_sample([ordinary, duplicate]) is ordinary
    point["sample_reasons"] = None
    assert _zero_kv_prefill_sample([ordinary, duplicate]) is ordinary
