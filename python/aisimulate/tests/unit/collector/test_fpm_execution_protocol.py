# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controls for observed execution, independent of planned capture expectations."""

import copy
import json

import pytest
from collector.fpm_forward import execution_evidence, repeatability, runtime_probe
from collector.fpm_forward.measurement_evidence import compare_measurements, extract_measurement_evidence

from .test_fpm_measurement_evidence import _add_execution_protocol, _artifacts, _edit, _remove_execution_protocol
from .test_fpm_profile_collection import no_models_or_timing_data  # noqa: F401
from .test_fpm_repeatability import _args, _fake_collector, campaign  # noqa: F401
from .test_fpm_runtime_probe import _inputs

pytestmark = pytest.mark.unit


def _execution(payload, *, clock_offset=0.0):
    _add_execution_protocol(payload, clock_offset=clock_offset)


def _pair(tmp_path, *, factor=1.0):
    left_root, right_root = tmp_path / "source", tmp_path / "repeat"
    left = _artifacts(left_root)
    right = _artifacts(right_root, run="run-b", factor=factor)
    _edit(left_root, _execution)
    _edit(right_root, lambda payload: _execution(payload, clock_offset=9999))
    return left_root, right_root, left, right


def _compare(roots):
    left_root, right_root, left_collection, right_collection = roots
    left = extract_measurement_evidence(left_root, left_collection)
    right = extract_measurement_evidence(right_root, right_collection)
    key = next(iter(left["points"]))
    return left, right, compare_measurements(left, right, point_key=key)


def test_observed_dispatch_keeps_slow_samples_and_process_clocks_separate(tmp_path):
    left, right, comparison = _compare(_pair(tmp_path, factor=6))
    assert comparison["status"] == "comparable"
    assert "observed_per_call_graph_dispatch" not in comparison["unobserved"]
    assert "kv_cache_tensors" in comparison["unobserved"]
    assert right["warmup_evidence"]["0"]["records"][0]["validation"]["status"] == "not_performed"
    point = next(iter(right["points"].values()))
    assert point["ranks"]["0"]["raw_fpms"][0]["wall_time"] == pytest.approx(0.06)
    assert point["wall_time_seconds"] == pytest.approx(next(iter(left["points"].values()))["wall_time_seconds"] * 6)


def test_async_and_sync_observations_do_not_compare_even_when_launch_defaults_match(tmp_path):
    roots = _pair(tmp_path)
    _edit(roots[1], lambda payload: payload["engine"]["scheduler"].update(async_scheduling=True))
    assert "effective async_scheduling differs" in " ".join(_compare(roots)[2]["reasons"])
    assert _compare(roots)[2]["status"] == "mismatch"


def test_missing_scheduling_is_not_inferred_from_same_config(tmp_path):
    roots = _pair(tmp_path)
    _edit(roots[1], lambda payload: payload.pop("engine"))
    comparison = _compare(roots)[2]
    assert comparison["status"] == "unestablished"
    assert "async_scheduling" in " ".join(comparison["reasons"])


def test_older_observations_are_unknown_not_a_proven_protocol_mismatch(tmp_path):
    roots = _pair(tmp_path)

    def older(payload):
        payload["measurement_protocol"].pop("execution_evidence")
        payload.pop("warmup_evidence")
        for row in payload["results"]:
            measurement = row["fpms"][0]["benchmark_measurement"]
            measurement["preparation"].pop("warmup_records_before")
            for raw in measurement["raw_fpms"]:
                raw.pop("benchmark_sample")

    _edit(roots[0], older)
    left, right, comparison = _compare(roots)
    for key in left["points"]:
        comparison = compare_measurements(left, right, point_key=key)
        assert comparison["status"] == "unestablished", comparison
        assert "measurement_protocol differs" not in comparison["reasons"]


def test_actual_dispatch_mismatch_is_detected_despite_equal_point_expectations(tmp_path):
    roots = _pair(tmp_path)
    _edit(
        roots[1],
        lambda payload: payload["results"][0]["fpms"][0]["benchmark_measurement"]["raw_fpms"][0]["benchmark_sample"][
            "cudagraph"
        ].update(runtime_mode="FULL"),
    )
    assert _compare(roots)[2]["status"] == "mismatch"


@pytest.mark.parametrize("both", [False, True])
def test_unavailable_dispatch_remains_unknown_instead_of_eager(tmp_path, both):
    roots = _pair(tmp_path)

    def unavailable(payload):
        graph = payload["results"][0]["fpms"][0]["benchmark_measurement"]["raw_fpms"][0]["benchmark_sample"][
            "cudagraph"
        ]
        graph.update(
            status="unavailable", runtime_mode=None, num_unpadded_tokens=None, num_padded_tokens=None, num_paddings=None
        )

    if both:
        _edit(roots[0], unavailable)
    _edit(roots[1], unavailable)
    comparison = _compare(roots)[2]
    assert comparison["status"] == "unestablished"
    assert "observed_per_call_graph_dispatch" in comparison["unobserved"]


@pytest.mark.parametrize("change", ["clock", "index", "contract", "warmup", "padding", "retained"])
def test_contradictory_sample_evidence_rejects(tmp_path, change):
    roots = _pair(tmp_path)

    def mutate(payload):
        retained = payload["results"][0]["fpms"][0]
        sample = retained["benchmark_measurement"]["raw_fpms"][0]["benchmark_sample"]
        if change == "clock":
            sample["timing"]["end_monotonic"] += 1
        elif change == "index":
            sample["forward_index"] = True
        elif change == "contract":
            payload["measurement_protocol"]["execution_evidence"]["timing_clock"] = "global"
        elif change == "warmup":
            retained["benchmark_measurement"]["preparation"]["warmup_records_before"] = 2
        elif change == "padding":
            sample["cudagraph"]["num_paddings"] = 1
        else:
            retained["benchmark_sample"] = copy.deepcopy(sample)

    _edit(roots[1], mutate)
    with pytest.raises(ValueError):
        _compare(roots)


def test_observed_preparation_change_affects_full_grid_but_not_cross_context_identity(tmp_path):
    roots = _pair(tmp_path)
    _edit(
        roots[1],
        lambda payload: payload["warmup_evidence"]["records"][0]["requested_shape"].update(prompt_lengths=[16]),
    )
    left, right, comparison = _compare(roots)
    assert comparison["status"] == "mismatch"
    comparison = compare_measurements(left, right, point_key=next(iter(left["points"])), same_context=False)
    assert comparison["identity_status"] == "comparable"
    assert comparison["status"] == "unestablished"


def test_running_warmup_cannot_establish_completed_preparation(tmp_path):
    roots = _pair(tmp_path)
    for root in roots[:2]:
        _edit(
            root,
            lambda payload: payload["warmup_evidence"]["records"][0].update(status="running", forward_index_end=None),
        )
    left, _, comparison = _compare(roots)
    assert comparison["status"] == "unestablished"
    measurement = next(iter(left["points"].values()))["ranks"]["0"]
    assert measurement["observed_execution"]["warmup_prefix_sha256"] is None


@pytest.mark.parametrize(
    "mutation", ["future_count", "omitted_record", "late_forward", "early_forward", "missing_end", "count"]
)
def test_impossible_warmup_lifecycle_rejects(tmp_path, mutation):
    roots = _pair(tmp_path)

    def change(payload):
        history = payload["warmup_evidence"]["records"]
        record = history[0]
        if mutation == "future_count":
            record["completed_points_before"] = 99
        elif mutation == "omitted_record":
            payload["results"][0]["fpms"][0]["benchmark_measurement"]["preparation"]["warmup_records_before"] = 0
        elif mutation == "late_forward":
            record.update(forward_index_start=4, forward_index_end=7)
        elif mutation == "early_forward":
            history.append({**copy.deepcopy(record), "kind": "eager_shape", "completed_points_before": 1})
            payload["results"][1]["fpms"][0]["benchmark_measurement"]["preparation"]["warmup_records_before"] = 2
        elif mutation == "missing_end":
            record.update(status="failed", forward_index_end=None)
        else:
            record["observed_forward_count"] = None

    _edit(roots[1], change)
    with pytest.raises(ValueError, match="warmup|preparation"):
        _compare(roots)


@pytest.mark.parametrize("observed_attempt", [False, True])
def test_failed_historical_warmup_is_visible_and_distinguishable(tmp_path, observed_attempt):
    roots = _pair(tmp_path)

    def failed(payload):
        record = payload["warmup_evidence"]["records"][0]
        record.update(kind="eager_shape", status="failed", validation={"status": "failed", "reason": "shape_mismatch"})
        if not observed_attempt:
            record.update(forward_index_start=None, forward_index_end=0, observed_forward_count=0)
            record["validation"]["status"] = "not_performed"

    _edit(roots[1], failed)
    assert _compare(roots)[2]["status"] == "mismatch"
    _edit(roots[0], failed)
    left, _, comparison = _compare(roots)
    assert comparison["status"] == "comparable"
    assert left["warmup_evidence"]["0"]["records"][0]["status"] == "failed"


def test_unreferenced_later_warmup_does_not_invalidate_completed_measurements(tmp_path):
    roots = _pair(tmp_path)

    def unfinished(payload):
        payload["warmup_evidence"]["records"].append(
            {
                "kind": "eager_shape",
                "status": "running",
                "validation": {"status": "not_performed"},
                "completed_points_before": 2,
                "forward_index_start": 5,
                "forward_index_end": None,
                "observed_forward_count": 0,
            }
        )

    _edit(roots[1], unfinished)
    assert _compare(roots)[2]["status"] == "comparable"


@pytest.mark.parametrize(
    "policy,eager,warning", [("runtime", False, False), ("explicit", False, True), ("explicit", True, False)]
)
def test_probe_graph_warning_is_early_actionable_and_does_not_reject_explicit(
    tmp_path, monkeypatch, policy, eager, warning
):
    launch, manifest, _ = _inputs(tmp_path, graph_policy=policy)
    launch["collection"]["enforce_eager"] = eager
    result = runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=tmp_path / "preview")
    assert result["status"] == "preview"
    messages = " ".join(result["configurations"]["tp4"]["diagnostics"])
    assert ("apply to prefill only" in messages) == warning
    if warning:
        assert "Compatibility is unresolved until observed" in messages


def test_legacy_frozen_assessment_keeps_exact_original_evidence(campaign, tmp_path, monkeypatch):  # noqa: F811
    from .test_fpm_repeatability import _mutate_native

    _mutate_native(campaign[1], _remove_execution_protocol)
    original_freeze = repeatability.freeze_repeatability_plan

    def old_freeze(*args, **kwargs):
        return original_freeze(*args, **{**kwargs, "observation_evidence_version": 1})

    monkeypatch.setattr(repeatability, "freeze_repeatability_plan", old_freeze)
    _fake_collector(monkeypatch)
    collect = repeatability.run_collection

    def historical_repeat(*args, **kwargs):
        from pathlib import Path

        result = collect(*args, **kwargs)
        _mutate_native(Path(kwargs["artifact_root"]), _remove_execution_protocol)
        return result

    monkeypatch.setattr(repeatability, "run_collection", historical_repeat)
    args = _args(campaign, tmp_path)
    original_report = repeatability.run_repeatability(**args)
    frozen_path = args["output_dir"] / repeatability.PLAN_FILENAME
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    assert "observation_evidence_version" not in frozen
    assert "scheduling" not in frozen["cells"][0]["execution"]
    monkeypatch.setattr(repeatability, "freeze_repeatability_plan", original_freeze)
    assessed = repeatability.assess_repeatability(frozen, original_report, cv_threshold=0.05)
    assert assessed["status"] == original_report["status"]
    resumed = repeatability.run_repeatability(**{**args, "resume": True})
    assert resumed["status"] == original_report["status"]
    assert frozen_path.read_bytes() == frozen_bytes


def test_requested_default_does_not_claim_an_effective_async_value():
    assert execution_evidence._requested_scheduling({}) == {"source": "runtime_default", "async_scheduling": None}
    assert execution_evidence._requested_scheduling({"--no-async-scheduling": True}) == {
        "source": "generated_flag",
        "async_scheduling": False,
    }
