# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical checks for comparison definitions, time coverage, and source joins."""

import math

import pytest

from scripts.e2e_accuracy.agentx_comparison import build, validate
from scripts.e2e_accuracy.agentx_curves import (
    build_comparisons,
    cumulative,
    curve_score,
    inflight,
    rebin,
    request_curve,
    series,
    smooth,
)

NAMESPACE = "weka:1111111111111111"
EPOCH = 1792599226807345062


def native(index, phase="profile", start=0, end=2000, **updates):
    return {
        "uuid": str(index),
        "request_id": f"{NAMESPACE}:request:outer:{index}",
        "phase": phase,
        "arrival_time_ms": start,
        "terminal_time_ms": end,
        "first_admit_ms": start,
        "first_token_ms": start + 200,
        "last_token_ms": end,
        "terminal_status": "completed",
        "input_length": 100,
        "output_length": 3,
        "e2e_latency_ms": 2000,
        "ttft_ms": 200,
        "itl_ms": 20,
        "agentic": {"conversation_id": "native-conversation"},
        **updates,
    }


def reference(index, phase="profiling", start=0, end=2000, **updates):
    return {
        "request_id": f"ref-{index}",
        "phase": phase,
        "source_trace_id": "trace",
        "source_outer_idx": index,
        "source_inner_idx": None,
        "conversation_id": "conversation",
        "start_ns": str(EPOCH + start * 1_000_000),
        "end_ns": str(EPOCH + end * 1_000_000),
        "input_tokens": 101,
        "output_tokens": 3,
        "e2e_ms": 1000,
        "ttft_ms": 100,
        "tpot_ms": 10,
        "error": None,
        **updates,
    }


def sample(start, end, **updates):
    return {
        "start": start,
        "t": end,
        "duration": end - start,
        "phase": "profile",
        "running": 1,
        "waiting": 0,
        "cache_pct": 50,
        "prefill_tokens": 10,
        "output_tokens": 2,
        "scheduler_hits": 8,
        "scheduler_queries": 10,
        "hbm": 8,
        "host": 0,
        "miss": 2,
        "admission_input": 10,
        "accept": 3,
        "accept_count": 2,
        **updates,
    }


def fixture(native_duration=4, reference_duration=4):
    dataset = {
        "point_id": 1,
        "requests": [native(1)],
        "reference": [reference(1)],
        "pairing": [],
        "samples": [
            sample(
                0,
                0,
                prefill_tokens=0,
                output_tokens=0,
                scheduler_hits=0,
                scheduler_queries=0,
                hbm=0,
                host=0,
                miss=0,
                admission_input=0,
                accept_count=0,
                accept=None,
            )
        ]
        + [sample(t, t + 1) for t in range(native_duration)],
        "telemetry": {
            "profile_start_ms": 0,
            "sample_interval_ms": 1000,
            "phase_boundaries": [
                {"phase": "profile", "start_ms": 0, "end_ms": native_duration * 1000},
                {"phase": "preparation", "start_ms": 0, "end_ms": 0},
            ],
        },
    }
    values = {
        "cache_used": 20,
        "cache_evictable": 30,
        "cache_capacity": 100,
        "running": 1,
        "waiting": 0,
        "acceptance": 3,
        "prefill_tokens": 10,
        "output_processed_tokens": 12,
        "scheduler_hits": 8,
        "scheduler_queries": 10,
        "hbm": 8,
        "host": 0,
        "miss": 2,
        "admission_input": 10,
    }
    silicon = {
        "point_id": 1,
        "phases": {
            "profile": {
                "start_ns": str(EPOCH),
                "end_ns": str(EPOCH + reference_duration * 1_000_000_000),
                "duration_s": reference_duration,
                "series": {
                    key: [{"start": t, "end": t + 1, "value": value, "coverage": 1} for t in range(reference_duration)]
                    for key, value in values.items()
                },
            }
        },
    }
    return dataset, silicon


def test_rebin_conserves_fractional_interval_counts_and_never_fills_gaps():
    intervals = [
        {"start": -0.5, "end": 0.5, "value": 10},
        {"start": 0.5, "end": 1.5, "value": 20},
    ]
    assert rebin(intervals, [1, 1.5], counter=True) == [15, 10]
    assert rebin(intervals, [1, 2], counter=True) == [15, None]
    assert rebin([{"start": 0, "end": 1, "value": 0}], [1, 2]) == [0, None]
    assert rebin([{"start": 0, "end": 1, "value": 7, "coverage": 0.5}], [1]) == [None]


def test_smoothing_is_duration_weighted_and_missing_is_not_zero():
    assert smooth([10, 20], [1, 1.5], counter=True) == [10, 20]
    assert smooth([10, 20], [1, 1.5]) == [10, pytest.approx(40 / 3)]
    assert smooth([1, None, 3], [1, 2, 3]) == [1, None, None]
    assert cumulative([None, 2, 3], start_index=1) == [None, 2, 5]


def test_curve_score_reports_zero_denominators_missing_and_false_activity():
    ref = series("x", "x", "silicon", [1, 2, 3, 4], [0, 10, None, 20])
    sim = series("x", "x", "simulated", [1, 2, 3, 4], [2, 20, 30, 10])
    score = curve_score("x", "x", ref, sim, 4, 4)
    assert score["mape_percent"] == 75
    assert score["count"] == 2 and score["coverage"] == 0.5
    assert score["zero_reference_count"] == 1 and score["missing_count"] == 1
    assert score["zero_reference_simulated_nonzero_count"] == 1
    assert score["mae"] == pytest.approx(22 / 3)
    assert score["aligned"][0] == {
        "t": 2,
        "reference": 10,
        "simulated": 20,
        "ape_percent": 100,
    }


def test_percentiles_step_on_arrival_and_inverse_tpot_is_not_tpot_mape():
    rows = [
        {"_start": 0, "terminal_status": "completed", "itl_ms": 10},
        {"_start": 2.5, "terminal_status": "completed", "itl_ms": 20},
    ]
    assert request_curve(rows, [1, 2, 3], "itl_ms", 75, inverse=True) == [
        100,
        100,
        pytest.approx(1000 / 17.5),
    ]
    dataset, silicon = fixture()
    phase = build_comparisons(dataset, silicon, {"trace": NAMESPACE})["phases"]["profile"]
    assert phase["matched"]["ttft_ms"]["mape_percent"] == 100
    assert phase["matched"]["interactivity"]["mape_percent"] == 50
    assert phase["charts"]["interactivity_p90"]["scores"][0]["mape_percent"] == 50
    assert phase["matched"]["e2e_latency_ms"]["pairs"][0]["input_length_matches"] is False


def test_phase_alignment_uses_actual_epoch_not_first_request_and_preserves_ns():
    dataset, silicon = fixture()
    dataset["reference"][0]["start_ns"] = str(EPOCH + 1_234_567)
    phase = build_comparisons(dataset, silicon, {"trace": NAMESPACE})["phases"]["profile"]
    row = phase["timeline"]["reference_requests"][0]
    assert row["arrival_phase_ms"] == 1.234567
    assert row["start_ns"] == str(EPOCH + 1_234_567)
    assert row["matched_simulated_uuid"] == "1"
    assert phase["timeline"]["native_groups"]["1"] == row["canonical_conversation"]


def test_unrecorded_cancellation_retains_unavailable_interval_and_unassigned_lineage():
    dataset, silicon = fixture()
    dataset["reference"].append(
        reference(
            2,
            conversation_id=None,
            source_trace_id=None,
            start_ns=None,
            end_ns=None,
            input_tokens=None,
            output_tokens=None,
            error="measurement_cancelled",
            chunks=[],
        )
    )
    phase = build_comparisons(dataset, silicon, {"trace": NAMESPACE})["phases"]["profile"]
    row = phase["timeline"]["reference_requests"][1]
    assert row["canonical_conversation"] == "silicon:Unassigned"
    assert row["arrival_time_ms"] is row["terminal_time_ms"] is None
    assert row["matched_simulated_uuid"] is None
    assert row["error"] == "measurement_cancelled"


def test_preparation_folds_primer_and_warmup_but_excludes_failures_and_length_mismatch():
    dataset, silicon = fixture()
    dataset["requests"] = [
        native(1, "primer"),
        native(2, "warmup"),
        native(3, "warmup", output_length=8),
    ]
    dataset["reference"] = [
        reference(1, "warmup"),
        reference(2, "warmup", error={"type": "failure"}),
        reference(3, "warmup"),
    ]
    dataset["telemetry"]["phase_boundaries"][1]["end_ms"] = 4000
    silicon["phases"]["preparation"] = silicon["phases"]["profile"]
    phase = build_comparisons(dataset, silicon, {"trace": NAMESPACE})["phases"]["preparation"]
    score = phase["matched"]["e2e_latency_ms"]
    assert score["count"] == 1 and score["mape_percent"] == 100
    assert "reference_error" in score["pairs"][1]["exclusions"]
    assert "output_length_mismatch_or_unavailable" in score["pairs"][2]["exclusions"]
    assert phase["counts"] == {
        "silicon": 3,
        "simulated": 3,
        "silicon_failed": 1,
        "simulated_not_completed": 0,
    }


def test_ambiguous_source_is_never_paired_by_time_or_length():
    dataset, silicon = fixture()
    dataset["requests"].append(native(1, uuid="duplicate"))
    phase = build_comparisons(dataset, silicon, {"trace": NAMESPACE})["phases"]["profile"]
    assert phase["matched"]["e2e_latency_ms"]["count"] == 0
    assert "ambiguous_simulated_source" in phase["matched"]["e2e_latency_ms"]["pairs"][0]["exclusions"]


def test_default_output_is_completion_accounted_and_incompatible_metrics_unscored():
    dataset, silicon = fixture(native_duration=5, reference_duration=4)
    charts = build_comparisons(dataset, silicon, {"trace": NAMESPACE})["phases"]["profile"]["charts"]
    values = charts["throughput_decode"]["series"]
    assert values[0]["y"][:4] == values[1]["y"][:4] == [0, 1.5, 1, 0.75]
    assert charts["throughput_decode"]["scores"][0]["mape_percent"] == 0
    assert charts["throughput_decode"]["score_window_end_s"] == 4
    for key in ("throughput_delivered", "acceptance", "completion"):
        assert charts[key]["scores"][0]["mape_percent"] is None
        assert charts[key]["scores"][0]["reason"]
    assert charts["cache"]["scores"][0]["mape_percent"] == 0
    assert charts["source"]["scores"][0]["mape_percent"] == 0


def test_inflight_proxy_counts_maximum_per_conversation_and_keeps_unknown():
    rows = [
        {"_start": 0, "_end": 3, "input_tokens": n, "conversation_id": c} for n, c in [(10, "a"), (20, "a"), (30, "b")]
    ]
    assert inflight(rows, [1, 3], "input_tokens", True, True) == [50, 0]
    rows[0]["input_tokens"] = None
    assert inflight(rows, [1], "input_tokens", True, True) == [None]


def test_missing_silicon_clock_preserves_native_data_without_invented_curve_score():
    dataset, _ = fixture()
    phase = build_comparisons(dataset, None, {"trace": NAMESPACE})["phases"]["profile"]
    chart = phase["charts"]["latency_ttft_ms_p90"]
    assert all(row["origin"] == "simulated" for row in chart["series"])
    assert chart["scores"][0]["mape_percent"] is None
    assert "origin" in chart["scores"][0]["reason"]
    assert phase["matched"]["ttft_ms"]["count"] == 1
    untimed = phase["timeline"]["reference_requests"][0]
    assert untimed["arrival_time_ms"] is None
    assert untimed["matched_simulated_uuid"] == "1"
    empty = build_comparisons({})
    assert empty["phases"]["profile"]["counts"]["simulated"] == 0


def test_prefix_counter_ratio_is_token_weighted_not_mean_of_ratios():
    dataset, silicon = fixture()
    silicon["phases"]["profile"]["series"]["scheduler_hits"] = [
        {"start": 0, "end": 1, "value": 1},
        {"start": 1, "end": 2, "value": 90},
    ]
    silicon["phases"]["profile"]["series"]["scheduler_queries"] = [
        {"start": 0, "end": 1, "value": 1},
        {"start": 1, "end": 2, "value": 100},
    ]
    chart = build_comparisons(dataset, silicon, {"trace": NAMESPACE})["phases"]["profile"]["charts"]["hits"]
    assert chart["series"][0]["y"][:2] == [100, pytest.approx(100 * 91 / 101)]
    assert chart["series"][0]["y"][2] is None


def test_metrics_are_finite_json_values():
    dataset, silicon = fixture()
    result = build_comparisons(dataset, silicon, {"trace": NAMESPACE})

    def check(value):
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, float):
            assert math.isfinite(value)

    check(result)


def test_supported_pass_counter_nulls_mean_no_queries_but_missing_intervals_do_not():
    dataset, silicon = fixture()
    for row in dataset["samples"][2:]:
        row["scheduler_hits"] = row["scheduler_queries"] = None
    result = build_comparisons(dataset, silicon, {"trace": NAMESPACE})
    chart = result["phases"]["profile"]["charts"]["hits"]
    assert chart["scores"][0]["mape_percent"] == pytest.approx(0)
    assert chart["series"][1]["y"] == pytest.approx([80, 80, 80, 80])
    assert dataset["samples"][2]["scheduler_queries"] is None
    del dataset["samples"][2]
    chart = build_comparisons(dataset, silicon)["phases"]["profile"]["charts"]["hits"]
    assert chart["series"][1]["y"] == [80, None, None, None]
    dataset["samples"][1]["scheduler_hits"] = dataset["samples"][1]["scheduler_queries"] = None
    chart = build_comparisons(dataset, silicon)["phases"]["profile"]["charts"]["hits"]
    assert chart["series"][1]["y"] == [None] * 4


def test_startup_smoothing_uses_common_covered_window_without_zero_filling():
    dataset, silicon = fixture()
    dataset["samples"][0]["cache_pct"] = 0
    for key in ("cache_used", "cache_evictable", "cache_capacity"):
        for row in silicon["phases"]["profile"]["series"][key]:
            row["start"] += 0.1
            row["end"] += 0.1
    chart = build_comparisons(dataset, silicon)["phases"]["profile"]["charts"]["cache"]
    assert chart["series"][0]["y"] == [None, 50, 50, 50]
    assert chart["series"][1]["y"] == [None, 50, 50, 50]
    assert chart["scores"][0]["count"] == 3
    assert chart["scores"][0]["mape_percent"] == 0


def test_proxy_uses_unambiguous_source_conversation_mapping():
    dataset, silicon = fixture()
    dataset["requests"].append(native(2, agentic={"conversation_id": "separate-native-fanout"}))
    dataset["reference"].append(reference(2))
    phase = build_comparisons(dataset, silicon, {"trace": NAMESPACE})["phases"]["profile"]
    assert phase["charts"]["unique"]["series"][1]["y"][0] == 100
    assert phase["timeline"]["native_groups"]["1"] == phase["timeline"]["native_groups"]["2"]


def test_native_lookup_remains_an_unscored_additional_series():
    dataset, silicon = fixture()
    for row in dataset["samples"]:
        row.update(lookup_hits=9, lookup_queries=10)
    chart = build_comparisons(dataset, silicon)["phases"]["profile"]["charts"]["hits"]
    assert {row["metric"] for row in chart["series"]} == {
        "scheduler_reuse",
        "native_lookup",
    }
    assert chart["scores"][0]["mape_percent"] == 0
    assert chart["scores"][1]["mape_percent"] is None
    assert "no equivalent" in chart["scores"][1]["reason"]


def public_comparison():
    dataset, silicon = fixture()
    report = {
        "samples": dataset["samples"],
        "telemetry": dataset["telemetry"],
        "phases": {
            r["phase"]: {k: v for k, v in r.items() if k != "phase"} for r in dataset["telemetry"]["phase_boundaries"]
        },
    }
    scores = {"request_errors": {m: {"count": 1, "sum_ape": 100, "mape": 100} for m in ("ttft_ms", "tpot_ms")}}
    return build(
        {
            "reference": dataset["reference"],
            "silicon": silicon,
            "namespace_map": {"trace": NAMESPACE},
        },
        report,
        dataset,
        scores,
    )


def test_public_comparison_keeps_point_request_denominators_and_unscored_preparation():
    value = public_comparison()
    assert value["phases"]["profile"]["request_scores"]["ttft_ms"]["count"] == 1
    assert all(s["mape_percent"] is None for c in value["phases"]["preparation"]["charts"] for s in c["scores"])
    rows = value["phases"]["profile"]["requests"]
    assert len(rows) == 2 and rows[0]["counterpart"] == rows[1]["uuid"]
    assert all(len(r["uuid"]) == 16 for r in rows)


def test_portable_preparation_joins_source_keys_instead_of_measured_uuids():
    from test_agentx_accuracy import observation

    from scripts.e2e_accuracy.agentx_contract import score
    from scripts.e2e_accuracy.run_agentx_accuracy import normalize_native

    dataset, silicon = fixture()
    dataset["requests"] = [native(1, end=1000), native(2, start=2000, end=4000)]
    measured = [
        reference(1, phase="preparation", end=1000, key=dataset["requests"][0]["request_id"]),
        reference(2, phase="profile", start=2000, end=4000, key=dataset["requests"][1]["request_id"]),
    ]
    observations = [observation(row["key"], phase=row["phase"]) for row in measured]
    normalized = normalize_native({"telemetry": {"requests": dataset["requests"]}}, observations, "self_collected")
    scores = score(observations, normalized, "complete_finite_cohort")
    assert scores["coverage"]["predicted"] == 1
    report = {
        "samples": dataset["samples"],
        "telemetry": dataset["telemetry"],
        "phases": {"preparation": {"start_ms": 0, "end_ms": 2000}, "profile": {"start_ms": 2000, "end_ms": 4000}},
    }
    value = build({"reference": measured, "silicon": silicon, "namespace_map": {}}, report, dataset, scores)
    validate(value)
    for phase in ("preparation", "profile"):
        rows = value["phases"][phase]["requests"]
        assert len(rows) == 2
        assert rows[0]["counterpart"] == rows[1]["uuid"]
        assert rows[1]["counterpart"] == rows[0]["uuid"]
    assert all(s["mape_percent"] is None for chart in value["phases"]["preparation"]["charts"] for s in chart["scores"])


def test_public_comparison_rejects_private_fields_and_wrong_scores():
    value = public_comparison()
    value["phases"]["profile"]["requests"][0]["prompt"] = "private"
    with pytest.raises(ValueError, match="fields"):
        validate(value)
    value = public_comparison()
    score = next(s for c in value["phases"]["profile"]["charts"] for s in c["scores"] if s["mape_percent"] is not None)
    score["mape_percent"] += 1
    with pytest.raises(ValueError, match="reconcile"):
        validate(value)


def test_untimed_failed_requests_remain_inspectable_without_fabricated_intervals():
    dataset, silicon = fixture()
    dataset["reference"][0].update(end_ns=None, error={"type": "failed"})
    phase = build_comparisons(dataset, silicon, {"trace": NAMESPACE})["phases"]["profile"]
    row = phase["timeline"]["reference_requests"][0]
    assert row["terminal_time_ms"] is None and row["terminal_status"] == "failed"
    assert row["matched_simulated_uuid"] == "1"
    assert phase["matched"]["ttft_ms"]["count"] == 0


@pytest.mark.parametrize(
    "malformation",
    ["self_pair", "one_way_pair", "preparation_score", "fractional_count"],
)
def test_public_comparison_rejects_invalid_pairing_and_preparation_scores(malformation):
    value = public_comparison()
    rows = value["phases"]["profile"]["requests"]
    if malformation == "self_pair":
        rows[0]["counterpart"] = rows[0]["uuid"]
    elif malformation == "one_way_pair":
        rows[1]["counterpart"] = None
    elif malformation == "preparation_score":
        value["phases"]["preparation"]["request_scores"]["ttft_ms"] = {
            "count": 1,
            "sum_ape": 1,
            "mape": 1,
        }
    else:
        value["phases"]["profile"]["request_scores"]["ttft_ms"]["count"] = 1.5
    with pytest.raises(ValueError):
        validate(value)


@pytest.mark.parametrize("clock", ["ms", "ns"])
def test_qwen_stream_counts_enable_delivery_comparison_without_token_arrays(clock):
    dataset, silicon = fixture()
    dataset["reference"][0]["chunks"] = [
        {"at_ms": EPOCH / 1e6 + 500, "count": 1},
        {"at_ms": EPOCH / 1e6 + 1000, "count": 2},
    ]
    if clock == "ns":
        dataset["reference"][0]["chunks"] = [
            {"at_ns": str(EPOCH + 500_000_000), "count": 1},
            {"at_ns": str(EPOCH + 1_000_000_000), "count": 2},
        ]
    charts = build_comparisons(dataset, silicon, {"trace": NAMESPACE})["phases"]["profile"]["charts"]
    score = charts["throughput_delivered"]["scores"][0]
    assert score["reason"] is None and score["count"] > 0


def test_missing_cancelled_stream_does_not_become_zero_delivered_tokens():
    dataset, silicon = fixture()
    dataset["reference"][0]["chunks"] = [{"at_ns": str(EPOCH + 500_000_000), "count": 3}]
    dataset["reference"].append(
        reference(2, error="measurement_cancelled", start_ns=None, end_ns=None, output_tokens=None)
    )
    chart = build_comparisons(dataset, silicon, {"trace": NAMESPACE})["phases"]["profile"]["charts"][
        "throughput_delivered"
    ]
    assert chart["scores"][0]["mape_percent"] is None
    assert "cancelled" in chart["scores"][0]["reason"]
    assert not any(row["origin"] == "silicon" for row in chart["series"])
