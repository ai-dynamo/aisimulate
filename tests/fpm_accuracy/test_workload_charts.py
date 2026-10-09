# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from scripts.fpm_accuracy.dashboard.workloads import (
    charts,
    histogram,
    workload_details,
    workload_summary,
)
from scripts.fpm_accuracy.exceptions import DataError
from scripts.fpm_accuracy.hf.parquet import validate_collection_manifest


def request(**changes):
    return dict(
        dict(
            request_id="request",
            stage="profiling",
            outcome="completed",
            input_tokens=100,
            output_tokens=3,
            start_offset_ms=1000,
            ttft_ms=200,
            tpot_ms=50,
            e2e_ms=300,
            timing_boundaries_compatible=True,
        ),
        **changes,
    )


def test_histograms_preserve_full_population_and_null_zero():
    result = histogram([None, float("nan"), -1, 0, 0, 10, 10])
    assert result["count"] == 4
    assert result["bins"] == [
        {"lower": 0, "upper": 0, "count": 2},
        {"lower": 10, "upper": 10, "count": 2},
    ]
    broad = histogram(range(10000))
    assert len(broad["bins"]) == 32
    assert sum(b["count"] for b in broad["bins"]) == 10000
    assert histogram([])["p90"] is None


def test_timings_use_seconds_reciprocal_and_compatible_boundaries():
    result = charts(
        [
            request(tpot_ms=None),
            request(request_id="one", output_tokens=1, tpot_ms=None),
            request(
                request_id="incompatible",
                tpot_ms=None,
                timing_boundaries_compatible=False,
            ),
            request(request_id="warmup", stage="warmup"),
        ]
    )
    assert result["request_count"] == 3
    assert result["ttft"]["points"] == [[1, 0.2]] * 3
    assert result["interactivity"]["points"] == [[1, 20]]
    assert result["interactivity"]["excluded"] == 2


def test_missing_and_cancelled_timings_never_become_zero():
    result = charts(
        [
            request(outcome="cancelled", ttft_ms=None, tpot_ms=None, e2e_ms=None),
            request(ttft_ms=0, tpot_ms=0),
            request(start_offset_ms=None),
        ]
    )
    assert result["ttft"]["points"] == [[1, 0]]
    assert result["interactivity"]["points"] == []
    assert result["input"]["count"] == 3


def test_rolling_percentile_uses_latest_fifty_valid_requests():
    records = [
        request(request_id=str(i), start_offset_ms=i * 1000, ttft_ms=i * 1000)
        for i in range(60)
    ]
    result = charts(list(reversed(records)))
    assert result["ttft"]["rolling_p90"][-1] == [59, 54.1]
    assert result["ttft"]["points"][0] == [0, 0]


def run(identity, spans, kind="trace_replay"):
    return dict(
        id=identity,
        collection_type=kind,
        benchmark_preset=None,
        benchmark_id=None,
        started_at=None,
        collector={"name": None, "version": None},
        replay_mode="unknown",
        dataset={"name": "recorded dataset"},
        workload={},
        serving={"layout": "unknown"},
        truth_bindings=[{"file_id": "truth", "record_ranges": spans}],
        request_metrics={
            "status": "unavailable",
            "file_ids": [],
            "reason": "Missing request evidence.",
        },
    )


def test_selected_truth_controls_metadata_and_supporting_only_runs_are_hidden():
    runs = [
        run("run-b", [[1, 3]]),
        run("run-a", [[3, 6]], "static_serving"),
        run("run-c", [[6, 8]]),
    ]
    case = SimpleNamespace(
        observations=[
            SimpleNamespace(source_file_id="truth", source_row=i) for i in (1, 4, 5)
        ],
        helper_files=[],
        collection_runs=runs,
    )
    details = workload_details(case)
    assert [r["id"] for r in details["runs"]] == ["run-a", "run-b"]
    assert [r["measurement_count"] for r in details["runs"]] == [2, 1]
    assert workload_summary(details)["run_count"] == 2
    case.collection_runs = []
    assert workload_details(case) == {"runs": [], "unattributed_measurements": 3}


def test_missing_selected_request_asset_disables_charts():
    collection = run("run-a", None)
    collection["request_metrics"].update(status="available", file_ids=["requests"])
    case = SimpleNamespace(
        observations=[SimpleNamespace(source_file_id="truth", source_row=1)],
        helper_files=[],
        collection_runs=[collection],
    )
    selected = workload_details(case)["runs"][0]
    assert selected["availability"] == "unavailable"
    assert selected["charts"] is None


@pytest.mark.parametrize(
    "mutation", ["overlap", "helper", "request", "duplicate", "negative"]
)
def test_corrupt_collection_references_fail_closed(mutation):
    manifest = {
        "files": [
            dict(
                measurement_file_id="truth",
                role="truth",
                format="parquet",
                storage_schema="fpm-iterations-parquet-v1",
                logical_row_count=3,
            )
        ],
        "collection_runs": [run("run-a", [[1, 3]])],
    }
    if mutation == "overlap":
        manifest["collection_runs"].append(run("run-b", [[2, 4]]))
    elif mutation == "helper":
        manifest["files"][0]["role"] = "supporting_evidence"
    elif mutation == "request":
        manifest["collection_runs"][0]["request_metrics"].update(
            status="available", file_ids=["absent"]
        )
    elif mutation == "duplicate":
        manifest["collection_runs"].append(run("run-a", [[3, 4]]))
    else:
        manifest["collection_runs"][0]["workload"]["concurrency"] = -1
    with pytest.raises(DataError):
        validate_collection_manifest(manifest)
