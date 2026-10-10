# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import io
import json
from dataclasses import replace
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_hf_dataset import _benchmark_point, _fpm_payload

from scripts.fpm_accuracy.dashboard.workloads import (
    charts,
    histogram,
    workload_details,
    workload_summary,
)
from scripts.fpm_accuracy.exceptions import DataError
from scripts.fpm_accuracy.hf import HfDataset, adapter_for
from scripts.fpm_accuracy.hf.models import MeasurementFile
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
    records = [request(request_id=str(i), start_offset_ms=i * 1000, ttft_ms=i * 1000) for i in range(60)]
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
        observations=[SimpleNamespace(source_file_id="truth", source_row=i) for i in (1, 4, 5)],
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


def test_summary_counts_distinct_recorded_concurrency_with_units():
    runs = []
    for value, unit in [
        (32, "session_trees"),
        (4, "session_trees"),
        (4, "session_trees"),
        (4, "requests"),
        (None, "session_trees"),
        (16, "unknown"),
    ]:
        item = run(str(len(runs)), None)
        item["workload"] = {"concurrency": value, "concurrency_unit": unit}
        runs.append(item)
    summary = workload_summary({"runs": runs, "unattributed_measurements": 0})
    assert summary["run_count"] == 6
    assert summary["concurrency_settings"] == [
        {"unit": "requests", "value": 4},
        {"unit": "session_trees", "value": 4},
        {"unit": "session_trees", "value": 32},
    ]
    assert workload_summary({"runs": [], "unattributed_measurements": 0})["concurrency_settings"] == []


@pytest.mark.parametrize("mutation", ["overlap", "helper", "request", "duplicate", "negative"])
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
        manifest["collection_runs"][0]["request_metrics"].update(status="available", file_ids=["absent"])
    elif mutation == "duplicate":
        manifest["collection_runs"].append(run("run-a", [[3, 4]]))
    else:
        manifest["collection_runs"][0]["workload"]["concurrency"] = -1
    with pytest.raises(DataError):
        validate_collection_manifest(manifest)


RECORD_SCHEMA = pa.schema(
    [
        ("source_row", pa.int64()),
        ("measurement_id", pa.string()),
        ("phase", pa.string()),
        ("batch_size", pa.int64()),
        ("total_prefill_tokens", pa.int64()),
        ("total_kv_read_tokens", pa.int64()),
        ("truth_latency_ms", pa.float64()),
        ("source_file", pa.string()),
        ("original_source_row", pa.int64()),
    ]
)


def reduced_record(source_row=7, **changes):
    return dict(
        dict(
            source_row=source_row,
            measurement_id=f"point-{source_row}",
            phase="decode",
            batch_size=2,
            total_prefill_tokens=0,
            total_kv_read_tokens=128,
            truth_latency_ms=13.0,
            source_file=None,
            original_source_row=None,
        ),
        **changes,
    )


def write_measurements(tmp_path, records, schema_id, *, schema=None, row_group_size=1, **metadata):
    path = tmp_path / "truth.parquet"
    table = pa.Table.from_pylist(records, schema=schema).replace_schema_metadata(
        {b"schema_id": schema_id.encode(), b"schema_version": b"1", b"file_metadata": json.dumps(metadata).encode()}
    )
    pq.write_table(table, path, row_group_size=row_group_size)
    return MeasurementFile(
        "truth",
        "truth.parquet",
        "0" * 64,
        "truth",
        path,
        "https://example.com/truth.parquet",
        storage_format="parquet",
        storage_schema=schema_id,
        logical_row_count=len(records),
    )


@pytest.mark.parametrize("layout", ["pre_grouped_rank_lists", "benchmark", "reduced"])
def test_parquet_logical_ids_reach_collection_bindings(tmp_path, layout):
    records = []
    for source_row in (7, 23):
        if layout == "reduced":
            records.append(reduced_record(source_row, source_file="original.csv", original_source_row=99))
            continue
        ranks = [
            dict(
                _fpm_payload(rank=rank, counter=source_row, wall_time=latency),
                received_at_ns=1789002372297086123 + source_row,
                queued_requests={"num_decode_requests": 0},
                extensions_json="{}",
            )
            for rank, latency in enumerate((0.01, 0.013))
        ]
        metadata = dict(
            benchmark_id=source_row,
            point=_benchmark_point(source_row),
            complete=True,
            expected_dp_ranks=[0, 1],
            wall_time=0.013,
            _rank_result_metadata=[{"dp_rank": 0}, {"dp_rank": 1}],
        )
        records.append(
            dict(source_row=source_row, layout=layout, rank_measurements=ranks, metadata_json=json.dumps(metadata))
        )
    baseline = None
    for size in (1, 2):
        file = write_measurements(
            tmp_path,
            records,
            "forward-pass-records-parquet-v1" if layout == "reduced" else "fpm-iterations-parquet-v1",
            schema=RECORD_SCHEMA if layout == "reduced" else None,
            row_group_size=size,
            has_iteration_groups=True,
        )
        file = replace(
            file,
            source_layout="benchmark",
            representation="pre_grouped_rank_lists" if layout == "pre_grouped_rank_lists" else None,
            iteration_count=2,
            rank_record_count=4,
        )
        protocol = "forward-pass-record-v1" if layout == "reduced" else "forward-pass-measurement-v1"
        parsed = adapter_for(protocol).parse("config", [file], 2)
        observations = HfDataset._materialize_observations("config", parsed.observations)
        assert [o.source_row for o in observations] == [7, 23]
        assert [o.actual_ms for o in observations] == [13.0, 13.0]
        assert [[r.dp_rank for r in o.iteration.ranks] for o in observations] == (
            [[0], [0]] if layout == "reduced" else [[0, 1], [0, 1]]
        )
        if layout == "reduced":
            assert [o.iteration.ranks[0].counter_id for o in observations] == [0, 1]
        identity = [(o.observation_id, o.prediction_input()) for o in observations]
        if baseline is not None:
            assert identity == baseline
        baseline = identity
        case = SimpleNamespace(
            observations=observations[:1],
            helper_files=[],
            collection_runs=[run("first", [[7, 8]]), run("second", [[23, 24]])],
        )
        details = workload_details(case)
        assert [(r["id"], r["measurement_count"]) for r in details["runs"]] == [("first", 1)]
        assert details["unattributed_measurements"] == 0


@pytest.mark.parametrize(
    "provenance",
    [{}, {"source_file": "original.csv"}, {"source_row": 99}, {"source_file": "original.csv", "source_row": 99}],
)
def test_reduced_parquet_preserves_optional_csv_provenance(tmp_path, provenance):
    record = reduced_record(2)
    legacy = {k: v for k, v in record.items() if k not in ("source_row", "source_file", "original_source_row")}
    legacy.update(provenance)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(legacy))
    writer.writeheader()
    writer.writerow(legacy)
    path = tmp_path / "truth.csv"
    path.write_text(stream.getvalue())
    original = MeasurementFile("truth", "truth.csv", "0" * 64, "derived_truth", path, "https://example.com")
    parser = adapter_for("forward-pass-record-v1")
    expected = parser.parse("config", [original], 1)
    record.update(source_file=provenance.get("source_file"), original_source_row=provenance.get("source_row"))
    columnar = write_measurements(tmp_path, [record], "forward-pass-records-parquet-v1", schema=RECORD_SCHEMA)
    actual = parser.parse("config", [columnar], 1)
    assert actual.issues == expected.issues == ()
    assert actual.observations[0].source_row == expected.observations[0].source_row == 2
    assert [r.to_aic_dict(include_observation=True) for r in actual.observations[0].iteration.ranks] == [
        r.to_aic_dict(include_observation=True) for r in expected.observations[0].iteration.ranks
    ]


@pytest.mark.parametrize(
    "changes, error", [({"source_file": ""}, "empty source_file"), ({"original_source_row": 0}, "source_row")]
)
def test_reduced_parquet_rejects_invalid_supplied_provenance(tmp_path, changes, error):
    file = write_measurements(
        tmp_path, [reduced_record(**changes)], "forward-pass-records-parquet-v1", schema=RECORD_SCHEMA
    )
    with pytest.raises(DataError, match=error):
        adapter_for("forward-pass-record-v1").parse("config", [file], 1)
