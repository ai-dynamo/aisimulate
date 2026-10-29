# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration checks for report inputs, portable silicon, and CLI passthrough."""

import copy
import hashlib
import json
import re
import sys
from unittest.mock import Mock

import interactive_report
import pytest
import render_report
from silicon_metrics import load_silicon_metrics

EPOCH = 1_792_599_226_802_738_376
POINT = 446063
NAMESPACE = "weka:1111111111111111"


def write_json(path, value):
    path.write_text(json.dumps(value))


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "observed-report"
    output.mkdir()
    prediction = {
        "mean_e2e_latency_ms": 500.25,
        "output_throughput_tok_s": 1.5,
        "telemetry": {
            "clock": "absolute_virtual_ms",
            "profile_start_ms": 0,
            "sample_interval_ms": 1000,
            "phase_boundaries": [
                {"phase": "profile", "start_ms": 0, "end_ms": 2000},
                {"phase": "preparation", "start_ms": 0, "end_ms": 0},
            ],
        },
    }
    reference = [
        {
            "request_id": "silicon-1",
            "phase": "profiling",
            "source_trace_id": "trace",
            "source_outer_idx": 1,
            "source_inner_idx": None,
            "conversation_id": "conversation",
            "turn_index": 0,
            "start_ns": EPOCH + 100_000_000,
            "end_ns": EPOCH + 1_100_000_000,
            "input_tokens": 100,
            "output_tokens": 3,
            "e2e_ms": 1000,
            "ttft_ms": 100,
            "tpot_ms": 10,
            "error": None,
        }
    ]
    reference_path = source / "reference-requests.json"
    write_json(reference_path, reference)
    write_json(source / "source_namespace_map.json", {"trace": NAMESPACE})
    requests = [
        {
            "uuid": "native-1",
            "request_id": f"{NAMESPACE}:request:outer:1",
            "phase": "profile",
            "arrival_time_ms": 100,
            "first_token_ms": 200,
            "terminal_time_ms": 1100,
            "last_token_ms": 1100,
            "terminal_status": "completed",
            "input_length": 100,
            "output_length": 3,
            "e2e_latency_ms": 1000,
            "ttft_ms": 100,
            "itl_ms": 10,
            "agentic": {"conversation_id": "native-conversation"},
            "routing_history": [{"pool": "agg"}],
        }
    ]
    measurement = {
        "point_id": POINT,
        "variant": "observed_profile_output_lengths",
        "wall_time_seconds": 12.3456789,
        "comparisons": [{"metric": "Unchanged existing metric", "mape_percent": 12.345678901, "count": 1}],
        "provenance": {"reference": {"path": str(reference_path)}},
    }
    samples = [
        {
            "sampled_at_ms": tick * 1000,
            "interval_start_ms": max(tick - 1, 0) * 1000,
            "phase": "profile",
            "traffic": {"duration_s": min(tick, 1), "delivered_output_tokens": tick},
            "decode_scheduler_metrics": [
                {
                    "worker_id": 0,
                    "dp_rank": 0,
                    "physical_cache_usage": 0.5,
                    "total_blocks": 100,
                    "running_requests": 1,
                    "waiting_requests": 0,
                }
            ],
        }
        for tick in range(3)
    ]
    raw_path = source / "server-metrics.json"
    write_json(
        raw_path,
        {
            "schema_version": "1.1",
            "metrics_phase": "profiling",
            "benchmark_id": "self-authored-adapter-fixture",
            "summary": {"phase_time_ranges": {"profiling": {"start_ns": EPOCH, "end_ns": EPOCH + 2_000_000_000}}},
            "metrics": {},
        },
    )
    return {
        "output": output,
        "prediction": prediction,
        "measurement": measurement,
        "requests": requests,
        "reference": reference,
        "lifecycle": [],
        "samples": samples,
        "provenance": {"reference_requests": str(reference_path)},
        "silicon": load_silicon_metrics(raw_path, POINT),
        "namespace_map": {"trace": NAMESPACE},
    }


def report_data(path):
    document = path.read_text()
    match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', document, re.S)
    assert match is not None
    return json.loads(match.group(1))


def test_portable_exports_preserve_inputs_and_existing_metrics(inputs):
    before = copy.deepcopy(inputs)
    source = inputs["output"].parent / "source" / "server-metrics.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    path = interactive_report.write_interactive_report(**inputs)
    document = path.read_bytes()
    data = report_data(path)
    compact_path = inputs["output"] / "silicon-telemetry.json"
    compact = load_silicon_metrics(compact_path, POINT)
    comparisons_path = inputs["output"] / "chart-comparisons.json"
    comparisons = json.loads(comparisons_path.read_text())
    assert compact == inputs["silicon"]
    assert compact["provenance"]["source_sha256"] == digest
    assert data["datasets"][0]["measurement"] == inputs["measurement"]
    assert comparisons["datasets"][0]["comparison"] == data["datasets"][0]["comparison"]
    assert comparisons["silicon_provenance"] == compact["provenance"]
    assert data["datasets"][0]["prediction"]["mean_e2e_latency_ms"] == 500.25
    assert inputs == before
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    interactive_report.write_interactive_report(**{**inputs, "silicon": compact})
    assert path.read_bytes() == document


def test_absent_silicon_removes_stale_optional_export(inputs):
    interactive_report.write_interactive_report(**inputs)
    assert (inputs["output"] / "silicon-telemetry.json").exists()
    path = interactive_report.write_interactive_report(**{**inputs, "silicon": None})
    data = report_data(path)
    assert data["silicon"] is None
    assert not (inputs["output"] / "silicon-telemetry.json").exists()
    comparisons = json.loads((inputs["output"] / "chart-comparisons.json").read_text())
    assert comparisons["silicon_provenance"] is None
    cache = data["datasets"][0]["comparison"]["phases"]["profile"]["charts"]["cache"]
    assert all(score["mape_percent"] is None for score in cache["scores"])


def test_point_mismatch_fails_before_writing_outputs(inputs):
    silicon = copy.deepcopy(inputs["silicon"])
    silicon["point_id"] = 438956
    with pytest.raises(ValueError, match="point IDs differ"):
        interactive_report.write_interactive_report(**{**inputs, "silicon": silicon})
    assert list(inputs["output"].iterdir()) == []


def add_sibling(inputs, point_id=POINT):
    parent = inputs["output"].parent
    run = parent / "raw-replay"
    report = parent / "raw-report"
    run.mkdir()
    report.mkdir()
    write_json(run / "prediction.json", inputs["prediction"])
    write_json(
        report / "measurement.json",
        {**inputs["measurement"], "point_id": point_id, "variant": "unmodified_full_corpus"},
    )
    for artifact, values in (
        ("telemetry-requests", "requests"),
        ("telemetry-lifecycle", "lifecycle"),
        ("telemetry-samples", "samples"),
    ):
        write_rows(run / f"{artifact}.jsonl", inputs[values])


def test_worker_identity_survives_switching_replay_variants(inputs):
    add_sibling(inputs)
    source = inputs["output"].parent / "source"
    write_json(
        source / "request-timeline.json",
        {
            "requests": [
                {
                    "phase": "profiling",
                    "srcTrace": "trace",
                    "srcOuter": 1,
                    "srcInner": None,
                    "cid": "conversation",
                    "ti": 0,
                    "wid": "silicon-worker",
                }
            ]
        },
    )
    before = copy.deepcopy(inputs["reference"])
    data = interactive_report.build_report_data(**inputs)
    assert {dataset["id"] for dataset in data["datasets"]} == {"raw", "observed"}
    assert all(dataset["reference"][0].get("worker_id") == "silicon-worker" for dataset in data["datasets"])
    assert inputs["reference"] == before


def test_sibling_from_different_point_is_not_mixed(inputs):
    add_sibling(inputs, point_id=438956)
    data = interactive_report.build_report_data(**inputs)
    assert [dataset["id"] for dataset in data["datasets"]] == ["observed"]


def test_cli_passes_compact_silicon_to_renderer(inputs, monkeypatch):
    parent = inputs["output"].parent
    run = parent / "observed-replay"
    run.mkdir()
    write_json(run / "prediction.json", inputs["prediction"])
    for artifact, key in (
        ("telemetry-requests", "requests"),
        ("telemetry-lifecycle", "lifecycle"),
        ("telemetry-samples", "samples"),
    ):
        write_rows(run / f"{artifact}.jsonl", inputs[key])
    measurement = parent / "measurement.json"
    write_json(measurement, inputs["measurement"])
    compact = parent / "silicon.json"
    write_json(compact, inputs["silicon"])
    before = measurement.read_bytes()
    for name in ("timeline", "request_metrics", "throughput", "scheduler_cache", "distributions", "comparison_plot"):
        monkeypatch.setattr(render_report, name, Mock())
    render = Mock()
    monkeypatch.setattr(render_report, "render_html", render)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_report.py",
            "--run-dir",
            str(run),
            "--reference",
            inputs["provenance"]["reference_requests"],
            "--measurement",
            str(measurement),
            "--output-dir",
            str(inputs["output"]),
            "--silicon-metrics",
            str(compact),
        ],
    )
    render_report.main()
    assert render.call_count == 1
    assert render.call_args.args[-1] == inputs["silicon"]
    assert render.call_args.args[3] == inputs["measurement"]
    assert measurement.read_bytes() == before
