# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic producer records cover omissions that aggregate metrics hide."""

import hashlib
import json

import pytest

from normalize import add_stream_observations, metric, reference_row, retain_cancelled_accounting, retain_harness_receipt
from collect import main, qualified_pilot


def test_resume_requires_checksum_pinned_successful_pilot(tmp_path):
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps({"schema": "aisimulate.qwen-weka.pilot-qualification.v1", "status": "qualified"}))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    assert qualified_pilot(path, checksum)["status"] == "qualified"
    path.write_text(json.dumps({"schema": "aisimulate.qwen-weka.pilot-qualification.v1", "status": "failed"}))
    with pytest.raises(ValueError, match="checksum"):
        qualified_pilot(path, checksum)
    with pytest.raises(ValueError, match="did not qualify"):
        qualified_pilot(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("lanes", [["4", "4"], ["4", "1"], ["4", "8", "10", "13", "14"]])
def test_partial_campaign_requires_qualified_evidence_and_ordered_unique_lanes(tmp_path, monkeypatch, lanes):
    monkeypatch.setattr("sys.argv", ["collect", "--root", str(tmp_path / "run"), "--lanes", *lanes])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_cached_harness_receipt_uses_actual_export_and_dataset_metadata(tmp_path):
    run, output = tmp_path / "run", tmp_path / "normalized"
    (run / "benchmark").mkdir(parents=True)
    (run / "client-record").mkdir()
    output.mkdir()
    summary = run / "benchmark/profile_export_aiperf.json"
    summary.write_text(json.dumps({"input_config": {"datasets": [{"name": "main"}], "phases": [{"duration": 3600}]}}))
    metadata = run / "client-record/dataset-metadata.json"
    metadata.write_text(json.dumps({"sampling_strategy": "sequential", "conversations": [{"conversation_id": "play", "is_root": True}]}))
    result = retain_harness_receipt(run, output, {"play": "namespace"})
    assert result["datasets"][0]["sampling"] == "sequential"
    origin = json.loads((output / "harness-receipt-origin.json").read_bytes())
    assert origin["configuration"]["sha256"] == hashlib.sha256(summary.read_bytes()).hexdigest()
    assert origin["sampling"]["sha256"] == hashlib.sha256(metadata.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="eligible corpus"):
        retain_harness_receipt(run, output, {"different": "namespace"})
    summary.write_text(json.dumps({"input_config": {"datasets": [{"sampling": "random"}]}}))
    with pytest.raises(ValueError, match="sampling disagree"):
        retain_harness_receipt(run, output, {"play": "namespace"})


def metadata(identity="a", phase="profiling", **changes):
    return {
        "x_request_id": identity,
        "phase_kind": phase,
        "source_trace_id": "play",
        "source_outer_idx": 1,
        "request_start_ns": 2_000_000_000_000_000_000,
        "request_end_ns": 2_000_000_000_020_000_000,
        **changes,
    }


def test_token_units_are_integral_and_invalid_measurements_stay_unavailable():
    assert metric({"metrics": {"length": {"value": 2.0, "unit": "tokens"}}}, "length", "tokens") == 2
    assert metric({"metrics": {"length": {"value": 2.5, "unit": "tokens"}}}, "length", "tokens") is None
    with pytest.raises(ValueError, match="unit"):
        metric({"metrics": {"length": {"value": 2, "unit": "ms"}}}, "length", "tokens")


def test_raw_overflow_records_remain_and_stream_counts_conserve_tokens(tmp_path):
    namespaces = {"play": "weka:0123456789abcdef"}
    row = {"metadata": metadata(), "metrics": {"output_sequence_length": {"value": 2, "unit": "tokens"}}}
    refs = [reference_row(row, namespaces)]
    event = {
        "choices": [{"token_ids": [1, 2], "delta": None}],
        "usage": {"prompt_tokens_details": {"cached_tokens": 2}},
    }
    records = [
        {
            "metadata": metadata(),
            "start_perf_ns": 1000,
            "responses": [{"perf_ns": 2_001_000, "packets": [{"name": "data", "value": json.dumps(event)}]}],
        },
        {
            "metadata": metadata("overflow", context_overflow_skip=True),
            "start_perf_ns": 1000,
            "responses": [],
            "error": {"message": "overflow"},
        },
    ]
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(record) + "\n" for record in records))
    add_stream_observations(refs, raw, namespaces)
    assert len(refs) == 2 and refs[1]["error"] == "context_overflow"
    assert refs[0]["chunks"] == [{"at_ns": "2000000000002000000", "count": 2}]
    assert refs[0]["cached_tokens"] == 2


def accounting_log(completed=1, cancelled=1):
    return (
        "Phase warmup (warmup) complete | completed=1, cancelled=0, errors=0\n"
        f"Phase profiling (profiling) complete | completed={completed}, cancelled={cancelled}, errors=0\n"
        "Received CREDIT_PHASE_COMPLETE message, Phase complete: PhaseRecordsStats(phase_kind='profiling', "
        "start_ns=1000000000, sent_end_ns=601000000000, requests_end_ns=631000000000, "
        f"final_requests_completed={completed}, final_requests_cancelled={cancelled}, final_request_errors=0)\n"
    )


def test_unrecorded_cancellations_are_counted_without_invented_timings():
    references = [{"phase": "preparation"}, {"phase": "profile"}]
    ranges = {
        "warmup": {"start_ns": 0, "end_ns": 1_000_000_000},
        "profiling": {"start_ns": 1_000_000_000, "end_ns": 631_000_000_000},
    }
    result = retain_cancelled_accounting(references, accounting_log(), ranges)
    assert result["profile"]["cancelled_without_request_record"] == 1
    assert len(references) == 3 and references[-1]["error"] == "measurement_cancelled"
    assert references[-1]["start_ns"] is references[-1]["end_ns"] is None
    assert references[-1]["identity_valid"] is False
    with pytest.raises(ValueError, match="reconcile"):
        retain_cancelled_accounting([{"phase": "preparation"}, {"phase": "profile"}], accounting_log(2, 0), ranges)
