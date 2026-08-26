# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json

import pytest

from tools.support_matrix.build_fpe_supermatrix import build_web_rows, load_artifacts, write_web_matrix

pytestmark = pytest.mark.unit


def _row(*, roles, phase, status="PASS", latency_ms=1.25, **overrides):
    row = {
        "model": "test/model",
        "architecture": "TestForCausalLM",
        "system": "b200_sxm",
        "backend": "vllm",
        "backend_version": "0.22.0",
        "forward_model": "op_level",
        "roles": roles,
        "tp_size": 1,
        "pp_size": 1,
        "attention_dp_size": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 1,
        "cp_size": 1,
        "gemm_quant_mode": "bfloat16",
        "moe_quant_mode": "bfloat16",
        "kvcache_quant_mode": "bfloat16",
        "fmha_quant_mode": "bfloat16",
        "comm_quant_mode": "half",
        "nextn": 0,
        "phase": phase,
        "status": status,
        "latency_ms": latency_ms,
        "source": "silicon" if status == "PASS" else "",
        "error_message": ("" if status == "PASS" else "missing /home/runner/work/repo/python/aisimulate/op_level data"),
    }
    row.update(overrides)
    return row


def _metadata():
    return {"schema_version": 1, "source_version": "0.12.0", "source_sha": "abc123", "workload": {}}


def test_build_web_rows_preserves_existing_agg_disagg_contract_and_real_latencies():
    rows = [
        _row(roles="agg|prefill|decode", phase="prefill", latency_ms=1.0),
        _row(roles="agg|prefill|decode", phase="decode_start", latency_ms=2.0),
        _row(roles="agg|prefill|decode", phase="decode_end", latency_ms=3.0),
        _row(roles="agg|prefill|decode", phase="mixed", latency_ms=4.0),
    ]

    result = build_web_rows(rows, _metadata())

    assert [(row["Mode"], row["Status"]) for row in result] == [("agg", "PASS"), ("disagg", "PASS")]
    assert result[0]["FPEPhaseLatencyMs"] == (
        "decode_end=3.000000, decode_start=2.000000, mixed=4.000000, prefill=1.000000"
    )
    assert result[0]["FPEProbeCount"] == "4"
    assert result[0]["SourceSHA"] == "abc123"


def test_build_web_rows_fails_closed_and_preserves_native_status_counts():
    rows = [
        _row(roles="agg|prefill|decode", phase=phase, status="PERF_DATA_MISSING", latency_ms=None)
        for phase in ("prefill", "decode_start", "decode_end", "mixed")
    ]

    result = build_web_rows(rows, _metadata())

    assert {row["Status"] for row in result} == {"FAIL"}
    assert {row["FPEStatusCounts"] for row in result} == {"PERF_DATA_MISSING=4"}
    assert all("source_sha=abc123" in row["ErrMsg"] for row in result)
    assert all("missing <repo>/python/aisimulate/op_level data" in row["ErrMsg"] for row in result)


def test_write_web_matrix_uses_split_csv_index(tmp_path):
    rows = build_web_rows([_row(roles="prefill", phase="prefill")], _metadata())

    index = write_web_matrix(rows, tmp_path)

    assert index == {"files": ["b200_sxm.csv"]}
    assert json.loads((tmp_path / "index.json").read_text()) == index
    with (tmp_path / "b200_sxm.csv").open(newline="") as handle:
        written = list(csv.DictReader(handle))
    assert written == rows


def test_load_artifacts_rejects_non_op_level_forward_models(tmp_path):
    path = tmp_path / "fpe_support_matrix.json"
    path.write_text(
        json.dumps(
            {
                "metadata": _metadata(),
                "results": [_row(roles="agg", phase="prefill", forward_model="fpm")],
            }
        )
    )

    with pytest.raises(ValueError, match="non-op-level"):
        load_artifacts([path])
