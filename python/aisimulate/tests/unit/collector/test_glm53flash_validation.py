# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic CPU protocol fixtures; these tests are not GPU accuracy evidence."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from collector.fpm_forward import glm53flash_validation as validation
from collector.glm53flash_protocol import PROTOCOL

pytestmark = pytest.mark.unit


def write_plan(tmp_path, key, role):
    backend, quant, tp, phase = key
    offset = 0 if role == "holdout" else 128
    points = [
        {
            "batch_size": 1,
            "total_kv_read_tokens": length + offset - (256 if phase == "prefill" else 0),
            "total_prefill_tokens": 256 if phase == "prefill" else 0,
        }
        for length in (2048, 60000, 120000)
    ]
    frozen = {
        "schema_version": 3,
        "prefill": points if phase == "prefill" else [],
        "decode": points if phase == "decode" else [],
    }
    corpus = ("a" if role == "calibration" else "b") * 64
    cell_id = f"{backend}-{quant}-{tp}-{phase}-{role}"
    cell = {
        "cell_id": cell_id,
        "backend": backend,
        "state_protocol": PROTOCOL,
        "weight_quantization": quant,
        "workload_kind": phase,
        "parallel_strategy": "pure_tp",
        "input_text_sha256": corpus,
        "topology": {"tp": tp, "pp": 1, "dp": 1, "cp": 1, "moe_tp": tp, "moe_ep": 1},
        "execution_identity": {
            "model_config_sha256": "c" * 64,
            "execution_profile": "full",
            "engram_residency": "none",
            "input_modality": "text",
        },
    }
    plan = {
        "schema_name": "aic_fpm_collection_plan",
        "system": "gb300",
        "backend": backend,
        "sha256": "d" * 64,
        "model_path": "zai-org/GLM-5.3-Flash" if quant == "fp8" else "nvidia/GLM-5.3-Flash-NVFP4",
        "cells": [cell],
        "options": {
            "dataset_role": role,
            "input_text_sha256": corpus,
            "benchmark_points": {"payload": frozen, "sha256": validation.digest(frozen)},
        },
    }
    path = tmp_path / f"{cell_id}.json"
    path.write_text(json.dumps(plan))
    return {
        "plan": {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "cell_id": cell_id,
        "attempt_id": f"attempt-{cell_id}",
        "raw_root": f"raw-{cell_id}",
    }


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    manifest = {
        "schema": validation.SCHEMA,
        "mode": "fpm",
        "entries": [
            {role: write_plan(tmp_path, key, role) for role in ("calibration", "holdout")}
            for key in validation.REQUIRED
        ],
    }

    def native(run, base):
        return {
            "values": {point["benchmark_id"]: point["benchmark_id"] * 10.0 for point in run["points"]},
            "request_ids": {run["spec"]["cell_id"]},
            "receipts": [{"path": "raw.json", "sha256": "e" * 64}],
            "runtime_run_id": run["spec"]["cell_id"],
            "backend_version": "0.30.0",
        }

    def predict(run, entry, mode, base, **kwargs):
        return {
            "rows": {point["benchmark_id"]: {"prediction_ms": point["benchmark_id"] * 10.5} for point in run["points"]},
            "config": {"mode": mode},
        }

    monkeypatch.setattr(validation, "_native_run", native)
    monkeypatch.setattr(
        validation,
        "_load_native",
        lambda run, base, mode: dict(validation._native_run(run, base), timing_boundary="test_gpu_boundary"),
    )
    monkeypatch.setattr(validation, "_predict", predict)
    monkeypatch.setattr(validation, "installed_consumer_identity", lambda: {"payload_sha256": "f" * 64})
    return manifest


def test_complete_matrix_scores_all_points_and_reports_every_required_group(campaign, tmp_path):
    report = validation.evaluate(campaign, tmp_path)
    assert report["acceptance"] == "PASSED"
    assert report["coverage"] == {
        "required_configurations": 8,
        "required_phase_cells": 16,
        "passed_phase_cells": 16,
        "requested_points": 48,
        "compared_points": 48,
    }
    for cell in report["cells"]:
        assert cell["metrics"]["mape_pct"] == 5
        assert cell["metrics"]["wape_pct"] == 5
        assert cell["metrics"]["p95_ape_pct"] == 5
        assert cell["metrics"]["max_ape_pct"] == 5
        assert all(cell["groups"][group]["requested"] == 1 for group in validation.GROUPS)
    assert report["http_metrics"]["acceptance"] == "NOT_EVALUATED"


@pytest.mark.parametrize("label", ["corpus", "geometry", "request_id"])
def test_calibration_holdout_overlap_blocks_every_acceptance(campaign, tmp_path, monkeypatch, label):
    if label == "request_id":
        original = validation._native_run

        def overlapping(*args):
            result = original(*args)
            result["request_ids"] = {"reused-request"}
            return result

        monkeypatch.setattr(validation, "_native_run", overlapping)
    else:
        spec = campaign["entries"][0]["holdout"]
        path = tmp_path / spec["plan"]["path"]
        plan = json.loads(path.read_text())
        if label == "corpus":
            plan["cells"][0]["input_text_sha256"] = "a" * 64
            plan["options"]["input_text_sha256"] = "a" * 64
        else:
            calibration = campaign["entries"][0]["calibration"]
            calibration_plan = json.loads((tmp_path / calibration["plan"]["path"]).read_text())
            plan["options"]["benchmark_points"] = calibration_plan["options"]["benchmark_points"]
        path.write_text(json.dumps(plan))
        spec["plan"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = validation.evaluate(campaign, tmp_path)
    assert report["acceptance"] == "FAILED"
    assert report["coverage"]["requested_points"] == 48
    assert any(label in error["error"] for error in report["errors"])


@pytest.mark.parametrize(
    "failure, expected",
    [(FileNotFoundError("not collected"), "NOT_EVALUATED"), (ValueError("missing native repetition"), "FAILED")],
)
def test_missing_or_failed_native_data_never_drops_requested_points(campaign, tmp_path, monkeypatch, failure, expected):
    def fail(*args):
        raise failure

    monkeypatch.setattr(validation, "_native_run", fail)
    report = validation.evaluate(campaign, tmp_path)
    assert report["acceptance"] == expected
    assert report["coverage"]["requested_points"] == 48
    assert report["coverage"]["compared_points"] == 0
    assert all(len(cell["points"]) == 3 for cell in report["cells"])


def test_threshold_is_per_cell_per_phase_and_ops_mode_is_explicit(campaign, tmp_path, monkeypatch):
    original = validation._predict

    def worse(run, *args, **kwargs):
        result = original(run, *args, **kwargs)
        if run["key"] == validation.REQUIRED[0]:
            for row in result["rows"].values():
                row["prediction_ms"] *= 1.15 / 1.05
        return result

    monkeypatch.setattr(validation, "_predict", worse)
    report = validation.evaluate(campaign, tmp_path)
    assert report["acceptance"] == "FAILED"
    assert report["coverage"]["passed_phase_cells"] == 15
    campaign["mode"] = "ops"
    assert validation.evaluate(campaign, tmp_path)["acceptance"] == "PASSED"


def test_failed_prediction_remains_in_denominator(campaign, tmp_path, monkeypatch):
    original = validation._predict

    def fail_one(*args, **kwargs):
        result = original(*args, **kwargs)
        result["rows"][2] = {"error": "missing strict measured operator"}
        return result

    monkeypatch.setattr(validation, "_predict", fail_one)
    report = validation.evaluate(campaign, tmp_path)
    assert report["acceptance"] == "FAILED"
    assert report["coverage"]["requested_points"] == 48
    assert report["coverage"]["compared_points"] == 32
    assert all(cell["metrics"]["coverage"] == pytest.approx(2 / 3) for cell in report["cells"])


def test_missing_matrix_cells_and_uninstalled_consumer_cannot_pass(campaign, tmp_path, monkeypatch):
    campaign["entries"] = campaign["entries"][:1]
    assert validation.evaluate(campaign, tmp_path)["acceptance"] == "NOT_EVALUATED"

    def unavailable():
        raise ValueError("editable install")

    monkeypatch.setattr(validation, "installed_consumer_identity", unavailable)
    report = validation.evaluate(campaign, tmp_path)
    assert report["acceptance"] == "NOT_EVALUATED"
    assert report["coverage"]["compared_points"] == 0


def test_receipt_tampering_is_rejected(campaign, tmp_path):
    receipt = campaign["entries"][0]["holdout"]["plan"]
    (tmp_path / receipt["path"]).write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch"):
        validation.evaluate(campaign, tmp_path)


def test_empty_campaign_cli_writes_not_evaluated_matrix(tmp_path):
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"schema": validation.SCHEMA, "mode": "fpm", "entries": []}))
    output = tmp_path / "report.json"
    assert validation.main(["--manifest", str(manifest), "--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["acceptance"] == "NOT_EVALUATED"
    assert len(report["cells"]) == 16


def test_metrics_keep_wape_distinct_and_use_nearest_rank_p95():
    rows = [
        {"status": "MEASURED_AND_PREDICTED", "measured_ms": 1, "prediction_ms": 2},
        {"status": "MEASURED_AND_PREDICTED", "measured_ms": 99, "prediction_ms": 99},
        {"status": "NOT_EVALUATED"},
    ]
    result = validation.metrics(rows)
    assert result["mape_pct"] == 50
    assert result["wape_pct"] == 1
    assert result["p95_ape_pct"] == 100
    assert result["requested"] == 3 and result["compared"] == 2


def test_native_adapter_uses_validated_producer_timings_and_real_request_ids(tmp_path, monkeypatch):
    key = validation.REQUIRED[0]
    spec = write_plan(tmp_path, key, "holdout")
    run = validation._plan_run(spec, tmp_path, "holdout")
    root = tmp_path / spec["raw_root"]
    root.mkdir()
    (root / "benchmark.token-streams.jsonl").write_text(
        json.dumps({"requests": [{"request_id": "actual-request"}]}) + "\n"
    )
    calls = []

    def reader(cell, path, **kwargs):
        calls.append((cell.cell_id, path, kwargs))
        return SimpleNamespace(
            points=[SimpleNamespace(point=point, rank_wall_times=((0, 0.012),)) for point in run["points"]],
            input_provenance={
                "text_sha256": "b" * 64,
                "tokenizer_revision": validation.MODEL_REVISIONS[run["plan"]["model_path"]],
            },
            runtime_run_id="actual-run",
            runtime_grid_digest="actual-grid",
            backend_version="0.30.0",
        )

    monkeypatch.setattr(validation, "validate_native_collection", reader)
    result = validation._native_run(run, tmp_path)
    assert result["values"] == {1: 12, 2: 12, 3: 12}
    assert result["request_ids"] == {"actual-request"}
    assert calls[0][2] == {"expected_plan_sha256": "d" * 64, "expected_attempt_id": spec["attempt_id"]}


def calibration_table(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    key = validation.REQUIRED[0]
    spec = write_plan(tmp_path, key, "calibration")
    run = validation._plan_run(spec, tmp_path, "calibration")
    native = {
        "values": {1: 10.0, 2: 20.0, 3: 30.0},
        "runtime_run_id": "real-run",
        "runtime_grid_digest": "f" * 64,
        "backend_version": "0.30.0",
        "input_provenance": {"token_ids_sha256": "e" * 64},
    }
    rows = [
        dict(
            point,
            model_path=run["plan"]["model_path"],
            backend="vllm",
            weight_quantization="fp8",
            tp=2,
            workload_kind="prefill",
            cell_id=run["cell"]["cell_id"],
            source_plan_sha256=run["plan"]["sha256"],
            collector_attempt_id=spec["attempt_id"],
            runtime_run_id=native["runtime_run_id"],
            runtime_grid_digest=native["runtime_grid_digest"],
            input_text_sha256=run["corpus"],
            input_token_ids_sha256="e" * 64,
            input_tokenizer_revision=validation.MODEL_REVISIONS[run["plan"]["model_path"]],
            state_protocol=PROTOCOL,
            timing_boundary=validation.TIMING_BOUNDARIES["vllm"],
            backend_version="0.30.0",
            pp=1,
            dp=1,
            cp=1,
            moe_tp=2,
            moe_ep=1,
            **run["cell"]["execution_identity"],
            latency_ms=native["values"][point["benchmark_id"]],
        )
        for point in run["points"]
    ]
    path = tmp_path / "fpm_forward_perf.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path, run, native, rows


def test_consumer_fpm_rows_bind_actual_native_calibration(tmp_path):
    path, run, native, _ = calibration_table(tmp_path)
    assert validation._bind_fpm_rows([path], run, native)["rows"] == 3


@pytest.mark.parametrize("change", ["holdout_corpus", "attempt", "latency", "geometry", "missing", "duplicate"])
def test_consumer_fpm_rejects_tainted_or_incomplete_calibration(tmp_path, change):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path, run, native, rows = calibration_table(tmp_path)
    if change == "holdout_corpus":
        rows[0]["input_text_sha256"] = "b" * 64
    elif change == "attempt":
        rows[0]["collector_attempt_id"] = "different-attempt"
    elif change == "latency":
        rows[0]["latency_ms"] = 11.0
    elif change == "geometry":
        rows[0]["total_prefill_tokens"] += 1
    elif change == "missing":
        rows.pop()
    else:
        rows.append(rows[0])
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="consumer FPM"):
        validation._bind_fpm_rows([path], run, native)


@pytest.mark.parametrize("mode", ["fpm", "ops"])
def test_predict_calls_public_sdk_for_every_frozen_point(tmp_path, monkeypatch, mode):
    from aisimulate_core.sdk import rust_engine_step

    path, calibration, native, _ = calibration_table(tmp_path)
    holdout = validation._plan_run(write_plan(tmp_path, validation.REQUIRED[0], "holdout"), tmp_path, "holdout")
    calls, closed = [], []
    if mode == "ops":
        import sys

        monkeypatch.setitem(
            sys.modules,
            "collector.glm53flash_validation",
            SimpleNamespace(
                bind_calibration=lambda paths, run, receipt: {"native_runtime_run_id": receipt["runtime_run_id"]}
            ),
        )
    model = SimpleNamespace(
        estimate_forward_pass_time_ms=lambda payload: calls.append(payload) or 12.0,
        diagnostics=lambda: {"public": True},
        close=lambda: closed.append(True),
    )
    configs = []
    monkeypatch.setattr(
        rust_engine_step.RustForwardPassPerfModel, "best_available", lambda cfg: configs.append(cfg) or model
    )
    receipts = [
        {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in tmp_path.iterdir()
        if p.is_file()
    ]
    entry = {"consumer_config": {"backend_version": "0.30.0", "systems_paths": ["."]}, "consumer_data": receipts}
    result = validation._predict(holdout, entry, mode, tmp_path, calibration=calibration, calibration_native=native)
    assert len(calls) == 3 and closed == [True]
    assert [row["prediction_ms"] for row in result["rows"].values()] == [12.0] * 3
    assert configs[0].estimation_mode == ("fpm_interpolation" if mode == "fpm" else "op_level")
    assert configs[0].fallback_policy == "deny"
    assert configs[0].database_mode == "SILICON"
    assert configs[0].fpm_fmha_quant_mode == ("fp8" if mode == "fpm" else None)
    assert result["calibration_binding"]["native_runtime_run_id"] == "real-run"


def test_installed_consumer_follows_unified_runtime_and_rejects_shim_replacement(monkeypatch):
    import base64
    from pathlib import Path

    import aisimulate
    import aisimulate_core
    from aisimulate import _runtime
    from aisimulate_core import _native
    from aisimulate_core.sdk import rust_engine_step

    modules = {
        "aisimulate/__init__.py": aisimulate,
        "aisimulate/" + Path(_runtime.__file__).name: _runtime,
        "aisimulate_core/__init__.py": aisimulate_core,
        "aisimulate_core/_native.py": _native,
        "aisimulate_core/sdk/rust_engine_step.py": rust_engine_step,
    }

    class File(str):
        pass

    files = []
    for name, module in modules.items():
        item = File(name)
        raw = Path(module.__file__).read_bytes()
        item.hash = SimpleNamespace(
            mode="sha256", value=base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        )
        item.size = len(raw)
        files.append(item)
    distribution = SimpleNamespace(
        files=files, version="test-only", locate_file=lambda name: modules[str(name)].__file__
    )
    monkeypatch.setattr(validation.importlib.metadata, "distribution", lambda name: distribution)
    assert validation.installed_consumer_identity()["version"] == "test-only"
    monkeypatch.setattr(_native, "RustForwardPassPerfModel", object())
    with pytest.raises(ValueError, match="canonical native binding"):
        validation.installed_consumer_identity()


def test_ops_native_loader_keeps_gpu_boundary_separate(tmp_path, monkeypatch):
    import sys

    run = validation._plan_run(write_plan(tmp_path, validation.REQUIRED[0], "holdout"), tmp_path, "holdout")
    result = {"values": {1: 1.0, 2: 2.0, 3: 3.0}, "timing_boundary": "embedding_to_logits_gpu_v1"}
    monkeypatch.setitem(
        sys.modules, "collector.glm53flash_validation", SimpleNamespace(load_native=lambda run, base: result)
    )
    assert validation._load_native(run, tmp_path, "ops")["timing_boundary"] == "embedding_to_logits_gpu_v1"
    result["values"].pop(3)
    with pytest.raises(ValueError, match="omits frozen requested points"):
        validation._load_native(run, tmp_path, "ops")


@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_inclusive_context_limit_preserves_database_features(batch, phase):
    query = 32 if phase == "prefill" else 1
    point = {
        "point_type": phase,
        "batch_size": batch,
        "total_prefill_tokens": batch * query if phase == "prefill" else 0,
        "total_kv_read_tokens": batch * (131072 - query),
    }
    assert validation._geometry(point) == (phase, batch, point["total_prefill_tokens"], point["total_kv_read_tokens"])
    assert validation._context_length(point) == 131072
    assert validation._group(point) == "128K"
    point["total_kv_read_tokens"] += batch
    with pytest.raises(ValueError, match="context128K"):
        validation._geometry(point)


@pytest.mark.parametrize(
    "past, group",
    [(1022, "below1K"), (1023, "1K-32K"), (32767, "1K-32K"), (32768, "64K"), (65535, "64K"), (65536, "128K")],
)
def test_decode_context_groups_include_current_token(past, group):
    point = {"point_type": "decode", "batch_size": 2, "total_prefill_tokens": 0, "total_kv_read_tokens": past * 2}
    assert validation._geometry(point)[2] == 0
    assert validation._group(point) == group


def test_sharded_consumer_binding_keeps_each_native_origin_and_rejects_donor(tmp_path):
    import copy

    import pyarrow as pa
    import pyarrow.parquet as pq

    path, run, native, rows = calibration_table(tmp_path)
    children, receipts = [], {}
    for index, row in enumerate(rows, 1):
        child = copy.deepcopy(run)
        cid = f"shard-{index}"
        child["cell"].update(cell_id=cid, weight_quantization="fp8_block")
        child["plan"]["sha256"] = str(index) * 64
        child["spec"]["attempt_id"] = f"attempt-{index}"
        child["points"] = [dict(run["points"][index - 1], benchmark_id=1)]
        receipt = dict(native, values={1: native["values"][index]}, runtime_run_id=f"run-{index}")
        row.update(
            cell_id=cid,
            weight_quantization="fp8_block",
            source_plan_sha256=child["plan"]["sha256"],
            collector_attempt_id=child["spec"]["attempt_id"],
            runtime_run_id=receipt["runtime_run_id"],
        )
        children.append(child)
        receipts[cid] = receipt
    parent = dict(run, children=children)
    pq.write_table(pa.Table.from_pylist(rows), path)
    bound = validation._bind_fpm_rows([path], parent, {"_children": receipts})
    assert len(bound["shards"]) == 3
    assert all(item["rows"] == 1 for item in bound["shards"])
    pq.write_table(pa.Table.from_pylist([*rows, dict(rows[0], cell_id="unfrozen-donor")]), path)
    with pytest.raises(ValueError, match="donor cell"):
        validation._bind_fpm_rows([path], parent, {"_children": receipts})
