# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import statistics
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.fpm_forward import cli, repeatability, runner
from collector.fpm_forward.config import with_kv_warmup_defaults
from collector.fpm_forward.database import aggregate_cell
from collector.fpm_forward.native_artifact import (
    _expected_scheduled,
)
from collector.fpm_forward.runtime import fpm_memory_observer as observer
from collector.fpm_forward.runtime_memory import validate_saved_plan

from .test_fpm_profile_collection import _plan, _profile, no_models_or_timing_data  # noqa: F401
from .test_fpm_runner import _native_payload, _write_provenance
from .test_fpm_runtime_memory import _vllm_config

pytestmark = pytest.mark.unit


def _point(coordinates, phase, index):
    point = {"point_type": phase, "benchmark_id": index, "total_prefill_tokens": 0, **coordinates}
    axis = point["total_prefill_tokens"] or point["batch_size"]
    capture = 1 << (axis - 1).bit_length() if axis <= 16 else None
    point.update(
        expected_cudagraph_mode=("FULL" if phase == "decode" else "PIECEWISE") if capture else "NONE",
        expected_capture_size=capture,
        padding_tokens=capture - axis if capture else None,
        sample_reasons=["kvwarm_real_kv"]
        if phase == "decode"
        else ["prefill_real_seed"]
        if point["total_kv_read_tokens"]
        else [],
    )
    return point


def _write_campaign(
    root, checkpoint_path, plan, *, attempt_id="source", factor=1.0, observed=True, generator_overrides=None
):
    deployment = generator_overrides or {}
    root.mkdir(parents=True, exist_ok=True)
    (root / "collection-plan.json").write_text(json.dumps(plan.to_dict()))
    (root / "generator-overrides.json").write_text(json.dumps(with_kv_warmup_defaults(deployment)))
    checkpoint = {"schema": runner.CHECKPOINT_SCHEMA, "plan_sha256": plan.sha256, "cells": {}}
    for cell in plan.cells:
        directory = root / "cells" / cell.cell_id
        raw = directory / "raw" / "pod"
        raw.mkdir(parents=True, exist_ok=True)
        runner._render_cell(plan, cell, directory, deployment)
        generated_model = json.loads((directory / "generator-request.json").read_text())["ServiceConfig"]["model_path"]
        _write_provenance(
            raw / "collector-provenance.json", cell_id=cell.cell_id, plan_sha256=plan.sha256, attempt_id=attempt_id
        )
        provenance = json.loads((raw / "collector-provenance.json").read_text())
        provenance["runtime"]["backend_version"] = plan.capability.aic_database_version
        (raw / "collector-provenance.json").write_text(json.dumps(provenance))
        coords = json.loads(plan.options.benchmark_points_json)[cell.workload_kind]
        points = [_point(item, cell.workload_kind, index + 1) for index, item in enumerate(coords)]
        for rank in range(cell.topology.dp):
            payload = _native_payload(phase=cell.workload_kind, rank=rank, dp=cell.topology.dp)
            groups = []
            results = []
            for point in points:
                rank_results = [
                    {
                        "dp_rank": dp,
                        "fpms": [
                            {
                                "counter_id": point["benchmark_id"],
                                "dp_rank": dp,
                                "wall_time": factor * (0.01 + dp * 0.001),
                                "scheduled_requests": _expected_scheduled(point),
                            }
                        ],
                    }
                    for dp in range(cell.topology.dp)
                ]
                groups.append(
                    {
                        "benchmark_id": point["benchmark_id"],
                        "point": point,
                        "expected_dp_ranks": list(range(cell.topology.dp)),
                        "complete": True,
                        "rank_results": rank_results,
                        "wall_time": max(row["fpms"][0]["wall_time"] for row in rank_results),
                    }
                )
                results.append(
                    {
                        "point": point,
                        "kv_seed_regime": "real_kv"
                        if cell.workload_kind == "decode"
                        else "real_prefix"
                        if point["total_kv_read_tokens"]
                        else "not_applicable",
                        "fpms": rank_results[rank]["fpms"],
                    }
                )
            payload.update(
                results=results,
                iteration_groups=groups,
                coverage={"expected_points": len(points), "completed_points": len(points), "skipped_points": 0},
                cudagraph={"mode": "FULL_AND_PIECEWISE", "capture_sizes": [1, 2, 4, 8, 16]},
                timing={
                    "benchmark_elapsed_seconds": 100.0,
                    "measured_iteration_seconds": sum(group["wall_time"] for group in groups),
                },
            )
            (raw / f"benchmark-dp{rank}.json").write_text(json.dumps(payload))
            if observed:
                for tp in range(cell.topology.tp):
                    execution = {
                        "schema_name": "aisimulate_fpm_runtime_execution",
                        "schema_version": 1,
                        "collector_provenance": provenance,
                        "backend_version": plan.capability.aic_database_version,
                        "dp_rank": rank,
                        "tp_rank": tp,
                        "pp_rank": 0,
                        "status": "observed",
                        "attention_groups": [
                            {
                                "kv_cache_group_id": 0,
                                "backend_class": "example.ActualBackend",
                                "layer_names": ["layer.0"],
                            }
                        ],
                        "graph_config": {
                            "mode": "VLLM_COMPILE",
                            "cudagraph_mode": "FULL_AND_PIECEWISE",
                            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
                            "max_cudagraph_capture_size": 16,
                        },
                        "resolved_config": _runtime_config(plan, cell),
                    }
                    execution["resolved_config"]["model_config"]["model"] = generated_model
                    (raw / f"fpm-execution-worker-dp{rank}-tp{tp}-pp0.json").write_text(json.dumps(execution))
        (raw / runner.POINTS_RECEIPT_FILENAME).write_text(
            json.dumps(
                {
                    "plan_sha256": plan.sha256,
                    "cell_id": cell.cell_id,
                    "attempt_id": attempt_id,
                    "sha256": plan.options.benchmark_points_sha256,
                    "phase": "after",
                }
            )
        )
        checkpoint["cells"][cell.cell_id] = {"status": "passed", "attempt_id": attempt_id}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(checkpoint))


def _runtime_config(plan, cell):
    """Synthetic initialized vLLM state for the real generated test launch."""
    config = observer.execution_config(_vllm_config(tp=cell.topology.tp, dp=cell.topology.dp))
    config["model_config"].update(
        model=plan.model_path,
        revision=plan.fpm_profile.model_revision,
        max_model_len=plan.options.vllm_max_model_len,
        enforce_eager=plan.options.enforce_eager,
    )
    if cell.gemm_quant_mode in {"fp8", "fp8_static", "fp8_block"}:
        config["model_config"]["quantization"] = "fp8"
        config["quantization_config"] = {
            "type": "vllm.model_executor.layers.quantization.fp8.Fp8Config",
            "activation_scheme": "static" if cell.gemm_quant_mode == "fp8_static" else "dynamic",
            "weight_block_size": [128, 128] if cell.gemm_quant_mode == "fp8_block" else None,
        }
    elif cell.gemm_quant_mode in {"bfloat16", "half"}:
        config["model_config"].update(dtype=cell.gemm_quant_mode, quantization=None)
        config["quantization_config"] = None
    config["cache_config"].update(
        cache_dtype=cell.kv_cache_dtype,
        gpu_memory_utilization=plan.options.gpu_memory_utilization or 0.9,
        num_gpu_blocks_override=None,
        enable_prefix_caching=cell.workload_kind == "decode" and runner._decode_prefix_caching_mode(cell) == "enabled",
    )
    resources = plan.deployment_profile(cell).resources
    config["scheduler_config"].update(
        max_num_batched_tokens=plan.options.max_num_batched_tokens or resources.max_num_tokens,
        max_num_seqs=plan.options.max_num_seqs or resources.max_batch_size,
    )
    if cell.workload_kind == "prefill":
        config["scheduler_config"]["max_num_batched_tokens"] = plan.options.prefill_sampling.max_total_prefill_tokens
        config["scheduler_config"]["max_num_seqs"] = (
            plan.options.prefill_sampling.max_batch_size or config["scheduler_config"]["max_num_seqs"]
        )
    config["parallel_config"].update(
        enable_expert_parallel=cell.topology.moe_ep > 1,
        enable_eplb=False,
        data_parallel_size_local=cell.topology.dp,
        data_parallel_external_lb=False,
        decode_context_parallel_size=cell.topology.cp,
    )
    config["compilation_config"].update(cudagraph_capture_sizes=[1, 2, 4, 8, 16], max_cudagraph_capture_size=16)
    return config


@pytest.fixture
def campaign(tmp_path, no_models_or_timing_data):  # noqa: F811
    profile = _profile()
    for deployment in profile["deployments"]:
        deployment["backend_version"] = "0.28.0"
    plan = _plan(profile)
    plan = replace(
        plan, options=replace(plan.options, prefill_cudagraph_policy="runtime", max_prefill_cudagraph_size=None)
    )
    coordinates = [
        {"batch_size": batch, "total_prefill_tokens": batch, "total_kv_read_tokens": batch * context}
        for batch in (1, 2, 4, 8, 16, 32)
        for context in (16, 1024)
    ]
    plan = repeatability._subset_plan(
        plan,
        {
            "cell_id": plan.cells[0].cell_id,
            "benchmark_points": {"schema_version": 3, "prefill": coordinates, "decode": []},
        },
    )
    source = tmp_path / "source"
    checkpoint = tmp_path / "source-checkpoint.json"
    _write_campaign(source, checkpoint, plan)
    return plan, source, checkpoint


def _args(campaign, tmp_path, **kwargs):
    plan, source, checkpoint = campaign
    return {
        "source_plan": plan,
        "generator_overrides": {},
        "source_campaign_dir": source,
        "source_checkpoint_path": checkpoint,
        "output_dir": tmp_path / "validation",
        **kwargs,
    }


def _fake_collector(monkeypatch, *, factors=None, fail_at=None, duplicate=False, mutate=None):
    calls = []

    def collect(plan, **kwargs):
        calls.append((plan, kwargs))
        assert kwargs["publish_database"] is False
        assert kwargs["resume"] is False and kwargs["retry_failed"] is False
        index = len(calls)
        root = Path(kwargs["artifact_root"]) / plan.sha256[:16]
        cp = Path(kwargs["checkpoint_dir"]) / "fpm_forward.json"
        _write_campaign(
            root,
            cp,
            plan,
            attempt_id="same" if duplicate else f"independent-{index}",
            factor=(factors or [1.0])[((index - 1) % len(factors or [1.0]))],
            generator_overrides=kwargs["generator_overrides"],
        )
        if mutate:
            mutate(root, index)
        if index == fail_at:
            return [{"classification": "campaign_cell_failed", "error_message": "runtime failed"}]
        return []

    monkeypatch.setattr(repeatability, "run_collection", collect)
    monkeypatch.setattr(repeatability, "_cell_runner", lambda *_args: SimpleNamespace(cleanup=lambda: None))
    return calls


def test_freeze_selects_validated_native_extremes_and_reports_unselected_capture_boundaries(campaign):
    plan, source, checkpoint = campaign
    frozen = repeatability.freeze_repeatability_plan(plan, source, checkpoint, max_points_per_cell=6)
    selected = frozen["cells"][0]
    assert len(selected["points"]) == 6 < selected["source_point_count"]
    assert {point["coordinates"]["batch_size"] for point in selected["points"]} >= {1, 32}
    assert {
        point["coordinates"]["total_kv_read_tokens"] / point["coordinates"]["batch_size"]
        for point in selected["points"]
    } == {16, 1024}
    assert selected["selection_coverage"]["unselected_capture_boundaries"] == [2, 4, 8]
    assert selected["execution"]["status"] == "qualified"
    assert selected["execution"]["per_point_dispatch"] == "unreported"
    validate_saved_plan(repeatability._subset_plan(plan, selected).to_dict())
    assert frozen == repeatability.freeze_repeatability_plan(plan, source, checkpoint, max_points_per_cell=6)


def test_too_small_budget_cannot_silently_drop_required_regimes(campaign):
    with pytest.raises(ValueError, match="cannot cover.*uncovered"):
        repeatability.freeze_repeatability_plan(*campaign, max_points_per_cell=1)


def _add_zero_kv_duplicates(root, *, seed_samples=3):
    for path in root.glob("cells/*/raw/*/benchmark-*.json"):
        payload = json.loads(path.read_text())
        ordinary_row = payload["results"][0]
        ordinary_group = payload["iteration_groups"][0]
        assert ordinary_row["point"]["total_kv_read_tokens"] == 0
        assert ordinary_row["kv_seed_regime"] == "not_applicable"
        for index in range(seed_samples):
            group = copy.deepcopy(ordinary_group)
            point = group["point"]
            point["benchmark_id"] = index + 2
            point["sample_reasons"].append("prefill_real_seed")
            group["benchmark_id"] = point["benchmark_id"]
            for result in group["rank_results"]:
                fpm = result["fpms"][0]
                fpm["counter_id"] = point["benchmark_id"]
                fpm["wall_time"] *= 0.5
            group["wall_time"] *= 0.5
            payload["iteration_groups"].append(group)
            payload["results"].append(
                {
                    "point": point,
                    "kv_seed_regime": "real_prefix",
                    "fpms": group["rank_results"][payload["dp"]["rank"]]["fpms"],
                }
            )
        count = len(payload["results"])
        payload["coverage"].update(expected_points=count, completed_points=count)
        payload["timing"]["measured_iteration_seconds"] = sum(
            group["wall_time"] for group in payload["iteration_groups"]
        )
        path.write_text(json.dumps(payload))


@pytest.fixture
def zero_kv_campaign(campaign):
    plan, source, checkpoint = campaign
    plan = repeatability._subset_plan(
        plan,
        {
            "cell_id": plan.cells[0].cell_id,
            "benchmark_points": {
                "schema_version": 3,
                "prefill": [{"batch_size": 1, "total_prefill_tokens": 1, "total_kv_read_tokens": 0}],
                "decode": [],
            },
        },
    )
    _write_campaign(source, checkpoint, plan)
    return plan, source, checkpoint


@pytest.mark.parametrize("seed_samples", [3, 4])
def test_freeze_zero_kv_duplicates_uses_published_ordinary_sample(zero_kv_campaign, seed_samples):
    plan, source, checkpoint = zero_kv_campaign
    _add_zero_kv_duplicates(source, seed_samples=seed_samples)
    original = runner._file_manifest(source)
    frozen = repeatability.freeze_repeatability_plan(plan, source, checkpoint)
    cell = frozen["cells"][0]
    assert cell["source_point_count"] == seed_samples + 1
    assert len(cell["points"]) == 1
    point = cell["points"][0]
    assert point["source_point"]["benchmark_id"] == 1
    assert point["source_point"]["sample_reasons"] == []
    assert point["kv_seed_regime"] == "not_applicable"
    rows = aggregate_cell(plan, plan.cells[0], source / "cells" / cell["cell_id"], expected_attempt_id="source")
    assert len(rows) == 1
    assert point["source_wall_time_seconds"] * 1000 == rows[0]["latency_ms"]
    assert point["source_wall_time_seconds"] == max(value for _, value in point["source_rank_wall_times"])
    assert runner._file_manifest(source) == original


@pytest.mark.parametrize(
    "conflict",
    [
        "duplicate_ordinary",
        "no_ordinary",
        "missing_provenance",
        "positive_kv",
        "expected_cudagraph_mode",
        "expected_capture_size",
        "padding_tokens",
        "partition",
        "rows",
        "sample_reasons",
        "context_clamped",
    ],
)
def test_freeze_rejects_conflicting_zero_kv_duplicates(zero_kv_campaign, conflict):
    plan, source, checkpoint = zero_kv_campaign
    _add_zero_kv_duplicates(source)
    for path in source.glob("cells/*/raw/*/benchmark-*.json"):
        payload = json.loads(path.read_text())
        for index, (row, group) in enumerate(zip(payload["results"], payload["iteration_groups"], strict=True)):
            point = row["point"]
            if conflict == "positive_kv":
                point["total_kv_read_tokens"] = 128
                for result in group["rank_results"]:
                    result["fpms"][0]["scheduled_requests"]["sum_prefill_kv_tokens"] = 128
                row["fpms"] = group["rank_results"][payload["dp"]["rank"]]["fpms"]
            elif conflict == "no_ordinary":
                row["kv_seed_regime"] = "real_prefix"
                point["sample_reasons"] = ["prefill_real_seed"]
            elif index == 1:
                if conflict == "duplicate_ordinary":
                    row["kv_seed_regime"] = "not_applicable"
                    point["sample_reasons"] = []
                elif conflict == "missing_provenance":
                    row.pop("kv_seed_regime")
                elif conflict == "context_clamped":
                    point["sample_reasons"].append("context_clamped")
                else:
                    point[conflict] = {
                        "expected_cudagraph_mode": "NONE",
                        "expected_capture_size": 2,
                        "padding_tokens": 1,
                        "partition": "different",
                        "rows": [{"tokens": 1}],
                        "sample_reasons": ["eager_tail", "prefill_real_seed"],
                    }[conflict]
            group["point"] = point
        path.write_text(json.dumps(payload))
    original = runner._file_manifest(source)
    with pytest.raises(ValueError, match="unclamped samples share one key"):
        repeatability.freeze_repeatability_plan(plan, source, checkpoint)
    assert runner._file_manifest(source) == original


def test_zero_kv_source_consolidation_preserves_independent_fresh_samples(zero_kv_campaign, tmp_path, monkeypatch):
    _add_zero_kv_duplicates(zero_kv_campaign[1])
    original = runner._file_manifest(zero_kv_campaign[1])
    calls = _fake_collector(monkeypatch, factors=[0.99, 1.0, 1.01, 1.0, 1.0])
    args = _args(zero_kv_campaign, tmp_path)
    report = repeatability.run_repeatability(**args)
    assert report["status"] == "passed" and len(calls) == 5
    assert len(report["points"]) == 1
    point = report["points"][0]
    assert point["sample_count"] == 5
    assert point["source_inclusive_sample_count"] == 6
    assert point["samples_seconds"] == pytest.approx(
        [point["source_wall_time_seconds"] * factor for factor in (0.99, 1.0, 1.01, 1.0, 1.0)]
    )
    assert runner._file_manifest(zero_kv_campaign[1]) == original
    assert repeatability.run_repeatability(**args, resume=True) == report


def test_zero_kv_duplicates_in_fresh_repeat_are_not_counted_as_independent_samples(
    zero_kv_campaign, tmp_path, monkeypatch
):
    _add_zero_kv_duplicates(zero_kv_campaign[1])
    calls = _fake_collector(monkeypatch, mutate=lambda root, _index: _add_zero_kv_duplicates(root))
    report = repeatability.run_repeatability(**_args(zero_kv_campaign, tmp_path))
    assert report["status"] == "failed" and len(calls) == 1
    assert report["points"][0]["sample_count"] == 0
    failed = report["samples"][zero_kv_campaign[0].cells[0].cell_id][0]["attempts"][-1]
    assert "did not measure exactly the frozen subset" in failed["error"]


def test_five_independent_samples_keep_raw_data_and_pass_without_formal_publication(campaign, tmp_path, monkeypatch):
    calls = _fake_collector(monkeypatch, factors=[0.99, 1.0, 1.01, 1.0, 1.0])
    originals = runner._file_manifest(campaign[1])
    args = _args(campaign, tmp_path, max_points_per_cell=6)
    report = repeatability.run_repeatability(**args)
    assert report["status"] == "passed" and len(calls) == 5
    assert all(len(json.loads(plan.options.benchmark_points_json)["prefill"]) == 6 for plan, _ in calls)
    assert len({kwargs["artifact_root"] for _, kwargs in calls}) == 5
    assert runner._file_manifest(campaign[1]) == originals
    assert all(point["sample_count"] == 5 and len(point["samples_seconds"]) == 5 for point in report["points"])
    values = report["points"][0]["samples_seconds"]
    assert report["points"][0]["sample_cv"] == statistics.stdev(values) / statistics.mean(values)
    assert report["points"][0]["sample_stddev_seconds"] == statistics.stdev(values)
    assert report["repeatability"]["standard_deviation_denominator"] == "n - 1"
    assert report["repeatability"]["time_unit"] == "seconds"
    assert repeatability.run_repeatability(**args, resume=True) == report
    assert len(calls) == 5


def _mutate_workers(root, action):
    for path in root.glob("cells/*/raw/*/fpm-execution-worker-*.json"):
        payload = json.loads(path.read_text())
        action(payload)
        path.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("model_config", "model", "other/checkpoint"),
        ("model_config", "revision", "different-revision"),
        ("model_config", "dtype", "torch.float32"),
        ("model_config", "quantization", None),
        ("model_config", "max_model_len", 4096),
        ("model_config", "enforce_eager", True),
        ("cache_config", "cache_dtype", "half"),
        ("parallel_config", "tensor_parallel_size", 999),
        ("scheduler_config", "max_num_batched_tokens", 4096),
    ],
)
def test_identically_wrong_source_and_repeats_fail(campaign, tmp_path, monkeypatch, section, field, value):
    def alter(payload):
        payload["resolved_config"][section][field] = value

    _mutate_workers(campaign[1], alter)
    calls = _fake_collector(monkeypatch, mutate=lambda root, _index: _mutate_workers(root, alter))
    report = repeatability.run_repeatability(**_args(campaign, tmp_path))
    assert report["status"] == "failed"
    assert report["execution"]["status"] == "failed"
    assert calls == []


@pytest.mark.parametrize("missing", ["all", "model", "revision", "quantization", "scheduler", "graph"])
def test_required_runtime_configuration_missing_is_incomplete(campaign, tmp_path, monkeypatch, missing):
    def alter(payload):
        config = payload["resolved_config"]
        if missing == "all":
            payload["resolved_config"] = None
        elif missing == "scheduler":
            del config["scheduler_config"]
        elif missing == "graph":
            del payload["graph_config"]["mode"]
        else:
            del config["model_config"][missing]

    _mutate_workers(campaign[1], alter)
    _fake_collector(monkeypatch, mutate=lambda root, _index: _mutate_workers(root, alter))
    report = repeatability.run_repeatability(**_args(campaign, tmp_path))
    assert report["status"] == "incomplete"


@pytest.mark.parametrize("change", ["duplicate", "mode", "sizes", "maximum", "compile", "eager"])
def test_invalid_initialized_graph_fails(campaign, tmp_path, monkeypatch, change):
    def alter(payload):
        graph = payload["graph_config"]
        if change == "duplicate":
            graph.update(cudagraph_mode="NONE", cudagraph_capture_sizes=[], max_cudagraph_capture_size=0)
            return
        if change == "mode":
            graph["cudagraph_mode"] = "MADE_UP"
        elif change == "sizes":
            graph["cudagraph_capture_sizes"] = [1, 2, 2, 4, 8, 16]
        elif change == "maximum":
            graph["max_cudagraph_capture_size"] = 17
        elif change == "compile":
            graph["mode"] = False
        else:
            payload["resolved_config"]["model_config"]["enforce_eager"] = True
        payload["resolved_config"]["compilation_config"] = dict(graph)

    _mutate_workers(campaign[1], alter)
    _fake_collector(monkeypatch, mutate=lambda root, _index: _mutate_workers(root, alter))
    assert repeatability.run_repeatability(**_args(campaign, tmp_path))["status"] == "failed"


def test_worker_graph_downgrade_preserves_scheduler_expectations(campaign, tmp_path, monkeypatch):
    def alter(payload):
        payload["graph_config"]["cudagraph_mode"] = "NONE"
        payload["resolved_config"]["compilation_config"]["cudagraph_mode"] = "NONE"

    _mutate_workers(campaign[1], alter)
    _fake_collector(monkeypatch, mutate=lambda root, _index: _mutate_workers(root, alter))
    report = repeatability.run_repeatability(**_args(campaign, tmp_path))
    assert report["status"] == "passed"


@pytest.mark.parametrize("evidence", ["correct", "missing", "wrong"])
def test_explicit_local_checkpoint_uses_loaded_revision_evidence(campaign, tmp_path, monkeypatch, evidence):
    plan, source, checkpoint = campaign
    profile = json.loads(plan._fpm_profile_json)
    profile["model_revision"] = "a" * 40
    deployment = {"K8sConfig": {"k8s_pvc_mount_path": "/models", "k8s_model_path_in_pvc": "checkpoint"}}
    plan = replace(
        plan,
        _fpm_profile_json=json.dumps(profile),
        generator_config_sha256=repeatability._canonical_hash(with_kv_warmup_defaults(deployment)),
    )
    plan = repeatability._subset_plan(
        plan, {"cell_id": plan.cells[0].cell_id, "benchmark_points": json.loads(plan.options.benchmark_points_json)}
    )
    _write_campaign(source, checkpoint, plan, generator_overrides=deployment)

    def alter(payload):
        model = payload["resolved_config"]["model_config"]
        assert model["model"] == "/models/checkpoint"
        model["revision"] = None
        if evidence != "missing":
            model["loaded_config_commit_hash"] = ("a" if evidence == "correct" else "b") * 40

    _mutate_workers(source, alter)
    calls = _fake_collector(monkeypatch, mutate=lambda root, _index: _mutate_workers(root, alter))
    result = repeatability.run_repeatability(
        **_args((plan, source, checkpoint), tmp_path, generator_overrides=deployment)
    )
    assert result["status"] == {"correct": "passed", "missing": "incomplete", "wrong": "failed"}[evidence]
    if evidence == "wrong":
        assert calls == []


@pytest.mark.parametrize("observed", [128, 256])
def test_explicit_cache_sizing_is_checked_against_launch(campaign, observed):
    plan, source, checkpoint = campaign
    cell = plan.cells[0]
    policy = repeatability.BackendPolicy(
        "explicit-cache", {"params": {"agg": {"extra_cli_args": ["--num-gpu-blocks-override", "128"]}}}, {}
    )
    plan = replace(plan, cells=(replace(cell, backend_policy=policy),), backend_policies=(policy,))
    plan = repeatability._subset_plan(
        plan, {"cell_id": cell.cell_id, "benchmark_points": json.loads(plan.options.benchmark_points_json)}
    )
    _write_campaign(source, checkpoint, plan)
    _mutate_workers(
        source, lambda payload: payload["resolved_config"]["cache_config"].update(num_gpu_blocks_override=observed)
    )
    result = repeatability.freeze_repeatability_plan(plan, source, checkpoint)
    assert result["cells"][0]["execution"]["status"] == ("qualified" if observed == 128 else "failed")


@pytest.mark.parametrize("last", [True, False])
@pytest.mark.parametrize("matches", [True, False])
def test_execution_only_boolean_overrides_use_last_flag(campaign, last, matches):
    plan, source, checkpoint = campaign
    positive, negative = "--async-scheduling", "--no-async-scheduling"
    args = [positive, negative, positive] if last else [positive, negative]
    policy = repeatability.BackendPolicy("boolean-override", {"params": {"agg": {"extra_cli_args": args}}}, {})
    cell = replace(plan.cells[0], backend_policy=policy)
    plan = replace(plan, cells=(cell,), backend_policies=(policy,))
    plan = repeatability._subset_plan(
        plan, {"cell_id": cell.cell_id, "benchmark_points": json.loads(plan.options.benchmark_points_json)}
    )
    assert not runner._observe_runtime_memory(plan, cell)
    _write_campaign(source, checkpoint, plan)
    _mutate_workers(
        source,
        lambda payload: payload["resolved_config"]["scheduler_config"].update(
            async_scheduling=last if matches else not last
        ),
    )
    result = repeatability.freeze_repeatability_plan(plan, source, checkpoint)
    assert result["cells"][0]["execution"]["status"] == ("qualified" if matches else "failed")


def test_unrequested_explicit_cache_sizing_fails(campaign):
    _mutate_workers(
        campaign[1], lambda payload: payload["resolved_config"]["cache_config"].update(kv_cache_memory_bytes=1024)
    )
    result = repeatability.freeze_repeatability_plan(*campaign)
    assert result["cells"][0]["execution"]["status"] == "failed"


@pytest.mark.parametrize("missing", ["checkpoint", "native_rank", "worker", "launch"])
def test_missing_sample_artifacts_stay_incomplete(campaign, tmp_path, monkeypatch, missing):
    def mutate(root, _index):
        if missing == "checkpoint":
            (root.parent.parent / "checkpoint" / "fpm_forward.json").unlink()
        elif missing == "native_rank":
            next(root.glob("cells/*/raw/*/benchmark-dp*.json")).unlink()
        elif missing == "worker":
            next(root.glob("cells/*/raw/*/fpm-execution-worker*.json")).unlink()
        else:
            next(root.glob("cells/*/run.sh")).unlink()

    _fake_collector(monkeypatch, mutate=mutate)
    report = repeatability.run_repeatability(**_args(campaign, tmp_path))
    assert report["status"] == "incomplete"


def test_backend_mismatch_fails_until_authorized_retry_and_preserves_history(campaign, tmp_path, monkeypatch):
    def mutate(root, index):
        if index == 1:
            _mutate_workers(root, lambda payload: payload["attention_groups"][0].update(backend_class="other.Backend"))

    _fake_collector(monkeypatch, mutate=mutate)
    args = _args(campaign, tmp_path)
    report = repeatability.run_repeatability(**args)
    assert report["status"] == "failed"
    assert all(point["sample_stddev_seconds"] is None for point in report["points"])
    report = repeatability.run_repeatability(**args, resume=True, retry_failed=True)
    assert report["status"] == "passed"
    attempts = report["samples"][campaign[0].cells[0].cell_id][0]["attempts"]
    assert [attempt["status"] for attempt in attempts] == ["failed", "passed"]
    assert attempts[0]["failure_kind"] == "validation_failed"


@pytest.mark.parametrize(
    "flag,value", [("--kv-cache-memory-bytes", "1073741824"), ("--num-gpu-blocks-override", "128")]
)
def test_execution_observer_keeps_explicit_declared_cache_settings(campaign, flag, value):
    cell = campaign[0].cells[0]
    policy = repeatability.BackendPolicy("explicit", {"params": {"agg": {"extra_cli_args": [flag, value]}}}, {})
    cell = replace(cell, backend_policy=policy)
    assert not runner._observe_runtime_memory(campaign[0], cell)
    args = runner._cell_generator_overrides(campaign[0], cell, {})["params"]["agg"]["extra_cli_args"]
    assert args[args.index(flag) + 1] == value


def test_execution_observer_owns_worker_class(campaign):
    cell = campaign[0].cells[0]
    policy = repeatability.BackendPolicy(
        "explicit", {"params": {"agg": {"extra_cli_args": ["--worker-cls=Other"]}}}, {}
    )
    with pytest.raises(ValueError, match="observer"):
        runner._cell_generator_overrides(campaign[0], replace(cell, backend_policy=policy), {})


def test_unstable_points_fail_and_can_be_reassessed_without_new_launches(campaign, tmp_path, monkeypatch):
    calls = _fake_collector(monkeypatch, factors=[0.5, 1.5])
    args = _args(campaign, tmp_path)
    report = repeatability.run_repeatability(**args)
    assert report["status"] == "failed" and report["repeatability"]["status"] == "unstable"
    frozen = json.loads((args["output_dir"] / repeatability.PLAN_FILENAME).read_text())
    reassessed = repeatability.assess_repeatability(frozen, report, cv_threshold=0.75)
    assert reassessed["status"] == "passed" and len(calls) == 5
    assert report["status"] == "failed"
    assert reassessed["assessment"]["cv_threshold"] == 0.75


def test_stable_new_samples_do_not_accept_an_outlying_published_measurement(campaign, tmp_path, monkeypatch):
    plan, source, checkpoint = campaign
    _write_campaign(source, checkpoint, plan, factor=10.0)
    _fake_collector(monkeypatch)
    originals = runner._file_manifest(source)
    report = repeatability.run_repeatability(**_args(campaign, tmp_path))
    assert report["status"] == "failed"
    assert all(point["sample_cv"] == 0 and point["source_inclusive_cv"] > 0.05 for point in report["points"])
    assert all(point["source_inclusive_sample_count"] == 6 for point in report["points"])
    assert runner._file_manifest(source) == originals


def test_failure_resume_preserves_failed_attempt_and_uses_new_directory(campaign, tmp_path, monkeypatch):
    calls = _fake_collector(monkeypatch, fail_at=2)
    args = _args(campaign, tmp_path)
    report = repeatability.run_repeatability(**args)
    assert report["status"] == "incomplete" and len(calls) == 2
    assert all(point["sample_count"] == 1 and point["sample_stddev_seconds"] is None for point in report["points"])
    failed_raw = Path(calls[1][1]["artifact_root"])
    saved_failed = runner._file_manifest(failed_raw)
    assert repeatability.run_repeatability(**args, resume=True) == report
    assert len(calls) == 2
    report = repeatability.run_repeatability(**args, resume=True, retry_failed=True)
    assert report["status"] == "passed" and len(calls) == 6
    entries = report["samples"][campaign[0].cells[0].cell_id]
    assert [item["status"] for item in entries[1]["attempts"]] == ["failed", "passed"]
    assert runner._file_manifest(failed_raw) == saved_failed


@pytest.mark.parametrize("mutation", ["source", "sample", "policy", "deployment"])
def test_resume_rejects_mutated_evidence_without_recollection(campaign, tmp_path, monkeypatch, mutation):
    calls = _fake_collector(monkeypatch)
    args = _args(campaign, tmp_path)
    repeatability.run_repeatability(**args)
    if mutation == "source":
        path = next(campaign[1].glob("cells/*/raw/*/benchmark-*.json"))
        path.write_text(path.read_text() + " ")
    elif mutation == "sample":
        path = next(args["output_dir"].glob("samples/**/raw/*/benchmark-*.json"))
        path.write_text(path.read_text() + " ")
    elif mutation == "policy":
        args["samples"] = 6
    else:
        args["generator_overrides"] = {"K8sConfig": {"k8s_image": "different"}}
    with pytest.raises(ValueError, match="changed|differ"):
        repeatability.run_repeatability(**args, resume=True)
    assert len(calls) == 5


def test_reused_attempt_is_not_an_independent_sample(campaign, tmp_path, monkeypatch):
    calls = _fake_collector(monkeypatch, duplicate=True)
    report = repeatability.run_repeatability(**_args(campaign, tmp_path))
    assert report["status"] == "failed" and len(calls) == 2
    entries = report["samples"][campaign[0].cells[0].cell_id]
    assert "independent launch" in entries[1]["attempts"][-1]["error"]


@pytest.mark.parametrize("change", ["backend", "graph", "seed", "coordinate"])
def test_repeat_changed_runtime_regime_never_enters_repeatability_statistics(campaign, tmp_path, monkeypatch, change):
    def mutate(root, _index):
        if change in {"backend", "graph"}:
            for path in root.glob("cells/*/raw/*/fpm-execution*.json"):
                value = json.loads(path.read_text())
                if change == "backend":
                    value["attention_groups"][0]["backend_class"] = "another.Backend"
                else:
                    value["graph_config"]["cudagraph_mode"] = "NONE"
                path.write_text(json.dumps(value))
        else:
            for path in root.glob("cells/*/raw/*/benchmark-*.json"):
                value = json.loads(path.read_text())
                if change == "seed":
                    value["results"][0]["kv_seed_regime"] = "fake_prefix"
                    value["results"][0]["point"]["sample_reasons"] = ["prefill_fake_prefix"]
                    value["iteration_groups"][0]["point"]["sample_reasons"] = ["prefill_fake_prefix"]
                else:
                    value["results"][0]["point"]["batch_size"] += 1
                path.write_text(json.dumps(value))

    calls = _fake_collector(monkeypatch, mutate=mutate)
    report = repeatability.run_repeatability(**_args(campaign, tmp_path))
    assert len(calls) == 1 and report["status"] == "failed"
    assert all(point["sample_count"] == 0 for point in report["points"])


def test_old_artifacts_without_execution_evidence_remain_incomplete(campaign):
    plan, source, checkpoint = campaign
    for path in source.glob("cells/*/raw/*/fpm-execution*.json"):
        path.unlink()
    frozen = repeatability.freeze_repeatability_plan(plan, source, checkpoint)
    evidence = frozen["cells"][0]["execution"]
    assert evidence["status"] == "incomplete"
    assert "missing" in evidence["missing_evidence"][0]


def test_execution_observer_preserves_mixed_selected_backends_and_graph_config(tmp_path, monkeypatch):
    monkeypatch.setattr(observer, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(observer.importlib.metadata, "version", lambda _name: "0.28.0")
    (tmp_path / "collector-provenance.json").write_text("{}")
    backend_a = type("BackendA", (), {"__module__": "actual.runtime"})
    backend_b = type("BackendB", (), {"__module__": "actual.runtime"})
    worker = SimpleNamespace(
        vllm_config=_vllm_config(),
        model_runner=SimpleNamespace(
            attn_groups=[
                [
                    SimpleNamespace(backend=backend_a, layer_names=["layer.0"], kv_cache_group_id=0),
                    SimpleNamespace(backend=backend_b, layer_names=["layer.1"], kv_cache_group_id=0),
                ]
            ]
        ),
    )
    worker.vllm_config.model_config.hf_config = SimpleNamespace(_commit_hash="a" * 40)
    del worker.vllm_config.offload_config
    observer.observe_execution(worker, dp_rank=0, tp_rank=0, pp_rank=0)
    result = json.loads((tmp_path / "fpm-execution-worker-dp0-tp0-pp0.json").read_text())
    assert result["status"] == "observed"
    assert [item["backend_class"] for item in result["attention_groups"]] == [
        "actual.runtime.BackendA",
        "actual.runtime.BackendB",
    ]
    assert result["graph_config"]["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert result["resolved_config"]["compilation_config"] == result["graph_config"]
    assert result["resolved_config"]["model_config"]["loaded_config_commit_hash"] == "a" * 40
    assert result["resolved_config"]["quantization_config"]["quant_method"] == "NVFP4"
    assert "offload_config" not in result["resolved_config"]
    assert result["per_point_dispatch"] == "unreported"
    assert not list(tmp_path.glob("fpm-memory-*.json"))
    with pytest.raises(RuntimeError, match="duplicate"):
        observer.observe_execution(worker, dp_rank=0, tp_rank=0, pp_rank=0)


@pytest.mark.parametrize("version", ["0.27.0", "0.28.0", "0.29.0"])
@pytest.mark.usefixtures("no_models_or_timing_data")
def test_execution_only_observation_does_not_enable_memory_conversion(version):
    profile = _profile()
    for deployment in profile["deployments"]:
        deployment["backend_version"] = version
    plan = _plan(profile)
    for cell in plan.cells:
        args = runner._cell_generator_overrides(plan, cell, {})["params"]["agg"]["extra_cli_args"]
        assert runner._observe_runtime_memory(plan, cell) is False
        if version in {"0.27.0", "0.28.0"}:
            assert args[args.index("--worker-cls") + 1] == "fpm_memory_worker.FpmExecutionWorker"
            assert "--scheduler-cls" not in args
        else:
            assert "--worker-cls" not in args


def test_cli_exposes_bounded_repeatability_defaults():
    args = cli._parser().parse_args(["--gpu", "gb300"])
    assert (args.repeatability_samples, args.repeatability_max_points, args.repeatability_cv_threshold) == (5, 12, 0.05)


def test_saved_loader_roundtrips_without_model_or_hardware_resolution(campaign):
    plan, source, _checkpoint = campaign
    assert repeatability.load_repeatability_source(source).to_dict() == plan.to_dict()
    assert repeatability.load_repeatability_deployment(source) == with_kv_warmup_defaults({})


def test_cli_preview_uses_saved_plan_without_model_or_gpu_arguments(campaign, tmp_path, monkeypatch, capsys):
    plan, source, checkpoint = campaign
    monkeypatch.setattr(cli, "build_collection_case_plan", lambda **kwargs: pytest.fail("re-resolved source plan"))
    assert (
        cli.main(
            [
                "--repeatability-source-campaign",
                str(source),
                "--repeatability-source-checkpoint",
                str(checkpoint),
                "--repeatability-output-dir",
                str(tmp_path / "validation"),
                "--plan-only",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["source_plan_sha256"] == plan.sha256
    assert not (tmp_path / "validation").exists()


def test_new_code_cannot_silently_remeasure_old_campaign(campaign, tmp_path, monkeypatch):
    calls = _fake_collector(monkeypatch)
    monkeypatch.setattr(repeatability.planner, "_git_revision", lambda: "different-code")
    with pytest.raises(ValueError, match="checkout/code.*unqualified"):
        repeatability.run_repeatability(**_args(campaign, tmp_path))
    assert calls == []


def test_interrupted_sample_is_checkpointed_without_deleting_evidence(campaign, tmp_path, monkeypatch):
    def interrupted(plan, **kwargs):
        root = Path(kwargs["artifact_root"])
        root.mkdir(parents=True)
        (root / "runtime-failure.log").write_text("partial native output")
        raise KeyboardInterrupt

    monkeypatch.setattr(repeatability, "run_collection", interrupted)
    args = _args(campaign, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        repeatability.run_repeatability(**args)
    report = json.loads((args["output_dir"] / repeatability.REPORT_FILENAME).read_text())
    sample = report["samples"][campaign[0].cells[0].cell_id][0]
    assert sample["attempts"][0]["status"] == "interrupted"
    assert next(args["output_dir"].rglob("runtime-failure.log")).read_text() == "partial native output"


def test_receipt_mismatch_is_a_failed_sample(campaign, tmp_path, monkeypatch):
    def mutate(root, _index):
        path = next(root.glob(f"cells/*/raw/*/{runner.POINTS_RECEIPT_FILENAME}"))
        value = json.loads(path.read_text())
        value["sha256"] = "wrong"
        path.write_text(json.dumps(value))

    _fake_collector(monkeypatch, mutate=mutate)
    report = repeatability.run_repeatability(**_args(campaign, tmp_path))
    sample = report["samples"][campaign[0].cells[0].cell_id][0]
    assert sample["attempts"][0]["status"] == "failed"
    assert "receipt mismatch" in sample["attempts"][0]["error"]


def test_failed_source_point_is_never_selected(campaign):
    plan, source, checkpoint = campaign
    path = next(source.glob("cells/*/raw/*/benchmark-*.json"))
    value = json.loads(path.read_text())
    value["timing_valid"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="invalid native terminal"):
        repeatability.freeze_repeatability_plan(plan, source, checkpoint)
