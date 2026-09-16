# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.metadata
import json
import sys
from dataclasses import replace

import pytest
import yaml
from collector.fpm_forward import capabilities, cli, memory_admission, planner, runner
from collector.fpm_forward.config import FPMCollectionOptions, reject_fpm_arguments_without_fpm
from collector.fpm_forward.model_capability import load_model_config

pytestmark = pytest.mark.unit


def _profile():
    """Real GLM deployment identities with explicitly declared test-only bounds."""
    return {
        "schema_version": 1,
        "model": "nvidia/GLM-5.2-NVFP4",
        "model_revision": "collector-test-snapshot",
        "architecture": "GlmMoeDsaForCausalLM",
        "context_length": 8192,
        "num_experts": 256,
        "provenance": "Collector test declaration; not a measured memory qualification.",
        "deployments": [
            {
                "system": "b200_sxm",
                "backend": "vllm",
                "backend_version": "0.25.1",
                "tp": tp,
                "dp": dp,
                "moe_tp": 1,
                "moe_ep": 8,
                "gemm_quant_mode": "nvfp4",
                "moe_quant_mode": "nvfp4",
                "fmha_quant_mode": "fp8",
                "comm_quant_mode": "half",
                "kv_cache_dtype": "fp8",
                "resources": {
                    "weights_bytes": 100_000_000_000,
                    "activations_bytes": 1_000_000_000,
                    "runtime_overhead_bytes": 1_000_000_000,
                    "comm_overhead_bytes": 1_000_000_000,
                    "kv_bytes_per_token": 60_216,
                    "cache_layout": "linear",
                    "max_num_tokens": 8192,
                    "max_batch_size": 1024,
                    "provenance": "Independent test envelope: 100 GB weights plus three 1 GB overhead bounds.",
                },
            }
            for tp, dp in ((1, 8), (8, 1))
        ],
    }


def _argv(profile):
    return [
        "--model-path",
        profile["model"],
        "--gpu",
        "b200_sxm",
        "--fpm-max-gpus",
        "8",
        "--fpm-gpu-counts",
        "8",
        "--fpm-parallel-presets",
        "dep",
        "tep",
        "--fpm-kv-cache-dtypes",
        "fp8",
    ]


def _plan(profile, **overrides):
    args = cli._parser().parse_args(_argv(profile))
    kwargs = {
        "backend": "vllm",
        "model_path": profile["model"],
        "system": "b200_sxm",
        "selected_ops": {"dsa_context_module", "dsa_generation_module"},
        "options": FPMCollectionOptions.from_args(args),
        "fpm_profile": profile,
    }
    kwargs.update(overrides)
    return planner.build_collection_plan(**kwargs)


@pytest.fixture
def no_models_or_timing_data(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("profile collection constructed a model or consulted timing data")

    from aiconfigurator_core.sdk import engine, memory, models, perf_database
    from aiconfigurator_core.sdk.models import base

    monkeypatch.setattr(models, "_MODEL_REGISTRY", {})
    monkeypatch.setattr(base, "_MODEL_REGISTRY", {})
    for module in (models, memory, engine):
        monkeypatch.setattr(module, "get_model", forbidden)
    monkeypatch.setattr(memory_admission.KVCacheEstimator, "from_request", forbidden)
    for module in (capabilities, perf_database):
        monkeypatch.setattr(module, "get_database", forbidden)
        monkeypatch.setattr(module, "get_latest_database_version", forbidden)
    monkeypatch.setattr(capabilities, "context_fmha_supported_modes", forbidden)
    monkeypatch.setattr(planner, "_git_revision", lambda: "profile-collection-test")


@pytest.mark.parametrize("file_format", ["json", "yaml"])
def test_glm_cli_plans_without_model_classes_or_any_timing_data(
    tmp_path, monkeypatch, capsys, no_models_or_timing_data, file_format
):
    profile = _profile()
    path = tmp_path / f"profile.{file_format}"
    path.write_text(json.dumps(profile) if file_format == "json" else yaml.safe_dump(profile))
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "")

    assert cli.main([*_argv(profile), "--fpm-model-profile", str(path), "--plan-only"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["counts"]["cells"] == 4
    assert {cell["cell_id"] for cell in output["cells"]} == {
        "fpm-6efe9f97155a97e4",
        "fpm-093958a8e9575a8a",
        "fpm-272431d375ba264f",
        "fpm-8ccaea64a44c2882",
    }
    assert {cell["parallel_strategy"] for cell in output["cells"]} == {"dep", "tep"}
    assert {cell["resolved_dtypes"]["fmha_quant_mode"] for cell in output["cells"]} == {"fp8"}
    assert {cell["resolved_dtypes"]["fmha_resolution"] for cell in output["cells"]} == {"checkpoint_native"}
    assert output["fpm_profile"]["model_revision"] == "collector-test-snapshot"
    assert {decision["source"] for decision in output["topology_memory_admission"]} == {"fpm_profile_declared"}
    for decision in output["topology_memory_admission"]:
        assert decision["estimates"][0]["estimated_non_kv_bytes"] == 103_000_000_000
        assert decision["estimates"][0]["provenance"] == profile["deployments"][0]["resources"]["provenance"]
        assert decision["activation_envelope"]["max_batch_size"] == 1024
        assert decision["activation_envelope"]["scope"] == "rank_local"


def test_unknown_architecture_plans_and_renders_from_real_config(tmp_path, no_models_or_timing_data):
    profile = _profile()
    config = load_model_config(profile["model"]).payload
    config["architectures"] = ["UnregisteredMoeForCausalLM"]
    profile.update(model="private-org/new-model", architecture=config["architectures"][0])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    plan = _plan(profile, model_config_path=str(path), has_model_cases=False)
    assert plan.capability.support_level == "bootstrap_template"
    cell_dir = tmp_path / "rendered"
    cell_dir.mkdir()
    runner._render_cell(plan, plan.cells[0], cell_dir, {})

    script = (cell_dir / "run.sh").read_text()
    assert "--model private-org/new-model" in script
    assert "--max-num-batched-tokens 8192" in script
    assert "--max-num-seqs 1024" in script


def test_frozen_profile_content_invalidates_resume_without_changing_cell_ids(tmp_path, no_models_or_timing_data):
    source = _profile()
    first = _plan(source)
    original = first.to_dict()
    source["model_revision"] = "different-test-snapshot"
    second = _plan(source)
    detached = first.fpm_profile
    detached.deployments[0].resources.weights_bytes += 10

    assert first.to_dict() == original
    assert first.sha256 != second.sha256
    assert [cell.cell_id for cell in first.cells] == [cell.cell_id for cell in second.cells]
    assert _plan(original["fpm_profile"]).sha256 == first.sha256
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"schema": runner.CHECKPOINT_SCHEMA, "plan_sha256": first.sha256, "cells": {}}))
    with pytest.raises(ValueError, match="checkpoint does not match the current frozen plan"):
        runner._load_checkpoint(checkpoint, second, resume=True)


@pytest.mark.parametrize("field,value", [("fmha_quant_mode", "bfloat16"), ("moe_backend", "flashinfer_cutlass")])
def test_profile_cannot_override_resolved_precision_or_backend(field, value, no_models_or_timing_data):
    profile = _profile()
    profile["deployments"][0][field] = value
    with pytest.raises(ValueError, match="profile identity mismatch"):
        _plan(profile)


def test_historical_minimax_transfer_precision_cannot_relabel_new_collection(no_models_or_timing_data):
    profile = _profile()
    profile.update(model="MiniMaxAI/MiniMax-M2.7", architecture="MiniMaxM2ForCausalLM")
    deployment = profile["deployments"][0]
    deployment.update(
        system="h200_sxm",
        tp=4,
        dp=1,
        moe_tp=4,
        moe_ep=1,
        gemm_quant_mode="fp8_block",
        moe_quant_mode="fp8_block",
        fmha_quant_mode="bfloat16",
    )
    profile["deployments"] = [deployment]
    options = replace(
        FPMCollectionOptions.from_args(cli._parser().parse_args(_argv(profile))),
        max_gpus=4,
        gpu_counts=(4,),
        parallel_presets=("pure_tp",),
    )
    with pytest.raises(ValueError, match="fmha_quant_mode: resolved='fp8', profile='bfloat16'"):
        _plan(profile, options=options, system="h200_sxm", selected_ops={"attention_context", "attention_generation"})


@pytest.mark.parametrize("field,value", [("model", "other/model"), ("architecture", "OtherArchitecture")])
def test_profile_model_identity_must_match_real_checkpoint(field, value, no_models_or_timing_data):
    profile = _profile()
    profile[field] = value
    with pytest.raises(ValueError, match="profile (model identity|architecture) mismatch"):
        _plan(profile, model_path="nvidia/GLM-5.2-NVFP4")


def test_missing_requested_topology_raises_before_memory_admission(monkeypatch, no_models_or_timing_data):
    profile = _profile()
    profile["deployments"] = profile["deployments"][:1]
    monkeypatch.setattr(memory_admission, "load_system_spec", lambda *_args: pytest.fail("admission ran"))
    with pytest.raises(ValueError, match="no matching FPM deployment profile"):
        _plan(profile)


def test_profile_collection_rejects_non_native_kv_dtype(no_models_or_timing_data):
    profile = _profile()
    options = replace(
        FPMCollectionOptions.from_args(cli._parser().parse_args(_argv(profile))), kv_cache_dtypes=("bfloat16",)
    )
    with pytest.raises(ValueError, match="checkpoint-native KV dtype"):
        _plan(profile, options=options)


@pytest.mark.parametrize("field,value", [("max_num_tokens", 4096), ("max_batch_size", 1)])
def test_resource_envelope_errors_do_not_fail_open(field, value, no_models_or_timing_data):
    profile = _profile()
    profile["deployments"][0]["resources"][field] = value
    options = replace(
        FPMCollectionOptions.from_args(cli._parser().parse_args(_argv(profile))), max_prefill_batch_size=4
    )
    with pytest.raises(ValueError, match="resource envelope exceeded"):
        _plan(profile, options=options)


def test_declared_size_capacity_filter_counts_rejected_topology(caplog, no_models_or_timing_data):
    profile = _profile()
    profile["deployments"][0]["resources"]["weights_bytes"] = 200_000_000_000
    plan = _plan(profile)
    assert {cell.parallel_strategy for cell in plan.cells} == {"tep"}
    assert [decision.disposition for decision in plan.topology_memory_admission] == ["rejected", "admitted"]
    assert "dropped 1/2 topologies" in caplog.text


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("smoke", [False, True])
def test_profile_bounds_native_scheduling_before_cases_are_queued(phase, smoke, no_models_or_timing_data):
    plan = _plan(_profile())
    cell = next(cell for cell in plan.cells if cell.workload_kind == phase)
    arguments = runner._cell_generator_overrides(plan, cell, {}, smoke=smoke)["params"]["agg"]["extra_cli_args"]
    assert arguments.count("--max-num-batched-tokens") == 1
    assert arguments[arguments.index("--max-num-batched-tokens") + 1] == "8192"
    assert arguments.count("--max-num-seqs") == 1
    assert arguments[arguments.index("--max-num-seqs") + 1] == "1024"


def test_profile_preserves_explicit_prefill_envelope(no_models_or_timing_data):
    profile = _profile()
    options = replace(
        FPMCollectionOptions.from_args(cli._parser().parse_args(_argv(profile))),
        max_prefill_isl=2048,
        max_prefill_batch_size=4,
    )
    plan = _plan(profile, options=options)
    for cell in plan.cells:
        arguments = runner._cell_generator_overrides(plan, cell, {})["params"]["agg"]["extra_cli_args"]
        expected_tokens, expected_batch = ("2048", "4") if cell.workload_kind == "prefill" else ("8192", "1024")
        assert arguments[arguments.index("--max-num-batched-tokens") + 1] == expected_tokens
        assert arguments[arguments.index("--max-num-seqs") + 1] == expected_batch


@pytest.mark.parametrize("actual_version", ["0.25.1", "0.24.0"])
def test_profile_runtime_version_is_observed_before_execution(tmp_path, monkeypatch, actual_version):
    resource = object.__new__(runner.KubernetesCellRunner)
    monkeypatch.setattr(runner, "FPM_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(importlib.metadata, "version", lambda _: actual_version)

    def execute(_pod, command, timeout):
        if command[:2] == ["python3", "-c"]:
            monkeypatch.setattr(sys, "argv", ["-c", *command[3:]])
            exec(command[2], {})

    resource._exec_checked = execute
    arguments = dict(cell_id="cell", plan_sha256="plan", attempt_id="attempt", expected_backend_version="0.25.1")
    if actual_version != "0.25.1":
        with pytest.raises(RuntimeError, match="profile runtime mismatch"):
            resource.prepare_attempt(["pod-0"], **arguments)
    else:
        resource.prepare_attempt(["pod-0"], **arguments)
    provenance = json.loads((tmp_path / "collector-provenance.json").read_text())
    assert provenance["runtime"]["backend_version"] == actual_version


def test_profile_input_is_not_valid_for_op_collection():
    from argparse import Namespace

    with pytest.raises(ValueError, match="--fpm-model-profile"):
        reject_fpm_arguments_without_fpm(Namespace(ops=["gemm"], fpm_model_profile="profile.json"))


def test_multiple_profile_runtime_versions_fail_explicitly(no_models_or_timing_data):
    profile = _profile()
    other = copy.deepcopy(profile["deployments"][0])
    other["backend_version"] = "0.24.0"
    profile["deployments"].append(other)
    with pytest.raises(ValueError, match="must identify one runtime version"):
        _plan(profile)
