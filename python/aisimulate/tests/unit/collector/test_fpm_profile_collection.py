# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
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

    from aisimulate_core.sdk import engine, memory, models, perf_database
    from aisimulate_core.sdk.models import base

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


def _multimodal_config():
    # Independently authored geometry; the wrapper is the checkpoint identity,
    # while the nested model type establishes the supported decoder layout.
    return {
        "architectures": ["ExampleMultimodalForConditionalGeneration"],
        "model_type": "example_multimodal",
        "text_config": {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 1024,
            "max_position_embeddings": 4096,
            "torch_dtype": "bfloat16",
        },
        "vision_config": {"hidden_size": 512, "num_hidden_layers": 8},
    }


def _onboard_collection_plan(tmp_path, document, overrides=None):
    from aisimulate.support.config_profile import derive_profile
    from aisimulate.support.config_profile import load_model_config as load_onboarding_config
    from aisimulate.support.fpm import fpm_cli_args
    from aisimulate.support.schema import SupportRequest

    path = tmp_path / "config.json"
    path.write_text(json.dumps(document))
    request = SupportRequest.model_validate(
        {
            "identity": {
                "model": "example/multimodal-checkpoint",
                "model_revision": "collector-test-snapshot",
                "model_kind": "dense",
                "framework_version": "0.25.1",
                "gpu": "h200_sxm",
                "interconnect": "NVLink",
            },
            "search": {"context_length": 4096},
            "workload": {"input_tokens": 128, "concurrency": 8, "request_count": 8},
        }
    )
    draft = derive_profile(
        load_onboarding_config(path),
        request,
        {
            "fmha_quant_mode": "bfloat16",
            "comm_quant_mode": "half",
            "kv_cache_dtype": "bfloat16",
            **(overrides or {}),
        },
    )
    assert draft.profile is not None, draft.missing
    assert hashlib.sha256(path.read_bytes()).hexdigest() in draft.profile.provenance
    request = SupportRequest.model_validate({**request.model_dump(), "fpm_profile": draft.profile})
    command = fpm_cli_args(request, output_dir=tmp_path / "plan", plan_only=True)
    args = cli._parser().parse_args(command[3:])
    return planner.build_collection_plan(
        backend="vllm",
        model_path=request.identity.model,
        model_architecture=args.model_architecture,
        model_config_path=str(path),
        system="h200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        options=FPMCollectionOptions.from_args(args),
        fpm_profile=draft.profile,
    )


@pytest.mark.parametrize("declare_text_architecture", [False, True])
@pytest.mark.parametrize(
    "wrapper_fields",
    [
        {},
        {"n_routed_experts": 32},
        {"kv_lora_rank": 64},
        {"q_lora_rank": 32, "num_experts": 16, "sliding_window": 64, "multi_query": True, "hidden_size": 1024},
    ],
)
def test_config_onboarding_architecture_matches_multimodal_collection(
    tmp_path, no_models_or_timing_data, declare_text_architecture, wrapper_fields
):
    document = {**_multimodal_config(), **wrapper_fields}
    if not declare_text_architecture:
        del document["text_config"]["architectures"]
    plan = _onboard_collection_plan(tmp_path, document)
    expected = "LlamaForCausalLM" if declare_text_architecture else "ExampleMultimodalForConditionalGeneration"
    assert plan.capability.architecture == plan.fpm_profile.architecture == expected
    assert plan.capability.is_moe is False
    assert plan.capability.attention_kind == "dense_gqa"
    assert len(plan.cells) == 2
    assert "Text decoder only" in plan.fpm_profile.provenance
    assert {decision["source"] for decision in plan.to_dict()["topology_memory_admission"]} == {"fpm_profile_declared"}
    parsed = plan.capability.model_config.parsed_payload()
    assert parsed["hidden_size"] == 128
    assert parsed["num_experts"] == 0
    assert all(key not in parsed["raw_config"] for key in wrapper_fields if key != "hidden_size")
    cell_dir = tmp_path / "rendered"
    cell_dir.mkdir()
    runner._render_cell(plan, plan.cells[0], cell_dir, {})
    assert "--model example/multimodal-checkpoint" in (cell_dir / "run.sh").read_text()


@pytest.mark.parametrize("outer_alias", ["quant_algo", "hf_quant_config"])
@pytest.mark.parametrize(
    "algorithm,expected_gemm,expected_moe", [("fp8", "fp8_static", "fp8"), ("nvfp4", "nvfp4", "nvfp4")]
)
@pytest.mark.parametrize("shared_precision", [False, True])
@pytest.mark.parametrize("dtype_key", ["dtype", "torch_dtype"])
def test_config_onboarding_precision_matches_multimodal_collection(
    tmp_path, no_models_or_timing_data, outer_alias, algorithm, expected_gemm, expected_moe, shared_precision, dtype_key
):
    document = _multimodal_config()
    text = document["text_config"]
    del text["torch_dtype"]
    selected = document if shared_precision else text
    selected[dtype_key] = "bfloat16"
    selected["quantization_config"] = {"quant_method": algorithm, "kv_cache_scheme": {"type": "float", "num_bits": 8}}
    other_dtype_key = "torch_dtype" if dtype_key == "dtype" else "dtype"
    outer_algorithm = algorithm if shared_precision else ("nvfp4" if algorithm == "fp8" else "fp8")
    document[outer_alias] = (
        outer_algorithm if outer_alias == "quant_algo" else {"quantization": {"quant_algo": outer_algorithm.upper()}}
    )
    if shared_precision:
        # Null aliases, like missing aliases, do not declare decoder precision.
        text[dtype_key] = None
        text["quantization_config"] = None
    else:
        document[other_dtype_key] = "float32"
        document.update(quant_dynamic=True, kv_cache_quant_algo="int8")
    plan = _onboard_collection_plan(
        tmp_path,
        document,
        {"fmha_quant_mode": "fp8", "kv_cache_dtype": "fp8", "weights_bytes": 1024, "activations_bytes": 1024},
    )
    assert len(plan.cells) == 2
    assert plan.dtype_profile.gemm_quant_mode == expected_gemm
    assert plan.dtype_profile.moe_quant_mode == expected_moe
    assert plan.dtype_profile.native_kv_cache_dtype == "fp8"
    evidence = plan.capability.model_config
    frozen = evidence.to_dict()
    assert frozen["payload"] == document
    assert (
        frozen["sha256"]
        == hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
    )
    raw = evidence.parsed_payload()["raw_config"]
    assert raw["quant_algo"] == algorithm
    assert raw["kv_cache_quant_algo"] == "fp8"
    assert "quant_dynamic" not in raw
    assert raw[dtype_key] == "bfloat16"
    assert other_dtype_key not in raw
    if not shared_precision:
        assert "hf_quant_config" not in raw
    raw["quantization_config"]["quant_method"] = "mutated"
    detached = evidence.effective_payload
    detached["quantization_config"]["quant_method"] = "mutated"
    (tmp_path / "config.json").unlink()
    cell_dir = tmp_path / "rendered"
    cell_dir.mkdir()
    runner._render_cell(plan, plan.cells[0], cell_dir, {})
    assert "--model example/multimodal-checkpoint" in (cell_dir / "run.sh").read_text()
    assert evidence.parsed_payload()["raw_config"]["quant_algo"] == algorithm
    assert evidence.to_dict() == frozen


def test_registered_multimodal_parser_keeps_context_out_of_decoder_metadata(tmp_path, no_models_or_timing_data):
    document = _multimodal_config()
    document.update(
        architectures=["Llama4ForConditionalGeneration"],
        n_routed_experts=32,
        kv_lora_rank=64,
        hf_quant_config={"quantization": {"quant_algo": "NVFP4"}},
        quant_dynamic=True,
        vision_config={
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_channels": 3,
            "intermediate_size": 64,
            "image_size": 16,
            "patch_size": 4,
            "pixel_shuffle_ratio": 0.5,
            "projector_input_dim": 32,
            "projector_output_dim": 16,
            "vision_output_dim": 16,
        },
        image_processor_config={"max_patches": 1, "resize_to_max_canvas": False, "add_global_tile": False},
    )
    del document["text_config"]["architectures"]
    document["text_config"]["quantization_config"] = {"quant_method": "fp8", "kv_cache_scheme": "FP8"}
    plan = _onboard_collection_plan(
        tmp_path,
        document,
        {"fmha_quant_mode": "fp8", "kv_cache_dtype": "fp8", "weights_bytes": 1024, "activations_bytes": 1024},
    )
    assert plan.capability.architecture == "Llama4ForConditionalGeneration"
    assert plan.capability.is_moe is False
    assert plan.capability.attention_kind == "dense_gqa"
    assert plan.dtype_profile.gemm_quant_mode == "fp8_static"
    evidence = plan.capability.model_config
    frozen = evidence.to_dict()
    parsed = evidence.parsed_payload()
    assert parsed["hidden_size"] == 128
    assert parsed["num_experts"] == 0
    assert parsed["extra_params"].vision_config.hidden_size == 16
    assert parsed["extra_params"].vision_config.max_num_tiles == 1
    assert parsed["raw_config"]["quant_algo"] == "fp8"
    assert all(
        key not in parsed["raw_config"]
        for key in ("n_routed_experts", "kv_lora_rank", "vision_config", "image_processor_config", "hf_quant_config")
    )
    (tmp_path / "config.json").unlink()
    cell_dir = tmp_path / "rendered"
    cell_dir.mkdir()
    runner._render_cell(plan, plan.cells[0], cell_dir, {})
    assert "--model example/multimodal-checkpoint" in (cell_dir / "run.sh").read_text()
    assert evidence.to_dict() == frozen


def test_unknown_decoder_quantization_is_not_replaced_by_wrapper_precision(tmp_path, no_models_or_timing_data):
    document = _multimodal_config()
    document["quantization_config"] = {"quant_method": "fp8"}
    document["text_config"]["quant_algo"] = "unknown_decoder_quantization"
    with pytest.raises(ValueError, match="Unsupported quant algorithm: unknown_decoder_quantization"):
        _onboard_collection_plan(
            tmp_path,
            document,
            {"gemm_quant_mode": "fp8_static", "moe_quant_mode": "fp8", "weights_bytes": 1024},
        )


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
        FPMCollectionOptions.from_args(cli._parser().parse_args(_argv(profile))),
        max_prefill_isl=8192,
        max_prefill_batch_size=4,
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
    # A profile's declared context is now authoritative; legacy profile
    # callers previously used -1 and could auto-fit beyond that contract.
    assert arguments[arguments.index("--max-model-len") + 1] == "8192"


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


@pytest.mark.parametrize("with_profile", [False, True])
@pytest.mark.parametrize("narrow_prefill", [False, True])
@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize(
    "max_tokens,max_sequences,prefill_tokens,prefill_sequences", [(4096, 64, 1024, 4), (256, 256, 128, 128)]
)
def test_cli_runtime_limits_reach_both_rendered_workers(
    tmp_path,
    monkeypatch,
    with_profile,
    narrow_prefill,
    smoke,
    max_tokens,
    max_sequences,
    prefill_tokens,
    prefill_sequences,
):
    """Exercise parsing, plan resolution and real Generator output without launching GPUs."""
    profile = _profile()
    argv = [
        *_argv(profile),
        "--fpm-max-model-len",
        "4096",
        "--fpm-max-num-batched-tokens",
        str(max_tokens),
        "--fpm-max-num-seqs",
        str(max_sequences),
    ]
    if with_profile:
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(profile))
        argv.extend(["--fpm-model-profile", str(path)])
    if narrow_prefill:
        argv.extend(
            ["--fpm-max-prefill-isl", str(prefill_tokens), "--fpm-max-prefill-batch-size", str(prefill_sequences)]
        )
    if smoke:
        argv.append("--smoke")

    def render_without_launch(_args, resolved):
        plan, overrides = resolved
        for phase in ("prefill", "decode"):
            cell = next(cell for cell in plan.cells if cell.workload_kind == phase)
            target = tmp_path / phase
            target.mkdir()
            runner._render_cell(plan, cell, target, overrides, smoke=_args.smoke)
        return []

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "")
    monkeypatch.setattr(cli, "run_resolved", render_without_launch)
    assert cli.main(argv) == 0
    for phase in ("prefill", "decode"):
        script = (tmp_path / phase / "run.sh").read_text()
        tokens, sequences = (
            (prefill_tokens, prefill_sequences)
            if narrow_prefill and phase == "prefill"
            else (max_tokens, max_sequences)
        )
        for option, value in (
            ("--max-model-len", 4096),
            ("--max-num-batched-tokens", tokens),
            ("--max-num-seqs", sequences),
        ):
            assert script.count(option) == 1
            assert f"{option} {value}" in script


@pytest.mark.parametrize("smoke", [False, True])
def test_omitted_prefill_defaults_stay_within_selected_profile_bounds(no_models_or_timing_data, smoke):
    profile = _profile()
    for deployment in profile["deployments"]:
        deployment["resources"].update(max_num_tokens=4096, max_batch_size=32)
    plan = _plan(profile)
    assert plan.options.prefill_sampling.max_total_prefill_tokens == 4096
    for cell in plan.cells:
        arguments = runner._cell_generator_overrides(plan, cell, {}, smoke=smoke)["params"]["agg"]["extra_cli_args"]
        for option, value in (
            ("--max-model-len", "8192"),
            ("--max-num-batched-tokens", "4096"),
            ("--max-num-seqs", "32"),
        ):
            assert arguments.count(option) == 1
            assert arguments[arguments.index(option) + 1] == value


@pytest.mark.parametrize(
    "with_profile,resource_limits,limits",
    [
        (False, {}, ["--fpm-max-num-batched-tokens", "128", "--fpm-max-num-seqs", "256"]),
        (True, {}, ["--fpm-max-num-batched-tokens", "128", "--fpm-max-num-seqs", "256"]),
        (
            False,
            {},
            ["--fpm-max-num-batched-tokens", "4096", "--fpm-max-num-seqs", "256", "--fpm-max-prefill-isl", "128"],
        ),
        (False, {}, ["--fpm-max-num-seqs", "16384"]),
        (True, {}, ["--fpm-max-num-batched-tokens", "128"]),
        (True, {}, ["--fpm-max-prefill-isl", "128"]),
        (True, {"max_num_tokens": 128, "max_batch_size": 256}, []),
        (
            True,
            {"max_num_tokens": 128, "max_batch_size": 256},
            ["--fpm-max-prefill-batch-size", "8"],
        ),
    ],
)
@pytest.mark.parametrize("smoke", [False, True])
def test_cli_rejects_scheduler_tokens_below_sequences_before_execution(
    tmp_path, monkeypatch, capsys, with_profile, resource_limits, limits, smoke
):
    profile = _profile()
    for deployment in profile["deployments"]:
        deployment["resources"].update(resource_limits)
    argv = [*_argv(profile), *limits]
    if with_profile:
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(profile))
        argv.extend(["--fpm-model-profile", str(path)])
    if smoke:
        argv.append("--smoke")
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "")
    monkeypatch.setattr(cli, "run_resolved", lambda *_args: pytest.fail("collection execution started"))

    with pytest.raises(SystemExit) as error:
        cli.main(argv)

    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "max_num_batched_tokens" in message
    assert "max_num_seqs" in message


@pytest.mark.parametrize(
    "limits, message",
    [
        (["--fpm-max-model-len", "8193"], "context"),
        (["--fpm-max-num-batched-tokens", "8192", "--fpm-max-prefill-isl", "1024"], "resource envelope exceeded"),
        (["--fpm-max-num-seqs", "128", "--fpm-max-prefill-batch-size", "4"], "resource envelope exceeded"),
        (["--fpm-max-prefill-isl", "8192"], "resource envelope exceeded"),
        (["--fpm-max-prefill-batch-size", "128"], "resource envelope exceeded"),
    ],
)
def test_cli_rejects_profile_limit_overshoots_before_execution(tmp_path, monkeypatch, capsys, limits, message):
    profile = _profile()
    for deployment in profile["deployments"]:
        deployment["resources"].update(max_num_tokens=4096, max_batch_size=64)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "")
    monkeypatch.setattr(cli, "run_resolved", lambda *_args: pytest.fail("collection execution started"))
    with pytest.raises(SystemExit) as error:
        cli.main([*_argv(profile), "--fpm-model-profile", str(path), *limits])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


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
