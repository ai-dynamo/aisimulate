# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import csv
import json
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest
from collector.case_generator import (
    get_attention_head_configs,
    get_gemm_case_specs,
    get_moe_quantization_specs,
    moe_model_allows_quantization,
)
from collector.model_cases import (
    BASE_OP_CASES_DIR,
    build_collection_case_plan,
    default_architecture_cases_path,
    load_yaml_file,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SUPPORT_MATRIX_ROOT = REPO_ROOT / "src" / "aisimulate_core" / "systems" / "support_matrix"


def _planned_moe_modes(monkeypatch, backend: str, model_path: str, sm_version: int) -> set[str]:
    from collector.op_plans import plan

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    return {case.moe_type for case in plan("moe", backend, sm_version).cases if case.model_name == model_path}


def test_model_case_plan_merges_required_base_and_framework_specific_ops():
    plan = build_collection_case_plan(backend="sglang", model_path="deepseek-ai/DeepSeek-V3")

    assert plan.model_architecture == "DeepseekV3ForCausalLM"
    assert plan.model_cases_paths == [default_architecture_cases_path("DeepseekV3ForCausalLM")]
    assert plan.has_op("gemm")
    assert not plan.has_op("attention_context")
    assert not plan.has_op("attention_generation")
    assert "moe" in plan.selected_ops
    assert "mla_context" in plan.selected_ops
    assert "wideep_mla_context" not in plan.selected_ops
    # The sglang plan keeps moe_ep out (separate WideEP 0.5.10 runtime); the
    # trtllm plan activates it (same image as stock trtllm) — see
    # tests/unit/collector/trtllm/test_collect_moe_ep.py.
    assert "moe_ep" not in plan.selected_ops
    assert "trtllm_moe_wideep" not in plan.selected_ops  # retired op name


def test_attention_head_configs_preserve_real_model_structures_without_cross_mixing():
    from collector.case_generator import get_attention_context_shape_sweeps, get_attention_generation_shape_sweeps

    expected_model_structures = {
        # Gemma 4 local/global attention.
        (16, 8, 256, 1024),
        (16, 2, 512, 0),
        # Llama 4 local attention.
        (40, 8, 128, 8192),
        # MiMo-V2 global/local attention.
        (64, 4, 192, 0),
        (64, 8, 192, 128),
    }
    impossible_cross_model_mixes = {
        (64, 8, 256, 1024),
        (40, 8, 192, 8192),
        (16, 8, 512, 1024),
        (64, 4, 64, 128),
    }

    for phase, get_shape_sweeps in (
        ("context", get_attention_context_shape_sweeps),
        ("generation", get_attention_generation_shape_sweeps),
    ):
        configs = {
            (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
            for sweep in get_shape_sweeps("sglang")
            for config in get_attention_head_configs(sweep, phase=phase)
        }

        assert expected_model_structures <= configs
        assert configs.isdisjoint(impossible_cross_model_mixes)


def test_native_attention_profiles_drop_non_integral_local_gqa(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "example/unregistered-model")
    shape_sweep = {
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "head_dim": 128,
        "window_size": 0,
        "tensor_parallel_sizes": [1, 2],
    }

    assert [
        (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
        for config in get_attention_head_configs(shape_sweep, phase="generation")
    ] == [
        (6, 3, 128, 0),
    ]


def test_targeted_attention_profile_uses_model_topology(monkeypatch):
    from collector.case_generator import get_attention_context_shape_sweeps

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "Qwen/Qwen3-32B-FP8")
    configs = {
        (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
        for sweep in get_attention_context_shape_sweeps("sglang")
        for config in get_attention_head_configs(sweep, phase="context")
    }

    assert configs == {
        (64, 8, 128, 0),
        (32, 4, 128, 0),
        (16, 2, 128, 0),
        (8, 1, 128, 0),
        (4, 1, 128, 0),
        (2, 1, 128, 0),
        (1, 1, 128, 0),
    }


def test_added_model_attention_profiles_resolve_targeted_topology(monkeypatch):
    from collector.case_generator import get_attention_context_shape_sweeps

    profiles = (
        (("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"), 8, 2, 256, (1, 2, 4, 8)),
        (("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"), 16, 4, 256, (1, 2, 4, 8, 16)),
        (("Qwen/Qwen3.5-122B-A10B",), 32, 2, 256, (1, 2, 4, 8, 16, 32)),
        (("MiniMaxAI/MiniMax-M2", "MiniMaxAI/MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.7"), 48, 8, 128, (1, 2, 4, 8, 16)),
        (("Qwen/Qwen3-30B-A3B",), 32, 4, 128, (1, 2, 4, 8)),
        (("Qwen/Qwen3.8-2.4T-A95B", "Qwen/Qwen3.8-2.4T-A95B-FP8"), 64, 4, 256, (1, 2, 4, 8, 16, 32)),
    )

    for model_paths, num_heads, num_kv_heads, head_dim, tp_sizes in profiles:
        expected = {
            (num_heads // tp, (num_kv_heads + tp - 1) // tp, head_dim, 0)
            for tp in tp_sizes
            if (num_heads // tp) % ((num_kv_heads + tp - 1) // tp) == 0
        }
        for model_path in model_paths:
            monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
            configs = {
                (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
                for sweep in get_attention_context_shape_sweeps("sglang")
                for config in get_attention_head_configs(sweep, phase="context")
            }
            assert configs == expected, model_path


def test_added_model_moe_profiles_resolve_targeted_aliases(monkeypatch):
    from collector.case_generator import get_common_moe_test_cases

    expected_by_model = {
        "Qwen/Qwen3.5-122B-A10B": ("Qwen/Qwen3.5-122B-A10B", 3072, 1024, 8, 256),
        "Qwen/Qwen3-235B-A22B-Instruct-2507": ("Qwen/Qwen3-235B-A22B", 4096, 1536, 8, 128),
        "MiniMaxAI/MiniMax-M2": ("MiniMaxAI/MiniMax-M2.5", 3072, 1536, 8, 256),
    }
    for model_path, expected in expected_by_model.items():
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
        cases = get_common_moe_test_cases()
        assert {
            (case.model_name, case.hidden_size, case.inter_size, case.topk, case.num_experts) for case in cases
        } == {expected}


@pytest.mark.parametrize(
    ("model_path", "quant_mode"),
    [
        ("nvidia/Qwen3.5-397B-A17B-NVFP4", "nvfp4"),
        ("nvidia/Qwen3.5-122B-A10B-NVFP4", "nvfp4"),
        ("nvidia/Gemma-4-26B-A4B-NVFP4", "bfloat16"),
        ("nvidia/Kimi-K2.6-NVFP4", "nvfp4"),
        ("nvidia/Kimi-K2.7-Code-NVFP4", "nvfp4"),
        ("nvidia/DeepSeek-V4-Flash-NVFP4", "nvfp4"),
        ("nvidia/DeepSeek-V4-Pro-NVFP4", "nvfp4"),
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", "nvfp4"),
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "nvfp4"),
        ("nvidia/MiniMax-M3-NVFP4", "nvfp4"),
    ],
)
def test_nvfp4_quant_artifacts_have_exact_moe_profiles_and_lanes(monkeypatch, model_path, quant_mode):
    from collector.case_generator import get_common_moe_test_cases

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    cases = get_common_moe_test_cases()

    assert cases
    assert {case.model_name for case in cases} == {model_path}
    for backend in ("sglang", "trtllm", "vllm"):
        assert moe_model_allows_quantization(backend, model_path, quant_mode)


@pytest.mark.parametrize("backend", ["sglang", "trtllm", "vllm"])
def test_qwen36_w4a16_nvfp4_moe_is_transferred_but_explicitly_deferred(monkeypatch, caplog, backend):
    from collector.case_generator import get_common_moe_test_cases
    from collector.op_plans import plan_cases

    model_path = "nvidia/Qwen3.6-35B-A3B-NVFP4"
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)

    assert "moe" in build_collection_case_plan(backend=backend, model_path=model_path).selected_ops
    transferred_shapes = get_common_moe_test_cases()
    assert transferred_shapes
    assert {case.model_name for case in transferred_shapes} == {model_path}
    assert "w4a16_nvfp4" not in {spec.name for spec in get_moe_quantization_specs()}

    with caplog.at_level("INFO", logger="collector.op_plans"):
        assert plan_cases("moe", backend, 100) == []
    assert f"plan no case (quantization_modes: []): {model_path}" in caplog.text


@pytest.mark.parametrize(
    "model_path",
    [
        "nvidia/Qwen3.6-27B-NVFP4",
        "nvidia/Gemma-4-31B-IT-NVFP4",
    ],
)
def test_nvfp4_dense_artifacts_have_targeted_attention_profiles(monkeypatch, model_path):
    from collector.case_generator import get_attention_context_shape_sweeps

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)

    assert get_attention_context_shape_sweeps("sglang")


@pytest.mark.parametrize(
    ("model_path", "d_model", "global_k_heads", "global_v_heads", "tp_sizes"),
    [
        ("Qwen/Qwen3.5-27B", 5120, 16, 48, (1, 2, 4, 8)),
        ("Qwen/Qwen3.5-35B-A3B", 2048, 16, 32, (1, 2, 4, 8, 16)),
    ],
)
def test_qwen35_gdn_getters_expand_tp_local_physical_keys(
    monkeypatch, model_path, d_model, global_k_heads, global_v_heads, tp_sizes
):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    expected = {
        (phase, d_model, 4, global_k_heads // tp, 128, global_v_heads // tp, 128)
        for phase in ("context", "generation")
        for tp in tp_sizes
    }

    from collector.op_plans import plan_cases

    for backend in ("sglang", "trtllm", "vllm"):
        cases = plan_cases("gdn", backend, 100)
        assert {(case[0], case[1], case[2], case[3], case[4], case[5], case[6]) for case in cases} == expected


def test_gdn_tp_declarations_fail_loud_and_dedupe_on_loader_key(monkeypatch):
    from collector import case_generator

    invalid = {
        "model_path": "example/invalid",
        "d_model": 2048,
        "d_conv": 4,
        "num_k_heads": 16,
        "head_k_dim": 128,
        "num_v_heads": 32,
        "head_v_dim": 128,
        "tensor_parallel_sizes": [3],
    }
    monkeypatch.setattr(case_generator, "_model_case_values", lambda op_name: [invalid])
    with pytest.raises(ValueError, match="both global head counts to be divisible"):
        case_generator.get_common_gdn_test_cases()

    def profile(model_path, d_model, num_k_heads, num_v_heads, tp):
        return {
            "model_path": model_path,
            "d_model": d_model,
            "d_conv": 4,
            "num_k_heads": num_k_heads,
            "head_k_dim": 128,
            "num_v_heads": num_v_heads,
            "head_v_dim": 128,
            "tensor_parallel_sizes": [tp],
        }

    profiles = [
        profile("example/first", 2048, 16, 32, 4),
        profile("example/duplicate", 2048, 4, 8, 1),
        profile("example/distinct-d-model", 4096, 4, 8, 1),
    ]
    monkeypatch.setattr(case_generator, "_model_case_values", lambda op_name: profiles)
    cases = case_generator.get_common_gdn_test_cases()

    assert len(cases) == 4
    assert {(case.phase, case.d_model, case.num_k_heads, case.num_v_heads, case.model_name) for case in cases} == {
        (phase, 2048, 4, 8, "example/first") for phase in ("context", "generation")
    } | {(phase, 4096, 4, 8, "example/distinct-d-model") for phase in ("context", "generation")}


def test_gdn_case_generation_rejects_unkeyed_non_fp32_state_dtype(monkeypatch):
    from collector import case_generator

    profile = {
        "model_path": "example/bfloat16-state",
        "d_model": 2048,
        "d_conv": 4,
        "num_k_heads": 16,
        "head_k_dim": 128,
        "num_v_heads": 32,
        "head_v_dim": 128,
        "tensor_parallel_sizes": [1],
        "mamba_ssm_dtype": "bfloat16",
    }
    monkeypatch.setattr(case_generator, "_model_case_values", lambda op_name: [profile])

    with pytest.raises(ValueError, match="only float32 collection is supported"):
        case_generator.get_common_gdn_test_cases()


def test_mimo_attention_profile_matches_aic_full_attention_window(monkeypatch):
    from collector.case_generator import get_attention_context_shape_sweeps

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "XiaomiMiMo/MiMo-7B-Base")
    configs = {
        (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
        for sweep in get_attention_context_shape_sweeps("vllm")
        for config in get_attention_head_configs(sweep, phase="context")
    }

    assert configs == {
        (32, 8, 128, 0),
        (16, 4, 128, 0),
        (8, 2, 128, 0),
        (4, 1, 128, 0),
        (2, 1, 128, 0),
        (1, 1, 128, 0),
    }


def test_qwen_vl_attention_profiles_stop_at_sdk_valid_tp16(monkeypatch):
    from collector.case_generator import get_attention_context_shape_sweeps

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "Qwen/Qwen3-VL-32B-Instruct")
    configs = {
        (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
        for sweep in get_attention_context_shape_sweeps("vllm")
        for config in get_attention_head_configs(sweep, phase="context")
    }

    assert configs == {
        (64, 8, 128, 0),
        (32, 4, 128, 0),
        (16, 2, 128, 0),
        (8, 1, 128, 0),
        (4, 1, 128, 0),
    }


def test_full_encoder_attention_profiles_combine_defaults_and_model_deltas(monkeypatch):
    from collector.case_generator import get_attention_encoder_head_configs, get_attention_encoder_shape_sweeps

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    for backend in ("sglang", "trtllm", "vllm"):
        sweeps = get_attention_encoder_shape_sweeps(backend)
        keys = {
            (config.num_heads, config.head_dim)
            for sweep in sweeps
            for config in get_attention_encoder_head_configs(sweep)
        }
        default_keys = {
            (num_heads, head_dim)
            for sweep in sweeps
            for head_dim in sweep["head_dims"]
            for num_heads in sweep["head_counts"]
        }

        assert default_keys <= keys
        # (1,64)/(1,72): TP=full shards of Qwen3-VL/Kimi ViTs down to one head.
        # (*,104): Mistral-Medium-3.5 Pixtral ViT (head_dim=104 is absent from the
        # base 64/72/80 grid), TP-sharded to num_heads {16,8,4,2,1}.
        assert keys - default_keys == {(1, 64), (1, 72), (1, 104), (2, 104), (4, 104), (8, 104), (16, 104)}


def test_targeted_encoder_attention_profile_is_model_exact(monkeypatch):
    from collector.case_generator import get_attention_encoder_head_configs, get_attention_encoder_shape_sweeps

    for model_path, head_dim in (
        ("Qwen/Qwen3-VL-4B-Instruct", 64),
        ("Qwen/Qwen3-VL-32B-Instruct", 72),
        ("Qwen/Qwen3-VL-235B-A22B-Instruct", 72),
        ("moonshotai/Kimi-K2.5", 72),
        ("nvidia/Kimi-K2.5-NVFP4", 72),
    ):
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
        configs = [
            config
            for sweep in get_attention_encoder_shape_sweeps("vllm")
            for config in get_attention_encoder_head_configs(sweep)
        ]

        assert {(config.num_heads, config.head_dim) for config in configs} == {
            (16, head_dim),
            (8, head_dim),
            (4, head_dim),
            (2, head_dim),
            (1, head_dim),
        }


def test_base_gemm_cases_are_readable_shape_specs():
    plan = build_collection_case_plan(backend="sglang", model_path="Qwen/Qwen3-32B")
    assert plan.has_op("gemm")
    assert plan.has_op("attention_context")
    assert plan.has_op("attention_generation")

    def base_specs(op_name):
        data = load_yaml_file(BASE_OP_CASES_DIR / f"{op_name}.yaml")
        return data["all_frameworks_op_cases"][op_name]["cases"]

    gemm_specs = base_specs("gemm")
    assert len(gemm_specs) == 1
    spec = gemm_specs[0]
    assert spec["id"] == "base_transformer_gemm_shape_sweep"
    assert spec["token_counts"][:5] == [1, 2, 3, 4, 5]
    assert spec["feature_sizes"][:3] == [32, 64, 128]

    context_spec = base_specs("attention_context")[0]
    assert context_spec["id"] == "base_attention_context_shape_sweep"
    assert context_spec["kv_head_options"] == ["self", 1, 2, 4, 8]
    generation_spec = base_specs("attention_generation")[0]
    assert generation_spec["id"] == "base_attention_generation_shape_sweep"
    assert generation_spec["xqa_query_head_counts"][-1] == 64


def test_moe_model_quantization_policy_is_yaml_backed():
    assert moe_model_allows_quantization("sglang", "deepseek-ai/DeepSeek-V4-Flash", "w4a8_mxfp4_mxfp8")
    assert not moe_model_allows_quantization("sglang", "deepseek-ai/DeepSeek-V4-Flash", "bfloat16")
    assert not moe_model_allows_quantization("sglang", "Qwen/Qwen3-235B-A22B", "w4a8_mxfp4_mxfp8")
    assert moe_model_allows_quantization("sglang", "nvidia/GLM-5.2-NVFP4", "nvfp4")
    assert not moe_model_allows_quantization("sglang", "nvidia/GLM-5.2-NVFP4", "bfloat16")
    assert moe_model_allows_quantization("sglang", "zai-org/GLM-5-FP8", "fp8_block")
    assert not moe_model_allows_quantization("sglang", "zai-org/GLM-5-FP8", "nvfp4")
    # zai-org/GLM-5.3 is Z.AI's FP8-block release; the BF16 weights carry the
    # -BF16 suffix (there is no zai-org/GLM-5.3-FP8 repository).
    assert moe_model_allows_quantization("sglang", "zai-org/GLM-5.3", "fp8_block")
    assert not moe_model_allows_quantization("sglang", "zai-org/GLM-5.3", "bfloat16")
    assert not moe_model_allows_quantization("sglang", "zai-org/GLM-5.3", "nvfp4")
    assert moe_model_allows_quantization("sglang", "zai-org/GLM-5.3-BF16", "bfloat16")
    assert not moe_model_allows_quantization("sglang", "zai-org/GLM-5.3-BF16", "fp8_block")

    assert moe_model_allows_quantization("sglang", "openai/gpt-oss-120b", "w4a16_mxfp4")
    assert moe_model_allows_quantization("sglang", "openai/gpt-oss-120b", "w4a8_mxfp4_mxfp8")
    assert not moe_model_allows_quantization("sglang", "openai/gpt-oss-120b", "bfloat16")

    assert moe_model_allows_quantization("trtllm", "moonshotai/Kimi-K2.5", "int4_wo")
    assert not moe_model_allows_quantization("trtllm", "moonshotai/Kimi-K2.5", "w4a16_mxfp4")
    assert not moe_model_allows_quantization("trtllm", "moonshotai/Kimi-K2.5", "bfloat16")
    assert not moe_model_allows_quantization("trtllm", "Qwen/Qwen3-235B-A22B", "w4a16_mxfp4")
    assert not moe_model_allows_quantization("trtllm", "openai/gpt-oss-20b", "fp8")


def test_dsv4_moe_quantization_policy_prunes_unrelated_modes():
    # AIC-1749: nvidia/DeepSeek-V4-{Flash,Pro}-NVFP4 are a real ModelOpt NVFP4
    # export. Every backend that serves NVFP4 MoE declares exactly [nvfp4] on
    # those rows (positive one-true-mode declarations, review follow-up), so
    # assert the singleton set per backend explicitly rather than omitting the
    # artifact (omission would look identical to "not yet declared").
    expected_by_backend = {
        "sglang": {
            "deepseek-ai/DeepSeek-V4-Flash": {"w4a8_mxfp4_mxfp8"},
            "deepseek-ai/DeepSeek-V4-Pro": {"w4a8_mxfp4_mxfp8"},
            "sgl-project/DeepSeek-V4-Flash-FP8": {"fp8_block"},
            "sgl-project/DeepSeek-V4-Pro-FP8": {"fp8_block"},
            # One true mode everywhere: the NVFP4 artifacts declare nvfp4 on
            # every backend that serves NVFP4 MoE (positive declarations,
            # nothing excluded).
            "nvidia/DeepSeek-V4-Flash-NVFP4": {"nvfp4"},
            "nvidia/DeepSeek-V4-Pro-NVFP4": {"nvfp4"},
        },
        "trtllm": {
            "deepseek-ai/DeepSeek-V4-Flash": {"w4a8_mxfp4_mxfp8"},
            "deepseek-ai/DeepSeek-V4-Pro": {"w4a8_mxfp4_mxfp8"},
            "sgl-project/DeepSeek-V4-Flash-FP8": {"fp8_block"},
            "sgl-project/DeepSeek-V4-Pro-FP8": {"fp8_block"},
            "nvidia/DeepSeek-V4-Flash-NVFP4": {"nvfp4"},
            "nvidia/DeepSeek-V4-Pro-NVFP4": {"nvfp4"},
        },
        "vllm": {
            "deepseek-ai/DeepSeek-V4-Flash": {"w4a8_mxfp4_mxfp8"},
            "deepseek-ai/DeepSeek-V4-Pro": {"w4a8_mxfp4_mxfp8"},
            "sgl-project/DeepSeek-V4-Flash-FP8": {"fp8_block"},
            "sgl-project/DeepSeek-V4-Pro-FP8": {"fp8_block"},
            "nvidia/DeepSeek-V4-Flash-NVFP4": {"nvfp4"},
            "nvidia/DeepSeek-V4-Pro-NVFP4": {"nvfp4"},
        },
    }

    for backend, expected_by_artifact in expected_by_backend.items():
        available_modes = {spec.name for spec in get_moe_quantization_specs(backend)}
        for model_path, expected in expected_by_artifact.items():
            allowed = {mode for mode in available_modes if moe_model_allows_quantization(backend, model_path, mode)}
            assert allowed == expected, (backend, model_path)


def test_qwen35_397b_nvfp4_moe_row_is_nvfp4_only_on_every_backend():
    # AIC-1715/1716 rebase-4 review (Blocker 2): the row briefly carried
    # ``frameworks: [sglang]``, which made ``_model_moe_backend_quantization``
    # skip the row entirely for trtllm/vllm (case_generator.py `continue` on
    # a framework mismatch) instead of narrowing it — an EMPTY
    # ``model_quantization`` inverts the gate rather than tightening it:
    # trtllm has no ``requires_model_quantization_config`` floor, so it fell
    # back to allow-everything (would have queued cases under every trtllm
    # MoE quant mode, not just nvfp4); vllm's nvfp4 spec DOES require a
    # model-quantization entry, so it lost its legitimate nvfp4 cases
    # instead. A per-backend, explicitly-named-backend assertion (never
    # ``backend=None``, which both counter helpers above are blind to) is
    # the only shape that would have caught this.
    model_path = "nvidia/Qwen3.5-397B-A17B-NVFP4"
    for backend in ("sglang", "trtllm", "vllm"):
        available_modes = {spec.name for spec in get_moe_quantization_specs(backend)}
        allowed = {mode for mode in available_modes if moe_model_allows_quantization(backend, model_path, mode)}
        assert allowed == {"nvfp4"}, (backend, model_path, allowed)


def test_kimi_moe_quantization_is_artifact_specific(monkeypatch):
    expected_by_artifact = {
        "moonshotai/Kimi-K2-Instruct": {"fp8_block"},
        "moonshotai/Kimi-K2.5": {"int4_wo"},
        "nvidia/Kimi-K2.5-NVFP4": {"nvfp4"},
    }

    available_modes = {spec.name for spec in get_moe_quantization_specs()}
    for backend in ("sglang", "trtllm", "vllm"):
        for model_path, expected in expected_by_artifact.items():
            allowed = {mode for mode in available_modes if moe_model_allows_quantization(backend, model_path, mode)}
            assert allowed == expected, (backend, model_path)

    from collector.case_generator import get_moe_quantization_module_config

    assert get_moe_quantization_module_config("sglang", "int4_wo", model_name="moonshotai/Kimi-K2.5") == {
        "group_size": 32
    }
    # SGLang pins the int4 runner on SM89/90/120 (Marlin) and SM100/103
    # (flashinfer_trtllm), not on SM80 (collector/skips.py).
    assert _planned_moe_modes(monkeypatch, "sglang", "moonshotai/Kimi-K2.5", 90) == {"int4_wo"}
    assert _planned_moe_modes(monkeypatch, "sglang", "moonshotai/Kimi-K2.5", 100) == {"int4_wo"}
    assert _planned_moe_modes(monkeypatch, "sglang", "moonshotai/Kimi-K2.5", 80) == set()
    assert _planned_moe_modes(monkeypatch, "sglang", "nvidia/Kimi-K2.5-NVFP4", 90) == set()
    assert _planned_moe_modes(monkeypatch, "sglang", "nvidia/Kimi-K2.5-NVFP4", 100) == {"nvfp4"}


def test_sglang_marlin_is_declared_only_for_weight_only_modes():
    def backend_maps(value):
        if isinstance(value, dict):
            if "sglang_moe_backends" in value:
                yield value["sglang_moe_backends"]
            for child in value.values():
                yield from backend_maps(child)
        elif isinstance(value, list):
            for child in value:
                yield from backend_maps(child)

    base = load_yaml_file(REPO_ROOT / "collector/cases/base_ops/moe.yaml")
    maps = [base["common_case_values"]["moe_sglang"]["backends"]]
    for path in (REPO_ROOT / "collector/cases/models").glob("*_cases.yaml"):
        maps.extend(backend_maps(load_yaml_file(path)))

    marlin_modes = {
        mode for backends in maps for mode, mapping in backends.items() if "marlin" in json.dumps(mapping).lower()
    }
    # Marlin is a weight-only (bf16-activation) runner: INT4-WO everywhere it
    # is declared, plus MXFP4 w4a16 on SM120 where SGLang 0.5.14 serving
    # itself selects Marlin (server_args.py:3876-3887). NVFP4 and
    # mxfp8-activation modes must never declare it (FP4/INT4 identity
    # reversal).
    assert marlin_modes == {"int4_wo", "w4a16_mxfp4"}


@pytest.mark.unit
def test_sglang_registry_marks_unvalidated_dsa_and_moe_platforms_explicitly():
    from collector.sglang.registry import REGISTRY

    sm90 = build_collection_case_plan(backend="sglang", full=True, sm_version=90)
    sm100 = build_collection_case_plan(backend="sglang", full=True, sm_version=100)
    entries = {entry.op: entry for entry in REGISTRY}

    # SM90 unparked by the h100/h200 probe collections (2026-08-14..15,
    # pipelines 62700025 + 62872230): 67,532 context + 4,896 generation
    # skip rows, fa3/flashmla buckets clean.
    for op in ("dsa_context_module_skip_indexer", "dsa_generation_module_skip_indexer"):
        assert op in sm90.selected_ops
        assert op in sm100.selected_ops
        assert entries[op].unverified_sms == (120,)

    # SM103 unparked by the B300 hardware probe (2026-07-13, pipeline
    # 57716023): sampled dsa cases ran clean across all three kernel buckets.
    for op in ("dsa_context_module", "dsa_generation_module"):
        assert entries[op].unverified_sms == (120,)

    # The SM120 MoE bring-up audit (RTX 6000 Pro) cleared the moe
    # maturity marker: every planned (quant mode x backend) family was probed
    # on hardware with verified constructed-method provenance.
    assert entries["moe"].unverified_sms == ()

    # SM120 sparse-round audit (RTX 6000 Pro): the framework
    # itself rejects the DSA/GLM-5 sparse family on SM120 (TRTLLM-GEN
    # fmhaRunner "Unsupported architecture"; DeepGEMM attention.hpp:184;
    # sgl-kernel sparse attention SM90a/SM100f-only), the CSA context pool
    # derivation is fail-closed pending an SM120 Torch-indexer workspace
    # policy, and the topk-calib producer is not yet SM120-variant-aware.
    for op in (
        "dsv4_csa_topk_calib",
        "dsv4_paged_mqa_logits_module",
        "glm5_mqa_logits_module",
        "glm5_topk_module",
        "glm5_dsa_attn_module",
    ):
        assert entries[op].unverified_sms == (120,)

    # SM89 bring-up round 2 (L40S): stock 0.5.14 cannot run
    # DSV4 on SM89 at all — the server_args DeepseekV4 hook leaves
    # SGLANG_OPT_DEEPGEMM_HC_PRENORM default-True on non-SM120/non-HIP
    # platforms while deep_gemm is unimported (NameError in both mHC
    # directions), the compressed FlashMLA family has no SM89 target
    # (in-kernel CUDA InternalError on every HCA/CSA-generation case), and
    # the CSA context pool derivation is fail-closed below SM90.
    assert entries["dsv4_csa_context_module"].unverified_sms == (89, 120)
    for op in (
        "mhc_module",
        "dsv4_hca_context_module",
        "dsv4_hca_generation_module",
        "dsv4_csa_generation_module",
    ):
        assert entries[op].unverified_sms == (89,)

    # Probed clean on SM120 and SM89 with only classified failure tails
    # (GDN int32 kernel-limit raises, decode grid-Y boundary, top-cell
    # OOMs): no markers.
    assert entries["gdn"].unverified_sms == ()


def test_deepseek_minimax_and_nemotron_moe_quantization_is_artifact_specific():
    shared_expected = {
        "deepseek-ai/DeepSeek-V3": {"fp8_block"},
        "deepseek-ai/DeepSeek-R1": {"fp8_block"},
        "deepseek-ai/DeepSeek-V3.2": {"fp8_block"},
        "nvidia/DeepSeek-V3.1-NVFP4": {"nvfp4"},
        "MiniMaxAI/MiniMax-M2": {"fp8_block"},
        "MiniMaxAI/MiniMax-M2.5": {"fp8_block"},
        "MiniMaxAI/MiniMax-M2.7": {"fp8_block"},
        "nvidia/MiniMax-M2.5-NVFP4": {"nvfp4"},
        "nvidia/MiniMax-M2.7-NVFP4": {"nvfp4"},
        "MiniMaxAI/MiniMax-M3": {"bfloat16"},
        # MIXED_PRECISION artifact: routed experts NVFP4 gs16 (the MoE axis);
        # its MXFP8 attention/dense/shared-expert side is not a MoE mode.
        "nvidia/MiniMax-M3-NVFP4": {"nvfp4"},
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": {"bfloat16"},
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": {"nvfp4"},
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": {"bfloat16"},
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4": {"nvfp4"},
    }

    expected_by_artifact = {
        **shared_expected,
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8": {"fp8"},
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8": {"fp8"},
    }
    available_modes = {spec.name for spec in get_moe_quantization_specs()}
    for backend in ("sglang", "trtllm", "vllm"):
        for model_path, expected in expected_by_artifact.items():
            allowed = {mode for mode in available_modes if moe_model_allows_quantization(backend, model_path, mode)}
            assert allowed == expected, (backend, model_path)


def test_gptoss_mxfp4_runs_w4a8_from_blackwell(monkeypatch):
    model_path = "openai/gpt-oss-120b"

    # SGLang (flashinfer_mxfp4 at its default precision, gpt_oss.py:64-75) and TRT-LLM
    # (ModelConfig.get_mxfp4_quant_algo) serve GPT-OSS MXFP4 with MXFP8 activations from
    # SM100, so collector/skips.py leaves the W4A16 cases out there.
    for backend in ("sglang", "trtllm"):
        assert _planned_moe_modes(monkeypatch, backend, model_path, 90) == {"w4a16_mxfp4"}, backend
        for sm in (100, 103, 120):
            assert _planned_moe_modes(monkeypatch, backend, model_path, sm) == {"w4a8_mxfp4_mxfp8"}, (backend, sm)
    # vLLM serves gpt-oss MXFP4 experts with BF16 activations on every SM, and
    # its collector builds the MXFP8-activation path for DeepSeek-V4 only
    # (collector/skips.py).
    for sm in (90, 100, 103, 120):
        assert _planned_moe_modes(monkeypatch, "vllm", model_path, sm) == {"w4a16_mxfp4"}, sm


@pytest.mark.parametrize("backend", ["sglang", "trtllm", "vllm"])
def test_dsv4_native_w4a8_mode_is_planned_from_sm100(monkeypatch, backend):
    model_path = "deepseek-ai/DeepSeek-V4-Flash"

    # The trtllm-gen MXFP4 x MXFP8 kernel starts at SM100 (collector/skips.py).
    assert _planned_moe_modes(monkeypatch, backend, model_path, 90) == set()
    assert _planned_moe_modes(monkeypatch, backend, model_path, 100) == {"w4a8_mxfp4_mxfp8"}
    assert _planned_moe_modes(monkeypatch, backend, model_path, 103) == {"w4a8_mxfp4_mxfp8"}
    # SGLang runs the FP4 experts as MXFP8 x MXFP4 on SM120 too
    # (model_overrides/deepseek_v4.py:75-88, mxfp4_flashinfer_cutlass_moe.py:36-44);
    # vLLM serves MXFP4 MoE on SM120 without MXFP8 activations (collector/skips.py).
    expected_sm120 = set() if backend == "vllm" else {"w4a8_mxfp4_mxfp8"}
    assert _planned_moe_modes(monkeypatch, backend, model_path, 120) == expected_sm120


def test_sglang_mxfp4_quant_labels_select_explicit_activation_precision():
    source_path = REPO_ROOT / "collector/sglang/collect_moe.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_mxfp4_activation_precision"
    )
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source_path), "exec"), namespace)

    assert namespace["_mxfp4_activation_precision"]("w4a16_mxfp4") == "bf16"
    assert namespace["_mxfp4_activation_precision"]("w4a8_mxfp4_mxfp8") == "default"


def test_attention_shape_specs_are_shared_with_an_xpu_override():
    from collector.case_generator import (
        get_attention_context_shape_sweeps,
        get_attention_generation_shape_sweeps,
        get_shared_base_op_case_specs,
    )

    # One grid for every CUDA backend: generic full attention at head_dim 128;
    # model profiles add every other head_dim and window.
    (context,) = get_shared_base_op_case_specs("attention_context")
    (generation,) = get_shared_base_op_case_specs("attention_generation")
    assert context["head_dims"] == [128] and context["window_sizes"] == [0]
    assert context["query_head_counts"][-1] == 64
    assert generation["head_dims"] == [128] and generation["window_sizes"] == [0]
    assert generation["mha_query_head_counts"] == [1, 2, 4, 8]
    for backend in ("sglang", "trtllm", "vllm"):
        assert get_attention_context_shape_sweeps(backend) == [context]
        assert get_attention_generation_shape_sweeps(backend) == [generation]

    # The vLLM XPU collector keeps its own grid.
    (xpu_context,) = get_attention_context_shape_sweeps("vllm_xpu")
    assert xpu_context["batch_sizes"] == [1, 2, 4, 8, 16, 32]
    assert xpu_context["kv_head_options"] == [1, 2, 4, 8]
    assert xpu_context["head_dims"] == [128, 64]


def test_gemm_common_cases_expand_from_base_op_yaml_shape_specs():
    from collector.case_generator import (
        ComputeScaleCommonTestCase,
        GemmCommonTestCase,
        get_compute_scale_case_specs,
        get_gemm_case_specs,
        get_gemm_type_specs,
    )

    cases = get_gemm_case_specs()
    xpu_cases = get_gemm_case_specs("vllm_xpu")

    # Base gemm sweep expansion, then model_case_values.gemm rows. Qwen3.8-Max
    # adds two output widths across the base token-count grid. DeepSeek-V4 Pro
    # adds (n, k) = (7168, 384) for all 74 standard / 21 XPU token counts;
    # its other shared-expert projections overlap the existing base shapes.
    # GLM-5.3-Flash adds ten new off-grid output widths at k=4096 (its
    # declared widths minus those another model already declares) across the
    # 74 standard / 21 XPU token counts.
    assert len(cases) == 38258
    assert cases[0] == GemmCommonTestCase(x=32768, n=65536, k=51200)
    assert cases[-1] == GemmCommonTestCase(x=1, n=1, k=4096)
    assert not any(case.n == 65536 and case.k == 65536 for case in cases)

    assert len(xpu_cases) == 10857
    assert xpu_cases[0] == GemmCommonTestCase(x=8192, n=65536, k=32768)
    assert xpu_cases[-1] == GemmCommonTestCase(x=1, n=1, k=4096)
    assert get_gemm_type_specs("vllm_xpu") == ["bfloat16", "fp8"]

    for backend_cases in (cases, xpu_cases):
        keys = {(case.x, case.n, case.k) for case in backend_cases}
        assert len(keys) == len(backend_cases)
        assert {x for x, n, k in keys if (n, k) == (7168, 384)} == {x for x, _, _ in keys}

    compute_scale_cases = get_compute_scale_case_specs()
    assert len(compute_scale_cases) == 1628
    assert compute_scale_cases[0] == ComputeScaleCommonTestCase(m=32768, k=51200)
    assert compute_scale_cases[-1] == ComputeScaleCommonTestCase(m=1, k=65536)


def test_cross_model_common_cases_expand_from_base_op_yaml_sweeps(monkeypatch):
    from collector.case_generator import (
        get_common_gdn_test_cases,
        get_common_mamba2_test_cases,
        get_common_mhc_test_cases,
        get_common_moe_test_cases,
        get_context_mla_case_specs,
        get_generation_mla_case_specs,
    )

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)

    moe_cases = get_common_moe_test_cases()
    # +117 per new GLM model path: GLM-5.1 (BF16/FP8/NVFP4) and GLM-5.2
    # (BF16/FP8) share GLM-5's MoE dims. nvidia/GLM-5.1-NVFP4 is also
    # registered in moe.yaml base_ops.
    # +114 for Kimi-K3's LatentMoE row (3584/3072, 896x16, w4a16_mxfp4),
    # plus exact quant-sensitive rows for the current NVIDIA NVFP4 artifacts.
    # +198 from Step-3.7-Flash: 99 cases for each physical BF16/FP8 artifact.
    # +117 for the vLLM Nemotron Super FP8 latent-MoE row (1024/2688, 512x22).
    # GLM-5.3 (BF16/FP8/NVFP4) adds no physical cases: its three identifiers
    # are model_aliases on the existing GLM-5.2 rows (same base, zero deltas).
    # +114 for nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4, the nvfp4
    # checkpoint release of the 30B/A3B shape (AIC-1743/AIC-1748); same
    # topk=6/e128/h2688/i1856 geometry as Nano-BF16, hence the identical
    # per-model case count (114).
    # +231 for AIC-1749's DeepSeek-V4 NVFP4 rows: +117 nvidia/DeepSeek-V4-Flash-NVFP4
    # (matches deepseek-ai/DeepSeek-V4-Flash's 4096/2048 count) and +114
    # nvidia/DeepSeek-V4-Pro-NVFP4 (matches deepseek-ai/DeepSeek-V4-Pro's
    # 7168/3072 count) -- every backend declares exactly [nvfp4] for both
    # new rows.
    # +114 for MiniMax-M3's MoE row (6144/3072, 128x4).
    # +114 for the nvidia/MiniMax-M3-NVFP4 row (same 6144/3072, 128x4
    # geometry; quant-distinct artifact — NVFP4 routed experts — so it is a
    # separate row, never merged with the BF16 parent).
    # AIC-1715/1716 rebase-4 review (Blocker 2): nvidia/Qwen3.5-397B-A17B-NVFP4's
    # moe row briefly carried frameworks: [sglang] (citing the InferenceX
    # serving pin), which does not change this total at all -- not
    # "coincidentally", but by construction: get_common_moe_test_cases counts
    # unique (hidden_size, inter_size, topk, num_experts, ...) geometry
    # tuples and never consults model_case_values.moe.frameworks, so a row's
    # framework restriction is invisible to this counter regardless of its
    # value. The frameworks: [sglang] key has since been dropped (it was
    # inverting the trtllm/vllm QUANTIZATION gate elsewhere -- see
    # test_qwen35_397b_nvfp4_moe_row_is_nvfp4_only_on_every_backend, the
    # actual regression); the row's own count (117, pinned separately below)
    # is unaffected either way. Qwen3.8-Max contributes one 117-case base
    # row and one 117-case RadixArk NVFP4 row. GLM-5.3-Flash adds two 99-case
    # rows (zai-org FP8 and nvidia NVFP4 checkpoints, 288 experts x 4096/2048).
    assert len(moe_cases) == 7152

    assert any(
        case.model_name == "nvidia/DeepSeek-V4-Flash-NVFP4"
        and case.hidden_size == 4096
        and case.inter_size == 2048
        and case.topk == 6
        and case.num_experts == 256
        for case in moe_cases
    )
    assert any(
        case.model_name == "nvidia/DeepSeek-V4-Pro-NVFP4"
        and case.hidden_size == 7168
        and case.inter_size == 3072
        and case.topk == 6
        and case.num_experts == 384
        for case in moe_cases
    )
    assert any(
        case.model_name == "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
        and case.hidden_size == 1024
        and case.inter_size == 2688
        for case in moe_cases
    )
    assert any(
        case.model_name == "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
        and case.hidden_size == 2048
        and case.inter_size == 5120
        for case in moe_cases
    )
    # MiniMax-M3's model-owned MoE dims must be present directly (both the
    # BF16 parent and the quant-distinct NVFP4 artifact row), not merely
    # inside the aggregate count above.
    assert any(
        case.model_name == "MiniMaxAI/MiniMax-M3" and case.hidden_size == 6144 and case.inter_size == 3072
        for case in moe_cases
    )
    assert any(
        case.model_name == "nvidia/MiniMax-M3-NVFP4" and case.hidden_size == 6144 and case.inter_size == 3072
        for case in moe_cases
    )
    # Every Lightning case, not just one: the documented +114 contribution,
    # each carrying the exact checkpoint geometry (an `any(...)` pin would
    # pass with 113 wrong-dimension siblings).
    lightning_cases = [
        case for case in moe_cases if case.model_name == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    ]
    assert len(lightning_cases) == 114
    assert all(
        case.hidden_size == 2688 and case.inter_size == 1856 and case.topk == 6 and case.num_experts == 128
        for case in lightning_cases
    )
    # Step-3.7-Flash: assert both physical artifact identities, the shape, and
    # the routing contract. MoE loads the model config by model_name, so the
    # BF16 artifact must not alias to the FP8 representative.
    step_cases = [case for case in moe_cases if "Step-3.7-Flash" in case.model_name]
    assert {case.model_name for case in step_cases} == {
        "stepfun-ai/Step-3.7-Flash",
        "stepfun-ai/Step-3.7-Flash-FP8",
    }
    assert sum(case.model_name == "stepfun-ai/Step-3.7-Flash" for case in step_cases) == 99
    assert sum(case.model_name == "stepfun-ai/Step-3.7-Flash-FP8" for case in step_cases) == 99
    assert all(
        case.hidden_size == 4096 and case.inter_size == 1280 and case.topk == 8 and case.num_experts == 288
        for case in step_cases
    )
    # Sigmoid gate + correction bias before top-k, renormalized and scaled by
    # 3.0. The defaults (softmax, no bias, no scaling) would benchmark a
    # different MoE invocation from the one that is actually served.
    assert all(
        case.sglang_moe_scoring_func == "sigmoid"
        and case.sglang_moe_has_correction_bias
        and case.sglang_moe_renormalize
        and case.sglang_moe_routed_scaling_factor == 3.0
        for case in step_cases
    )

    # Kimi-K3 declares the native 96-head MLA profile (DeepSeek geometry),
    # expanding the MLA spec grids.
    assert len(get_context_mla_case_specs()) == 330
    assert len(get_generation_mla_case_specs()) == 543
    mamba_cases = get_common_mamba2_test_cases()
    assert len(mamba_cases) == 12
    assert {case.model_name for case in mamba_cases} >= {"MAMBA2_GENERIC_4K", "MAMBA2_GENERIC_1K"}
    # Qwen3.8-Max adds context/generation at TP 1/2/4/8/16.
    assert len(get_common_gdn_test_cases()) == 84
    mhc_cases = get_common_mhc_test_cases()
    # GLM-5.3-Flash adds one aliased (FP8/NVFP4) mHC row.
    assert len(mhc_cases) == 10
    assert {(case.model_name, case.architecture, case.phase, case.hidden_size, case.hc_mult) for case in mhc_cases} == {
        (model_name, architecture, phase, hidden_size, 4)
        for model_name, architecture, hidden_size in (
            ("deepseek-ai/DeepSeek-V4-Flash", "DeepseekV4ForCausalLM", 4096),
            ("sgl-project/DeepSeek-V4-Flash-FP8", "DeepseekV4ForCausalLM", 4096),
            ("deepseek-ai/DeepSeek-V4-Pro", "DeepseekV4ForCausalLM", 7168),
            ("sgl-project/DeepSeek-V4-Pro-FP8", "DeepseekV4ForCausalLM", 7168),
            ("zai-org/GLM-5.3-Flash", "Glm5NextForConditionalGeneration", 4096),
        )
        for phase in ("pre", "post")
    }
    assert {(case.phase, case.hidden_size, case.hc_mult) for case in mhc_cases} == {
        (phase, hidden_size, 4) for hidden_size in (4096, 7168) for phase in ("pre", "post")
    }


@pytest.mark.parametrize("op", ["mla_context", "mla_generation"])
def test_mla_plan_dedupes_on_loader_physical_keys(monkeypatch, op):
    """The perf key holds local heads, not total heads or TP: one case per key, at the smallest TP."""
    from collector.op_plans import plan

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    cases = plan(op, "trtllm", 100).cases

    keys = [(case.kv_cache_dtype, case.num_heads // case.tp_size, case.batch_size, case.input_len) for case in cases]
    assert len(keys) == len(set(keys))
    assert 1 in {case.num_heads // case.tp_size for case in cases}
    assert {case.kv_cache_dtype for case in cases} == {"bfloat16", "fp8"}


def test_kimi_mla_plan_includes_generation_bmm_helpers():
    required_ops = {"mla_context", "mla_generation", "mla_bmm_gen_pre", "mla_bmm_gen_post"}
    for backend in ("sglang", "trtllm"):
        plan = build_collection_case_plan(backend=backend, model_path="moonshotai/Kimi-K2.5")
        assert required_ops <= plan.selected_ops


def test_kimi_k3_vllm_plan_activates_mla_module_ops():
    vllm_plan = build_collection_case_plan(backend="vllm", model_path="moonshotai/Kimi-K3")
    assert {"mla_context_module", "mla_generation_module"} <= vllm_plan.selected_ops
    assert {"mla_context", "mla_generation"}.isdisjoint(vllm_plan.selected_ops)

    # Framework scoping: sglang keeps the granular MLA lane.
    sglang_plan = build_collection_case_plan(backend="sglang", model_path="moonshotai/Kimi-K3")
    assert {"mla_context_module", "mla_generation_module"}.isdisjoint(sglang_plan.selected_ops)
    assert {"mla_context", "mla_generation"} <= sglang_plan.selected_ops


def test_kimi_k3_scoped_run_activates_moe_for_sglang_and_vllm_only():
    # K3 has no trtllm serving lane. moe activation is framework-specific
    # (sglang/vllm), so a K3-scoped trtllm run plans NO moe at all — a
    # planned-op zero-case expansion with no logged drop is structurally
    # impossible (case_authoring.md; review).
    for backend in ("sglang", "vllm"):
        plan = build_collection_case_plan(backend=backend, model_path="moonshotai/Kimi-K3")
        assert "moe" in plan.selected_ops, backend
    trtllm_plan = build_collection_case_plan(backend="trtllm", model_path="moonshotai/Kimi-K3")
    assert "moe" not in trtllm_plan.selected_ops

    # A full-plan run plans K3's checkpoint modes on every backend.
    from collector.case_generator import moe_model_allows_quantization

    for backend in ("sglang", "trtllm", "vllm"):
        assert moe_model_allows_quantization(backend, "moonshotai/Kimi-K3", "w4a8_mxfp4_mxfp8"), backend


def test_module_prefix_context_sweeps_are_yaml_backed():
    from collector.case_generator import get_mla_module_sweep_spec

    assert get_mla_module_sweep_spec("plan").context_prefix_lengths == [
        0, 1, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288,
        16384, 32768, 65536, 131072, 262144, 524288, 1048575,
    ]  # fmt: skip
    assert 128 in get_mla_module_sweep_spec("sglang").context_prefix_lengths


def test_vllm_moe_quantization_metadata_is_yaml_backed():
    from collector.case_generator import get_moe_quantization_module_config, moe_model_allows_quantization

    assert [spec.name for spec in get_moe_quantization_specs("vllm")] == [
        "bfloat16",
        "fp8",
        "fp8_block",
        "int4_wo",
        "nvfp4",
        "w4a16_mxfp4",
        "w4a8_mxfp4_mxfp8",
    ]

    assert moe_model_allows_quantization("vllm", "openai/gpt-oss-20b", "w4a16_mxfp4")
    assert not moe_model_allows_quantization("vllm", "openai/gpt-oss-20b", "bfloat16")
    assert not moe_model_allows_quantization("vllm", "Qwen/Qwen3-235B-A22B", "w4a16_mxfp4")
    assert moe_model_allows_quantization("vllm", "Qwen/Qwen3-235B-A22B", "bfloat16")
    assert get_moe_quantization_module_config("vllm", "w4a16_mxfp4", model_name="openai/gpt-oss-20b") == {
        "has_bias": True,
        "activation": "swigluoai",
    }
    assert get_moe_quantization_module_config("vllm", "w4a16_mxfp4", model_name="Qwen/Qwen3-235B-A22B") == {}
    assert get_moe_quantization_module_config("vllm", "int4_wo", model_name="moonshotai/Kimi-K2.5") == {
        "group_size": 32
    }


def test_vllm_xpu_moe_metadata_is_yaml_backed(monkeypatch):
    from collector.case_generator import (
        get_moe_backend_model_activation,
        get_moe_backend_test_cases,
        get_moe_quantization_modes,
        moe_model_allows_quantization,
    )

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)

    cases = get_moe_backend_test_cases("vllm_xpu")

    assert len(cases) == 327
    assert {case.model_name for case in cases} == {
        "Qwen/Qwen1.5-MoE-A2.7B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "meta-llama/Llama-4-Scout-17B-16E",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    }
    assert not any(case.model_name == "Qwen/Qwen3-30B-A3B" and case.tp >= 8 for case in cases)
    assert get_moe_backend_model_activation("vllm_xpu", "openai/gpt-oss-20b") == "swigluoai"
    assert get_moe_backend_model_activation("vllm_xpu", "Qwen/Qwen1.5-MoE-A2.7B") == "silu"

    assert get_moe_quantization_modes("vllm_xpu", sm_version=0, runtime_features={}) == [
        "bfloat16",
        "w4a16_mxfp4",
    ]
    assert get_moe_quantization_modes(
        "vllm_xpu",
        sm_version=0,
        runtime_features={"torch_fp8_e4m3fn": True},
    ) == ["bfloat16", "fp8", "w4a16_mxfp4"]
    assert moe_model_allows_quantization("vllm_xpu", "openai/gpt-oss-20b", "w4a16_mxfp4")
    assert not moe_model_allows_quantization("vllm_xpu", "openai/gpt-oss-20b", "bfloat16")
    assert not moe_model_allows_quantization("vllm_xpu", "Qwen/Qwen1.5-MoE-A2.7B", "w4a16_mxfp4")

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "openai/gpt-oss-20b")
    targeted_cases = get_moe_backend_test_cases("vllm_xpu")
    assert targeted_cases
    assert {case.model_name for case in targeted_cases} == {"openai/gpt-oss-20b"}


def test_mla_bmm_cases_expand_from_base_op_yaml():
    from collector.case_generator import MLABMMCommonTestCase, get_mla_bmm_case_specs

    pre_cases = get_mla_bmm_case_specs("sglang", "mla_bmm_gen_pre")
    post_cases = get_mla_bmm_case_specs("sglang", "mla_bmm_gen_post")

    # 600/672 since the Kimi-K3 96-head family (96/48/24/12) joined the
    # base head_counts grid alongside the DeepSeek 128-family.
    assert len(pre_cases) == 600
    assert len(post_cases) == 672
    assert pre_cases[0] == MLABMMCommonTestCase(
        num_tokens=1,
        num_heads=128,
        dtype="bfloat16",
        num_warmups=2,
        num_runs=10,
    )
    assert pre_cases[1] == MLABMMCommonTestCase(
        num_tokens=1,
        num_heads=128,
        dtype="fp8",
        num_warmups=2,
        num_runs=10,
    )
    assert post_cases[-1] == MLABMMCommonTestCase(
        num_tokens=20480,
        num_heads=1,
        dtype="fp8",
        num_warmups=2,
        num_runs=10,
    )


def test_mla_module_metadata_and_micro_sweeps_are_yaml_backed():
    from collector.case_generator import (
        get_mla_module_model_specs,
        get_mla_module_precision_specs,
        get_mla_module_sweep_spec,
    )

    sweep = get_mla_module_sweep_spec()
    dsa_specs = get_mla_module_model_specs(attention_type="dsa", apply_model_filter=False)
    kimi_specs = get_mla_module_model_specs(
        attention_type="mla",
        backend="vllm",
        wideep_mla=False,
        apply_model_filter=False,
    )
    wideep_specs = get_mla_module_model_specs(attention_type="mla", wideep_mla=True, apply_model_filter=False)
    trtllm_specs = get_mla_module_model_specs(backend="trtllm")
    vllm_specs = get_mla_module_model_specs(backend="vllm")

    assert sweep.batch_sizes == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    assert sweep.sequence_lengths[-2:] == [8192, 16384]
    assert sweep.inner_sweep_head_counts == [128, 64, 32, 16, 8]
    assert sweep.top_level_head_counts == [128, 64, 32, 16, 8]
    assert sweep.module_precision_combos == [("bfloat16", "bfloat16", "bfloat16")]

    assert [
        (spec.compute_dtype, spec.kv_cache_dtype, spec.gemm_type)
        for spec in get_mla_module_precision_specs("sglang", phase="context", sm_version=90)
    ] == [
        ("bfloat16", "bfloat16", "bfloat16"),
        ("bfloat16", "fp8", "bfloat16"),
        ("bfloat16", "bfloat16", "fp8_block"),
        ("bfloat16", "fp8", "fp8_block"),
    ]
    assert get_mla_module_sweep_spec("sglang").context_sequence_lengths[-2:] == [8192, 16384]

    # FP8 prefill-query compute exists for the dense-MLA prefill path only: the
    # sparse DSA and MSA builders have no prefill-query quantization concept, so
    # the fp8 compute combos are scoped attention_types: [mla] and the DSA and
    # MSA plans do not expand them.
    for attention_type in ("dsa", "msa"):
        assert [
            (spec.compute_dtype, spec.kv_cache_dtype, spec.gemm_type)
            for spec in get_mla_module_precision_specs("plan", phase="context", attention_type=attention_type)
        ] == [
            ("bfloat16", "bfloat16", "bfloat16"),
            ("bfloat16", "fp8", "bfloat16"),
            ("bfloat16", "bfloat16", "fp8_block"),
            ("bfloat16", "fp8", "fp8_block"),
            ("bfloat16", "bfloat16", "nvfp4"),
            ("bfloat16", "fp8", "nvfp4"),
        ]
    assert [
        (spec.compute_dtype, spec.kv_cache_dtype, spec.gemm_type)
        for spec in get_mla_module_precision_specs("plan", phase="context", attention_type="mla")
    ] == [
        ("bfloat16", "bfloat16", "bfloat16"),
        ("bfloat16", "fp8", "bfloat16"),
        ("fp8", "fp8", "bfloat16"),
        ("bfloat16", "bfloat16", "fp8_block"),
        ("bfloat16", "fp8", "fp8_block"),
        ("fp8", "fp8", "fp8_block"),
        ("bfloat16", "bfloat16", "nvfp4"),
        ("bfloat16", "fp8", "nvfp4"),
        ("fp8", "fp8", "nvfp4"),
    ]

    with pytest.raises(ValueError, match="attention_type"):
        get_mla_module_precision_specs("plan", phase="context", sm_version=100, attention_type="dense")

    assert {spec.model_path for spec in dsa_specs} == {
        "deepseek-ai/DeepSeek-V3.2",
        "nvidia/DeepSeek-V3.2-NVFP4",
        "zai-org/GLM-5",
        "zai-org/GLM-5-FP8",
        "nvidia/GLM-5-NVFP4",
        "zai-org/GLM-5.1",
        "zai-org/GLM-5.1-FP8",
        "nvidia/GLM-5.1-NVFP4",
        "zai-org/GLM-5.2",
        "zai-org/GLM-5.2-FP8",
        "nvidia/GLM-5.2-NVFP4",
        "zai-org/GLM-5.3-BF16",
        "zai-org/GLM-5.3",
        "nvidia/GLM-5.3-NVFP4",
    }
    assert {spec.native_num_heads for spec in dsa_specs if spec.architecture == "GlmMoeDsaForCausalLM"} == {64}
    assert {(spec.model_path, spec.architecture, spec.native_num_heads) for spec in kimi_specs} == {
        ("moonshotai/Kimi-K2-Instruct", "DeepseekV3ForCausalLM", 64),
        ("moonshotai/Kimi-K2.5", "KimiK25ForConditionalGeneration", 64),
        ("nvidia/Kimi-K2.5-NVFP4", "KimiK25ForConditionalGeneration", 64),
        ("nvidia/Kimi-K2.6-NVFP4", "KimiK25ForConditionalGeneration", 64),
        ("nvidia/Kimi-K2.7-Code-NVFP4", "KimiK25ForConditionalGeneration", 64),
        ("moonshotai/Kimi-K3", "KimiK3ForConditionalGeneration", 96),
    }
    assert {spec.model_path for spec in wideep_specs} == {
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3",
        "nvidia/DeepSeek-V3.1-NVFP4",
    }
    assert {(spec.attention_type, spec.model_path, spec.architecture) for spec in vllm_specs} == {
        ("mla", "deepseek-ai/DeepSeek-V3", "DeepseekV3ForCausalLM"),
        ("dsa", "deepseek-ai/DeepSeek-V3.2", "DeepseekV32ForCausalLM"),
        ("dsa", "zai-org/GLM-5.2", "GlmMoeDsaForCausalLM"),
        ("msa", "MiniMaxAI/MiniMax-M3", "MiniMaxM3ForCausalLM"),
    }
    assert {(spec.attention_type, spec.model_path, spec.architecture) for spec in trtllm_specs} == {
        ("mla", "deepseek-ai/DeepSeek-V3", "DeepseekV3ForCausalLM"),
        ("dsa", "deepseek-ai/DeepSeek-V3.2", "DeepseekV32ForCausalLM"),
        ("dsa", "zai-org/GLM-5", "GlmMoeDsaForCausalLM"),
        ("msa", "MiniMaxAI/MiniMax-M3", "MiniMaxM3ForCausalLM"),
    }


def test_mla_module_targeted_artifacts_keep_requested_checkpoint(monkeypatch):
    from collector.case_generator import get_mla_module_model_specs

    for backend, model_path, attention_type, architecture in (
        ("trtllm", "nvidia/DeepSeek-V3.1-NVFP4", "mla", "DeepseekV3ForCausalLM"),
        ("trtllm", "nvidia/GLM-5-NVFP4", "dsa", "GlmMoeDsaForCausalLM"),
        ("vllm", "nvidia/DeepSeek-V3.1-NVFP4", "mla", "DeepseekV3ForCausalLM"),
        ("vllm", "moonshotai/Kimi-K2-Instruct", "mla", "DeepseekV3ForCausalLM"),
        ("vllm", "moonshotai/Kimi-K2.5", "mla", "KimiK25ForConditionalGeneration"),
        ("vllm", "nvidia/Kimi-K2.5-NVFP4", "mla", "KimiK25ForConditionalGeneration"),
        ("vllm", "nvidia/GLM-5-NVFP4", "dsa", "GlmMoeDsaForCausalLM"),
        ("vllm", "zai-org/GLM-5.2", "dsa", "GlmMoeDsaForCausalLM"),
        ("vllm", "zai-org/GLM-5.2-FP8", "dsa", "GlmMoeDsaForCausalLM"),
    ):
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
        specs = get_mla_module_model_specs(attention_type=attention_type, backend=backend)
        assert [(spec.model_path, spec.architecture) for spec in specs] == [(model_path, architecture)]


def test_vllm_dsa_reference_covers_native_full_and_reuse_without_duplicate_artifacts(monkeypatch):
    from collector.case_generator import get_mla_module_model_specs

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    specs = get_mla_module_model_specs(attention_type="dsa", backend="vllm")
    glm = [spec for spec in specs if spec.architecture == "GlmMoeDsaForCausalLM"]
    assert [spec.model_path for spec in glm] == ["zai-org/GLM-5.2"]
    config_path = REPO_ROOT / "src/aisimulate_core/model_configs/zai-org--GLM-5.2_config.json"
    config = json.loads(config_path.read_text())
    # The native layer-zero rule executes a full indexer, while this
    # checkpoint declares reuse for later layers; both rows must be sampled.
    assert config["index_topk_freq"] > 1
    assert max(0 - config["index_skip_topk_offset"] + 1, 0) % config["index_topk_freq"] == 0
    all_specs = get_mla_module_model_specs(attention_type="dsa", backend="vllm", apply_model_filter=False)
    assert len(all_specs) == len(set(all_specs))
    # GLM-5.3 reverses the zai-org naming: the plain repository is the FP8
    # release and the BF16 weights carry -BF16 (no zai-org/GLM-5.3-FP8 exists).
    assert {s.model_path for s in all_specs if s.architecture == "GlmMoeDsaForCausalLM"} == {
        f"{prefix}/GLM-{version}{suffix}"
        for version in ("5", "5.1", "5.2")
        for prefix, suffix in (("zai-org", ""), ("zai-org", "-FP8"), ("nvidia", "-NVFP4"))
    } | {"zai-org/GLM-5.3", "zai-org/GLM-5.3-BF16", "nvidia/GLM-5.3-NVFP4"}


def test_vllm_mla_module_artifacts_have_local_configs():
    from collector.case_generator import get_mla_module_model_specs

    config_root = REPO_ROOT / "src" / "aisimulate_core" / "model_configs"
    for spec in get_mla_module_model_specs(backend="vllm", apply_model_filter=False):
        config_path = config_root / f"{spec.model_path.replace('/', '--')}_config.json"
        assert config_path.is_file(), f"{spec.model_path} would require a runtime Hub download"


def test_shape_only_mla_alias_uses_canonical_model(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "moonshotai/Kimi-K2-Instruct")
    from collector.case_generator import get_context_mla_case_specs

    kimi_specs = get_context_mla_case_specs()
    assert kimi_specs
    assert {spec.model_name for spec in kimi_specs} == {"moonshotai/Kimi-K2.5"}
    assert {spec.num_heads for spec in kimi_specs} == {64, 128}


def test_model_cases_path_can_infer_model_path():
    model_cases_path = default_architecture_cases_path("DeepseekV4ForCausalLM")

    plan = build_collection_case_plan(backend="sglang", model_cases_path=str(model_cases_path))

    assert plan.model_path == "sgl-project/DeepSeek-V4-Flash-FP8"
    assert plan.model_architecture == "DeepseekV4ForCausalLM"
    assert "dsv4_csa_context_module" in plan.selected_ops
    assert "dsv4_csa_topk_calib" in plan.selected_ops
    assert "mhc_module" in plan.selected_ops
    # the SDK's only live sparse sidecar (CP chunk walk) is planned; the FMLA sidecars
    # (hca_attn / csa_attn) were removed from the registry
    assert "dsv4_paged_mqa_logits_module" in plan.selected_ops
    assert {"dsv4_hca_attn_module", "dsv4_csa_attn_module"}.isdisjoint(plan.selected_ops)


def test_plan_rejects_model_declared_ops_unknown_to_backend_registry(tmp_path):
    """A typo in a model-declared op name must fail plan building loudly, not
    silently collect nothing for the intended benchmark."""
    case_file = tmp_path / "FakeArchForCausalLM_cases.yaml"
    case_file.write_text(
        "architecture: FakeArchForCausalLM\n"
        "model_path: fake/model\n"
        "model_ops:\n"
        "  - attention_context\n"
        "  - attention_contxt_typo\n"
    )

    with pytest.raises(ValueError, match="attention_contxt_typo"):
        build_collection_case_plan(backend="vllm", model_cases_path=str(case_file))


def test_dsv4_plan_only_uses_backend_specific_case_plan():
    model_path = "deepseek-ai/DeepSeek-V4-Pro"
    expected_ops = build_collection_case_plan(backend="sglang", model_path=model_path).ops

    result = subprocess.run(
        [
            sys.executable,
            "collector/collect.py",
            "--backend",
            "sglang",
            "--model-path",
            model_path,
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ops"] == expected_ops
    assert "dsv4_csa_topk_calib" in payload["ops"]
    assert "moe_ep" not in payload["ops"]


def test_vllm_024_schedules_consumed_dsv4_modules_only():
    from collector.vllm.registry import REGISTRY

    consumed_dsv4_ops = {
        "dsv4_csa_context_module",
        "dsv4_hca_context_module",
        "dsv4_csa_generation_module",
        "dsv4_hca_generation_module",
    }
    removed_sidecar_ops = {"dsv4_hca_attn_module", "dsv4_csa_attn_module"}  # removed 2026-10-15
    plan = build_collection_case_plan(backend="vllm", model_path="sgl-project/DeepSeek-V4-Pro-FP8")

    assert plan.ops == [
        "dsv4_csa_context_module",
        "dsv4_csa_generation_module",
        "dsv4_hca_context_module",
        "dsv4_hca_generation_module",
        "dsv4_paged_mqa_logits_module",  # consumed by the SDK CP chunk walk; shipped for vllm 0.24/0.25
        "gemm",
        # No mhc_module: collector/skips.py leaves DeepSeek-V4 out of vLLM's mHC table.
        "moe",
    ]
    registered_ops = {entry.op for entry in REGISTRY}
    assert consumed_dsv4_ops <= plan.selected_ops
    assert consumed_dsv4_ops <= registered_ops
    assert removed_sidecar_ops.isdisjoint(registered_ops)


def test_model_architecture_can_select_case_file():
    plan = build_collection_case_plan(backend="trtllm", model_architecture="Qwen3MoeForCausalLM")

    assert plan.model_path == "Qwen/Qwen3-235B-A22B"
    assert plan.model_architecture == "Qwen3MoeForCausalLM"
    assert plan.model_cases_paths == [default_architecture_cases_path("Qwen3MoeForCausalLM")]
    assert "moe" in plan.selected_ops
    assert "mla_module" not in plan.selected_ops


def test_model_path_alias_resolves_architecture_case_file():
    plan = build_collection_case_plan(backend="trtllm", model_path="Qwen/Qwen3-235B-A22B-FP8")

    assert plan.model_path == "Qwen/Qwen3-235B-A22B-FP8"
    assert plan.model_architecture == "Qwen3MoeForCausalLM"
    assert plan.model_cases_paths == [default_architecture_cases_path("Qwen3MoeForCausalLM")]
    assert "moe" in plan.selected_ops


def test_encoder_attention_plan_matches_sdk_model_and_backend_support():
    dense_plan = build_collection_case_plan(backend="sglang", model_path="Qwen/Qwen3-32B")
    assert dense_plan.ops == ["attention_context", "attention_generation", "gemm"]
    assert not dense_plan.has_op("encoder_attention")

    for backend in ("sglang", "trtllm", "vllm"):
        assert build_collection_case_plan(
            backend=backend,
            model_path="Qwen/Qwen3-VL-32B-Instruct",
        ).has_op("encoder_attention")
    assert not build_collection_case_plan(
        backend="vllm_xpu",
        model_path="Qwen/Qwen3-VL-32B-Instruct",
    ).has_op("encoder_attention")


def test_vllm_024_model_plans_only_schedule_representable_attention_paths():
    kimi_path = "moonshotai/Kimi-K2.5"
    assert build_collection_case_plan(backend="vllm", model_path=kimi_path).ops == [
        "encoder_attention",
        "gemm",
        "mla_context_module",
        "mla_generation_module",
        "moe",
    ]
    assert build_collection_case_plan(backend="vllm", model_path="moonshotai/Kimi-K2-Instruct").ops == [
        "gemm",
        "mla_context_module",
        "mla_generation_module",
        "moe",
    ]
    assert build_collection_case_plan(backend="sglang", model_path=kimi_path).ops == [
        "gemm",
        "mla_bmm_gen_post",
        "mla_bmm_gen_pre",
        "mla_context",
        "mla_generation",
        "moe",
    ]
    # TRT-LLM 1.3.0rc20 serves the full K2.5 VLM including MoonViT3d, so the
    # trtllm plan carries the vision encoder_attention profile like vLLM.
    assert build_collection_case_plan(backend="trtllm", model_path=kimi_path).ops == [
        "encoder_attention",
        "gemm",
        "mla_bmm_gen_post",
        "mla_bmm_gen_pre",
        "mla_context",
        "mla_generation",
        "moe",
    ]  # moe_ep left the default trtllm plan with the separate WideEP pin
    assert build_collection_case_plan(backend="vllm_xpu", model_path=kimi_path).ops == ["gemm", "moe"]

    models_with_unrepresentable_vllm_attention = (
        "XiaomiMiMo/MiMo-V2-Flash",
        "google/gemma-4-26B-A4B",
        "openai/gpt-oss-120b",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    )
    legacy_plan = ["attention_context", "attention_generation", "gemm", "moe"]
    for model_path in models_with_unrepresentable_vllm_attention:
        assert build_collection_case_plan(backend="vllm", model_path=model_path).ops == ["gemm", "moe"]
        assert build_collection_case_plan(backend="vllm_xpu", model_path=model_path).ops == ["gemm", "moe"]
        assert build_collection_case_plan(backend="sglang", model_path=model_path).ops == legacy_plan
        # collector/skips.py: TRT-LLM cannot serve MiMo-V2-Flash, so a model run plans nothing.
        expected_trtllm = [] if model_path == "XiaomiMiMo/MiMo-V2-Flash" else legacy_plan
        assert build_collection_case_plan(backend="trtllm", model_path=model_path).ops == expected_trtllm


def test_compute_scale_is_selected_only_for_static_fp8_artifact():
    static_model = "Qwen/Qwen3-32B-FP8-Static-PerTensor"
    non_static_models = ("Qwen/Qwen3-32B", "Qwen/Qwen3-32B-FP8", "Qwen/Qwen3-0.6B")

    for backend in ("sglang", "trtllm", "vllm"):
        static_plan = build_collection_case_plan(backend=backend, model_path=static_model, sm_version=100)
        assert "compute_scale" in static_plan.selected_ops
        assert "compute_scale" in build_collection_case_plan(backend=backend, full=True).selected_ops

        for model_path in non_static_models:
            plan = build_collection_case_plan(backend=backend, model_path=model_path, sm_version=100)
            assert "compute_scale" not in plan.selected_ops

    xpu_plan = build_collection_case_plan(backend="vllm_xpu", model_path=static_model)
    assert "compute_scale" not in xpu_plan.selected_ops


def test_model_plans_do_not_request_ops_missing_from_backend_registry():
    deepseek_vllm = build_collection_case_plan(backend="vllm", model_path="deepseek-ai/DeepSeek-V3")
    kimi_vllm = build_collection_case_plan(backend="vllm", model_path="moonshotai/Kimi-K2.5")
    nemotron_sglang = build_collection_case_plan(
        backend="sglang",
        model_path="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    )
    nemotron_trtllm = build_collection_case_plan(
        backend="trtllm",
        model_path="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    )

    assert not deepseek_vllm.has_op("mla_context")
    assert not deepseek_vllm.has_op("mla_generation")
    assert deepseek_vllm.has_op("mla_context_module")
    assert not kimi_vllm.has_op("attention_context")
    assert not kimi_vllm.has_op("attention_generation")
    assert kimi_vllm.has_op("encoder_attention")
    assert kimi_vllm.has_op("mla_context_module")
    assert kimi_vllm.has_op("mla_generation_module")
    assert not kimi_vllm.has_op("mla_context")
    assert not kimi_vllm.has_op("mla_generation")
    assert not nemotron_sglang.has_op("mamba2")
    assert nemotron_trtllm.has_op("mamba2")


def test_full_mode_aggregates_all_model_case_files():
    plan = build_collection_case_plan(backend="sglang", full=True)

    assert plan.model_path is None
    assert len(plan.model_cases_paths) >= 18
    assert "wideep_mla_context" not in plan.selected_ops
    assert "dsv4_csa_context_module" in plan.selected_ops
    assert "gdn" in plan.selected_ops


def test_full_mode_ops_are_a_union_of_model_plan_ops():
    for backend in ("sglang", "trtllm"):
        full_plan = build_collection_case_plan(backend=backend, full=True)
        for model_path in ("deepseek-ai/DeepSeek-V3", "moonshotai/Kimi-K2.5", "Qwen/Qwen3-32B"):
            model_plan = build_collection_case_plan(backend=backend, model_path=model_path)
            assert model_plan.selected_ops <= full_plan.selected_ops, f"{backend}/{model_path}"


def test_mla_module_metadata_canonicalizes_consumer_keyed_backends():
    from collector.case_generator import get_mla_module_model_specs

    original_artifacts = {
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1",
        "nvidia/DeepSeek-V3.1-NVFP4",
    }

    def paths(backend):
        return {
            spec.model_path
            for spec in get_mla_module_model_specs(
                attention_type="mla",
                backend=backend,
                apply_model_filter=backend in {"trtllm", "vllm"},
            )
        }

    assert paths("vllm") == {"deepseek-ai/DeepSeek-V3"}
    assert paths("sglang") == original_artifacts
    assert paths("trtllm") == {"deepseek-ai/DeepSeek-V3"}


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
def test_dsa_sub_8_head_cells_leave_the_plan_on_sm100_and_sm103(backend):
    """b200 2026-10-06 (jobs 471043976 / 469988228): per-rank heads 1/2/4 have no sparse-MLA
    kernel on SM100/103 and failed 68,435 + 15,408 cases after a full module load each.
    A skips.py entry leaves them out there; SM90 runs FlashMLA and keeps them."""
    from collector.op_plans import plan

    for op in ("dsa_context_module", "dsa_generation_module"):
        # GLM-5.x at TP16 runs 4 heads per rank.
        assert 4 in {case.num_heads for case in plan(op, backend, 90).cases}
        for sm in (100, 103):
            assert min(case.num_heads for case in plan(op, backend, sm).cases) >= 8


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
def test_module_plan_keeps_only_the_heads_of_declared_tensor_parallel_shards(backend):
    """A model row that declares tensor_parallel_sizes (case_authoring model-correlated
    narrowing, as the MSA rows do) measures only native_num_heads // tp; the shared head
    grid's other entries (96/48/24/12 for the 96-head Kimi-K3 MLA module, 128 for a 64-head
    model) were 40-50% of the b200 vllm dsa_context rows."""
    from collector.op_plans import plan

    by_model = {}
    for case in plan("dsa_context_module", backend, 90).cases:
        by_model.setdefault(case.model_path, set()).add(case.num_heads)
    assert by_model["zai-org/GLM-5.2"] == {64, 32, 16, 8, 4}
    assert by_model["deepseek-ai/DeepSeek-V3.2"] == {128, 64, 32, 16, 8}


def test_support_matrix_models_have_model_case_aliases():
    case_aliases = set()
    for path in (REPO_ROOT / "collector" / "cases" / "models").glob("*_cases.yaml"):
        data = path.read_text(encoding="utf-8")
        for line in data.splitlines():
            stripped = line.strip()
            if stripped.startswith("model_path: "):
                case_aliases.add(stripped.removeprefix("model_path: ").strip())
            elif stripped.startswith("- "):
                case_aliases.add(stripped.removeprefix("- ").strip())

    support_matrix_models = set()
    for path in SUPPORT_MATRIX_ROOT.glob("*.csv"):
        with path.open(encoding="utf-8") as f:
            support_matrix_models.update(row["HuggingFaceID"] for row in csv.DictReader(f))

    assert support_matrix_models <= case_aliases


def test_support_matrix_moe_alias_generates_targeted_cases(monkeypatch):
    from collector.case_generator import get_common_moe_test_cases

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "Qwen/Qwen3-235B-A22B-FP8")

    cases = get_common_moe_test_cases()

    assert cases
    assert {case.model_name for case in cases} == {"Qwen/Qwen3-235B-A22B"}


def test_qwen3_30b_fp8_alias_reuses_canonical_case_and_tp_constraints(monkeypatch):
    from collector.case_generator import get_common_moe_test_cases

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "Qwen/Qwen3-30B-A3B-FP8")

    cases = get_common_moe_test_cases()

    assert cases
    assert {case.model_name for case in cases} == {"Qwen/Qwen3-30B-A3B"}
    assert all(case.tp < 8 for case in cases)


def test_quant_sensitive_moe_artifacts_use_quant_equivalent_representatives(monkeypatch):
    from collector.case_generator import get_common_moe_test_cases

    expected_representatives = {
        "nvidia/DeepSeek-V3.1-NVFP4": "nvidia/DeepSeek-V3.1-NVFP4",
        "nvidia/MiniMax-M2.5-NVFP4": "nvidia/MiniMax-M2.5-NVFP4",
        "nvidia/MiniMax-M2.7-NVFP4": "nvidia/MiniMax-M2.5-NVFP4",
        "nvidia/MiniMax-M3-NVFP4": "nvidia/MiniMax-M3-NVFP4",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    }

    for model_path, expected_representative in expected_representatives.items():
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
        cases = get_common_moe_test_cases()
        assert cases and {case.model_name for case in cases} == {expected_representative}


def test_nemotron_super_fp8_vllm_moe_case_covers_missing_consumer_key(monkeypatch):
    from collector.case_generator import get_common_moe_test_cases

    model_path = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)

    cases = get_common_moe_test_cases()

    assert cases
    assert {case.model_name for case in cases} == {model_path}
    assert any(
        case.hidden_size == 1024
        and case.inter_size == 2688
        and case.topk == 22
        and case.num_experts == 512
        and case.tp == 1
        and case.ep == 4
        and case.token_expert_distribution == "power_law"
        and case.power_law_alpha == 1.01
        for case in cases
    )
    assert moe_model_allows_quantization("vllm", model_path, "fp8")
    assert not moe_model_allows_quantization("vllm", model_path, "bfloat16")

    config_path = REPO_ROOT / "src/aisimulate_core/model_configs" / f"{model_path.replace('/', '--')}_config.json"
    assert config_path.is_file()


def test_qwen35_397b_nvfp4_moe_cases_are_declared_with_correct_shape_and_runner(monkeypatch):
    from collector.case_generator import (
        get_common_moe_test_cases,
        get_sglang_moe_backend,
        moe_model_allows_quantization,
    )

    # The full-roster case counts below assume no model filter: an inherited
    # COLLECTOR_MODEL_PATH would silently narrow the expansion.
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)

    cases = get_common_moe_test_cases()
    nvfp4_cases = [case for case in cases if case.model_name == "nvidia/Qwen3.5-397B-A17B-NVFP4"]

    # Row exists and carries the 397B shape tuple. The count itself is
    # re-derived, not hardcoded from the model YAML's own "+117" comment: the
    # NVFP4 row declares the IDENTICAL shape tuple (4096/1024, topk10, 512
    # experts) as the bf16/fp8_block Qwen/Qwen3.5-397B-A17B row, so the
    # shared moe.yaml sweep grid (tp/ep combos x token-count x workload
    # distribution, filtered by that one shape) must expand to the exact
    # same case count for both -- 117, per
    # test_cross_model_common_cases_expand_from_base_op_yaml_sweeps's own
    # "+117 for nvidia/Qwen3.5-397B-A17B-NVFP4" delta comment.
    base_397b_cases = [case for case in cases if case.model_name == "Qwen/Qwen3.5-397B-A17B"]
    assert nvfp4_cases, "nvidia/Qwen3.5-397B-A17B-NVFP4 moe cases not found"
    assert len(nvfp4_cases) == len(base_397b_cases) == 117, (
        f"nvfp4 case count must match the bf16/fp8_block 397B row's identical-shape expansion; "
        f"got {len(nvfp4_cases)} nvfp4 vs {len(base_397b_cases)} base"
    )
    assert all(case.hidden_size == 4096 for case in nvfp4_cases)
    assert all(case.inter_size == 1024 for case in nvfp4_cases)
    assert all(case.topk == 10 for case in nvfp4_cases)
    assert all(case.num_experts == 512 for case in nvfp4_cases)

    # Runner map resolves flashinfer_trtllm at sm100 and sm103
    sample = nvfp4_cases[0]
    assert get_sglang_moe_backend(sample, "nvfp4", 100) == "flashinfer_trtllm"
    assert get_sglang_moe_backend(sample, "nvfp4", 103) == "flashinfer_trtllm"

    # Quant policy: nvfp4 allowed for sglang; bfloat16 and fp8_block excluded
    assert moe_model_allows_quantization("sglang", "nvidia/Qwen3.5-397B-A17B-NVFP4", "nvfp4")
    assert not moe_model_allows_quantization("sglang", "nvidia/Qwen3.5-397B-A17B-NVFP4", "bfloat16")
    assert not moe_model_allows_quantization("sglang", "nvidia/Qwen3.5-397B-A17B-NVFP4", "fp8_block")


def test_qwen38_max_nvfp4_moe_cases_are_declared_with_correct_shape_and_runner(monkeypatch):
    from collector.case_generator import (
        get_common_moe_test_cases,
        get_sglang_moe_backend,
        moe_model_allows_quantization,
    )

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    cases = get_common_moe_test_cases()
    nvfp4_cases = [case for case in cases if case.model_name == "RadixArk/Qwen3.8-2.4T-A95B-NVFP4"]
    base_cases = [case for case in cases if case.model_name == "Qwen/Qwen3.8-2.4T-A95B"]

    assert nvfp4_cases, "RadixArk/Qwen3.8-2.4T-A95B-NVFP4 moe cases not found"
    assert len(nvfp4_cases) == len(base_cases) == 117
    assert all(
        case.hidden_size == 8192 and case.inter_size == 2048 and case.topk == 10 and case.num_experts == 512
        for case in nvfp4_cases
    )

    sample = nvfp4_cases[0]
    assert get_sglang_moe_backend(sample, "nvfp4", 100) == "flashinfer_trtllm"
    assert get_sglang_moe_backend(sample, "nvfp4", 103) == "flashinfer_trtllm"

    base_sample = base_cases[0]
    assert get_sglang_moe_backend(base_sample, "bfloat16", 100) == "triton"
    assert get_sglang_moe_backend(base_sample, "fp8_block", 100) == "flashinfer_trtllm"

    assert moe_model_allows_quantization("sglang", "RadixArk/Qwen3.8-2.4T-A95B-NVFP4", "nvfp4")
    assert not moe_model_allows_quantization("sglang", "RadixArk/Qwen3.8-2.4T-A95B-NVFP4", "bfloat16")
    assert not moe_model_allows_quantization("sglang", "RadixArk/Qwen3.8-2.4T-A95B-NVFP4", "fp8_block")
    assert moe_model_allows_quantization("sglang", "Qwen/Qwen3.8-2.4T-A95B", "bfloat16")
    assert moe_model_allows_quantization("sglang", "Qwen/Qwen3.8-2.4T-A95B", "fp8_block")
    assert not moe_model_allows_quantization("sglang", "Qwen/Qwen3.8-2.4T-A95B", "nvfp4")


def test_qwen38_max_plan_only_schedules_attention_on_compatible_collector():
    model_path = "Qwen/Qwen3.8-2.4T-A95B"
    attention_ops = {"attention_context", "attention_generation"}

    for backend in ("sglang", "vllm"):
        plan = build_collection_case_plan(backend=backend, model_path=model_path)
        assert attention_ops.isdisjoint(plan.selected_ops), (backend, plan.selected_ops)

    trtllm_plan = build_collection_case_plan(backend="trtllm", model_path=model_path)
    assert attention_ops <= trtllm_plan.selected_ops


@pytest.mark.parametrize(
    ("model_path", "expected"),
    [
        ("Qwen/Qwen3.8-2.4T-A95B", {"bfloat16", "fp8_block"}),
        ("RadixArk/Qwen3.8-2.4T-A95B-NVFP4", {"nvfp4"}),
    ],
)
def test_qwen38_max_moe_rows_plan_their_checkpoint_modes(model_path, expected):
    available_modes = {spec.name for spec in get_moe_quantization_specs()}
    for backend in ("sglang", "trtllm", "vllm"):
        allowed = {mode for mode in available_modes if moe_model_allows_quantization(backend, model_path, mode)}
        assert allowed == expected, (backend, model_path, allowed)


def test_nemotron_ultra_quant_artifact_keeps_moe_path_but_reuses_mamba_profile(monkeypatch):
    from collector.case_generator import get_common_mamba2_test_cases, get_common_moe_test_cases

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8")

    moe_cases = get_common_moe_test_cases()
    mamba_cases = get_common_mamba2_test_cases()

    assert moe_cases and {case.model_name for case in moe_cases} == {"nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8"}
    assert {case.hidden_size for case in moe_cases} == {2048, 8192}
    assert mamba_cases and {case.model_name for case in mamba_cases} == {
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
    }


def test_unverified_nemotron_rl_artifact_has_no_moe_profile(monkeypatch):
    from collector.case_generator import get_common_mamba2_test_cases, get_common_moe_test_cases

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/nemotron-ultra-rl-050826")

    assert get_common_moe_test_cases() == []
    assert get_common_mamba2_test_cases()


def test_support_matrix_mamba_alias_generates_targeted_cases(monkeypatch):
    from collector.case_generator import get_common_mamba2_test_cases

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")

    cases = get_common_mamba2_test_cases()

    assert cases
    assert {case.model_name for case in cases} == {"nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"}


def test_collector_case_yaml_numeric_lists_are_sorted():
    def is_number(value):
        return isinstance(value, int | float) and not isinstance(value, bool)

    def walk_numeric_lists(value, path):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield from walk_numeric_lists(nested, (*path, str(key)))
        elif isinstance(value, list):
            # Correlated history triples are coordinates, not sortable sweep axes.
            if len(path) >= 3 and path[-3] == "long_context_cases" and path[-2] in {"context", "generation"}:
                return
            if len(value) > 1 and all(is_number(item) for item in value):
                yield path, value
            for index, nested in enumerate(value):
                yield from walk_numeric_lists(nested, (*path, str(index)))

    violations = []
    for path in sorted((REPO_ROOT / "collector" / "cases").glob("**/*.yaml")):
        for yaml_path, values in walk_numeric_lists(load_yaml_file(path), ()):
            adjacent_values = list(pairwise(values))
            ascending = all(left <= right for left, right in adjacent_values)
            descending = all(left >= right for left, right in adjacent_values)
            if not (ascending or descending):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{'.'.join(yaml_path)} = {values}")

    assert violations == []


def test_step3p7_plans_correlated_attention_topologies(monkeypatch):
    """Step-3.7 must plan (64 Q, window 0) and (96 Q, window 512), not their cross-product.

    The pinned vLLM Step3p5 block substitutes 96 query heads on
    sliding_attention layers while global layers keep 64. A single row
    cross-producting num_attention_heads=64 with window_sizes [0, 512] plans
    64-head SWA cases that never run and omits the 96-head SWA cases that do --
    without changing the total case count, so an aggregate assertion misses it.
    """
    from collector.case_generator import get_attention_context_shape_sweeps

    # Attention is shape-only, so both artifacts intentionally share the rows.
    for model_path in ("stepfun-ai/Step-3.7-Flash-FP8", "stepfun-ai/Step-3.7-Flash"):
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
        configs = {
            (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
            for sweep in get_attention_context_shape_sweeps("vllm")
            for config in get_attention_head_configs(sweep, phase="context")
        }

        assert configs == {
            # global attention: 64 Q / 8 KV, sharded over TP 1/2/4/8
            (64, 8, 128, 0),
            (32, 4, 128, 0),
            (16, 2, 128, 0),
            (8, 1, 128, 0),
            # sliding attention (window 512): 96 Q / 8 KV
            (96, 8, 128, 512),
            (48, 4, 128, 512),
            (24, 2, 128, 512),
            (12, 1, 128, 512),
        }, model_path


def test_qwen35_gemm_model_rows_add_exact_below_grid_widths():
    """model_case_values.gemm supplies exact widths under the base feature grid
    (scalar expert gate n=1, GDN b/a projections); token density comes from the
    base sweeps and cases dedupe on the physical (x, n, k) tuple."""
    specs = get_gemm_case_specs()
    shapes = {(case.n, case.k) for case in specs}
    assert {(1, 2048), (8, 2048), (16, 4096), (12, 5120)} <= shapes

    base_tokens = set()
    for sweep in load_yaml_file(BASE_OP_CASES_DIR / "gemm.yaml")["all_frameworks_op_cases"]["gemm"]["cases"]:
        base_tokens.update(int(token) for token in sweep["token_counts"])
    assert {case.x for case in specs if (case.n, case.k) == (1, 2048)} == base_tokens

    assert len(specs) == len({(case.x, case.n, case.k) for case in specs})


def test_vllm_msa_persist_row_raises_when_log_perf_fails():
    """A false return from log_perf (lock exhaustion / write failure) must
    fail the case: a worker that returns normally lets the checkpoint advance
    with no row persisted, silently shrinking the dataset. Mirrors the
    TRT-LLM/SGLang collectors' behavior."""
    source_path = REPO_ROOT / "collector/vllm/collect_msa_module.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_persist_msa_row")

    def run(log_perf_result):
        calls = {}

        def fake_log_perf(**kwargs):
            calls.update(kwargs)
            return log_perf_result

        namespace = {"log_perf": fake_log_perf}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source_path), "exec"), namespace)
        namespace["_persist_msa_row"](
            item={"latency": "1.0"},
            vllm_version="0.24.0",
            device_name="test-device",
            op_name="msa_generation_module",
            kernel_source="MiniMaxM3SparseTritonImpl",
            perf_filename="msa_generation_module_perf.txt",
            power_stats=None,
        )
        return calls

    calls = run(True)
    assert calls["perf_filename"] == "msa_generation_module_perf.txt"

    with pytest.raises(RuntimeError, match="failed to persist MSA row"):
        run(False)


def test_nvfp4_checkpoint_targets_msa_module_specs(monkeypatch):
    """Exact-targeting the advertised NVFP4 artifact must populate the MSA
    plan for every backend (review 4969690316 Spec-2: a declared artifact
    expanding to zero cases with no logged drop is a population bug). The
    artifact aliases onto the canonical MiniMax-M3 MSA row — the module
    benchmark is artifact-insensitive (dummy weights, identical sparse
    geometry, precision as explicit sweep axes)."""
    from collector.case_generator import get_mla_module_model_specs

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/MiniMax-M3-NVFP4")
    for backend in ("trtllm", "vllm", "sglang"):
        specs = get_mla_module_model_specs("msa", backend=backend)
        assert specs, f"NVFP4 exact-targeting produced zero MSA specs for {backend}"
        assert all(s.model_path == "MiniMaxAI/MiniMax-M3" for s in specs), (
            "alias rows must stay keyed to the canonical model path"
        )
