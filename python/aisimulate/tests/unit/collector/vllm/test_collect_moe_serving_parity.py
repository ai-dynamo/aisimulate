# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free checks of the serving-parity rules collector/vllm/collect_moe.py
applies at vLLM 0.32.0: Kimi-K3 routing and TP padding, the FlashInfer
autotune bucket wiring and the non-gated ckpt_names. Citations live in the
collector."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
COLLECT_MOE = REPO_ROOT / "collector" / "vllm" / "collect_moe.py"
K3 = "moonshotai/Kimi-K3"


@pytest.fixture
def collect_moe(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    version = types.ModuleType("vllm.version")
    version.__version__ = "0.32.0"
    monkeypatch.setitem(sys.modules, "vllm.version", version)
    name = "collector.vllm.collect_moe"
    monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(name, COLLECT_MOE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_kimi_k3_routes_with_sigmoid_and_native_situ(collect_moe):
    config = collect_moe._resolve_moe_runtime_config(K3, {})
    # KimiMoE: scoring_func=config.moe_router_activation_func, renormalize=config.moe_renormalize.
    assert config["scoring_func"] == "sigmoid"
    assert config["renormalize"] is True
    assert (config["use_grouped_topk"], config["num_expert_group"], config["topk_group"]) == (True, 1, 1)
    assert config["use_routing_bias"] is True and config["router_logits_float32"] is True
    assert config["activation"] == "situ"
    assert (config["activation_situ_beta"], config["activation_situ_linear_beta"]) == (4.0, 25.0)
    assert config["min_moe_intermediate_per_partition"] == 256


def test_only_kimi_k3_declares_an_intermediate_floor(collect_moe):
    for model in ("deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-235B-A22B", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"):
        assert collect_moe._resolve_moe_runtime_config(model, {})["min_moe_intermediate_per_partition"] is None


@pytest.mark.parametrize(
    ("tp", "ep", "expected"),
    [
        (1, 1, 3072),
        (8, 1, 3072),  # 384 per rank, above the floor
        (16, 1, 4096),  # 192 per rank -> 256 per rank
        (32, 1, 8192),  # 96 per rank -> 256 per rank
        (1, 32, 3072),  # EP keeps whole experts per rank
    ],
)
def test_kimi_k3_pads_pure_tp_shards_like_serving(collect_moe, tp, ep, expected):
    assert collect_moe._moe_intermediate_size(3072, tp, ep, 256) == expected


def test_models_without_a_floor_keep_their_intermediate_size(collect_moe):
    assert collect_moe._moe_intermediate_size(192, 32, 1, None) == 192


def test_moe_tunes_at_the_bucket_of_vllms_warmup_list():
    """The bucket mapping itself is covered in test_helper_flashinfer_autotune.py;
    pin that the GPU-only loop asks for it with vLLM's warmup ceiling and the
    experts' own clamp, on SM90+ only."""
    source = COLLECT_MOE.read_text(encoding="utf-8")
    assert "serving_autotunes = has_flashinfer() and current_platform.has_device_capability(90)" in source
    assert "max_tuned_tokens = vllm_config.scheduler_config.max_num_batched_tokens" in source
    assert "tune_max_num_tokens=_flashinfer_moe_tune_max_num_tokens(leaf_experts)" in source


def _experts(name, **attrs):
    return type(name, (), attrs)()


@pytest.fixture
def fake_fi_moe_largest_bucket(monkeypatch):
    """vllm fused_moe/utils.py fi_moe_largest_bucket at v0.32.0: max(max_num_tokens * dp, 8192)."""
    utils = types.ModuleType("vllm.model_executor.layers.fused_moe.utils")
    utils.fi_moe_largest_bucket = lambda config: max(config.max_num_tokens * config.dp_size, 8192)
    for name in ("vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.fused_moe"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, utils.__name__, utils)


@pytest.mark.parametrize("name", ["FlashInferExperts", "TrtLlmBf16ExpertsModular", "TrtLlmMxint4ExpertsMonolithic"])
def test_experts_without_a_tune_max_run_flashinfers_8192_tactic_above_it(collect_moe, name):
    assert collect_moe._flashinfer_moe_tune_max_num_tokens(_experts(name)) == 8192


@pytest.mark.parametrize(
    "name",
    [
        "TrtLlmBf16ExpertsMonolithic",
        "TrtLlmFp8ExpertsModular",
        "TrtLlmFp8ExpertsMonolithic",
        "TrtLlmMxfp4ExpertsMonolithic",
        "TrtLlmMxfp4ExpertsModular",
        "TrtLlmNvFp4ExpertsMonolithic",
    ],
)
def test_trtllm_gen_experts_clamp_at_their_largest_bucket(collect_moe, fake_fi_moe_largest_bucket, name):
    config = SimpleNamespace(max_num_tokens=65536, dp_size=1)
    assert collect_moe._flashinfer_moe_tune_max_num_tokens(_experts(name, moe_config=config)) == 65536
    small = SimpleNamespace(max_num_tokens=2048, dp_size=1)
    assert collect_moe._flashinfer_moe_tune_max_num_tokens(_experts(name, moe_config=small)) == 8192


def test_nvfp4_modular_experts_also_clamp_at_their_launch_chunk(collect_moe, fake_fi_moe_largest_bucket):
    config = SimpleNamespace(max_num_tokens=65536, dp_size=1)
    experts = _experts("TrtLlmNvFp4ExpertsModular", moe_config=config, _get_chunk_size=lambda self: 40000)
    assert collect_moe._flashinfer_moe_tune_max_num_tokens(experts) == 40000


@pytest.mark.parametrize("name", ["FlashInferCuteDSLExperts", "TritonExperts", "MarlinExperts", "CutlassExpertsMxfp4"])
def test_uncapped_and_non_flashinfer_experts_have_no_clamp(collect_moe, name):
    assert collect_moe._flashinfer_moe_tune_max_num_tokens(_experts(name)) is None
    assert collect_moe._flashinfer_moe_tune_max_num_tokens(None) is None


def test_an_untraced_flashinfer_experts_class_fails_the_case(collect_moe):
    with pytest.raises(RuntimeError, match="FlashInferB12xExperts run a FlashInfer op"):
        collect_moe._flashinfer_moe_tune_max_num_tokens(_experts("FlashInferB12xExperts"))


def test_nemotron_resolves_a_non_gated_activation(collect_moe):
    config = collect_moe._resolve_moe_runtime_config("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", {})
    assert config["activation"] == "relu2_no_mul"


def test_non_gated_layers_pass_serving_ckpt_names():
    """RoutedExperts rejects an up projection name next to a non-gated
    activation at vLLM 0.32.0; the GPU-only call site must pass Nemotron-H's
    names. Pin the wiring statically."""
    source = COLLECT_MOE.read_text(encoding="utf-8")
    assert 'ckpt_names = {"ckpt_names": ("up_proj", "down_proj", None)}' in source
    assert 'MoEActivation.from_str(runtime_config["activation"]).is_gated' in source
    assert "**ckpt_names," in source


def test_situ_betas_are_always_passed_natively():
    """The 0.24-era situ-as-silu Marlin approximation is gone: its constructor
    probe inspected the factory function's __init__ and never saw the betas,
    so every stock run timed silu on forced Marlin."""
    source = COLLECT_MOE.read_text(encoding="utf-8")
    assert "situ_as_silu" not in source
    assert "_supports_current_device = staticmethod" not in source
    assert 'activation_situ_beta=runtime_config["activation_situ_beta"]' in source
