# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted and modified from vLLM at commit 98dff2a81d747d1dba01a47f939f48c3526d4206:
# https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/models/inkling/nvidia/sconv_swa_attn.py

"""Config onboarding uses declared geometry, without analytical model construction."""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import re
from pathlib import Path

import pytest

from aisimulate.support import config_profile
from aisimulate.support.config_profile import (
    _ATTENTION_DIMENSIONS,
    ProfileRequestError,
    _geometry,
    _linear_layout_gap,
    _runtime_layout,
    derive_profile,
    linear_block_bytes,
    linear_block_size,
    linear_runtime_resources,
    load_model_config,
    profile_from_observations,
    validate_overrides,
)
from aisimulate.support.schema import SupportRequest

pytestmark = pytest.mark.unit

GLM = Path(__file__).parents[2] / "src/aisimulate_core/model_configs/zai-org--GLM-5.3-Flash_config.json"


def _config(tmp_path, **updates):
    # Original small synthetic geometry makes every tensor count hand-checkable.
    raw = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 64,
        "max_position_embeddings": 2048,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
    }
    raw.update(updates)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_model_config(path)


def _request(kind="dense", **topology):
    return SupportRequest.model_validate(
        {
            "identity": {
                "model": "example/configured-model",
                "model_revision": "checkpoint-123",
                "model_kind": kind,
                "framework_version": "0.25.1",
                "gpu": "h200_sxm",
                "interconnect": "NVLink",
            },
            "search": {"context_length": 2048, **topology},
            "workload": {"concurrency": 2},
        }
    )


def _runtime(**updates):
    return {"kv_cache_dtype": "bfloat16", "fmha_quant_mode": "bfloat16", "comm_quant_mode": "half", **updates}


def _glm_request(**topology):
    # The deployment from ai-dynamo/aisimulate#475.
    return SupportRequest.model_validate(
        {
            "identity": {
                "model": "zai-org/GLM-5.3-Flash",
                "model_revision": "91b69215804f2d2036a8867f6ce44848e278dcda",
                "model_kind": "moe",
                "framework_version": "0.31.0",
                "gpu": "b300_sxm",
                "interconnect": "nvswitch",
            },
            "search": {"context_length": 1048576, "tensor_parallel": 4, "moe_tensor_parallel": 4, **topology},
            "collection": {"max_num_tokens": 8192, "max_batch_size": 256, "gpu_memory_utilization": 0.92},
        }
    )


def _zero_dim_config(tmp_path, nested=False, **updates):
    # A decoder whose ordinary attention dimensions are declared as 0, as linear-attention checkpoints do.
    raw = {"num_hidden_layers": 2, "head_dim": 0, "qk_rope_head_dim": 0, **updates}
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"text_config": raw} if nested else raw))
    return load_model_config(path)


def test_glm53_flash_config_is_runtime_resolved_not_rejected():
    config = load_model_config(GLM)
    assert config.sha256 == "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
    assert config.runtime_layout == (
        "layer_types deepseek_sparse_attention x11",
        "layer_types linear_attention x34",
        "recurrent-state field linear_attn_config",
    )
    assert config.suggestions["architecture"] == "Glm5NextForConditionalGeneration"
    assert config.suggestions["num_experts"] == 288
    assert config.suggestions["context_length"] == 1048576
    assert config.notes["cache_layout"] == (
        f"cache layout pending the runtime probe ({', '.join(config.runtime_layout)}). "
        "config.json does not establish cache or state geometry. onboard probe-runtime measures it."
    )


_NEMOTRON_H = {
    "nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_config.json",
    "nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4_config.json",
    "nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-FP8_config.json",
    "nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4_config.json",
    "nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16_config.json",
    "nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8_config.json",
    "nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4_config.json",
    "nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16_config.json",
    "nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4_config.json",
    "nvidia--Nemotron-H-56B-Base-8K_config.json",
}
# Packaged configs whose declared architecture intake does not recognize, by the architecture their reason names.
# The Qwen3-VL and Llama 4 text configs name no decoder architecture, so their wrapper names it.
_UNRECOGNIZED = {
    **dict.fromkeys(
        (f"Qwen--Qwen3-VL-{size}" for size in ("2B-Instruct", "4B-Instruct", "8B-Instruct", "32B-Instruct")),
        "Qwen3VLForConditionalGeneration",
    ),
    "Qwen--Qwen3-VL-32B-Thinking": "Qwen3VLForConditionalGeneration",
    "Qwen--Qwen3-VL-235B-A22B-Instruct": "Qwen3VLMoeForConditionalGeneration",
    "Qwen--Qwen3-VL-30B-A3B-Instruct": "Qwen3VLMoeForConditionalGeneration",
    "meta-llama--Llama-4-Maverick-17B-128E-Instruct": "Llama4ForConditionalGeneration",
    "meta-llama--Llama-4-Scout-17B-16E-Instruct": "Llama4ForConditionalGeneration",
    "MiniMaxAI--MiniMax-M3": "MiniMaxM3ForCausalLM",
    "nvidia--MiniMax-M3-NVFP4": "MiniMaxM3SparseForCausalLM",
    "XiaomiMiMo--MiMo-7B-Base": "MiMoForCausalLM",
    "XiaomiMiMo--MiMo-V2-Flash": "MiMoV2FlashForCausalLM",
    "nvidia--Llama-3_3-Nemotron-Super-49B-v1": "DeciLMForCausalLM",
}
# DeepSeek, Kimi K2 and GLM-5 MLA/DSA configs keep explicit cache accounting.
_MLA = (
    "deepseek-ai--DeepSeek-R1",
    "deepseek-ai--DeepSeek-V3",
    "deepseek-ai--DeepSeek-V3.2",
    "nvidia--DeepSeek-V3.1-NVFP4",
    "nvidia--DeepSeek-V3.2-NVFP4",
    "moonshotai--Kimi-K2-Instruct",
    "moonshotai--Kimi-K2.5",
    "nvidia--Kimi-K2.5-NVFP4",
    "nvidia--Kimi-K2.6-NVFP4",
    "nvidia--Kimi-K2.7-Code-NVFP4",
    "zai-org--GLM-5",
    "zai-org--GLM-5-FP8",
    "zai-org--GLM-5.1",
    "zai-org--GLM-5.1-FP8",
    "zai-org--GLM-5.2",
    "zai-org--GLM-5.2-FP8",
    "zai-org--GLM-5.3",
    "nvidia--GLM-5-NVFP4",
    "nvidia--GLM-5.1-NVFP4",
    "nvidia--GLM-5.2-NVFP4",
    "nvidia--GLM-5.3-NVFP4",
)


def test_packaged_nemotron_h_hybrids_are_runtime_resolved():
    paths = sorted(GLM.parent.glob("*_config.json"))
    nemotron = {
        path.name for path in paths if json.loads(path.read_text()).get("architectures") == ["NemotronHForCausalLM"]
    }
    assert nemotron == _NEMOTRON_H
    geometry = {"cache_layout", "kv_bytes_per_token", "cache_groups", "cache_block_sizes"}
    for name in sorted(nemotron):
        # Every one loads, including those whose unread time_step_limit spells Infinity.
        config = load_model_config(GLM.parent / name)
        assert {"recurrent-state field mamba_num_heads", "recurrent-state field ssm_state_size"} <= set(
            config.runtime_layout
        ), name
        draft = derive_profile(config, None)
        assert set(draft.pending_runtime) == {"cache_geometry"}, name
        assert not geometry & draft.missing.keys(), name


def test_packaged_unrecognized_architectures_are_pending_and_mla_keeps_explicit_accounting():
    unrecognized, unloadable = {}, set()
    for path in sorted(GLM.parent.glob("*_config.json")):
        name = path.name.removesuffix("_config.json")
        try:
            config = load_model_config(path)
        except ValueError:
            unloadable.add(name)
            continue
        reasons = [reason for reason in config.runtime_layout if reason.startswith("unrecognized architecture")]
        if reasons:
            unrecognized[name] = reasons
    # Unrelated quantization and expert-count metadata errors stop these before any layout is classified.
    assert unloadable == {
        "nvidia--DeepSeek-V4-Flash-NVFP4",
        "nvidia--DeepSeek-V4-Pro-NVFP4",
        "nvidia--DeepSeek-V4.1-Flash-NVFP4",
        "nvidia--Gemma-4-31B-IT-NVFP4",
    }
    assert unrecognized == {name: [f"unrecognized architecture {arch}"] for name, arch in _UNRECOGNIZED.items()}
    geometry = {"cache_layout", "kv_bytes_per_token"}
    for name, architecture in _UNRECOGNIZED.items():
        draft = derive_profile(load_model_config(GLM.parent / f"{name}_config.json"), None)
        assert (
            draft.profile is None
            and f"unrecognized architecture {architecture}" in draft.pending_runtime["cache_geometry"]
        ), name
        assert geometry.isdisjoint(draft.missing), name
    for name in _MLA:
        config = load_model_config(GLM.parent / f"{name}_config.json")
        draft = derive_profile(config, None)
        assert config.runtime_layout == () and draft.pending_runtime == {}, name
        assert geometry <= draft.missing.keys(), name
        reason = draft.missing["kv_bytes_per_token"]
        assert "MLA/DSA" in reason and "unknown layouts" not in reason, name
    # A file that names no architecture, such as a ModelOpt sidecar, keeps asking for explicit accounting.
    sidecar = derive_profile(load_model_config(GLM.parent / "nvidia--GLM-5-NVFP4_hf_quant_config.json"), None)
    assert sidecar.pending_runtime == {} and {"architecture", *geometry} <= sidecar.missing.keys()
    assert "MLA/DSA and unknown layouts" in sidecar.missing["kv_bytes_per_token"]


@pytest.mark.parametrize(
    "raw,architecture,expected",
    [
        ({"layer_types": ["full_attention", "sliding_attention"]}, None, ()),
        ({"layer_types": ["deepseek_sparse_attention"]}, "GlmMoeDsaForCausalLM", ()),
        ({"layer_types": ["deepseek_sparse_attention"]}, "DeepseekV32ForCausalLM", ()),
        (
            {"layer_types": ["deepseek_sparse_attention"]},
            "LlamaForCausalLM",
            ("layer_types deepseek_sparse_attention x1",),
        ),
        ({"layer_types": ["gdn", "full_attention", "gdn"]}, None, ("layer_types gdn x2",)),
        # Null state fields are absent, not markers.
        (
            {
                "ssm_cfg": None,
                "mamba_d_state": None,
                "linear_conv_kernel_dim": None,
                "linear_attn_config": None,
                "ssm_state_size": None,
                "mamba_num_heads": None,
            },
            None,
            (),
        ),
        # Architecture names match case-insensitively; the name is reported as declared.
        ({}, "ToyJambaForCausalLM", ("recurrent architecture ToyJambaForCausalLM",)),
        ({}, "RWKVForCausalLM", ("recurrent architecture RWKVForCausalLM",)),
        # An all-1 attn_type_list is full attention; other integers are one reason per distinct value.
        ({"attn_type_list": [1, 1, 1]}, None, ()),
        ({"attn_type_list": [1, 0, 0, 2, 1]}, None, ("attn_type_list 0 x2", "attn_type_list 2 x1")),
        # Only real integers count; malformed lists are structural errors raised elsewhere.
        ({"attn_type_list": [True, False, "0", None, 1.5]}, None, ()),
        ({"attn_type_list": "0"}, None, ()),
        # State markers win over recognized layer names, and reasons come back sorted.
        (
            {"layer_types": ["linear_attention", "full_attention"], "mamba_d_state": 16},
            "ToyMambaForCausalLM",
            (
                "layer_types linear_attention x1",
                "recurrent architecture ToyMambaForCausalLM",
                "recurrent-state field mamba_d_state",
            ),
        ),
        (
            {"layer_types": ["linear_attention"], "attn_type_list": [0], "ssm_state_size": 128},
            None,
            (
                "attn_type_list 0 x1",
                "layer_types linear_attention x1",
                "recurrent-state field ssm_state_size",
            ),
        ),
    ],
)
def test_runtime_layout_reasons(raw, architecture, expected):
    assert _runtime_layout(raw, architecture, architecture) == expected


@pytest.mark.parametrize(
    "raw,architecture,named,expected",
    [
        ({}, "MiMoForCausalLM", "MiMoForCausalLM", ("unrecognized architecture MiMoForCausalLM",)),
        # A multimodal wrapper names a text config that names no decoder architecture.
        ({}, None, "Qwen3VLForConditionalGeneration", ("unrecognized architecture Qwen3VLForConditionalGeneration",)),
        # A config that names no architecture keeps today's explicit inputs.
        ({}, None, None, ()),
        ({}, None, "LlamaForCausalLM", ()),
        ({}, "Qwen3MoeForCausalLM", "Qwen3MoeForCausalLM", ()),
        # MLA/DSA families keep explicit cache accounting.
        ({}, "DeepseekV3ForCausalLM", "DeepseekV3ForCausalLM", ()),
        ({}, "DeepseekV32ForCausalLM", "DeepseekV32ForCausalLM", ()),
        ({}, "GlmMoeDsaForCausalLM", "GlmMoeDsaForCausalLM", ()),
        # Declared windows or convolution state derive a grouped cache, whatever the architecture.
        ({"num_hidden_layers": 2, "sliding_window": 128}, "GptOssForCausalLM", "GptOssForCausalLM", ()),
        (
            {"num_hidden_layers": 2, "layer_types": ["full_attention", "sliding_attention"], "sliding_window": 128},
            "Gemma4ForConditionalGeneration",
            "Gemma4ForConditionalGeneration",
            (),
        ),
        (
            {"num_hidden_layers": 2, "sliding_window": 128, "use_sliding_window": False},
            "MiMoForCausalLM",
            "MiMoForCausalLM",
            ("unrecognized architecture MiMoForCausalLM",),
        ),
        # A marker already makes the layout pending, so it is the only reason.
        ({"mamba_d_state": 16}, "MiMoForCausalLM", "MiMoForCausalLM", ("recurrent-state field mamba_d_state",)),
    ],
)
def test_unrecognized_architecture_reason(raw, architecture, named, expected):
    assert _runtime_layout(raw, architecture, named) == expected


_GEMMA3 = {"architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3_text"}
_SLIDING_FIRST = ["sliding_attention", "full_attention"]


@pytest.mark.parametrize(
    "fields,message",
    [
        # A window with sliding_window_pattern but no explicit layer list.
        ({"sliding_window": 512, "sliding_window_pattern": 6}, "unsupported implicit sliding-window schedule"),
        # A window with hybrid_layer_pattern, as in MiMo-V2-Flash, but no explicit layer list.
        ({"sliding_window": 128, "hybrid_layer_pattern": [0, 1]}, "unsupported implicit sliding-window schedule"),
        ({"use_sliding_window": True}, "use_sliding_window=true requires a positive sliding_window"),
        ({"sliding_window": 128, "local_layer_ids": [-1]}, "local_layer_ids must be a list of nonnegative layer"),
        ({"sliding_window": 128, "local_layer_ids": None}, "local_layer_ids must be a list of nonnegative layer"),
        ({"sliding_window": 128, "local_layer_ids": [0, 0]}, "local_layer_ids must be unique indexes below"),
        (
            {"sliding_window": 128, "layer_types": _SLIDING_FIRST, "local_layer_ids": [1]},
            "conflicting layer_types and local_layer_ids attention layouts",
        ),
        (
            {"sliding_window": 128, "use_sliding_window": False, "layer_types": _SLIDING_FIRST},
            "use_sliding_window=false conflicts with sliding attention layers",
        ),
        ({"layer_types": _SLIDING_FIRST}, "sliding attention layers require a positive sliding_window"),
    ],
)
def test_unrecognized_architecture_with_an_underivable_window_layout_is_pending(tmp_path, fields, message):
    # Intake cannot derive cache groups from these declarations, so the runtime probe resolves the layout,
    # as it does for a marker layout with the same fields.
    config = _config(tmp_path, **_GEMMA3, **fields)
    assert config.runtime_layout == ("unrecognized architecture Gemma3ForCausalLM",)
    draft = derive_profile(config, _request(), _runtime())
    assert draft.profile is None
    assert "unrecognized architecture Gemma3ForCausalLM" in draft.pending_runtime["cache_geometry"]
    marker = _config(tmp_path, **_GEMMA3, mamba_d_state=16, **fields)
    assert marker.runtime_layout == ("recurrent-state field mamba_d_state",)
    # A recognized architecture keeps today's message.
    with pytest.raises(ValueError, match=re.escape(message)):
        _config(tmp_path, **fields)


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"sliding_window": 128, "sliding_window_size": 256}, "conflicting sliding-window fields"),
        ({"layer_types": ["full_attention"]}, "layer_types must have one entry per num_hidden_layers"),
        ({"num_hidden_layers": 2.0}, "num_hidden_layers must be an integer"),
    ],
)
@pytest.mark.parametrize("marker", [{}, {"mamba_d_state": 16}])
def test_unrecognized_architecture_keeps_the_marker_layout_structure_checks(tmp_path, fields, message, marker):
    # These checks also run for a marker layout, so they still fail for an unrecognized architecture.
    with pytest.raises(ValueError, match=re.escape(message)):
        _config(tmp_path, **_GEMMA3, **marker, **fields)


def test_attention_dimensions_are_listed_explicitly():
    expected = {
        "kv_lora_rank",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "index_head_dim",
        "index_n_heads",
    }
    assert expected == _ATTENTION_DIMENSIONS


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "marker,reasons",
    [
        (
            {"layer_types": ["linear_attention", "deepseek_sparse_attention"]},
            ("layer_types deepseek_sparse_attention x1", "layer_types linear_attention x1"),
        ),
        ({"mamba_d_state": 16}, ("recurrent-state field mamba_d_state",)),
        ({"ssm_cfg": {}}, ("recurrent-state field ssm_cfg",)),
        ({"linear_conv_kernel_dim": 4}, ("recurrent-state field linear_conv_kernel_dim",)),
        ({"linear_attn_config": {"head_dim": 128}}, ("recurrent-state field linear_attn_config",)),
        # NemotronH hybrids declare Mamba state with these two fields.
        ({"ssm_state_size": 128}, ("recurrent-state field ssm_state_size",)),
        ({"mamba_num_heads": 64}, ("recurrent-state field mamba_num_heads",)),
        ({"architectures": ["ToyMambaForCausalLM"]}, ("recurrent architecture ToyMambaForCausalLM",)),
        # A mixed attn_type_list (0 = linear attention) is runtime-resolved, not rejected.
        ({"attn_type_list": [1, 0]}, ("attn_type_list 0 x1",)),
        ({"attn_type_list": [0, 2]}, ("attn_type_list 0 x1", "attn_type_list 2 x1")),
        # State markers win over recognized layer names.
        (
            {"layer_types": ["full_attention", "full_attention"], "mamba_d_state": 16},
            ("recurrent-state field mamba_d_state",),
        ),
    ],
)
def test_unrecognized_layouts_load_with_zero_ordinary_dimensions(tmp_path, nested, marker, reasons):
    config = _zero_dim_config(tmp_path, nested, **marker)
    assert config.runtime_layout == reasons
    assert "head_dim" not in config.notes["cache_layout"]
    assert "onboard probe-runtime" in config.notes["cache_layout"]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "update,message",
    [
        ({"layer_types": ["linear_attention"]}, "one entry per num_hidden_layers"),
        ({"layer_types": "linear_attention"}, "one entry per num_hidden_layers"),
        ({"layer_types": ["linear_attention", 3]}, "layer_types entries must be nonempty strings"),
        ({"layer_types": ["linear_attention", " "]}, "layer_types entries must be nonempty strings"),
        ({"attn_type_list": [0]}, "attn_type_list must have one entry per num_hidden_layers"),
        ({"attn_type_list": "0"}, "attn_type_list must have one entry per num_hidden_layers"),
        ({"attn_type_list": [0, "1"]}, "attn_type_list entries must be integers"),
        ({"attn_type_list": [0, 1.5]}, "attn_type_list entries must be integers"),
        ({"attn_type_list": [0, True]}, "attn_type_list entries must be integers"),
        ({"num_hidden_layers": 2.0}, "num_hidden_layers"),
        ({"is_encoder_decoder": True}, "unsupported encoder-decoder"),
        ({"sliding_window": 64, "sliding_window_size": 128}, "conflicting sliding-window"),
    ],
)
def test_runtime_resolved_layouts_keep_structural_checks(tmp_path, nested, update, message):
    with pytest.raises(ValueError, match=message):
        _zero_dim_config(tmp_path, nested, mamba_d_state=16, **update)


@pytest.mark.parametrize("update", [{"layer_types": ["full_attention", 3]}, {"layer_types": ["full_attention", ""]}])
def test_recognized_layouts_keep_layer_type_structure_checks(tmp_path, update):
    with pytest.raises(ValueError, match="layer_types entries must be nonempty strings"):
        _config(tmp_path, **update)


@pytest.mark.parametrize(
    "update,message",
    [
        ({"attn_type_list": [1]}, "attn_type_list must have one entry per num_hidden_layers"),
        ({"attn_type_list": [1, "1"]}, "attn_type_list entries must be integers"),
        ({"attn_type_list": [1, True]}, "attn_type_list entries must be integers"),
        ({"attn_type_list": [True, True]}, "attn_type_list entries must be integers"),
    ],
)
def test_recognized_layouts_keep_attn_type_list_structure_checks(tmp_path, update, message):
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, **update)


def test_all_full_attention_type_list_is_not_runtime_resolved(tmp_path):
    assert _config(tmp_path, attn_type_list=[1, 1]).runtime_layout == ()


def test_recognized_layouts_still_reject_zero_ordinary_dimensions(tmp_path):
    with pytest.raises(ValueError, match="head_dim"):
        _zero_dim_config(tmp_path, layer_types=["full_attention", "sliding_attention"], sliding_window=8)


def test_runtime_resolved_geometry_skips_head_derived_values():
    config = load_model_config(GLM)
    dims = _geometry(config, None, config.runtime_layout, _glm_request())
    assert "local_kv_heads" not in dims and dims["num_hidden_layers"] == 45
    # A declared 0 is "no such dimension", not a size to compute with.
    assert "head_dim" not in dims


def test_runtime_resolved_geometry_keeps_the_attention_head_tp_check():
    config = load_model_config(GLM)
    with pytest.raises(ProfileRequestError, match="num_attention_heads must divide"):
        _geometry(config, None, config.runtime_layout, _glm_request(tensor_parallel=3, moe_tensor_parallel=3))


_GLM_PRECISION = {
    "gemm_quant_mode": "fp8_block",
    "moe_quant_mode": "fp8_block",
    "fmha_quant_mode": "bfloat16",
    "comm_quant_mode": "half",
    "kv_cache_dtype": "bfloat16",
}
# The groups stand in for the runtime report, which always reports a grouped layout, names its groups
# runtime_group_<i> and states each group's sliding_window. A real GLM probe also reports state groups. The spec
# guard in collector/fpm_forward/runtime_memory.py keeps that probe incomplete until state groups are supported.
_RUNTIME_GROUP = {
    "name": "runtime_group_0",
    "kind": "attention",
    "num_layers": 11,
    "block_size_tokens": 64,
    "page_size_bytes": 4096,
    "sliding_window": None,
}
_WINDOWED_GROUP = {**_RUNTIME_GROUP, "name": "runtime_group_1", "num_layers": 3, "sliding_window": 128}
_GROUPED_OBSERVATION = {
    "cache_layout": "grouped",
    "cache_groups": [_RUNTIME_GROUP],
    "max_num_tokens": 8192,
    "max_batch_size": 256,
    "provenance": "observed runtime resources",
}
# Import also accepts a linear observation, which the vLLM probe never reports; it is not a claim about GLM.
_LINEAR_OBSERVATION = {
    "cache_layout": "linear",
    "kv_bytes_per_token": 6000,
    "max_num_tokens": 8192,
    "max_batch_size": 256,
    "provenance": "observed runtime resources",
}
_CHECKED = "user override; checked against the runtime probe's measurement"
_LLAMA_PRECISION = {"gemm_quant_mode": "bfloat16", "moe_quant_mode": "bfloat16", **_runtime()}


def _import(resources, overrides=None):
    return profile_from_observations(load_model_config(GLM), _glm_request(), _GLM_PRECISION, resources, overrides)


def _sources(profile):
    return {key: entry["source"] for key, entry in json.loads(profile.provenance)["user_overrides"].items()}


def test_runtime_resolved_config_accepts_observed_cache_geometry():
    # The runtime import path revalidates the same config that intake accepted.
    profile = _import(_GROUPED_OBSERVATION)
    assert profile.architecture == "Glm5NextForConditionalGeneration"
    assert profile.deployments[0].resources.cache_groups[0].num_layers == 11


def test_imported_provenance_says_the_probe_measured_the_pending_geometry():
    config = load_model_config(GLM)
    notes = json.loads(_import(_GROUPED_OBSERVATION).provenance)["config_notes"]
    assert notes["cache_layout"].startswith("cache layout measured by the runtime probe (")
    assert "pending" not in notes["cache_layout"] and "linear_attention x34" in notes["cache_layout"]
    assert {key: value for key, value in notes.items() if key != "cache_layout"} == {
        key: value for key, value in config.notes.items() if key != "cache_layout"
    }


@pytest.mark.parametrize(
    "typed,resources",
    [
        ({"cache_layout": "linear", "kv_bytes_per_token": 6000}, _LINEAR_OBSERVATION),
        (
            {"cache_layout": "grouped", "cache_groups": _GROUPED_OBSERVATION["cache_groups"]},
            _GROUPED_OBSERVATION,
        ),
    ],
)
def test_typed_geometry_that_matches_the_probe_builds_the_profile(typed, resources):
    profile = _import(resources, typed)
    assert profile.deployments[0].resources.cache_layout == resources["cache_layout"]
    recorded = json.loads(profile.provenance)["user_overrides"]
    assert recorded.keys() == typed.keys()
    for key, value in validate_overrides(typed).items():
        assert recorded[key] == {"value": value, "source": _CHECKED}


def test_typed_kv_bytes_that_differ_from_the_probe_name_both_values():
    with pytest.raises(ValueError) as error:
        _import(_LINEAR_OBSERVATION, {"cache_layout": "linear", "kv_bytes_per_token": 5000})
    message = str(error.value)
    assert "resource_overrides.kv_bytes_per_token=5000" in message
    assert "measured kv_bytes_per_token=6000" in message
    assert "the measurement is authoritative" in message
    assert "Correct or remove the override; if the runtime changed, run onboard probe-runtime again." in message


def test_claim_without_a_measured_counterpart_names_the_measured_layout():
    with pytest.raises(ValueError) as error:
        _import(_LINEAR_OBSERVATION, {"cache_groups": [_RUNTIME_GROUP]})
    message = str(error.value)
    assert "resource_overrides.cache_groups=[" in message
    assert "the runtime probe measured no cache_groups for its linear cache_layout" in message


@pytest.mark.parametrize("claim", [{"cache_layout": "linear"}, {"kv_bytes_per_token": 6000}])
def test_per_token_claim_against_a_linear_observation_without_kv_bytes_names_the_missing_field(claim):
    # A malformed observation: FpmModelProfile alone rejects a linear cache without kv_bytes_per_token.
    observed = {key: value for key, value in _LINEAR_OBSERVATION.items() if key != "kv_bytes_per_token"}
    with pytest.raises(ValueError) as error:
        _import(observed, claim)
    message = str(error.value)
    ((key, value),) = claim.items()
    assert (
        f"resource_overrides.{key}={value}, but the runtime probe measured no kv_bytes_per_token "
        "for its linear cache_layout"
    ) in message
    assert "measured cache_layout=linear" not in message


def test_cache_group_claims_match_without_names_or_order():
    # The user names a group; the probe names it runtime_group_<i>. A full-history group may omit sliding_window.
    named = {**_RUNTIME_GROUP, "name": "full_attention"}
    unwindowed = {key: value for key, value in named.items() if key != "sliding_window"}
    for claim in ([named], [unwindowed]):
        _import(_GROUPED_OBSERVATION, {"cache_layout": "grouped", "cache_groups": claim})
    bare = {key: value for key, value in _RUNTIME_GROUP.items() if key != "sliding_window"}
    _import({**_GROUPED_OBSERVATION, "cache_groups": [bare]}, {"cache_groups": [named]})
    # Order carries no geometry either, and the profile keeps the measured groups as reported.
    observed = {**_GROUPED_OBSERVATION, "cache_groups": [_RUNTIME_GROUP, _WINDOWED_GROUP]}
    profile = _import(observed, {"cache_groups": [{**_WINDOWED_GROUP, "name": "sliding"}, named]})
    measured = profile.deployments[0].resources.cache_groups
    assert [group.name for group in measured] == ["runtime_group_0", "runtime_group_1"]
    assert _sources(profile) == {"cache_groups": _CHECKED}


def test_cache_group_claim_with_another_page_size_shows_both_lists():
    claim = [{**_RUNTIME_GROUP, "name": "full_attention", "page_size_bytes": 8192}]
    with pytest.raises(ValueError) as error:
        _import(_GROUPED_OBSERVATION, {"cache_groups": claim})
    message = str(error.value)
    assert (
        'resource_overrides.cache_groups=[{"block_size_tokens":64,"kind":"attention","name":"full_attention",'
        '"num_layers":11,"page_size_bytes":8192,"sliding_window":null}] differs'
    ) in message
    assert (
        'measured cache_groups=[{"block_size_tokens":64,"kind":"attention","name":"runtime_group_0",'
        '"num_layers":11,"page_size_bytes":4096,"sliding_window":null}]'
    ) in message


_RUNTIME_MEMORY = {
    "kv_cache_bytes": 409_600,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 2048,
    "provenance": "observed runtime capacity",
}
_LINEAR_REASON = (
    "linear: import wrote the linear layout because the supplied {claimed} matched the measured full-attention groups"
)


def _groups(pages):
    return [
        {**_RUNTIME_GROUP, "name": f"runtime_group_{index}", "page_size_bytes": page, "block_size_tokens": block}
        for index, (page, block) in enumerate(pages)
    ]


@pytest.mark.parametrize(
    "pages,expected",
    [
        # One full-attention group: 4096 bytes per 64-token block is 64 bytes per token.
        ([(4096, 64)], 64),
        # 0.2 + 27.9 + 3.9 bytes per token is exactly 32, which a floating-point sum misses.
        ([(2, 10), (279, 10), (39, 10)], 32),
    ],
)
def test_linear_runtime_resources_sums_full_attention_groups(pages, expected):
    measured = {**_GROUPED_OBSERVATION, "cache_groups": _groups(pages), "runtime_memory": dict(_RUNTIME_MEMORY)}
    before = json.loads(json.dumps(measured))
    linear = linear_runtime_resources(measured)
    # Every other field, including the runtime memory and the provenance, is kept.
    assert linear == {
        **{key: value for key, value in before.items() if key != "cache_groups"},
        "cache_layout": "linear",
        "kv_bytes_per_token": expected,
    }
    assert type(linear["kv_bytes_per_token"]) is int
    assert measured == before


@pytest.mark.parametrize(
    "resources,reason",
    [
        (
            {**_GROUPED_OBSERVATION, "cache_groups": [_RUNTIME_GROUP, _WINDOWED_GROUP]},
            "cache group runtime_group_1 has sliding_window=128",
        ),
        (
            {
                **_GROUPED_OBSERVATION,
                "cache_groups": [_RUNTIME_GROUP, {**_WINDOWED_GROUP, "kind": "convolution", "sliding_window": 4}],
            },
            "cache group runtime_group_1 has kind=convolution",
        ),
        # 25 bytes per 4-token block is 25/4 bytes per token.
        (
            {
                **_GROUPED_OBSERVATION,
                "cache_groups": [{**_RUNTIME_GROUP, "block_size_tokens": 4, "page_size_bytes": 25}],
            },
            "the measured bytes per token, 25/4, are not a whole number",
        ),
        # 128 / 16 + 256 / 32 is 16 bytes per token, but replay of a linear profile uses one block size.
        (
            {**_GROUPED_OBSERVATION, "cache_groups": _groups([(128, 16), (256, 32)])},
            "cache groups runtime_group_0 and runtime_group_1 have different block sizes, 16 and 32 tokens",
        ),
        (
            {
                **_GROUPED_OBSERVATION,
                "cache_groups": [{**_RUNTIME_GROUP, "block_size_tokens": 1, "page_size_bytes": 64}],
            },
            "cache group runtime_group_0 has block_size_tokens=1, and vLLM replay needs blocks of at least 2 tokens",
        ),
        (_LINEAR_OBSERVATION, "the measurement's cache_layout is linear, not grouped"),
    ],
    ids=[
        "sliding-window-group",
        "convolution-group",
        "fractional-sum",
        "different-block-sizes",
        "one-token-blocks",
        "linear-input",
    ],
)
def test_linear_runtime_resources_has_no_linear_layout_for_other_measurements(resources, reason):
    assert linear_runtime_resources(resources) is None
    assert linear_block_size(resources) is None
    # Finalization gives this reason when it refuses a linear profile.
    assert _linear_layout_gap(resources) == reason


@pytest.mark.parametrize(
    "groups,block",
    [
        # One block size: 16 tokens fill one 128-byte page of each group.
        (_groups([(128, 16), (128, 16)]), 256),
        # Block sizes of 16 and 32 tokens: a linear layout has one block size, so it does not describe these groups.
        (_groups([(128, 16), (128, 32)]), None),
        ([_RUNTIME_GROUP, _WINDOWED_GROUP], None),
    ],
    ids=["one-block-size", "two-block-sizes", "windowed-group"],
)
def test_linear_block_bytes_fill_whole_pages_in_every_group(groups, block):
    assert linear_block_bytes({**_GROUPED_OBSERVATION, "cache_groups": groups}) == block


@pytest.mark.parametrize(
    "pages",
    [
        # One full-attention group: 512 bytes per 16-token block is 32 bytes per token.
        [(512, 16)],
        # Synthetic groups whose bytes per token are not whole numbers: 0.2 + 27.9 + 3.9 is exactly 32, which a
        # floating-point sum misses.
        [(2, 10), (279, 10), (39, 10)],
    ],
)
@pytest.mark.parametrize("layout", [False, True])
def test_matching_linear_claims_give_a_linear_profile(tmp_path, pages, layout):
    config = _config(tmp_path)
    request = _request(tensor_parallel=2)
    # Intake's exact per-rank value at TP2: 2 K/V * 2 layers * 1 rank-local KV head * head_dim 4 * 2 bytes.
    claim = derive_profile(config, request, _runtime()).resolved["kv_bytes_per_token"]
    assert claim == 32
    claims = {"cache_layout": "linear", "kv_bytes_per_token": claim} if layout else {"kv_bytes_per_token": claim}
    observed = {**_GROUPED_OBSERVATION, "cache_groups": _groups(pages), "runtime_memory": dict(_RUNTIME_MEMORY)}
    profile = profile_from_observations(config, request, _LLAMA_PRECISION, observed, claims)
    resources = profile.deployments[0].resources.model_dump(mode="json")
    assert resources["cache_layout"] == "linear" and resources["kv_bytes_per_token"] == 32
    assert "cache_groups" not in resources
    assert resources["runtime_memory"] == _RUNTIME_MEMORY and resources["provenance"] == observed["provenance"]
    provenance = json.loads(profile.provenance)
    # The measured groups stay in the provenance for audit, with their block sizes.
    assert provenance["measured_cache_groups"] == observed["cache_groups"]
    claimed = "cache_layout and kv_bytes_per_token" if layout else "kv_bytes_per_token"
    assert provenance["cache_layout_reason"] == _LINEAR_REASON.format(claimed=claimed)
    assert _sources(profile) == dict.fromkeys(claims, _CHECKED)


def test_linear_layout_claim_alone_uses_the_measured_whole_number(tmp_path):
    observed = {**_GROUPED_OBSERVATION, "cache_groups": _groups([(512, 16)]), "runtime_memory": dict(_RUNTIME_MEMORY)}
    profile = profile_from_observations(
        _config(tmp_path), _request(tensor_parallel=2), _LLAMA_PRECISION, observed, {"cache_layout": "linear"}
    )
    resources = profile.deployments[0].resources
    assert resources.cache_layout == "linear" and resources.kv_bytes_per_token == 32
    assert json.loads(profile.provenance)["cache_layout_reason"] == _LINEAR_REASON.format(claimed="cache_layout")


def test_linear_layout_claim_with_a_fractional_measurement_keeps_the_grouped_layout(tmp_path):
    # 25 bytes per 4-token block is 25/4 bytes per token; a linear profile needs a whole number.
    groups = [{**_RUNTIME_GROUP, "block_size_tokens": 4, "page_size_bytes": 25}]
    observed = {**_GROUPED_OBSERVATION, "cache_groups": groups}
    profile = profile_from_observations(
        _config(tmp_path), _request(tensor_parallel=2), _LLAMA_PRECISION, observed, {"cache_layout": "linear"}
    )
    assert profile.deployments[0].resources.cache_layout == "grouped"
    provenance = json.loads(profile.provenance)
    assert provenance["cache_layout_reason"] == (
        "grouped: the supplied cache_layout matched the measured full-attention groups, but the measured bytes per "
        "token, 25/4, are not a whole number"
    )
    assert "measured_cache_groups" not in provenance


@pytest.mark.parametrize(
    "claims",
    [{}, {"cache_layout": "grouped"}, {"cache_groups": _groups([(512, 16)])}],
    ids=["no-claim", "grouped-claim", "cache-groups-claim"],
)
def test_without_a_linear_claim_the_profile_stays_grouped(tmp_path, claims):
    observed = {**_GROUPED_OBSERVATION, "cache_groups": _groups([(512, 16)])}
    profile = profile_from_observations(
        _config(tmp_path), _request(tensor_parallel=2), _LLAMA_PRECISION, observed, claims
    )
    assert profile.deployments[0].resources.cache_layout == "grouped"
    provenance = json.loads(profile.provenance)
    assert "cache_layout_reason" not in provenance and "measured_cache_groups" not in provenance


def test_grouped_layout_claim_with_kv_bytes_per_token_is_refused_at_import(tmp_path):
    # 512 bytes per 16-token block is 32 bytes per token.
    observed = {**_GROUPED_OBSERVATION, "cache_groups": _groups([(512, 16)])}
    config, request = _config(tmp_path), _request(tensor_parallel=2)
    # Each claim alone matches the measurement, and the two give different layouts.
    grouped = profile_from_observations(config, request, _LLAMA_PRECISION, observed, {"cache_layout": "grouped"})
    linear = profile_from_observations(config, request, _LLAMA_PRECISION, observed, {"kv_bytes_per_token": 32})
    assert grouped.deployments[0].resources.cache_layout == "grouped"
    assert linear.deployments[0].resources.cache_layout == "linear"
    with pytest.raises(ValueError, match="kv_bytes_per_token cannot be combined with grouped cache_layout"):
        profile_from_observations(
            config, request, _LLAMA_PRECISION, observed, {"cache_layout": "grouped", "kv_bytes_per_token": 32}
        )


def test_linear_and_grouped_profiles_of_one_measurement_differ_by_less_than_a_block(tmp_path, monkeypatch):
    from aisimulate_core.sdk import engine, memory

    def forbidden(*_args, **_kwargs):
        pytest.fail("the profile capacity path loaded a model graph or timing data")

    for name in ("get_model", "build_model_config"):
        monkeypatch.setattr(memory, name, forbidden)
    monkeypatch.setattr(memory.perf_database, "get_database", forbidden)
    monkeypatch.setattr(engine, "_literal_backend_version", lambda _s, _b, version, *_args: version)
    # A pool that is not a whole number of 4096-byte pages: 100 pages plus 2000 bytes.
    runtime = {**_RUNTIME_MEMORY, "kv_cache_bytes": 409_600 + 2000}
    observed = {**_GROUPED_OBSERVATION, "cache_groups": [_RUNTIME_GROUP], "runtime_memory": runtime}
    config = _config(tmp_path)
    # The capacity path checks the profile architecture against this local checkpoint config.
    payload = _request(tensor_parallel=2).model_dump(mode="json")
    payload["identity"]["model"] = str(tmp_path)
    request = SupportRequest.model_validate(payload)
    capacities = {}
    for claims in ({}, {"kv_bytes_per_token": 64}):
        profile = profile_from_observations(config, request, _LLAMA_PRECISION, observed, claims)
        capacities[profile.deployments[0].resources.cache_layout] = memory.estimate_kv_cache(
            request.identity.model,
            request.identity.gpu,
            "vllm",
            request.identity.framework_version,
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
            tp_size=2,
            context_length=2048,
            gpu_memory_capacity_bytes_override=10**9,
            fpm_profile=profile.model_dump(mode="json"),
        )
    linear_tokens = capacities["linear"]["total_kv_size_tokens"]
    # The grouped allocator stores whole 64-token blocks of 4096 bytes.
    grouped_tokens = capacities["grouped"]["total_kv_size_bytes"] // 4096 * 64
    assert (linear_tokens, grouped_tokens) == (6431, 6400)
    assert abs(linear_tokens - grouped_tokens) < 64


def test_linear_kv_claim_one_byte_off_shows_both_values():
    # 4096 bytes per 64-token block is 64 bytes per token.
    with pytest.raises(ValueError) as error:
        _import(_GROUPED_OBSERVATION, {"cache_layout": "linear", "kv_bytes_per_token": 65})
    message = str(error.value)
    assert (
        "resource_overrides.kv_bytes_per_token=65 differs from the runtime probe's measured kv_bytes_per_token=64 "
        "(page_size_bytes / block_size_tokens summed over its cache groups)"
    ) in message
    assert "the measurement is authoritative" in message


def test_kv_claim_against_a_fractional_measurement_says_no_whole_number_matches():
    # 25 bytes per 4-token block is 25/4 bytes per token, which no integer claim can equal.
    group = {**_RUNTIME_GROUP, "block_size_tokens": 4, "page_size_bytes": 25}
    with pytest.raises(ValueError) as error:
        _import({**_GROUPED_OBSERVATION, "cache_groups": [group]}, {"kv_bytes_per_token": 6})
    message = str(error.value)
    assert (
        "resource_overrides.kv_bytes_per_token=6 differs from the runtime probe's measured kv_bytes_per_token=25/4"
        in (message)
    )
    assert "The measured bytes per token are not a whole number, so no integer claim can match" in message


@pytest.mark.parametrize("claim", [{"cache_layout": "linear"}, {"kv_bytes_per_token": 64}])
@pytest.mark.parametrize(
    "group,field",
    [
        (_WINDOWED_GROUP, "sliding_window=128"),
        ({**_WINDOWED_GROUP, "kind": "convolution", "sliding_window": 4}, "kind=convolution"),
    ],
)
def test_linear_claim_against_a_windowed_group_names_that_group(claim, group, field):
    observed = {**_GROUPED_OBSERVATION, "cache_groups": [_RUNTIME_GROUP, group]}
    with pytest.raises(ValueError) as error:
        _import(observed, claim)
    message = str(error.value)
    ((key, value),) = claim.items()
    assert (
        f"resource_overrides.{key}={value}, but the runtime probe measured no per-token cache size: "
        f"its cache group runtime_group_1 has {field}"
    ) in message


def test_import_labels_every_compared_override_that_matches():
    overrides = {
        "cache_layout": "grouped",
        "max_num_tokens": 8192,
        "max_batch_size": 256,
        "kv_cache_dtype": "bfloat16",
        "weights_bytes": 1024,
    }
    # Import does not compare weights_bytes with the probe, so it stays the user's assumption.
    assert _sources(_import(_GROUPED_OBSERVATION, overrides)) == {
        **dict.fromkeys(("cache_layout", "max_num_tokens", "max_batch_size", "kv_cache_dtype"), _CHECKED),
        "weights_bytes": "user assumption",
    }


def test_import_refuses_cache_block_sizes_for_a_pending_layout():
    overrides = {"cache_block_sizes": {"full_attention": 16}}
    with pytest.raises(ValueError) as intake:
        derive_profile(load_model_config(GLM), _glm_request(), overrides)
    with pytest.raises(ValueError) as imported:
        _import(_GROUPED_OBSERVATION, overrides)
    assert str(imported.value) == str(intake.value)
    assert "cache_block_sizes names config-derived cache groups, and this layout has none" in str(imported.value)


def test_import_records_unused_cache_block_sizes_for_a_derivable_layout(tmp_path):
    config = _config(tmp_path, sliding_window=128)
    assert config.runtime_layout == ()
    observed = {
        **_GROUPED_OBSERVATION,
        "cache_groups": [{**_WINDOWED_GROUP, "name": "runtime_group_0", "num_layers": 2}],
    }
    profile = profile_from_observations(
        config, _request(), _LLAMA_PRECISION, observed, {"cache_block_sizes": {"sliding_attention": 16}}
    )
    assert _sources(profile) == {
        "cache_block_sizes": "user override; not used, because the runtime probe measured the cache block sizes"
    }
    assert profile.deployments[0].resources.cache_groups[0].block_size_tokens == 64


@pytest.mark.parametrize("claim", [False, True])
def test_unrecognized_architecture_imports_the_measured_groups(claim):
    config = load_model_config(GLM.parent / "XiaomiMiMo--MiMo-7B-Base_config.json")
    assert config.runtime_layout == ("unrecognized architecture MiMoForCausalLM",)
    # At TP2: 2 K/V * 36 layers * 4 rank-local KV heads * head_dim 128 * 2 bytes = 73728 bytes per token.
    group = {**_RUNTIME_GROUP, "num_layers": 36, "block_size_tokens": 16, "page_size_bytes": 16 * 73728}
    overrides = {"num_experts": 0, **({"kv_bytes_per_token": 73728} if claim else {})}
    profile = profile_from_observations(
        config,
        _request(tensor_parallel=2),
        _LLAMA_PRECISION,
        {**_GROUPED_OBSERVATION, "cache_groups": [group]},
        overrides,
    )
    assert profile.architecture == "MiMoForCausalLM"
    resources = profile.deployments[0].resources
    if claim:
        # The matching kv_bytes_per_token gives back the linear profile that a typed size gave on main.
        assert resources.cache_layout == "linear" and resources.kv_bytes_per_token == 73728
    else:
        assert resources.cache_layout == "grouped" and resources.cache_groups[0].num_layers == 36
    notes = json.loads(profile.provenance)["config_notes"]
    assert notes["cache_layout"].startswith(
        "cache layout measured by the runtime probe (unrecognized architecture MiMoForCausalLM)"
    )
    assert _sources(profile) == {
        "num_experts": "user assumption",
        **({"kv_bytes_per_token": _CHECKED} if claim else {}),
    }


@pytest.mark.parametrize(
    "typed,expected",
    [
        (
            {
                "cache_layout": "grouped",
                "cache_groups": [{**_GROUPED_OBSERVATION["cache_groups"][0], "num_layers": 12}],
            },
            'resource_overrides.cache_groups=[{"block_size_tokens":64,"kind":"attention","name":"runtime_group_0",'
            '"num_layers":12,"page_size_bytes":4096,"sliding_window":null}] differs',
        ),
        ({"max_num_tokens": 4096}, "resource_overrides.max_num_tokens=4096 differs"),
        ({"max_batch_size": 128}, "measured max_batch_size=256; the measurement is authoritative"),
    ],
)
def test_import_mismatches_render_values_compactly(typed, expected):
    with pytest.raises(ValueError, match=re.escape(expected)):
        _import(_GROUPED_OBSERVATION, typed)


def test_observation_import_checks_layout_against_the_architecture_intake_used(tmp_path):
    # The wrapper names LlamaForCausalLM, but the text config names no decoder architecture, so intake does not
    # treat the decoder as Llama's layout: TP8 with four attention heads completes a profile.
    text = {key: value for key, value in _config(tmp_path).raw.items() if key not in ("architectures", "model_type")}
    config = _config(tmp_path, text_config=text)
    assert config.decoder_architecture is None and config.suggestions["architecture"] == "LlamaForCausalLM"
    request = _request(tensor_parallel=8)
    overrides = _runtime(num_experts=0, cache_layout="linear", kv_bytes_per_token=64)
    assert derive_profile(config, request, overrides).profile is not None
    # Import revalidates the same deployment against the same architecture, so it accepts it too.
    precision = {"gemm_quant_mode": "bfloat16", "moe_quant_mode": "bfloat16", **_runtime()}
    resources = {**_LINEAR_OBSERVATION, "kv_bytes_per_token": 64}
    profile = profile_from_observations(config, request, precision, resources, overrides)
    assert profile.architecture == "LlamaForCausalLM" and profile.deployments[0].tp == 8


@pytest.mark.parametrize("preview", [False, True])
def test_runtime_resolved_layout_reports_pending_geometry(preview):
    draft = derive_profile(load_model_config(GLM), None if preview else _glm_request())
    assert draft.profile is None
    assert set(draft.pending_runtime) == {"cache_geometry"}
    reason = draft.pending_runtime["cache_geometry"]
    assert "onboard probe-runtime" in reason and "linear_attention x34" in reason
    assert "onboard import-observations" in reason
    geometry = {"cache_layout", "kv_bytes_per_token", "cache_groups", "cache_block_sizes"}
    assert not geometry & draft.missing.keys()
    assert geometry.isdisjoint(draft.resolved)
    assert draft.resolved["architecture"] == "Glm5NextForConditionalGeneration"


def test_runtime_resolved_layout_keeps_precision_and_memory_overrides():
    draft = derive_profile(load_model_config(GLM), _glm_request(), _runtime(weights_bytes=1000, activations_bytes=2000))
    assert draft.profile is None and set(draft.pending_runtime) == {"cache_geometry"}
    assert draft.resolved["weights_bytes"] == 1000 and draft.resolved["kv_cache_dtype"] == "bfloat16"
    assert draft.sources["weights_bytes"].startswith("user override")
    assert draft.missing == {}


def test_recognized_layout_reports_nothing_pending(tmp_path):
    draft = derive_profile(_config(tmp_path), _request(), _runtime())
    assert draft.profile is not None and draft.pending_runtime == {}


@pytest.mark.parametrize(
    "marker",
    [
        {"layer_types": ["full_attention", "linear_attention"]},
        {"ssm_cfg": {}},
        {"ssm_state_size": 128},
        {"attn_type_list": [1, 0]},
    ],
)
def test_full_attention_architecture_with_runtime_resolved_layout_assumes_no_linear_cache(tmp_path, marker):
    # Llama is a known full-attention architecture; the marker says this checkpoint is not that layout.
    draft = derive_profile(_config(tmp_path, **marker), _request(), _runtime())
    assert draft.profile is None and set(draft.pending_runtime) == {"cache_geometry"}
    assert {"cache_layout", "kv_bytes_per_token"}.isdisjoint(draft.resolved | draft.missing)


_CHECKED_BY_IMPORT = "user override; onboard import-observations checks it against the runtime probe's measurement"
# Every non-KV memory field declared: the declared-memory path must still not trust typed geometry.
_DECLARED_MEMORY = {"weights_bytes": 1, "activations_bytes": 1, "runtime_overhead_bytes": 1, "comm_overhead_bytes": 1}


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("memory", [{}, _DECLARED_MEMORY])
@pytest.mark.parametrize(
    "typed",
    [
        {"cache_layout": "linear"},
        {"kv_bytes_per_token": 6000},
        {"cache_layout": "linear", "kv_bytes_per_token": 6000},
        {"cache_layout": "grouped", "cache_groups": _GROUPED_OBSERVATION["cache_groups"]},
    ],
)
def test_runtime_resolved_layout_records_typed_geometry_as_a_claim(preview, memory, typed):
    draft = derive_profile(load_model_config(GLM), None if preview else _glm_request(), _runtime(**memory, **typed))
    # A claim never completes a profile, and never changes the classification.
    assert draft.profile is None
    assert draft.pending_runtime == derive_profile(load_model_config(GLM), None).pending_runtime
    for key, value in validate_overrides(typed).items():
        assert draft.resolved[key] == value
        assert draft.sources[key] == _CHECKED_BY_IMPORT
    assert not {"cache_layout", "kv_bytes_per_token", "cache_groups", "cache_block_sizes"} & draft.missing.keys()


@pytest.mark.parametrize(
    "typed,message",
    [
        ({"cache_layout": "hybrid"}, "unknown cache_layout"),
        ({"kv_bytes_per_token": 0}, "kv_bytes_per_token must be an integer"),
        ({"cache_groups": []}, "cache_groups must be a nonempty list"),
        (
            {"cache_layout": "linear", "cache_groups": _GROUPED_OBSERVATION["cache_groups"]},
            "cache_groups cannot be combined with linear cache_layout",
        ),
        (
            {"cache_layout": "grouped", "kv_bytes_per_token": 6000},
            "kv_bytes_per_token cannot be combined with grouped cache_layout",
        ),
    ],
)
def test_runtime_resolved_layout_validates_typed_geometry_like_any_override(typed, message):
    with pytest.raises(ValueError, match=message):
        derive_profile(load_model_config(GLM), _glm_request(), typed)


@pytest.mark.parametrize("groups", [False, True], ids=["no-cache-groups", "cache-groups"])
@pytest.mark.parametrize("kv_bytes", [False, True], ids=["no-kv-bytes", "kv-bytes"])
@pytest.mark.parametrize("layout", [None, "linear", "grouped"], ids=["no-layout", "linear-layout", "grouped-layout"])
def test_overrides_that_state_different_cache_layouts_are_refused(layout, kv_bytes, groups):
    overrides = {
        **({"cache_layout": layout} if layout else {}),
        **({"kv_bytes_per_token": 6000} if kv_bytes else {}),
        **({"cache_groups": _GROUPED_OBSERVATION["cache_groups"]} if groups else {}),
    }
    # Each override states a layout: cache_layout its value, kv_bytes_per_token linear, cache_groups grouped.
    stated = {state for state in (layout, "linear" if kv_bytes else None, "grouped" if groups else None) if state}
    if len(stated) > 1:
        with pytest.raises(ValueError, match="cannot be combined"):
            validate_overrides(overrides)
    else:
        # The values are already in their validated form.
        assert validate_overrides(overrides) == overrides


@pytest.mark.parametrize("preview", [False, True])
# The refusal is decided by key, so it wins whatever the value is.
@pytest.mark.parametrize("blocks", [{"full_attention": 64}, {"unknown_group": 0}])
def test_runtime_resolved_layout_has_no_cache_groups_for_block_sizes(preview, blocks):
    with pytest.raises(ValueError) as error:
        derive_profile(load_model_config(GLM), None if preview else _glm_request(), {"cache_block_sizes": blocks})
    message = str(error.value)
    assert "cache_block_sizes names config-derived cache groups, and this layout has none" in message
    assert "onboard probe-runtime reports the runtime cache groups with their block sizes" in message
    assert "layer_types linear_attention x34" in message


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("preview", [False, True])
def test_architecture_override_cannot_bypass_runtime_resolved_geometry(tmp_path, preview, nested):
    config = _config(tmp_path, architectures=None, model_type="unknown")
    if nested:
        config = _config(tmp_path, architectures=None, model_type="unknown_wrapper", text_config=config.raw)
    assert config.runtime_layout == ()
    request = None if preview else _request()
    overrides = _runtime(architecture="MambaForCausalLM", num_experts=0)
    draft = derive_profile(config, request, overrides)
    assert draft.profile is None and set(draft.pending_runtime) == {"cache_geometry"}
    assert "recurrent architecture MambaForCausalLM" in draft.pending_runtime["cache_geometry"]
    # Typed geometry with every non-KV memory field declared is still only a claim for the import to check.
    typed = derive_profile(
        config, request, {**overrides, **_DECLARED_MEMORY, "kv_bytes_per_token": 1, "cache_layout": "linear"}
    )
    assert typed.profile is None and typed.pending_runtime == draft.pending_runtime
    assert typed.sources["kv_bytes_per_token"] == typed.sources["cache_layout"] == _CHECKED_BY_IMPORT
    # Declaring the same architecture in the original config is classified identically.
    declared = _config(tmp_path, architectures=["MambaForCausalLM"], model_type="unknown", num_experts=0)
    assert declared.runtime_layout == ("recurrent architecture MambaForCausalLM",)
    assert derive_profile(declared, request, _runtime()).pending_runtime == draft.pending_runtime


@pytest.mark.parametrize(
    "architecture,pending",
    [
        ("LlamaForCausalLM", None),
        ("MambaForCausalLM", "recurrent architecture MambaForCausalLM"),
        ("MiMoForCausalLM", "unrecognized architecture MiMoForCausalLM"),
    ],
)
def test_cache_layout_note_follows_an_architecture_override(tmp_path, architecture, pending):
    # The config names no architecture, so its own notes cannot say how an override classifies the layout.
    config = _config(tmp_path, architectures=None, model_type="unknown")
    assert "cache_layout" not in config.notes
    draft = derive_profile(config, _request(), _runtime(architecture=architecture, num_experts=0))
    if pending is None:
        assert draft.profile is not None and "cache_layout" not in draft.notes
        assert "cache_layout" not in json.loads(draft.profile.provenance)["config_notes"]
    else:
        assert draft.notes["cache_layout"].startswith(f"cache layout pending the runtime probe ({pending})")
        assert pending in draft.pending_runtime["cache_geometry"]
    assert {key: value for key, value in draft.notes.items() if key != "cache_layout"} == config.notes


def test_cache_layout_note_is_not_pending_when_an_override_recognizes_the_layout(tmp_path):
    # Sparse-attention layer types are recognized only for the DSA architectures, so the config's own note is stale.
    config = _config(tmp_path, architectures=None, model_type="unknown", layer_types=["deepseek_sparse_attention"] * 2)
    assert config.notes["cache_layout"].startswith("cache layout pending the runtime probe")
    overrides = _runtime(
        architecture="GlmMoeDsaForCausalLM", num_experts=4, cache_layout="linear", kv_bytes_per_token=64
    )
    draft = derive_profile(config, _request("moe"), overrides)
    assert draft.pending_runtime == {} and draft.profile is not None
    assert "cache_layout" not in draft.notes
    assert "cache_layout" not in json.loads(draft.profile.provenance)["config_notes"]


def test_explicit_decoder_layout_is_not_overridden_by_inkling_wrapper(tmp_path):
    decoder = _config(tmp_path).raw
    config = _config(tmp_path, architectures=["InklingForConditionalGeneration"], text_config=decoder)
    assert config.decoder_architecture == "LlamaForCausalLM"
    assert config.suggestions["architecture"] == "LlamaForCausalLM"


def test_config_hash_and_identity_are_content_facts_not_checkpoint_revision(tmp_path):
    config = _config(tmp_path, _name_or_path="source/repository", auto_map={"AutoConfig": "do_not_execute.Config"})
    assert config.sha256 == hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    assert config.suggestions == {
        "model": "source/repository",
        "architecture": "LlamaForCausalLM",
        "model_kind": "dense",
        "context_length": 2048,
        "num_experts": 0,
    }
    draft = derive_profile(config, _request(), _runtime())
    assert draft.profile is not None
    assert draft.profile.model == "example/configured-model"
    assert draft.profile.model_revision == "checkpoint-123"
    assert config.sha256 in draft.profile.provenance
    assert config.sha256 in draft.profile.deployments[0].resources.provenance
    assert "auto_map" in config.notes


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("declared_architecture", [False, True])
def test_nested_text_geometry_and_architecture_are_independent_of_wrapper_and_encoders(
    tmp_path, kind, declared_architecture
):
    text = _config(tmp_path).raw
    if kind == "moe":
        text.update(architectures=["MixtralForCausalLM"], model_type="mixtral", num_local_experts=4)
    architecture = text["architectures"][0]
    baseline = derive_profile(_config(tmp_path, **text), _request(kind), _runtime())
    if not declared_architecture:
        del text["architectures"]
    config = _config(
        tmp_path,
        architectures=["ExampleMultimodalForConditionalGeneration"],
        model_type="example_multimodal",
        _name_or_path="example/original-checkpoint",
        hidden_size=1024,
        n_layer=99,
        max_position_embeddings=8192,
        num_experts=64,
        text_config={**text, "_name_or_path": "example/decoder-base"},
        vision_config={"hidden_size": 512, "num_hidden_layers": 20, "num_experts": 8},
        audio_config={"hidden_size": 256, "num_hidden_layers": 10},
    )

    draft = derive_profile(config, _request(kind), _runtime())

    published_architecture = architecture if declared_architecture else "ExampleMultimodalForConditionalGeneration"
    assert draft.resolved == {**baseline.resolved, "architecture": published_architecture}
    assert draft.profile.architecture == published_architecture
    assert config.decoder_architecture == architecture
    assert config.suggestions["model"] == "example/original-checkpoint"
    assert config.sha256 == hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    notes = json.loads(draft.profile.provenance)["config_notes"]
    assert "text_config" in notes["decoder_config"]
    assert f"decoder architecture={architecture}" in notes["decoder_architecture"]
    assert "Text decoder only" in notes["modeling_scope"]
    assert "full multimodal latency is not modeled" in notes["modeling_scope"]
    assert json.loads(draft.profile.deployments[0].resources.provenance)["config_notes"] == notes


def test_nested_config_inherits_only_shared_precision_metadata_and_original_identity(tmp_path):
    text = _config(tmp_path).raw
    del text["torch_dtype"]
    config = _config(
        tmp_path,
        _name_or_path="example/quantized-checkpoint",
        auto_map={"AutoConfig": "do_not_execute.Config"},
        text_config={**text, "_name_or_path": "example/decoder-base"},
        quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4", "kv_cache_scheme": "FP8"},
    )
    draft = derive_profile(config, _request(), _runtime(weights_bytes=1024))
    assert draft.profile is not None
    assert draft.resolved["gemm_quant_mode"] == draft.resolved["moe_quant_mode"] == "nvfp4"
    assert config.raw["torch_dtype"] == "bfloat16"
    assert config.raw["quantization_config"]["quant_algo"] == "NVFP4"
    assert config.suggestions["model"] == "example/quantized-checkpoint"
    assert "torch_dtype, quantization_config inherited" in config.notes["shared_metadata"]
    assert "auto_map" in config.notes
    assert config.sha256 in draft.profile.provenance


def test_nested_precision_metadata_takes_precedence_as_a_group(tmp_path):
    text = _config(tmp_path).raw
    del text["torch_dtype"]
    config = _config(
        tmp_path,
        text_config={**text, "dtype": "float16", "quantization_config": {"quant_method": "fp8"}},
        quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4"},
    )
    draft = derive_profile(config, _request())
    assert draft.resolved["gemm_quant_mode"] == "fp8_static"
    assert draft.resolved["moe_quant_mode"] == "fp8"
    assert "torch_dtype" not in config.raw
    assert "config tensor dtype=float16" in config.notes["dtype"]
    assert "shared_metadata" not in config.notes


@pytest.mark.parametrize("metadata", [{"hf_quant_config": {"format": "custom"}}, {"quant_algo": "CUSTOM"}])
def test_shared_unrecognized_quantization_prevents_bfloat16_storage_assumptions(tmp_path, metadata):
    config = _config(tmp_path, text_config=_config(tmp_path).raw, **metadata)
    draft = derive_profile(config, _request())
    assert {"gemm_quant_mode", "moe_quant_mode"} <= draft.missing.keys()
    assert {"weights_bytes"}.isdisjoint(draft.resolved)
    for key, value in metadata.items():
        assert config.raw[key] == value
        assert key in config.notes["shared_metadata"]


@pytest.mark.parametrize("metadata", [{"hf_quant_config": {"format": "custom"}}, {"quant_algo": "CUSTOM"}])
def test_nested_unrecognized_quantization_is_not_replaced_by_shared_metadata(tmp_path, metadata):
    config = _config(
        tmp_path,
        text_config={**_config(tmp_path).raw, **metadata},
        quantization_config={"quant_method": "fp8"},
    )
    draft = derive_profile(config, _request())
    assert {"gemm_quant_mode", "moe_quant_mode"} <= draft.missing.keys()
    assert {"weights_bytes"}.isdisjoint(draft.resolved)
    assert "quantization_config" not in config.raw


@pytest.mark.parametrize("model_type", [None, "example_custom_text"])
def test_unknown_nested_decoder_retains_wrapper_identity_and_requires_resource_bounds(tmp_path, model_type):
    text = _config(tmp_path).raw
    for key in ("architectures", "model_type", "max_position_embeddings"):
        del text[key]
    text.update(model_max_length=4096, n_routed_experts=4, local_layer_ids=[0], sliding_window_size=128)
    if model_type:
        text["model_type"] = model_type
    config = _config(
        tmp_path,
        architectures=["ExampleMultimodalForConditionalGeneration"],
        model_type="example_multimodal",
        text_config=text,
    )
    draft = derive_profile(config, _request("moe"), _runtime())
    assert draft.resolved["architecture"] == "ExampleMultimodalForConditionalGeneration"
    assert draft.resolved["context_length"] == 4096
    assert draft.resolved["num_experts"] == 4
    assert {"cache_block_sizes"} <= draft.missing.keys()
    assert {"weights_bytes", "activations_bytes"}.isdisjoint(draft.resolved)
    assert draft.resolved["cache_layout"] == "grouped"
    assert "wrapper architecture" in draft.sources["architecture"]
    assert "assumption" in draft.sources["gemm_quant_mode"]
    with pytest.raises(ValueError, match="unsupported linear cache"):
        derive_profile(config, _request("moe"), _runtime(kv_bytes_per_token=64, cache_layout="linear"))
    complete = derive_profile(
        config,
        _request("moe"),
        _runtime(
            weights_bytes=1024,
            activations_bytes=2048,
            cache_block_sizes={"full_attention": 16, "sliding_attention": 16},
        ),
    )
    assert complete.profile is not None
    assert len(complete.profile.deployments[0].resources.cache_groups) == 2


def test_missing_text_geometry_does_not_fall_back_to_wrapper_dimensions(tmp_path):
    config = _config(tmp_path, text_config={"model_type": "llama", "model_max_length": 2048})
    draft = derive_profile(config, _request(), _runtime())
    assert {"kv_bytes_per_token"} <= draft.missing.keys()
    assert {"weights_bytes", "activations_bytes"}.isdisjoint(draft.resolved)
    assert "hidden_size" not in config.raw
    assert "weights_bytes" not in draft.missing


def test_known_wrapper_does_not_establish_unknown_nested_decoder_resource_layout(tmp_path):
    text = {key: value for key, value in _config(tmp_path).raw.items() if key not in ("architectures", "model_type")}
    config = _config(tmp_path, text_config=text)
    draft = derive_profile(config, _request(), _runtime())
    assert draft.resolved["architecture"] == "LlamaForCausalLM"
    assert config.decoder_architecture is None
    assert {"num_experts", "kv_bytes_per_token", "cache_layout"} <= draft.missing.keys()
    assert {"weights_bytes", "activations_bytes"}.isdisjoint(draft.resolved)


@pytest.mark.parametrize("text", [None, {}, [], "decoder", [{"hidden_size": 16}], {"text_config": {"hidden_size": 16}}])
def test_malformed_or_ambiguous_text_sections_have_actionable_diagnostics(tmp_path, text):
    with pytest.raises(ValueError, match="text_config.*decoder.*--fpm-profile"):
        _config(tmp_path, text_config=text)


def test_nested_architecture_ambiguity_is_not_hidden_by_wrapper_fallback(tmp_path):
    text = {**_config(tmp_path).raw, "architectures": ["LlamaForCausalLM", "MixtralForCausalLM"]}
    with pytest.raises(ValueError, match="exactly one decoder architecture"):
        _config(tmp_path, text_config=text)


def test_flat_multimodal_metadata_does_not_change_decoder_estimates(tmp_path):
    baseline = derive_profile(_config(tmp_path), _request(), _runtime())
    config = _config(
        tmp_path,
        vision_config={"hidden_size": 512, "num_hidden_layers": 20},
        audio_config={"hidden_size": 256, "num_hidden_layers": 10},
    )
    draft = derive_profile(config, _request(), _runtime())
    assert draft.resolved == baseline.resolved
    assert "Text decoder only" in json.loads(draft.profile.provenance)["config_notes"]["modeling_scope"]


@pytest.mark.parametrize("tp,weights,kv", [(1, 13472, 64), (2, 6816, 32), (4, 3744, 32)])
def test_dense_tensor_counts_include_replicated_norms_and_kv_heads(tmp_path, tp, weights, kv):
    # TP1: 2 embedding/head matrices * 64*16 + 2 layers *
    # (Q/O 2*16*16 + K/V 2*16*8 + MLP 3*16*32 + norms 2*16) + final norm 16.
    draft = derive_profile(_config(tmp_path), _request(tensor_parallel=tp), _runtime())
    assert draft.missing == {}
    resources = draft.profile.deployments[0].resources
    assert resources.weights_bytes is None
    assert draft.resolved["weights_bytes"] == weights
    assert resources.kv_bytes_per_token == kv
    assert resources.max_num_tokens == 8192
    assert resources.max_batch_size == 256
    assert "estimate" in draft.sources["weights_bytes"]
    assert "exact" in draft.sources["kv_bytes_per_token"]
    assert "estimate" in draft.sources["runtime_overhead_bytes"]
    assert "sha256" in draft.sources["runtime_overhead_bytes"]


def test_weight_tying_biases_and_qk_norm_have_independent_counts(tmp_path):
    request = _request(tensor_parallel=2)
    base = derive_profile(_config(tmp_path), request, _runtime()).resolved["weights_bytes"]
    tied = derive_profile(_config(tmp_path, tie_word_embeddings=True), request, _runtime())
    assert tied.resolved["weights_bytes"] == base - 64 // 2 * 16 * 2
    biased = derive_profile(_config(tmp_path, attention_bias=True, mlp_bias=True), request, _runtime())
    # Per layer: QKV (8+4+4), output (16), gated MLP (16+16+16), all two-byte.
    assert biased.resolved["weights_bytes"] == base + 2 * (16 + 16 + 48) * 2
    qwen = derive_profile(
        _config(tmp_path, architectures=["Qwen3ForCausalLM"], model_type="qwen3"), request, _runtime()
    )
    assert qwen.resolved["weights_bytes"] == base + 2 * (2 * 4) * 2


@pytest.mark.parametrize("tp,expected_bias_bytes", [(1, 128), (2, 64), (4, 48)])
def test_qwen3_attention_bias_counts_qkv_without_output_bias(tmp_path, tp, expected_bias_bytes):
    request = _request(tensor_parallel=tp)
    facts = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}
    base = derive_profile(_config(tmp_path, **facts), request, _runtime())
    biased = derive_profile(_config(tmp_path, **facts, attention_bias=True), request, _runtime())
    # Two layers, two-byte Q/K/V biases: widths (16,8,8), (8,4,4), or (4,4,4).
    # The output projection is bias-free; KV heads replicate at TP4.
    assert biased.resolved["weights_bytes"] - base.resolved["weights_bytes"] == expected_bias_bytes


@pytest.mark.parametrize(
    "topology,weights,kv,parallel",
    [
        ({"tensor_parallel": 2}, 16288, 32, (2, 1, 1, 2, 1, 1)),
        (
            {"attention_data_parallel": 2, "moe_tensor_parallel": 1, "moe_expert_parallel": 2},
            19872,
            64,
            (1, 1, 2, 1, 2, 1),
        ),
        ({"tensor_parallel": 2, "moe_tensor_parallel": 1, "moe_expert_parallel": 2}, 16288, 32, (2, 1, 1, 1, 2, 1)),
    ],
)
def test_moe_resources_are_rank_local_for_tp_dep_and_tep(tmp_path, topology, weights, kv, parallel):
    config = _config(tmp_path, architectures=["MixtralForCausalLM"], model_type="mixtral", num_local_experts=4)
    draft = derive_profile(config, _request("moe", **topology), _runtime(comm_overhead_bytes=321))
    assert draft.missing == {}
    deployment = draft.profile.deployments[0]
    assert deployment.parallel_tuple == parallel
    assert deployment.resources.weights_bytes is None
    assert draft.resolved["weights_bytes"] == weights
    assert deployment.resources.kv_bytes_per_token == kv
    assert deployment.resources.comm_overhead_bytes is None
    assert draft.resolved["comm_overhead_bytes"] == 321
    assert deployment.resources.memory_source == "pending"


def test_activation_estimate_uses_declared_scheduler_envelope(tmp_path):
    config = _config(tmp_path, hidden_size=4096, intermediate_size=8192, num_attention_heads=32, head_dim=128)
    small = derive_profile(config, _request(tensor_parallel=2), _runtime(max_num_tokens=16))
    large = derive_profile(config, _request(tensor_parallel=2), _runtime(max_num_tokens=4096))
    assert small.resolved["activations_bytes"] == 70 * 1024 * 1024
    assert large.resolved["activations_bytes"] == 2 * 4096 * 4096 * 6.5
    assert "estimate" in large.sources["activations_bytes"]
    assert "max_num_tokens=4096" in large.sources["activations_bytes"]


def test_mha_and_weight_padding_have_separate_supported_domains(tmp_path):
    draft = derive_profile(_config(tmp_path, num_key_value_heads=4, vocab_size=65), _request(), _runtime())
    assert draft.resolved["kv_bytes_per_token"] == 128
    assert "weights_bytes" not in draft.resolved
    assert "weights_bytes" not in draft.missing
    assert draft.profile.deployments[0].resources.memory_source == "pending"


@pytest.mark.parametrize("field", ["gemm_quant_mode", "kv_cache_dtype", "weights_bytes", "activations_bytes"])
def test_explicit_overrides_survive_derivation_and_are_recorded_per_field(tmp_path, field):
    value = "fp8" if "mode" in field or "dtype" in field else 123456
    draft = derive_profile(_config(tmp_path), _request(), _runtime(**{field: value}))
    assert draft.resolved[field] == value
    assert "user override" in draft.sources[field]


def test_hardware_defaults_do_not_reuse_tp_communication_for_dep(tmp_path):
    config = _config(tmp_path, architectures=["MixtralForCausalLM"], model_type="mixtral", num_local_experts=4)
    request = _request("moe", attention_data_parallel=2, moe_tensor_parallel=1, moe_expert_parallel=2)
    draft = derive_profile(config, request, _runtime())
    assert "comm_overhead_bytes" not in draft.resolved
    assert "comm_overhead_bytes" not in draft.missing
    assert draft.resolved["runtime_overhead_bytes"] == 3758096384


def test_unknown_hardware_has_no_fallback_reservations(tmp_path):
    request = _request()
    request.identity.gpu = "custom_gpu"
    draft = derive_profile(_config(tmp_path), request, _runtime())
    assert {"comm_overhead_bytes", "runtime_overhead_bytes"}.isdisjoint(draft.resolved)


def test_dynamic_kv_scales_need_explicit_accounting(tmp_path):
    config = _config(
        tmp_path,
        quantization_config={
            "quant_method": "fp8",
            "kv_cache_scheme": {"num_bits": 8, "type": "float", "dynamic": True},
        },
    )
    draft = derive_profile(config, _request())
    assert draft.resolved["kv_cache_dtype"] == "fp8"
    assert "kv_bytes_per_token" in draft.missing
    assert "scales" in draft.missing["kv_bytes_per_token"]


@pytest.mark.parametrize("cache_type", [[], {}, True, 8, 1.5, "", "\x00"])
def test_malformed_kv_cache_type_is_rejected_at_config_load(tmp_path, cache_type):
    with pytest.raises(ValueError, match=r"quantization_config\.kv_cache_scheme\.type"):
        _config(tmp_path, quantization_config={"kv_cache_scheme": {"num_bits": 8, "type": cache_type}})


@pytest.mark.parametrize(
    "scheme,dtype",
    [
        ("FP8", "fp8"),
        ({"num_bits": 8, "type": "float"}, "fp8"),
        ({"num_bits": 8, "type": "int"}, "int8"),
        ({"num_bits": 8, "type": None}, None),
        ({"num_bits": 8}, None),
        ({"num_bits": 8, "type": "unknown"}, None),
    ],
)
def test_kv_cache_type_preserves_known_precision_and_unresolved_metadata(tmp_path, scheme, dtype):
    config = _config(tmp_path, quantization_config={"kv_cache_scheme": scheme})
    draft = derive_profile(config, _request())
    if dtype is None:
        assert "kv_cache_dtype" not in draft.resolved
        assert "kv_cache_dtype" in draft.missing
    else:
        assert draft.resolved["kv_cache_dtype"] == dtype
        assert draft.resolved["kv_bytes_per_token"] == 32


def test_weight_quantization_does_not_select_runtime_attention_or_kv_dtype(tmp_path):
    config = _config(tmp_path, quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]})
    draft = derive_profile(config, _request())
    assert draft.resolved["gemm_quant_mode"] == "fp8_block"
    assert draft.resolved["moe_quant_mode"] == "fp8_block"
    assert {"fmha_quant_mode", "kv_cache_dtype"} <= draft.missing.keys()
    assert {"weights_bytes"}.isdisjoint(draft.resolved)
    assert draft.resolved["comm_quant_mode"] == "half"
    assert "collector FPM identity default" in draft.sources["comm_quant_mode"]
    assert draft.profile is None
    assert "weights_bytes" not in draft.missing


@pytest.mark.parametrize(
    "quant,gemm,moe",
    [
        ({"quant_method": "fp8"}, "fp8_static", "fp8"),
        ({"quant_method": "fp8", "activation_scheme": "static"}, "fp8_static", "fp8"),
        ({"quant_method": "fp8", "activation_scheme": "dynamic"}, "fp8", "fp8"),
        (
            {
                "quant_method": "modelopt",
                "quant_algo": "FP8",
                "config_groups": {"g": {"input_activations": {"dynamic": True}}},
            },
            "fp8",
            "fp8",
        ),
        ({"quant_method": "fp8", "weight_block_size": [128, 128]}, "fp8_block", "fp8_block"),
        ({"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["self_attn*"]}, "nvfp4", "nvfp4"),
    ],
)
def test_config_quantization_uses_existing_fpm_checkpoint_identity(tmp_path, quant, gemm, moe):
    from aisimulate.support.fpm import fpm_cli_args
    from aisimulate_core.sdk.models.helpers import _infer_quant_modes_from_raw_config
    from aisimulate_core.sdk.utils import _infer_quantization_fields

    config = _config(tmp_path, quantization_config=quant)
    draft = derive_profile(config, _request(), _runtime(weights_bytes=1024))
    assert draft.resolved["gemm_quant_mode"] == gemm
    assert draft.resolved["moe_quant_mode"] == moe
    # Compare the real collector's identity helper, not a duplicate test formula.
    existing = _infer_quant_modes_from_raw_config({**config.raw, **_infer_quantization_fields(config.raw)})
    assert existing["gemm_quant_mode"].name == gemm
    assert existing["moe_quant_mode"].name == moe
    request = SupportRequest.model_validate({**_request().model_dump(), "fpm_profile": draft.profile})
    command = fpm_cli_args(request, output_dir=tmp_path / "plan", plan_only=True)
    assert command[command.index("--fpm-weight-quantizations") + 1] == gemm
    assert draft.profile.deployments[0].match_identity()[:2] == [gemm, moe]
    assert "checkpoint/FPM" in draft.sources["gemm_quant_mode"]


def test_unknown_quantization_never_falls_back_to_residual_bfloat16(tmp_path):
    config = _config(tmp_path, quantization_config={"quant_method": "custom", "quant_algo": "custom"})
    draft = derive_profile(config, _request())
    assert {"gemm_quant_mode", "moe_quant_mode"} <= set(draft.missing)
    assert {"weights_bytes"}.isdisjoint(draft.resolved)


def test_unknown_architecture_is_pending_and_still_asks_for_user_owned_inputs(tmp_path):
    config = _config(tmp_path, architectures=["NewDecoderForCausalLM"], model_type="new_decoder")
    assert "num_experts" not in config.suggestions
    assert config.runtime_layout == ("unrecognized architecture NewDecoderForCausalLM",)
    draft = derive_profile(config, _request())
    assert set(draft.pending_runtime) == {"cache_geometry"}
    assert {"num_experts", "fmha_quant_mode", "kv_cache_dtype"} <= set(draft.missing)
    assert {"kv_bytes_per_token", "cache_layout"}.isdisjoint(draft.missing)
    assert {"weights_bytes", "activations_bytes"}.isdisjoint(draft.resolved)
    explicit = derive_profile(
        config,
        _request(),
        _runtime(
            num_experts=0, weights_bytes=1000, activations_bytes=2000, kv_bytes_per_token=32, cache_layout="linear"
        ),
    )
    assert explicit.profile is None and explicit.missing == {}
    assert "user override" in explicit.sources["weights_bytes"]
    assert explicit.sources["kv_bytes_per_token"] == _CHECKED_BY_IMPORT


def test_missing_fields_are_complete_and_mla_layout_can_use_explicit_resources(tmp_path):
    # The MLA/DSA families keep explicit cache accounting, so explicit resources still complete their profile.
    config = _config(tmp_path, architectures=["DeepseekV3ForCausalLM"], model_type="deepseek_v3")
    assert "num_experts" not in config.suggestions and config.runtime_layout == ()
    draft = derive_profile(config, _request("moe"))
    assert {"num_experts", "kv_bytes_per_token", "cache_layout"} <= set(draft.missing)
    assert {"weights_bytes", "activations_bytes"}.isdisjoint(draft.resolved)
    complete = derive_profile(
        config,
        _request("moe"),
        _runtime(
            num_experts=4, weights_bytes=1000, activations_bytes=2000, kv_bytes_per_token=32, cache_layout="linear"
        ),
    )
    assert complete.profile is not None
    assert "user override" in complete.sources["weights_bytes"]


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"hidden_size": True}, "hidden_size"),
        ({"num_hidden_layers": 2.0}, "num_hidden_layers"),
        ({"num_attention_heads": 0}, "num_attention_heads"),
        ({"architectures": ["LlamaForCausalLM", "OtherForCausalLM"]}, "architectures"),
        ({"n_layer": 4}, "conflicting"),
        ({"model_max_length": 4096}, "conflicting context_length"),
        ({"dtype": "float32"}, "conflicting"),
        ({"num_key_value_heads": 3}, "num_key_value_heads"),
        ({"attention_bias": "false"}, "attention_bias"),
        ({"model_type": "qwen3"}, "model_type"),
        ({"quantization_config": []}, "quantization_config"),
        ({"num_mtp_modules": -1}, "num_mtp_modules"),
        (
            {"quantization_config": {"config_groups": {"g": {"weights": {"dynamic": "false"}}}}},
            "dynamic",
        ),
    ],
)
def test_corrupt_or_ambiguous_config_fails_before_profile_overrides(tmp_path, updates, match):
    with pytest.raises(ValueError, match=match):
        _config(tmp_path, **updates)


@pytest.mark.parametrize("raw", ['{"hidden_size":16,"hidden_size":32}', '{"hidden_size":NaN}', "[]"])
def test_json_must_be_finite_object_with_unique_keys(tmp_path, raw):
    path = tmp_path / "config.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_model_config(path)


def _sidecar(tmp_path, **updates):
    sidecar = {
        "producer": {"name": "modelopt", "version": "0.0.0"},
        "quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"},
        **updates,
    }
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(sidecar))


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
@pytest.mark.parametrize("sidecar", [False, True])
def test_unread_non_finite_numbers_load(tmp_path, value, sidecar):
    # NemotronH configs spell time_step_limit as [0.0, Infinity]; intake never reads that field.
    if sidecar:
        _sidecar(tmp_path, calibration_range=[0.0, value])
    config = _config(tmp_path, time_step_limit=[0.0, value])
    assert config.source_files.keys() == ({"hf_quant_config.json"} if sidecar else set())
    assert all("time_step_limit" not in note and "calibration_range" not in note for note in config.notes.values())


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
@pytest.mark.parametrize(
    "updates,field",
    [
        ({"max_position_embeddings": "VALUE"}, "max_position_embeddings"),
        ({"hidden_size": "VALUE"}, "hidden_size"),
        ({"sliding_window": "VALUE"}, "sliding_window"),
        ({"layer_types": ["full_attention", "VALUE"]}, "layer_types"),
        # Fields read without an integer, string or boolean check, including echoed metadata.
        ({"use_mtp": "VALUE"}, "use_mtp"),
        ({"compress_ratios": [4, "VALUE"]}, r"compress_ratios\[1\]"),
        ({"max_window_layers": "VALUE"}, "max_window_layers"),
        ({"hybrid_layer_pattern": [0, "VALUE"]}, r"hybrid_layer_pattern\[1\]"),
        (
            {"quantization_config": {"kv_cache_scheme": {"num_bits": 8, "type": "float", "group_size": "VALUE"}}},
            r"quantization_config\.kv_cache_scheme\.group_size",
        ),
    ],
)
def test_non_finite_numbers_in_read_fields_name_the_field(tmp_path, value, updates, field):
    updates = json.loads(json.dumps(updates).replace('"VALUE"', json.dumps(value)))
    with pytest.raises(ValueError, match=field):
        _config(tmp_path, **updates)


@pytest.mark.parametrize("value", [math.inf, math.nan])
def test_non_finite_numbers_in_the_read_sidecar_section_name_the_field(tmp_path, value):
    _sidecar(tmp_path, quantization={"quant_algo": "FP8", "group_size": value})
    with pytest.raises(ValueError, match=r"hf_quant_config\.json quantization\.group_size"):
        _config(tmp_path)


def test_an_unlisted_non_finite_read_fails_closed(tmp_path, monkeypatch):
    # If a future read is missing from _UNTYPED_READ_FIELDS, intake's JSON output refuses the value instead of
    # writing NaN into a note.
    unlisted = tuple(field for field in config_profile._UNTYPED_READ_FIELDS if field != "quantization_config")
    monkeypatch.setattr(config_profile, "_UNTYPED_READ_FIELDS", unlisted)
    scheme = {"num_bits": 8, "type": "float", "group_size": math.nan}
    with pytest.raises(ValueError, match="not JSON compliant"):
        _config(tmp_path, quantization_config={"quant_method": "fp8", "kv_cache_scheme": scheme})


def test_unread_infinity_never_reaches_profile_provenance(tmp_path):
    def reject(constant):
        raise AssertionError(f"non-finite {constant} reached an output")

    _sidecar(tmp_path, calibration_range=[0.0, math.inf])
    config = _config(tmp_path, time_step_limit=[0.0, math.inf])
    draft = derive_profile(config, _request(), _runtime(weights_bytes=1024))
    assert draft.profile is not None
    for provenance in (draft.profile.provenance, draft.profile.deployments[0].resources.provenance):
        json.loads(provenance, parse_constant=reject)
    json.loads(draft.profile.model_dump_json(), parse_constant=reject)
    json.loads(
        json.dumps({"resolved": draft.resolved, "sources": draft.sources, "notes": draft.notes}), parse_constant=reject
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"weights_byte": 7},
        {"weights_bytes": True},
        {"weights_bytes": 7.0},
        {"weights_bytes": -1},
        {"weights_bytes": 2**53 + 1},
        {"max_num_tokens": 0},
        {"kv_bytes_per_token": 0},
        {"num_experts": -1},
        {"cache_layout": "hybrid"},
        {"gemm_quant_mode": "fp16"},
        {"kv_cache_dtype": "auto"},
        {"provenance": " "},
        {"context_length": "2048"},
        {"weights_bytes": 2**53, "activations_bytes": 1},
    ],
)
def test_overrides_are_strict_even_before_identity_is_complete(tmp_path, overrides):
    with pytest.raises(ValueError):
        derive_profile(_config(tmp_path), None, overrides)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "updates,match",
    [
        ({"sliding_window": 128}, "unsupported"),
        ({"is_encoder_decoder": True}, "unsupported"),
        ({"kv_lora_rank": 8}, "unsupported"),
    ],
)
def test_known_incompatible_state_cannot_be_papered_over_with_overrides(tmp_path, updates, match, nested):
    if nested:
        updates = {"text_config": {**_config(tmp_path).raw, **updates}}
    with pytest.raises(ValueError, match=match):
        config = _config(tmp_path, **updates)
        derive_profile(config, _request(), _runtime(weights_bytes=1, kv_bytes_per_token=1, cache_layout="linear"))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "updates", [{"layer_types": ["full_attention", "linear_attention"]}, {"ssm_cfg": {"d_state": 16}}]
)
def test_runtime_resolved_state_cannot_be_papered_over_with_typed_geometry(tmp_path, updates, nested):
    # These layouts load, and their geometry is measured by the runtime probe: typed geometry is only a claim.
    if nested:
        updates = {"text_config": {**_config(tmp_path).raw, **updates}}
    config = _config(tmp_path, **updates)
    draft = derive_profile(
        config, _request(), _runtime(**_DECLARED_MEMORY, kv_bytes_per_token=1, cache_layout="linear")
    )
    assert draft.profile is None and set(draft.pending_runtime) == {"cache_geometry"}
    assert draft.sources["kv_bytes_per_token"] == _CHECKED_BY_IMPORT


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("explicit_resources", [False, True])
@pytest.mark.parametrize(
    "facts,architecture,experts,match",
    [
        ({"num_experts": 4}, "LlamaForCausalLM", 4, "dense architecture"),
        ({}, "MixtralForCausalLM", 0, "MoE architecture"),
        ({"kv_lora_rank": 8}, "LlamaForCausalLM", 0, "cache/projection modifiers"),
        ({"num_experts_per_tok": 8}, "MixtralForCausalLM", 4, "routed expert count"),
    ],
)
def test_late_architecture_and_expert_inputs_cannot_bypass_structural_guards(
    tmp_path, preview, explicit_resources, facts, architecture, experts, match, nested
):
    config = _config(tmp_path, architectures=None, model_type="unknown", **facts)
    if nested:
        config = _config(tmp_path, architectures=None, model_type="unknown_wrapper", text_config=config.raw)
    overrides = _runtime(architecture=architecture, num_experts=experts)
    if explicit_resources:
        overrides.update(weights_bytes=1, kv_bytes_per_token=1, cache_layout="linear")
    with pytest.raises(ValueError, match=match):
        derive_profile(config, None if preview else _request("moe" if experts else "dense"), overrides)
    # Supplying the same effective facts in the original config also fails.
    with pytest.raises(ValueError, match=match):
        _config(tmp_path, **{**facts, "architectures": [architecture], "model_type": "unknown", "num_experts": experts})


@pytest.mark.parametrize("field", ["num_experts_per_tok", "first_k_dense_replace"])
@pytest.mark.parametrize("late_experts", [False, True])
@pytest.mark.parametrize("preview", [False, True])
def test_optional_null_moe_metadata_matches_absent_metadata(tmp_path, field, late_experts, preview):
    facts = {"architectures": ["MixtralForCausalLM"], "model_type": "mixtral"}
    overrides = _runtime()
    if late_experts:
        overrides["num_experts"] = 4
    else:
        facts["num_local_experts"] = 4
    request = None if preview else _request("moe")
    absent = derive_profile(_config(tmp_path, **facts), request, overrides)
    config = _config(tmp_path, **facts, **{field: None})

    draft = derive_profile(config, request, overrides)

    assert config.raw[field] is None
    assert field not in config.notes
    assert draft.resolved == absent.resolved
    assert draft.missing == absent.missing
    assert draft.resolved["num_experts"] == 4
    assert (draft.profile is None) == preview


@pytest.mark.parametrize("field", ["num_experts_per_tok", "first_k_dense_replace"])
@pytest.mark.parametrize("value", [True, False, -1, 1.5, "1"])
def test_optional_moe_metadata_rejects_noninteger_or_negative_values(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        _config(tmp_path, **{field: value})


@pytest.mark.parametrize(
    "facts,match",
    [
        ({"num_experts_per_tok": 0}, "num_experts_per_tok"),
        ({"num_experts_per_tok": 5}, "routed expert count"),
        ({"first_k_dense_replace": 3}, "num_hidden_layers"),
    ],
)
def test_optional_moe_metadata_preserves_count_validation(tmp_path, facts, match):
    with pytest.raises(ValueError, match=match):
        _config(tmp_path, architectures=["MixtralForCausalLM"], model_type="mixtral", num_local_experts=4, **facts)


def test_bundled_modelopt_string_kv_scheme_preserves_usable_geometry():
    path = (
        Path(__file__).parents[2] / "src/aisimulate_core/model_configs/Qwen--Qwen3-32B-FP8-Static-PerTensor_config.json"
    )
    config = load_model_config(path)
    draft = derive_profile(config, _request(tensor_parallel=4))
    assert config.raw["quantization_config"]["kv_cache_scheme"] == "FP8"
    assert config.suggestions["architecture"] == "Qwen3ForCausalLM"
    assert draft.resolved["gemm_quant_mode"] == "fp8_static"
    assert draft.resolved["moe_quant_mode"] == draft.resolved["kv_cache_dtype"] == "fp8"
    assert draft.resolved["kv_bytes_per_token"] == 2 * 64 * 2 * 128
    assert {"fmha_quant_mode"} <= draft.missing.keys()
    assert {"weights_bytes", "activations_bytes"}.isdisjoint(draft.resolved)


def test_bundled_custom_layout_keeps_null_geometry_unresolved():
    path = (
        Path(__file__).parents[2]
        / "src/aisimulate_core/model_configs/nvidia--Llama-3_3-Nemotron-Super-49B-v1_config.json"
    )
    config = load_model_config(path)
    draft = derive_profile(config, _request(), _runtime())
    assert config.suggestions["architecture"] == "DeciLMForCausalLM"
    assert config.suggestions["context_length"] == 131072
    assert config.raw["intermediate_size"] is None
    assert "intermediate_size" not in config.notes
    assert "unrecognized architecture DeciLMForCausalLM" in draft.pending_runtime["cache_geometry"]
    assert "num_experts" in draft.missing and "kv_bytes_per_token" not in draft.missing
    assert {"weights_bytes", "activations_bytes"}.isdisjoint(draft.resolved)


def test_selected_topology_and_context_must_match_structural_config(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="attention heads"):
        derive_profile(config, _request(tensor_parallel=8), _runtime(weights_bytes=7, kv_bytes_per_token=7))
    with pytest.raises(ValueError, match="num_experts"):
        derive_profile(config, _request(), _runtime(num_experts=2))
    with pytest.raises(ValueError, match="context_length"):
        derive_profile(config, _request(), _runtime(context_length=4096))


@pytest.mark.parametrize(
    "filename,architecture,context,quant,kv",
    [
        ("MiniMaxAI--MiniMax-M2.7_config.json", "MiniMaxM2ForCausalLM", 204800, "fp8_block", None),
        ("nvidia--MiniMax-M2.7-NVFP4_config.json", "MiniMaxM2ForCausalLM", 196608, "nvfp4", "fp8"),
        ("zai-org--GLM-5.2_config.json", "GlmMoeDsaForCausalLM", 1048576, "bfloat16", None),
        ("nvidia--GLM-5.2-NVFP4_config.json", "GlmMoeDsaForCausalLM", 1048576, "nvfp4", "fp8"),
    ],
)
def test_real_minimax_and_glm_extract_independent_facts_without_naive_storage_rules(
    filename, architecture, context, quant, kv
):
    path = Path(__file__).parents[2] / "src/aisimulate_core/model_configs" / filename
    config = load_model_config(path)
    draft = derive_profile(config, _request("moe", tensor_parallel=4))
    assert config.suggestions["architecture"] == architecture
    assert config.suggestions["context_length"] == context
    assert config.suggestions["num_experts"] == 256
    assert config.suggestions["model_kind"] == "moe"
    assert draft.resolved["gemm_quant_mode"] == quant
    assert draft.resolved.get("kv_cache_dtype") == kv
    assert "fmha_quant_mode" in draft.missing
    assert "weights_bytes" not in draft.resolved
    assert "weights_bytes" not in draft.missing
    if architecture == "GlmMoeDsaForCausalLM":
        assert "kv_bytes_per_token" in draft.missing
        assert "DSA" in draft.missing["kv_bytes_per_token"]
    else:
        finished_kv = derive_profile(config, _request("moe", tensor_parallel=4), {"kv_cache_dtype": "bfloat16"})
        assert finished_kv.resolved["kv_bytes_per_token"] == 62 * 2 * 2 * 128 * 2


@pytest.mark.parametrize("nested", [False, True])
def test_config_route_does_not_import_model_construction_or_remote_code(tmp_path, monkeypatch, nested):
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        assert not name.startswith(("transformers", "huggingface_hub", "collector")), name
        assert ".sdk.models" not in name, name
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)
    config = _config(tmp_path)
    if nested:
        config = _config(tmp_path, text_config=config.raw, vision_config={"hidden_size": 512})
    draft = derive_profile(config, _request(), _runtime())
    assert draft.profile is not None


@pytest.mark.parametrize("mixed", [False, True])
def test_windowed_cache_derives_separate_group_pages_without_linear_rate(tmp_path, mixed):
    updates = {"sliding_window": 128}
    if mixed:
        updates["layer_types"] = ["full_attention", "sliding_attention"]
    config = _config(tmp_path, **updates)
    missing = derive_profile(config, _request(tensor_parallel=2), _runtime())
    assert missing.resolved["cache_layout"] == "grouped"
    assert "cache_block_sizes" in missing.missing
    assert "kv_bytes_per_token" not in missing.missing
    blocks = {"sliding_attention": 8, **({"full_attention": 16} if mixed else {})}
    draft = derive_profile(config, _request(tensor_parallel=2), _runtime(cache_block_sizes=blocks))
    assert draft.profile is not None
    resource = draft.profile.deployments[0].resources
    assert resource.kv_bytes_per_token is None
    groups = {group.name: group for group in resource.cache_groups}
    assert groups["sliding_attention"].page_size_bytes == (128 if mixed else 256)
    assert groups["sliding_attention"].sliding_window == 128
    if mixed:
        assert groups["full_attention"].page_size_bytes == 256
        assert groups["full_attention"].sliding_window is None
    serialized = draft.profile.model_dump(mode="json")
    assert "kv_bytes_per_token" not in serialized["deployments"][0]["resources"]
    assert type(draft.profile).model_validate(serialized) == draft.profile
    assert "minimum packed" in draft.sources["cache_groups"]


def _synthetic_inkling(tmp_path, **updates):
    # Original small geometry. Expected packing/TP constraints follow the pinned
    # Apache-2.0 vLLM implementation (copyright vLLM contributors):
    # https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/models/inkling/nvidia/sconv_swa_attn.py
    # Modified for metadata-only tests; no checkpoint or upstream fixture copied.
    text = {
        "hidden_size": 16,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "local_layer_ids": [0, 1, 2, 4, 5, 6],
        "swa_num_attention_heads": 8,
        "swa_num_key_value_heads": 4,
        "swa_head_dim": 4,
        "sliding_window_size": 128,
        "use_sconv": True,
        "sconv_kernel_size": 4,
        "model_max_length": 4096,
        "n_routed_experts": 8,
        "torch_dtype": "bfloat16",
        **updates,
    }
    return _config(
        tmp_path, architectures=["InklingForConditionalGeneration"], model_type="inkling_mm_model", text_config=text
    )


@pytest.mark.parametrize("dtype,attention_factor", [("bfloat16", 2), ("fp8", 1)])
@pytest.mark.parametrize(
    "topology",
    [
        {"tensor_parallel": 2},
        {"tensor_parallel": 2, "attention_data_parallel": 1, "moe_tensor_parallel": 1, "moe_expert_parallel": 2},
        {"tensor_parallel": 1, "attention_data_parallel": 8, "moe_tensor_parallel": 1, "moe_expert_parallel": 8},
    ],
)
def test_inkling_mixed_kv_and_bfloat16_convolution_are_rank_local(tmp_path, dtype, attention_factor, topology):
    config = _synthetic_inkling(tmp_path)
    draft = derive_profile(
        config,
        _request("moe", **topology),
        _runtime(
            kv_cache_dtype=dtype,
            weights_bytes=4096,
            activations_bytes=1024,
            comm_overhead_bytes=0,
            cache_block_sizes={"full_attention": 16, "sliding_attention": 8},
        ),
    )
    assert draft.profile is not None
    groups = {group.name: group for group in draft.profile.deployments[0].resources.cache_groups}
    rank_factor = 2 // topology["tensor_parallel"]
    assert groups["full_attention"].page_size_bytes == 256 * attention_factor * rank_factor
    assert groups["sliding_attention"].page_size_bytes == 768 * attention_factor * rank_factor
    assert groups["full_convolution"].page_size_bytes == 512 * rank_factor
    assert groups["sliding_convolution"].page_size_bytes == 1536 * rank_factor
    assert groups["full_convolution"].sliding_window == groups["sliding_convolution"].sliding_window == 4
    assert groups["full_attention"].num_layers == 2
    assert groups["sliding_attention"].num_layers == 6
    assert "rank_local_heads" in draft.sources["cache_groups"]
    assert '"dtype": "bfloat16"' in draft.sources["cache_groups"]


def test_inkling_convolution_caps_attention_tp_without_limiting_dep_width(tmp_path):
    config = _synthetic_inkling(tmp_path)
    with pytest.raises(ValueError, match="attention TP <= 2"):
        derive_profile(config, _request("moe", tensor_parallel=4), _runtime())
    dep = derive_profile(
        config, _request("moe", attention_data_parallel=8, moe_tensor_parallel=1, moe_expert_parallel=8), _runtime()
    )
    assert dep.resolved["cache_layout"] == "grouped"
    assert "cache_block_sizes" in dep.missing


@pytest.mark.parametrize("flag", [False, "absent"])
def test_inkling_cannot_silently_omit_runtime_convolution_state(tmp_path, flag):
    _synthetic_inkling(tmp_path)
    path = tmp_path / "config.json"
    document = json.loads(path.read_text())
    if flag == "absent":
        document["text_config"].pop("use_sconv")
    else:
        document["text_config"]["use_sconv"] = flag
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="allocates convolution state in every decoder layer"):
        load_model_config(path)


@pytest.mark.parametrize("local_ids", [None, "absent"])
def test_inkling_nominal_window_does_not_enable_absent_local_layers(tmp_path, local_ids):
    _synthetic_inkling(tmp_path)
    path = tmp_path / "config.json"
    document = json.loads(path.read_text())
    if local_ids == "absent":
        document["text_config"].pop("local_layer_ids")
    else:
        document["text_config"]["local_layer_ids"] = local_ids
    path.write_text(json.dumps(document))
    draft = derive_profile(load_model_config(path), _request("moe"), _runtime(cache_block_sizes={"full_attention": 16}))
    assert [group["name"] for group in draft.resolved["cache_groups"]] == ["full_attention", "full_convolution"]
    assert all(group["num_layers"] == 8 for group in draft.resolved["cache_groups"])


def test_group_override_preserves_runtime_padding_and_required_states(tmp_path):
    config = _synthetic_inkling(tmp_path)
    overrides = _runtime(
        weights_bytes=4096, activations_bytes=1024, cache_block_sizes={"full_attention": 16, "sliding_attention": 8}
    )
    initial = derive_profile(config, _request("moe", tensor_parallel=2), overrides)
    groups = initial.resolved["cache_groups"]
    groups[0]["page_size_bytes"] += 128
    overrides.pop("cache_block_sizes")
    overrides["cache_groups"] = groups
    edited = derive_profile(config, _request("moe", tensor_parallel=2), overrides)
    assert edited.profile is not None
    assert edited.resolved["cache_groups"][0]["page_size_bytes"] == 640
    assert "user override" in edited.sources["cache_groups"]
    assert "cache_block_sizes" not in edited.missing
    with pytest.raises(ValueError, match="layer counts and retention windows"):
        derive_profile(config, _request("moe", tensor_parallel=2), {**overrides, "cache_groups": groups[:2]})


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"sliding_window": 64, "sliding_window_size": 128}, "conflicting sliding-window"),
        ({"sliding_window": 64, "local_layer_ids": [0, 0]}, "unique indexes"),
        ({"sliding_window": 64, "local_layer_ids": [2]}, "below num_hidden_layers"),
        (
            {"sliding_window": 64, "local_layer_ids": [0], "layer_types": ["full_attention", "sliding_attention"]},
            "conflicting layer_types",
        ),
        ({"sliding_window": 64, "local_layer_ids": [0], "use_sliding_window": False}, "conflicts"),
        ({"layer_types": ["full_attention", "sliding_attention"]}, "positive sliding_window"),
        ({"use_sconv": True, "sconv_kernel_size": 4}, "unsupported use_sconv"),
        ({"use_sliding_window": True}, "requires a positive sliding_window"),
        ({"sliding_window": 128, "attn_type_list": [1, 1]}, "conflicts"),
    ],
)
def test_cache_alias_conflicts_and_unknown_states_fail_at_intake(tmp_path, updates, match):
    with pytest.raises(ValueError, match=match):
        _config(tmp_path, **updates)


def test_grouped_sizing_requires_declared_kv_dtype_and_dynamic_scale_bounds(tmp_path):
    config = _config(tmp_path, sliding_window=128)
    unknown_dtype = derive_profile(config, _request(), {"cache_block_sizes": {"sliding_attention": 16}})
    assert {"kv_cache_dtype", "cache_groups"} <= unknown_dtype.missing.keys()
    assert "cache_groups" not in unknown_dtype.resolved
    dynamic = _config(tmp_path, sliding_window=128, quantization_config={"kv_cache_scheme": {"dynamic": True}})
    unknown_scales = derive_profile(dynamic, _request(), _runtime(cache_block_sizes={"sliding_attention": 16}))
    assert "cache_groups" in unknown_scales.missing
    assert "nonstandard KV" in unknown_scales.missing["cache_groups"]
