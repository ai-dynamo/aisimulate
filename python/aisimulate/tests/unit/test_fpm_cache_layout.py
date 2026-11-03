# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cache layout check that config intake and the FPM collector share."""

from __future__ import annotations

import pytest

from aisimulate.fpm_cache_layout import cache_layout_markers, recognized_layer_types

pytestmark = pytest.mark.unit

_STATE_FIELDS = (
    "ssm_cfg",
    "mamba_d_state",
    "linear_conv_kernel_dim",
    "linear_attn_config",
    "ssm_state_size",
    "mamba_num_heads",
)


@pytest.mark.parametrize("architecture", [None, "LlamaForCausalLM", "Glm5NextForConditionalGeneration"])
def test_sparse_attention_is_recognized_only_for_the_dsa_decoders(architecture):
    assert recognized_layer_types(architecture) == {"full_attention", "sliding_attention"}
    for decoder in ("GlmMoeDsaForCausalLM", "DeepseekV32ForCausalLM"):
        assert recognized_layer_types(decoder) == {"full_attention", "sliding_attention", "deepseek_sparse_attention"}


@pytest.mark.parametrize(
    "raw,architecture,expected",
    [
        ({}, None, ()),
        ({"layer_types": ["full_attention", "sliding_attention"]}, None, ()),
        (
            {"layer_types": ["linear_attention", "full_attention", "linear_attention"]},
            None,
            ("layer_types linear_attention x2",),
        ),
        ({"layer_types": ["deepseek_sparse_attention"]}, "GlmMoeDsaForCausalLM", ()),
        # A multimodal wrapper is not a DSA decoder.
        (
            {"layer_types": ["deepseek_sparse_attention"]},
            "Glm5NextForConditionalGeneration",
            ("layer_types deepseek_sparse_attention x1",),
        ),
        # Malformed lists are structural errors that each caller raises itself; they are not markers.
        ({"layer_types": [1, "", None]}, None, ()),
        ({"layer_types": "linear_attention"}, None, ()),
        # 1 is full attention; every other integer is one reason per distinct value.
        ({"attn_type_list": [1, 1]}, None, ()),
        ({"attn_type_list": [0, 1, 0, 2]}, None, ("attn_type_list 0 x2", "attn_type_list 2 x1")),
        ({"attn_type_list": [True, "0", 1.5]}, None, ()),
        # Null state fields are absent, not markers.
        (dict.fromkeys(_STATE_FIELDS), None, ()),
        *(({field: 0}, None, (f"recurrent-state field {field}",)) for field in _STATE_FIELDS),
        # Architecture names match in any letter case and are reported as declared.
        ({}, "FalconMambaForCausalLM", ("recurrent architecture FalconMambaForCausalLM",)),
        ({}, "ToyJAMBAForCausalLM", ("recurrent architecture ToyJAMBAForCausalLM",)),
        ({}, "Rwkv7ForCausalLM", ("recurrent architecture Rwkv7ForCausalLM",)),
        # Every marker counts, also next to recognized layer names, and the reasons come back sorted.
        (
            {"layer_types": ["linear_attention", "full_attention"], "attn_type_list": [0, 1], "ssm_state_size": 128},
            "ToyMambaForCausalLM",
            (
                "attn_type_list 0 x1",
                "layer_types linear_attention x1",
                "recurrent architecture ToyMambaForCausalLM",
                "recurrent-state field ssm_state_size",
            ),
        ),
        # The rule for architectures that intake does not recognize is not part of this check.
        ({}, "MiMoForCausalLM", ()),
    ],
)
def test_cache_layout_markers(raw, architecture, expected):
    assert cache_layout_markers(raw, architecture) == expected
