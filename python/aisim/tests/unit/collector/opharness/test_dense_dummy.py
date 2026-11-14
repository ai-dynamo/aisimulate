# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The serving probes refill only low-bit weight matrices: scales, zero points and integer
index tensors keep the framework's values (an index tensor of random bits would fault)."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PATH = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components" / "probes" / "_dense_dummy.py"
_spec = importlib.util.spec_from_file_location("_dense_dummy", _PATH)
dense = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dense)


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.1.mlp.experts.w13_weight",
        "model.layers.1.mlp.experts.w2_weight_packed",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.down_proj.qweight",
    ],
)
def test_weight_matrices_are_refilled(name):
    assert dense.is_weight_matrix(name)


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.1.mlp.experts.w13_weight_scale",
        "model.layers.1.mlp.experts.w13_weight_scale_inv",
        "model.layers.1.mlp.experts.w2_weight_scale_2",
        "model.layers.0.mlp.down_proj.g_idx",
        "model.layers.0.mlp.down_proj.qzeros",
        "model.layers.1.mlp.experts.expert_map",
        "model.layers.1.mlp.gate.e_score_correction_bias",
        "model.layers.0.self_attn.attn.k_scale",
    ],
)
def test_scales_zero_points_and_index_tensors_are_kept(name):
    assert not dense.is_weight_matrix(name)


@pytest.mark.parametrize(
    "name, norm",
    [
        ("model.layers.0.input_layernorm.weight", True),
        ("model.layers.0.self_attn.kv_a_layernorm.weight", True),
        ("model.norm.weight", True),
        ("model.layers.0.self_attn.q_norm.weight", True),
        ("model.layers.1.mlp.gate.weight", False),
        ("model.embed_tokens.weight", False),
    ],
)
def test_norm_weights_are_told_apart_from_matrices(name, norm):
    """Norm weights go to 1 and float matrices to N(0, 0.02), so hidden states keep unit scale
    and a router's logits (about 1.7 at width 7168) outweigh the U(-1e-3, 1e-3) bias."""
    assert dense.is_weight_matrix(name)
    assert dense.is_norm_weight(name) is norm
