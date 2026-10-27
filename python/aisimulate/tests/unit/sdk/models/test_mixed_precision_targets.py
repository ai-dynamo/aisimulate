# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A MIXED_PRECISION export names what it quantizes; everything else is unquantized."""

import pytest

from aisimulate_core.sdk.models import helpers

pytestmark = pytest.mark.unit

GLM53_LIKE = {
    "quant_algo": "mixed_precision",
    "hf_quant_config": {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                **{
                    f"model.layers.{i}.mlp.{p}": {"quant_algo": "FP8"}
                    for i in range(3)
                    for p in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
                },
                **{f"model.layers.{i}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16} for i in range(3, 78)},
            },
            "exclude_modules": ["model.layers.78*"],
        }
    },
    "quantization_config": {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "ignore": ["model.layers.78*"],
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 8, "type": "float", "dynamic": False},
                "targets": ["model.layers.0.mlp.gate_proj"],
            }
        },
    },
}


def test_quantized_paths_collects_both_sources_lower_cased():
    paths = helpers.mixed_precision_quantized_paths(GLM53_LIKE)
    assert "model.layers.0.mlp.gate_proj" in paths and "model.layers.77.mlp.experts" in paths
    assert len(paths) == 87


def test_quantized_paths_is_none_outside_mixed_precision():
    assert (
        helpers.mixed_precision_quantized_paths(
            {"quant_algo": "nvfp4", "hf_quant_config": GLM53_LIKE["hf_quant_config"]}
        )
        is None
    )


def test_quantized_paths_is_none_for_class_name_targets():
    raw = {
        "quant_algo": "mixed_precision",
        "quantization_config": {
            "config_groups": {"g": {"weights": {"num_bits": 4, "type": "float"}, "targets": ["Linear"]}}
        },
    }
    assert helpers.mixed_precision_quantized_paths(raw) is None


def test_covers_matches_exact_and_prefix_paths():
    paths = frozenset({"model.layers.3.mlp.experts", "model.layers.0.mlp.gate_up_proj"})
    assert helpers.mixed_precision_covers(paths, "model.layers.3.mlp.experts.17.down_proj")
    assert helpers.mixed_precision_covers(paths, "model.layers.0.mlp.gate_up_proj")
    assert not helpers.mixed_precision_covers(paths, "model.layers.0.mlp.gate_proj")
    assert not helpers.mixed_precision_covers(paths, "model.layers.30.mlp.shared_experts.gate_proj")


def test_attention_exclusions_are_the_unlisted_groups():
    paths = helpers.mixed_precision_quantized_paths(GLM53_LIKE)
    assert helpers.mixed_precision_attention_exclusions(paths) == frozenset({"q", "kv", "o", "indexer"})
    listed = frozenset({"model.layers.0.self_attn.q_b_proj", "model.layers.0.self_attn.o_proj"})
    assert helpers.mixed_precision_attention_exclusions(listed) == frozenset({"kv", "indexer"})
    assert helpers.mixed_precision_attention_exclusions(frozenset({"model.layers.0.self_attn"})) == frozenset()


def test_shared_experts_quantized_only_when_listed():
    assert not helpers.mixed_precision_shared_experts_quantized(helpers.mixed_precision_quantized_paths(GLM53_LIKE))
    assert helpers.mixed_precision_shared_experts_quantized(frozenset({"model.layers.4.mlp.shared_experts.gate_proj"}))
