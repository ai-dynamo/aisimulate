# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiconfigurator_core.sdk._cuda_graph_component_model import (
    component_observation,
    interpolate_component,
    reconstruct_reservation_bytes,
)
from aiconfigurator_core.sdk._cuda_graph_features import (
    derived_feature_row,
    graph_shape_features,
    model_architecture_features,
)

pytestmark = pytest.mark.unit


def _component_row(max_num_seqs: int = 8) -> dict[str, object]:
    return {
        "attention_backend": "FLASH_ATTN",
        "attention_dp_size": 1,
        "compilation_backend": "inductor",
        "compilation_mode": "NONE",
        "compute_dtype": "bfloat16",
        "cuda_graph_capture_sizes": [1, 2, 4],
        "cuda_graph_mode": "FULL_AND_PIECEWISE",
        "dcp_size": 1,
        "flashinfer_autotune": False,
        "kv_cache_dtype": "fp8",
        "linear_backend": "auto",
        "max_model_len": 4096,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": max_num_seqs,
        "model_id": "MiniMaxAI/MiniMax-M2.7",
        "moe_backend": "auto",
        "moe_ep_size": 1,
        "moe_tp_size": 4,
        "pcp_size": 1,
        "pp_size": 1,
        "profile_id": f"profile-{max_num_seqs}",
        "quantization": "fp8",
        "speculative_method": "none",
        "speculative_tokens": 0,
        "system": "h200_sxm",
        "tp_size": 4,
        "backend_version": "0.25.1",
    }


def test_minimax_m27_architecture_is_resolved_from_packaged_config() -> None:
    features = model_architecture_features("MiniMaxAI/MiniMax-M2.7")
    assert features["model_architecture"] == "MiniMaxM2ForCausalLM"
    assert features["model_hidden_size"] == 3072
    assert features["model_num_hidden_layers"] == 62
    assert features["model_num_experts"] == 256
    assert features["model_experts_per_token"] == 8
    assert len(features["model_architecture_config_sha256"]) == 64


def test_minimax_m27_graph_distribution_uses_all_capture_sizes() -> None:
    sizes = [
        1,
        2,
        4,
        *range(8, 257, 8),
        *range(272, 513, 16),
        *range(544, 2049, 32),
    ]
    features = graph_shape_features(
        {
            "cuda_graph_capture_sizes": sizes,
            "cuda_graph_mode": "FULL_AND_PIECEWISE",
            "max_num_seqs": 1024,
            "speculative_tokens": 0,
        }
    )
    assert features["cuda_graph_capture_count"] == 99
    assert features["cuda_graph_full_count"] == 67
    assert features["cuda_graph_full_largest_capture_size"] == 1024
    assert features["cuda_graph_piecewise_count"] == 99
    assert features["cuda_graph_piecewise_largest_capture_size"] == 2048


@pytest.mark.parametrize(
    ("topology", "mode"),
    [
        ({"tp_size": 4, "attention_dp_size": 1, "moe_tp_size": 4, "moe_ep_size": 1}, "tp"),
        ({"tp_size": 4, "attention_dp_size": 1, "moe_tp_size": 1, "moe_ep_size": 4}, "tep"),
        ({"tp_size": 1, "attention_dp_size": 4, "moe_tp_size": 1, "moe_ep_size": 4}, "dep"),
    ],
)
def test_parallel_modes_have_distinct_interaction_features(topology: dict[str, int], mode: str) -> None:
    features = derived_feature_row(
        {
            "model_id": "MiniMaxAI/MiniMax-M2.7",
            "cuda_graph_capture_sizes": [1, 2, 4, 8],
            "cuda_graph_mode": "FULL_AND_PIECEWISE",
            "max_num_seqs": 8,
            "pp_size": 1,
            "dcp_size": 1,
            "pcp_size": 1,
            **topology,
        }
    )
    assert features["parallel_mode"] == mode
    assert features["capture_attention_elements"] > 0
    assert features["capture_moe_active_elements"] > 0


def test_component_interpolation_stays_inside_exact_categorical_cell_and_numeric_range() -> None:
    observations = [
        component_observation(_component_row(8), "full_first_capture", target_bytes=800),
        component_observation(_component_row(16), "full_first_capture", target_bytes=1600),
    ]
    prediction = interpolate_component(observations, _component_row(12), "full_first_capture")
    assert prediction is not None
    assert 800 < prediction < 1600
    assert interpolate_component(observations, _component_row(32), "full_first_capture") is None
    assert (
        interpolate_component(
            observations,
            {**_component_row(12), "system": "b200_sxm"},
            "full_first_capture",
        )
        is None
    )


def test_component_reconstruction_uses_shared_first_capture_pool() -> None:
    reservation = reconstruct_reservation_bytes(
        _component_row(),
        {
            "full_first_capture": 224,
            "full_per_graph": 16,
            "piecewise_first_capture": 128,
            "piecewise_per_graph": 8,
        },
    )
    assert reservation == 272
