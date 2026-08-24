# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aisimulate.public_config import (
    PredictionConfig,
    RecommendationConfig,
    WorkerConfig,
    WorkersConfig,
    split_config_sections,
)
from aisimulate.recommend import recommendation_to_sweeper


def _engine() -> dict:
    return {
        "model": "example/model",
        "hardware": "h200_sxm",
        "workers": {"aggregated": {}},
    }


def test_prediction_uses_reviewed_default_traffic() -> None:
    config = PredictionConfig.model_validate({"engine": _engine()})

    assert config.traffic.source.type == "synthetic"
    assert config.traffic.load.type == "concurrency"
    assert config.traffic.load.concurrency == 10
    assert config.traffic.stop is not None
    assert config.traffic.stop.requests == 100


def test_prediction_scheduler_defaults_are_role_aware() -> None:
    aggregated = PredictionConfig.model_validate({"engine": _engine()})
    assert aggregated.engine.workers.aggregated is not None
    assert aggregated.engine.workers.aggregated.scheduler.max_batched_tokens == 8192
    assert aggregated.engine.workers.aggregated.scheduler.max_sequences == 256

    disaggregated = PredictionConfig.model_validate(
        {
            "engine": {
                **_engine(),
                "mode": "disaggregated",
                "workers": {"prefill": {}, "decode": {}},
            }
        }
    )
    assert disaggregated.engine.workers.prefill is not None
    assert disaggregated.engine.workers.decode is not None
    assert disaggregated.engine.workers.prefill.scheduler.max_batched_tokens == 8192
    assert disaggregated.engine.workers.prefill.scheduler.max_sequences == 1
    assert disaggregated.engine.workers.decode.scheduler.max_batched_tokens == 8192
    assert disaggregated.engine.workers.decode.scheduler.max_sequences == 256

    programmatic = WorkersConfig(
        prefill=WorkerConfig(), decode=WorkerConfig()
    )
    assert programmatic.prefill is not None
    assert programmatic.decode is not None
    assert programmatic.prefill.scheduler.max_sequences == 1
    assert programmatic.decode.scheduler.max_sequences == 256


def test_prediction_rejects_recommendation_domain() -> None:
    with pytest.raises(ValidationError):
        PredictionConfig.model_validate(
            {
                "engine": {
                    **_engine(),
                    "backend": {"choices": ["vllm", "sglang"]},
                }
            }
        )


def test_synthetic_session_rate_uses_session_units() -> None:
    config = PredictionConfig.model_validate(
        {
            "engine": _engine(),
            "traffic": {
                "source": {
                    "type": "synthetic-session",
                    "new_input_tokens_per_turn": 128,
                    "output_tokens_per_turn": 16,
                    "session": {"turns": 3},
                },
                "load": {
                    "type": "constant_rate",
                    "sessions_per_second": 4,
                },
                "stop": {"sessions": 20},
            },
        }
    )

    assert config.traffic.source.type == "synthetic-session"
    assert config.traffic.load.sessions_per_second == 4


def test_recommendation_rejects_unknown_nested_field() -> None:
    with pytest.raises(ValidationError, match="unknown fields"):
        RecommendationConfig.model_validate(
            {
                "engine": {**_engine(), "mystery": 1},
                "optimization": {},
            }
        )


def test_recommendation_accepts_domains_and_parallel_preset() -> None:
    config = RecommendationConfig.model_validate(
        {
            "engine": {
                "mode": {"choices": ["aggregated", "disaggregated"]},
                "model": "example/model",
                "hardware": "auto",
                "backend": {"choices": ["vllm", "sglang"]},
                "workers": {
                    role: {"parallelism": {"preset": "default"}}
                    for role in ("aggregated", "prefill", "decode")
                },
            },
            "optimization": {"hardware": "h200_sxm"},
        }
    )

    assert config.optimization.hardware == "h200_sxm"


def test_parallel_preset_modes_lower_without_conflating_semantics() -> None:
    independent = {
        "preset": False,
        "replicas": {"range": {"min": 1, "max": 2, "step": 1}},
    }
    config = RecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "aggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "context_length": 4096,
                "workers": {"aggregated": {"parallelism": independent}},
            },
            "optimization": {},
        }
    )

    lowered = recommendation_to_sweeper(config)

    assert lowered.search_space.parallel_configs_by_mode == {}
    assert lowered.search_space.parallel_independent_by_mode["agg"]["replicas"] == [
        1,
        2,
    ]


def test_custom_parallel_preset_lowers_as_flat_atomic_choices() -> None:
    mapping = {
        "replicas": 1,
        "tensor": 1,
        "pipeline": 1,
        "attention_data": 1,
        "moe_tensor": 1,
        "moe_expert": 1,
    }
    config = RecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "aggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "context_length": 4096,
                "workers": {
                    "aggregated": {"parallelism": {"preset": [mapping]}}
                },
            },
            "optimization": {},
        }
    )

    lowered = recommendation_to_sweeper(config)

    assert lowered.search_space.flat_parallel_modes == ["agg"]
    assert lowered.search_space.parallel_configs_by_mode["agg"] == [
        {
            "replicas": 1,
            "tp": 1,
            "pp": 1,
            "attention_dp": 1,
            "moe_tp": 1,
            "moe_ep": 1,
        }
    ]


def test_present_non_core_section_is_split_for_adapter_validation() -> None:
    base = {
        "engine": {
            **_engine(),
            "mode": "aggregated",
            "context_length": 4096,
        },
        "optimization": {},
        "placement": {"policy": {"choices": ["first", "least_loaded"]}},
    }
    core, adapters = split_config_sections(base, command="recommend")
    config = RecommendationConfig.model_validate(core)
    lowered = recommendation_to_sweeper(
        config, adapter_configs=adapters, stack="example"
    )

    assert "placement" not in core
    assert adapters == {
        "placement": {"policy": {"choices": ["first", "least_loaded"]}}
    }
    assert lowered.adapters["example.placement"].search_space == adapters["placement"]


def test_core_models_reject_stack_owned_sections() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RecommendationConfig.model_validate(
            {
                "engine": {**_engine(), "context_length": 4096},
                "placement": {"policy": "first"},
                "optimization": {},
            }
        )


@pytest.mark.parametrize("section", ["", "nested.section", 7])
def test_adapter_section_names_are_unambiguous(section) -> None:
    with pytest.raises(ValueError, match="section names without dots"):
        split_config_sections(
            {
                "engine": {**_engine(), "context_length": 4096},
                "optimization": {},
                section: {},
            },
            command="recommend",
        )
