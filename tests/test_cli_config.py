# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.config.common import split_config_sections
from aisimulate.config.engine import (
    EngineRecommendationConfig,
    WorkerPredictionConfig,
    WorkersPredictionConfig,
)
from aisimulate.recommend import recommendation_to_sweeper


def _engine() -> dict:
    return {
        "model": "example/model",
        "hardware": "h200_sxm",
        "workers": {"aggregated": {}},
    }


def test_prediction_uses_reviewed_default_traffic() -> None:
    config = CorePredictionConfig.model_validate({"engine": _engine()})

    assert config.traffic.source.type == "synthetic"
    assert config.traffic.load.type == "concurrency"
    assert config.traffic.load.concurrency == 10
    assert config.traffic.stop is not None
    assert config.traffic.stop.requests == 100


def test_prediction_scheduler_defaults_are_role_aware() -> None:
    aggregated = CorePredictionConfig.model_validate({"engine": _engine()})
    assert aggregated.engine.workers.aggregated is not None
    assert aggregated.engine.workers.aggregated.scheduler.max_batched_tokens == 8192
    assert aggregated.engine.workers.aggregated.scheduler.max_sequences == 256

    disaggregated = CorePredictionConfig.model_validate(
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

    programmatic = WorkersPredictionConfig(
        prefill=WorkerPredictionConfig(), decode=WorkerPredictionConfig()
    )
    assert programmatic.prefill is not None
    assert programmatic.decode is not None
    assert programmatic.prefill.scheduler.max_sequences == 1
    assert programmatic.decode.scheduler.max_sequences == 256


def test_prediction_accepts_trtllm_disaggregated_dp1() -> None:
    config = CorePredictionConfig.model_validate(
        {
            "engine": {
                **_engine(),
                "mode": "disaggregated",
                "backend": "trtllm",
                "workers": {
                    "prefill": {"parallelism": {"attention_data": 1}},
                    "decode": {"parallelism": {"attention_data": 1}},
                },
            }
        }
    )

    assert config.engine.backend == "trtllm"
    assert config.engine.mode == "disaggregated"
    assert config.engine.workers.prefill is not None
    assert config.engine.workers.decode is not None


def test_prediction_rejects_recommendation_domain() -> None:
    with pytest.raises(ValidationError):
        CorePredictionConfig.model_validate(
            {
                "engine": {
                    **_engine(),
                    "backend": {"choices": ["vllm", "sglang"]},
                }
            }
        )


def test_synthetic_session_rate_uses_session_units() -> None:
    config = CorePredictionConfig.model_validate(
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
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CoreRecommendationConfig.model_validate(
            {
                "engine": {**_engine(), "mystery": 1},
                "optimization": {},
            }
        )


def test_recommendation_accepts_domains_and_parallel_preset() -> None:
    config = CoreRecommendationConfig.model_validate(
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
    assert isinstance(config.engine, EngineRecommendationConfig)


def test_parallel_preset_modes_lower_without_conflating_semantics() -> None:
    independent = {
        "preset": False,
        "replicas": {"range": {"min": 1, "max": 2, "step": 1}},
    }
    config = CoreRecommendationConfig.model_validate(
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
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "aggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "context_length": 4096,
                "workers": {"aggregated": {"parallelism": {"preset": [mapping]}}},
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
    config = CoreRecommendationConfig.model_validate(core)
    lowered = recommendation_to_sweeper(
        config, adapter_configs=adapters, stack="example"
    )

    assert "placement" not in core
    assert adapters == {"placement": {"policy": {"choices": ["first", "least_loaded"]}}}
    assert lowered.adapters["example.placement"].search_space == adapters["placement"]


def test_core_models_reject_stack_owned_sections() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CoreRecommendationConfig.model_validate(
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


def test_parallel_custom_preset_requires_complete_strict_mapping() -> None:
    base = {
        "engine": {
            **_engine(),
            "mode": "aggregated",
            "context_length": 4096,
            "workers": {"aggregated": {"parallelism": {"preset": [{"replicas": 2}]}}},
        },
        "optimization": {},
    }
    with pytest.raises(ValidationError, match="cover exactly all knobs"):
        CoreRecommendationConfig.model_validate(base)


@pytest.mark.parametrize("invalid", [1.9, True, "2"])
def test_parallel_custom_preset_rejects_coercible_leaves(invalid) -> None:
    mapping = {
        "replicas": 1,
        "tensor": invalid,
        "pipeline": 1,
        "attention_data": 1,
        "moe_tensor": 1,
        "moe_expert": 1,
    }
    with pytest.raises(ValidationError):
        CoreRecommendationConfig.model_validate(
            {
                "engine": {
                    **_engine(),
                    "mode": "aggregated",
                    "context_length": 4096,
                    "workers": {"aggregated": {"parallelism": {"preset": [mapping]}}},
                },
                "optimization": {},
            }
        )


def test_engine_scheduler_domains_replace_defaults_and_preserve_log_scale() -> None:
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                **_engine(),
                "mode": "aggregated",
                "backend_version": "0.19.0",
                "context_length": 4096,
                "workers": {
                    "aggregated": {
                        "parallelism": {"preset": "default"},
                        "scheduler": {
                            "max_batched_tokens": {"choices": [4096]},
                            "max_sequences": {
                                "range": {"min": 1, "max": 8, "scale": "log"}
                            },
                        },
                    }
                },
            },
            "optimization": {},
        }
    )
    lowered = recommendation_to_sweeper(config)
    assert lowered.search_space.agg_max_num_batched_tokens == [4096]
    assert lowered.search_space.agg_max_num_seqs == [1]
    assert lowered.search_space.engine_integer_log_ranges["agg_max_num_seqs"] == [
        1,
        8,
    ]
    assert lowered.search_space.backend_version == "0.19.0"


def test_large_integer_log_ranges_lower_as_compact_bounds() -> None:
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                **_engine(),
                "mode": "aggregated",
                "context_length": 4096,
                "workers": {
                    "aggregated": {
                        "parallelism": {
                            "preset": False,
                            "replicas": {
                                "range": {"min": 1, "max": 1_000_000, "scale": "log"}
                            },
                        }
                    }
                },
            },
            "optimization": {},
        }
    )

    lowered = recommendation_to_sweeper(config)

    assert lowered.search_space.parallel_independent_by_mode["agg"]["replicas"] == [1]
    assert lowered.search_space.parallel_independent_log_ranges_by_mode["agg"][
        "replicas"
    ] == [
        1,
        1_000_000,
    ]


def test_disagg_mixed_parallel_presets_preserve_each_role_semantics() -> None:
    mapping = {
        "replicas": 1,
        "tensor": 1,
        "pipeline": 1,
        "attention_data": 1,
        "moe_tensor": 1,
        "moe_expert": 1,
    }
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "disaggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "context_length": 4096,
                "workers": {
                    "prefill": {"parallelism": {"preset": [mapping]}},
                    "decode": {
                        "parallelism": {
                            "preset": False,
                            "replicas": {"choices": [1, 2]},
                        }
                    },
                },
            },
            "optimization": {},
        }
    )

    lowered = recommendation_to_sweeper(config)

    assert lowered.search_space.parallel_custom_configs_by_mode["disagg"][
        "prefill"
    ] == [
        {
            "replicas": 1,
            "tp": 1,
            "pp": 1,
            "attention_dp": 1,
            "moe_tp": 1,
            "moe_ep": 1,
        }
    ]
    assert lowered.search_space.parallel_independent_by_mode["disagg"][
        "decode_replicas"
    ] == [
        1,
        2,
    ]
    assert not any(
        name.startswith("prefill_")
        for name in lowered.search_space.parallel_independent_by_mode["disagg"]
    )


def test_backend_specific_block_size_validation() -> None:
    with pytest.raises(ValidationError, match="require block_size >= 2"):
        CorePredictionConfig.model_validate(
            {
                "engine": {
                    **_engine(),
                    "backend": "vllm",
                    "workers": {"aggregated": {"kv_cache": {"block_size": 1}}},
                }
            }
        )

    config = CorePredictionConfig.model_validate(
        {
            "engine": {
                **_engine(),
                "backend": "sglang",
                "workers": {"aggregated": {"kv_cache": {"block_size": 1}}},
            }
        }
    )
    assert config.engine.workers.aggregated.kv_cache.block_size == 1


def test_prediction_rejects_auto_hardware_and_kv_relative_load() -> None:
    with pytest.raises(ValidationError, match="recommendation-only"):
        CorePredictionConfig.model_validate(
            {"engine": {**_engine(), "hardware": "auto"}}
        )
    with pytest.raises(ValidationError):
        CorePredictionConfig.model_validate(
            {
                "engine": _engine(),
                "traffic": {
                    "source": {"type": "synthetic"},
                    "load": {"type": "kv_capacity_fraction", "fraction": 1.2},
                    "stop": {"requests": 10},
                },
            }
        )


def test_traffic_and_optimizer_strict_defaults_and_finite_values() -> None:
    prediction = CorePredictionConfig.model_validate(
        {
            "engine": _engine(),
            "traffic": {
                "source": {"type": "synthetic"},
                "load": {},
                "stop": {"requests": 10},
            },
        }
    )
    assert prediction.traffic.load.concurrency == 10
    with pytest.raises(ValidationError):
        CoreRecommendationConfig.model_validate(
            {
                "engine": {
                    **_engine(),
                    "mode": "aggregated",
                    "context_length": 4096,
                },
                "optimization": {"hardware": "auto"},
            }
        )


@pytest.mark.parametrize(
    ("field", "bound"), [("ttft_ms", 800.0), ("itl_ms", 30.0), ("e2e_ms", 1000.0)]
)
@pytest.mark.parametrize("target", ["goodput", "goodput_per_gpu"])
def test_goodput_requires_at_least_one_sla_bound_in_typed_cli_config(
    target: str, field: str, bound: float
) -> None:
    base = {
        "engine": {
            **_engine(),
            "mode": "aggregated",
            "context_length": 4096,
        },
        "optimization": {"target": target},
    }
    with pytest.raises(ValidationError, match="requires an evaluation.sla bound"):
        CoreRecommendationConfig.model_validate(base)
    accepted = CoreRecommendationConfig.model_validate(
        {**base, "evaluation": {"sla": {field: bound}}}
    )
    assert getattr(accepted.evaluation.sla, field) == bound


def test_strict_sla_is_public_and_lowers_to_sweeper_goal() -> None:
    base = {
        "engine": {
            **_engine(),
            "mode": "aggregated",
            "context_length": 4096,
        },
        "optimization": {"target": "throughput", "strict_sla": True},
    }
    with pytest.raises(ValidationError, match="requires at least one"):
        CoreRecommendationConfig.model_validate(base)

    public = CoreRecommendationConfig.model_validate(
        {**base, "evaluation": {"sla": {"itl_ms": 30}}}
    )
    lowered = recommendation_to_sweeper(public)

    assert public.optimization.strict_sla is True
    assert lowered.goal.strict_sla is True
    assert lowered.goal.sla is not None
    assert lowered.goal.sla.itl_ms == 30

    with pytest.raises(ValidationError):
        CoreRecommendationConfig.model_validate(
            {**base, "optimization": {"strict_sla": "true"}}
        )


@pytest.mark.parametrize(("field", "bound"), [("ttft_ms", 800.0), ("itl_ms", 30.0)])
def test_prediction_accepts_independent_token_sla_bounds(
    field: str, bound: float
) -> None:
    config = CorePredictionConfig.model_validate(
        {"engine": _engine(), "evaluation": {"sla": {field: bound}}}
    )

    assert getattr(config.evaluation.sla, field) == bound


@pytest.mark.parametrize(("field", "bound"), [("ttft_ms", 800.0), ("itl_ms", 30.0)])
def test_recommendation_accepts_independent_token_sla_bounds(
    field: str, bound: float
) -> None:
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {**_engine(), "mode": "aggregated"},
            "evaluation": {"sla": {field: bound}},
            "optimization": {},
        }
    )

    assert getattr(config.evaluation.sla, field) == bound
    assert config.optimization.strict_sla is False


def test_kv_relative_load_choices_lower_as_generic_search_dimension() -> None:
    config = CoreRecommendationConfig.model_validate(
        {
            "traffic": {
                "source": {"type": "synthetic"},
                "load": {
                    "type": "kv_capacity_fraction",
                    "fraction": {"choices": [0.5, 1.5]},
                },
                "stop": {"requests": 10},
            },
            "engine": {
                **_engine(),
                "mode": "aggregated",
                "context_length": 4096,
            },
            "optimization": {},
        }
    )
    lowered = recommendation_to_sweeper(config)
    assert lowered.workload.load_search_field == "kv_load_ratio"
    assert lowered.workload.load_choices == [0.5, 1.5]


def test_trace_block_default_and_finite_rate_contract() -> None:
    config = CorePredictionConfig.model_validate(
        {
            "traffic": {
                "source": {
                    "type": "trace",
                    "paths": ["trace.jsonl"],
                    "format": "mooncake",
                },
                "load": {"type": "trace_timestamps"},
            },
            "engine": _engine(),
        }
    )
    assert config.traffic.source.block_size == 512

    weka = CorePredictionConfig.model_validate(
        {
            "traffic": {
                "source": {"type": "trace", "paths": ["weka-corpus"], "format": "weka"},
                "load": {"type": "trace_timestamps", "agentic_lanes": 2},
            },
            "engine": _engine(),
        }
    )
    assert weka.traffic.source.block_size is None
    assert weka.traffic.load.agentic_lanes == 2


@pytest.mark.parametrize(
    "traffic",
    [
        {
            "source": {"type": "trace", "paths": ["trace.jsonl"], "format": "mooncake"},
            "load": {"type": "trace_timestamps", "agentic_lanes": 1},
        },
        {
            "source": {"type": "trace", "paths": ["weka-corpus"], "format": "weka"},
            "load": {"type": "concurrency", "concurrency": 1},
        },
        {
            "source": {"type": "trace", "paths": ["weka-corpus"], "format": "weka"},
            "load": {"type": "trace_timestamps", "agentic_lanes": 0},
        },
    ],
)
def test_agentic_lane_contract_rejects_unsupported_inputs(traffic: dict) -> None:
    with pytest.raises(ValidationError):
        CorePredictionConfig.model_validate({"traffic": traffic, "engine": _engine()})


def test_weka_requires_aggregated_engine() -> None:
    with pytest.raises(ValidationError, match="weka requires aggregated"):
        CorePredictionConfig.model_validate(
            {
                "traffic": {
                    "source": {
                        "type": "trace",
                        "paths": ["weka-corpus"],
                        "format": "weka",
                    },
                    "load": {"type": "trace_timestamps"},
                },
                "engine": {
                    **_engine(),
                    "mode": "disaggregated",
                    "workers": {"prefill": {}, "decode": {}},
                },
            }
        )


def test_finite_rate_and_timeout_contract() -> None:
    with pytest.raises(ValidationError):
        CorePredictionConfig.model_validate(
            {
                "traffic": {
                    "source": {"type": "synthetic"},
                    "load": {
                        "type": "constant_rate",
                        "requests_per_second": float("inf"),
                    },
                    "stop": {"requests": 10},
                },
                "engine": _engine(),
            }
        )
    with pytest.raises(ValidationError):
        CoreRecommendationConfig.model_validate(
            {
                "engine": {
                    **_engine(),
                    "mode": "aggregated",
                    "context_length": 4096,
                },
                "optimization": {},
                "optimizer": {"candidate_timeout_seconds": float("inf")},
            }
        )
