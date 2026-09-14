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
from aisimulate.recommend import _candidate_prediction, recommendation_to_sweeper
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import ReplaySpec
from aisimulate.sweeper.sample import unroll_sample


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
    assert aggregated.engine.workers.aggregated.scheduler.prefill_schedule_interval == 1

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
    assert disaggregated.engine.workers.prefill.scheduler.prefill_schedule_interval == 1
    assert disaggregated.engine.workers.decode.scheduler.max_batched_tokens == 8192
    assert disaggregated.engine.workers.decode.scheduler.max_sequences == 256
    assert disaggregated.engine.workers.decode.scheduler.prefill_schedule_interval == 1

    programmatic = WorkersPredictionConfig(prefill=WorkerPredictionConfig(), decode=WorkerPredictionConfig())
    assert programmatic.prefill is not None
    assert programmatic.decode is not None
    assert programmatic.prefill.scheduler.max_sequences == 1
    assert programmatic.decode.scheduler.max_sequences == 256


def test_prediction_rejects_nonpositive_prefill_schedule_interval() -> None:
    engine = _engine()
    engine["workers"]["aggregated"] = {"scheduler": {"prefill_schedule_interval": 0}}

    with pytest.raises(ValidationError, match="prefill_schedule_interval"):
        CorePredictionConfig.model_validate({"engine": engine})


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


@pytest.mark.parametrize("value", [-1, (1 << 53) + 1])
def test_prediction_rejects_invalid_cuda_graph_reservation(value: int) -> None:
    engine = _engine()
    engine["workers"]["aggregated"] = {
        "kv_cache": {
            "capacity": {
                "type": "default",
                "cuda_graph_reserved_bytes": value,
            }
        }
    }

    with pytest.raises(ValidationError, match="cuda_graph_reserved_bytes"):
        CorePredictionConfig.model_validate({"engine": engine})


def test_prediction_rejects_cuda_graph_reservation_with_fixed_capacity() -> None:
    engine = _engine()
    engine["workers"]["aggregated"] = {
        "kv_cache": {
            "capacity": {
                "type": "fixed",
                "blocks": 128,
                "cuda_graph_reserved_bytes": 1 << 30,
            }
        }
    }

    with pytest.raises(ValidationError, match="fixed KV capacity rejects cuda_graph_reserved_bytes"):
        CorePredictionConfig.model_validate({"engine": engine})


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
                    role: {"parallelism": {"preset": "default"}} for role in ("aggregated", "prefill", "decode")
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
    lowered = recommendation_to_sweeper(config, adapter_configs=adapters, stack="example")

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
                            "max_sequences": {"range": {"min": 1, "max": 8, "scale": "log"}},
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
                            "replicas": {"range": {"min": 1, "max": 1_000_000, "scale": "log"}},
                        }
                    }
                },
            },
            "optimization": {},
        }
    )

    lowered = recommendation_to_sweeper(config)

    assert lowered.search_space.parallel_independent_by_mode["agg"]["replicas"] == [1]
    assert lowered.search_space.parallel_independent_log_ranges_by_mode["agg"]["replicas"] == [
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

    assert lowered.search_space.parallel_custom_configs_by_mode["disagg"]["prefill"] == [
        {
            "replicas": 1,
            "tp": 1,
            "pp": 1,
            "attention_dp": 1,
            "moe_tp": 1,
            "moe_ep": 1,
        }
    ]
    assert lowered.search_space.parallel_independent_by_mode["disagg"]["decode_replicas"] == [
        1,
        2,
    ]
    assert not any(name.startswith("prefill_") for name in lowered.search_space.parallel_independent_by_mode["disagg"])


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
        CorePredictionConfig.model_validate({"engine": {**_engine(), "hardware": "auto"}})
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


@pytest.mark.parametrize(("field", "bound"), [("ttft_ms", 800.0), ("itl_ms", 30.0), ("e2e_ms", 1000.0)])
@pytest.mark.parametrize("target", ["goodput", "goodput_per_gpu"])
def test_goodput_requires_at_least_one_sla_bound_in_typed_cli_config(target: str, field: str, bound: float) -> None:
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
    accepted = CoreRecommendationConfig.model_validate({**base, "evaluation": {"sla": {field: bound}}})
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

    public = CoreRecommendationConfig.model_validate({**base, "evaluation": {"sla": {"itl_ms": 30}}})
    lowered = recommendation_to_sweeper(public)

    assert public.optimization.strict_sla is True
    assert lowered.goal.strict_sla is True
    assert lowered.goal.sla is not None
    assert lowered.goal.sla.itl_ms == 30

    with pytest.raises(ValidationError):
        CoreRecommendationConfig.model_validate({**base, "optimization": {"strict_sla": "true"}})


@pytest.mark.parametrize(("field", "bound"), [("ttft_ms", 800.0), ("itl_ms", 30.0)])
def test_prediction_accepts_independent_token_sla_bounds(field: str, bound: float) -> None:
    config = CorePredictionConfig.model_validate({"engine": _engine(), "evaluation": {"sla": {field: bound}}})

    assert getattr(config.evaluation.sla, field) == bound


@pytest.mark.parametrize(("field", "bound"), [("ttft_ms", 800.0), ("itl_ms", 30.0)])
def test_recommendation_accepts_independent_token_sla_bounds(field: str, bound: float) -> None:
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
    assert weka.traffic.source.nested_timestamp_basis is None
    assert weka.traffic.load.agentic_lanes == 2

    configured_weka = CorePredictionConfig.model_validate(
        {
            "traffic": {
                "source": {
                    "type": "trace",
                    "paths": ["weka-corpus"],
                    "format": "weka",
                    "nested_timestamp_basis": "relative",
                },
                "load": {"type": "trace_timestamps"},
            },
            "engine": _engine(),
        }
    )
    assert configured_weka.traffic.source.nested_timestamp_basis == "relative"

    with pytest.raises(ValidationError, match="only valid for trace format 'weka'"):
        CorePredictionConfig.model_validate(
            {
                "traffic": {
                    "source": {
                        "type": "trace",
                        "paths": ["trace.jsonl"],
                        "format": "mooncake",
                        "nested_timestamp_basis": "absolute",
                    },
                    "load": {"type": "trace_timestamps"},
                },
                "engine": _engine(),
            }
        )


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


def test_prediction_timing_forward_model_defaults_to_op_level() -> None:
    config = CorePredictionConfig.model_validate({"engine": _engine()})

    assert config.engine.workers.aggregated is not None
    assert config.engine.workers.aggregated.timing.type == "default"
    assert config.engine.workers.aggregated.timing.forward_model == "op_level"


def test_prediction_timing_accepts_fpm_forward_model_with_default_timing() -> None:
    engine = _engine()
    engine["workers"]["aggregated"] = {"timing": {"type": "default", "forward_model": "fpm"}}

    config = CorePredictionConfig.model_validate({"engine": engine})

    assert config.engine.workers.aggregated is not None
    assert config.engine.workers.aggregated.timing.forward_model == "fpm"


@pytest.mark.parametrize(
    "timing",
    [
        {"type": "fixed", "prefill_ms": 1, "decode_ms": 1, "forward_model": "fpm"},
        {"type": "polynomial", "forward_model": "fpm"},
    ],
)
def test_prediction_timing_rejects_fpm_forward_model_without_default_timing(timing: dict) -> None:
    engine = _engine()
    engine["workers"]["aggregated"] = {"timing": timing}

    with pytest.raises(ValidationError, match="forward_model applies to default timing only"):
        CorePredictionConfig.model_validate({"engine": engine})


def test_prediction_timing_rejects_unknown_forward_model() -> None:
    engine = _engine()
    engine["workers"]["aggregated"] = {"timing": {"forward_model": "layerwise"}}

    with pytest.raises(ValidationError, match="forward_model"):
        CorePredictionConfig.model_validate({"engine": engine})


def test_recommendation_timing_accepts_forward_model_per_role() -> None:
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "disaggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "workers": {
                    "prefill": {"timing": {"type": "default"}},
                    "decode": {"timing": {"type": "default", "forward_model": "fpm"}},
                },
            },
            "optimization": {"constraints": {"max_candidate_gpus": 8}},
        }
    )

    assert config.engine.workers.prefill is not None
    assert config.engine.workers.decode is not None
    assert config.engine.workers.prefill.timing.forward_model == "op_level"
    assert config.engine.workers.decode.timing.forward_model == "fpm"


def _fpm_recommendation() -> CoreRecommendationConfig:
    return CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "aggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "context_length": 2048,
                "workers": {
                    "aggregated": {
                        "parallelism": {"preset": False, "tensor": 1, "moe_tensor": 1, "moe_expert": 1},
                        "kv_cache": {"capacity": {"type": "fixed", "blocks": 256}},
                        "timing": {"type": "default", "forward_model": "fpm"},
                    }
                },
            },
            "optimization": {"constraints": {"max_candidate_gpus": 8}},
        }
    )


def test_recommendation_lowers_forward_model_per_role() -> None:
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "disaggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "context_length": 4096,
                "workers": {
                    "prefill": {"timing": {"type": "default"}},
                    "decode": {"timing": {"type": "default", "forward_model": "fpm"}},
                },
            },
            "optimization": {"constraints": {"max_candidate_gpus": 8}},
        }
    )

    space = recommendation_to_sweeper(config).search_space

    assert space.prefill_forward_model == "op_level"
    assert space.decode_forward_model == "fpm"
    assert space.agg_forward_model == "op_level"


def test_recommendation_candidate_yaml_round_trips_forward_model() -> None:
    config = _fpm_recommendation()
    smart = recommendation_to_sweeper(config)
    assert smart.search_space.agg_forward_model == "fpm"

    sample = unroll_sample(
        search_space=smart.search_space,
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 256,
        },
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1),
    )
    assert sample["agg_forward_model"] == "fpm"
    deployment = build_backend_deployment(sample, backend_version="test")
    assert deployment.agg_engine_args["aic_forward_model"] == "fpm"

    prediction = _candidate_prediction(
        config,
        sample,
        ReplaySpec(backend_deployment=deployment, workload={}, goal={}),
        adapter_sections={},
    )

    assert prediction["engine"]["workers"]["aggregated"]["timing"] == {"type": "default", "forward_model": "fpm"}
    CorePredictionConfig.model_validate(prediction)


def test_recommendation_candidate_yaml_spells_out_op_level_like_other_defaults() -> None:
    config = _fpm_recommendation()
    raw = config.model_dump(mode="python", exclude_none=True)
    raw["engine"]["workers"]["aggregated"]["timing"] = {"type": "default"}
    config = CoreRecommendationConfig.model_validate(raw)
    smart = recommendation_to_sweeper(config)
    sample = unroll_sample(
        search_space=smart.search_space,
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 256,
        },
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1),
    )
    deployment = build_backend_deployment(sample, backend_version="test")

    prediction = _candidate_prediction(
        config,
        sample,
        ReplaySpec(backend_deployment=deployment, workload={}, goal={}),
        adapter_sections={},
    )

    # Normalization materializes every schema default into the candidate; forward_model is no exception.
    assert prediction["engine"]["workers"]["aggregated"]["timing"] == {"type": "default", "forward_model": "op_level"}


def _pd_hardware_config(*, recommend=False, **overrides):
    engine = {
        **_engine(),
        "mode": "disaggregated",
        "backend": "vllm",
        "backend_version": "0.24.0",
        "context_length": 4096,
        "workers": {role: {"hardware": value} for role, value in overrides.items()},
    }
    for role in ("prefill", "decode"):
        engine["workers"].setdefault(role, {})
    raw = {"engine": engine}
    if recommend:
        raw["optimization"] = {"constraints": {"max_candidate_gpus": 4}}
    return raw


@pytest.mark.parametrize("recommend", [False, True])
@pytest.mark.parametrize("value", ["", " ", "auto", {"choices": ["h200_sxm", "gb200"]}])
def test_worker_hardware_requires_concrete_sku(recommend, value):
    schema = CoreRecommendationConfig if recommend else CorePredictionConfig
    with pytest.raises(ValidationError, match="hardware"):
        schema.model_validate(_pd_hardware_config(recommend=recommend, decode=value))


@pytest.mark.parametrize("recommend", [False, True])
@pytest.mark.parametrize("mode,role", [("aggregated", "aggregated"), ("afd", "decode")])
def test_worker_hardware_rejects_non_pd_modes(recommend, mode, role):
    schema = CoreRecommendationConfig if recommend else CorePredictionConfig
    raw = _pd_hardware_config(recommend=recommend)
    raw["engine"].update(mode=mode, workers={role: {"hardware": "gb200"}})
    with pytest.raises(ValidationError, match="hardware overrides require prefill/decode"):
        schema.model_validate(raw)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, ("h200_sxm", "h200_sxm")),
        ({"decode": "gb200"}, ("h200_sxm", "gb200")),
        ({"prefill": "gb200"}, ("gb200", "h200_sxm")),
        ({"prefill": "gb200", "decode": "gb200"}, ("gb200", "gb200")),
    ],
)
def test_pd_hardware_survives_search_candidate_yaml_and_predict(overrides, expected):
    import yaml

    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.sweeper.parallel_enum import DisaggParallelConfig

    source = CoreRecommendationConfig.model_validate(_pd_hardware_config(recommend=True, **overrides))
    smart = recommendation_to_sweeper(source)
    assert tuple(smart.search_space.hardware_sku_for(role) for role in ("prefill", "decode")) == expected
    parallel = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    sample = unroll_sample(
        search_space=smart.search_space,
        selection={
            "deployment_mode": "disagg",
            "backend": "vllm",
            "prefill_max_num_batched_tokens": 8192,
            "prefill_max_num_seqs": 1,
            "decode_max_num_batched_tokens": 8192,
            "decode_max_num_seqs": 256,
        },
        parallel_config=DisaggParallelConfig(prefill=parallel, decode=parallel),
    )
    sample["backend_version"] = "0.24.0"
    deployment = build_backend_deployment(sample, backend_version="0.24.0")
    mapping = _candidate_prediction(
        source, sample, ReplaySpec(backend_deployment=deployment, workload={}, goal={}), adapter_sections={}
    )
    concrete = CorePredictionConfig.model_validate(yaml.safe_load(yaml.safe_dump(mapping)))
    compiled = prediction_to_replay_spec(concrete).backend_deployment
    assert concrete.engine.hardware == "h200_sxm"
    for role, hardware in zip(("prefill", "decode"), expected, strict=True):
        assert ("hardware" in mapping["engine"]["workers"][role]) == (role in overrides)
        assert getattr(compiled, f"{role}_engine_args")["aic_system"] == hardware
        assert compiled.performance_model_metadata[role]["config"]["system"] == hardware
        assert getattr(deployment, f"{role}_engine_args")["aic_system"] == hardware


def test_pd_hardware_mixed_modes_and_auto_fallback():
    raw = _pd_hardware_config(recommend=True, decode="gb200")
    raw["engine"]["mode"] = {"choices": ["aggregated", "disaggregated"]}
    raw["engine"]["workers"]["aggregated"] = {}
    raw["engine"]["hardware"] = "auto"
    raw["optimization"]["hardware"] = "h200_sxm"
    space = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw)).search_space
    assert space.hardware_sku_for("agg") == "h200_sxm"
    assert space.hardware_sku_for("prefill") == "h200_sxm"
    assert space.hardware_sku_for("decode") == "gb200"


@pytest.mark.parametrize("same_version", [False, True])
def test_pd_predict_requires_shared_implicit_backend_version(monkeypatch, same_version):
    from aisimulate.compiler import prediction_to_replay_spec

    calls = []

    def resolve(hardware, backend):
        calls.append((hardware, backend))
        return "0.24.0" if same_version or hardware == "h200_sxm" else "0.23.0"

    monkeypatch.setattr("aisimulate.compiler.resolve_backend_version", resolve)
    raw = _pd_hardware_config(decode="gb200")
    del raw["engine"]["backend_version"]
    config = CorePredictionConfig.model_validate(raw)
    if same_version:
        deployment = prediction_to_replay_spec(config).backend_deployment
        assert deployment.backend_version == "0.24.0"
        for role in ("prefill", "decode"):
            assert getattr(deployment, f"{role}_engine_args")["aic_backend_version"] == "0.24.0"
    else:
        with pytest.raises(ValueError, match="Set engine.backend_version"):
            prediction_to_replay_spec(config)
    assert set(calls) == {("h200_sxm", "vllm"), ("gb200", "vllm")}


@pytest.mark.parametrize("override", [False, True])
def test_pd_predict_keeps_legacy_version_defaults_without_hardware_override(monkeypatch, override):
    from aisimulate.compiler import prediction_to_replay_spec

    def resolve(hardware, backend):
        assert hardware == "gb200"
        return "0.24.0"

    monkeypatch.setattr("aisimulate.compiler.resolve_backend_version", resolve)
    raw = _pd_hardware_config(**({"prefill": "gb200", "decode": "gb200"} if override else {}))
    del raw["engine"]["backend_version"]
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment
    assert deployment.backend_version == ("0.24.0" if override else "")


@pytest.mark.parametrize("router_hardware", ["h200_sxm", "gb200", None])
def test_pd_predict_checks_effective_prefill_hardware_in_router_hook(router_hardware):
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.sweeper.provider import AdapterReplaySpec, RuntimeHookSpec

    config = CorePredictionConfig.model_validate(_pd_hardware_config(prefill="gb200", decode="gb200"))
    spec = AdapterReplaySpec(
        runtime_hooks=(
            RuntimeHookSpec(
                provider="dynamo.router",
                kind="placement_policy",
                api_version=1,
                config={
                    "router_config": {"router_prefill_load_model": "aic"},
                    "aic_perf_config": {"aic_system": router_hardware},
                },
            ),
        ),
    )
    if router_hardware == "gb200":
        prediction_to_replay_spec(config, adapter_specs={"dynamo.router": spec})
    else:
        with pytest.raises(ValueError, match="does not match effective prefill_hardware_sku"):
            prediction_to_replay_spec(config, adapter_specs={"dynamo.router": spec})
