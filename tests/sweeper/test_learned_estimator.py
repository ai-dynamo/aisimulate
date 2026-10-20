# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learned configuration must reject callers that cannot supply full FPM metrics."""

import json

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_learned_configuration_round_trip_and_phase_only_replay_rejection(tmp_path):
    """Accept the canonical model contract; never fabricate missing batch features."""
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config.cli import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate.sweeper.config import SearchSpace

    source = tmp_path / "forest.json"
    source.write_text(
        json.dumps(
            {
                "schema": "aic_fpm_learned_forward_perf",
                "schema_version": 1,
                "worker_type": "aggregated",
                "target": "log_ms",
                "features": ["num_decode_requests"],
                "stores": {"pure_decode": {"baseline": 1.0, "trees": []}},
            }
        )
    )
    controls = {"fpm_learned": {"artifact_path": str(source), "adaptation": "frozen"}}
    space = SearchSpace(
        model_name="Qwen/Qwen3-32B",
        hardware_sku="h200_sxm",
        deployment_mode=["agg"],
        estimation_mode="fpm_learned",
        estimator_config=controls,
    )
    restored_space = SearchSpace.model_validate_json(space.model_dump_json())
    assert restored_space.estimator_config == controls
    config = CorePredictionConfig.model_validate(
        {
            "engine": {
                "model": "Qwen/Qwen3-32B",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.24.0",
                "context_length": 4096,
                "estimation_mode": "fpm_learned",
                "estimator_config": controls,
                "workers": {
                    "aggregated": {
                        "parallelism": {"tensor": 2},
                        "kv_cache": {"capacity": {"type": "fixed", "blocks": 128}},
                    }
                },
            },
            "traffic": {
                "source": {"type": "synthetic", "input_tokens": 8, "output_tokens": 2},
                "load": {"type": "concurrency", "concurrency": 1},
                "stop": {"requests": 1},
            },
        }
    )
    config = CorePredictionConfig.model_validate_json(config.model_dump_json())
    replay = prediction_to_replay_spec(config)
    runner = EngineReplayRunnerFactory().create(0)
    try:
        with pytest.raises(RuntimeError, match="complete ForwardPassMetrics"):
            runner.run(replay)
    finally:
        runner.close()
