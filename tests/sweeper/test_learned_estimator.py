# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learned configuration must reject callers that cannot supply full FPM metrics."""

import json

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize("source_field", ["artifact_path", "checkpoint_path"])
@pytest.mark.parametrize("boundary", ["runner", "native"])
def test_raw_learned_replay_rejection_precedes_source_loading(tmp_path, source_field, boundary):
    """Low-level replay must enforce the capability boundary before opening a source."""
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config.cli import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory, _materialize_engine_execution_spec

    config = CorePredictionConfig.model_validate(
        {
            "engine": {
                "model": "Qwen/Qwen3-32B",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.24.0",
                "context_length": 4096,
                "workers": {
                    "aggregated": {
                        "parallelism": {"tensor": 2},
                        "kv_cache": {"capacity": {"type": "fixed", "blocks": 128}},
                        "timing": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
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
    # Public prediction/SearchSpace validation rejects this mode already. Inject
    # it into raw timing input to exercise native replay's independent guard.
    source = tmp_path / "missing-learned-source"
    assert not source.exists()
    timing = {
        "type": "external",
        "provider": "aic",
        "config": {
            "model": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.24.0",
            "worker_type": "aggregated",
            "tp": 2,
            "estimation_mode": "fpm_learned",
            "estimator_config": {"fpm_learned": {source_field: str(source)}},
        },
    }
    if boundary == "native":
        from aisimulate import _runtime

        execution = _materialize_engine_execution_spec(replay, trace_block_size=512, record_per_request=False)
        execution["spec"]["engine"]["rank"]["timing_model"] = timing
        with pytest.raises(RuntimeError, match="engine replay's phase-only timing"):
            _runtime.run_replay_json(json.dumps(execution))
        return
    replay.backend_deployment.agg_engine_args["timing_model"] = timing
    runner = EngineReplayRunnerFactory().create(0)
    try:
        with pytest.raises(ValueError, match="engine replay's phase-only timing"):
            runner.run(replay)
    finally:
        runner.close()
