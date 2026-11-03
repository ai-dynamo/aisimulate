# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine replay prices every pass of a learned role with the canonical iteration query."""

import json

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

_MODEL = "Qwen/Qwen3-0.6B"
_DECODE_MS = 5.0


def _prediction(decode_timing: dict | None = None, *, fixed: bool = False) -> dict:
    timing = {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0} if fixed else {}
    return {
        "engine": {
            "mode": "disaggregated",
            "model": _MODEL,
            "hardware": "h200_sxm",
            "backend": "vllm",
            "context_length": 256,
            "workers": {
                role: {
                    "parallelism": {"tensor": 1},
                    "kv_cache": {"block_size": 16, "capacity": {"type": "fixed", "blocks": 512}},
                    **({"timing": timing} if timing else {}),
                    **({"timing": decode_timing} if role == "decode" and decode_timing else {}),
                }
                for role in ("prefill", "decode")
            },
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 64, "output_tokens": 16},
            "load": {"type": "concurrency", "concurrency": 2},
            "stop": {"requests": 4},
        },
    }


def _replay(raw: dict):
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config.cli import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory

    runner = EngineReplayRunnerFactory().create(0)
    try:
        return runner.run(prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)))
    finally:
        runner.close()


@pytest.mark.parametrize("source_field", ["artifact_path", "checkpoint_path"])
@pytest.mark.parametrize("boundary", ["runner", "native"])
def test_raw_learned_replay_reads_its_source_at_construction(tmp_path, source_field, boundary):
    """Low-level replay accepts learned timing and fails only on the missing source."""
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config.cli import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory, _materialize_engine_execution_spec

    replay = prediction_to_replay_spec(CorePredictionConfig.model_validate(_prediction(fixed=True)))
    source = tmp_path / "missing-learned-source"
    assert not source.exists()
    timing = {
        "type": "external",
        "provider": "aic",
        "config": {
            "model": _MODEL,
            "system": "h200_sxm",
            "backend": "vllm",
            "worker_type": "decode",
            "tp": 1,
            "estimation_mode": "fpm_learned",
            "estimator_config": {"fpm_learned": {source_field: str(source), "adaptation": "frozen"}},
        },
    }
    if boundary == "native":
        from aisimulate import _runtime

        execution = _materialize_engine_execution_spec(replay, trace_block_size=512, record_per_request=False)
        execution["spec"]["engine"]["decode"]["rank"]["timing_model"] = timing
        with pytest.raises((RuntimeError, OSError), match="missing-learned-source"):
            _runtime.run_replay_json(json.dumps(execution))
        return
    replay.backend_deployment.decode_engine_args["timing_model"] = timing
    runner = EngineReplayRunnerFactory().create(0)
    try:
        with pytest.raises((RuntimeError, ValueError, OSError), match="missing-learned-source"):
            runner.run(replay)
    finally:
        runner.close()


def test_a_learned_decode_role_prices_each_decode_iteration(tmp_path):
    """A tiny forest fitted to a constant decode time times every decode pass."""
    pytest.importorskip("sklearn")
    from aisimulate_core.sdk.fpm_learned import train

    iterations = []
    for index in range(64):
        requests = 1 + index % 4
        rank = {
            "version": 1,
            "wall_time": _DECODE_MS / 1000.0,
            "scheduled_requests": {
                "num_decode_requests": requests,
                "sum_decode_kv_tokens": requests * (64 + 7 * (index % 13)),
            },
        }
        iterations.append([rank])
    artifact = train(iterations, "decode", max_iter=5, min_store_rows=8)
    path = tmp_path / "decode-forest.json"
    path.write_text(json.dumps(artifact))
    learned = {
        "estimation_mode": "fpm_learned",
        "fallback_policy": "deny",
        "estimator_config": {"fpm_learned": {"artifact_path": str(path), "adaptation": "frozen"}},
    }

    result = _replay(_prediction(learned))
    baseline = _replay(_prediction())

    # The forest predicts the constant it was fitted to for every decode pass.
    assert result.metrics["mean_tpot_ms"] == pytest.approx(_DECODE_MS, rel=0.05)
    assert baseline.metrics["mean_tpot_ms"] != pytest.approx(_DECODE_MS, rel=0.05)
    # Learned timing is latency-only.
    assert result.metrics["power_w"] is None
