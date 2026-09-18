# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt lookup must couple target verification cost and sampled progress."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aisimulate import EngineReplayRunnerFactory, ReplayOutputRequirements
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.config.engine import NgramSpeculationConfig
from aisimulate.main import main
from aisimulate.recommend import recommendation_to_sweeper
from aisimulate.sweeper.config import SearchSpace

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = {"kind": "ngram", "num_speculative_tokens": 2, "acceptance_rates": [1.0, 1.0], "seed": 42}
_COST = {"kind": "ngram", "params": {"num_speculative_tokens": 2}}


def _prediction(*, mode="aggregated", rates=(1.0, 1.0), timing="fixed"):
    worker = {
        "scheduler": {"max_batched_tokens": 64, "max_sequences": 4},
        "kv_cache": {"block_size": 4, "capacity": {"type": "fixed", "blocks": 128}},
    }
    if timing == "fixed":
        worker["timing"] = {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}
    return {
        "engine": {
            "mode": mode,
            "model": "meta-llama/Meta-Llama-3.1-8B",
            "hardware": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.24.0",
            "context_length": 2048,
            "speculation": {**_SPEC, "acceptance_rates": list(rates)},
            "workers": {
                role: deepcopy(worker) for role in (("aggregated",) if mode == "aggregated" else ("prefill", "decode"))
            },
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 8, "output_tokens": 7},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        },
    }


def _run(raw):
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    runner = EngineReplayRunnerFactory().create(0)
    try:
        return runner.run(spec, output_requirements=ReplayOutputRequirements(include_raw_report=True))
    finally:
        runner.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_speculative_tokens": 0},
        {"num_speculative_tokens": 6},
        {"num_speculative_tokens": True},
        {"num_speculative_tokens": 2.5},
        {"acceptance_rates": [1.0]},
        {"acceptance_rates": [1.0, 1.0, 1.0]},
        {"acceptance_rates": [float("nan"), 1]},
        {"acceptance_rates": [-0.1, 1]},
        {"acceptance_rates": [1.1, 1]},
        {"acceptance_rates": [True, 1]},
        {"seed": -1},
        {"seed": 1 << 64},
        {"seed": True},
        {"trigger_rate": 0.5},
        {"kind": "mtp"},
    ],
)
def test_invalid_assumptions_are_rejected(overrides):
    with pytest.raises(ValidationError):
        NgramSpeculationConfig.model_validate({**_SPEC, **overrides})


def test_acceptance_is_required():
    with pytest.raises(ValidationError, match="acceptance_rates"):
        NgramSpeculationConfig.model_validate({"kind": "ngram", "num_speculative_tokens": 2})


@pytest.mark.parametrize("backend", ["sglang", "trtllm"])
def test_unqualified_backends_are_rejected(backend):
    raw = _prediction()
    raw["engine"]["backend"] = backend
    with pytest.raises(ValidationError, match="requires vllm"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"timing": {"forward_model": "fpm"}}, "op_level"),
        ({"kv_cache": {"host_offload": {"num_host_blocks": 64}}}, "host_offload"),
    ],
)
def test_unsupported_compositions_are_rejected(change, error):
    raw = _prediction()
    raw["engine"]["workers"]["aggregated"].update(change)
    with pytest.raises(ValidationError, match=error):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
def test_native_replay_samples_conditional_progress_per_verification(mode):
    accepted = _run(_prediction(mode=mode))
    rejected = _run(_prediction(mode=mode, rates=(0.0, 1.0)))
    baseline = _prediction(mode=mode)
    baseline["engine"].pop("speculation")
    ordinary = _run(baseline)
    assert accepted.metrics["completed_requests"] == rejected.metrics["completed_requests"] == 1
    assert accepted.metrics["mean_e2e_latency_ms"] < rejected.metrics["mean_e2e_latency_ms"]
    assert rejected.metrics["mean_e2e_latency_ms"] == ordinary.metrics["mean_e2e_latency_ms"]
    assert accepted.metrics["mean_ttft_ms"] == rejected.metrics["mean_ttft_ms"]


def test_native_aic_compiles_ngram_cost_without_mtp_draft_layers(monkeypatch):
    from aiconfigurator_core.sdk import engine

    calls = []
    original = engine.compile_engine
    original_spec = engine.build_engine_spec_json

    def capture_graph(model, **kwargs):
        assert model.spec_scheme.kind == "ngram"
        assert model.spec_scheme.verify_width() == 3
        assert model.spec_scheme.build_draft_generation_ops(model) == []
        assert model.spec_scheme.build_draft_context_ops(model) == []
        assert model.spec_scheme.draft_weights_bytes(model) == 0
        assert model.spec_scheme.draft_kv_bytes_per_sequence(model, 128) == 0
        return original_spec(model, **kwargs)

    monkeypatch.setattr(engine, "build_engine_spec_json", capture_graph)

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "compile_engine", capture)
    report = _run(_prediction(timing="default"))
    assert report.metrics["completed_requests"] == 1
    assert report.metrics["mean_e2e_latency_ms"] > 0
    assert calls and all(call["speculation"] == _COST and call["nextn"] == 0 for call in calls)


def test_recommend_predict_round_trip_preserves_speculation(tmp_path):
    raw = yaml.safe_load(
        (_ROOT / "tests/e2e/configs/unified_cli/recommend/engine/02-custom-preset-throughput-per-gpu.yaml").read_text()
    )
    raw["engine"]["speculation"] = deepcopy(_SPEC)
    raw["traffic"]["source"]["output_tokens"] = 7
    raw["optimizer"]["max_trials"] = 1
    raw["engine"]["workers"]["aggregated"]["parallelism"]["preset"] = raw["engine"]["workers"]["aggregated"][
        "parallelism"
    ]["preset"][:1]
    path = tmp_path / "recommend.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "recommend"
    assert main(["recommend", "-c", str(path), "--output-dir", str(output)]) == 0
    saved = sorted((output / "recommendations").glob("*.yaml"))
    assert saved
    prediction = CorePredictionConfig.from_yaml(saved[0])
    assert prediction.engine.speculation.model_dump() == _SPEC
    assert main(["predict", "-c", str(saved[0]), "--output-dir", str(tmp_path / "predict")]) == 0


def test_set_override_reaches_native_replay(tmp_path):
    raw = _prediction()
    raw["engine"].pop("speculation")
    path = tmp_path / "prediction.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "output"
    assert (
        main(
            [
                "predict",
                "-c",
                str(path),
                "--set",
                "engine.speculation=" + yaml.safe_dump(_SPEC, default_flow_style=True).strip(),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads((output / "prediction.json").read_text())
    summary = report.get("summary", report)
    assert summary["mean_e2e_latency_ms"] == pytest.approx(4.0)


def test_lower_level_sweeper_rejects_legacy_mtp_combination():
    with pytest.raises(ValidationError, match="cannot be combined"):
        SearchSpace(
            model_name="test",
            hardware_sku="h200_sxm",
            deployment_mode=["agg"],
            backend=["vllm"],
            speculation=_SPEC,
            aic_nextn=2,
        )


@pytest.mark.parametrize("mode,inactive_role", [("agg", "prefill"), ("agg", "decode"), ("disagg", "agg")])
@pytest.mark.parametrize(
    "field,value,error",
    [
        ("forward_model", "fpm", "op_level"),
        ("native_host_offload", {"num_host_blocks": 64}, "host_offload"),
    ],
)
def test_ngram_sweeper_ignores_inactive_roles_until_selected(mode, inactive_role, field, value, error):
    from aisimulate.sweeper.parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
    from aisimulate.sweeper.sample import unroll_sample

    inactive_key = f"{inactive_role}_{field}"
    values = {
        "model_name": "test",
        "hardware_sku": "h200_sxm",
        "deployment_mode": [mode],
        "backend": ["vllm"],
        "speculation": _SPEC,
        inactive_key: value,
    }
    space = SearchSpace(**values)
    replica = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    selection = {"deployment_mode": mode, "backend": "vllm"}
    for role in ("agg",) if mode == "agg" else ("prefill", "decode"):
        selection[f"{role}_max_num_batched_tokens"] = 64
        selection[f"{role}_max_num_seqs"] = 4
    sample = unroll_sample(
        search_space=space,
        selection=selection,
        parallel_config=replica if mode == "agg" else DisaggParallelConfig(prefill=replica, decode=replica),
    )
    assert inactive_key not in sample
    assert sample["speculation"] == _SPEC

    with pytest.raises(ValidationError, match=error):
        SearchSpace(**{**values, "deployment_mode": ["agg", "disagg"]})


def test_generator_does_not_silently_drop_prompt_lookup():
    from aiconfigurator.generator.request import SweeperCandidateError, from_sweeper_candidate

    with pytest.raises(SweeperCandidateError, match="ngram deployment generation is unsupported"):
        from_sweeper_candidate({"config": {"speculation": _SPEC}})


def test_recommendation_config_lowers_speculation_as_pinned_setting():
    raw = _prediction()
    raw["optimization"] = {"target": "throughput", "constraints": {"max_candidate_gpus": 1}}
    config = CoreRecommendationConfig.model_validate(raw)
    smart = recommendation_to_sweeper(config)
    assert smart.search_space.speculation.model_dump() == _SPEC


@pytest.mark.parametrize("output_tokens", [1, 2, 5, 7])
def test_native_progress_clamps_final_burst(output_tokens):
    raw = _prediction()
    raw["traffic"]["source"]["output_tokens"] = output_tokens
    report = _run(raw)
    # The native scheduler performs prefill, then up to three tokens per decode round.
    rounds = (output_tokens + 2) // 3
    assert report.metrics["mean_e2e_latency_ms"] == pytest.approx(1 + rounds)


def test_ngram_rejection_still_pays_target_verification_cost():
    raw = _prediction(timing="default", rates=(0.0, 1.0))
    speculative = _run(raw)
    raw["engine"].pop("speculation")
    ordinary = _run(raw)
    assert speculative.metrics["mean_e2e_latency_ms"] > ordinary.metrics["mean_e2e_latency_ms"]


def test_agentic_execution_rejects_ngram_before_loading_trace():
    raw = _prediction()
    raw["traffic"] = {
        "source": {"type": "trace", "paths": ["not-loaded.jsonl"], "format": "weka"},
        "load": {"type": "trace_timestamps"},
    }
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    with pytest.raises(ValueError, match="speculative decoding disabled"):
        EngineReplayRunnerFactory().capabilities().require_compatible(spec)


def test_online_prediction_rejects_ngram():
    config = CorePredictionConfig.model_validate(_prediction())
    with pytest.raises(ValueError, match="offline engine stack"):
        prediction_to_replay_spec(config, execution_mode="online")
