# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-width DSpark uses its own identity with the shared replay machinery."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from test_ngram_cli import _prediction, _run

from aisimulate.config import CorePredictionConfig
from aisimulate.config.engine import DSparkSpeculationConfig
from aisimulate.main import main
from aisimulate_core.sdk.memory import KVCacheEstimator
from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel
from aisimulate_core.sdk.speculation.base import from_forward_pass_config
from aisimulate_core.sdk.utils import _load_model_config_from_model_path


def _dspark(*, k=6, accepted=2.77, mode="aggregated", backend="vllm", output=19):
    raw = _prediction(mode=mode)
    raw["engine"]["backend"] = backend
    raw["engine"]["speculation"] = {
        "kind": "dspark",
        "num_speculative_tokens": k,
        "accepted_tokens": accepted,
        "adaptive_verification": False,
        "seed": 42,
    }
    raw["traffic"]["source"]["output_tokens"] = output
    return raw


@pytest.mark.parametrize("k", [3, 5, 6, 7])
@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_fixed_width_progress_and_short_completion(k, mode, backend):
    full = _run(_dspark(k=k, accepted=k, mode=mode, backend=backend))
    zero = _run(_dspark(k=k, accepted=0, mode=mode, backend=backend))
    short = _run(_dspark(k=k, accepted=k, mode=mode, backend=backend, output=2))
    assert full.metrics["completed_requests"] == zero.metrics["completed_requests"] == 1
    assert short.metrics["completed_requests"] == 1
    assert short.metrics["total_output_tokens"] == 2
    assert full.metrics["total_output_tokens"] == zero.metrics["total_output_tokens"] == 19
    assert full.metrics["mean_e2e_latency_ms"] < zero.metrics["mean_e2e_latency_ms"]


def test_fractional_acceptance_seed_repeats():
    first = _run(_dspark(output=123))
    second = _run(_dspark(output=123))
    assert first.metrics["mean_e2e_latency_ms"] == second.metrics["mean_e2e_latency_ms"]


@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_context_limit_clips_multi_token_completion(mode, backend):
    raw = _dspark(k=7, accepted=7, output=32, mode=mode, backend=backend)
    raw["engine"]["context_length"] = 20
    report = _run(raw)
    assert report.metrics["completed_requests"] == 1
    assert report.metrics["total_output_tokens"] == 12


@pytest.mark.parametrize(
    "overrides",
    [
        {"adaptive_verification": True},
        {"num_speculative_tokens": 0},
        {"num_speculative_tokens": True},
        {"accepted_tokens": 6.1},
        {"accepted_tokens": -1},
        {"accepted_tokens": float("nan")},
        {"acceptance_rates": [1.0] * 6},
        {"accepted_tokens": None},
        {"accepted_tokens": None, "acceptance_rates": [1.0] * 5},
    ],
)
def test_invalid_configuration(overrides):
    raw = _dspark()["engine"]["speculation"]
    with pytest.raises(ValidationError):
        DSparkSpeculationConfig.model_validate({**raw, **overrides})


@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("warmup", [None, False, True])
def test_agentx_dependencies_complete(mode, backend, warmup):
    raw = _dspark(mode=mode, backend=backend)
    raw["traffic"] = {
        "source": {
            "type": "trace",
            "format": "weka",
            "paths": [str(Path(__file__).parent / "e2e/configs/unified_cli/fixtures/traces/weka-relative.json")],
        },
        "load": {
            "type": "trace_timestamps",
            "agentic_lanes": 1,
        },
    }
    if warmup is not None:
        raw["traffic"]["load"].update(agentic_snapshot={"seed": 42}, agentic_warmup=warmup)
    report = _run(raw)
    # Seed 42 starts after the first saved turn; warmup primes it without scoring it.
    assert report.metrics["completed_requests"] == (4 if warmup is None else 3)
    assert report.metadata["agentic_qualification"] == "functional_only"


def test_draft_revision_participates_in_compiled_identity():
    value = {
        "kind": "dspark",
        "params": {
            "num_speculative_tokens": 6,
            "draft_model_path": "example/draft",
            "draft_revision": "revision-a",
            "draft_config": {"dspark_block_size": 5},
        },
    }
    first = from_forward_pass_config(value).identity_hash()
    value["params"]["draft_revision"] = "revision-b"
    assert from_forward_pass_config(value).identity_hash() != first


@pytest.mark.parametrize("backend,op_level", [("vllm", False), ("sglang", False), ("vllm", True)])
def test_recommendation_saved_prediction_preserves_dspark(tmp_path, backend, op_level):
    root = Path(__file__).parent / "e2e/configs/unified_cli"
    raw = yaml.safe_load((root / "recommend/engine/02-custom-preset-throughput-per-gpu.yaml").read_text())
    raw["engine"]["backend"] = backend
    spec = DSparkSpeculationConfig.model_validate(_dspark()["engine"]["speculation"]).model_dump()
    raw["engine"]["speculation"] = spec
    raw["optimizer"]["max_trials"] = 1
    expected = dict(spec)
    if op_level:
        snapshot = dict(_load_model_config_from_model_path("deepseek-ai/DeepSeek-V4-Flash"))
        snapshot.update(dspark_block_size=5, dspark_target_layer_ids=[40, 41, 42])
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps(snapshot))
        raw["engine"].update(model=str(model_dir), hardware="b200_sxm", backend_version="0.24.0")
        worker = raw["engine"]["workers"]["aggregated"]
        worker["timing"] = {"type": "default"}
        preset = worker["parallelism"]["preset"][0]
        preset.update(tensor=4, moe_expert=4, moe_tensor=1)
        worker["parallelism"] = {"preset": [preset]}
        worker["kv_cache"]["block_size"] = 256
        raw["optimization"]["constraints"]["max_candidate_gpus"] = 4
        expected["draft_config"] = snapshot
    path = tmp_path / "recommend.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "recommend"
    assert main(["recommend", "-c", str(path), "--output-dir", str(output)]) == 0
    saved = sorted((output / "recommendations").glob("*.yaml"))
    assert saved
    prediction = CorePredictionConfig.from_yaml(saved[0])
    assert prediction.engine.speculation.model_dump() == expected
    assert main(["predict", "-c", str(saved[0]), "--output-dir", str(tmp_path / "predict")]) == 0


def test_pd_auto_transfer_and_real_op_cost():
    raw = _dspark(mode="disaggregated")
    engine = raw["engine"]
    engine.update(model="deepseek-ai/DeepSeek-V4-Flash", hardware="b200_sxm")
    engine["speculation"]["draft_config"] = {
        "dspark_block_size": 5,
        "dspark_target_layer_ids": [40, 41, 42],
    }
    engine["kv_transfer"] = {"bandwidth_gb_per_second": 50, "bytes_per_token": "auto"}
    for worker in engine["workers"].values():
        worker.pop("timing")
        worker["parallelism"] = {"tensor": 4, "moe_expert": 4, "moe_tensor": 1}
        worker["kv_cache"]["block_size"] = 256
    report = _run(raw)
    assert report.metrics["completed_requests"] == 1
    assert report.metrics["total_output_tokens"] == 19
    assert report.metrics["mean_ttft_ms"] > 0


def test_cost_identity_is_resolved_and_excludes_acceptance():
    spec = DSparkSpeculationConfig.model_validate(_dspark()["engine"]["speculation"])
    cost = spec.cost_config()
    cost["params"]["draft_config"] = {
        "dspark_block_size": 5,
        "dspark_target_layer_ids": [40, 41, 42],
    }
    config = {
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "system": "b200_sxm",
        "backend": "vllm",
        "worker_type": "aggregated",
        "estimation_mode": "op_level",
        "speculation": cost,
    }
    normalized = RustForwardPassPerfModel.normalize_config(config)
    assert normalized["speculation"] == cost
    assert "accepted_tokens" not in cost["params"]
    with pytest.raises(ValueError, match="op_level"):
        RustForwardPassPerfModel.normalize_config({**config, "estimation_mode": "fpm_interpolation"})
    with pytest.raises(ValueError, match="nextn"):
        CorePredictionConfig.model_validate({**_dspark(), "engine": {**_dspark()["engine"], "nextn": 1}})


@pytest.mark.parametrize(
    "name,backend,version,tp,ep",
    [
        ("Flash", "vllm", "0.24.0", 4, 4),
        ("Pro", "vllm", "0.24.0", 8, 1),
        ("Pro", "sglang", "0.5.14", 8, 1),
    ],
)
def test_canonical_op_cost_and_draft_memory(name, backend, version, tp, ep):
    model_name = f"deepseek-ai/DeepSeek-V4-{name}"
    layers = _load_model_config_from_model_path(model_name)["num_hidden_layers"]
    cost = {
        "kind": "dspark",
        "params": {
            "num_speculative_tokens": 6,
            "draft_config": {"dspark_block_size": 5, "dspark_target_layer_ids": list(range(layers - 3, layers))},
        },
    }
    config = ForwardPassPerfModelConfig(
        model=model_name,
        system="b200_sxm",
        backend=backend,
        backend_version=version,
        worker_type="aggregated",
        tp=tp,
        moe_tp_size=tp // ep,
        moe_ep_size=ep,
        estimation_mode="op_level",
        speculation=cost,
    )
    model = RustForwardPassPerfModel.best_available(config)
    assert model.diagnostics()["provenance"]["modeling_assumptions"]
    assert model.predict_decode_latency_total(1, 2048) > 0
    operations = model.static_phase_diagnostics(batch_size=1, context_length=2048, prefill=False)
    assert operations
    kwargs = dict(
        max_num_tokens=128,
        max_batch_size=4,
        tp_size=tp,
        moe_tp_size=tp // ep,
        moe_ep_size=ep,
    )
    ordinary = KVCacheEstimator.from_request(
        config.model,
        config.system,
        config.backend,
        config.backend_version,
        **kwargs,
    ).breakdown
    speculative = KVCacheEstimator.from_request(
        config.model,
        config.system,
        config.backend,
        config.backend_version,
        speculation=cost,
        **kwargs,
    ).breakdown
    assert speculative["weights_bytes"] > ordinary["weights_bytes"]
    assert speculative["tokens_from_kv_bytes"](1e9) < ordinary["tokens_from_kv_bytes"](1e9)
