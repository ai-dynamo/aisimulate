# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public MTP override normalization, compatibility and real target replay."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from aisimulate import EngineReplayRunnerFactory, ReplayOutputRequirements
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.config.engine import MtpSpeculationConfig
from aisimulate.runner import normalize_mtp_engine_args
from aisimulate.speculation import normalize_speculation_engine_args, speculation_report_metadata
from aisimulate.sweeper.replay import BackendDeploymentSpec, ReplaySpec

MTP = {
    "kind": "mtp",
    "num_speculative_tokens": 3,
    "expected_accepted_tokens": 1.5,
    "seed": 42,
}


def prediction(model="nvidia/GLM-5.2-NVFP4", backend="sglang", topology="aggregated"):
    worker = {
        "parallelism": {"tensor": 8, "moe_tensor": 8},
        "scheduler": {"max_batched_tokens": 8192, "max_sequences": 8},
        "kv_cache": {"block_size": 64, "capacity": {"type": "fixed", "blocks": 4096}},
        "timing": {"type": "default"},
    }
    return {
        "engine": {
            "mode": topology,
            "model": model,
            "backend": backend,
            "hardware": "b300_sxm",
            "backend_version": "0.5.14" if backend == "sglang" else "0.24.0",
            "context_length": 262144,
            "speculation": deepcopy(MTP),
            "workers": {
                role: deepcopy(worker)
                for role in (("aggregated",) if topology == "aggregated" else ("prefill", "decode"))
            },
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 128, "output_tokens": 9},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 2},
        },
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_speculative_tokens", 0),
        ("num_speculative_tokens", 6),
        ("num_speculative_tokens", True),
        ("expected_accepted_tokens", -1),
        ("expected_accepted_tokens", 3.1),
        ("expected_accepted_tokens", float("nan")),
        ("expected_accepted_tokens", True),
        ("seed", -1),
        ("seed", 1 << 64),
        ("seed", True),
    ],
)
def test_invalid_mtp_assumption(field, value):
    with pytest.raises(ValidationError):
        MtpSpeculationConfig.model_validate({**MTP, field: value})


def test_acceptance_required_and_independent_default_seed():
    with pytest.raises(ValidationError, match="expected_accepted_tokens"):
        MtpSpeculationConfig.model_validate({"kind": "mtp", "num_speculative_tokens": 3})
    config = MtpSpeculationConfig.model_validate({k: v for k, v in MTP.items() if k != "seed"})
    assert config.seed == 42
    assert MtpSpeculationConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize("expected,rates", [(0, "0,0,0"), (1.5, "1,0.5,0"), (3, "1,1,1")])
def test_shared_normalizer_equates_old_and_new_sampler(expected, rates):
    new = {
        "speculation": {**MTP, "expected_accepted_tokens": expected},
        "timing_model": {"type": "external", "provider": "aic", "config": {}},
    }
    old = {"nextn": 3, "nextn_accepted": expected, "mtp_seed": 42}
    normalized = normalize_mtp_engine_args(new, role="decode")
    assert {k: v for k, v in normalized.items() if k != "timing_model"} == normalize_mtp_engine_args(old, role="decode")
    assert normalized["timing_model"]["config"]["speculation"] == MtpSpeculationConfig.model_validate(MTP).cost_config()
    assert normalized["aic_nextn_accept_rates"] == rates
    assert normalize_mtp_engine_args(normalized, role="decode") == normalized
    assert new["speculation"]["expected_accepted_tokens"] == expected


def test_legacy_normalizer_import_is_the_stable_shared_api():
    assert normalize_mtp_engine_args is normalize_speculation_engine_args


@pytest.mark.parametrize("field", ["aic_nextn", "nextn"])
@pytest.mark.parametrize("value", [0, None])
def test_disabled_depth_normalizes_idempotently(field, value):
    normalized = normalize_speculation_engine_args({field: value, "aic_mtp_seed": 42}, role="decode")
    assert normalized == {}
    assert normalize_speculation_engine_args(normalized, role="decode") == normalized


@pytest.mark.parametrize("dependent", [{"aic_nextn_accepted": 0}, {"nextn_accept_rates": "1"}, {"mtp_seed": 73}])
def test_disabled_depth_rejects_dependent_sampler_controls(dependent):
    with pytest.raises(ValueError, match="requires aic_nextn"):
        normalize_speculation_engine_args({"aic_nextn": 0, **dependent}, role="decode")


@pytest.mark.parametrize(
    "selection",
    [MTP, {"kind": "ngram", "num_speculative_tokens": 3, "acceptance_rates": [1.0, 0.5, 0.0]}],
)
def test_shared_normalizer_never_loses_method_without_cost_identity(selection):
    with pytest.raises(ValueError, match="materialize the target cost identity first"):
        normalize_speculation_engine_args({"speculation": selection}, role="decode")


@pytest.mark.parametrize("kind,model", [("ngram", "meta-llama/Meta-Llama-3.1-8B"), ("mtp", "moonshotai/Kimi-K3")])
def test_flat_materialization_preserves_method_in_canonical_cost(kind, model):
    from aisimulate.runner import _materialize_engine_role
    from aisimulate_core.sdk import RustForwardPassPerfModel

    selection = (
        MTP if kind == "mtp" else {"kind": "ngram", "num_speculative_tokens": 3, "acceptance_rates": [1.0, 0.5, 0.0]}
    )
    authored = {
        "aic_model_path": model,
        "aic_system": "b200_sxm",
        "aic_tp_size": 8,
        "num_gpu_blocks": 8192,
        "speculation": selection,
    }
    if kind == "mtp":
        authored.update(aic_moe_tp_size=8, aic_moe_ep_size=1)
    resolved = _materialize_engine_role("vllm", "0.24.0", {"tp": 8}, authored, "aggregated")
    config = resolved["rank"]["timing_model"]["config"]
    assert config["speculation"] == {"kind": kind, "params": {"num_speculative_tokens": 3}}
    assert config.get("nextn", 0) == 0
    assert normalize_speculation_engine_args(resolved["rank"], role="aggregated") == resolved["rank"]
    request = {**config, "worker_type": "aggregated", "estimation_mode": "op_level"}
    if kind == "mtp":
        with pytest.raises(ValueError, match="DSPARK"):
            RustForwardPassPerfModel.best_available(request)
    else:
        cost = RustForwardPassPerfModel.best_available(request)
        try:
            assert cost.diagnostics()["provenance"]["config"]["speculation"]["kind"] == "ngram"
        finally:
            cost.close()


def _metadata_spec(args, metadata=None):
    return ReplaySpec(
        BackendDeploymentSpec("agg", "vllm", "0.24.0", agg_engine_args=args, performance_model_metadata=metadata or {}),
        {},
        {},
    )


def test_metadata_uses_resolved_cost_and_original_inferred_capacity():
    original = {"aic_model_path": "moonshotai/Kimi-K3", "aic_nextn": 3, "aic_nextn_accepted": 1.5}
    resolved = {
        "num_gpu_blocks_is_explicit": False,
        "rank": {
            "aic_nextn": 3,
            "aic_nextn_accept_rates": "1,0.5,0",
            "num_gpu_blocks": 8192,
            "timing_model": {
                "type": "external",
                "provider": "aic",
                "config": {"model": original["aic_model_path"], "nextn": 3},
            },
        },
    }
    metadata = speculation_report_metadata(_metadata_spec(original), resolved_role_args={"aggregated": resolved})[
        "aggregated"
    ]
    assert metadata["resolved_method"] == "dspark"
    assert metadata["requested"] == {"kind": "legacy_nextn", "num_speculative_tokens": 3}
    assert metadata["cost_approximation"] == "legacy_dspark_graph"
    assert metadata["capacity_source"] == "inferred"
    assert metadata["expected_accepted_draft_tokens"] == 1.5
    original.pop("aic_nextn_accepted")
    metadata = speculation_report_metadata(_metadata_spec(original), resolved_role_args={"aggregated": resolved})[
        "aggregated"
    ]
    assert metadata["conditional_acceptance_rates"] == "1,0.5,0"
    assert metadata["expected_accepted_draft_tokens"] is None


@pytest.mark.parametrize("kind", ["mtp", "ngram"])
@pytest.mark.parametrize("authored_canonical", [False, True])
def test_metadata_preserves_explicit_canonical_request(kind, authored_canonical):
    selection = {"kind": kind, "params": {"num_speculative_tokens": 3}}
    rank = {
        "aic_nextn": 3,
        "aic_nextn_accept_rates": "1,0.5,0",
        "aic_mtp_seed": 73,
        "timing_model": {
            "type": "external",
            "provider": "aic",
            "config": {"model": "configured-target", "speculation": selection},
        },
    }
    authored = deepcopy(rank)
    if not authored_canonical:
        # Adapters may carry their authored identity separately, then provide
        # the complete canonical cost in the resolved role descriptor.
        authored.pop("timing_model")
    original = deepcopy(authored)
    metadata = speculation_report_metadata(
        _metadata_spec(authored),
        resolved_role_args={"aggregated": {"rank": rank, "num_gpu_blocks_is_explicit": True}},
    )["aggregated"]
    assert metadata["requested"] == selection
    assert metadata["resolved_method"] == kind
    assert metadata["num_speculative_tokens"] == 3
    assert metadata["seed"] == 73
    assert metadata["capacity_source"] == "explicit_fixed"
    assert authored == original


def test_no_speculation_metadata_accepts_null_timing_without_lookup(monkeypatch):
    from aisimulate_core.sdk.models import helpers

    def unexpected(*args, **kwargs):
        pytest.fail("no-SD metadata must not look up a model")

    monkeypatch.setattr(helpers, "_get_model_info", unexpected)
    assert speculation_report_metadata(_metadata_spec({"timing_model": None})) == {}


def test_metadata_failure_happens_before_simulation(monkeypatch):
    import aisimulate.runner as runner_module

    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(prediction()))
    events = []

    def reject_metadata(*args, **kwargs):
        assert "aggregated" in kwargs["resolved_role_args"]
        raise ValueError("invalid speculative provenance")

    class Runtime:
        def run_replay_json(self, value):
            events.append("executed")
            raise AssertionError("runtime must not run before metadata validation")

    monkeypatch.setattr(runner_module, "speculation_report_metadata", reject_metadata)
    runner = EngineReplayRunnerFactory(runtime=Runtime()).create(0)
    with pytest.raises(ValueError, match="invalid speculative provenance"):
        runner.run(spec)
    assert events == []


@pytest.mark.parametrize("legacy", [{"nextn": 2}, {"nextn_accepted": 0}, {"aic_nextn_accept_rates": "1,1,1"}])
def test_normalizer_rejects_conflicting_sources(legacy):
    with pytest.raises(ValueError, match="cannot be combined"):
        normalize_mtp_engine_args({"speculation": MTP, **legacy}, role="decode")


@pytest.mark.parametrize("model", ["nvidia/GLM-5.2-NVFP4", "deepseek-ai/DeepSeek-V4-Pro"])
@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
def test_new_and_legacy_configuration_have_equal_native_execution(model, topology):
    raw = prediction(model=model, topology=topology)
    configured = CorePredictionConfig.model_validate(raw)
    assert (
        CorePredictionConfig.model_validate_json(configured.model_dump_json()).engine.speculation
        == configured.engine.speculation
    )
    new = prediction_to_replay_spec(configured)
    raw["engine"].pop("speculation")
    raw["engine"].update(nextn=3, nextn_accepted=1.5)
    old = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    runner = EngineReplayRunnerFactory().create(0)
    try:
        results = [
            runner.run(
                spec,
                output_requirements=ReplayOutputRequirements(include_raw_report=True),
            )
            for spec in (new, old)
        ]
    finally:
        runner.close()
    wall_clock_metrics = {
        "wall_time_ms",
        "processed_tokens_per_s",
        "processed_output_tokens_per_s",
    }
    assert {k: v for k, v in results[0].metrics.items() if k not in wall_clock_metrics} == {
        k: v for k, v in results[1].metrics.items() if k not in wall_clock_metrics
    }
    assert results[0].metrics["completed_requests"] == 2
    assert results[0].metadata["speculative_acceptance"] == results[1].metadata["speculative_acceptance"]
    for role in results[0].metadata["speculation"].values():
        assert role["target_model"] == model
        assert role["resolved_method"] == "mtp"
        assert role["expected_accepted_draft_tokens"] == 1.5
        assert role["capacity_source"] == "explicit_fixed"


def test_agentic_mtp_rejects_inferred_capacity_before_cost_lookup():
    raw = prediction()
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["capacity"] = {"type": "default"}
    raw["traffic"] = {
        "source": {"type": "trace", "format": "weka", "paths": ["unused"]},
        "load": {"type": "trace_timestamps", "agentic_lanes": 1},
    }
    with pytest.raises(ValueError, match="explicit fixed KV capacity"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))


def _agentic_capability_spec(args):
    return ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode="agg",
            backend="sglang",
            backend_version="0.5.14",
            agg_engine_args=args,
            num_workers=1,
        ),
        workload={"trace_format": "weka", "agentic_lanes": 1},
        goal={},
    )


def _canonical_mtp_rank():
    return {
        "num_gpu_blocks": 128,
        "aic_nextn_accept_rates": "1,0.5,0",
        "timing_model": {
            "type": "external",
            "provider": "aic",
            "config": {"speculation": {"kind": "mtp", "params": {"num_speculative_tokens": 3}}},
        },
    }


@pytest.mark.parametrize("canonical_legacy", [False, True])
@pytest.mark.parametrize("rates", [None, "", "  "])
def test_agentic_capability_requires_acceptance_for_canonical_cost(canonical_legacy, rates):
    rank = _canonical_mtp_rank()
    rank["aic_nextn_accept_rates"] = rates
    if canonical_legacy:
        rank["timing_model"]["config"] = {"nextn": 3}
    with pytest.raises(ValueError, match="explicit acceptance"):
        EngineReplayRunnerFactory().capabilities().require_compatible(_agentic_capability_spec(rank))


@pytest.mark.parametrize("wrapped", [False, True])
def test_agentic_capability_honors_inferred_capacity_provenance(wrapped):
    rank = _canonical_mtp_rank()
    args = {"rank": rank} if wrapped else rank
    args["num_gpu_blocks_is_explicit"] = False
    with pytest.raises(ValueError, match="explicit fixed KV capacity"):
        EngineReplayRunnerFactory().capabilities().require_compatible(_agentic_capability_spec(args))


@pytest.mark.parametrize(
    "timing",
    [
        {"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
        {"type": "polynomial"},
        {"type": "external", "provider": "custom", "config": {}},
        {"type": "external", "provider": "aic", "config": {"estimation_mode": "fpm_interpolation"}},
        {"type": "external", "provider": "aic", "config": {"forward_model": "fpm"}},
    ],
)
def test_agentic_capability_rejects_timing_without_mtp_cost(timing):
    rank = _canonical_mtp_rank()
    rank.update(aic_nextn=3, timing_model=timing)
    with pytest.raises(ValueError, match="AIC op_level timing"):
        EngineReplayRunnerFactory().capabilities().require_compatible(_agentic_capability_spec(rank))


def test_agentic_capability_does_not_let_public_mtp_mask_canonical_ngram():
    rank = _canonical_mtp_rank()
    rank["speculation"] = deepcopy(MTP)
    rank["timing_model"]["config"]["speculation"]["kind"] = "ngram"
    with pytest.raises(ValueError, match="only MTP"):
        EngineReplayRunnerFactory().capabilities().require_compatible(_agentic_capability_spec(rank))


@pytest.mark.parametrize("timing", [None, {"type": "default"}])
def test_agentic_capability_accepts_public_expected_zero_before_flat_aic_resolution(timing):
    rank = {
        "aic_model_path": "nvidia/GLM-5.2-NVFP4",
        "num_gpu_blocks": 128,
        "speculation": {**MTP, "expected_accepted_tokens": 0},
        "timing_model": timing,
    }
    EngineReplayRunnerFactory().capabilities().require_compatible(_agentic_capability_spec(rank))


def test_agentic_capability_keeps_zero_depth_fixed_timing_compatible():
    rank = {"aic_nextn": 0, "timing_model": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}}
    EngineReplayRunnerFactory().capabilities().require_compatible(_agentic_capability_spec(rank))


def test_mtp_recommendation_saves_cost_identity_and_runs_prediction(tmp_path):
    import json
    from pathlib import Path

    import yaml

    from aisimulate.main import main

    example = (
        Path(__file__).parent / "e2e/configs/unified_cli/recommend/engine/02-custom-preset-throughput-per-gpu.yaml"
    )
    raw = yaml.safe_load(example.read_text())
    raw["engine"].update(
        model="nvidia/GLM-5.2-NVFP4",
        hardware="b300_sxm",
        backend="sglang",
        backend_version="0.5.14",
        speculation=deepcopy(MTP),
        estimation_mode="op_level",
    )
    worker = raw["engine"]["workers"]["aggregated"]
    worker["timing"] = {"type": "default"}
    worker["parallelism"]["preset"] = [
        {"replicas": 1, "tensor": 8, "pipeline": 1, "attention_data": 1, "moe_tensor": 8, "moe_expert": 1}
    ]
    raw["optimization"]["constraints"]["max_candidate_gpus"] = 8
    raw["optimizer"]["max_trials"] = 1
    raw["traffic"]["source"]["output_tokens"] = 9
    config = tmp_path / "recommend.yaml"
    config.write_text(yaml.safe_dump(raw))
    output = tmp_path / "recommend"
    assert main(["recommend", "-c", str(config), "--output-dir", str(output)]) == 0
    saved = list((output / "recommendations").glob("*.yaml"))
    assert len(saved) == 1
    prediction_config = CorePredictionConfig.from_yaml(saved[0])
    assert prediction_config.engine.speculation.model_dump() == MTP
    prediction_output = tmp_path / "predict"
    assert main(["predict", "-c", str(saved[0]), "--output-dir", str(prediction_output)]) == 0
    report = json.loads((prediction_output / "prediction.json").read_text())
    assert report.get("summary", report)["completed_requests"] == 6
