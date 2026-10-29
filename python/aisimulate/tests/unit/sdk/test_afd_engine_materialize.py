# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native AFD rank materialization (`_materialize_afd_engine_role`) contracts.

Pins rank-owned topology, backend, MTP depth and context-window identity.
The timing provider reads these from the rank instead of carrying independent
copies. Also pins the
continuous-batching scheduler derivation (max_num_seqs and the prefill
pins; num_gpu_blocks is derived natively from the memory gate) and the
fail-loud gates for unsupported backends and invalid role-local topology.
"""

from __future__ import annotations

import dataclasses

import pytest

import aisimulate.runner as runner
from aisimulate.runner import InvalidRunnerError, _accept_rates_for_expected, _materialize_afd_engine_role
from aisimulate.sweeper.replay import BackendDeploymentSpec, ReplaySpec

pytestmark = pytest.mark.unit


def _deployment(metadata: dict, *, phase: str = "both") -> BackendDeploymentSpec:
    topology = {
        "num_experts": 128,
        "a_tp_size": 1,
        "n_a_nodes": 1,
        "batch_size_per_a_replica": 18,
        "f_moe_ep_size": 8,
        "n_f_nodes": 1,
        "gpus_per_node": 8,
        "num_microbatches": 3,
    }
    return BackendDeploymentSpec(
        deployment_mode="agg",
        backend="sglang",
        backend_version="0.5.14",
        parallel_config={"aggregated": {"topology": topology}},
        agg_engine_args={"afd": topology},
        num_workers=1,
        performance_model_metadata={"aggregated": metadata},
    )


def _spec(deployment: BackendDeploymentSpec, workload: dict | None = None) -> ReplaySpec:
    return ReplaySpec(
        backend_deployment=deployment,
        workload=workload if workload is not None else {"isl": 1024, "osl": 128},
        goal={},
    )


def _metadata(**overrides):
    metadata = {
        "provider": "afd",
        "measurement_required": False,
        "model": "test/model",
        "hardware": "h200_sxm",
    }
    metadata.update(overrides)
    return metadata


@pytest.mark.parametrize("value", [None, 0, True, "1", -1])
def test_invalid_attention_tp_rejects_before_loading_model(monkeypatch, value):
    deployment = _deployment(_metadata())
    if value is None:
        deployment.agg_engine_args["afd"].pop("a_tp_size")
    else:
        deployment.agg_engine_args["afd"]["a_tp_size"] = value
    monkeypatch.setattr(runner, "validate_afd_model", lambda _: pytest.fail("model metadata loaded"))
    with pytest.raises(ValueError, match="a_tp_size.*positive integer"):
        _materialize_afd_engine_role(deployment, _spec(deployment), "aggregated")


@pytest.mark.parametrize("provider", ["afd", "aic-afd"])
def test_saved_afd_metadata_materializes_with_the_canonical_provider(provider):
    spec = _spec(_deployment(_metadata(provider=provider)))
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec, "aggregated")
    assert engine["rank"]["timing_model"]["provider"] == "afd"
    assert spec.backend_deployment.performance_model_metadata["aggregated"]["provider"] == provider


@pytest.fixture(autouse=True)
def _fake_model_config(monkeypatch):
    monkeypatch.setattr(
        runner,
        "validate_afd_model",
        lambda model: {
            "hidden_size": 5120,
            "layers": 64,
            "num_experts": 128,
            "topk": 8,
            "num_shared_experts": 0,
            "num_moe_layers": 64,
        },
    )


def test_mtp_wires_depth_and_accept_rates_into_the_rank_config():
    spec = _spec(_deployment(_metadata(nextn=2, nextn_accepted=1.5)))
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"]

    assert engine["aic_nextn"] == 2
    assert engine["aic_nextn_accept_rates"] == _accept_rates_for_expected(2, 1.5, role="afd")
    assert "nextn" not in engine["timing_model"]["config"]


@pytest.mark.parametrize("layout", ["four_stage", "two_stage"])
def test_pipeline_layout_passes_through_to_native_configuration(layout):
    deployment = _deployment(_metadata())
    deployment.agg_engine_args["afd"]["pipeline_layout"] = layout
    spec = _spec(deployment)
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"]
    assert "pipeline_layout" not in engine["timing_model"]["config"]
    assert engine["afd"]["pipeline_layout"] == layout


def test_pipeline_layout_defaults_to_four_stage():
    spec = _spec(_deployment(_metadata()))
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"]
    assert engine["afd"]["pipeline_layout"] == "four_stage"


def test_unsupported_topology_rejects_before_model_metadata(monkeypatch):
    deployment = _deployment(_metadata())
    deployment.agg_engine_args["afd"]["f_moe_ep_size"] = 1
    monkeypatch.setattr(runner, "validate_afd_model", lambda model: pytest.fail("model metadata loaded"))
    with pytest.raises(NotImplementedError):
        _materialize_afd_engine_role(deployment, _spec(deployment), deployment.afd_roles[0])


@pytest.mark.parametrize("field", ["f_moe_ep_size", "chunked_prefill_size"])
@pytest.mark.parametrize("value", [0, False])
def test_explicit_invalid_topology_values_do_not_become_defaults(field, value):
    deployment = _deployment(_metadata())
    deployment.agg_engine_args["afd"][field] = value
    spec = _spec(deployment)
    with pytest.raises(ValueError, match=field):
        _materialize_afd_engine_role(deployment, spec, deployment.afd_roles[0])


def test_zero_transfer_factor_reaches_native_validation_unchanged():
    deployment = _deployment(_metadata())
    deployment.agg_engine_args["afd"]["comm_overhead_factor"] = 0.0
    spec = _spec(deployment)
    engine = _materialize_afd_engine_role(deployment, spec, deployment.afd_roles[0])["rank"]
    assert engine["afd"]["comm_overhead_factor"] == 0.0
    assert "comm_overhead_factor" not in engine["timing_model"]["config"]


def test_zero_context_window_is_rejected():
    spec = _spec(_deployment(_metadata(context_length=0)))
    with pytest.raises(ValueError, match="context_length"):
        _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])


def test_zero_model_layers_do_not_fall_back_to_another_metadata_field(monkeypatch):
    monkeypatch.setattr(
        runner,
        "validate_afd_model",
        lambda model: {
            "hidden_size": 5120,
            "layers": 0,
            "num_hidden_layers": 64,
        },
    )
    spec = _spec(_deployment(_metadata()))
    with pytest.raises(ValueError, match="layers"):
        _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])


@pytest.mark.parametrize("field", ["isl", "osl"])
def test_trace_placeholder_defaults_do_not_replace_explicit_zero(monkeypatch, field):
    monkeypatch.setattr(runner, "resolve_model_context_length", lambda model: 40960)
    spec = _spec(_deployment(_metadata()), {"trace_path": "/tmp/trace.jsonl", "isl": None, "osl": None})
    config = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"][
        "timing_model"
    ]["config"]
    assert (config["isl"], config["osl"]) == (1, 2)
    spec.workload[field] = 0
    with pytest.raises(ValueError, match=field):
        _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])


@pytest.mark.parametrize(
    "field,value",
    [
        ("unknown_option", "unused"),
        ("phase", "unused"),
        ("pipeline_fill_mode", "unused"),
        ("boundary_on_attn", True),
        ("boundary_on_attn", False),
    ],
)
def test_materialization_rejects_unknown_topology_fields(field, value, monkeypatch):
    deployment = _deployment(_metadata())
    deployment.agg_engine_args["afd"][field] = value
    monkeypatch.setattr(runner, "validate_afd_model", lambda model: pytest.fail("model metadata loaded"))
    spec = _spec(deployment)
    with pytest.raises(InvalidRunnerError, match=f"unknown AFD topology fields.*{field}"):
        _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])


def test_explicit_chunked_prefill_size_overrides_the_prompt_window_pin():
    topology = {
        "a_tp_size": 1,
        "n_a_nodes": 1,
        "batch_size_per_a_replica": 18,
        "f_moe_ep_size": 8,
        "n_f_nodes": 1,
        "gpus_per_node": 8,
        "num_microbatches": 3,
        "chunked_prefill_size": 256,
    }
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="sglang",
        backend_version="0.5.14",
        parallel_config={"aggregated": {"topology": topology}},
        agg_engine_args={"afd": topology},
        num_workers=1,
        performance_model_metadata={"aggregated": _metadata()},
    )
    spec = _spec(deployment)
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"]

    # The explicit window replaces the prompt pin on BOTH prefill budgets;
    # the absent context declaration is unaffected; Rust derives its runtime cap.
    assert engine["sglang"] == {"chunked_prefill_size": 256, "max_prefill_tokens": 256}
    assert engine["max_model_len"] is None


def _plus_pd_deployment(metadata: dict, *, phase: str = "decode") -> BackendDeploymentSpec:
    topology = {
        "a_tp_size": 1,
        "n_a_nodes": 1,
        "batch_size_per_a_replica": 18,
        "f_moe_ep_size": 8,
        "n_f_nodes": 1,
        "gpus_per_node": 8,
        "num_microbatches": 3,
    }
    companion_role = "prefill" if phase == "decode" else "decode"
    return BackendDeploymentSpec(
        deployment_mode="disagg",
        backend="sglang",
        backend_version="0.5.14",
        parallel_config={f"{phase}_replicas": 1, f"{companion_role}_replicas": 2},
        performance_model_metadata={phase: metadata},
        **{
            f"{phase}_engine_args": {"afd": topology},
            f"num_{phase}_workers": 1,
            f"{companion_role}_engine_args": {
                "timing_model": {"type": "fixed", "prefill_ms": 2.0, "decode_ms": 2.0},
                "num_gpu_blocks": 512,
                "block_size": 16,
            },
            f"num_{companion_role}_workers": 2,
        },
    )


def test_afd_plus_pd_materializes_the_two_role_disaggregated_topology():
    spec = _spec(
        _plus_pd_deployment(_metadata()),
        workload={"isl": 1024, "osl": 128, "concurrency": 2, "request_count": 4},
    )
    execution = runner._materialize_engine_execution_spec(spec, trace_block_size=64, record_per_request=False)

    assert execution["topology"]["kind"] == "disaggregated"
    # decode-phase AFD pool sits in the decode slot with one pool-pair rank;
    # the prefill companion scales by its replica count.
    assert execution["topology"]["decode"] == {"initial_workers": 1, "startup_delay_ms": 0.0}
    assert execution["topology"]["prefill"]["initial_workers"] == 2
    afd_rank = execution["engine"]["decode"]["rank"]
    assert "phase" not in afd_rank["afd"]
    assert afd_rank["timing_model"]["provider"] == "afd"
    prefill_rank = execution["engine"]["prefill"]["rank"]
    assert "afd" not in prefill_rank


@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
def test_nested_afd_rank_is_rejected_before_model_loading(monkeypatch, role):
    deployment = _deployment(_metadata()) if role == "aggregated" else _plus_pd_deployment(_metadata(), phase=role)
    field = "agg_engine_args" if role == "aggregated" else f"{role}_engine_args"
    raw = getattr(deployment, field)
    raw["afd"]["a_tp_size"] = 0
    nested = {
        "rank": {
            **raw,
            "num_gpu_blocks": 16,
            "block_size": 1,
            "timing_model": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
        }
    }
    deployment = dataclasses.replace(deployment, **{field: nested})
    monkeypatch.setattr(runner, "validate_afd_model", lambda _: pytest.fail("model metadata loaded"))
    with pytest.raises(InvalidRunnerError, match=f"AFD {role} requires flat role arguments"):
        runner._materialize_engine_execution_spec(_spec(deployment), trace_block_size=64, record_per_request=False)


@pytest.mark.parametrize("afd_role", ["prefill", "decode"])
def test_nested_monolithic_companion_remains_supported(afd_role):
    deployment = _plus_pd_deployment(_metadata(), phase=afd_role)
    companion = "decode" if afd_role == "prefill" else "prefill"
    field = f"{companion}_engine_args"
    deployment = dataclasses.replace(deployment, **{field: {"rank": getattr(deployment, field)}})
    execution = runner._materialize_engine_execution_spec(
        _spec(deployment, workload={"isl": 1024, "osl": 128, "concurrency": 2, "request_count": 4}),
        trace_block_size=64,
        record_per_request=False,
    )
    rank = execution["engine"][companion]["rank"]
    assert "afd" not in rank
    assert rank["timing_model"] == {"type": "fixed", "prefill_ms": 2.0, "decode_ms": 2.0}
    assert execution["engine"][afd_role]["rank"]["timing_model"]["provider"] == "afd"


def test_non_sglang_backend_is_rejected_loudly():
    deployment = dataclasses.replace(_deployment(_metadata()), backend="trtllm")
    spec = _spec(deployment)
    with pytest.raises(InvalidRunnerError, match="backend='sglang'"):
        _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])


def test_mtp_without_nextn_accepted_fails_loudly():
    spec = _spec(_deployment(_metadata(nextn=2)))
    with pytest.raises(InvalidRunnerError, match="nextn_accepted"):
        _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])


def test_mtp_depth_above_five_is_rejected():
    spec = _spec(_deployment(_metadata(nextn=6, nextn_accepted=3.0)))
    with pytest.raises(InvalidRunnerError, match="1..=5"):
        _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])


def test_no_mtp_leaves_the_rank_config_without_speculation_keys():
    spec = _spec(_deployment(_metadata(nextn=0)))
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"]

    assert "aic_nextn" not in engine
    assert "aic_nextn_accept_rates" not in engine
    assert "nextn" not in engine["timing_model"]["config"]


def test_scheduler_derivation_for_synthetic_workload():
    # max_num_seqs mirrors the pool's in-flight capacity; num_gpu_blocks is
    # derived natively from the memory gate (physical capacity, not a
    # sentinel); the prefill knobs pin one prompt per pass (the timing identity
    # charges one frozen prefill step).
    spec = _spec(_deployment(_metadata()))
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"]

    assert engine["max_num_seqs"] == 144
    assert "num_gpu_blocks" not in engine
    assert engine["block_size"] == 1
    assert engine["max_model_len"] is None
    assert engine["sglang"] == {"chunked_prefill_size": 1024, "max_prefill_tokens": 1024}
    assert "engine_core" not in engine


def test_scheduler_derivation_uses_the_context_window_when_present():
    spec = _spec(_deployment(_metadata(context_length=8192)))
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"]

    assert engine["max_num_seqs"] == 144
    assert "num_gpu_blocks" not in engine
    assert engine["block_size"] == 1
    assert engine["max_model_len"] == 8192
    assert engine["sglang"] == {"chunked_prefill_size": 8192, "max_prefill_tokens": 8192}


def test_context_length_below_isl_plus_osl_is_rejected_loudly():
    # The derivation pins the prompt window AND the KV window at
    # context_length, so a window that cannot hold the declared workload would
    # silently chunk the prompt (and undersize the derived KV capacity).
    spec = _spec(_deployment(_metadata(context_length=512)))
    with pytest.raises(InvalidRunnerError, match="cannot hold isl\\+osl"):
        _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])


def test_variable_length_traffic_is_exempt_from_the_window_lower_bound():
    # Sampled lengths are upper bounds, not per-request lengths, so the
    # window check must not fire; SglangCore's max_model_len gate handles a
    # genuinely over-long single prompt instead.
    spec = _spec(
        _deployment(_metadata(context_length=512)),
        workload={"isl": 1024, "osl": 128, "random_range_ratio": 0.5},
    )
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"]

    assert engine["max_model_len"] == 512
    assert "num_gpu_blocks" not in engine


def test_afd_capability_surface_uses_existing_deployment_topologies():
    capabilities = runner.EngineReplayRunnerFactory().capabilities()
    assert capabilities.supports_backend_topology("sglang", "agg")
    assert capabilities.supports_backend_topology("sglang", "disagg")
    assert not any(mode.startswith("afd") for _, mode in capabilities.supported_backend_topologies)


def test_trace_workload_without_context_length_reserves_the_model_window(monkeypatch):
    # The memory gate must not size KV from the placeholder isl/osl (1/2) of
    # variable-length traffic: reserve the model's full context, matching the
    # compiler path's "max" resolution.
    monkeypatch.setattr(runner, "resolve_model_context_length", lambda model: 40960)
    spec = _spec(_deployment(_metadata()), workload={"trace_path": "/tmp/trace.jsonl"})
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"]

    assert engine["max_model_len"] == 40960
    assert "num_gpu_blocks" not in engine
    assert engine["block_size"] == 1
    assert engine["max_model_len"] == 40960


def test_explicit_context_length_wins_for_trace_workloads(monkeypatch):
    monkeypatch.setattr(runner, "resolve_model_context_length", lambda model: 40960)
    spec = _spec(_deployment(_metadata(context_length=8192)), workload={"trace_path": "/tmp/trace.jsonl"})
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"]

    assert engine["max_model_len"] == 8192


def test_synthetic_workload_keeps_the_isl_osl_window(monkeypatch):
    def no_resolve(model):
        raise AssertionError("resolve_model_context_length must not be called for synthetic traffic")

    monkeypatch.setattr(runner, "resolve_model_context_length", no_resolve)
    spec = _spec(_deployment(_metadata()))
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"]

    assert "max_seq_len" not in engine["timing_model"]["config"]
    assert engine["max_model_len"] is None


def test_fixed_synthetic_traffic_keeps_the_frozen_decode_identity():
    spec = _spec(_deployment(_metadata()), workload={"isl": 1024, "osl": 128, "random_range_ratio": 1.0})
    config = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"][
        "timing_model"
    ]["config"]

    assert config["frozen_decode_identity"] is True


@pytest.mark.parametrize("explicit", [True, False])
def test_materializer_uses_serialized_pricing_control_and_ignores_environment(monkeypatch, explicit):
    spec = _spec(
        _deployment(
            _metadata(
                estimator={
                    "estimator_config": {"afd": {"frozen_decode_identity": explicit}},
                }
            )
        )
    )
    configs = []
    for value in ("0", "1", "invalid"):
        monkeypatch.setenv("AISIMULATE_AFD_FROZEN_DECODE_IDENTITY", value)
        config = _materialize_afd_engine_role(spec.backend_deployment, spec, "aggregated")["rank"]["timing_model"][
            "config"
        ]
        assert config["frozen_decode_identity"] is explicit
        configs.append(config)
    assert configs[0] == configs[1] == configs[2]


def test_variable_length_synthetic_traffic_drops_the_frozen_decode_identity(monkeypatch):
    # random_range_ratio < 1.0 samples per-request lengths around the declared
    # isl/osl, so pricing every decode at the frozen identity would silently
    # reuse a measurement taken at a different sequence length. The declared
    # isl/osl are still real upper bounds, so the KV window must NOT fall back
    # to the whole model context the way placeholder-length traffic does.
    def no_resolve(model):
        raise AssertionError("declared isl/osl are real bounds; the model window must not be reserved")

    monkeypatch.setattr(runner, "resolve_model_context_length", no_resolve)
    spec = _spec(_deployment(_metadata()), workload={"isl": 1024, "osl": 128, "random_range_ratio": 0.5})
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"]

    assert engine["timing_model"]["config"]["frozen_decode_identity"] is False
    assert "max_seq_len" not in engine["timing_model"]["config"]
    assert "num_gpu_blocks" not in engine
    assert engine["block_size"] == 1
    assert engine["max_model_len"] is None


def test_placeholder_length_traffic_drops_the_frozen_decode_identity(monkeypatch):
    monkeypatch.setattr(runner, "resolve_model_context_length", lambda model: 40960)
    spec = _spec(_deployment(_metadata()), workload={"trace_path": "/tmp/trace.jsonl"})
    config = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])["rank"][
        "timing_model"
    ]["config"]

    assert config["frozen_decode_identity"] is False


def test_routing_defaults_and_explicit_a_reach_rank_and_timing():
    for explicit in [False, True]:
        deployment = _deployment(_metadata())
        if explicit:
            deployment.agg_engine_args["afd"]["routing_location"] = "a"
        spec = _spec(deployment)
        engine = _materialize_afd_engine_role(spec.backend_deployment, spec, spec.backend_deployment.afd_roles[0])[
            "rank"
        ]
        assert engine["afd"]["routing_location"] == "a"
        assert "routing_location" not in engine["timing_model"]["config"]


def test_f_routing_replay_fails_before_model_metadata_resolution(monkeypatch):
    deployment = _deployment(_metadata())
    deployment.agg_engine_args["afd"]["routing_location"] = "f"
    monkeypatch.setattr(runner, "validate_afd_model", lambda *args: pytest.fail("unexpected model lookup"))
    with pytest.raises(NotImplementedError, match="routing_location='f'"):
        _materialize_afd_engine_role(deployment, _spec(deployment), deployment.afd_roles[0])


@pytest.mark.parametrize("provider", ["afd", "aic-afd"])
@pytest.mark.parametrize(
    ("rank_location", "timing_location", "exception", "message"),
    [
        ("f", "f", NotImplementedError, "routing_location='f' is not implemented"),
        ("a", "f", RuntimeError, "routing_location conflicts"),
        ("f", "a", RuntimeError, "routing_location conflicts"),
    ],
)
def test_direct_native_replay_checks_routing_before_data_loading(
    provider, rank_location, timing_location, exception, message
):
    import json

    from aisimulate import _runtime

    spec = _spec(_deployment(_metadata()), workload={"isl": 1024, "osl": 128, "concurrency": 2, "request_count": 4})
    execution = runner._materialize_engine_execution_spec(spec, trace_block_size=64, record_per_request=False)
    rank = execution["engine"]["rank"]
    rank["timing_model"]["provider"] = provider
    rank["afd"]["routing_location"] = rank_location
    timing = rank["timing_model"]["config"]
    timing["routing_location"] = timing_location
    timing["model"] = "missing/model"
    rank["afd"]["a_pool"]["system"] = "missing_system"
    with pytest.raises(exception, match=message):
        _runtime.run_replay_json(json.dumps(execution))


def test_prediction_source_policies_survive_saved_config_and_materialization():
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config import CorePredictionConfig

    cfg = CorePredictionConfig.model_validate(
        {
            "traffic": {
                "source": {"type": "synthetic", "input_tokens": 128, "output_tokens": 8},
                "load": {"type": "concurrency", "concurrency": 1},
                "stop": {"requests": 1},
            },
            "engine": {
                "model": "Qwen/Qwen3-30B-A3B",
                "hardware": "h200_sxm",
                "backend": "sglang",
                "backend_version": "0.5.14",
                "context_length": 256,
                "enable_shared_layer": False,
                "strict_provenance": True,
                "workers": {
                    "aggregated": {
                        "type": "afd",
                        "afd": {"n_a_nodes": 1, "n_f_nodes": 1, "tp_a": 1, "a_batch_size": 18},
                    }
                },
            },
        }
    )
    for config in [cfg, CorePredictionConfig.model_validate_json(cfg.model_dump_json())]:
        spec = prediction_to_replay_spec(config)
        rank = _materialize_afd_engine_role(spec.backend_deployment, spec, "aggregated")["rank"]
        assert rank["timing_model"]["config"]["estimator"] == {
            "enable_shared_layer": False,
            "strict_provenance": True,
        }


def test_materialization_keeps_independent_f_width_and_shared_version():
    deployment = _deployment(_metadata())
    deployment.agg_engine_args["afd"].update(f_gpus_per_node=4, f_moe_ep_size=4)
    spec = _spec(deployment)
    rank = _materialize_afd_engine_role(deployment, spec, "aggregated")["rank"]
    assert rank["afd"]["gpus_per_node"] == 8
    assert rank["afd"]["f_gpus_per_node"] == 4
    assert rank["afd"]["f_pool"]["f_moe_ep_size"] == 4
    assert rank["timing_model"]["config"]["backend_version"] == deployment.backend_version
