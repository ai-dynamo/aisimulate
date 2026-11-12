# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Role composition, search, saved replay and native P/D execution contracts."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.recommend import _candidate_prediction, recommendation_to_sweeper
from aisimulate.runner import EngineReplayRunnerFactory, InvalidRunnerError, _materialize_engine_execution_spec
from aisimulate.sweeper.afd_parallel import AFDInfeasible
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.replay import ReplayOutputRequirements, ReplaySpec, RunnerCapabilities
from aisimulate.sweeper.sample import unroll_sample
from aisimulate.sweeper.sampler import RandomBranchSampler
from aisimulate.sweeper.search_space import RunnerIncompatibleError, enumerate_branches

COMBINATIONS = [
    ("aggregated", ("aggregated",)),
    ("disaggregated", ("prefill",)),
    ("disaggregated", ("decode",)),
    ("disaggregated", ("prefill", "decode")),
]


def prediction(mode="disaggregated", afd_roles=("prefill", "decode")):
    roles = ("aggregated",) if mode == "aggregated" else ("prefill", "decode")
    return {
        "engine": {
            "mode": mode,
            "model": "Qwen/Qwen3-30B-A3B",
            "hardware": "h200_sxm",
            "backend": "sglang",
            "backend_version": "0.5.14",
            "context_length": 2048,
            "workers": {
                role: {
                    "type": "afd",
                    "afd": {
                        "n_a_nodes": 1,
                        "n_f_nodes": 1,
                        "tp_a": 1,
                        "a_batch_size": 12 if role == "prefill" else 18,
                        "num_microbatches": 3,
                    },
                }
                if role in afd_roles
                else {"kv_cache": {"capacity": {"type": "fixed", "blocks": 8192}}}
                for role in roles
            },
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 16, "output_tokens": 4},
            "load": {"type": "concurrency", "concurrency": 2},
            "stop": {"requests": 4},
        },
    }


def recommendation(mode="disaggregated", afd_roles=("prefill", "decode")):
    raw = prediction(mode, afd_roles)
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 32}}
    for worker in raw["engine"]["workers"].values():
        if worker.get("type") == "afd":
            del worker["afd"]["n_a_nodes"], worker["afd"]["n_f_nodes"]
        else:
            worker["parallelism"] = {
                "preset": [
                    {
                        "replicas": 1,
                        "tensor": 1,
                        "pipeline": 1,
                        "attention_data": 1,
                        "moe_tensor": 1,
                        "moe_expert": 1,
                    }
                ]
            }
    return raw


@pytest.mark.parametrize(("mode", "afd_roles"), COMBINATIONS)
def test_saved_afd_search_does_not_invent_ordinary_estimator_overrides(mode, afd_roles):
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(recommendation(mode, afd_roles)))
    restored = type(smart).model_validate(smart.model_dump(mode="json"))
    assert restored.search_space.afd_roles == smart.search_space.afd_roles
    assert restored.search_space.role_estimator_controls == smart.search_space.role_estimator_controls


@pytest.mark.parametrize(("mode", "afd_roles"), COMBINATIONS)
def test_explicit_legacy_fpm_selection_on_afd_is_rejected_instead_of_overwritten(mode, afd_roles):
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(recommendation(mode, afd_roles)))
    payload = smart.model_dump(mode="json")
    role = "agg" if afd_roles[0] == "aggregated" else afd_roles[0]
    payload["search_space"][f"{role}_forward_model"] = "fpm"
    with pytest.raises(ValueError, match="role_estimator_controls require regular language workers"):
        type(smart).model_validate(payload)


@pytest.mark.parametrize("mode,afd_roles", COMBINATIONS)
@pytest.mark.parametrize("field,value", [("enable_eplb", True), ("wideep_num_slots", 8), ("moe_backend", "custom")])
def test_public_and_sdk_afd_search_reject_unsupported_controls(mode, afd_roles, field, value):
    raw = recommendation(mode, afd_roles)
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))
    raw["engine"][field] = value
    with pytest.raises(ValueError, match=f"{field}.*unsupported for AFD"):
        CoreRecommendationConfig.model_validate(raw)
    payload = smart.model_dump(mode="python")
    payload["search_space"][field] = value
    with pytest.raises(ValueError, match=f"{field}.*unsupported for AFD"):
        type(smart).model_validate(payload)


def test_afd_search_preserves_supported_controls_through_saved_prediction():
    raw = recommendation("aggregated", ("aggregated",))
    quant = dict.fromkeys(("gemm_quant_mode", "moe_quant_mode", "fmha_quant_mode", "kvcache_quant_mode"), "bfloat16")
    quant["comm_quant_mode"] = "half"
    selectors = {"attention_backend": "test_attention", "moe_kernel_source": "test_source"}
    raw["engine"].update(quant, **selectors, nextn=2, nextn_accepted=1.5, moe_backend="default", enable_eplb=False)
    source = CoreRecommendationConfig.model_validate(raw)
    smart = recommendation_to_sweeper(source)
    smart = type(smart).model_validate_json(smart.model_dump_json())
    (branch,) = enumerate_branches(smart, max_seq_len=2048)
    suggestion = RandomBranchSampler(branch, seed=7).suggest(1)[0]
    sample = unroll_sample(
        search_space=smart.search_space, selection=suggestion.selection, parallel_config=suggestion.parallel_config
    )
    deployment = build_backend_deployment(sample, backend_version="0.5.14")
    meta = deployment.performance_model_metadata["aggregated"]
    assert meta["quant"] == quant
    assert {name: meta[name] for name in selectors} == selectors
    assert (meta["nextn"], meta["nextn_accepted"]) == (2, 1.5)
    spec = ReplaySpec(backend_deployment=deployment, workload=smart.workload.model_dump(), goal={})
    saved = _candidate_prediction(source, sample, spec, adapter_sections={})
    reloaded = CorePredictionConfig.model_validate_json(CorePredictionConfig.model_validate(saved).model_dump_json())
    recompiled = prediction_to_replay_spec(reloaded)
    restored_meta = recompiled.backend_deployment.performance_model_metadata["aggregated"]
    for name in ("quant", *selectors, "nextn", "nextn_accepted"):
        assert restored_meta[name] == meta[name]


@pytest.mark.parametrize(
    "field,value", [("vision", {}), ("host_profile", {"path": "/tmp/unused.json", "frontend": "python"})]
)
def test_afd_companion_rejects_unsupported_host_fields(field, value):
    raw = prediction("disaggregated", ("decode",))
    raw["engine"]["workers"]["prefill"][field] = value
    with pytest.raises(ValueError, match=f"AFD companion \\(prefill\\).*{field}"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))


def test_afd_companion_checks_frontend_measurement():
    raw = prediction("disaggregated", ("decode",))
    raw["engine"]["workers"]["prefill"]["frontend"] = {
        "stages": [{"service_ms": 1.0}],
        "measured_for": {
            "model": raw["engine"]["model"],
            "frontend": "python",
            "feature_transport": "shm",
            "height": 224,
            "width": 224,
        },
    }
    with pytest.raises(ValueError, match="AFD companion \\(prefill\\) frontend:.*no images"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))


@pytest.mark.parametrize("frontend", [False, True])
def test_afd_companion_preserves_and_executes_host_loop_and_frontend(frontend):
    raw = prediction("disaggregated", ("decode",))
    raw["engine"]["workers"]["prefill"]["host_loop"] = True
    if frontend:
        raw["engine"]["workers"]["prefill"]["frontend"] = {"stages": [{"service_ms": 5.0}]}
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    args = spec.backend_deployment.prefill_engine_args
    assert args["sglang"]["host_loop"] is True
    if frontend:
        assert args["frontend"]["stages"][0]["service_ms"] == 5.0
    report = EngineReplayRunnerFactory().create(0).run(spec)
    assert report.metrics["completed_requests"] == 4
    if frontend:
        assert report.metrics["mean_ttft_ms"] >= 5.0


@pytest.mark.parametrize("shadow_root", [False, True])
def test_native_afd_memory_and_kv_budget_use_custom_pool_specs(tmp_path, shadow_root):
    import aisimulate_core

    packaged = Path(aisimulate_core.__file__).parent / "systems"
    system = yaml.safe_load((packaged / "h200_sxm.yaml").read_text())
    system["data_dir"] = str(packaged / "data/h200_sxm")
    for name, capacity in (("custom_a", 64), ("custom_f", 32)):
        system["gpu"]["mem_capacity"] = capacity * (1 << 30)
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(system))
    raw = prediction("aggregated", ("aggregated",))
    raw["engine"]["hardware"] = "custom_a"
    raw["engine"]["systems_paths"] = [str(tmp_path)]
    if shadow_root:
        # The first root knows the SKU but lacks timing data. The canonical
        # model selects the second root; capacity must use that same spec.
        shadow = tmp_path / "shadow"
        shadow.mkdir()
        system["gpu"]["mem_capacity"] = 8 * (1 << 30)
        system["data_dir"] = str(shadow / "missing-data")
        for name in ("custom_a", "custom_f"):
            (shadow / f"{name}.yaml").write_text(yaml.safe_dump(system))
        raw["engine"]["systems_paths"].insert(0, str(shadow))
    raw["engine"]["workers"]["aggregated"]["afd"]["f_system"] = "custom_f"
    report = (
        EngineReplayRunnerFactory()
        .create(0)
        .run(
            prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)),
            output_requirements=ReplayOutputRequirements(capture_memory_diagnostics=True),
        )
    )
    assert report.metrics["completed_requests"] == 4
    memory = report.metadata["native_report"]["memory_diagnostics"]["aggregated"]
    for phase in ("prefill", "decode"):
        assert memory["pools"][phase]["attention"]["capacity_gib_per_gpu"] == 64
        assert memory["pools"][phase]["ffn"]["capacity_gib_per_gpu"] == 32
    assert memory["kv_budget_bytes_per_a_gpu"] < 64 * (1 << 30)


@pytest.mark.parametrize("mode,afd_roles", COMBINATIONS)
def test_four_combinations_compile_into_role_local_descriptors(mode, afd_roles):
    source = CorePredictionConfig.model_validate(prediction(mode, afd_roles))
    restored = CorePredictionConfig.model_validate_json(source.model_dump_json())
    spec = prediction_to_replay_spec(restored)
    deployment = spec.backend_deployment
    assert deployment.afd_roles == afd_roles
    assert deployment.deployment_mode == ("agg" if mode == "aggregated" else "disagg")
    wire = _materialize_engine_execution_spec(spec, trace_block_size=64, record_per_request=True)["spec"]
    for role in afd_roles:
        descriptor = wire["engine"] if mode == "aggregated" else wire["engine"][role]
        assert "phase" not in descriptor["rank"]["afd"]
        assert descriptor["rank"]["timing_model"]["provider"] == "afd"
        assert descriptor["rank"]["max_num_seqs"] == 8 * (12 if role == "prefill" else 18)
        assert deployment.parallel_config[role]["schema_version"] == 2
    assert spec.api_version == 1


@pytest.mark.parametrize("budget", [16, 24, 32])
def test_recommendation_fixed_ep_preserves_feasible_topologies_with_larger_budget(budget):
    raw = recommendation("aggregated", ("aggregated",))
    raw["optimization"]["constraints"]["max_candidate_gpus"] = budget
    raw["engine"]["workers"]["aggregated"]["afd"].update(f_moe_ep_size=8, num_microbatches=3)
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))
    (branch,) = enumerate_branches(smart, max_seq_len=2048)
    assert [(t.n_a_nodes, t.n_f_nodes, t.f_moe_ep_size) for t in branch.parallel_configs] == [
        (a, 1, 8) for a in range(1, budget // 8)
    ]


@pytest.mark.parametrize("mode,afd_roles", COMBINATIONS)
def test_search_sampler_saved_prediction_and_replay_agree(mode, afd_roles):
    source = CoreRecommendationConfig.model_validate(recommendation(mode, afd_roles))
    smart = recommendation_to_sweeper(source)
    (branch,) = enumerate_branches(smart, max_seq_len=2048)
    assert all(c.total_gpus <= 32 for c in branch.parallel_configs)
    # A pinned ordinary role stays pinned; it must not expand into another shape/replica count.
    for candidate in branch.parallel_configs:
        if mode == "disaggregated":
            for role in {"prefill", "decode"} - set(afd_roles):
                assert getattr(candidate, role).total_gpus == 1
    suggestion = RandomBranchSampler(branch, seed=7).suggest(1)[0]
    sample = unroll_sample(
        search_space=smart.search_space, selection=suggestion.selection, parallel_config=suggestion.parallel_config
    )
    deployment = build_backend_deployment(sample, backend_version="0.5.14")
    spec = ReplaySpec(backend_deployment=deployment, workload=smart.workload.model_dump(), goal={})
    saved = _candidate_prediction(source, sample, spec, adapter_sections={})
    recompiled = prediction_to_replay_spec(CorePredictionConfig.model_validate(saved))
    assert recompiled.backend_deployment.afd_roles == afd_roles
    for role in afd_roles:
        assert recompiled.backend_deployment.parallel_config[role] == deployment.parallel_config[role]
    assert sample["used_gpus"] == suggestion.parallel_config.total_gpus


@pytest.mark.parametrize("afd_role,ordinary_role", [("prefill", "decode"), ("decode", "prefill")])
@pytest.mark.parametrize("scale", ["linear", "log"])
def test_mixed_afd_memory_fraction_range_reaches_samples_and_saved_prediction(afd_role, ordinary_role, scale):
    raw = recommendation(afd_roles=(afd_role,))
    raw["engine"]["workers"][ordinary_role]["kv_cache"]["capacity"] = {
        "type": "default",
        "memory_fraction": {"range": {"min": 0.7, "max": 0.9, "scale": scale}},
    }
    source = CoreRecommendationConfig.model_validate(raw)
    smart = recommendation_to_sweeper(source)
    name = f"{ordinary_role}_gpu_memory_utilization"
    # Other modes and AFD roles must not contribute ordinary-worker ranges.
    smart.search_space.engine_float_ranges.update(
        {"agg_gpu_memory_utilization": [0.5, 0.6], f"{afd_role}_gpu_memory_utilization": [0.5, 0.6]}
    )
    (branch,) = enumerate_branches(smart, max_seq_len=2048)

    assert branch.float_ranges == {name: (0.7, 0.9)}
    assert name not in branch.knob_choices
    assert branch.log_float_ranges == (frozenset({name}) if scale == "log" else frozenset())
    suggestions = RandomBranchSampler(branch, seed=7).suggest(5)
    samples = [
        unroll_sample(search_space=smart.search_space, selection=s.selection, parallel_config=s.parallel_config)
        for s in suggestions
    ]
    values = [sample[name] for sample in samples]
    assert all(0.7 < value < 0.9 for value in values)
    assert len(set(values)) > 1

    sample = samples[0]
    deployment = build_backend_deployment(sample, backend_version="0.5.14")
    spec = ReplaySpec(backend_deployment=deployment, workload=smart.workload.model_dump(), goal={})
    saved = _candidate_prediction(source, sample, spec, adapter_sections={})
    restored = CorePredictionConfig.model_validate_json(CorePredictionConfig.model_validate(saved).model_dump_json())
    worker = getattr(restored.engine.workers, ordinary_role)
    assert worker.kv_cache.capacity.memory_fraction == sample[name]


@pytest.mark.parametrize("mode,afd_roles", COMBINATIONS)
def test_native_replay_completes_exact_work_and_accounts_for_all_pools(mode, afd_roles):
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(prediction(mode, afd_roles)))
    report = (
        EngineReplayRunnerFactory(determinism="canonical_v1")
        .create(0)
        .run(spec, output_requirements=ReplayOutputRequirements(capture_per_request=True, include_raw_report=True))
    )
    assert report.metrics["completed_requests"] == 4
    assert report.metrics["total_input_tokens"] == 64
    assert report.metrics["total_output_tokens"] == 16
    records = report.metadata["native_report"]["per_request"]
    assert len(records) == 4
    assert all(
        r["terminal_status"] == "completed" and r["output_length"] == r["requested_output_length"] for r in records
    )
    if mode == "disaggregated":
        assert all(r["admission_count"] == 2 for r in records)
        for record in records:
            assert record["source_released_ms"] is not None
            assert record["decode_admit_ms"] >= record["destination_activated_ms"] >= record["source_held_ms"]
        assert report.metrics["prefill_gpus_per_worker"] == (16 if "prefill" in afd_roles else 1)
        assert report.metrics["decode_gpus_per_worker"] == (16 if "decode" in afd_roles else 1)


@pytest.mark.parametrize("timing_mode", ["full_prompt", "destination_missing"])
def test_double_afd_kv_transfer_is_applied_once_and_deterministic(timing_mode):
    raw = prediction()
    raw["traffic"]["load"]["concurrency"] = 1
    baseline = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    raw["engine"]["kv_transfer"] = {
        "bytes_per_token": 1000,
        "bandwidth_gb_per_second": 0.001,
        "timing_mode": timing_mode,
    }
    delayed = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    factory = EngineReplayRunnerFactory(determinism="canonical_v1")
    reports = [factory.create(0).run(spec) for spec in (baseline, delayed, delayed)]
    assert reports[1].metrics["mean_ttft_ms"] - reports[0].metrics["mean_ttft_ms"] == pytest.approx(16)
    assert reports[1].metrics["total_output_tokens"] == 16
    assert {
        k: v
        for k, v in reports[1].metrics.items()
        if k not in {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}
    } == {
        k: v
        for k, v in reports[2].metrics.items()
        if k not in {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}
    }


@pytest.mark.parametrize("explicit_ep", [None, 8])
def test_auto_kv_transfer_uses_per_request_geometry_for_both_roles(explicit_ep):
    raw = prediction()
    raw["engine"]["kv_transfer"] = {"bytes_per_token": "auto", "bandwidth_gb_per_second": 50}
    if explicit_ep is not None:
        raw["engine"]["workers"]["prefill"]["afd"]["f_moe_ep_size"] = explicit_ep
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    p, d = spec.backend_deployment.prefill_engine_args, spec.backend_deployment.decode_engine_args
    assert p["kv_transfer_bytes_per_token"] == d["kv_transfer_bytes_per_token"] > 0
    changed = deepcopy(raw)
    changed["engine"]["workers"]["prefill"]["afd"]["a_batch_size"] = 64
    larger = prediction_to_replay_spec(CorePredictionConfig.model_validate(changed))
    assert (
        larger.backend_deployment.prefill_engine_args["kv_transfer_bytes_per_token"] == p["kv_transfer_bytes_per_token"]
    )


@pytest.mark.parametrize("key,value", [("phase", "both"), ("combined_with_pd", True), ("max_candidates", 10)])
def test_removed_afd_fields_are_rejected(key, value):
    raw = prediction()
    raw["engine"]["workers"]["decode"]["afd"][key] = value
    with pytest.raises(ValidationError, match=key):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("key", ["parallelism", "scheduler", "kv_cache", "timing"])
def test_afd_worker_accepts_empty_common_parameter_blocks(key):
    raw = prediction()
    raw["engine"]["workers"]["decode"][key] = {}
    CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("missing", ["prefill", "decode"])
def test_pd_requires_both_roles(missing):
    raw = prediction()
    del raw["engine"]["workers"][missing]
    with pytest.raises(ValidationError, match="requires prefill and decode"):
        CorePredictionConfig.model_validate(raw)


def test_search_fails_when_budget_cannot_fit_two_afd_pairs():
    raw = recommendation()
    raw["optimization"]["constraints"]["max_candidate_gpus"] = 24
    with pytest.raises(ValueError, match="no viable AFD role composition"):
        enumerate_branches(recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw)))


@pytest.mark.parametrize("modes", [["aggregated", "disaggregated"], ["disaggregated", "aggregated"]])
@pytest.mark.parametrize("budget,ep", [(8, None), (16, None), (16, 3)])
def test_infeasible_afd_mode_keeps_viable_ordinary_candidates(modes, budget, ep):
    raw = recommendation()
    raw["engine"]["mode"] = {"choices": modes}
    raw["engine"]["workers"]["aggregated"] = {}
    raw["optimization"]["constraints"]["max_candidate_gpus"] = budget
    if ep is not None:
        raw["engine"]["workers"]["decode"]["afd"]["f_moe_ep_size"] = ep
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))
    ordinary = smart.model_copy(
        update={"search_space": smart.search_space.model_copy(update={"deployment_mode": ["agg"]})}
    )
    expected = enumerate_branches(ordinary, max_seq_len=2048)

    with pytest.warns(UserWarning, match="disagg.*skipped"):
        branches = enumerate_branches(smart, max_seq_len=2048)

    assert branches == expected


def test_all_afd_modes_infeasible_still_raise():
    raw = recommendation()
    raw["engine"]["mode"] = {"choices": ["aggregated", "disaggregated"]}
    raw["engine"]["workers"]["aggregated"] = deepcopy(raw["engine"]["workers"]["decode"])
    raw["optimization"]["constraints"]["max_candidate_gpus"] = 8
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))

    with pytest.raises(ValueError, match="no deployment_mode has a viable parallel config"):
        enumerate_branches(smart)


@pytest.mark.parametrize("failure", ["invalid_tp", "candidate_limit", "pinned_companion"])
def test_afd_configuration_errors_remain_fatal_with_a_viable_ordinary_mode(failure):
    raw = recommendation(afd_roles=("decode",) if failure == "pinned_companion" else ("prefill", "decode"))
    raw["engine"]["mode"] = {"choices": ["aggregated", "disaggregated"]}
    raw["engine"]["workers"]["aggregated"] = {}
    if failure == "invalid_tp":
        raw["engine"]["workers"]["decode"]["afd"]["tp_a"] = 3
        expected = "must divide gpus_per_node"
    elif failure == "candidate_limit":
        raw["optimization"]["max_topology_candidates"] = 1
        expected = "exceeding"
    else:
        raw["optimization"]["constraints"]["max_candidate_gpus"] = 16
        expected = "no viable AFD role composition"
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))

    with pytest.raises(ValueError, match=expected):
        enumerate_branches(smart, max_seq_len=2048)


def test_complete_domain_limit_never_truncates():
    raw = recommendation("aggregated", ("aggregated",))
    raw["optimization"]["max_topology_candidates"] = 1
    with pytest.raises(AFDInfeasible, match="exceeding"):
        enumerate_branches(recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw)))


def test_runner_capability_gate_runs_before_model_lookup(monkeypatch):
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(recommendation()))
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.resolve_model_hardware", lambda *a, **k: pytest.fail("hardware lookup")
    )
    with pytest.raises(RunnerIncompatibleError):
        enumerate_branches(
            smart, runner_capabilities=RunnerCapabilities(supported_backend_topologies=(("vllm", "disagg"),))
        )


@pytest.mark.parametrize("field", ["enable_eplb", "wideep_num_slots", "moe_backend"])
def test_afd_rejects_unsupported_pool_shaping_controls(field):
    raw = prediction()
    raw["engine"][field] = {"enable_eplb": True, "wideep_num_slots": 128, "moe_backend": "deepep_moe"}[field]
    with pytest.raises(ValidationError, match="unsupported for AFD"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("afd_role,ordinary_role", [("prefill", "decode"), ("decode", "prefill")])
@pytest.mark.parametrize("timing", [{"type": "fixed", "prefill_ms": 1, "decode_ms": 1}, {"type": "polynomial"}])
@pytest.mark.parametrize(
    "field,value",
    [
        ("attention_backend", "fa3"),
        ("moe_kernel_source", "sglang_fused_moe_triton"),
        ("gemm_quant_mode", "fp8"),
        ("moe_quant_mode", "fp8"),
        ("kvcache_quant_mode", "fp8"),
        ("fmha_quant_mode", "fp8"),
        ("comm_quant_mode", "fp8"),
        ("nextn", 2),
    ],
)
def test_afd_controls_require_default_timing_in_ordinary_roles(afd_role, ordinary_role, timing, field, value):
    raw = prediction("disaggregated", (afd_role,))
    raw["engine"][field] = value
    if field == "nextn":
        raw["engine"]["nextn_accepted"] = 1.25
    raw["engine"]["workers"][ordinary_role]["timing"] = timing
    with pytest.raises(ValidationError, match="require default timing"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("mode,afd_roles", COMBINATIONS)
def test_kernel_mtp_and_quantization_identity_survives_search_and_reload(mode, afd_roles):
    raw = recommendation(mode, afd_roles)
    controls = {
        "attention_backend": "fa3",
        "moe_kernel_source": "sglang_fused_moe_triton",
        "kvcache_quant_mode": "fp8",
        "nextn": 2,
        "nextn_accepted": 1.25,
        "estimator_config": {},
    }
    raw["engine"].update(controls)
    source = CoreRecommendationConfig.model_validate(raw)
    smart = recommendation_to_sweeper(source)
    (branch,) = enumerate_branches(smart, max_seq_len=2048)
    suggestion = RandomBranchSampler(branch, seed=7).suggest(1)[0]
    sample = unroll_sample(
        search_space=smart.search_space, selection=suggestion.selection, parallel_config=suggestion.parallel_config
    )
    from aisimulate.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver

    estimators = ForwardPassEstimatorResolver(smart.search_space).resolve_candidate(sample)
    deployment = build_backend_deployment(sample, backend_version="0.5.14", forward_pass_estimators=estimators)
    spec = ReplaySpec(backend_deployment=deployment, workload=smart.workload.model_dump(), goal={})
    saved = _candidate_prediction(source, sample, spec, adapter_sections={})
    assert saved["engine"]["estimator_config"] == controls["estimator_config"]
    reloaded = prediction_to_replay_spec(CorePredictionConfig.model_validate(saved))
    execution = _materialize_engine_execution_spec(reloaded, trace_block_size=64, record_per_request=True)["spec"]
    for role in afd_roles:
        rank = (execution["engine"] if mode == "aggregated" else execution["engine"][role])["rank"]
        identity = rank["timing_model"]["config"]
        assert identity["attention_backend"] == controls["attention_backend"]
        assert identity["moe_kernel_source"] == controls["moe_kernel_source"]
        assert identity["quant"]["kvcache_quant_mode"] == "fp8"
        assert "frozen_decode_identity" not in identity
        assert "nextn" not in identity
        assert rank["aic_nextn"] == 2
        assert rank["aic_nextn_accept_rates"]


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_memory_failure_in_either_afd_role_prevents_replay(role):
    raw = prediction()
    raw["engine"]["workers"][role]["afd"]["a_batch_size"] = 1_000_000
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    with pytest.raises(RuntimeError, match="out_of_memory"):
        EngineReplayRunnerFactory().create(0).run(spec)


def test_search_candidate_uses_dynamic_pricing_before_execution():
    from aisimulate.sweeper.search import _materialize_one

    raw = recommendation("aggregated", ("aggregated",))
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))
    (branch,) = enumerate_branches(smart, max_seq_len=2048)
    suggestion = RandomBranchSampler(branch, seed=7).suggest(1)[0]
    prepared, error = _materialize_one(
        suggestion.selection,
        suggestion.parallel_config,
        config=smart,
        goal=smart.goal,
        providers={},
        provider_plans={},
        runner_factory=EngineReplayRunnerFactory(),
    )
    assert error is None
    assert prepared.replay_spec.api_version == 1
    assert (
        "frozen_decode_identity" not in prepared.replay_spec.backend_deployment.performance_model_metadata["aggregated"]
    )


def test_native_memory_diagnostics_cover_each_role_and_can_qualify_the_artifact():
    from aisimulate.afd_artifacts import build_afd_qualification

    raw = prediction()
    raw["engine"]["workers"]["decode"]["afd"].update(n_a_nodes=2, n_f_nodes=2, pipeline_layout="two_stage")
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    wire = _materialize_engine_execution_spec(spec, trace_block_size=64, record_per_request=True)["spec"]
    assert wire["engine"]["prefill"]["rank"]["max_num_seqs"] == 8 * 12
    assert wire["engine"]["decode"]["rank"]["max_num_seqs"] == 16 * 18
    assert wire["engine"]["prefill"]["rank"]["afd"]["pipeline_layout"] == "four_stage"
    assert wire["engine"]["decode"]["rank"]["afd"]["pipeline_layout"] == "two_stage"
    report = (
        EngineReplayRunnerFactory()
        .create(0)
        .run(
            spec, output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_memory_diagnostics=True)
        )
    )
    memory = report.metadata["native_report"]["afd_memory"]
    assert set(memory) == {"prefill", "decode"}
    for role in memory:
        assert set(memory[role]["pools"]) == {role}
        assert set(memory[role]["pools"][role]) == {"attention", "ffn"}
        assert memory[role]["kv_budget_bytes_per_a_gpu"] > 0
    artifact = build_afd_qualification(spec, memory=memory)
    assert all(artifact["deployment_plan"]["roles"][role]["memory"]["status"] == "fits" for role in memory)
    assert report.metrics["prefill_gpus_per_worker"] == 16
    assert report.metrics["decode_gpus_per_worker"] == 32
    assert report.metrics["completed_requests"] == 4
    assert report.metrics["total_output_tokens"] == 16
    assert artifact["deployment_plan"]["gpu_accounting"]["total_gpus"] == 48


def test_afd_native_config_has_one_source_for_rank_invariants():
    raw = prediction("aggregated", ("aggregated",))
    raw["engine"].update(nextn=2, nextn_accepted=1.25, comm_quant_mode="fp8")
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    wire = _materialize_engine_execution_spec(spec, trace_block_size=64, record_per_request=True)["spec"]
    rank = wire["engine"]["rank"]
    timing = rank["timing_model"]["config"]
    assert rank["backend"] == "sglang"
    assert rank["aic_nextn"] == 2
    assert rank["max_model_len"] == 2048
    assert timing["quant"]["comm_quant_mode"] == "fp8"
    duplicates = set(rank["afd"]) | {"backend", "nextn", "comm_quant_mode", "max_seq_len"}
    assert not duplicates.intersection(timing)


def test_afd_communication_precision_changes_native_sol_prediction():
    # At SOL, fp8 halves the transfer payload relative to half. Pool compute
    # geometry and workload are identical, so the complete serial pass shrinks.
    durations = {}
    for mode in ("half", "fp8"):
        raw = prediction("aggregated", ("aggregated",))
        raw["engine"].update(database_mode="SOL", comm_quant_mode=mode)
        raw["engine"]["workers"]["aggregated"]["afd"].update(num_microbatches=1)
        spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
        report = EngineReplayRunnerFactory().create(0).run(spec)
        assert report.metrics["completed_requests"] == 4
        durations[mode] = report.metrics["duration_ms"]
    assert durations["fp8"] < durations["half"]


def test_native_afd_topology_drives_both_timing_and_gpu_accounting():
    import json

    from aisimulate import _runtime

    raw = prediction("aggregated", ("aggregated",))
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    wire = _materialize_engine_execution_spec(spec, trace_block_size=64, record_per_request=True)
    first = json.loads(_runtime.run_replay_json(json.dumps(wire)))
    rank = wire["spec"]["engine"]["rank"]
    rank["afd"]["a_pool"]["n_a_nodes"] = 2
    second = json.loads(_runtime.run_replay_json(json.dumps(wire)))
    assert first["decode_gpus_per_worker"] == 16
    assert second["decode_gpus_per_worker"] == 24
    assert second["duration_ms"] == first["duration_ms"]  # same actual pass geometry while underloaded
    assert first["completed_requests"] == second["completed_requests"] == 4


def test_native_afd_backend_is_validated_before_model_loading():
    import json

    from aisimulate import _runtime

    raw = prediction("aggregated", ("aggregated",))
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    wire = _materialize_engine_execution_spec(spec, trace_block_size=64, record_per_request=False)
    rank = wire["spec"]["engine"]["rank"]
    rank.update(backend="vllm", block_size=16)
    rank["timing_model"]["config"]["model"] = "missing/model"
    with pytest.raises(RuntimeError, match="AFD ranks require backend=sglang"):
        _runtime.run_replay_json(json.dumps(wire))


@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("context_length", [None, 146, 2048])
@pytest.mark.parametrize("legacy", [False, True])
def test_native_afd_memory_reserves_each_phase_without_promoting_derived_context(mode, context_length, legacy):
    import json

    from aisimulate import _runtime

    roles = ("aggregated",) if mode == "aggregated" else ("prefill", "decode")
    raw = prediction(mode, roles)
    raw["traffic"]["source"].update(input_tokens=16, output_tokens=128)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    # Generic Sweeper permits an absent context window, unlike predict's
    # default `max`. Exercise that supported metadata at the native boundary.
    for role in roles:
        spec.backend_deployment.performance_model_metadata[role]["context_length"] = context_length
    wire = _materialize_engine_execution_spec(spec, trace_block_size=64, record_per_request=False)
    for role in roles:
        descriptor = wire["spec"]["engine"]
        rank = (descriptor if mode == "aggregated" else descriptor[role])["rank"]
        assert rank["max_model_len"] == context_length
        if legacy:
            rank["max_model_len"] = context_length or 144
            rank["timing_model"]["config"].update(backend=rank["backend"], max_seq_len=context_length)
    report = json.loads(_runtime.run_replay_json(json.dumps(wire)))
    assert report["completed_requests"] == 4
    # Qwen3-30B-A3B: 48 layers * K/V * 4 KV heads * 128 head width * fp16.
    kv_bytes_per_token = 48 * 2 * 4 * 128 * 2
    for role in roles:
        memory = report["afd_memory"][role]["pools"]
        reserved_batch = 12 if role == "prefill" else 18
        for phase in ("prefill", "decode") if role == "aggregated" else (role,):
            seq_len = context_length or 144
            expected_gib = reserved_batch * seq_len * kv_bytes_per_token / (1 << 30)
            assert memory[phase]["attention"]["components_gib_per_gpu"]["kvcache"] == expected_gib


@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
@pytest.mark.parametrize("output_tokens", [3, 8])
@pytest.mark.parametrize("at_limit", [False, True])
def test_native_afd_context_boundaries_preserve_requested_output(role, output_tokens, at_limit):
    # With ISL=16, SGLang admission needs 23 positions; complete generation
    # needs 18+OSL. A prefill-only role hands off after its first output token.
    minimum = {"aggregated": max(23, 18 + output_tokens), "prefill": 23, "decode": 18 + output_tokens}[role]
    window = minimum if at_limit else minimum - 1
    raw = prediction("aggregated" if role == "aggregated" else "disaggregated", (role,))
    raw["traffic"]["source"].update(input_tokens=16, output_tokens=output_tokens)
    raw["traffic"]["load"]["concurrency"] = 1
    raw["traffic"]["stop"]["requests"] = 1
    if role == "aggregated":
        raw["engine"]["context_length"] = window
    else:
        raw["engine"]["workers"][role]["context_length"] = window
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    runner = EngineReplayRunnerFactory(determinism="canonical_v1").create(0)
    try:
        if not at_limit:
            with pytest.raises(InvalidRunnerError, match=f"AFD {role} context_length=.*cannot hold"):
                runner.run(spec)
            return
        report = runner.run(
            spec, output_requirements=ReplayOutputRequirements(capture_per_request=True, include_raw_report=True)
        )
    finally:
        runner.close()
    assert report.metrics["completed_requests"] == 1
    assert report.metrics["total_output_tokens"] == output_tokens
    record = report.metadata["native_report"]["per_request"][0]
    assert record["terminal_status"] == "completed"
    assert record["output_length"] == output_tokens


@pytest.mark.parametrize("afd_roles", [("prefill",), ("prefill", "decode")])
def test_native_afd_prefill_context_does_not_reserve_the_decode_output(afd_roles):
    raw = prediction("disaggregated", afd_roles)
    raw["traffic"]["source"].update(input_tokens=16, output_tokens=128)
    # Leave SGLang's native admission headroom while staying below ISL+OSL=144.
    raw["engine"]["workers"]["prefill"]["context_length"] = 32
    config = CorePredictionConfig.model_validate(raw)
    restored = CorePredictionConfig.model_validate_json(config.model_dump_json())
    spec = prediction_to_replay_spec(restored)
    runner = EngineReplayRunnerFactory(determinism="canonical_v1").create(0)
    try:
        report = runner.run(
            spec, output_requirements=ReplayOutputRequirements(capture_per_request=True, include_raw_report=True)
        )
    finally:
        runner.close()

    assert report.metrics["completed_requests"] == 4
    assert report.metrics["total_input_tokens"] == 64
    assert report.metrics["total_output_tokens"] == 512
    records = report.metadata["native_report"]["per_request"]
    assert all(r["terminal_status"] == "completed" and r["output_length"] == 128 for r in records)
    memory = report.metadata["native_report"]["afd_memory"]["prefill"]["pools"]["prefill"]
    kv_bytes_per_token = 48 * 2 * 4 * 128 * 2
    assert memory["attention"]["components_gib_per_gpu"]["kvcache"] == 12 * 32 * kv_bytes_per_token / (1 << 30)


def test_native_afd_undeclared_window_still_limits_runtime_prompt_length():
    import json

    from aisimulate import _runtime

    raw = prediction("aggregated", ("aggregated",))
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    spec.backend_deployment.performance_model_metadata["aggregated"]["context_length"] = None
    wire = _materialize_engine_execution_spec(spec, trace_block_size=64, record_per_request=True)
    assert wire["spec"]["engine"]["rank"]["max_model_len"] is None
    # The provider declares 16 input + 4 output tokens. A prompt at its derived
    # limit must still fail even though the serialized declaration stays None.
    wire["traffic"]["isl"] = 20
    report = json.loads(_runtime.run_replay_json(json.dumps(wire)))
    assert report["completed_requests"] == report["total_output_tokens"] == 0
    assert len(report["per_request"]) == 4
    assert all(request["terminal_status"] == "rejected" for request in report["per_request"])


def hetero_specs(tmp_path, a_width, f_width):
    import aisimulate_core

    root = Path(aisimulate_core.__file__).parent / "systems"
    system = yaml.safe_load((root / "h200_sxm.yaml").read_text())
    system["data_dir"] = str(root / "data/h200_sxm")
    for name, width in (("hetero_a", a_width), ("hetero_f", f_width)):
        system["node"]["num_gpus_per_node"] = width
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(system))


@pytest.mark.parametrize("a_width,f_width", [(8, 4), (4, 8)])
@pytest.mark.parametrize("mode,afd_roles", COMBINATIONS)
def test_heterogeneous_roles_compile_predict_and_replay_saved_identity(tmp_path, a_width, f_width, mode, afd_roles):
    hetero_specs(tmp_path, a_width, f_width)
    raw = prediction(mode, afd_roles)
    raw["engine"].update(hardware="hetero_f", systems_paths=[str(tmp_path)])
    for role in afd_roles:
        raw["engine"]["workers"][role]["afd"].update(a_system="hetero_a", f_system="hetero_f", tp_a=2)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    wire = _materialize_engine_execution_spec(spec, trace_block_size=64, record_per_request=True)["spec"]
    for role in afd_roles:
        topo = spec.backend_deployment.parallel_config[role]["topology"]
        assert (topo["gpus_per_node"], topo["f_gpus_per_node"]) == (a_width, f_width)
        assert topo["a_replica_count"] == a_width // 2
        assert topo["f_rank_count"] == f_width
        descriptor = wire["engine"] if mode == "aggregated" else wire["engine"][role]
        assert descriptor["rank"]["afd"]["f_gpus_per_node"] == f_width
        assert descriptor["rank"]["timing_model"]["config"]["backend_version"] == "0.5.14"
    report = EngineReplayRunnerFactory().create(0).run(spec)
    assert report.metrics["completed_requests"] == 4
    assert report.metrics["total_input_tokens"] == 64
    assert report.metrics["total_output_tokens"] == 16
    for role in afd_roles:
        for reported_role in ("decode",) if role == "aggregated" else (role,):
            assert report.metrics[f"{reported_role}_gpus_per_worker"] == a_width + f_width
    # The public saved prediction reproduces the exact internal hardware geometry.
    restored = prediction_to_replay_spec(
        CorePredictionConfig.model_validate_json(CorePredictionConfig.model_validate(raw).model_dump_json())
    )
    assert restored.backend_deployment.parallel_config == spec.backend_deployment.parallel_config


@pytest.mark.parametrize("a_width,f_width", [(8, 4), (4, 8)])
def test_heterogeneous_search_saved_prediction_and_replay(tmp_path, a_width, f_width):
    hetero_specs(tmp_path, a_width, f_width)
    raw = recommendation("aggregated", ("aggregated",))
    raw["engine"].update(hardware="hetero_a", systems_paths=[str(tmp_path)])
    raw["engine"]["workers"]["aggregated"]["afd"].update(f_system="hetero_f", tp_a=2)
    raw["optimization"]["constraints"]["max_candidate_gpus"] = a_width + f_width
    source = CoreRecommendationConfig.model_validate(raw)
    smart = recommendation_to_sweeper(source)
    (branch,) = enumerate_branches(smart, max_seq_len=2048)
    assert len(branch.parallel_configs) == 1
    candidate = branch.parallel_configs[0]
    assert candidate.total_gpus == a_width + f_width
    assert candidate.f_rank_count == f_width
    suggestion = RandomBranchSampler(branch, seed=7).suggest(1)[0]
    sample = unroll_sample(
        search_space=smart.search_space, selection=suggestion.selection, parallel_config=suggestion.parallel_config
    )
    deployment = build_backend_deployment(sample, backend_version="0.5.14")
    spec = ReplaySpec(backend_deployment=deployment, workload=smart.workload.model_dump(), goal={})
    saved = _candidate_prediction(source, sample, spec, adapter_sections={})
    recompiled = prediction_to_replay_spec(CorePredictionConfig.model_validate(saved))
    assert recompiled.backend_deployment.parallel_config == deployment.parallel_config
    assert saved["engine"]["backend_version"] == "0.5.14"
    assert EngineReplayRunnerFactory().create(0).run(recompiled).metrics["completed_requests"] == 4


@pytest.mark.parametrize("requested", [None, "current", "previous", "next"])
def test_afd_retains_one_shared_version_across_hardware(monkeypatch, requested):
    raw = prediction("aggregated", ("aggregated",))
    raw["engine"]["backend_version"] = requested
    raw["engine"]["workers"]["aggregated"]["afd"]["f_system"] = "b200_sxm"

    def resolve(system, backend, *args, **kwargs):
        return "0.5.14" if system == "h200_sxm" else "0.5.15"

    monkeypatch.setattr("aisimulate.sweeper.kv_estimate.resolve_backend_version", resolve)
    monkeypatch.setattr("aisimulate_core.sdk.perf_database.resolve_query_version", resolve)
    with pytest.raises(ValueError, match="common|resolves differently"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))


@pytest.mark.parametrize("requested", [None, "current", "previous", "next"])
@pytest.mark.parametrize("ordinary_companion", [False, True])
def test_recommendation_pins_one_afd_version_and_reuses_resolution(monkeypatch, requested, ordinary_companion):
    from aisimulate.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver
    from aisimulate.sweeper.search import _materialize_one

    raw = recommendation(afd_roles=("decode",)) if ordinary_companion else recommendation("aggregated", ("aggregated",))
    raw["engine"]["backend_version"] = requested
    source = CoreRecommendationConfig.model_validate(raw)
    smart = recommendation_to_sweeper(source)
    calls = []

    def resolve(system, backend, *args, **kwargs):
        calls.append((system, tuple(kwargs["systems_paths"])))
        return "0.5.14"

    monkeypatch.setattr("aisimulate.sweeper.kv_estimate.resolve_backend_version", resolve)
    monkeypatch.setattr("aisimulate_core.sdk.perf_database.resolve_query_version", resolve)
    (branch,) = enumerate_branches(smart, max_seq_len=2048)
    suggestion = RandomBranchSampler(branch, seed=7).suggest(1)[0]
    resolver = ForwardPassEstimatorResolver(smart.search_space)
    prepared, error = _materialize_one(
        suggestion.selection,
        suggestion.parallel_config,
        config=smart,
        goal=smart.goal,
        providers={},
        provider_plans={},
        runner_factory=EngineReplayRunnerFactory(),
        estimator_resolver=resolver,
        prediction_config_factory=lambda sample, spec: _candidate_prediction(source, sample, spec, adapter_sections={}),
    )
    assert error is None
    assert prepared.replay_spec.backend_deployment.backend_version == "0.5.14"
    saved = prepared.prediction_config
    assert saved["engine"]["backend_version"] == "0.5.14"
    count = len(calls)
    estimators = resolver.resolve_candidate(prepared.sample)
    assert resolver.resolve_afd_version(prepared.sample, estimators) == "0.5.14"
    assert len(calls) == count
    reloaded = prediction_to_replay_spec(CorePredictionConfig.model_validate(saved))
    assert reloaded.backend_deployment.backend_version == "0.5.14"


@pytest.mark.parametrize("requested", [None, "current"])
def test_recommendation_rejects_different_afd_pool_versions(tmp_path, requested):
    from aisimulate.sweeper.search import _materialize_one

    hetero_specs(tmp_path, 8, 8)
    (tmp_path / "query_versions.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "defaults": {"sglang": {"current": "0.5.14"}},
                "overrides": {"hetero_f": {"sglang": {"current": "0.5.16"}}},
            }
        )
    )
    raw = recommendation("aggregated", ("aggregated",))
    raw["engine"].update(hardware="hetero_a", systems_paths=[str(tmp_path)], backend_version=requested)
    raw["engine"]["workers"]["aggregated"]["afd"]["f_system"] = "hetero_f"
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))
    (branch,) = enumerate_branches(smart, max_seq_len=2048)
    suggestion = RandomBranchSampler(branch, seed=7).suggest(1)[0]
    prepared, error = _materialize_one(
        suggestion.selection,
        suggestion.parallel_config,
        config=smart,
        goal=smart.goal,
        providers={},
        provider_plans={},
        runner_factory=EngineReplayRunnerFactory(),
    )
    assert prepared is None
    assert "common literal" in error.reason


@pytest.mark.parametrize("trace", [False, True])
def test_prefill_memory_rejects_oom_for_synthetic_and_trace(tmp_path, trace):
    import json

    hetero_specs(tmp_path, 8, 8)
    path = tmp_path / "hetero_f.yaml"
    system = yaml.safe_load(path.read_text())
    system["gpu"]["mem_capacity"] = 7 * (1 << 30)
    path.write_text(yaml.safe_dump(system))
    raw = prediction("aggregated", ("aggregated",))
    raw["engine"].update(hardware="hetero_a", systems_paths=[str(tmp_path)])
    raw["engine"]["workers"]["aggregated"]["afd"].update(a_batch_size=1, num_microbatches=1, f_system="hetero_f")
    raw["traffic"]["source"]["input_tokens"] = 1024
    if trace:
        path = tmp_path / "trace.jsonl"
        path.write_text(
            json.dumps({"timestamp": 0, "input_length": 1024, "output_length": 4, "hash_ids": [1, 2]}) + "\n"
        )
        raw["traffic"]["source"] = {"type": "trace", "paths": [str(path)], "format": "mooncake"}
        raw["traffic"].pop("stop")
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    # The F pool has 6.75 GiB of routed weights. A 1024-token prefill needs
    # another 5.52 GiB for activations, workspace and fixed reservations;
    # pricing a trace at its one-token placeholder incorrectly fits 12 GiB.
    with pytest.raises(RuntimeError, match="out_of_memory.*F pool, phase=prefill"):
        EngineReplayRunnerFactory().create(0).run(spec)


@pytest.mark.parametrize("trace", [False, True])
def test_prefill_chunk_bounds_buffers_without_shortening_kv_window(tmp_path, trace):
    import json

    memories = []
    for tokens in (16, 1024):
        raw = prediction("aggregated", ("aggregated",))
        raw["engine"]["workers"]["aggregated"]["afd"].update(
            a_batch_size=1, num_microbatches=1, chunked_prefill_size=16
        )
        raw["traffic"]["source"]["input_tokens"] = tokens
        raw["traffic"]["stop"]["requests"] = 1
        if trace:
            path = tmp_path / f"trace-{tokens}.jsonl"
            path.write_text(
                json.dumps({"timestamp": 0, "input_length": tokens, "output_length": 4, "hash_ids": [1, 2]}) + "\n"
            )
            raw["traffic"]["source"] = {"type": "trace", "paths": [str(path)], "format": "mooncake"}
            raw["traffic"].pop("stop")
        spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
        report = (
            EngineReplayRunnerFactory()
            .create(0)
            .run(spec, output_requirements=ReplayOutputRequirements(include_raw_report=True))
        )
        assert report.metrics["completed_requests"] == 1
        memories.append(report.metadata["native_report"]["afd_memory"]["aggregated"])
        # Qwen's 48 layers, four 128-wide KV heads and BF16 K/V reserve
        # 2048 * 48 * 4 * 128 * 2 * 2 bytes = 0.1875 GiB per A GPU.
        assert memories[-1]["pools"]["prefill"]["attention"]["components_gib_per_gpu"]["kvcache"] == pytest.approx(
            0.1875
        )
    # Both passes process at most 16 new tokens, while both keep a 2048-token
    # declared KV window. Fit diagnostics and derived KV capacity must agree.
    assert memories[0] == memories[1]


def test_afd_rejects_missing_explicit_common_database_version():
    raw = prediction("disaggregated", ("prefill", "decode"))
    raw["engine"]["backend_version"] = "999.0.0"
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    assert spec.backend_deployment.backend_version == "999.0.0"
    with pytest.raises((ValueError, RuntimeError), match="999.0.0"):
        EngineReplayRunnerFactory().create(0).run(spec)


@pytest.mark.parametrize("version", [1, 3, 4, 5, 6])
def test_saved_replay_rejects_unsupported_topology_schema_before_model_lookup(version, monkeypatch):
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(prediction()))
    role = spec.backend_deployment.afd_roles[0]
    spec.backend_deployment.parallel_config[role]["schema_version"] = version
    monkeypatch.setattr("aisimulate.runner.validate_afd_model", lambda _: pytest.fail("model lookup"))
    runner = EngineReplayRunnerFactory().create(0)
    try:
        with pytest.raises(ValueError, match="AFD topology schema"):
            runner.run(spec)
    finally:
        runner.close()


def test_current_saved_json_replays_with_identical_semantics_despite_environment(monkeypatch):
    import json

    from aisimulate.sweeper.replay import BackendDeploymentSpec, canonical_json

    current = prediction_to_replay_spec(CorePredictionConfig.model_validate(prediction()))
    wire = json.loads(canonical_json(current))

    wire["backend_deployment"] = BackendDeploymentSpec(**wire["backend_deployment"])
    restored = ReplaySpec(**wire)
    assert restored.api_version == 1
    factory = EngineReplayRunnerFactory(determinism="canonical_v1")
    reports = []
    for s, environment in ((current, "0"), (restored, "1")):
        monkeypatch.setenv("AISIMULATE_AFD_FROZEN_DECODE_IDENTITY", environment)
        reports.append(
            factory.create(0).run(
                s, output_requirements=ReplayOutputRequirements(capture_per_request=True, include_raw_report=True)
            )
        )
    host_metrics = {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}
    assert {k: v for k, v in reports[0].metrics.items() if k not in host_metrics} == {
        k: v for k, v in reports[1].metrics.items() if k not in host_metrics
    }
    assert reports[0].metadata["native_report"]["per_request"] == reports[1].metadata["native_report"]["per_request"]


@pytest.mark.parametrize("pool,capacity_gib,label", [("a", 6, "A"), ("f", 12, "F")])
def test_heterogeneous_memory_gate_checks_the_actual_pool_width(tmp_path, pool, capacity_gib, label):
    hetero_specs(tmp_path, 8, 4)
    path = tmp_path / f"hetero_{pool}.yaml"
    system = yaml.safe_load(path.read_text())
    system["gpu"]["mem_capacity"] = capacity_gib * (1 << 30)
    path.write_text(yaml.safe_dump(system))
    raw = prediction("aggregated", ("aggregated",))
    raw["engine"].update(hardware="hetero_a", systems_paths=[str(tmp_path)])
    raw["engine"]["workers"]["aggregated"]["afd"].update(f_system="hetero_f", tp_a=2)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    with pytest.raises(RuntimeError, match=f"{label} pool"):
        EngineReplayRunnerFactory().create(0).run(spec)


@pytest.mark.parametrize(("mode", "afd_roles"), COMBINATIONS)
def test_afd_roles_reject_main_decode_radix_cache(mode, afd_roles):
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(recommendation(mode, afd_roles)))
    payload = smart.model_dump(mode="python")
    payload["search_space"]["decode_radix_cache"] = True
    payload["search_space"]["decode_enable_prefix_caching"] = True
    with pytest.raises(ValidationError, match="decode_radix_cache is unsupported for AFD"):
        type(smart).model_validate(payload)


def test_afd_prefill_ordinary_decode_rejects_radix_cache_from_direct_sample():
    smart = recommendation_to_sweeper(
        CoreRecommendationConfig.model_validate(recommendation("disaggregated", ("prefill",)))
    )
    branch = enumerate_branches(smart, max_seq_len=2048)[0]
    suggestion = RandomBranchSampler(branch, seed=7).suggest(1)[0]
    sample = unroll_sample(
        search_space=smart.search_space, selection=suggestion.selection, parallel_config=suggestion.parallel_config
    )
    assert build_backend_deployment(sample, backend_version="0.5.14").decode_engine_args is not None
    sample["decode_radix_cache"] = True
    with pytest.raises(ValueError, match="requires native SGLang disaggregated decode"):
        build_backend_deployment(sample, backend_version="0.5.14")


@pytest.mark.parametrize("field", ["a_batch_size", "tp_a", "num_microbatches"])
def test_afd_log_integer_domain_rejects_at_public_input(field):
    raw = recommendation("aggregated", ("aggregated",))
    raw["engine"]["workers"]["aggregated"]["afd"][field] = {"range": {"min": 1, "max": 8, "scale": "log"}}
    with pytest.raises(ValueError, match="AFD integer domains do not support log"):
        CoreRecommendationConfig.model_validate(raw)


@pytest.mark.parametrize("ratio", [0.5, [0.1, 0.9], "generic"])
def test_sdk_afd_rejects_every_kv_ratio_load_form(ratio):
    from aisimulate.sweeper.config import SmartSearchConfig

    payload = recommendation_to_sweeper(
        CoreRecommendationConfig.model_validate(recommendation("aggregated", ("aggregated",)))
    ).model_dump(mode="json")
    workload = payload["workload"]
    workload["concurrency"] = None
    if ratio == "generic":
        workload["concurrency"] = 1
        workload.update(load_search_field="kv_load_ratio", load_choices=[0.1, 0.9])
    else:
        workload["kv_load_ratio"] = ratio
    with pytest.raises(ValueError, match="AFD does not support kv_load_ratio"):
        SmartSearchConfig.model_validate(payload)


@pytest.mark.parametrize("key", ["parallel_configs", "parallel_configs_by_mode"])
def test_sdk_afd_rejects_obsolete_parallel_pins(key):
    from aisimulate.sweeper.config import SmartSearchConfig

    payload = recommendation_to_sweeper(
        CoreRecommendationConfig.model_validate(recommendation("aggregated", ("aggregated",)))
    ).model_dump(mode="json")
    entries = [{"tp": 2}]
    payload["search_space"][key] = entries if key == "parallel_configs" else {"agg": entries}
    with pytest.raises(ValueError, match="AFD modes do not support parallel_configs"):
        SmartSearchConfig.model_validate(payload)


@pytest.mark.parametrize("value", [None, False, True])
@pytest.mark.parametrize("mode,roles", COMBINATIONS)
def test_retired_pricing_config_rejected_before_replay(value, mode, roles):
    raw = prediction(mode, roles)
    raw["engine"]["estimator_config"] = {"afd": {"frozen_decode_identity": value}}
    with pytest.raises(ValueError, match="removed; regenerate"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("blocks", [None, 64])
@pytest.mark.parametrize("size", [2, 16])
def test_raw_afd_block_units_reject_before_any_memory_query(monkeypatch, blocks, size):
    import json

    from aisimulate import _runtime

    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(prediction("aggregated", ("aggregated",))))
    execution = _materialize_engine_execution_spec(spec, trace_block_size=64, record_per_request=True)
    rank = execution["spec"]["engine"]["rank"]
    rank["block_size"] = size
    if blocks is not None:
        rank["num_gpu_blocks"] = blocks
    monkeypatch.setattr(
        "aisimulate.sdk.afd_memory.check_afd_memory_fit", lambda **kwargs: pytest.fail("memory queried")
    )
    monkeypatch.setattr(
        "aisimulate.sdk.afd_memory.estimate_afd_kv_budget_bytes", lambda **kwargs: pytest.fail("capacity queried")
    )
    with pytest.raises(RuntimeError, match="AFD requires block_size=1"):
        _runtime.run_replay_json(json.dumps(execution))


@pytest.mark.parametrize("mode,roles", COMBINATIONS)
def test_available_memory_diagnostics_preserve_native_fit_status(mode, roles):
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(prediction(mode, roles)))
    requirements = ReplayOutputRequirements(capture_memory_diagnostics=True)
    report = EngineReplayRunnerFactory().create(0).run(spec, output_requirements=requirements)
    native = report.metadata["native_report"]
    for role in roles:
        assert native["afd_memory"][role]["status"] == "fits"
        assert native["memory_diagnostics"][role]["status"] == "available"


def test_prefill_pool_budget_accepts_the_reported_two_a_four_f_geometry():
    raw = prediction("aggregated", ("aggregated",))
    raw["engine"]["context_length"] = 4226
    raw["engine"]["workers"]["aggregated"]["afd"].update(n_a_nodes=2, n_f_nodes=4, a_batch_size=64, num_microbatches=3)
    raw["traffic"]["source"].update(input_tokens=4096, output_tokens=128)
    raw["traffic"]["stop"]["requests"] = 1
    report = (
        EngineReplayRunnerFactory().create(0).run(prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)))
    )
    assert report.metrics["completed_requests"] == 1
    assert report.metrics["total_output_tokens"] == 128


@pytest.mark.parametrize("batched_tokens", [None, 1, 32768])
@pytest.mark.parametrize("chunk_size", [None, 4096])
def test_long_prefill_memory_uses_sglang_limits_for_fit_and_kv_budget(monkeypatch, batched_tokens, chunk_size):
    import json

    from aisimulate import _runtime
    from aisimulate.sdk import afd_memory

    raw = prediction("aggregated", ("aggregated",))
    raw["engine"]["context_length"] = 20000
    raw["traffic"]["source"].update(input_tokens=16384, output_tokens=1)
    raw["traffic"]["stop"]["requests"] = 1
    raw["traffic"]["load"]["concurrency"] = 1
    if chunk_size is not None:
        raw["engine"]["workers"]["aggregated"]["afd"]["chunked_prefill_size"] = chunk_size
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    execution = _materialize_engine_execution_spec(spec, trace_block_size=64, record_per_request=True)
    rank = execution["spec"]["engine"]["rank"]
    if batched_tokens is not None:
        rank["max_num_batched_tokens"] = batched_tokens

    captured = {}
    for name in ("check_afd_memory_fit", "estimate_afd_kv_budget_bytes"):
        original = getattr(afd_memory, name)

        def capture(*args, _name=name, _original=original, **kwargs):
            captured[_name] = kwargs
            return _original(*args, **kwargs)

        monkeypatch.setattr(afd_memory, name, capture)

    report = json.loads(_runtime.run_replay_json(json.dumps(execution)))
    assert report["completed_requests"] == report["total_output_tokens"] == 1
    assert report["total_input_tokens"] == report["committed_prefill_tokens"] == 16384
    assert len(captured) == 2
    for kwargs in captured.values():
        # Reserve the declared pool capacity, including a full long prompt
        # when unchunked, regardless of the ignored batched-token setting.
        assert kwargs["prefill_tokens"] == kwargs["a_prefill_tokens"] == (chunk_size or 20000)
    fit = captured["check_afd_memory_fit"]
    budget = captured["estimate_afd_kv_budget_bytes"]
    assert fit["a_workspace_gb"] == budget["a_workspace_gb"]
    assert fit["a_activation_gb"] == budget["a_activation_gb"]


def test_unit_block_auto_capacity_preserves_controlled_64_token_admission(monkeypatch):
    import json

    from aisimulate import _runtime

    raw = prediction("aggregated", ("aggregated",))
    raw["engine"]["context_length"] = 128
    raw["traffic"].update(source={"type": "synthetic", "input_tokens": 32, "output_tokens": 16}, stop={"requests": 2})
    raw["engine"]["workers"]["aggregated"]["afd"].update(tp_a=8, a_batch_size=8, num_microbatches=1)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    execution = _materialize_engine_execution_spec(spec, trace_block_size=64, record_per_request=True)
    rank = execution["spec"]["engine"]["rank"]
    rank["enable_prefix_caching"] = False
    rank["sglang"].update(schedule_conservativeness=0.0, chunked_prefill_size=128, max_prefill_tokens=128)
    monkeypatch.setattr(
        "aisimulate.sdk.afd_memory.estimate_afd_kv_budget_bytes", lambda **kwargs: (64 * 32768.0, 32768.0)
    )
    result = json.loads(_runtime.run_replay_json(json.dumps(execution)))
    assert result["completed_requests"] == 2
    assert result["total_input_tokens"] == 64 and result["total_output_tokens"] == 32
    admissions = sorted(request["first_admit_ms"] for request in result["per_request"])
    assert admissions[0] == 0 < admissions[1]


@pytest.mark.parametrize("field", ["parallel_custom_configs_by_mode", "parallel_independent_by_mode"])
def test_sdk_keeps_supported_ordinary_companion_search_domains(field):
    from aisimulate.sweeper.config import SmartSearchConfig

    payload = recommendation_to_sweeper(
        CoreRecommendationConfig.model_validate(recommendation("disaggregated", ("decode",)))
    ).model_dump(mode="json")
    search = payload["search_space"]
    search["parallel_custom_configs_by_mode"] = {}
    search["parallel_independent_by_mode"] = {}
    search["parallel_independent_log_ranges_by_mode"] = {}
    search[field] = (
        {"disagg": {"prefill": [{"tp": 2, "moe_tp": 2, "moe_ep": 1}]}}
        if field.startswith("parallel_custom")
        else {"disagg": {"prefill_tp": [2]}}
    )
    config = SmartSearchConfig.model_validate(payload)
    branches = enumerate_branches(config, max_seq_len=2048)
    assert branches
    assert {parallel.prefill.shape.tp for branch in branches for parallel in branch.parallel_configs} == {2}


@pytest.mark.parametrize(("mode", "afd_roles"), COMBINATIONS)
def test_common_afd_runtime_fields_survive_prediction_and_execution(mode, afd_roles):
    raw = prediction(mode, afd_roles)
    for role in afd_roles:
        raw["engine"]["workers"][role].update(
            scheduler={"max_sequences": 1, "max_batched_tokens": 8},
            kv_cache={
                "prefix_caching": False,
                "capacity": {"type": "fixed", "blocks": 8192},
            },
        )
    config = CorePredictionConfig.model_validate(raw)
    restored = CorePredictionConfig.model_validate_json(config.model_dump_json())
    spec = prediction_to_replay_spec(restored)
    execution = _materialize_engine_execution_spec(spec, trace_block_size=1, record_per_request=True)
    for role in afd_roles:
        descriptor = execution["spec"]["engine"]
        rank = descriptor["rank"] if role == "aggregated" else descriptor[role]["rank"]
        assert rank["max_num_seqs"] == 1
        assert rank["num_gpu_blocks"] == 8192
        assert rank["enable_prefix_caching"] is False
        assert rank["sglang"]["chunked_prefill_size"] == 8
    report = EngineReplayRunnerFactory().create(0).run(spec)
    assert report.metrics["completed_requests"] == 4
    assert report.metrics["total_input_tokens"] == 64
    assert report.metrics["total_output_tokens"] == 16


def test_public_afd_parallelism_aliases_have_one_canonical_identity():
    raw = prediction("aggregated", ("aggregated",))
    worker = raw["engine"]["workers"]["aggregated"]
    del worker["afd"]["tp_a"]
    worker["parallelism"] = {"tensor": 2, "attention_data": 4, "moe_expert": 8}
    worker["hardware"] = "h200_sxm"
    config = CorePredictionConfig.model_validate(raw)
    saved = config.model_dump(mode="json")["engine"]["workers"]["aggregated"]
    assert "tensor" not in saved["parallelism"] and "moe_expert" not in saved["parallelism"]
    assert saved["afd"]["tp_a"] == 2 and saved["afd"]["f_moe_ep_size"] == 8
    spec = prediction_to_replay_spec(config)
    assert spec.backend_deployment.agg_engine_args["afd"]["a_replica_count"] == 4
    worker["afd"]["tp_a"] = 1
    with pytest.raises(ValueError, match="conflicts"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("kind", ["independent", "custom"])
def test_sdk_afd_parallel_constraints_are_enforced(kind):
    from aisimulate.sweeper.config import SmartSearchConfig

    payload = recommendation_to_sweeper(
        CoreRecommendationConfig.model_validate(recommendation("aggregated", ("aggregated",)))
    ).model_dump(mode="json")
    payload["search_space"]["afd_roles"]["aggregated"]["afd"].pop("tp_a")
    if kind == "independent":
        payload["search_space"]["parallel_independent_by_mode"] = {"agg": {"tp": [2]}}
    else:
        payload["search_space"]["parallel_custom_configs_by_mode"] = {
            "agg": {"agg": [{"tp": 2, "attention_dp": 4, "moe_ep": 8}]}
        }
    config = SmartSearchConfig.model_validate(payload)
    (branch,) = enumerate_branches(config, max_seq_len=2048)
    assert {t.a_tp_size for t in branch.parallel_configs} == {2}
    if kind == "custom":
        assert {t.a_replica_count for t in branch.parallel_configs} == {4}
    payload["search_space"]["afd_roles"]["aggregated"]["afd"]["tp_a"] = 1
    with pytest.raises(ValueError, match="infeasible|conflicts"):
        enumerate_branches(SmartSearchConfig.model_validate(payload), max_seq_len=2048)


def test_sdk_common_afd_runtime_fields_survive_search_save_and_recompile():
    from aisimulate.sweeper.config import SmartSearchConfig

    source = CoreRecommendationConfig.model_validate(recommendation("aggregated", ("aggregated",)))
    payload = recommendation_to_sweeper(source).model_dump(mode="json")
    payload["search_space"].update(
        agg_max_num_seqs=[1, 2],
        agg_num_gpu_blocks=64,
        agg_enable_prefix_caching=False,
    )
    config = SmartSearchConfig.model_validate(payload)
    config = SmartSearchConfig.model_validate_json(config.model_dump_json())
    assert "agg_max_num_seqs" not in config.search_space.model_dump(mode="json")
    (branch,) = enumerate_branches(config, max_seq_len=2048)
    assert branch.knob_choices["agg_max_num_seqs"] == [1, 2]
    suggestion = RandomBranchSampler(branch, seed=7).suggest(1)[0]
    sample = unroll_sample(
        search_space=config.search_space,
        selection=suggestion.selection,
        parallel_config=suggestion.parallel_config,
    )
    deployment = build_backend_deployment(sample, backend_version="0.5.14")
    spec = ReplaySpec(backend_deployment=deployment, workload=config.workload.model_dump(), goal={})
    saved = _candidate_prediction(source, sample, spec, adapter_sections={})
    restored = prediction_to_replay_spec(CorePredictionConfig.model_validate(saved))
    args = restored.backend_deployment.agg_engine_args
    assert args["max_num_seqs"] == suggestion.selection["agg_max_num_seqs"]
    assert args["num_gpu_blocks"] == 64 and args["enable_prefix_caching"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("agg_gpu_memory_utilization", 0.1),
        ("agg_block_size", 2),
        ("agg_timing_model", {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}),
    ],
)
def test_sdk_afd_unsupported_runtime_inputs_are_rejected(field, value):
    from aisimulate.sweeper.config import SmartSearchConfig

    payload = recommendation_to_sweeper(
        CoreRecommendationConfig.model_validate(recommendation("aggregated", ("aggregated",)))
    ).model_dump(mode="json")
    payload["search_space"][field] = value
    with pytest.raises(ValueError, match="AFD"):
        SmartSearchConfig.model_validate(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_size", 2),
        ("max_num_seqs", 10000),
        ("aic_tp_size", 2),
        ("gpu_memory_utilization", 0.1),
        ("unknown_option", True),
        ("timing_model", {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}),
        ("aic_nextn", 2),
    ],
)
def test_raw_afd_runtime_inputs_cannot_disappear(field, value):
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(prediction("aggregated", ("aggregated",))))
    spec.backend_deployment.agg_engine_args[field] = value
    with pytest.raises(ValueError, match="AFD"):
        _materialize_engine_execution_spec(spec, trace_block_size=1, record_per_request=False)


def test_common_afd_timing_and_search_domains_survive_saved_prediction():
    raw = recommendation("aggregated", ("aggregated",))
    worker = raw["engine"]["workers"]["aggregated"]
    worker["afd"].pop("tp_a")
    worker.update(
        parallelism={"preset": False, "tensor": {"choices": [1, 2]}},
        scheduler={"max_sequences": {"range": {"min": 1, "max": 4, "scale": "log"}}},
        timing={"database_mode": "SOL", "gemm_quant_mode": "bfloat16"},
    )
    source = CoreRecommendationConfig.model_validate(raw)
    smart = recommendation_to_sweeper(source)
    smart = type(smart).model_validate_json(smart.model_dump_json())
    (branch,) = enumerate_branches(smart, max_seq_len=2048)
    assert {t.a_tp_size for t in branch.parallel_configs} == {1, 2}
    assert branch.integer_ranges["agg_max_num_seqs"] == (1, 4)
    suggestion = RandomBranchSampler(branch, seed=7).suggest(1)[0]
    sample = unroll_sample(
        search_space=smart.search_space, selection=suggestion.selection, parallel_config=suggestion.parallel_config
    )
    deployment = build_backend_deployment(sample, backend_version="0.5.14")
    meta = deployment.performance_model_metadata["aggregated"]
    assert meta["estimator"]["database_mode"] == "SOL"
    assert meta["quant"]["gemm_quant_mode"] == "bfloat16"
    saved = _candidate_prediction(
        source,
        sample,
        ReplaySpec(backend_deployment=deployment, workload=smart.workload.model_dump(), goal={}),
        adapter_sections={},
    )
    restored = prediction_to_replay_spec(CorePredictionConfig.model_validate(saved))
    restored_meta = restored.backend_deployment.performance_model_metadata["aggregated"]
    assert restored_meta["quant"] == meta["quant"]
    assert restored_meta["estimator"]["database_mode"] == "SOL"


def test_afd_role_kv_precision_override_is_used_for_pd_transfer():
    raw = prediction()
    raw["engine"]["kv_transfer"] = {}
    raw["engine"]["kvcache_quant_mode"] = "bfloat16"
    base = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    raw["engine"]["workers"]["prefill"]["timing"] = {"kvcache_quant_mode": "fp8"}
    overridden = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    assert (
        overridden.backend_deployment.prefill_engine_args["kv_transfer_bytes_per_token"] * 2
        == (base.backend_deployment.prefill_engine_args["kv_transfer_bytes_per_token"])
    )


@pytest.mark.parametrize("dp_size", [2, 8])
def test_native_afd_rejects_outer_dp(dp_size):
    import json

    from aisimulate import _runtime

    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(prediction("aggregated", ("aggregated",))))
    execution = _materialize_engine_execution_spec(spec, trace_block_size=1, record_per_request=False)
    execution["spec"]["engine"]["dp_size"] = dp_size
    with pytest.raises(RuntimeError, match="one logical scheduler"):
        _runtime.run_replay_json(json.dumps(execution))


def test_flat_afd_aliases_validate_topology_and_keep_common_runtime_args():
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(prediction("aggregated", ("aggregated",))))
    spec.backend_deployment.agg_engine_args.update(
        aic_tp_size=1,
        aic_attention_dp_size=8,
        num_gpu_blocks=64,
        max_num_seqs=1,
        enable_prefix_caching=False,
        sglang={"chunked_prefill_size": 8, "max_prefill_tokens": 16},
    )
    execution = _materialize_engine_execution_spec(spec, trace_block_size=1, record_per_request=False)
    engine = execution["spec"]["engine"]
    assert engine.get("dp_size", 1) == 1
    assert engine["rank"]["num_gpu_blocks"] == 64
    assert engine["rank"]["max_num_seqs"] == 1
    assert engine["rank"]["enable_prefix_caching"] is False
    assert engine["rank"]["sglang"] == {"chunked_prefill_size": 8, "max_prefill_tokens": 16}


def test_sdk_afd_compact_integer_ranges_are_normalized_without_role_knobs():
    from aisimulate.sweeper.config import SmartSearchConfig

    payload = recommendation_to_sweeper(
        CoreRecommendationConfig.model_validate(recommendation("aggregated", ("aggregated",)))
    ).model_dump(mode="json")
    payload["search_space"]["engine_integer_log_ranges"] = {"agg_max_num_seqs": [1, 4]}
    config = SmartSearchConfig.model_validate(payload)
    config = SmartSearchConfig.model_validate_json(config.model_dump_json())
    (branch,) = enumerate_branches(config, max_seq_len=2048)
    assert branch.integer_ranges["agg_max_num_seqs"] == (1, 4)
    payload["search_space"]["engine_float_ranges"] = {"agg_gpu_memory_utilization": [0.1, 0.9]}
    with pytest.raises(ValueError, match="AFD engine_float_ranges"):
        SmartSearchConfig.model_validate(payload)
