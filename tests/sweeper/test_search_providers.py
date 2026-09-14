# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for adapter search-space and ReplaySpec materialization."""

from pathlib import Path

import pytest

import aisimulate.sweeper.search as search_module
from aisimulate.sweeper.config import SmartSearchConfig
from aisimulate.sweeper.parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.provider import (
    AdapterReplaySpec,
    AdapterSearchPlan,
    ConditionalSearchSpace,
    InfeasibleCandidate,
    RuntimeHookSpec,
    SearchSpaceFragment,
)
from aisimulate.sweeper.replay import HookCapability, ReplayReport, RunnerCapabilities
from aisimulate.sweeper.result import ReasonCategory
from aisimulate.sweeper.sampler import Suggestion
from aisimulate.sweeper.search_space import BranchSpace

TRACE = str(Path(__file__).parent / "data" / "mooncake_tiny.jsonl")
_HOOK = RuntimeHookSpec(
    provider="test.feature",
    kind="policy",
    api_version=1,
)


def _config() -> SmartSearchConfig:
    return SmartSearchConfig(
        search_space={
            "model_name": "model",
            "hardware_sku": "h200_sxm",
            "backend": ["vllm"],
            "deployment_mode": ["agg"],
        },
        adapters={
            "test.feature": {
                "search_space": {"modes": ["fast"]},
            }
        },
        workload={"trace_path": TRACE},
        sweep={"max_rounds": 1, "candidates_per_round": 1, "parallel_evals": 1},
    )


def _run_sweep(
    config: SmartSearchConfig,
    *,
    runner_factory,
    providers,
    sampler_factory,
    show_progress: bool,
):
    return (
        search_module.Sweeper(
            runner_factory=runner_factory,
            providers=providers,
            sampler_factory=sampler_factory,
            show_progress=show_progress,
        )
        .run(config, top_n=None)
        .selected_candidates
    )


class _Adapter:
    name = "test.feature"
    api_version = 1

    def __init__(self) -> None:
        self.generated = []
        self.materialized = []

    def generate_search_space(self, search_spec, context):
        self.generated.append((search_spec, context))
        return AdapterSearchPlan(
            fragment=SearchSpaceFragment(choices_by_branch={"agg": {"mode": list(search_spec["modes"])}}),
            potential_runtime_hooks=(_HOOK,),
        )

    def materialize_replay(self, plan, selection, context):
        self.materialized.append((plan, selection, context))
        hook = RuntimeHookSpec(
            provider=_HOOK.provider,
            kind=_HOOK.kind,
            api_version=_HOOK.api_version,
            config={"mode": selection["mode"]},
        )
        return AdapterReplaySpec(
            config={"mode": selection["mode"]},
            runtime_hooks=(hook,),
        )


class _MutatingAdapter(_Adapter):
    def materialize_replay(self, plan, selection, context):
        context.sample["backend"] = "mutated"
        context.backend_deployment.agg_engine_args["max_num_seqs"] = 1
        return super().materialize_replay(plan, selection, context)


class _SharedOutputAdapter(_Adapter):
    def __init__(self) -> None:
        super().__init__()
        self.shared = {}

    def generate_search_space(self, search_spec, context):
        del search_spec, context
        return AdapterSearchPlan(
            fragment=SearchSpaceFragment(choices_by_branch={"agg": {"mode": ["first", "second"]}}),
            potential_runtime_hooks=(_HOOK,),
        )

    def materialize_replay(self, plan, selection, context):
        del plan, context
        self.shared["mode"] = selection["mode"]
        return AdapterReplaySpec(
            config=self.shared,
            runtime_hooks=(
                RuntimeHookSpec(
                    provider=_HOOK.provider,
                    kind=_HOOK.kind,
                    api_version=_HOOK.api_version,
                    config=self.shared,
                ),
            ),
        )


class _Sampler:
    def __init__(self, branch, study_id, objectives=None):
        del study_id, objectives
        self.branch = branch
        assert branch.knob_choices["adapter::test.feature::mode"] == ["fast"]

    def suggest(self, count):
        assert count == 1
        selection = {
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 256,
            "adapter::test.feature::mode": "fast",
        }
        return [
            Suggestion(
                selection=selection,
                parallel_config=self.branch.parallel_configs[0],
                handle=selection,
            )
        ]

    def observe(self, suggestion, metrics):
        del suggestion, metrics

    def observe_infeasible(self, suggestion, reason):
        pytest.fail(f"unexpected infeasible suggestion {suggestion}: {reason}")


class _TwoCandidateSampler(_Sampler):
    def __init__(self, branch, study_id, objectives=None):
        del study_id, objectives
        self.branch = branch
        assert branch.knob_choices["adapter::test.feature::mode"] == [
            "first",
            "second",
        ]

    def suggest(self, count):
        assert count == 2
        return [
            Suggestion(
                selection={
                    "deployment_mode": "agg",
                    "backend": "vllm",
                    "agg_max_num_batched_tokens": 8192,
                    "agg_max_num_seqs": 256,
                    "adapter::test.feature::mode": mode,
                },
                parallel_config=self.branch.parallel_configs[0],
                handle=mode,
            )
            for mode in ("first", "second")
        ]


class _Runner:
    def __init__(self) -> None:
        self.specs = []
        self.closed = False

    def run(self, spec):
        self.specs.append(spec)
        return ReplayReport(metrics={"output_throughput_tok_s": 12.0})

    def close(self):
        self.closed = True


class _RunnerFactory:
    def __init__(self, *, support_hook: bool = True) -> None:
        self.runner = _Runner()
        self.support_hook = support_hook
        self.created = 0

    def capabilities(self):
        hooks = (HookCapability("test.feature", "policy", 1),) if self.support_hook else ()
        return RunnerCapabilities(
            supported_backend_topologies=(("*", "*"),),
            supported_hooks=hooks,
        )

    def create(self, worker_id):
        assert worker_id == 0
        self.created += 1
        return self.runner


def _stub_branch(monkeypatch) -> None:
    parallel = ReplicaParallelConfig(
        shape=ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1),
        replicas=1,
    )
    branch = BranchSpace(
        deployment_mode="agg",
        parallel_configs=(parallel,),
        supported_backends={parallel: frozenset({"vllm"})},
        knob_choices={
            "backend": ["vllm"],
            "agg_max_num_batched_tokens": [8192],
            "agg_max_num_seqs": [256],
        },
    )
    monkeypatch.setattr(
        search_module,
        "enumerate_branches",
        lambda config, *, max_seq_len=None, runner_capabilities=None: [branch],
    )
    monkeypatch.setattr(
        search_module,
        "resolve_backend_version",
        lambda hardware, backend, systems_path=None: "0.11.0",
    )


def test_conditional_adapter_fragment_is_validated_namespaced_and_merged() -> None:
    parallel = ReplicaParallelConfig(
        shape=ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1),
        replicas=1,
    )
    branch = BranchSpace(
        deployment_mode="agg",
        parallel_configs=(parallel,),
        supported_backends={parallel: frozenset({"vllm"})},
        knob_choices={"backend": ["vllm"]},
    )
    fragment = SearchSpaceFragment(
        choices_by_branch={"agg": {"mode": ["round_robin", "kv_router"]}},
        conditional_by_branch={
            "agg": [
                ConditionalSearchSpace(
                    selector="mode",
                    values=["kv_router"],
                    choices={"load_model": ["none", "aic"]},
                    float_ranges={"kv_weight": (0.01, 10.0)},
                    log_float_ranges=["kv_weight"],
                )
            ]
        },
    )

    search_module._validate_search_fragment(fragment)
    merged = search_module._merge_adapter_spaces(
        [branch],
        {"test.feature": AdapterSearchPlan(fragment=fragment)},
    )[0]

    assert merged.knob_choices["adapter::test.feature::mode"] == [
        "round_robin",
        "kv_router",
    ]
    assert "adapter::test.feature::kv_weight" not in merged.float_ranges
    assert merged.conditional_dimensions == (
        search_module.ConditionalDimensionSpace(
            selector="adapter::test.feature::mode",
            values=("kv_router",),
            knob_choices={"adapter::test.feature::load_model": ["none", "aic"]},
            float_ranges={"adapter::test.feature::kv_weight": (0.01, 10.0)},
            log_float_ranges=frozenset({"adapter::test.feature::kv_weight"}),
        ),
    )


def test_search_space_fragment_preserves_legacy_positional_field_order() -> None:
    fragment = SearchSpaceFragment(
        {"agg": {"mode": ["one"]}},
        {"agg": {"weight": (0.1, 1.0)}},
        {"agg": ["weight"]},
        {"agg": []},
    )

    assert fragment.choices_by_branch == {"agg": {"mode": ["one"]}}
    assert fragment.float_ranges_by_branch == {"agg": {"weight": (0.1, 1.0)}}
    assert fragment.api_version == 1
    search_module._validate_search_fragment(fragment)


def test_conditional_adapter_fragment_rejects_non_root_selector() -> None:
    fragment = SearchSpaceFragment(
        conditional_by_branch={
            "agg": [
                ConditionalSearchSpace(
                    selector="mode",
                    values=["kv_router"],
                    float_ranges={"kv_weight": (0.01, 10.0)},
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="is not a root categorical parameter"):
        search_module._validate_search_fragment(fragment)


def test_search_fragment_rejects_unknown_api_version() -> None:
    with pytest.raises(ValueError, match="requires version 1"):
        search_module._validate_search_fragment(SearchSpaceFragment(api_version=2))


def test_adapter_accepts_search_space_and_materializes_spec_on_main(
    monkeypatch,
) -> None:
    _stub_branch(monkeypatch)
    adapter = _Adapter()
    factory = _RunnerFactory()

    candidates = _run_sweep(
        _config(),
        runner_factory=factory,
        providers={"test.feature": adapter},
        sampler_factory=_Sampler,
        show_progress=False,
    )

    assert adapter.generated[0][0] == {"modes": ["fast"]}
    assert adapter.materialized[0][1] == {"mode": "fast"}
    assert factory.created == 1
    assert factory.runner.closed
    spec = factory.runner.specs[0]
    assert spec.adapters["test.feature"].config == {"mode": "fast"}
    assert spec.runtime_hooks[0].config == {"mode": "fast"}
    assert candidates[0].config["adapters"] == {"test.feature": {"mode": "fast"}}


def test_adapter_infeasible_selection_is_gated_before_replay(monkeypatch) -> None:
    config = _config()
    parallel = ReplicaParallelConfig(
        shape=ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1),
        replicas=1,
    )

    class InfeasibleAdapter(_Adapter):
        def materialize_replay(self, plan, selection, context):
            del plan, selection, context
            raise InfeasibleCandidate("invalid correlated leaves")

    monkeypatch.setattr(
        search_module,
        "resolve_backend_version",
        lambda hardware, backend, systems_path=None: "0.11.0",
    )
    prepared, result = search_module._materialize_one(
        {
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 256,
            "adapter::test.feature::mode": "fast",
        },
        parallel,
        config=config,
        goal=config.goal,
        providers={"test.feature": InfeasibleAdapter()},
        provider_plans={"test.feature": AdapterSearchPlan()},
        runner_factory=_RunnerFactory(),
    )

    assert prepared is None
    assert result is not None
    assert result.outcome == "infeasible"
    assert result.reason_category is ReasonCategory.ADAPTER_CONSTRAINT
    assert "invalid correlated leaves" in result.reason


@pytest.mark.parametrize(
    ("mode", "prefill", "decode", "load_model", "system", "accepted"),
    [
        ("disagg", "gb200", "h200_sxm", "aic", "h200_sxm", False),
        ("disagg", "gb200", "gb200", "aic", "h200_sxm", False),
        ("disagg", "gb200", "h200_sxm", "aic", None, False),
        ("disagg", "gb200", "h200_sxm", "aic", "gb200", True),
        ("disagg", "gb200", "gb200", "aic", "gb200", True),
        ("disagg", "h200_sxm", "gb200", "aic", "h200_sxm", True),
        ("disagg", None, None, "aic", "h200_sxm", True),
        ("disagg", "gb200", "h200_sxm", "none", None, True),
        ("agg", "gb200", "gb200", "aic", "h200_sxm", True),
    ],
)
def test_router_aic_role_hardware_is_checked_before_replay(mode, prefill, decode, load_model, system, accepted):
    config = _config()
    config.search_space.deployment_mode = ["agg", "disagg"]
    config.search_space.prefill_hardware_sku = prefill
    config.search_space.decode_hardware_sku = decode
    config.search_space.backend_version = "0.24.0"
    role = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    parallel = DisaggParallelConfig(prefill=role, decode=role) if mode == "disagg" else role
    hook = RuntimeHookSpec(
        provider="dynamo.router",
        kind="placement_policy",
        api_version=1,
        config={
            "router_config": {"router_prefill_load_model": load_model},
            "aic_perf_config": {"aic_system": system} if system is not None else None,
        },
    )

    class RouterAdapter(_Adapter):
        def materialize_replay(self, plan, selection, context):
            assert context.sample["hardware_sku"] == "h200_sxm"
            return AdapterReplaySpec(runtime_hooks=(hook,))

    class RouterRunnerFactory(_RunnerFactory):
        def capabilities(self):
            return RunnerCapabilities(
                supported_backend_topologies=(("*", "*"),),
                supported_hooks=(HookCapability("dynamo.router", "placement_policy", 1),),
            )

    selection = {"deployment_mode": mode, "backend": "vllm"}
    for engine_role in ("prefill", "decode") if mode == "disagg" else ("agg",):
        selection[f"{engine_role}_max_num_batched_tokens"] = 8192
        selection[f"{engine_role}_max_num_seqs"] = 256
    factory = RouterRunnerFactory()
    prepared, result = search_module._materialize_one(
        selection,
        parallel,
        config=config,
        goal=config.goal,
        providers={"test.feature": RouterAdapter()},
        provider_plans={"test.feature": AdapterSearchPlan()},
        runner_factory=factory,
    )

    if accepted:
        assert result is None
        assert prepared is not None
        assert prepared.replay_spec.runtime_hooks == (hook,)
        engine = (
            prepared.replay_spec.backend_deployment.prefill_engine_args
            if mode == "disagg"
            else prepared.replay_spec.backend_deployment.agg_engine_args
        )
        expected = (prefill or "h200_sxm") if mode == "disagg" else "h200_sxm"
        assert engine["aic_system"] == expected
    else:
        assert prepared is None
        assert result.outcome == "infeasible"
        assert result.reason_category is ReasonCategory.ADAPTER_CONSTRAINT
        assert "does not match effective prefill_hardware_sku='gb200'" in result.reason
        assert factory.runner.specs == []


def test_runner_hook_capability_is_checked_before_runner_creation(monkeypatch) -> None:
    _stub_branch(monkeypatch)
    factory = _RunnerFactory(support_hook=False)

    with pytest.raises(ValueError, match="unsupported runtime hook"):
        _run_sweep(
            _config(),
            runner_factory=factory,
            providers={"test.feature": _Adapter()},
            sampler_factory=_Sampler,
            show_progress=False,
        )

    assert factory.created == 0


@pytest.mark.parametrize("decode", ["h200_sxm", "gb200"])
@pytest.mark.parametrize(("system", "accepted"), [("h200_sxm", False), ("gb200", True)])
def test_sweep_submits_only_router_aic_hooks_matching_prefill_hardware(monkeypatch, decode, system, accepted):
    config = _config()
    config.search_space.deployment_mode = ["disagg"]
    config.search_space.prefill_hardware_sku = "gb200"
    config.search_space.decode_hardware_sku = decode
    config.search_space.backend_version = "0.24.0"
    config.sweep.max_trials = 1
    config.sweep.max_eval_seconds = None  # Exercise the sequential runner in this process.
    role = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    parallel = DisaggParallelConfig(prefill=role, decode=role)
    branch = BranchSpace(
        deployment_mode="disagg",
        parallel_configs=(parallel,),
        supported_backends={parallel: frozenset({"vllm"})},
        knob_choices={
            "backend": ["vllm"],
            "prefill_max_num_batched_tokens": [8192],
            "prefill_max_num_seqs": [1],
            "decode_max_num_batched_tokens": [8192],
            "decode_max_num_seqs": [256],
        },
    )
    monkeypatch.setattr(search_module, "enumerate_branches", lambda *args, **kwargs: [branch])
    hook = RuntimeHookSpec(
        provider="dynamo.router",
        kind="placement_policy",
        api_version=1,
        config={
            "router_config": {"router_prefill_load_model": "aic"},
            "aic_perf_config": {"aic_system": system},
        },
    )

    class RouterAdapter(_Adapter):
        def generate_search_space(self, search_spec, context):
            return AdapterSearchPlan(
                fragment=SearchSpaceFragment(choices_by_branch={"disagg": {"mode": ["fast"]}}),
                potential_runtime_hooks=(hook,),
            )

        def materialize_replay(self, plan, selection, context):
            self.materialized.append(context)
            return AdapterReplaySpec(runtime_hooks=(hook,))

    rejected = []

    class Sampler(_Sampler):
        def __init__(self, branch, study_id, objectives=None, **kwargs):
            super().__init__(branch, study_id, objectives)

        def suggest(self, count):
            assert count == 1
            selection = {
                "deployment_mode": "disagg",
                **{name: values[0] for name, values in self.branch.knob_choices.items()},
            }
            return [Suggestion(selection=selection, parallel_config=parallel, handle=selection)]

        def observe_infeasible(self, suggestion, reason):
            rejected.append(reason)

    class RouterRunnerFactory(_RunnerFactory):
        def capabilities(self):
            return RunnerCapabilities(
                supported_backend_topologies=(("*", "*"),),
                supported_hooks=(HookCapability("dynamo.router", "placement_policy", 1),),
            )

    adapter = RouterAdapter()
    factory = RouterRunnerFactory()
    result = search_module.Sweeper(
        runner_factory=factory,
        providers={"test.feature": adapter},
        sampler_factory=Sampler,
        show_progress=False,
    ).run(config, top_n=None)

    assert len(adapter.materialized) == 1
    if accepted:
        assert not rejected
        assert len(result.selected_candidates) == 1
        assert len(factory.runner.specs) == 1
        assert factory.runner.specs[0].runtime_hooks == (hook,)
    else:
        assert not result.selected_candidates
        assert factory.runner.specs == []
        assert len(rejected) == 1
        assert "does not match effective prefill_hardware_sku='gb200'" in rejected[0]
        assert result.counts.infeasible == 1
        assert result.candidates[0].reason_category is ReasonCategory.ADAPTER_CONSTRAINT


def test_core_branch_preflight_runs_before_adapter_preparation(monkeypatch) -> None:
    adapter = _Adapter()

    def reject_branches(*args, **kwargs):
        del args, kwargs
        raise ValueError("no viable backend/topology branch")

    monkeypatch.setattr(search_module, "enumerate_branches", reject_branches)

    with pytest.raises(ValueError, match="no viable backend/topology branch"):
        _run_sweep(
            _config(),
            runner_factory=_RunnerFactory(),
            providers={"test.feature": adapter},
            sampler_factory=_Sampler,
            show_progress=False,
        )

    assert adapter.generated == []


def test_adapter_candidate_context_is_isolated_from_core_candidate(monkeypatch) -> None:
    _stub_branch(monkeypatch)
    factory = _RunnerFactory()

    candidates = _run_sweep(
        _config(),
        runner_factory=factory,
        providers={"test.feature": _MutatingAdapter()},
        sampler_factory=_Sampler,
        show_progress=False,
    )

    spec = factory.runner.specs[0]
    assert spec.backend_deployment.backend == "vllm"
    assert spec.backend_deployment.agg_engine_args["max_num_seqs"] == 256
    assert candidates[0].config["backend"] == "vllm"


def test_adapter_reused_output_buffer_is_isolated_per_candidate(monkeypatch) -> None:
    _stub_branch(monkeypatch)
    config_data = _config().model_dump(mode="python")
    config_data["sweep"]["candidates_per_round"] = 2
    config = SmartSearchConfig.model_validate(config_data)
    factory = _RunnerFactory()

    candidates = _run_sweep(
        config,
        runner_factory=factory,
        providers={"test.feature": _SharedOutputAdapter()},
        sampler_factory=_TwoCandidateSampler,
        show_progress=False,
    )

    replay_modes = [spec.adapters["test.feature"].config["mode"] for spec in factory.runner.specs]
    hook_modes = [spec.runtime_hooks[0].config["mode"] for spec in factory.runner.specs]
    candidate_modes = [candidate.config["adapters"]["test.feature"]["mode"] for candidate in candidates]
    assert replay_modes == ["first", "second"]
    assert hook_modes == ["first", "second"]
    assert candidate_modes == ["first", "second"]


def test_adapter_search_contexts_are_isolated() -> None:
    config_data = _config().model_dump(mode="python")
    config_data["adapters"] = {
        "mutator": {"search_space": {}},
        "observer": {"search_space": {}},
    }
    config = SmartSearchConfig.model_validate(config_data)
    observed_backends = []

    class Mutator:
        name = "mutator"
        api_version = 1

        def generate_search_space(self, search_spec, context):
            context.core_search_space["backend"].append("mutated")
            return AdapterSearchPlan()

        def materialize_replay(self, plan, selection, context):
            return AdapterReplaySpec()

    class Observer:
        name = "observer"
        api_version = 1

        def generate_search_space(self, search_spec, context):
            observed_backends.extend(context.core_search_space["backend"])
            return AdapterSearchPlan()

        def materialize_replay(self, plan, selection, context):
            return AdapterReplaySpec()

    search_module._prepare_providers(
        config,
        injected={"mutator": Mutator(), "observer": Observer()},
        show_progress=False,
    )

    assert observed_backends == ["vllm"]


def test_adapter_contract_rejects_non_json_values_before_worker_submission() -> None:
    with pytest.raises(TypeError, match="test.feature.*non-JSON replay spec"):
        search_module._validate_provider_replay_spec(
            "test.feature",
            AdapterReplaySpec(config={"invalid": object()}),
        )

    with pytest.raises(TypeError, match="test.feature.*non-JSON search plan"):
        search_module._validate_search_plan(
            "test.feature",
            AdapterSearchPlan(state={"invalid": object()}),
        )


@pytest.mark.parametrize(
    "spec",
    [
        AdapterReplaySpec(config=[]),
        AdapterReplaySpec(runtime_hooks=[]),
        AdapterReplaySpec(runtime_hooks=(RuntimeHookSpec(provider="", kind="policy", api_version=1),)),
        AdapterReplaySpec(runtime_hooks=(RuntimeHookSpec(provider="test", kind="policy", api_version=True),)),
    ],
)
def test_adapter_replay_contract_rejects_invalid_field_shapes(spec) -> None:
    with pytest.raises(TypeError, match="invalid/non-JSON replay spec"):
        search_module._validate_provider_replay_spec("test.feature", spec)


@pytest.mark.parametrize(
    "plan",
    [
        AdapterSearchPlan(diagnostics=[]),
        AdapterSearchPlan(potential_runtime_hooks=[]),
        AdapterSearchPlan(fragment=SearchSpaceFragment(choices_by_branch={"agg": {"mode": (1,)}})),
        AdapterSearchPlan(
            fragment=SearchSpaceFragment(float_ranges_by_branch={"agg": {"weight": (0.0, float("inf"))}})
        ),
    ],
)
def test_adapter_search_contract_rejects_invalid_field_shapes(plan) -> None:
    with pytest.raises(TypeError, match="invalid/non-JSON search plan"):
        search_module._validate_search_plan("test.feature", plan)


def test_adapter_parameter_separator_collisions_are_rejected() -> None:
    branch = BranchSpace(
        deployment_mode="agg",
        parallel_configs=(),
        supported_backends={},
        knob_choices={},
    )
    plan = AdapterSearchPlan(fragment=SearchSpaceFragment(choices_by_branch={"agg": {"ambiguous::parameter": [1]}}))

    with pytest.raises(ValueError, match="reserved separator"):
        search_module._merge_adapter_spaces([branch], {"test.feature": plan})

    config_data = _config().model_dump(mode="python")
    config_data["adapters"] = {"ambiguous::adapter": {"search_space": {}}}
    config = SmartSearchConfig.model_validate(config_data)
    with pytest.raises(ValueError, match="adapter names.*reserved separator"):
        search_module._prepare_providers(
            config,
            injected={"ambiguous::adapter": _Adapter()},
            show_progress=False,
        )
