# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analytical EPD integration, tested against the shared in-tree AIC implementation."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import aisimulate.sweeper.search as search_mod
from aiconfigurator.sdk.sweep import _overlay_encoder_stage
from aisimulate.runner import EngineReplayRunnerFactory, InvalidRunnerError
from aisimulate.sweeper import (
    BackendDeploymentSpec,
    EncoderPoolSpec,
    EncoderSearch,
    ReplayOutputRequirements,
    ReplayReport,
    ReplaySpec,
    RunnerCapabilities,
    SmartSearchConfig,
    Sweeper,
    SweepResult,
)
from aisimulate.sweeper.epd import apply_encoder_overlay, resolve_encoder_catalog
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.sampler import Suggestion
from aisimulate.sweeper.search_space import BranchSpace


def _config(**kwargs):
    payload = dict(
        search_space=dict(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            hardware_sku="h200_sxm",
            backend=["sglang"],
            deployment_mode=["agg"],
            gpu_budget=8,
            encoder=dict(tp=[1], batch_size=[1], workers=[1, 2]),
        ),
        workload=dict(
            isl=128,
            osl=32,
            concurrency=8,
            num_request_ratio=2,
            images=dict(height=448, width=448, count=1),
        ),
        goal=dict(target="throughput_per_gpu"),
        sweep=dict(
            max_rounds=1,
            parallel_evals=1,
            candidates_per_round=2,
            max_trials=2,
            max_eval_seconds=None,
        ),
    )
    payload.update(kwargs)
    return SmartSearchConfig.model_validate(payload)


def _encoder(**kwargs):
    values = dict(
        model="Qwen/Qwen3-VL-8B-Instruct",
        system="h200_sxm",
        backend="sglang",
        backend_version="0.5.14",
        tp=1,
        batch_size=1,
        workers=1,
        latency_ms=50.0,
        throughput_rps=80.0,
        memory_gib=2.0,
        rate_degradation=0.9,
        visual_tokens=196,
        image_height=448,
        image_width=448,
        image_count=1,
    )
    return EncoderPoolSpec(**(values | kwargs))


def _report():
    return ReplayReport(
        metrics=dict(
            completed_requests=100.0,
            duration_ms=1000.0,
            output_throughput_tok_s=3200.0,
            mean_ttft_ms=60.0,
            mean_tpot_ms=8.0,
            mean_e2e_latency_ms=308.0,
            num_ttft_samples=100.0,
            num_tpot_samples=100.0,
            num_e2e_latency_samples=100.0,
            mean_output_token_throughput_per_user=125.0,
            goodput_output_throughput_tok_s=3000.0,
            p99_ttft_ms=99.0,
        ),
        metadata={"native_report": {"language_only": True}},
    )


def _spec(mode="agg", **encoder_kwargs):
    pc = {"tp": 2, "pp": 1, "attention_dp": 1}
    if mode == "disagg":
        pc = {f"{role}_{name}": value for role in ("prefill", "decode") for name, value in pc.items()}
    return ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode=mode,
            backend="sglang",
            backend_version="0.5.14",
            agg_engine_args={"aic_model_path": _encoder().model},
            prefill_engine_args={"aic_model_path": _encoder().model},
            decode_engine_args={"aic_model_path": _encoder().model},
            parallel_config=pc,
            num_workers=1,
            num_prefill_workers=1,
            num_decode_workers=2,
            encoder=_encoder(**encoder_kwargs),
        ),
        workload=_config().workload.model_dump(mode="json"),
        goal={"target": "throughput_per_gpu"},
    )


@pytest.mark.parametrize("mode,language_gpus", [("agg", 2), ("disagg", 6)])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_overlay_differential_against_aic(mode, language_gpus, workers):
    spec = _spec(mode, workers=workers)
    actual = apply_encoder_overlay(_report(), spec)
    expected = _overlay_encoder_stage(
        {
            "seq/s": 100.0,
            "tokens/s": 3200.0,
            "ttft": 60.0,
            "tpot": 8.0,
            "request_latency": 308.0,
            "osl": 32,
            "num_total_gpus": language_gpus,
        },
        {
            "seq/s": 80.0,
            "encoder_latency": 50.0,
            "num_total_gpus": 1,
            "tp": 1,
            "bs": 1,
            "memory": 2.0,
        },
        workers,
        encoder_degradation=0.9,
    )
    assert actual.metrics["output_throughput_tok_s"] == expected["tokens/s"]
    assert actual.metrics["mean_ttft_ms"] == expected["ttft"]
    assert actual.metrics["mean_e2e_latency_ms"] == expected["request_latency"]
    assert actual.metrics["mean_tpot_ms"] == expected["tpot"]
    assert (
        actual.metrics["mean_output_token_throughput_per_user"]
        == _report().metrics["mean_output_token_throughput_per_user"]
    )
    assert actual.metrics["completed_requests"] == _report().metrics["completed_requests"]
    avg_gpus = actual.metrics["gpu_hours"] / (actual.metrics["duration_ms"] / 3_600_000)
    assert avg_gpus == pytest.approx(expected["num_total_gpus"])
    assert actual.metrics["output_throughput_tok_s"] / avg_gpus == pytest.approx(expected["tokens/s/gpu"])
    assert actual.metadata["metric_semantics"] == "analytical_epd_overlay"
    assert "goodput_output_throughput_tok_s" not in actual.metrics
    assert "p99_ttft_ms" not in actual.metrics
    assert "native_report" not in actual.metadata
    assert "encoder_power_w" not in actual.metrics
    assert actual.metadata["encoder"]["power_w"] is None


def test_positive_power_is_encoder_only():
    report = apply_encoder_overlay(_report(), _spec(power_w=200.0, power_coverage=0.75))
    assert report.metrics["encoder_power_w"] == 200
    assert report.metrics["encoder_power_coverage"] == 0.75
    assert "power_w" not in report.metrics


@pytest.mark.parametrize(
    "field,value",
    [("tp", [True]), ("workers", []), ("batch_size", [9]), ("workers", [0])],
)
def test_invalid_encoder_domains(field, value):
    with pytest.raises(ValueError):
        EncoderSearch.model_validate({field: value})


def test_encoder_batch_size_upper_boundary():
    assert EncoderSearch.model_validate({"batch_size": [8]}).batch_size == [8]


def test_documented_example_validates():
    example = Path(__file__).resolve().parents[2] / "examples" / "sweeper" / "epd.yaml"
    config = SmartSearchConfig.from_yaml(example)
    assert config.workload.images.count == 1
    assert config.search_space.encoder.batch_size == [1, 2, 4]


@pytest.mark.parametrize(
    "updates",
    [
        {"images": None},
        {"random_range_ratio": 0.5},
        {"turns_per_session": 2},
        {"shared_prefix_ratio": 0.5},
        {"num_prefix_groups": 2},
        {"inter_turn_delay_ms": 1.0},
        {"max_sim_time_ms": 100.0},
    ],
)
def test_unsupported_workloads_fail_closed(updates):
    workload = _config().workload.model_dump(mode="json") | updates
    with pytest.raises(ValueError):
        _config(workload=workload)


def test_goodput_and_implicit_sla_are_not_claimed():
    for goal in (
        {"target": "goodput", "sla": {"ttft_ms": 500.0}},
        {"target": "throughput", "sla": {"ttft_ms": 500.0}},
    ):
        with pytest.raises(ValueError, match="goodput"):
            _config(goal=goal)
    _config(goal={"target": "throughput", "sla": {"ttft_ms": 500.0}, "strict_sla": True})


def test_runner_and_export_guards():
    with pytest.raises(ValueError, match="analytical EPD"):
        RunnerCapabilities(supported_backend_topologies=(("*", "*"),)).require_compatible(_spec())
    for outputs in (
        ReplayOutputRequirements(capture_per_request=True),
        ReplayOutputRequirements(include_raw_report=True),
    ):
        with pytest.raises(InvalidRunnerError, match="per-request"):
            EngineReplayRunnerFactory().create(0).run(_spec(), output_requirements=outputs)
    from aiconfigurator.generator.request.sweeper import (
        SweeperCandidateError,
        from_sweeper_candidate,
    )

    with pytest.raises(SweeperCandidateError, match="encoder pool"):
        from_sweeper_candidate({"config": {"encoder": asdict(_encoder())}}, workload={})


@pytest.mark.parametrize(
    "gpu_budget,failure",
    [(2, None), (8, None), (8, "build"), (8, "version"), (8, "kv"), (8, "backend"), (8, "projection")],
)
def test_complete_sweeper_selection_and_serialization(monkeypatch, gpu_budget, failure):
    config = _config(search_space=_config().search_space.model_dump() | {"gpu_budget": gpu_budget})
    parallel = ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    branch = BranchSpace(
        deployment_mode="agg",
        parallel_configs=(parallel,),
        supported_backends={parallel: frozenset({"vllm" if failure == "backend" else "sglang"})},
        knob_choices={"backend": ["sglang"]},
    )
    catalog = {"one": _encoder(), "two": _encoder(workers=2)}
    monkeypatch.setattr(search_mod, "enumerate_branches", lambda *a, **kw: [branch])
    monkeypatch.setattr(search_mod, "resolve_encoder_catalog", lambda c: catalog)
    monkeypatch.setattr(search_mod, "resolve_backend_version", lambda *a: "0.5.14")
    if failure in {"build", "version", "kv"}:

        def fail_build(*args, **kwargs):
            if failure == "kv":
                raise search_mod.InfeasibleKVCapacity("test capacity failure")
            raise ValueError("missing language performance data")

        function = {"build": "build_backend_deployment", "version": "resolve_backend_version", "kv": "resolve_kv_load"}[
            failure
        ]
        monkeypatch.setattr(search_mod, function, fail_build)

    class Sampler:
        def __init__(self, branch, **kwargs):
            assert branch.knob_choices["encoder_candidate"] == ["one", "two"]
            self.index = 0

        def suggest(self, count):
            suggestions = []
            for _ in range(count):
                key = ["one", "two"][self.index % 2]
                self.index += 1
                selection = dict(
                    deployment_mode="agg",
                    backend="sglang",
                    agg_max_num_batched_tokens=8192,
                    agg_max_num_seqs=256,
                    encoder_candidate=key,
                )
                if failure == "kv":
                    selection["kv_load_ratio"] = 0.5
                suggestions.append(
                    Suggestion(
                        selection=selection,
                        parallel_config=parallel,
                        handle=key,
                        infeasible_reason="test projection failure" if failure == "projection" else None,
                    )
                )
            return suggestions

        def observe(self, *args):
            pass

        def observe_infeasible(self, *args):
            pass

    class Factory:
        def capabilities(self):
            return RunnerCapabilities(
                supported_backend_topologies=(("sglang", "agg"),),
                supports_analytical_epd=True,
            )

        def create(self, worker_id):
            return self

        def run(self, spec):
            return apply_encoder_overlay(_report(), spec)

        def close(self):
            pass

    sweeper = Sweeper(runner_factory=Factory(), sampler_factory=Sampler, show_progress=False)
    result = sweeper.run(config)
    assert len(result.selected_candidates) == (2 if gpu_budget == 8 and failure is None else 0)
    assert {c.used_gpus for c in result.candidates} == {3, 4}
    for candidate in result.candidates:
        assert candidate.config["encoder"]["backend_version"] == "0.5.14"
        if failure in {"build", "kv"} or (failure is None and gpu_budget == 8):
            assert candidate.config["backend_version"] == "0.5.14"
        assert candidate.prediction_config is None
        assert candidate.config["deployment_artifact_generation_supported"] is False
        assert candidate.provenance.topology["encoder"]["image_height"] == 448
        assert candidate.provenance.topology["total_gpus"] == candidate.used_gpus
        assert any(
            p["role"] == "encoder" and p["backend_version"] == "0.5.14" for p in candidate.provenance.performance_data
        )
    assert SweepResult.from_json(result.to_json()).to_json() == result.to_json()
    if failure is None and gpu_budget == 8:
        invalid = Sweeper(
            runner_factory=Factory(),
            sampler_factory=Sampler,
            show_progress=False,
            prediction_config_factory=lambda *args: {},
        ).run(config)
        assert not invalid.selected_candidates
        assert all("prediction-ready" in candidate.reason for candidate in invalid.candidates)


def test_unresolved_parallel_snapshot_keeps_encoder_without_inventing_gpu_total():
    suggestion = Suggestion(
        selection={"deployment_mode": "agg", "encoder_candidate": "one"}, parallel_config=None, handle=None
    )
    snapshot = search_mod._suggestion_snapshot(suggestion, _config(), encoder_catalog={"one": _encoder()})
    assert snapshot["encoder"] == asdict(_encoder())
    assert snapshot["encoder_candidate"] == "one"
    assert snapshot["used_gpus"] is None
    assert snapshot["language_gpus"] is None
    assert snapshot["deployment_artifact_generation_supported"] is False


def test_catalog_uses_aic_geometry_memory_and_identity(monkeypatch):
    import aiconfigurator.sdk.sweep as aic_sweep
    from aiconfigurator_core.sdk import perf_database

    calls = []
    monkeypatch.setattr(
        perf_database, "get_database_view", lambda *a, **kw: calls.append(a) or SimpleNamespace(version="resolved")
    )
    monkeypatch.setattr("aisimulate.sweeper.epd.resolve_backend_version", lambda *args: "pinned")
    monkeypatch.setattr(
        aic_sweep,
        "_get_encoder_worker_candidates",
        lambda **kw: [
            {
                "encoder_latency": 10.0,
                "seq/s": 100.0,
                "num_total_gpus": 1,
                "tp": 1,
                "bs": 1,
                "memory": 2.0,
                "power_w": 0.0,
                "power_coverage": 0.0,
            }
        ],
    )
    catalog = resolve_encoder_catalog(_config())
    assert len(catalog) == 2
    assert calls == [("h200_sxm", "sglang", "pinned")]
    assert all(point.visual_tokens > 0 and point.backend_version == "resolved" for point in catalog.values())
    assert all(
        (point.image_height, point.image_width, point.image_count) == (448, 448, 1) for point in catalog.values()
    )
    assert all(point.power_w is None for point in catalog.values())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_invalid_metrics_do_not_become_epd_results(bad):
    for field in ("completed_requests", "duration_ms", "mean_ttft_ms"):
        values = _report().metrics | {field: bad}
        with pytest.raises(ValueError, match="finite"):
            apply_encoder_overlay(ReplayReport(values), _spec())


@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("timing_scope", ["flat", "nested"])
def test_runner_adds_visual_context_exactly_once(monkeypatch, mode, timing_scope):
    captured = []
    monkeypatch.setattr(
        "aisimulate.runner._materialize_engine_execution_spec",
        lambda spec, **kwargs: captured.append(spec.workload) or {},
    )
    runtime = SimpleNamespace(run_replay_json=lambda _: json.dumps(_report().metrics))
    spec = _spec(mode)
    args = {"aic_model_path": _encoder().model}
    args.update({"rank": {"forward_model": "op_level"}} if timing_scope == "nested" else {"forward_model": "op_level"})
    roles = ["agg"] if mode == "agg" else ["prefill", "decode"]
    spec = replace(
        spec, backend_deployment=replace(spec.backend_deployment, **{f"{role}_engine_args": args for role in roles})
    )
    result = EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    assert len(captured) == 1
    assert captured[0]["isl"] == 128 + 196
    assert spec.workload["isl"] == 128
    assert result.metrics["mean_ttft_ms"] == 110.0


@pytest.mark.parametrize(
    "updates",
    [
        {"images": {"height": 896, "width": 448, "count": 1}},
        {"images": {"height": 448, "width": 448, "count": 2}},
        {"random_range_ratio": 0.5},
        {"turns_per_session": 2},
        {"shared_prefix_ratio": 0.5},
        {"num_prefix_groups": 2},
        {"inter_turn_delay_ms": 1.0},
        {"max_sim_time_ms": 100.0},
        {"load_type": "concurrency"},
        {"trace_path": "unused.jsonl"},
    ],
)
def test_direct_runner_rejects_mismatched_or_unsupported_workload(updates):
    spec = _spec()
    with pytest.raises((ValueError, InvalidRunnerError)):
        EngineReplayRunnerFactory().create(0).run(replace(spec, workload=spec.workload | updates))


@pytest.mark.parametrize("role", ["agg", "prefill", "decode"])
@pytest.mark.parametrize(
    "updates",
    [
        {"aic_model_path": "another-model"},
        {"aic_forward_model": "fpm"},
        {"forward_model": "fpm"},
        {"timing_model": {"type": "fixed"}},
        {"startup_time": 1.0},
        {"rank": {"aic_forward_model": "fpm"}},
        {"rank": {"forward_model": "fpm"}},
        {"rank": {"timing_model": {"type": "fixed"}}},
        {"rank": {"startup_time": 1.0}},
    ],
)
def test_direct_runner_rejects_mismatched_language_estimates(updates, role):
    spec = _spec("agg" if role == "agg" else "disagg")
    key = f"{role}_engine_args"
    deployment = replace(spec.backend_deployment, **{key: getattr(spec.backend_deployment, key) | updates})
    with pytest.raises(InvalidRunnerError):
        EngineReplayRunnerFactory().create(0).run(replace(spec, backend_deployment=deployment))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"power_coverage": True},
        {"power_w": True, "power_coverage": 1},
        {"power_coverage": 0.5},
        {"power_w": 0.0},
        {"image_count": True},
    ],
)
def test_encoder_identity_and_power_validation(kwargs):
    with pytest.raises(ValueError):
        _encoder(**kwargs)


@pytest.mark.parametrize("mode,total_gpus", [("agg", 2), ("disagg", 3)])
def test_native_epd_search_smoke(mode, total_gpus):
    """Exercise packaged encoder data and the real language replay in both layouts."""
    pytest.importorskip("aisimulate._runtime")
    parallel = {"tp": 1, "replicas": 1}
    if mode == "disagg":
        parallel = {"prefill": parallel.copy(), "decode": parallel.copy()}
    config = _config(
        search_space=_config().search_space.model_dump()
        | {
            "deployment_mode": [mode],
            "parallel_configs": [parallel],
            "gpu_budget": 4,
            "encoder": {"tp": [1], "batch_size": [1], "workers": [1]},
        },
        workload=_config().workload.model_dump() | {"concurrency": 2, "num_request_ratio": 1},
        sweep={"algorithm": "random", "max_trials": 1, "parallel_evals": 1, "max_eval_seconds": None},
    )
    result = Sweeper(runner_factory=EngineReplayRunnerFactory(), show_progress=False).run(config)
    assert len(result.selected_candidates) == 1, result.to_json()
    candidate = result.candidates[0]
    assert candidate.used_gpus == total_gpus
    assert candidate.metrics["completed_requests"] == 2
    assert candidate.metrics["output_throughput_tok_s"] > 0
    assert candidate.provenance.runner_metadata["metric_semantics"] == "analytical_epd_overlay"
    assert candidate.config["encoder"]["image_count"] == 1
