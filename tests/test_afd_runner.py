# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end analytical runner tests for AFD deployments."""

from dataclasses import replace

import pytest

from aisimulate.runner import (
    AFDCompanionTiming,
    AICAFDCompanionPerformanceModel,
    EngineReplayRunnerFactory,
    InvalidRunnerError,
)
from aisimulate.sweeper import (
    AFDTopology,
    BackendDeploymentSpec,
    ReplayOutputRequirements,
    ReplaySpec,
)


def _metadata(*phases: str) -> dict:
    return {
        "afd": {
            "provider": "test",
            "measurement_required": False,
            "measurement_api_version": 1,
            "measurements": [
                {
                    "phase": phase,
                    "attention_ms": 1.0,
                    "ffn_ms": 1.0,
                    "a_to_f_ms": 0.0,
                    "f_to_a_ms": 0.0,
                    "num_layers": 1,
                    "provenance": {"provider": "test"},
                }
                for phase in phases
            ],
        }
    }


def _spec(topology: AFDTopology, *, companion_role: str | None = None) -> ReplaySpec:
    parallel_config = {
        "afd": topology.provenance()["topology"],
        "afd_provenance": {
            "gpu_accounting": {
                "total_gpus": topology.total_gpus + (2 if companion_role else 0)
            }
        },
    }
    kwargs = {}
    if companion_role is not None:
        prefix = f"{companion_role}_"
        parallel_config.update(
            {
                f"{prefix}tp": 2,
                f"{prefix}pp": 1,
                f"{prefix}attention_dp": 1,
                f"{prefix}moe_tp": 1,
                f"{prefix}moe_ep": 1,
                f"{prefix}replicas": 1,
            }
        )
        kwargs[f"{companion_role}_engine_args"] = {
            "max_num_batched_tokens": 256,
            "max_num_seqs": 2,
            "timing_model": {
                "type": "fixed",
                "prefill_ms": 2.0,
                "decode_ms": 2.0,
            },
        }
        kwargs[f"num_{companion_role}_workers"] = 1
    return ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode=topology.adapter_topology,
            backend="vllm",
            backend_version="test",
            parallel_config=parallel_config,
            performance_model_metadata=_metadata(
                *("prefill", "decode")
                if topology.phase.value == "both"
                else (topology.phase.value,)
            ),
            **kwargs,
        ),
        workload={
            "isl": 8,
            "osl": 3,
            "concurrency": 2,
            "num_request_ratio": 2,
        },
        goal={"target": "throughput", "sla": None},
        concurrency=2,
    )


def _topology(*, phase: str = "both", combined_with_pd: bool = False) -> AFDTopology:
    return AFDTopology(
        n_a_nodes=1,
        n_f_nodes=1,
        gpus_per_node=4,
        tp_a=2,
        a_batch_size=2,
        num_microbatches=2,
        phase=phase,
        combined_with_pd=combined_with_pd,
    )


def test_factory_advertises_afd_only_after_executable_runner_exists():
    capabilities = EngineReplayRunnerFactory().capabilities()

    for backend in ("vllm", "sglang", "trtllm"):
        assert capabilities.supports_backend_topology(backend, "afd")
        assert capabilities.supports_backend_topology(backend, "afd+pd")


def test_pure_both_phase_afd_runs_without_native_aggregate_fallback():
    report = (
        EngineReplayRunnerFactory()
        .create(0)
        .run(
            _spec(_topology()),
            output_requirements=ReplayOutputRequirements(
                include_raw_report=True,
                capture_per_request=True,
            ),
        )
    )

    assert report.metrics["completed_requests"] == 4.0
    assert report.metrics["duration_ms"] == pytest.approx(18.0)
    assert report.metrics["mean_tpot_ms"] == pytest.approx(3.0)
    assert report.metrics["output_throughput_tok_s"] == pytest.approx(12_000.0 / 18.0)
    assert report.metrics["num_total_gpus"] == 8.0
    assert "goodput_output_throughput_tok_s" not in report.metrics
    assert report.metadata["afd_replay"]["executor"] == "afd_foreground"
    assert report.metadata["afd_replay"]["afd_passes"] == 6
    assert len(report.metadata["per_request"]) == 4
    assert report.metadata["native_report"]["summary"] == report.metrics
    assert (
        report.metadata["native_report"]["per_request"]
        == report.metadata["per_request"]
    )
    assert "afd_report" in report.metadata


@pytest.mark.parametrize(
    ("concurrency", "request_count", "expected_arrivals"),
    [(1, 3, [0, 9, 18]), (2, 4, [0, 0, 9, 9]), (5, 1, [0])],
)
def test_closed_loop_replacements_arrive_when_requests_complete(concurrency, request_count, expected_arrivals):
    spec = _spec(_topology())
    spec.workload.update(
        source_type="synthetic", load_type="concurrency", concurrency=concurrency, request_count=request_count
    )
    spec = replace(spec, concurrency=concurrency)

    report = (
        EngineReplayRunnerFactory()
        .create(0)
        .run(spec, output_requirements=ReplayOutputRequirements(capture_per_request=True))
    )

    assert [request["arrival_time_ms"] for request in report.metadata["per_request"]] == expected_arrivals
    assert report.metrics["mean_ttft_ms"] == pytest.approx(3.0)


def test_afd_plus_pd_respects_global_concurrency_and_uses_companion_contract():
    calls = []

    class CompanionModel:
        def measure(self, spec):
            calls.append(spec)
            return AFDCompanionTiming(
                phase="decode",
                latency_ms=2.0,
                batch_capacity_per_worker=2,
                workers=1,
                provenance={"provider": "test-companion"},
            )

    topology = _topology(phase="prefill", combined_with_pd=True)
    report = (
        EngineReplayRunnerFactory(afd_companion_model=CompanionModel())
        .create(0)
        .run(_spec(topology, companion_role="decode"))
    )

    assert len(calls) == 1
    # Two requests fill the concurrency limit until both phases finish at 7 ms.
    assert report.metrics["duration_ms"] == pytest.approx(14.0)
    assert report.metrics["mean_tpot_ms"] == pytest.approx(2.0)
    assert report.metrics["num_total_gpus"] == 10.0
    assert report.metadata["afd_replay"]["batches"] == 2
    assert report.metadata["afd_replay"]["companion"]["provider"] == "test-companion"


@pytest.mark.parametrize(
    ("phase", "companion_role", "expected_arrivals", "expected_duration"),
    [
        ("prefill", "decode", [0, 0, 0, 7, 7, 11, 15], 23),
        ("decode", "prefill", [0, 0, 0, 8, 8, 14, 20], 32),
    ],
)
def test_closed_loop_admission_handles_partial_batches_across_both_pools(
    phase, companion_role, expected_arrivals, expected_duration
):
    spec = _spec(_topology(phase=phase, combined_with_pd=True), companion_role=companion_role)
    spec.workload.update(concurrency=3, request_count=7)
    spec = replace(spec, concurrency=3)

    report = (
        EngineReplayRunnerFactory()
        .create(0)
        .run(spec, output_requirements=ReplayOutputRequirements(capture_per_request=True))
    )

    records = report.metadata["per_request"]
    assert [record["arrival_time_ms"] for record in records] == expected_arrivals
    assert report.metrics["duration_ms"] == pytest.approx(expected_duration)
    assert report.metadata["afd_replay"]["batches"] == 5
    for record in records:
        now = record["arrival_time_ms"]
        in_flight = sum(
            other["arrival_time_ms"] <= now < other["arrival_time_ms"] + other["e2e_latency_ms"] for other in records
        )
        assert in_flight <= 3


@pytest.mark.parametrize(
    ("phase", "companion_role", "expected_duration"), [("prefill", "decode", 11), ("decode", "prefill", 14)]
)
def test_open_loop_bursts_preserve_independent_pool_overlap(phase, companion_role, expected_duration):
    spec = _spec(_topology(phase=phase, combined_with_pd=True), companion_role=companion_role)
    spec.workload.pop("concurrency")
    spec.workload.update(arrival_interval_ms=0.0, request_count=4)
    spec = replace(spec, concurrency=None)

    report = EngineReplayRunnerFactory().create(0).run(spec)

    assert report.metrics["duration_ms"] == pytest.approx(expected_duration)


@pytest.mark.parametrize(
    ("phase", "companion_role", "expected_tpots", "itl_sla", "expected_duration"),
    [("prefill", "decode", [10, 18.5, 18.5, 27], 11, 63), ("decode", "prefill", [3, 5, 5, 7], 4, 20)],
)
def test_open_loop_tpot_and_sla_include_decode_queueing(
    phase, companion_role, expected_tpots, itl_sla, expected_duration
):
    spec = _spec(_topology(phase=phase, combined_with_pd=True), companion_role=companion_role)
    spec.workload.pop("concurrency")
    spec.workload.update(arrival_interval_ms=0.01, request_count=4)
    spec = replace(spec, concurrency=None)
    if companion_role == "decode":
        spec.backend_deployment.decode_engine_args["timing_model"]["decode_ms"] = 10.0
    spec.goal["sla"] = {"itl_ms": itl_sla}

    report = (
        EngineReplayRunnerFactory()
        .create(0)
        .run(spec, output_requirements=ReplayOutputRequirements(capture_per_request=True))
    )

    assert [record["tpot_ms"] for record in report.metadata["per_request"]] == pytest.approx(expected_tpots)
    assert report.metrics["mean_tpot_ms"] == pytest.approx(sum(expected_tpots) / 4)
    assert report.metrics["goodput_completed_requests"] == 1.0
    assert report.metrics["goodput_output_throughput_tok_s"] == pytest.approx(3_000.0 / expected_duration)


@pytest.mark.parametrize(
    ("phase", "companion_role", "expected_duration", "expected_first_ttft", "expected_first_tpot"),
    [("prefill", "decode", 10_011, 3, 5_000.5), ("decode", "prefill", 10_016, 10_002, 3)],
)
def test_companion_startup_seconds_delay_the_configured_phase(
    phase, companion_role, expected_duration, expected_first_ttft, expected_first_tpot
):
    spec = _spec(_topology(phase=phase, combined_with_pd=True), companion_role=companion_role)
    engine_args = getattr(spec.backend_deployment, f"{companion_role}_engine_args")
    engine_args["startup_time"] = 10.0

    report = (
        EngineReplayRunnerFactory()
        .create(0)
        .run(spec, output_requirements=ReplayOutputRequirements(capture_per_request=True))
    )

    assert report.metrics["duration_ms"] == pytest.approx(expected_duration)
    assert report.metadata["per_request"][0]["ttft_ms"] == pytest.approx(expected_first_ttft)
    assert report.metadata["per_request"][0]["tpot_ms"] == pytest.approx(expected_first_tpot)


@pytest.mark.parametrize(
    ("phase", "companion_role", "expected_duration"),
    [("both", None, 6), ("prefill", "decode", 6), ("decode", "prefill", 10_004)],
)
def test_single_output_token_completes_at_prefill_without_decode(phase, companion_role, expected_duration):
    spec = _spec(_topology(phase=phase, combined_with_pd=companion_role is not None), companion_role=companion_role)
    spec.workload["osl"] = 1
    spec.goal["sla"] = {"itl_ms": 0.1}
    if companion_role is not None:
        getattr(spec.backend_deployment, f"{companion_role}_engine_args")["startup_time"] = 10.0

    report = (
        EngineReplayRunnerFactory()
        .create(0)
        .run(spec, output_requirements=ReplayOutputRequirements(capture_per_request=True))
    )

    assert report.metrics["duration_ms"] == pytest.approx(expected_duration)
    assert report.metrics["num_tpot_samples"] == 0
    assert report.metrics["goodput_completed_requests"] == 4
    for record in report.metadata["per_request"]:
        assert record["e2e_latency_ms"] == record["ttft_ms"]
        assert record["tpot_ms"] == 0.0


def test_default_companion_model_consumes_fixed_timing_without_aic_lookup():
    spec = _spec(
        _topology(phase="prefill", combined_with_pd=True),
        companion_role="decode",
    )

    timing = AICAFDCompanionPerformanceModel().measure(spec)

    assert timing.phase.value == "decode"
    assert timing.latency_ms == 2.0
    assert timing.total_batch_capacity == 2
    assert timing.provenance["provider"] == "fixed"


def test_afd_runner_rejects_unresolved_measurement_before_execution():
    spec = _spec(_topology())
    spec.backend_deployment.performance_model_metadata["afd"][
        "measurement_required"
    ] = True

    with pytest.raises(ValueError, match="measurement is unresolved"):
        EngineReplayRunnerFactory().create(0).run(spec)


def test_pure_single_phase_afd_does_not_claim_end_to_end_replay():
    with pytest.raises(ValueError, match="pure AFD replay requires phase='both'"):
        EngineReplayRunnerFactory().create(0).run(_spec(_topology(phase="decode")))


def test_afd_runner_applies_sla_to_goodput():
    spec = _spec(_topology())
    spec.goal["sla"] = {"ttft_ms": 2.0}

    report = EngineReplayRunnerFactory().create(0).run(spec)

    assert report.metrics["goodput_output_throughput_tok_s"] == 0.0


def test_afd_runner_rejects_conflicting_gpu_accounting():
    spec = _spec(_topology())
    spec.backend_deployment.parallel_config["afd_provenance"]["gpu_accounting"][
        "total_gpus"
    ] = 99

    with pytest.raises(ValueError, match="conflicts with topology accounting"):
        EngineReplayRunnerFactory().create(0).run(spec)


def test_afd_runner_fails_closed_for_unimplemented_simulation_deadline():
    spec = _spec(_topology())
    spec.workload["max_sim_time_ms"] = 10.0

    with pytest.raises(ValueError, match="max_sim_time_ms"):
        EngineReplayRunnerFactory().create(0).run(spec)


@pytest.mark.parametrize("combined_with_pd", [False, True])
def test_afd_runner_rejects_images_before_analytical_dispatch(combined_with_pd):
    spec = _spec(
        _topology(phase="decode" if combined_with_pd else "both", combined_with_pd=combined_with_pd),
        companion_role="prefill" if combined_with_pd else None,
    )
    spec = replace(spec, workload={**spec.workload, "images": {"height": 448, "width": 448, "count": 1}})
    with pytest.raises(InvalidRunnerError, match="image workloads require an encoder pool"):
        EngineReplayRunnerFactory().create(0).run(spec)
