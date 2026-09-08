# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end analytical runner tests for AFD deployments."""

import pytest

from aisimulate.runner import (
    AFDCompanionTiming,
    AICAFDCompanionPerformanceModel,
    EngineReplayRunnerFactory,
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


def test_afd_plus_pd_overlaps_separate_phase_pools_and_uses_companion_contract():
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
    assert report.metrics["duration_ms"] == pytest.approx(11.0)
    assert report.metrics["mean_tpot_ms"] == pytest.approx(2.0)
    assert report.metrics["num_total_gpus"] == 10.0
    assert report.metadata["afd_replay"]["batches"] == 2
    assert report.metadata["afd_replay"]["companion"]["provider"] == "test-companion"


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
    spec.goal["sla"] = {"ttft_ms": 4.0}

    report = EngineReplayRunnerFactory().create(0).run(spec)

    assert (
        report.metrics["goodput_output_throughput_tok_s"]
        < report.metrics["output_throughput_tok_s"]
    )


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
