# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden and fail-closed contracts for analytical AFD artifacts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from aisimulate.afd_artifacts import (
    AFDQualificationError,
    build_afd_qualification,
    write_afd_qualification_artifacts,
)
from aisimulate.sweeper.afd_parallel import AFDParallelConfig, AFDTopology
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import BackendDeploymentSpec, ReplaySpec


def _measurement(phase: str) -> dict:
    return {
        "phase": phase,
        "attention_ms": 1.25,
        "ffn_ms": 2.5,
        "a_to_f_ms": 0.125,
        "f_to_a_ms": 0.25,
        "num_layers": 64,
        "provenance": {
            "provider": "golden",
            "backend_version": "1.3.0rc14",
            "input_length": 1024,
            "output_length": 128,
        },
    }


def _spec(*, combined_with_pd: bool = False, afd_phase: str = "decode") -> ReplaySpec:
    topology = AFDTopology(
        n_a_nodes=1,
        n_f_nodes=1,
        gpus_per_node=8,
        tp_a=4,
        a_batch_size=64,
        f_moe_ep_size=1,
        num_microbatches=4,
        pipeline_model="conservative",
        phase=afd_phase if combined_with_pd else "both",
        combined_with_pd=combined_with_pd,
        is_moe=False,
    )
    companion = (
        ReplicaParallelConfig(
            shape=ParallelShape(tp=2, pp=1, dp=1, moe_tp=1, moe_ep=1),
            replicas=2,
        )
        if combined_with_pd
        else None
    )
    parallel = AFDParallelConfig(topology=topology, companion=companion)
    parallel_config = {
        "afd": topology.provenance()["topology"],
        "afd_provenance": parallel.provenance(),
    }
    deployment_kwargs = {}
    if companion is not None:
        companion_role = "prefill" if afd_phase == "decode" else "decode"
        prefix = f"{companion_role}_"
        parallel_config.update(
            {
                f"{prefix}tp": 2,
                f"{prefix}pp": 1,
                f"{prefix}attention_dp": 1,
                f"{prefix}moe_tp": 1,
                f"{prefix}moe_ep": 1,
                f"{prefix}strategy": "tp",
                f"{prefix}replicas": 2,
            }
        )
        deployment_kwargs = {
            f"{companion_role}_engine_args": {
                "worker_type": companion_role,
                "timing_model": {"type": "fixed", f"{companion_role}_ms": 5.0},
            },
            f"num_{companion_role}_workers": 2,
        }
    measurements = [_measurement(afd_phase)]
    if not combined_with_pd:
        measurements.insert(0, _measurement("prefill"))
    return ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode=topology.adapter_topology,
            backend="trtllm",
            backend_version="1.3.0rc14",
            parallel_config=parallel_config,
            performance_model_metadata={
                "afd": {
                    "provider": "golden",
                    "measurement_required": False,
                    "measurement_api_version": 1,
                    "measurements": measurements,
                }
            },
            **deployment_kwargs,
        ),
        workload={
            "kind": "synthetic",
            "isl": 1024,
            "osl": 128,
            "request_count": 16,
            "concurrency": 4,
        },
        goal={"sla": {"ttft_ms": 800.0, "itl_ms": 30.0}},
        concurrency=4,
    )


def test_pure_afd_qualification_matches_golden() -> None:
    rendered = json.dumps(build_afd_qualification(_spec()), indent=2, sort_keys=True) + "\n"
    golden = Path(__file__).parent / "golden" / "afd-qualification.json"

    assert rendered == golden.read_text(encoding="utf-8")


def test_afd_plus_pd_qualification_preserves_companion_and_gpu_accounting() -> None:
    artifact = build_afd_qualification(_spec(combined_with_pd=True))

    companion = artifact["deployment_plan"]["pools"]["companion"]
    assert companion == {
        "role": "prefill",
        "workers": 2,
        "gpus": 4,
        "parallelism": {
            "tp": 2,
            "pp": 1,
            "attention_dp": 1,
            "moe_tp": 1,
            "moe_ep": 1,
            "strategy": "tp",
            "replicas": 2,
        },
        "engine_args": {
            "worker_type": "prefill",
            "timing_model": {"type": "fixed", "prefill_ms": 5.0},
        },
    }
    assert artifact["deployment_plan"]["gpu_accounting"]["total_gpus"] == 20


def test_afd_plus_pd_qualification_supports_decode_companion() -> None:
    artifact = build_afd_qualification(_spec(combined_with_pd=True, afd_phase="prefill"))

    companion = artifact["deployment_plan"]["pools"]["companion"]
    assert companion["role"] == "decode"
    assert companion["workers"] == 2
    assert companion["engine_args"]["worker_type"] == "decode"


def test_qualification_rejects_unresolved_measurements() -> None:
    spec = _spec()
    deployment = replace(
        spec.backend_deployment,
        performance_model_metadata={"afd": {"provider": "unresolved", "measurement_required": True}},
    )

    with pytest.raises(AFDQualificationError, match="resolved layer measurements"):
        build_afd_qualification(replace(spec, backend_deployment=deployment))


def test_qualification_rejects_inconsistent_gpu_accounting() -> None:
    spec = _spec()
    parallel_config = dict(spec.backend_deployment.parallel_config)
    provenance = dict(parallel_config["afd_provenance"])
    provenance["gpu_accounting"] = {
        **provenance["gpu_accounting"],
        "total_gpus": 999,
    }
    parallel_config["afd_provenance"] = provenance
    deployment = replace(spec.backend_deployment, parallel_config=parallel_config)

    with pytest.raises(AFDQualificationError, match="gpu_accounting is inconsistent"):
        build_afd_qualification(replace(spec, backend_deployment=deployment))


def test_write_afd_artifacts_is_deterministic_and_non_afd_is_noop(tmp_path) -> None:
    paths = write_afd_qualification_artifacts(tmp_path, _spec())
    assert paths is not None
    replay_path, qualification_path = paths
    first = (replay_path.read_bytes(), qualification_path.read_bytes())

    write_afd_qualification_artifacts(tmp_path, _spec())
    assert first == (replay_path.read_bytes(), qualification_path.read_bytes())

    non_afd = replace(
        _spec(),
        backend_deployment=BackendDeploymentSpec(
            deployment_mode="agg",
            backend="vllm",
            backend_version="0.20.1",
        ),
    )
    assert write_afd_qualification_artifacts(tmp_path, non_afd) is None
