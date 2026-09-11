# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic qualification artifacts for analytical AFD predictions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .sweeper.afd_parallel import AFDTopology
from .sweeper.provider import JSONValue
from .sweeper.replay import REPLAY_SPEC_API_VERSION, ReplaySpec, canonical_json

AFD_QUALIFICATION_SCHEMA_VERSION = 1
AFD_REPLAY_SPEC_FILENAME = "afd-replay-spec.json"
AFD_QUALIFICATION_FILENAME = "afd-qualification.json"


class AFDQualificationError(ValueError):
    """The replay contract cannot be represented as a qualified AFD artifact."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AFDQualificationError(f"{path} must be a mapping")
    return value


def _positive_int(value: Any, path: str) -> int:
    if type(value) is not int or value < 1:
        raise AFDQualificationError(f"{path} must be a positive integer")
    return value


def _topology_from_deployment(spec: ReplaySpec) -> tuple[AFDTopology, Mapping[str, Any]]:
    deployment = spec.backend_deployment
    if deployment.deployment_mode not in {"afd", "afd+pd"}:
        raise AFDQualificationError("AFD qualification artifacts require deployment_mode 'afd' or 'afd+pd'")
    if deployment.encoder is not None or spec.workload.get("images") is not None:
        raise AFDQualificationError("AFD qualification does not support analytical EPD encoder pools or images")
    if spec.api_version != REPLAY_SPEC_API_VERSION:
        raise AFDQualificationError(f"ReplaySpec API version {spec.api_version!r} is not supported")
    if not deployment.backend or not deployment.backend_version:
        raise AFDQualificationError("AFD artifacts require resolved backend and backend_version")

    raw_topology = _mapping(deployment.parallel_config.get("afd"), "parallel_config.afd")
    topology_fields = {
        "n_a_nodes",
        "n_f_nodes",
        "gpus_per_node",
        "tp_a",
        "a_batch_size",
        "f_moe_ep_size",
        "num_microbatches",
        "pipeline_model",
        "phase",
        "combined_with_pd",
        "comm_overhead_factor",
        "boundary_on_attn",
        "is_moe",
        "num_experts",
    }
    missing = topology_fields - set(raw_topology)
    if missing:
        raise AFDQualificationError(f"parallel_config.afd is missing required fields {sorted(missing)}")
    topology = AFDTopology(**{name: raw_topology[name] for name in topology_fields})
    if raw_topology.get("ffn_tp") != topology.ffn_tp:
        raise AFDQualificationError("parallel_config.afd.ffn_tp is inconsistent with the F pool")
    if topology.adapter_topology != deployment.deployment_mode:
        raise AFDQualificationError("parallel_config.afd topology mode does not match deployment_mode")

    provenance = _mapping(
        deployment.parallel_config.get("afd_provenance"),
        "parallel_config.afd_provenance",
    )
    expected_topology = topology.provenance()
    if json.loads(canonical_json(provenance.get("topology"))) != json.loads(canonical_json(expected_topology)):
        raise AFDQualificationError("parallel_config.afd_provenance.topology does not match the concrete topology")
    if provenance.get("mode") != deployment.deployment_mode:
        raise AFDQualificationError("parallel_config.afd_provenance.mode does not match deployment_mode")
    return topology, provenance


def _validate_measurements(spec: ReplaySpec, topology: AFDTopology) -> Mapping[str, Any]:
    metadata = _mapping(
        spec.backend_deployment.performance_model_metadata.get("afd"),
        "performance_model_metadata.afd",
    )
    if metadata.get("measurement_required") is not False:
        raise AFDQualificationError("AFD qualification requires resolved layer measurements")
    measurements = metadata.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise AFDQualificationError("performance_model_metadata.afd.measurements must not be empty")
    phases: set[str] = set()
    for index, value in enumerate(measurements):
        measurement = _mapping(value, f"performance_model_metadata.afd.measurements[{index}]")
        phase = measurement.get("phase")
        if phase not in {"prefill", "decode"}:
            raise AFDQualificationError(f"AFD measurement {index} has invalid phase {phase!r}")
        phases.add(str(phase))
    expected_phases = {"prefill", "decode"} if topology.phase.value == "both" else {topology.phase.value}
    if phases != expected_phases or len(measurements) != len(expected_phases):
        raise AFDQualificationError("AFD measurements do not cover each configured phase exactly once")
    return metadata


def _companion_plan(
    spec: ReplaySpec,
    provenance: Mapping[str, Any],
) -> dict[str, JSONValue] | None:
    deployment = spec.backend_deployment
    role = provenance.get("companion_role")
    if deployment.deployment_mode == "afd":
        if role is not None or provenance.get("companion") is not None:
            raise AFDQualificationError("pure AFD cannot contain a P/D companion")
        return None
    if role not in {"prefill", "decode"}:
        raise AFDQualificationError("AFD+P/D requires one prefill or decode companion")

    prefix = f"{role}_"
    workers = deployment.num_prefill_workers if role == "prefill" else deployment.num_decode_workers
    engine_args = deployment.prefill_engine_args if role == "prefill" else deployment.decode_engine_args
    if workers < 1 or engine_args is None:
        raise AFDQualificationError(f"AFD+P/D {role} companion is not fully materialized")
    names = ("tp", "pp", "attention_dp", "moe_tp", "moe_ep", "strategy", "replicas")
    missing = [name for name in names if f"{prefix}{name}" not in deployment.parallel_config]
    if missing:
        raise AFDQualificationError(f"AFD+P/D {role} companion is missing parallel fields {missing}")
    parallel = {name: deployment.parallel_config[f"{prefix}{name}"] for name in names}
    if _positive_int(parallel["replicas"], f"parallel_config.{prefix}replicas") != workers:
        raise AFDQualificationError(f"AFD+P/D {role} worker count does not match its parallel config")
    companion_gpus = (
        _positive_int(parallel["tp"], f"parallel_config.{prefix}tp")
        * _positive_int(parallel["pp"], f"parallel_config.{prefix}pp")
        * _positive_int(parallel["attention_dp"], f"parallel_config.{prefix}attention_dp")
        * workers
    )
    return {
        "role": str(role),
        "workers": workers,
        "gpus": companion_gpus,
        "parallelism": parallel,
        "engine_args": dict(engine_args),
    }


def build_afd_qualification(spec: ReplaySpec) -> dict[str, JSONValue]:
    """Build and validate a logical A/F pool plan for analytical replay.

    The artifact deliberately records that native launch generation is not
    implemented. It is a reproducibility and release-qualification contract,
    not a Kubernetes or shell deployment manifest.
    """

    topology, provenance = _topology_from_deployment(spec)
    measurement_metadata = _validate_measurements(spec, topology)
    companion = _companion_plan(spec, provenance)
    accounting = _mapping(provenance.get("gpu_accounting"), "afd_provenance.gpu_accounting")
    expected_companion_gpus = 0 if companion is None else int(companion["gpus"])
    expected_accounting = {
        "attention_gpus": topology.attention_gpus,
        "ffn_gpus": topology.ffn_gpus,
        "companion_gpus": expected_companion_gpus,
        "total_gpus": topology.total_gpus + expected_companion_gpus,
    }
    if dict(accounting) != expected_accounting:
        raise AFDQualificationError(
            "parallel_config.afd_provenance.gpu_accounting is inconsistent with the worker pools"
        )

    replay_json = canonical_json(spec)
    replay_sha256 = hashlib.sha256(replay_json.encode("utf-8")).hexdigest()
    deployment = spec.backend_deployment
    return {
        "schema_version": AFD_QUALIFICATION_SCHEMA_VERSION,
        "kind": "aisimulate.afd.qualification",
        "qualification": {
            "status": "qualified_for_analytical_replay",
            "execution": "analytical_foreground",
            "native_deployment_supported": False,
            "native_deployment_reason": (
                "AISimulate does not provide a physical AFD serving adapter or launch renderer"
            ),
        },
        "identity": {
            "replay_spec_api_version": spec.api_version,
            "replay_spec_sha256": replay_sha256,
            "backend": deployment.backend,
            "backend_version": deployment.backend_version,
            "deployment_mode": deployment.deployment_mode,
        },
        "deployment_plan": {
            "routing": {
                "stages": ["attention", "ffn"],
                "boundary_on_attention": topology.boundary_on_attn,
                "phase": topology.phase.value,
                "combined_with_pd": topology.combined_with_pd,
            },
            "pools": {
                "attention": {
                    "nodes": topology.n_a_nodes,
                    "workers": topology.attention_workers,
                    "gpus": topology.attention_gpus,
                    "tensor_parallel_size": topology.tp_a,
                    "batch_size_per_worker": topology.a_batch_size,
                },
                "ffn": {
                    "nodes": topology.n_f_nodes,
                    "workers": topology.ffn_workers,
                    "gpus": topology.ffn_gpus,
                    "tensor_parallel_size": topology.ffn_tp,
                    "moe_expert_parallel_size": topology.f_moe_ep_size,
                },
                "companion": companion,
            },
            "gpu_accounting": expected_accounting,
            "launch": {
                "supported": False,
                "arguments": None,
            },
        },
        "performance_model": dict(measurement_metadata),
        "workload": json.loads(canonical_json(spec.workload)),
        "goal": json.loads(canonical_json(spec.goal)),
        "concurrency": spec.concurrency,
        "adapters": json.loads(canonical_json(spec.adapters)),
    }


def write_afd_qualification_artifacts(root: Path, spec: ReplaySpec) -> tuple[Path, Path] | None:
    """Write AFD replay and qualification artifacts, or return ``None`` for other modes."""

    if spec.backend_deployment.deployment_mode not in {"afd", "afd+pd"}:
        return None
    qualification = build_afd_qualification(spec)
    replay_path = root / AFD_REPLAY_SPEC_FILENAME
    replay_path.write_text(
        json.dumps(json.loads(canonical_json(spec)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    qualification_path = root / AFD_QUALIFICATION_FILENAME
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return replay_path, qualification_path


__all__ = [
    "AFD_QUALIFICATION_FILENAME",
    "AFD_QUALIFICATION_SCHEMA_VERSION",
    "AFD_REPLAY_SPEC_FILENAME",
    "AFDQualificationError",
    "build_afd_qualification",
    "write_afd_qualification_artifacts",
]
