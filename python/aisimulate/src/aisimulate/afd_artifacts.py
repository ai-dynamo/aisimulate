# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic qualification artifacts for analytical AFD predictions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aisimulate_core._native import validate_afd_routing_location

from .sweeper.afd_parallel import AFD_SCHEMA_VERSION, AFDTopology, validate_afd_model
from .sweeper.provider import JSONValue
from .sweeper.replay import REPLAY_SPEC_API_VERSION, ReplaySpec, canonical_json, resolve_afd_replay_identity

AFD_QUALIFICATION_SCHEMA_VERSION = 2
AFD_REPLAY_SPEC_FILENAME = "afd-replay-spec.json"
AFD_QUALIFICATION_FILENAME = "afd-qualification.json"


class AFDQualificationError(ValueError):
    """The replay contract cannot be represented as a qualified AFD artifact."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AFDQualificationError(f"{path} must be a mapping")
    return value


def build_afd_qualification(spec: ReplaySpec, *, memory: Mapping[str, Any] | None = None) -> dict[str, JSONValue]:
    """Validate every role's reproducible contract; serving accuracy is separate."""
    if type(spec.api_version) is not int or spec.api_version != REPLAY_SPEC_API_VERSION:
        raise AFDQualificationError(f"unsupported ReplaySpec API version {spec.api_version}")
    deployment = spec.backend_deployment
    if deployment.encoder is not None or spec.workload.get("images") is not None:
        raise AFDQualificationError("AFD qualification rejects encoder pools and images")
    if not deployment.has_afd or deployment.deployment_mode not in {"agg", "disagg"}:
        raise AFDQualificationError("qualification requires an aggregated or P/D deployment containing AFD roles")
    if deployment.backend != "sglang" or not deployment.backend_version:
        raise AFDQualificationError("AFD artifacts require sglang and a resolved backend_version")
    expected_roles = ("aggregated",) if deployment.deployment_mode == "agg" else ("prefill", "decode")
    if set(deployment.afd_roles) - set(expected_roles):
        raise AFDQualificationError("AFD role is outside the configured deployment topology")
    try:
        spec = resolve_afd_replay_identity(spec)
    except ValueError as error:
        raise AFDQualificationError(str(error)) from error
    deployment = spec.backend_deployment
    plans, total_gpus, source_models = {}, 0, {}
    for role in expected_roles:
        raw = getattr(deployment, "agg_engine_args" if role == "aggregated" else f"{role}_engine_args")
        raw = _mapping(raw, f"{role}.engine_args")
        count = getattr(deployment, "num_workers" if role == "aggregated" else f"num_{role}_workers")
        if type(count) is not int or count < 1:
            raise AFDQualificationError(f"{role} worker count must be positive")
        if role not in deployment.afd_roles:
            prefix = "" if role == "aggregated" else f"{role}_"
            values = [deployment.parallel_config.get(prefix + k, 1) for k in ("tp", "pp", "attention_dp")]
            if any(type(v) is not int or v < 1 for v in values):
                raise AFDQualificationError(f"invalid ordinary {role} GPU geometry")
            gpus = count * values[0] * values[1] * values[2]
            plans[role] = {"type": "monolithic", "workers": count, "gpu_accounting": {"total_gpus": gpus}}
        else:
            provenance = _mapping(deployment.parallel_config.get(role), f"parallel_config.{role}")
            schema = provenance.get("schema_version")
            if type(schema) is not int or schema != AFD_SCHEMA_VERSION:
                raise AFDQualificationError(f"unsupported {role} AFD topology schema; expected {AFD_SCHEMA_VERSION}")
            topology = AFDTopology.from_mapping(
                {k: v for k, v in _mapping(raw.get("afd"), f"{role}.afd").items() if k != "chunked_prefill_size"}
            )
            validate_afd_routing_location(topology.routing_location)
            expected = topology.provenance()
            saved_topology = AFDTopology.from_mapping(_mapping(provenance.get("topology"), f"{role}.topology"))
            if saved_topology != topology or provenance.get("gpu_accounting") != expected["gpu_accounting"]:
                raise AFDQualificationError(f"{role} provenance does not match concrete topology/GPU accounting")
            metadata = _mapping(deployment.performance_model_metadata.get(role), f"performance_model_metadata.{role}")
            if metadata.get("provider") not in ("afd", "aic-afd") or metadata.get("measurement_required") is not False:
                raise AFDQualificationError(f"{role} requires the native afd provider")
            if metadata["provider"] == "aic-afd":
                metadata = {**metadata, "provider": "afd"}
            if AFDTopology.from_mapping(_mapping(metadata.get("config"), f"{role}.config")) != topology:
                raise AFDQualificationError(f"{role} model identity disagrees with topology")
            model = metadata.get("model")
            if not isinstance(model, str) or not model:
                raise AFDQualificationError(f"{role} requires source model identity")
            if model not in source_models:
                source_models[model] = validate_afd_model(model)
            if source_models[model]["num_experts"] != topology.num_experts:
                raise AFDQualificationError(f"{role} expert count disagrees with source model")
            if count != 1:
                raise AFDQualificationError(f"{role} AFD pool requires one logical worker")
            gpus = topology.total_gpus
            plans[role] = {
                "type": "afd",
                "workers": 1,
                **expected,
                "performance_model": dict(metadata),
                "memory": {
                    "phase": "both" if role == "aggregated" else role,
                    "validation": "required_at_native_provider_construction",
                },
                "qualification": {"status": "qualified_for_replay_contract"},
            }
            from aisimulate_core.sdk.models.afd import layer_layout
            from aisimulate_core.sdk.models.helpers import _moe_source_config

            info = source_models[model]
            layout = layer_layout(info)
            raw = _moe_source_config(info)
            shared_width = int(
                raw.get("shared_expert_intermediate_size")
                or raw.get("moe_shared_expert_intermediate_size")
                or info["num_shared_experts"] * info["moe_inter_size"]
            )
            if info.get("architecture") == "Gemma4ForConditionalGeneration":
                shared_width = info["inter_size"]
            plans[role]["execution"] = {
                "attention_tp": topology.a_tp_size,
                "attention_dp": topology.a_replica_count,
                "dense_pool": "a",
                "dense_tp": topology.a_tp_size,
                "shared_pool": "a",
                "shared_tp": 1,
                "shared_intermediate_size": shared_width,
                "shared_token_dp": topology.a_tp_size,
                "shared_stage": "a",
                "return_restore_stage": "f_to_a",
                "sender_ranks": topology.a_gpu_count,
                "sender_tokens": "balanced_rank_slices",
                "f_receiver_rank": "f_node * f_gpus_per_node + a_rank % f_gpus_per_node",
                "layer_plan": "source_order",
                "layers": [
                    {"attention": a, "ffn": "routed" if m else "dense", "dsa_indexer": indexer}
                    for a, m, indexer in layout
                ],
            }
            if memory is not None:
                resolved = _mapping(memory.get(role), f"afd_memory.{role}")
                if resolved.get("status") != "fits" or resolved.get("phase") != (
                    "both" if role == "aggregated" else role
                ):
                    raise AFDQualificationError(f"{role} requires a successful matching native memory check")
                plans[role]["memory"] = dict(resolved)
        total_gpus += gpus
    replay_sha256 = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()
    return {
        "schema_version": AFD_QUALIFICATION_SCHEMA_VERSION,
        "kind": "aisimulate.afd.qualification",
        "qualification": {
            "status": "qualified_for_replay_contract",
            "native_deployment_supported": False,
            "native_deployment_reason": "AISimulate does not provide a physical AFD serving adapter or launch renderer",
        },
        "identity": {
            "replay_spec_api_version": spec.api_version,
            "replay_spec_sha256": replay_sha256,
            "backend": deployment.backend,
            "backend_version": deployment.backend_version,
            "deployment_mode": deployment.deployment_mode,
        },
        "deployment_plan": {
            "roles": plans,
            "gpu_accounting": {"total_gpus": total_gpus},
            "launch": {"supported": False, "arguments": None},
        },
        "workload": json.loads(canonical_json(spec.workload)),
        "goal": json.loads(canonical_json(spec.goal)),
        "concurrency": spec.concurrency,
        "adapters": json.loads(canonical_json(spec.adapters)),
    }


def write_afd_qualification_artifacts(
    root: Path, spec: ReplaySpec, *, memory: Mapping[str, Any] | None = None
) -> tuple[Path, Path] | None:
    """Write role-indexed AFD contract artifacts after validating the entire deployment."""
    if not spec.backend_deployment.has_afd:
        return None
    qualification = build_afd_qualification(spec, memory=memory)
    spec = resolve_afd_replay_identity(spec)
    replay_path, qualification_path = root / AFD_REPLAY_SPEC_FILENAME, root / AFD_QUALIFICATION_FILENAME
    replay_path.write_text(
        json.dumps(json.loads(canonical_json(spec)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    qualification_path.write_text(json.dumps(qualification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return replay_path, qualification_path


__all__ = [
    "AFD_QUALIFICATION_FILENAME",
    "AFD_QUALIFICATION_SCHEMA_VERSION",
    "AFD_REPLAY_SPEC_FILENAME",
    "AFDQualificationError",
    "build_afd_qualification",
    "write_afd_qualification_artifacts",
]
