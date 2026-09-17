# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve exact per-role Sweeper forward-pass estimator contracts."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from importlib import resources
from typing import Any

from aiconfigurator_core.sdk import (
    ForwardPassPerfModelConfig,
    RustForwardPassPerfModel,
)

from .config import SearchSpace
from .deploy import _role_hardware_sku
from .replay import ForwardPassEstimatorSpec


class ForwardPassEstimatorResolutionError(ValueError):
    """A configured forward-pass estimator identity cannot be resolved exactly."""


def resolve_systems_paths(configured: list[str]) -> tuple[str, ...]:
    """Expand and validate request-scoped system roots without setting globals."""

    packaged = os.fspath(resources.files("aiconfigurator_core") / "systems")
    resolved: list[str] = []
    for entry in configured:
        path = packaged if entry.lower() == "default" else os.path.abspath(os.path.expanduser(entry))
        if not os.path.isdir(path):
            raise ForwardPassEstimatorResolutionError(f"systems_paths entry is not an existing directory: {entry!r}")
        if path not in resolved:
            resolved.append(path)
    return tuple(resolved)


_DEFAULT_BLOCK_SIZE = {"vllm": 64, "sglang": 1, "trtllm": 32}


def _role_prefix(role: str) -> str:
    return "" if role == "agg" else f"{role}_"


class ForwardPassEstimatorResolver:
    """Resolve and cache Core-owned estimator identities for concrete roles.

    The search-space request alone is not an estimator identity: whole-forward
    FPM cells include topology and backend block size. ``resolve_candidate`` is
    called after a suggestion has been unrolled but before its ``ReplaySpec``
    reaches a runner. Every exact role request crosses the canonical Core
    constructor, and the returned config/provenance is carried unchanged.
    """

    def __init__(self, search_space: SearchSpace) -> None:
        self._search_space = search_space
        self._resolved: dict[str, ForwardPassEstimatorSpec] = {}

    def _request(self, sample: dict[str, Any], role: str) -> ForwardPassPerfModelConfig:
        backend = str(sample["backend"])
        prefix = _role_prefix(role)
        moe_tp = int(sample[f"{prefix}moe_tp"])
        moe_ep = int(sample[f"{prefix}moe_ep"])
        block_size = sample[f"{role}_block_size"]
        if block_size is None:
            block_size = _DEFAULT_BLOCK_SIZE[backend]
        transfer_policy: Any = self._search_space.transfer_policy
        if isinstance(transfer_policy, list):
            transfer_policy = tuple(transfer_policy)
        controls = self._search_space.role_estimator_controls.get(role, {})
        nextn = sample.get("aic_nextn")
        if nextn is None:
            nextn = self._search_space.aic_nextn
        return ForwardPassPerfModelConfig(
            model=self._search_space.model_name,
            system=_role_hardware_sku(sample, role),
            backend=backend,
            worker_type="aggregated" if role == "agg" else role,
            backend_version=self._search_space.requested_backend_version(backend),
            tp=int(sample[f"{prefix}tp"]),
            pp=int(sample[f"{prefix}pp"]),
            attention_dp=int(sample[f"{prefix}attention_dp"]),
            moe_tp_size=moe_tp if moe_tp * moe_ep > 1 else None,
            moe_ep_size=moe_ep if moe_tp * moe_ep > 1 else None,
            nextn=int(nextn or 0),
            kv_block_size=int(block_size),
            estimation_mode=controls.get("estimation_mode", self._search_space.estimation_mode),
            database_mode=controls.get("database_mode", self._search_space.database_mode),
            transfer_policy=controls.get("transfer_policy", transfer_policy),
            systems_paths=resolve_systems_paths(self._search_space.systems_paths_for(role)),
            fallback_policy=controls.get("fallback_policy", self._search_space.fallback_policy),
            estimator_config=controls.get("estimator_config", self._search_space.estimator_config),
        )

    def _resolve(self, request: ForwardPassPerfModelConfig, role: str) -> ForwardPassEstimatorSpec:
        request_payload = vars(request)
        cache_key = json.dumps(request.to_dict(), sort_keys=True)
        cached = self._resolved.get(cache_key)
        if cached is not None:
            return deepcopy(cached)

        model: RustForwardPassPerfModel | None = None
        try:
            model = RustForwardPassPerfModel.best_available(request)
            diagnostics = model.diagnostics()
        except Exception as exc:
            raise ForwardPassEstimatorResolutionError(
                "Core cannot construct the exact forward-pass estimator for "
                f"role={role}, model={request.model}, system={request.system}, "
                f"backend={request.backend}, tp={request.tp}, pp={request.pp}, "
                f"attention_dp={request.attention_dp}, moe_tp={request.moe_tp_size}, "
                f"moe_ep={request.moe_ep_size}, kv_block_size={request.kv_block_size}: {exc}"
            ) from exc
        finally:
            if model is not None:
                model.close()

        provenance = diagnostics.get("provenance")
        if not isinstance(provenance, dict) or not isinstance(provenance.get("config"), dict):
            raise ForwardPassEstimatorResolutionError(
                f"Core returned no resolved provenance for {request.system}/{request.backend}/{role}"
            )
        resolved_config = dict(provenance["config"])
        if diagnostics.get("readiness") != "ready":
            raise ForwardPassEstimatorResolutionError(
                f"estimator for {role} is not ready; regression requires training observations"
            )
        if not resolved_config.get("backend_version"):
            raise ForwardPassEstimatorResolutionError(
                f"Core did not resolve an exact backend version for {request.system}/{request.backend}/{role}"
            )
        selected_root = provenance.get("selected_systems_root")
        if selected_root and resolved_config.get("systems_paths") != [str(selected_root)]:
            raise ForwardPassEstimatorResolutionError(
                "Core provenance must pin its selected systems root in the resolved config: "
                f"config={resolved_config.get('systems_paths')!r}, selected={selected_root!r}"
            )
        for field in (
            "model",
            "system",
            "backend",
            "tp",
            "pp",
            "attention_dp",
            "moe_tp_size",
            "moe_ep_size",
            "nextn",
            "kv_block_size",
            "worker_type",
        ):
            if resolved_config.get(field) != request_payload.get(field):
                raise ForwardPassEstimatorResolutionError(
                    f"Core changed exact candidate field {field!r}: "
                    f"requested={request_payload.get(field)!r}, resolved={resolved_config.get(field)!r}"
                )
        spec = ForwardPassEstimatorSpec(
            config=resolved_config,
            options=None,
            diagnostics=diagnostics,
        )
        self._resolved[cache_key] = deepcopy(spec)
        return spec

    def resolve_candidate(self, sample: dict[str, Any]) -> dict[str, ForwardPassEstimatorSpec]:
        """Resolve every concrete engine role before replay trial execution."""

        if sample["deployment_mode"] not in {"agg", "disagg"} or self._search_space.encoder is not None:
            return {}
        roles = ("agg",) if sample["deployment_mode"] == "agg" else ("prefill", "decode")
        resolved = {
            role: self._resolve(self._request(sample, role), role)
            for role in roles
            if sample.get(f"{role}_timing_model") is None
        }
        versions = {spec.backend_version for spec in resolved.values()}
        if len(versions) > 1:
            raise ForwardPassEstimatorResolutionError(
                f"Core resolved inconsistent backend versions across candidate roles: {sorted(versions)}"
            )
        return resolved
