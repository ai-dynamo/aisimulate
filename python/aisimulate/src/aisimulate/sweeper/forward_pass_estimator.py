# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve pinned Sweeper forward-pass estimator controls before optimization."""

from __future__ import annotations

import os
from importlib import resources
from typing import Any

from aiconfigurator_core.sdk import (
    ForwardPassPerfModelConfig,
    ForwardPassPerfOptions,
    RustForwardPassPerfModel,
)

from .config import SearchSpace
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


def resolve_forward_pass_estimator_specs(search_space: SearchSpace) -> dict[str, ForwardPassEstimatorSpec]:
    """Ask Core to resolve each searched backend before branch enumeration.

    Sweeper deliberately owns no model/database selection logic. It supplies the
    typed request, then persists Core's exact config and provenance unchanged.
    """

    systems_paths = resolve_systems_paths(search_space.systems_paths)
    raw_options = search_space.forward_pass_options
    try:
        options = None if raw_options is None else ForwardPassPerfOptions(**raw_options)
    except TypeError as exc:
        raise ForwardPassEstimatorResolutionError(f"invalid forward_pass_options: {exc}") from exc

    resolved: dict[str, ForwardPassEstimatorSpec] = {}
    for backend in dict.fromkeys(search_space.backend):
        transfer_policy: Any = search_space.transfer_policy
        if isinstance(transfer_policy, list):
            transfer_policy = tuple(transfer_policy)
        request = ForwardPassPerfModelConfig(
            model=search_space.model_name,
            system=search_space.hardware_sku,
            backend=backend,
            backend_version=search_space.requested_backend_version(backend),
            nextn=int(search_space.aic_nextn or 0),
            forward_model=search_space.forward_model.value,
            database_mode=search_space.database_mode.value,
            transfer_policy=transfer_policy,
            systems_paths=systems_paths,
            fallback_policy=search_space.forward_pass_fallback_policy.value,
        )
        model: RustForwardPassPerfModel | None = None
        try:
            model = RustForwardPassPerfModel.best_available(request, options)
            diagnostics = model.diagnostics()
        except Exception as exc:
            raise ForwardPassEstimatorResolutionError(
                "Core cannot construct the forward-pass estimator for "
                f"{search_space.model_name}/{search_space.hardware_sku}/{backend}: {exc}"
            ) from exc
        finally:
            if model is not None:
                model.close()

        provenance = diagnostics.get("provenance")
        if not isinstance(provenance, dict) or not isinstance(provenance.get("config"), dict):
            raise ForwardPassEstimatorResolutionError(
                f"Core returned no resolved provenance for {search_space.hardware_sku}/{backend}"
            )
        resolved_config = dict(provenance["config"])
        selected_root = provenance.get("selected_systems_root")
        if selected_root:
            resolved_config["systems_paths"] = [str(selected_root)]
        if not resolved_config.get("backend_version"):
            raise ForwardPassEstimatorResolutionError(
                f"Core did not resolve an exact backend version for {search_space.hardware_sku}/{backend}"
            )
        resolved[backend] = ForwardPassEstimatorSpec(
            config=resolved_config,
            options=None if options is None else options.to_dict(),
            diagnostics=diagnostics,
        )
    return resolved
