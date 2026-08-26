# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve pinned Sweeper forward-pass estimator controls before optimization."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

from aiconfigurator_core.sdk import common, perf_database
from aiconfigurator_core.sdk.utils import get_model_config_from_model_path

from .config import ForwardModel, SearchSpace
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


def _require_fpm_data(
    *,
    systems_root: str,
    system_spec: dict,
    system: str,
    backend: str,
    version: str,
) -> None:
    data_dir = system_spec.get("data_dir")
    if not data_dir:
        raise ForwardPassEstimatorResolutionError(f"system {system!r} has no data_dir in its system definition")
    data_root = Path(systems_root, str(data_dir))
    version_dirs = list(data_root.glob(f"**/{backend}/{version}"))
    complete = [
        path
        for path in version_dirs
        if (path / "fpm_forward_perf.parquet").is_file() and (path / "fpm_forward_perf.metadata.json").is_file()
    ]
    if not complete:
        raise ForwardPassEstimatorResolutionError(
            "forward_model='fpm' requires fpm_forward_perf.parquet and its "
            f"metadata sidecar for model/system/backend/version; no FPM data found "
            f"under {data_root} for {system}/{backend}/{version}"
        )


def resolve_forward_pass_estimator_specs(search_space: SearchSpace) -> dict[str, ForwardPassEstimatorSpec]:
    """Resolve every searched backend to one immutable forward-pass estimator contract.

    Resolution happens once before branch enumeration. A bad model, system path,
    backend version, database mode, or FPM data identity therefore fails before
    the sampler creates a study or spends a trial.
    """

    systems_paths = resolve_systems_paths(search_space.systems_paths)
    try:
        enabled_transfers = common.resolve_transfer_policy(search_space.transfer_policy)
    except (TypeError, ValueError) as exc:
        raise ForwardPassEstimatorResolutionError(
            f"cannot resolve transfer_policy {search_space.transfer_policy!r}: {exc}"
        ) from exc
    transfer_policy = tuple(
        kind.value for kind in common.TransferKind if kind in enabled_transfers
    )
    try:
        model_config = get_model_config_from_model_path(search_space.model_name)
    except Exception as exc:
        raise ForwardPassEstimatorResolutionError(f"cannot resolve model {search_space.model_name!r}: {exc}") from exc
    architecture = str(model_config.get("architecture") or "").strip()
    if not architecture:
        raise ForwardPassEstimatorResolutionError(f"model {search_space.model_name!r} exposes no architecture")

    system_spec = perf_database.load_system_spec(search_space.hardware_sku, systems_paths=systems_paths)
    if not system_spec:
        raise ForwardPassEstimatorResolutionError(
            f"unknown system {search_space.hardware_sku!r} under systems_paths={list(systems_paths)!r}"
        )

    available = perf_database.get_supported_databases(systems_paths=list(systems_paths))
    available_by_backend = available.get(search_space.hardware_sku, {})
    mode = search_space.database_mode.value
    allow_missing_data = mode != "SILICON"

    resolved: dict[str, ForwardPassEstimatorSpec] = {}
    for backend in dict.fromkeys(search_space.backend):
        versions = available_by_backend.get(backend, [])
        requested = search_space.requested_backend_version(backend)
        if requested is not None:
            if requested not in versions:
                raise ForwardPassEstimatorResolutionError(
                    f"unsupported backend_version {requested!r} for "
                    f"{search_space.hardware_sku}/{backend}; available versions: {versions}"
                )
            version = requested
        else:
            version = perf_database.get_latest_database_version(
                search_space.hardware_sku,
                backend,
                systems_paths=list(systems_paths),
            )
            if version is None:
                raise ForwardPassEstimatorResolutionError(
                    f"no performance-data version for "
                    f"{search_space.hardware_sku}/{backend} under systems_paths="
                    f"{list(systems_paths)!r}"
                )

        try:
            database = perf_database.get_database_view(
                search_space.hardware_sku,
                backend,
                version,
                systems_paths=list(systems_paths),
                allow_missing_data=allow_missing_data,
                database_mode=mode,
                transfer_policy=list(transfer_policy),
            )
        except Exception as exc:
            raise ForwardPassEstimatorResolutionError(
                "cannot load forward-pass estimator data for "
                f"{search_space.hardware_sku}/{backend}/{version} in {mode} mode: {exc}"
            ) from exc
        if database is None:
            raise ForwardPassEstimatorResolutionError(
                f"performance data unavailable for {search_space.hardware_sku}/{backend}/{version} in {mode} mode"
            )

        systems_root = os.path.abspath(str(database.systems_root))
        if search_space.forward_model is ForwardModel.FPM:
            if search_space.aic_nextn:
                raise ForwardPassEstimatorResolutionError(
                    "forward_model='fpm' does not support aic_nextn/MTP; use forward_model='op_level'"
                )
            _require_fpm_data(
                systems_root=systems_root,
                system_spec=dict(database.system_spec),
                system=search_space.hardware_sku,
                backend=backend,
                version=version,
            )

        resolved[backend] = ForwardPassEstimatorSpec(
            model_path=search_space.model_name,
            model_architecture=architecture,
            system=search_space.hardware_sku,
            backend=backend,
            backend_version=version,
            database_mode=mode,
            transfer_policy=transfer_policy,
            forward_model=search_space.forward_model.value,
            systems_paths=systems_paths,
            performance_data_root=systems_root,
        )
    return resolved
