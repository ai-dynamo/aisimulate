# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve pinned Sweeper estimator controls before optimizer execution."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

from aiconfigurator_core.sdk import perf_database
from aiconfigurator_core.sdk.utils import get_model_config_from_model_path

from .config import ForwardModel, SearchSpace
from .replay import EstimatorSpec


class EstimatorResolutionError(ValueError):
    """A configured estimator/data identity cannot be resolved exactly."""


def resolve_systems_paths(configured: list[str]) -> tuple[str, ...]:
    """Expand and validate request-scoped system roots without setting globals."""

    packaged = os.fspath(resources.files("aiconfigurator_core") / "systems")
    resolved: list[str] = []
    for entry in configured:
        path = packaged if entry.lower() == "default" else os.path.abspath(os.path.expanduser(entry))
        if not os.path.isdir(path):
            raise EstimatorResolutionError(f"systems_paths entry is not an existing directory: {entry!r}")
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
        raise EstimatorResolutionError(f"system {system!r} has no data_dir in its system definition")
    data_root = Path(systems_root, str(data_dir))
    version_dirs = list(data_root.glob(f"**/{backend}/{version}"))
    complete = [
        path
        for path in version_dirs
        if (path / "fpm_forward_perf.parquet").is_file() and (path / "fpm_forward_perf.metadata.json").is_file()
    ]
    if not complete:
        raise EstimatorResolutionError(
            "forward_model='fpm' requires fpm_forward_perf.parquet and its "
            f"metadata sidecar for model/system/backend/version; no FPM data found "
            f"under {data_root} for {system}/{backend}/{version}"
        )


def resolve_estimator_specs(search_space: SearchSpace) -> dict[str, EstimatorSpec]:
    """Resolve every searched backend to one immutable estimator contract.

    Resolution happens once before branch enumeration. A bad model, system path,
    backend version, database mode, or FPM data identity therefore fails before
    the sampler creates a study or spends a trial.
    """

    systems_paths = resolve_systems_paths(search_space.systems_paths)
    try:
        model_config = get_model_config_from_model_path(search_space.model_name)
    except Exception as exc:
        raise EstimatorResolutionError(f"cannot resolve model {search_space.model_name!r}: {exc}") from exc
    architecture = str(model_config.get("architecture") or "").strip()
    if not architecture:
        raise EstimatorResolutionError(f"model {search_space.model_name!r} exposes no architecture")

    system_spec = perf_database.load_system_spec(search_space.hardware_sku, systems_paths=systems_paths)
    if not system_spec:
        raise EstimatorResolutionError(
            f"unknown system {search_space.hardware_sku!r} under systems_paths={list(systems_paths)!r}"
        )

    available = perf_database.get_supported_databases(systems_paths=list(systems_paths))
    available_by_backend = available.get(search_space.hardware_sku, {})
    transfer_policy = tuple(kind.value for kind in search_space.transfer_policy)
    mode = search_space.database_mode.value
    allow_missing_data = mode in {"EMPIRICAL", "SOL"}

    resolved: dict[str, EstimatorSpec] = {}
    for backend in dict.fromkeys(search_space.backend):
        versions = available_by_backend.get(backend, [])
        requested = search_space.requested_backend_version(backend)
        if requested is not None:
            if requested not in versions:
                raise EstimatorResolutionError(
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
                raise EstimatorResolutionError(
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
            raise EstimatorResolutionError(
                f"cannot load estimator data for {search_space.hardware_sku}/{backend}/{version} in {mode} mode: {exc}"
            ) from exc
        if database is None:
            raise EstimatorResolutionError(
                f"performance data unavailable for {search_space.hardware_sku}/{backend}/{version} in {mode} mode"
            )

        systems_root = os.path.abspath(str(database.systems_root))
        if search_space.forward_model is ForwardModel.FPM:
            if search_space.aic_nextn:
                raise EstimatorResolutionError(
                    "forward_model='fpm' does not support aic_nextn/MTP; use forward_model='op_level'"
                )
            _require_fpm_data(
                systems_root=systems_root,
                system_spec=dict(database.system_spec),
                system=search_space.hardware_sku,
                backend=backend,
                version=version,
            )

        resolved[backend] = EstimatorSpec(
            model_path=search_space.model_name,
            model_architecture=architecture,
            system=search_space.hardware_sku,
            backend=backend,
            backend_version=version,
            performance_data_version=version,
            database_mode=mode,
            transfer_policy=transfer_policy,
            forward_model=search_space.forward_model.value,
            engine_step_backend=search_space.engine_step_backend.value,
            systems_paths=systems_paths,
            performance_data_root=systems_root,
        )
    return resolved
