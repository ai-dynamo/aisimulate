# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a standalone ``systems/`` root for consumers that cannot run the Python data library.

Rust ``perf_database`` users, sweeper/planner embeddings and Rust tests read operation data
from a directory (``systems_paths`` / ``AICONFIGURATOR_SYSTEMS_PATH``) and never call
:func:`~aisim_core.data.ensure.ensure_data`. :func:`materialize_systems_root` runs
that step once, in Python, and writes one self-contained root those consumers can be
pointed at: the packaged system specs and small support files plus the verified
``data/<system>/...`` tables from the pinned dataset. After the call no network or Python
data library is needed.

Preparation is explicit and additive: nothing here changes how the packaged data is read.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from aisim_core.data.ensure import DataPin, MissingDataError, ensure_data
from aisim_core.data.fetch import Transport
from aisim_core.data.systems import PACKAGED_SYSTEMS_DIR

MARKER = ".aisim-systems-root.json"
SYSTEMS_ENV = "AICONFIGURATOR_SYSTEMS_PATH"
# Packaged production op data is the part a later release removes from the wheel; everything
# else under ``systems/`` (specs, version tables, profiles) stays packaged.
_DATA_DIR = "data"


@dataclass(frozen=True)
class MaterializedSystems:
    """A composed systems root and where its operation data came from."""

    root: Path
    dataset: dict[str, Any]  # provenance reported by ensure_data for ``op-latency``
    data_files: int
    data_bytes: int

    @property
    def env(self) -> dict[str, str]:
        """Environment for a child process (Rust CLI, sweeper, planner) to use this root."""
        return {SYSTEMS_ENV: str(self.root)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "env": self.env,
            "dataset": self.dataset,
            "data_files": self.data_files,
            "data_bytes": self.data_bytes,
        }


def materialize_systems_root(
    config: Mapping[str, Any],
    pin: DataPin,
    destination: Path,
    *,
    transport: Transport | None = None,
    packaged_systems: Path = PACKAGED_SYSTEMS_DIR,
) -> MaterializedSystems:
    """Write a self-contained ``systems/`` root for ``config`` at ``destination``.

    ``config`` carries ``system``, ``backend`` and ``backend_version`` (the same keys
    ``ensure_data`` reads). The pin is forced to the ``op-latency`` category; with
    ``pin.offline`` or ``pin.local_root`` the call makes no network request. The data
    comes only from the pin: a missing or invalid dataset is an error, never a fallback to
    packaged tables. ``destination`` must not exist; a failed call removes it.
    """
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; choose a new directory")
    ensured = ensure_data(config, replace(pin, categories=("op-latency",)), transport=transport)
    provenance = ensured.datasets.get("op-latency")
    if provenance is None:  # pragma: no cover - ensure_data raises instead
        raise MissingDataError("the pin produced no op-latency data")
    loaded_root = Path(ensured.config["systems_paths"][0])
    destination.mkdir(parents=True)
    try:
        for entry in sorted(packaged_systems.iterdir()):
            if entry.name == _DATA_DIR:
                continue
            _copy(entry, destination / entry.name)
        count, size = 0, 0
        for source in sorted(p for p in loaded_root.rglob("*") if p.is_file()):
            target = destination / source.relative_to(loaded_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(source, target)
            count, size = count + 1, size + source.stat().st_size
        (destination / MARKER).write_text(json.dumps({"dataset": provenance}, indent=2, sort_keys=True) + "\n")
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return MaterializedSystems(destination, provenance, count, size)


def _copy(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copyfile(source, target)


def _link_or_copy(source: Path, target: Path) -> None:
    """Hard-link when the filesystem allows it (same device), else copy; never alias writes."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
