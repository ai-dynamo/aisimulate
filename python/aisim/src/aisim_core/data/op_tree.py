# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the whole op-latency tree (Hugging Face layout) into one ``systems/`` root (AISIM-2063).

:class:`~aisim_core.data.loaders.OpLatencyLoader` materializes one catalog entry (one
``system/family/backend/version``). A predictor needs many entries at once, so this module walks the
category index, verifies every selected entry through that loader and writes a single root that
``systems_paths`` / ``AICONFIGURATOR_SYSTEMS_PATH`` can point at, next to the packaged path (which it never
modifies). The two lookup YAML files (``perf_data_reuse_manifest.yaml``, ``query_versions.yaml``) are rebuilt
byte-for-byte from the source text embedded in their published Parquet tables. Packaged system specs and other
non-op files are copied from the packaged systems directory because the op-latency category does not publish them.

Offline: reads only a local dataset-layout directory. Nothing here is wired into a default path.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from aisim_core.data import CategoryIndex, DatasetIndex, Entry, _file, _verify
from aisim_core.data.fetch import DATASET_INDEX_PATH
from aisim_core.data.loaders import LoaderError, OpLatencyLoader
from aisim_core.data.op_schema import DATA_PREFIX, QUERY_VERSIONS_NAME, REUSE_MANIFEST_NAME
from aisim_core.data.systems import PACKAGED_SYSTEMS_DIR

_DATA_DIR = "data"
_SOURCE_YAML_KEY = b"aisim.source_yaml"


@dataclass(frozen=True)
class OpTreeLoad:
    """What :func:`load_op_latency_tree` wrote."""

    root: Path
    entries: int
    files: int
    bytes: int
    lookup_files: tuple[str, ...]
    runtime_paths: frozenset[str]  # every runtime input the manifests declare, relative to ``root``
    seconds: float


def op_latency_entries(
    tree: Path,
    *,
    hardware: Iterable[str] | None = None,
    framework: Iterable[str] | None = None,
) -> list[Entry]:
    """Catalog entries of the ``op-latency`` category under ``tree`` (verified index files)."""
    tree = Path(tree)
    dataset = DatasetIndex.model_validate_json(_file(tree, DATASET_INDEX_PATH).read_bytes())
    reference = dataset.categories.get("op-latency")
    if reference is None:
        raise LoaderError(f"{tree} has no op-latency category")
    index = CategoryIndex.model_validate_json(_verify(tree, reference.index).read_bytes())
    hardware, framework = set(hardware or ()), set(framework or ())
    return [
        e
        for e in index.entries
        if (not hardware or e.identity.hardware in hardware) and (not framework or e.identity.framework in framework)
    ]


def load_op_latency_tree(
    tree: Path,
    destination: Path,
    *,
    hardware: Iterable[str] | None = None,
    framework: Iterable[str] | None = None,
    packaged_systems: Path | None = PACKAGED_SYSTEMS_DIR,
) -> OpTreeLoad:
    """Verify the selected entries of ``tree`` and write one ``systems/`` root at ``destination``.

    ``destination`` must not exist; a failed call removes it. Every artifact is hash-checked in the tree and again
    after the copy. ``packaged_systems=None`` skips copying specs and other non-op files (the root then holds only
    what the tree publishes).
    """
    started = time.perf_counter()
    tree, destination = Path(tree), Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; choose a new directory")
    loader = OpLatencyLoader()
    entries = op_latency_entries(tree, hardware=hardware, framework=framework)
    if not entries:
        raise LoaderError("no op-latency entries selected")
    destination.mkdir(parents=True)
    files = size = 0
    runtime: set[str] = set()
    try:
        for entry in sorted(entries, key=lambda e: e.manifest.path):
            manifest = loader.read_manifest(tree, entry.manifest.path)
            if manifest.identity.model is not None:
                raise LoaderError("operation table storage must remain model-independent")
            for artifact in manifest.artifacts:
                if artifact.runtime_path is None:
                    continue
                target = destination / artifact.runtime_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_verify(tree, artifact), target)
                _verify(destination, artifact.model_copy(update={"path": artifact.runtime_path}))
                files, size = files + 1, size + artifact.bytes
                runtime.add(artifact.runtime_path)
        lookups = _write_lookup_yaml(tree, destination)
        if packaged_systems is not None:
            for source in sorted(Path(packaged_systems).iterdir()):
                if source.name == _DATA_DIR or (destination / source.name).exists():
                    continue
                if source.is_dir():
                    shutil.copytree(source, destination / source.name)
                else:
                    shutil.copyfile(source, destination / source.name)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return OpTreeLoad(
        destination, len(entries), files, size, lookups, frozenset(runtime), time.perf_counter() - started
    )


def _write_lookup_yaml(tree: Path, destination: Path) -> tuple[str, ...]:
    """Rebuild the packaged lookup YAML files from the source text embedded in their Parquet tables."""
    written = []
    for name in (REUSE_MANIFEST_NAME, QUERY_VERSIONS_NAME):
        path = tree / DATA_PREFIX / f"{name}.parquet"
        if not path.is_file() or path.is_symlink():
            continue
        metadata = pq.read_metadata(path).metadata or {}
        text = metadata.get(_SOURCE_YAML_KEY)
        expected = metadata.get(b"aisim.source_sha256")
        if text is None or expected is None or hashlib.sha256(text).hexdigest() != expected.decode():
            raise LoaderError(f"{name}.parquet does not embed its source YAML intact")
        (destination / f"{name}.yaml").write_bytes(text)
        written.append(f"{name}.yaml")
    return tuple(written)
