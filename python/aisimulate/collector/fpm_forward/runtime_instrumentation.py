# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect and freeze campaign-local observer files without importing code."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

INSTRUMENTATION_SCHEMA = "aisimulate-runtime-instrumentation/v1"
OBSERVATION_SCHEMA = "aisimulate-runtime-observation/v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} requires a lowercase SHA-256")
    return value


def relative_path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(char) < 32 for char in value)
        or "\\" in value
        or ":" in value
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"artifact paths must be normalized relative paths: {value!r}")
    return value


def contained_file(root: Path, value: Any) -> Path:
    name = relative_path(value)
    current = root
    for part in PurePosixPath(name).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"artifact path contains a symlink: {name}")
    if not current.is_file() or not current.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"artifact is not a contained regular file: {name}")
    return current


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError(f"duplicate or non-string object key: {key!r}")
        result[key] = value
    return result


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _yaml_mapping(loader: _UniqueSafeLoader, node: yaml.MappingNode) -> dict[str, Any]:
    return _unique_pairs([(loader.construct_object(k), loader.construct_object(v)) for k, v in node.value])


_UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
        canonical_json(value)
    except (OSError, UnicodeError, TypeError, RecursionError, ValueError) as error:
        raise ValueError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


@dataclass(frozen=True, slots=True)
class InstrumentationBundle:
    root: Path
    manifest_path: Path
    _manifest_json: str
    _files_json: str
    sha256: str

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest_json)

    @property
    def files(self) -> dict[str, str]:
        return json.loads(self._files_json)


def load_instrumentation(path: str | Path, expected_version: str | None = None) -> InstrumentationBundle:
    """Parse a safe manifest and hash its complete declared bundle, without imports."""
    path = Path(path).absolute()
    if path.is_symlink():
        raise ValueError("instrumentation manifest cannot be a symlink")
    try:
        manifest = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueSafeLoader)
        canonical_json(manifest)
    except (OSError, UnicodeError, yaml.YAMLError, TypeError, RecursionError, ValueError) as error:
        raise ValueError(f"invalid instrumentation manifest: {error}") from error
    required = {"schema_version", "runtime", "files", "worker_class", "scheduler_class", "observation_schema"}
    if (
        not isinstance(manifest, dict)
        or not required <= manifest.keys()
        or manifest.keys() - required - {"source_notes"}
    ):
        raise ValueError("instrumentation manifest has missing or unknown fields")
    if manifest["schema_version"] != INSTRUMENTATION_SCHEMA or manifest["observation_schema"] != OBSERVATION_SCHEMA:
        raise ValueError("unsupported instrumentation or observation schema")
    runtime = manifest["runtime"]
    if (
        not isinstance(runtime, dict)
        or not {"framework", "version", "source_revision"} <= runtime.keys()
        or runtime.keys() - {"framework", "version", "source_revision", "source_files"}
        or runtime["framework"] != "vllm"
        or not isinstance(runtime["version"], str)
        or not runtime["version"].strip()
        or not isinstance(runtime["source_revision"], str)
        or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", runtime["source_revision"])
    ):
        raise ValueError("instrumentation requires a pinned vLLM version and exact source revision")
    if expected_version is not None and runtime["version"] != expected_version:
        raise ValueError("instrumentation runtime version differs from the requested version")
    if "source_files" in runtime:
        if not isinstance(runtime["source_files"], dict) or not runtime["source_files"]:
            raise ValueError("runtime source_files must contain installation-relative paths and hashes")
        for name, digest in runtime["source_files"].items():
            relative_path(name)
            validate_sha256(digest, "runtime source file")
    names = manifest["files"]
    if not isinstance(names, list) or not names:
        raise ValueError("instrumentation requires declared files")
    for name in names:
        relative_path(name)
    if len({name.casefold() for name in names}) != len(names) or "manifest.json" in names:
        raise ValueError("instrumentation file paths must be unique and cannot use reserved manifest.json")
    files = {name: sha256_bytes(contained_file(path.parent, name).read_bytes()) for name in names}
    modules: dict[str, str] = {}
    for name in names:
        if not name.endswith(".py"):
            continue
        parts = list(PurePosixPath(name).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        module = ".".join(parts)
        if not module or any(not part.isidentifier() for part in parts):
            raise ValueError(f"instrumentation module path is ambiguous: {name}")
        if module in modules:
            raise ValueError(f"instrumentation module is ambiguous: {module}")
        modules[module] = name
    for field in ("worker_class", "scheduler_class"):
        value = manifest[field]
        parts = value.split(".") if isinstance(value, str) else []
        if len(parts) < 2 or any(not part.isidentifier() for part in parts) or ".".join(parts[:-1]) not in modules:
            raise ValueError(f"{field} must reference an unambiguous declared Python module and class")
        for depth in range(1, len(parts) - 1):
            if "/".join(parts[:depth]) + "/__init__.py" not in files:
                raise ValueError(f"{field} package must have a declared __init__.py")
    if "source_notes" in manifest and (
        not isinstance(manifest["source_notes"], str) or manifest["source_notes"] not in files
    ):
        raise ValueError("source_notes must name a declared bundle file")
    identity = sha256_bytes(canonical_json({"manifest": manifest, "files": files}).encode())
    return InstrumentationBundle(path.parent, path, canonical_json(manifest), canonical_json(files), identity)


def freeze_instrumentation(bundle: InstrumentationBundle, target: str | Path) -> InstrumentationBundle:
    """Copy verified content to a fresh attempt; never rewrite an existing bundle."""
    target = Path(target).absolute()
    if target.exists() or target.is_symlink():
        raise ValueError("instrumentation freezing requires a fresh directory")
    if load_instrumentation(bundle.manifest_path).sha256 != bundle.sha256:
        raise ValueError("instrumentation changed after it was loaded")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".instrumentation-", dir=target.parent))
    try:
        for name, digest in bundle.files.items():
            raw = contained_file(bundle.root, name).read_bytes()
            if sha256_bytes(raw) != digest:
                raise ValueError(f"instrumentation file changed while freezing: {name}")
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        (temporary / "manifest.json").write_text(bundle._manifest_json + "\n", encoding="utf-8")
        if load_instrumentation(temporary / "manifest.json").sha256 != bundle.sha256:
            raise ValueError("instrumentation bundle changed while freezing")
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return load_instrumentation(target / "manifest.json")
