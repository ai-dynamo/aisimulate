# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline packaging for the canonical learned-model API.

This module never selects an estimator or implements inference. The producer
copies authenticated inputs, and the canonical Rust constructor validates the
result before it is published. Public entry points live in ``fpm_learned``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from aisimulate_core.sdk.fpm_learned import TRAINING_CHECKPOINT_SCHEMA, _file_sha256

BUNDLE_KIND = "aisimulate_learned_bundle"
CATALOGUE_KIND = "aisimulate_learned_catalogue"
SCHEMA_VERSION = 1
PRODUCER_VERSION = "1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value.lower()


def _relative_file(root: Path, relative: Any) -> Path:
    """Require portable paths and refuse symlinks, including intermediate ones."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"invalid bundle dependency path: {relative!r}")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in relative.split("/")) or ":" in path.parts[0]:
        raise ValueError(f"bundle dependency path must be a portable relative path: {relative!r}")
    current = root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"bundle dependencies cannot be symlinks: {current}")
    if not current.is_file():
        raise ValueError(f"bundle dependency is not a regular file: {current}")
    return current


def _copy_file(source: Path, destination: Path, expected: str, purpose: str, root: Path) -> dict[str, Any]:
    """Hash the copied bytes, so concurrent input changes cannot escape validation."""
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"bundle source must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    length = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(block)
            digest.update(block)
            length += len(block)
        writer.flush()
        os.fsync(writer.fileno())
    if digest.hexdigest() != _digest(expected, str(source)):
        raise ValueError(f"bundle source hash mismatch: {source}")
    return {
        "path": destination.relative_to(root).as_posix(),
        "byte_length": length,
        "sha256": digest.hexdigest(),
        "purpose": purpose,
    }


def _provenance(value: Mapping[str, Any], accepted_rows: dict[str, int]) -> dict[str, Any]:
    if value.get("anchor_kind") not in {"a", "b"}:
        raise ValueError("bundle provenance requires explicit anchor_kind 'a' or 'b'")
    result = copy.deepcopy(dict(value))
    for key in ("sources", "evaluation_sources"):
        references = result.get(key, [])
        if not isinstance(references, list) or (key == "sources" and not references):
            raise ValueError("bundle provenance requires nonempty sources and an evaluation_sources list")
        for reference in references:
            if not isinstance(reference, dict) or not isinstance(reference.get("reference"), str):
                raise ValueError(f"{key} entries require a reference and sha256")
            if not reference["reference"].strip():
                raise ValueError(f"{key} reference must not be empty")
            reference["sha256"] = _digest(reference.get("sha256"), f"{key}.sha256")
        result[key] = references
    training = {item["sha256"] for item in result["sources"]}
    if any(item["sha256"] in training for item in result["evaluation_sources"]):
        raise ValueError("training and evaluation source files must be disjoint for bundle publication")
    if result.get("accepted_rows", accepted_rows) != accepted_rows:
        raise ValueError("provenance accepted_rows conflicts with checkpoint training rows")
    if result.get("c_admission_lineage") is not None:
        raise ValueError("training seed bundles cannot claim resumed C admission lineage")
    result["accepted_rows"] = accepted_rows
    result["c_admission_lineage"] = None
    return result


def _request_config(bundle_path: Path, config: Mapping[str, Any] | None, adaptation: str | None) -> dict[str, Any]:
    manifest = _read_json(bundle_path)
    request = copy.deepcopy(dict(config)) if config is not None else copy.deepcopy(manifest["identity"])
    if config is not None and request.get("estimation_mode", "fpm_learned") != "fpm_learned":
        raise ValueError("bundle inspection requires estimation_mode=fpm_learned")
    request["estimation_mode"] = "fpm_learned"
    estimator = request.setdefault("estimator_config", {})
    if "state_snapshot" in estimator:
        raise ValueError("bundle inspection cannot replace a state snapshot source")
    learned = estimator.setdefault("fpm_learned", {})
    for key in ("artifact_path", "checkpoint_path", "resolved_source", "resolved_fallback", "source_sha256"):
        learned.pop(key, None)
    learned.update(bundle_path=str(bundle_path), source_sha256=_file_sha256(bundle_path))
    if adaptation is not None:
        learned["adaptation"] = adaptation
    return request


def _normalize_bundle(bundle_path: Path, config: Mapping[str, Any] | None, adaptation: str | None) -> dict[str, Any]:
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    return RustForwardPassPerfModel.normalize_config(_request_config(bundle_path, config, adaptation))


def inspect_bundle(
    bundle_path: str | os.PathLike[str], *, config: Mapping[str, Any] | None = None, adaptation: str | None = None
) -> dict[str, Any]:
    """Validate through Rust and return a complete pinned prediction configuration.

    An optional configuration supplies the expected deployment identity and
    explicit controls. Its source is replaced by this bundle; Rust checks the
    identity and every constructor dependency. This does not admit observations.
    """
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    path = Path(bundle_path).resolve(strict=True)
    resolved = _normalize_bundle(path, config, adaptation)
    model = RustForwardPassPerfModel.best_available(resolved)
    try:
        return model.diagnostics()["provenance"]["config"]
    finally:
        model.close()


def _publish_directory(staging: Path, destination: Path) -> None:
    # Reserve with mkdir rather than replace a concurrently created directory.
    destination.mkdir()
    try:
        os.rename(staging, destination)
    except OSError:
        destination.rmdir()
        raise


def _copy_prior(prior: Mapping[str, Any], staging: Path, dependencies: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve original-A evidence bytes and bind relocated files explicitly."""
    result = copy.deepcopy(dict(prior))
    original = result["original_a"]
    path = Path(original["manifest_path"]).resolve(strict=True)
    expected = _digest(original["source_sha256"], "original_a.source_sha256")
    destination = staging / "prior" / "original-a.json"
    dependencies.append(_copy_file(path, destination, expected, "original_a_manifest", staging))
    manifest = _read_json(destination)
    references = list(manifest["provenance"]["sources"]) + list(manifest["seed_artifacts"])
    for seed in manifest["seed_artifacts"]:
        if seed.get("checkpoint_manifest_sha256") is not None:
            references.append(
                {
                    "path": str(Path(seed["path"]).parent / "manifest.json"),
                    "sha256": seed["checkpoint_manifest_sha256"],
                }
            )
    remap = []
    seen: dict[str, str] = {}
    copied: set[str] = set()
    for reference in references:
        original_path = reference["path"]
        digest = _digest(reference["sha256"], "original_a dependency")
        if original_path in seen:
            if seen[original_path] != digest:
                raise ValueError(f"conflicting original-A dependency hashes: {original_path}")
            continue
        seen[original_path] = digest
        source = Path(original_path)
        if not source.is_absolute():
            source = path.parent / source
        relative = f"prior/objects/{digest}/{source.name}"
        if relative not in copied:
            dependencies.append(_copy_file(source, staging / relative, digest, "original_a_evidence", staging))
            copied.add(relative)
        remap.append({"original_path": original_path, "path": relative, "sha256": digest})
    result["original_a"] = {"manifest_path": "prior/original-a.json", "source_sha256": expected}
    result["dependency_remap"] = remap
    return result


def export_bundle(
    checkpoint_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    provenance: Mapping[str, Any],
    recipe_version: str | None = None,
    prior: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Package a pristine A/B training checkpoint, validate it, and return its config.

    The output directory must be new. Checkpoint bytes, accepted-row provenance
    and hashes are preserved. Declaring anchor A does not manufacture an
    interpolation prior or grant source redistribution rights.
    """
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    destination = Path(output_dir).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"bundle destination already exists: {destination}")
    manifest_path = _relative_file(checkpoint, "manifest.json")
    manifest_hash = _file_sha256(manifest_path)
    checkpoint_manifest = _read_json(manifest_path)
    if checkpoint_manifest.get("schema") != TRAINING_CHECKPOINT_SCHEMA:
        raise ValueError("bundle producer requires a pristine training checkpoint")
    if checkpoint_manifest.get("schema_version") != 1:
        raise ValueError("unsupported training checkpoint version")
    stores = checkpoint_manifest["stores"]
    accepted = {name: store["row_count"] for name, store in stores.items()}
    source_provenance = _provenance(provenance, accepted)
    artifact_name = checkpoint_manifest["artifact_file"]
    artifact = _read_json(_relative_file(checkpoint, artifact_name))
    recipe_version = "sklearn_training_checkpoint_v1" if recipe_version is None else recipe_version
    if not recipe_version.strip():
        raise ValueError("recipe_version must not be empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.tmp-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        files = [("manifest.json", manifest_hash, "training_checkpoint_manifest")]
        files.append((artifact_name, checkpoint_manifest["artifact_sha256"], "learned_forest"))
        files.extend((store["rows_file"], store["rows_sha256"], "training_rows") for store in stores.values())
        dependencies = [
            _copy_file(_relative_file(checkpoint, name), staging / "seed" / name, digest, purpose, staging)
            for name, digest, purpose in files
        ]
        bundled_prior = _copy_prior(prior, staging, dependencies) if prior is not None else None
        bundle = {
            "kind": BUNDLE_KIND,
            "schema_version": SCHEMA_VERSION,
            "producer_version": PRODUCER_VERSION,
            "recipe_version": recipe_version,
            "identity": checkpoint_manifest["identity"],
            "seed": {
                "kind": "training_checkpoint",
                "path": "seed",
                "sha256": manifest_hash,
                "adaptations": ["online", "frozen", "causal_selector"],
                "covered_stores": sorted(stores),
            },
            "training_recipe": {
                "name": artifact["metadata"]["model"],
                "features": checkpoint_manifest["features"],
                "target": checkpoint_manifest["target"],
                "trainer_version": checkpoint_manifest["sklearn_version"],
                "hyperparameters": artifact["metadata"]["hyperparameters"],
            },
            "provenance": source_provenance,
            "prior": bundled_prior,
            "dependencies": sorted(dependencies, key=lambda item: item["path"]),
        }
        with (staging / "bundle.json").open("xb") as handle:
            handle.write(_json_bytes(bundle))
            handle.flush()
            os.fsync(handle.fileno())
        inspect_bundle(staging / "bundle.json", config=config)
        _publish_directory(staging, destination)
    return inspect_bundle(destination / "bundle.json", config=config)


def train_bundle(
    iterations: Any,
    worker_type: str,
    *,
    output_dir: str | os.PathLike[str],
    identity: dict[str, Any],
    provenance: Mapping[str, Any],
    prior: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    **train_kwargs: Any,
) -> dict[str, Any]:
    """Fit accepted A/B rows with the existing trainer, then publish a bundle.

    ``train_kwargs`` are the existing ``train`` controls. Training/evaluation
    separation and collection acceptance remain the producer's responsibility;
    ``iterations`` contains training rows only. Evaluation files are recorded in
    provenance and are never read or admitted by this function.
    """
    from aisimulate_core.sdk.fpm_learned import train

    destination = Path(output_dir).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"bundle destination already exists: {destination}")
    if "online_checkpoint_dir" in train_kwargs:
        raise ValueError("train_bundle owns online_checkpoint_dir")
    # Validate the caller's claims before doing potentially expensive fitting.
    preliminary = dict(provenance)
    preliminary.pop("accepted_rows", None)
    _provenance(preliminary, {})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".learned-training-", dir=destination.parent) as temporary:
        checkpoint = Path(temporary) / "checkpoint"
        train(iterations, worker_type, identity=identity, online_checkpoint_dir=checkpoint, **train_kwargs)
        return export_bundle(checkpoint, destination, provenance=provenance, prior=prior, config=config)


def _copy_bundle(source_manifest: Path, staging: Path, expected_manifest_digest: str) -> None:
    manifest = _read_json(source_manifest)
    if manifest.get("kind") != BUNDLE_KIND or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported learned bundle manifest")
    seen = {"bundle.json"}
    for item in manifest["dependencies"]:
        relative = item["path"]
        if relative in seen:
            raise ValueError(f"duplicate bundle dependency: {relative}")
        seen.add(relative)
        source = _relative_file(source_manifest.parent, relative)
        copied = _copy_file(source, staging / relative, item["sha256"], item["purpose"], staging)
        if copied["byte_length"] != item["byte_length"]:
            raise ValueError(f"bundle dependency length mismatch: {relative}")
    _copy_file(source_manifest, staging / "bundle.json", expected_manifest_digest, "bundle_manifest", staging)


def install_bundle(
    bundle_path: str | os.PathLike[str],
    systems_root: str | os.PathLike[str],
    *,
    make_default: bool = False,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Install a validated immutable seed and atomically register its catalogue entry.

    The first bundle for an exact identity becomes its default. Installing a
    second preserves that designation unless ``make_default=True``. Existing
    pinned configurations continue to refer to the original immutable bundle.
    Concurrent installers fail with an actionable lock error; they never lose
    one another's catalogue entries.
    """
    source = Path(bundle_path).resolve(strict=True)
    inspect_bundle(source, config=config)
    manifest = _read_json(source)
    if manifest["seed"]["kind"] != "training_checkpoint" or manifest["provenance"].get("c_admission_lineage"):
        raise ValueError("catalogue publication requires a pristine A/B training seed")
    identity = manifest["identity"]
    if '"unknown"' in _json_bytes(identity).decode("utf-8"):
        raise ValueError("catalogue publication requires a complete known deployment identity")
    digest = _file_sha256(source)
    catalogue_root = Path(systems_root).resolve() / "fpm_learned"
    catalogue_root.mkdir(parents=True, exist_ok=True)
    lock = catalogue_root / ".install.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"another bundle installation owns {lock}; retry after it completes") from exc
    try:
        index_path = catalogue_root / "index.json"
        catalogue = (
            _read_json(index_path)
            if index_path.exists()
            else {"kind": CATALOGUE_KIND, "schema_version": SCHEMA_VERSION, "entries": []}
        )
        if catalogue.get("kind") != CATALOGUE_KIND or catalogue.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported learned catalogue")
        entries = catalogue["entries"]
        matches = [item for item in entries if item["identity"] == identity]
        if matches and sum(item["default"] is True for item in matches) != 1:
            raise ValueError("catalogue identity must have exactly one designated default")
        relative = f"bundles/{digest}/bundle.json"
        installed = catalogue_root / relative
        installed.parent.parent.mkdir(parents=True, exist_ok=True)
        if installed.parent.exists():
            if _file_sha256(_relative_file(catalogue_root, relative)) != digest:
                raise ValueError("immutable installed bundle has changed")
        else:
            with tempfile.TemporaryDirectory(prefix=".install-", dir=installed.parent.parent) as temporary:
                staging = Path(temporary)
                _copy_bundle(source, staging, digest)
                _publish_directory(staging, installed.parent)
        installed_config = inspect_bundle(installed, config=config)
        if make_default:
            for item in matches:
                item["default"] = False
        existing = next((item for item in matches if item["sha256"] == digest), None)
        if existing is None:
            entries.append(
                {
                    "identity": identity,
                    "recipe_version": manifest["recipe_version"],
                    "bundle_path": relative,
                    "sha256": digest,
                    "default": make_default or not matches,
                }
            )
        elif make_default:
            existing["default"] = True
        with tempfile.NamedTemporaryFile(prefix=".index-", dir=catalogue_root, delete=False) as handle:
            temporary_index = Path(handle.name)
            try:
                handle.write(_json_bytes(catalogue))
                handle.flush()
                os.fsync(handle.fileno())
                os.replace(temporary_index, index_path)
            finally:
                temporary_index.unlink(missing_ok=True)
        return installed_config
    finally:
        os.close(lock_fd)
        lock.unlink()
