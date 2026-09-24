# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact deployment identities admitted by GLM native evidence consumers.

A local version suffix is a different runtime. Repair candidates remain
unqualified until their native Engine, source and binary evidence is reviewed;
neither a version prefix nor a matching upstream tag admits them implicitly.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

BASELINE_VERSIONS = {"vllm": "0.30.0", "sglang": "0.5.20"}
VLLM_KPOOL_CANDIDATE = "0.30.0+glm53kpool.bf5f6b0e689d"
# Deliberately empty until native Engine qualification has passed and its
# immutable receipt has been reviewed. A build receipt or version suffix alone
# cannot promote a repair. Values will be reviewed qualification receipt hashes.
ADMITTED_VLLM_REPAIRS: dict[str, str] = {}
_BUILD_SHA256 = "3b72d70800e2ea244944580c1ce6a4faaa3dedf68af41b323aa690999abd9444"
_WHEEL_SHA256 = "a3b63cb3c95cf976f717077102e8172a33501c7d05092bc84cc58e3aaef47d36"
_ENGINE_IDENTITY_SHA256 = "d412233edffae84ae4b36a2e44d08bc4d3652a7e7f2bc62e8fc1193967a1cb22"
_V2_SOURCE_SHA256 = "48a6f6689d176be79d0ac38db0e68078050e63f73669c649b332d32dff076f1e"


def _canonical_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _candidate_build() -> dict:
    path = Path(__file__).parent / "fpm_forward/runtime/glm53flash_vllm_kpool_candidate/build-receipt.json"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _BUILD_SHA256:
        raise ValueError("GLM repair build receipt differs from the reviewed immutable build")
    build = json.loads(raw)
    if build["version"] != VLLM_KPOOL_CANDIDATE or build["wheel_sha256"] != _WHEEL_SHA256:
        raise ValueError("GLM repair wheel identity differs from its reviewed build")
    return build


def validate_backend_version(backend: str, version: str) -> str:
    repaired = backend == "vllm" and version == VLLM_KPOOL_CANDIDATE and version in ADMITTED_VLLM_REPAIRS
    if backend not in BASELINE_VERSIONS or (version != BASELINE_VERSIONS[backend] and not repaired):
        raise ValueError(f"unqualified GLM backend runtime: {backend} {version!r}")
    return version


def vllm_source_pins(version: str, manifest: Path) -> dict[str, str]:
    """Return the effective source closure for an admitted exact runtime."""
    validate_backend_version("vllm", version)
    pins = json.loads(manifest.read_bytes())
    if version == VLLM_KPOOL_CANDIDATE:
        patch = _candidate_build()["patch"]
        if pins.get(patch["source_path"]) != patch["base_sha256"]:
            raise ValueError("GLM repair source base differs from its reviewed build")
        pins[patch["source_path"]] = patch["patched_sha256"]
        root = Path(__file__).parent / "fpm_forward/runtime/glm53flash_vllm_kpool_candidate"
        native_identity = (root / "qualification/expected-runtime.json").read_bytes()
        v2_source = (root / "v2-source-sha256.json").read_bytes()
        if (
            hashlib.sha256(native_identity).hexdigest() != _ENGINE_IDENTITY_SHA256
            or hashlib.sha256(v2_source).hexdigest() != _V2_SOURCE_SHA256
        ):
            raise ValueError("GLM repaired Engine/V2 source manifest differs")
        for name, sha in {**json.loads(native_identity)["source_pins"], **json.loads(v2_source)}.items():
            if name in pins and pins[name] != sha:
                raise ValueError("GLM repair source closure has conflicting identities")
            pins[name] = sha
    return pins


def vllm_source_manifest_sha256(version: str, manifest: Path) -> str:
    pins = vllm_source_pins(version, manifest)
    if version == BASELINE_VERSIONS["vllm"]:
        # Preserve the already published stock protocol byte-for-byte.
        return hashlib.sha256(manifest.read_bytes()).hexdigest()
    return _canonical_sha256(pins)


def vllm_runtime_closure(version: str, manifest: Path) -> dict | None:
    """Expected repaired-runtime files; construction grants no admission."""
    sources = vllm_source_pins(version, manifest)
    if version == BASELINE_VERSIONS["vllm"]:
        return None
    build = _candidate_build()
    return {
        "schema_version": 1,
        "backend_version": version,
        "wheel_sha256": build["wheel_sha256"],
        "build_receipt_sha256": _BUILD_SHA256,
        "qualification_receipt_sha256": ADMITTED_VLLM_REPAIRS[version],
        "runtime_source_manifest_sha256": vllm_source_manifest_sha256(version, manifest),
        "files": {
            **{path: sha for path, sha in sources.items() if path.startswith("vllm/")},
            **build["unchanged_native_binaries"],
        },
    }


def observe_vllm_runtime_closure(version: str, manifest: Path) -> dict | None:
    """Read actual per-worker source and native binary bytes before timing."""
    closure = vllm_runtime_closure(version, manifest)
    if closure is None:
        return None
    import vllm

    if vllm.__version__ != version or importlib.metadata.version("vllm") != version:
        raise ValueError("GLM repair imported package and distribution versions differ")
    package = Path(vllm.__file__).resolve().parent
    observed = {}
    for name, expected in closure["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "vllm":
            raise ValueError("GLM repair closure contains a non-package file")
        path = package.joinpath(*relative.parts[1:]).resolve(strict=True)
        if not path.is_relative_to(package):
            raise ValueError("GLM repair closure file resolves outside the imported package")
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"GLM repair source/native binary differs: {name}")
        observed[name] = actual
    return {"contract_sha256": _canonical_sha256(closure), "observed_files": observed}


def validate_vllm_runtime_closure(version: str, manifest: Path, observed: dict | None) -> None:
    closure = vllm_runtime_closure(version, manifest)
    expected = (
        None if closure is None else {"contract_sha256": _canonical_sha256(closure), "observed_files": closure["files"]}
    )
    if observed != expected:
        raise ValueError("GLM vLLM native hardware repair source/binary closure is missing or differs")


def vllm_unaligned_prefill_admitted(version: str) -> bool:
    validate_backend_version("vllm", version)
    return version in ADMITTED_VLLM_REPAIRS


def validate_vllm_source_identity(producer: dict, manifest: Path) -> dict[str, str]:
    pins = vllm_source_pins(producer.get("vllm_package_version"), manifest)
    if producer.get("runtime_source_manifest_sha256") != vllm_source_manifest_sha256(
        producer.get("vllm_package_version"), manifest
    ):
        raise ValueError("GLM vLLM runtime source manifest differs from the admitted source closure")
    return pins


def validate_runtime_pair(backend: str, calibration: dict, holdout: dict) -> str:
    version = validate_backend_version(backend, calibration.get("backend_version"))
    if holdout.get("backend_version") != version:
        raise ValueError("GLM calibration and holdout use different native runtime versions")
    return version
