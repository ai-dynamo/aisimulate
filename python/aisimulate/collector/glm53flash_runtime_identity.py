# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact deployment identities admitted by GLM native evidence consumers.

A local version suffix is a different runtime. Repair candidates remain
unqualified until their native Engine, source and binary evidence is reviewed;
neither a version prefix nor a matching upstream tag admits them implicitly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASELINE_VERSIONS = {"vllm": "0.30.0", "sglang": "0.5.20"}


def validate_backend_version(backend: str, version: str) -> str:
    if backend not in BASELINE_VERSIONS or version != BASELINE_VERSIONS[backend]:
        raise ValueError(f"unqualified GLM backend runtime: {backend} {version!r}")
    return version


def vllm_source_pins(version: str, manifest: Path) -> dict[str, str]:
    """Return the effective source closure for an admitted exact runtime."""
    validate_backend_version("vllm", version)
    return json.loads(manifest.read_bytes())


def validate_vllm_source_identity(producer: dict, manifest: Path) -> dict[str, str]:
    pins = vllm_source_pins(producer.get("vllm_package_version"), manifest)
    # Existing stock producers bind these exact manifest bytes. A future
    # repaired runtime must bind its distinct effective closure here as well.
    if producer.get("runtime_source_manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError("GLM vLLM runtime source manifest differs from the admitted source closure")
    return pins


def validate_runtime_pair(backend: str, calibration: dict, holdout: dict) -> str:
    version = validate_backend_version(backend, calibration.get("backend_version"))
    if holdout.get("backend_version") != version:
        raise ValueError("GLM calibration and holdout use different native runtime versions")
    return version
