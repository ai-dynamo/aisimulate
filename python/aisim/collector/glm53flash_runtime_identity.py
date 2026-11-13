# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact backend runtimes admitted by the GLM-5.3-Flash native FPM producers.

A local version suffix is a different runtime and is rejected; neither a
version prefix nor a matching upstream tag admits it implicitly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Stock vLLM v0.31.0 (tag commit db9527a46873454610df6dbedf79a36d6bf1a7f6), no local patch.
BASELINE_VERSIONS = {"vllm": "0.31.0", "sglang": "0.5.20"}


def validate_backend_version(backend: str, version: str) -> str:
    if backend not in BASELINE_VERSIONS or version != BASELINE_VERSIONS[backend]:
        raise ValueError(f"unqualified GLM backend runtime: {backend} {version!r}")
    return version


def vllm_source_pins(version: str, manifest: Path) -> dict[str, str]:
    """Return the pinned vLLM/Dynamo source hashes for the admitted runtime."""
    validate_backend_version("vllm", version)
    return json.loads(manifest.read_bytes())


def vllm_source_manifest_sha256(version: str, manifest: Path) -> str:
    validate_backend_version("vllm", version)
    return hashlib.sha256(manifest.read_bytes()).hexdigest()
