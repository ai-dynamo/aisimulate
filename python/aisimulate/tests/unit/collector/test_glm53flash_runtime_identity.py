# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Only the exact stock runtimes are admitted; the source manifest binds the producer."""

import hashlib
from pathlib import Path

import pytest
from collector import glm53flash_runtime_identity as identity

pytestmark = pytest.mark.unit

MANIFEST = Path(identity.__file__).parent / "fpm_forward/runtime/glm53flash/runtime-source-sha256.json"


@pytest.mark.parametrize("version", [None, "0.30", "0.30.0", "0.31.0+local", "0.31.1"])
def test_other_vllm_versions_are_not_implicitly_qualified(version):
    with pytest.raises(ValueError, match="unqualified"):
        identity.validate_backend_version("vllm", version)
    with pytest.raises(ValueError, match="unqualified"):
        identity.vllm_source_pins(version, MANIFEST)


def test_stock_runtimes_are_admitted():
    assert identity.validate_backend_version("vllm", "0.31.0") == "0.31.0"
    assert identity.validate_backend_version("sglang", "0.5.20") == "0.5.20"
    with pytest.raises(ValueError, match="unqualified"):
        identity.validate_backend_version("trtllm", "0.31.0")


def test_stock_source_manifest_is_bound_byte_for_byte():
    pins = identity.vllm_source_pins("0.31.0", MANIFEST)
    assert "vllm/models/glm5next/nvidia/sparse_indexer.py" in pins
    assert "vllm/v1/worker/gpu_worker.py" in pins
    assert identity.vllm_source_manifest_sha256("0.31.0", MANIFEST) == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
