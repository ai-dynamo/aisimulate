# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unknown repair suffixes and cross-runtime evidence must fail admission."""

import hashlib
from pathlib import Path

import pytest

from collector import glm53flash_runtime_identity as identity
from collector.fpm_forward import hybrid_artifact

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("version", [None, "0.30", "0.30.0+unknown", "0.30.0+glm53kpool.bf5f6b0e689d", "0.31.0"])
def test_candidate_versions_are_not_implicitly_qualified(version):
    with pytest.raises(ValueError, match="unqualified"):
        identity.validate_backend_version("vllm", version)


def test_actual_source_closure_is_required_even_for_stock_version():
    manifest = Path(hybrid_artifact.__file__).parent / "runtime/glm53flash/runtime-source-sha256.json"
    producer = {
        "vllm_package_version": "0.30.0",
        "runtime_source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    assert "vllm/model_executor/layers/sparse_attn_indexer_kpool.py" in identity.validate_vllm_source_identity(
        producer, manifest
    )
    producer["runtime_source_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source manifest"):
        identity.validate_vllm_source_identity(producer, manifest)


def test_holdout_runtime_must_equal_calibration_runtime():
    calibration = {"backend_version": "0.30.0"}
    assert identity.validate_runtime_pair("vllm", calibration, calibration) == "0.30.0"
    with pytest.raises(ValueError, match="different native runtime"):
        identity.validate_runtime_pair("vllm", calibration, {"backend_version": "0.30.0+other"})
