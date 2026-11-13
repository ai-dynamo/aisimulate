# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash model-scoped collector runtime pins (stock vLLM 0.32.0; the SGLang 0.5.20 pin
retired 2026-10-30, one release per collector file)."""

import pytest
from collector.framework_manifest import require_collector_runtime, validate_resolution

pytestmark = pytest.mark.unit

GLM_MODEL_PATHS = ("zai-org/GLM-5.3-Flash", "nvidia/GLM-5.3-Flash-NVFP4")
VLLM_INDEX = "vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34"


@pytest.mark.parametrize("model_path", GLM_MODEL_PATHS)
def test_glm_vllm_pin_is_the_stock_0_32_0_release(model_path):
    runtime = require_collector_runtime(
        "vllm", "0.32.0", requested_ops={"gemm", "moe", "kda", "mhc_module"}, model_path=model_path
    )
    assert runtime.version == "0.32.0"
    assert runtime.image() == VLLM_INDEX
    assert runtime.source_commit == "5df63a7203cdb9328d824a27b682e77c7b9928be"
    # Stock image: no overlay build metadata.
    assert runtime.abi is None
    assert runtime.family is None


@pytest.mark.parametrize(
    "backend,wrong_version,expected",
    [("vllm", "0.30.0", "0.32.0"), ("vllm", "0.31.0", "0.32.0")],
)
def test_glm_pin_rejects_other_runtimes(backend, wrong_version, expected):
    with pytest.raises(RuntimeError, match=f"requires exactly {expected}"):
        require_collector_runtime(backend, wrong_version, requested_ops={"gemm"}, model_path=GLM_MODEL_PATHS[0])


@pytest.mark.parametrize(
    "backend,version,model_path",
    [
        ("vllm", "0.32.0", None),  # gemm family pin
        ("vllm", "0.32.0", "zai-org/GLM-5.2-FP8"),
        ("sglang", "0.5.23", "zai-org/GLM-5.2-FP8"),
    ],
)
def test_other_models_keep_their_runtime(backend, version, model_path):
    runtime = require_collector_runtime(backend, version, requested_ops={"gemm"}, model_path=model_path)
    assert runtime.version == version
    assert runtime.abi is None or "overlay_version" not in runtime.abi


def test_manifest_still_resolves_every_registry_op():
    assert validate_resolution() == []
