# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash model-scoped collector runtime pins (stock vLLM 0.31.0, SGLang 0.5.20)."""

import pytest
from collector.framework_manifest import require_collector_runtime, validate_resolution

pytestmark = pytest.mark.unit

GLM_MODEL_PATHS = ("zai-org/GLM-5.3-Flash", "nvidia/GLM-5.3-Flash-NVFP4")
VLLM_INDEX = "vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
SGLANG_INDEX = "lmsysorg/sglang:v0.5.20@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f"


@pytest.mark.parametrize("model_path", GLM_MODEL_PATHS)
def test_glm_vllm_pin_is_the_stock_0_31_0_release(model_path):
    runtime = require_collector_runtime(
        "vllm", "0.31.0", requested_ops={"gemm", "moe", "compute_scale"}, model_path=model_path
    )
    assert runtime.version == "0.31.0"
    assert runtime.image() == VLLM_INDEX
    assert runtime.source_commit == "db9527a46873454610df6dbedf79a36d6bf1a7f6"
    # Stock image: no overlay build metadata.
    assert runtime.abi is None
    assert runtime.family is None


@pytest.mark.parametrize("model_path", GLM_MODEL_PATHS)
def test_glm_vllm_pin_rejects_the_retired_tail_overlay_runtime(model_path):
    with pytest.raises(RuntimeError, match="requires exactly 0.31.0"):
        require_collector_runtime(
            "vllm", "0.30.0+glm53tail.eb4704514fdf", requested_ops={"gemm"}, model_path=model_path
        )


@pytest.mark.parametrize("model_path", GLM_MODEL_PATHS)
def test_glm_sglang_pin_resolves_every_op_to_0_5_20(model_path):
    runtime = require_collector_runtime(
        "sglang", "0.5.20", requested_ops={"gemm", "moe", "compute_scale"}, model_path=model_path
    )
    assert runtime.version == "0.5.20"
    assert runtime.image() == SGLANG_INDEX
    assert runtime.image("cu130").endswith(SGLANG_INDEX.split("@", 1)[1])
    assert runtime.source_commit == "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"


@pytest.mark.parametrize(
    "backend,wrong_version,expected",
    [("vllm", "0.27.1", "0.31.0"), ("vllm", "0.30.0", "0.31.0"), ("sglang", "0.5.17", "0.5.20")],
)
def test_glm_pin_rejects_other_runtimes(backend, wrong_version, expected):
    with pytest.raises(RuntimeError, match=f"requires exactly {expected}"):
        require_collector_runtime(backend, wrong_version, requested_ops={"gemm"}, model_path=GLM_MODEL_PATHS[0])


@pytest.mark.parametrize(
    "backend,version,model_path",
    [
        ("vllm", "0.24.0", None),
        ("vllm", "0.24.0", "zai-org/GLM-5.2-FP8"),
        ("vllm", "0.27.1", "Qwen/Qwen3.8-2.4T-A95B"),
        ("sglang", "0.5.14", None),
        ("sglang", "0.5.14", "zai-org/GLM-5.2-FP8"),
        ("sglang", "0.5.17", "Qwen/Qwen3.8-2.4T-A95B"),
    ],
)
def test_other_models_keep_their_runtime(backend, version, model_path):
    runtime = require_collector_runtime(backend, version, requested_ops={"gemm"}, model_path=model_path)
    assert runtime.version == version
    assert runtime.abi is None or "overlay_version" not in runtime.abi


def test_manifest_still_resolves_every_registry_op():
    assert validate_resolution() == []
