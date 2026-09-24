# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 DSA checkpoints on vLLM render an explicit fp8 kv-cache dtype.

vLLM 0.29.0 resolves `--kv-cache-dtype auto` for the modelopt NVFP4 DSA
artifacts (hf_quant kv_cache_quant_algo FP8) to the literal fp8_e4m3 and its
sparse-MLA selector rejects that spelling; the same engine with an explicit
`fp8` loads FlashMLASparseImpl. The prescription is keyed on the checkpoint
architecture and the task's nvfp4 GEMM quant mode — never on model names.
"""
from __future__ import annotations

import pytest

from aisimulate.generator import utils
from aisimulate.generator.rendering.engine import render_backend_parameters

pytestmark = pytest.mark.unit

_ARCH = {
    "nvidia/GLM-5.3-NVFP4": "GlmMoeDsaForCausalLM",
    "nvidia/GLM-5.2-NVFP4": "GlmMoeDsaForCausalLM",
    "nvidia/DeepSeek-V3.2-NVFP4": "DeepseekV32ForCausalLM",
    "zai-org/GLM-5.3": "GlmMoeDsaForCausalLM",
    "nvidia/Llama-3.3-70B-Instruct-FP4": "LlamaForCausalLM",
    "nvidia/DeepSeek-R1-0528-FP4": "DeepseekV3ForCausalLM",
    "local/unbundled": None,
}


_MODELOPT_FP8_KV = {"quant_method": "modelopt", "quant_algo": "NVFP4",
                    "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"}}
# what utils._bundled_quantization returns: config.json quantization block
# merged with the SDK-attached hf_quant_config (kv_cache_quant_algo lives there)
_QUANT = {
    "nvidia/GLM-5.2-NVFP4": _MODELOPT_FP8_KV,          # config.json carries the KV scheme
    "nvidia/GLM-5.3-NVFP4": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                             "kv_cache_quant_algo": "FP8"},            # hf_quant only
    "nvidia/DeepSeek-V3.2-NVFP4": {"quant_method": "nvfp4", "quant_algo": "NVFP4",
                                   "kv_cache_quant_algo": "FP8"},      # hf_quant only
    "nvidia/GLM-5-NVFP4-unquant-bundle": {"quant_method": "nvfp4"},    # neither spelling present
    "zai-org/GLM-5.3": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]},
    "nvidia/Llama-3.3-70B-Instruct-FP4": _MODELOPT_FP8_KV,
}
_ARCH["nvidia/GLM-5-NVFP4-unquant-bundle"] = "GlmMoeDsaForCausalLM"


@pytest.fixture(autouse=True)
def _stub_sdk_lookups(monkeypatch):
    monkeypatch.setattr(utils, "_model_architecture", lambda model_path: _ARCH.get(model_path))
    monkeypatch.setattr(utils, "_bundled_quantization", lambda model_path: _QUANT.get(model_path))


@pytest.mark.parametrize(
    ("backend", "model", "gemm", "expected"),
    [
        # optimized path: the task's nvfp4 GEMM quant mode is the artifact fact
        ("vllm", "nvidia/GLM-5.3-NVFP4", "nvfp4", "fp8"),
        ("vllm", "nvidia/DeepSeek-V3.2-NVFP4", "nvfp4", "fp8"),
        # naive path (no quant mode): the artifact's fp8 KV fact, either spelling
        ("vllm", "nvidia/GLM-5.2-NVFP4", None, "fp8"),             # config.json kv_cache_scheme
        ("vllm", "nvidia/GLM-5.3-NVFP4", None, "fp8"),             # hf_quant kv_cache_quant_algo
        ("vllm", "nvidia/DeepSeek-V3.2-NVFP4", None, "fp8"),
        ("vllm", "nvidia/GLM-5-NVFP4-unquant-bundle", None, None),  # no KV fact anywhere: no prescription
        ("vllm", "zai-org/GLM-5.3", "fp8_block", None),          # fp8 DSA artifact: auto is fine
        ("vllm", "zai-org/GLM-5.3", None, None),
        ("vllm", "nvidia/Llama-3.3-70B-Instruct-FP4", "nvfp4", None),  # NVFP4 but not DSA
        ("vllm", "nvidia/Llama-3.3-70B-Instruct-FP4", None, None),
        ("vllm", "nvidia/DeepSeek-R1-0528-FP4", "nvfp4", None),  # plain MLA, not DSA
        ("vllm", "local/unbundled", "nvfp4", None),               # unresolvable config: no prescription
        ("sglang", "nvidia/GLM-5.2-NVFP4", "nvfp4", None),        # vllm-only spelling gap
        ("trtllm", "nvidia/GLM-5.2-NVFP4", None, None),
    ],
)
def test_prescription_keyed_on_dsa_architecture_and_an_artifact_fact(backend, model, gemm, expected):
    assert utils.vllm_dsa_kv_cache_dtype(backend, model, gemm) == expected


def test_artifact_kv_scheme_detection():
    assert utils._artifact_pins_fp8_kv(_MODELOPT_FP8_KV)
    assert utils._artifact_pins_fp8_kv({"kv_cache_quant_algo": "FP8"})
    assert not utils._artifact_pins_fp8_kv({"quant_method": "fp8", "fmt": "e4m3"})
    assert not utils._artifact_pins_fp8_kv(None)


class _FakeTask:
    def __init__(self, backend, model):
        self.primary_backend_name = backend
        self.primary_model_path = model


def test_bridge_wrapper_passes_task_facts_through():
    # module_bridge imports the SDK (native runtime); skip where it is not built
    bridge = pytest.importorskip("aisimulate.generator.module_bridge", exc_type=ImportError)
    _vllm_dsa_kv_cache_dtype = bridge._vllm_dsa_kv_cache_dtype

    assert _vllm_dsa_kv_cache_dtype(_FakeTask("vllm", "nvidia/GLM-5.3-NVFP4"), "nvfp4") == "fp8"
    assert _vllm_dsa_kv_cache_dtype(_FakeTask("vllm", "zai-org/GLM-5.3"), "fp8_block") is None


def test_fp8_kv_dtype_renders_as_the_explicit_vllm_flag():
    out = render_backend_parameters({"kv_cache_dtype": "fp8"}, "vllm")
    assert out["kv_cache_dtype"]["kv-cache-dtype"] == "fp8"
    # and the bf16 modelling default still renders as auto
    out = render_backend_parameters({"kv_cache_dtype": "bfloat16"}, "vllm")
    assert out["kv_cache_dtype"]["kv-cache-dtype"] == "auto"
