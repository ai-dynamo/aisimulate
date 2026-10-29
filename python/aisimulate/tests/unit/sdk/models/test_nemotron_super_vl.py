# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4, text path only.

The packaged configuration files are unmodified copies of the upstream files at
revision da5b2650a835292c678994a1fa3fb653250fb508
(model_configs/NEMOTRON35_SUPER_VL_NVFP4_PROVENANCE.md). Vision inputs are out of
scope: AISimulate models the text path of this vision-language model.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

import aisimulate_core
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, common, config
from aisimulate_core.sdk.engine import EngineHandle, compile_engine
from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.models.helpers import _get_model_info
from aisimulate_core.sdk.utils import _load_model_config_from_model_path, get_model_config_from_model_path

pytestmark = pytest.mark.unit

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
_CONFIGS = Path(aisimulate_core.__file__).parent / "model_configs"
# nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@da5b2650a835292c678994a1fa3fb653250fb508, byte for byte.
_PACKAGED_SHA256 = {
    "config": "35189a8038d1b1f934e58cba50eb7e97b42a4da569424e35b9fd1048058bc7ea",
    "hf_quant_config": "b797654d89e25e9bd13617cb1d216ff5fad78cb6a8b99ba0cd0fbcf20050d75a",
}


@pytest.mark.parametrize("suffix", list(_PACKAGED_SHA256))
def test_bundle_is_the_pinned_upstream_bytes(suffix):
    data = (_CONFIGS / f"nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4_{suffix}.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == _PACKAGED_SHA256[suffix]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    import aisimulate.sdk.utils as sdk_utils

    def _no_network(*args, **kwargs):
        raise AssertionError("network path reached")

    monkeypatch.setattr(sdk_utils, "_download_hf_config", _no_network)
    monkeypatch.setattr(sdk_utils, "_download_hf_json", _no_network)
    caches = (get_model_config_from_model_path, _load_model_config_from_model_path, _get_model_info)
    for cached in caches:
        cached.cache_clear()
    yield
    for cached in caches:
        cached.cache_clear()


_FP8_STATIC_GEMMS = (
    "mamba_in_proj_gemm",
    "mamba_out_proj_gemm",
    "qkv_gemm",
    "proj_gemm",
    "shared_up_gemm",
    "shared_down_gemm",
    "fc1_latent_proj_gemm",
    "fc2_latent_proj_gemm",
)


def test_bundle_loads_offline_with_the_text_geometry():
    assert MODEL in common.DefaultHFModels
    info = get_model_config_from_model_path(MODEL)
    assert info["architecture"] == "NemotronH_Omni_Reasoning_V3"
    assert (info["layers"], info["hidden_size"], info["inter_size"], info["topk"], info["num_experts"]) == (
        88,
        4096,
        2688,
        22,
        512,
    )
    assert (info["n"], info["n_kv"], info["d"], info["vocab"], info["context"]) == (32, 2, 128, 131072, 1048576)
    extra = info["extra_params"]
    assert isinstance(extra, common.NemotronHConfig)
    assert Counter(extra.hybrid_override_pattern) == {"M": 40, "E": 40, "*": 8}
    assert (
        extra.mamba_num_heads,
        extra.mamba_head_dim,
        extra.ssm_state_size,
        extra.n_groups,
        extra.chunk_size,
        extra.conv_kernel,
    ) == (128, 64, 128, 8, 128, 4)
    assert (extra.moe_shared_expert_intermediate_size, extra.moe_latent_size) == (5376, 1024)
    assert info["raw_config"]["quant_algo"] == "mixed_precision"
    assert info["raw_config"]["hf_quant_config"]["quantization"]["quant_algo"] == "MIXED_PRECISION"


@pytest.mark.parametrize(
    ("backend", "fmha"),
    [
        ("trtllm", common.FMHAQuantMode.fp8),
        ("sglang", common.FMHAQuantMode.fp8),
        ("vllm", common.FMHAQuantMode.bfloat16),
    ],
)
def test_quant_defaults_follow_the_published_recipe(backend, fmha):
    model_config = config.ModelConfig(tp_size=8, pp_size=1, moe_tp_size=1, moe_ep_size=8, attention_dp_size=1)
    model = get_model(MODEL, model_config, backend_name=backend)
    assert model.model_family == "NEMOTRONH"
    assert (
        model_config.gemm_quant_mode,
        model_config.moe_quant_mode,
        model_config.kvcache_quant_mode,
        model_config.fmha_quant_mode,
    ) == (common.GEMMQuantMode.fp8_static, common.MoEQuantMode.nvfp4, common.KVCacheQuantMode.fp8, fmha)
    blocks = tuple(
        sum(op._scale_factor for op in model.context_ops if op._name == name)
        for name in ("context_mamba_norm", "context_moe_norm", "context_attn_norm")
    )
    assert blocks == (40, 40, 8)


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_text_graph_compiles_in_sol_on_blackwell(backend, monkeypatch):
    from aisimulate_core.sdk import engine as engine_module

    captured = {}
    build = engine_module.build_engine_spec_json

    def capture(model, **kwargs):
        captured["model"] = model
        return build(model, **kwargs)

    monkeypatch.setattr(engine_module, "build_engine_spec_json", capture)
    handle = EngineHandle(
        compile_engine(MODEL, "b200_sxm", backend, tp_size=4, moe_tp_size=4, moe_ep_size=1, database_mode="SOL")
    )
    assert handle.predict_prefill_latency(1, 128) > 0
    assert handle.predict_decode_latency(8, 1024) > 0
    model = captured["model"]
    kinds = {next(iter(json.loads(op._spec_json()))) for op in model.context_ops + model.generation_ops}
    # AISIM-2132 acceptance: a complete graph with no Fallback op.
    assert kinds == {
        "ContextAttention",
        "CustomAllReduce",
        "Elementwise",
        "Embedding",
        "Gemm",
        "GenerationAttention",
        "Mamba2",
        "Moe",
        "MoeDispatch",
        "P2P",
    }


def _silicon(backend, version):
    return ForwardPassPerfModelConfig(
        model=MODEL,
        system="b200_sxm",
        backend=backend,
        backend_version=version,
        worker_type="aggregated",
        tp=4,
        moe_tp_size=4,
        moe_ep_size=1,
        database_mode="SILICON",
        estimation_mode="op_level",
        fallback_policy="deny",
        enable_shared_layer=False,
    )


def _sources(model, *, prefill):
    ops = model.static_phase_diagnostics(prefill=prefill, batch_size=1 if prefill else 8, context_length=1024)
    return {op["name"]: op["source"] for op in ops}


def test_sglang_silicon_reads_the_packaged_moe_and_attention_tables():
    model = RustForwardPassPerfModel.best_available(_silicon("sglang", "0.5.14"))
    try:
        prefill, decode = _sources(model, prefill=True), _sources(model, prefill=False)
    finally:
        model.close()
    assert {n: prefill[n] for n in ("context_moe", "context_attention", "context_router_gemm")} == {
        "context_moe": "silicon",
        "context_attention": "mixed",
        "context_router_gemm": "silicon",
    }
    assert {prefill[f"context_{name}"] for name in _FP8_STATIC_GEMMS} == {"estimated"}
    assert {n: decode[n] for n in ("generation_moe", "generation_attention")} == {
        "generation_moe": "silicon",
        "generation_attention": "silicon",
    }
    # No SGLang Mamba2 table ships, and readiness accepts that today: both kernels price at SOL.
    assert {prefill["context_mamba_conv1d"], prefill["context_mamba_ssm"]} == {"sol"}
    assert {decode["generation_mamba_conv1d"], decode["generation_mamba_ssm"]} == {"sol"}


def test_vllm_silicon_has_no_nvfp4_moe_rows_at_top_k_22():
    model = RustForwardPassPerfModel.best_available(_silicon("vllm", "0.24.0"))
    try:
        with pytest.raises(PerfDataNotAvailableError, match=r'MoE data missing for MoeKey \{ quant: "nvfp4".*topk: 22'):
            model.static_phase_diagnostics(prefill=True, batch_size=1, context_length=1024)
    finally:
        model.close()
