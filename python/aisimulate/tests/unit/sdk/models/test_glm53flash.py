# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM geometry assertions derived from pinned config; see THIRD_PARTY_NOTICES.md."""

import json
from collections import Counter
from dataclasses import replace

import pytest

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, common
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from aisimulate_core.sdk.glm53flash import MODEL_REVISIONS
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.utils import get_model_config_from_model_path

pytestmark = pytest.mark.unit


def build(path, backend="vllm", tp=2, **kwargs):
    return get_model(path, ModelConfig(tp_size=tp, moe_tp_size=tp, moe_ep_size=1, **kwargs), backend)


@pytest.mark.parametrize("path", MODEL_REVISIONS)
def test_pinned_text_schedule_and_no_vision(path):
    info = get_model_config_from_model_path(path)
    d = info["extra_params"]
    assert common.ARCHITECTURE_TO_MODEL_FAMILY[info["architecture"]] == "GLM53FLASH"
    assert (info["layers"], info["hidden_size"], info["n"], info["d"], info["vocab"]) == (45, 4096, 64, 256, 154880)
    assert Counter(d.layer_types) == {"linear_attention": 34, "deepseek_sparse_attention": 11}
    assert tuple(i for i, value in enumerate(d.layer_types) if value == "deepseek_sparse_attention") == tuple(
        range(3, 45, 4)
    )
    assert d.mlp_layer_types[:3] == ("dense",) * 3
    assert d.mlp_layer_types[3:] == ("sparse",) * 42
    assert (d.linear_num_heads, d.linear_head_dim, d.hc_mult, d.hc_sinkhorn_iters) == (64, 128, 4, 20)
    assert (d.n_routed_experts, d.num_experts_per_tok, d.n_shared_experts) == (288, 8, 1)
    assert info.get("encoder_config") is None


@pytest.mark.parametrize("path", MODEL_REVISIONS)
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("tp", [2, 4])
def test_native_graph_boundary_precision_and_state(path, backend, tp):
    model = build(path, backend, tp)
    expected_format = "nvfp4" if "NVFP4" in path else "fp8"
    for phase in (model.context_ops, model.generation_ops):
        native = [json.loads(op._spec_json()) for op in phase]
        attn = [op["Glm53Attention"] for op in native if "Glm53Attention" in op]
        assert Counter(op["layer_kind"] for op in attn) == {"kda": 34, "sparse_mla": 11}
        assert all(op["checkpoint_format"] == expected_format for op in attn)
        assert all(op["projection_quant_mode"] == "bfloat16" for op in attn if op["layer_kind"] == "kda")
        sparse_quant = "fp8_block" if backend == "sglang" and expected_format == "fp8" else "bfloat16"
        assert {op["projection_quant_mode"] for op in attn if op["layer_kind"] == "sparse_mla"} == {sparse_quant}
        mhc = Counter(op["Glm53Mhc"]["role"] for op in native if "Glm53Mhc" in op)
        assert {op["Glm53Mhc"]["tp_size"] for op in native if "Glm53Mhc" in op} == {tp}
        assert {op["Glm53Mhc"]["is_context"] for op in native if "Glm53Mhc" in op} == {attn[0]["is_context"]}
        assert mhc == (
            {"expand": 1, "contract": 1, "pre": 1, "fused_post_pre": 89, "post": 1}
            if backend == "vllm"
            else {"expand": 1, "contract": 1, "pre": 90, "post": 90}
        )
        assert not any("Glm53Router" in op or "Moe" in op for op in native)
        ffns = [op["Glm53Ffn"] for op in native if "Glm53Ffn" in op]
        assert len(ffns) == 45
        assert sum(op["is_dense"] for op in ffns) == 3
        assert all(op["swiglu_limit"] == 10 and op["scoring_func"] == "sigmoid" for op in ffns)
        assert sum(any("Glm53Router" in child for child in op["children"]) for op in ffns) == 42
        assert not any("attn_norm" in op._name or "ffn_norm" in op._name for op in phase)
        primitives = [op["Glm53Primitive"] for op in native if "Glm53Primitive" in op]
        assert Counter(op["role"] for op in primitives) == {
            "embedding": 1,
            "final_norm": 1,
            "logits": 1,
            "allreduce": 91,
        }
        assert all(op["tp_size"] == tp for op in primitives)
        assert not any(key in op for op in native for key in ("Embedding", "Elementwise", "Gemm", "Nccl"))
        logits = next(op for op in primitives if op["role"] == "logits")
        assert logits["name"] == "logits" and logits["token_selection"] == "last_per_request"
        assert logits["output_dtype"] == ("float32" if backend == "sglang" else "bfloat16")
        assert logits["children"][1]["Nccl"]["operation"] == "all_gather"
        assert logits["children"][1]["Nccl"]["hidden_size"] == 154880
    # 34 FP32 recurrent states and qkv history, plus 11 replicated latent/index
    # caches. Payload does not depend on the checkpoint's weight precision.
    state = 34 * (64 // tp * 128 * 128 * 4 + 3 * (64 // tp) * 128 * 3 * 2)
    assert model.get_kvcache_bytes_per_sequence(0) == state + 11 * 2048
    assert model.get_kvcache_bytes_per_sequence(131072) == state + 11 * 71436288
    assert model.get_kvcache_bytes_per_sequence(4) - model.get_kvcache_bytes_per_sequence(3) == 11 * 644
    fixed = model.get_kvcache_bytes_per_sequence(0)
    assert model.get_kvcache_batch_capacity(2 * fixed + 5995 * 8192, 2) == 8192


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_nvfp4_shared_experts_are_bf16(backend):
    model = build("nvidia/GLM-5.3-Flash-NVFP4", backend)
    ffns = [body["Glm53Ffn"] for op in model.context_ops if "Glm53Ffn" in (body := json.loads(op._spec_json()))]
    gemms = [child["Gemm"] for op in ffns if not op["is_dense"] for child in op["children"] if "Gemm" in child]
    assert len(gemms) == 84
    assert {op["quant_mode"] for op in gemms} == {"bfloat16"}


def test_unsupported_topology_and_precision_fail_explicitly():
    path = "zai-org/GLM-5.3-Flash"
    with pytest.raises(NotImplementedError, match="vLLM and SGLang"):
        build(path, "trtllm")
    with pytest.raises(NotImplementedError, match="TP1/2/4"):
        build(path, tp=8)
    with pytest.raises(NotImplementedError, match="nextn=0"):
        build(path, nextn=1)
    with pytest.raises(NotImplementedError, match="precision partition"):
        build(path, gemm_quant_mode=common.GEMMQuantMode.bfloat16)
    with pytest.raises(ValueError, match="indices disagree"):
        info = get_model_config_from_model_path(path)
        config = dict(info["raw_config"]["text_config"])
        config["layer_types"] = list(replace(info["extra_params"], layer_types=("linear_attention",) * 45).layer_types)
        type(info["extra_params"]).from_text_config(config)


@pytest.mark.parametrize("path", MODEL_REVISIONS)
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_canonical_sol_constructor_and_saved_identity(path, backend):
    config = ForwardPassPerfModelConfig(
        model=path,
        system="gb300",
        backend=backend,
        worker_type="aggregated",
        tp=2,
        moe_tp_size=2,
        moe_ep_size=1,
        database_mode="SOL",
        estimation_mode="op_level",
        fallback_policy="deny",
    )
    model = RustForwardPassPerfModel.best_available(config)
    try:
        resolved = model.diagnostics()["provenance"]["config"]
        assert resolved["model"] == path
        assert ForwardPassPerfModelConfig(**resolved).to_dict() == resolved
        latency = model.static_phase_latency(batch_size=1, input_tokens=131072, output_tokens=2, prefill=False)
        assert latency > 0
    finally:
        model.close()


def test_formal_glm_ops_do_not_accept_generic_moe_tables():
    # The shipped GB300 generic operator dataset exists, but it cannot attest
    # GLM's whole FFN routing/clamp or hybrid attention execution boundary.
    config = ForwardPassPerfModelConfig(
        model="zai-org/GLM-5.3-Flash",
        system="gb300",
        backend="vllm",
        worker_type="aggregated",
        tp=2,
        moe_tp_size=2,
        moe_ep_size=1,
        database_mode="SILICON",
        estimation_mode="op_level",
        fallback_policy="deny",
    )
    with pytest.raises(PerfDataNotAvailableError, match="GLM-5.3-Flash measured module tables are unavailable"):
        RustForwardPassPerfModel.best_available(config)
