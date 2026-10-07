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
        # Rust omits an unset optional cp_size; Python to_dict keeps it as None.
        assert "cp_size" not in resolved
        saved = ForwardPassPerfModelConfig(**resolved).to_dict()
        assert saved.pop("cp_size") is None
        assert saved == resolved
        latency = model.static_phase_latency(batch_size=1, input_tokens=131072, output_tokens=2, prefill=False)
        assert latency > 0
    finally:
        model.close()


def _config(database_mode, backend="vllm", **kwargs):
    return ForwardPassPerfModelConfig(
        model="zai-org/GLM-5.3-Flash",
        system="gb300",
        backend=backend,
        worker_type="aggregated",
        tp=2,
        moe_tp_size=2,
        moe_ep_size=1,
        database_mode=database_mode,
        estimation_mode="op_level",
        fallback_policy="deny",
        **kwargs,
    )


def test_silicon_requires_exact_runtime_glm_tables():
    # Shipped GB300 generic tables predate the pinned GLM runtimes and carry no
    # GLM KDA/MoE/attention geometry: SILICON must fail closed, never use SOL
    # and never borrow earlier-version rows for the GLM-specific families.
    with pytest.raises(PerfDataNotAvailableError, match="exact runtime"):
        RustForwardPassPerfModel.best_available(_config("SILICON"))


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_hybrid_is_constructible_and_labels_analytical_fallbacks(backend):
    model = RustForwardPassPerfModel.best_available(_config("HYBRID", backend))
    try:
        ops = model.static_phase_diagnostics(batch_size=2, context_length=4096, prefix=0, prefill=True)
        sources = {op["name"]: op["source"] for op in ops}
        # No GLM attention table ships yet: the sparse layers report SOL.
        assert {sources[f"attention_{layer}"] for layer in range(3, 45, 4)} == {"sol"}
        assert all(op["latency_ms"] > 0 for op in ops if op["name"].startswith("attention_"))
    finally:
        model.close()


@pytest.mark.parametrize("path", MODEL_REVISIONS)
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("tp", [2, 4])
def test_measured_composition_uses_generic_ops_and_one_glm_attention_table(path, backend, tp):
    from aisimulate_core.sdk.models.glm53flash import KDA_KERNELS

    model = build(path, backend, tp)
    nvfp4 = "NVFP4" in path
    for phase, ops in (("context", model.context_ops), ("generation", model.generation_ops)):
        native = [json.loads(op._spec_json()) for op in ops]
        for body in native:
            kind, op = next(iter(body.items()))
            measured = op.get("measured", [])
            kinds = Counter(next(iter(child)) for child in measured)
            assert not any(k.startswith("Glm53") for k in kinds), (kind, kinds)
            if kind == "Glm53Attention" and op["layer_kind"] == "sparse_mla":
                assert measured == []  # the only GLM table op
            elif kind == "Glm53Attention":
                heads, p = 64 // tp, 64 // tp * 128
                kernels = [child["Kda"] for child in measured if "Kda" in child]
                assert [k["kernel_source"] for k in kernels] == list(KDA_KERNELS[(backend, phase)])
                assert {(k["phase"], k["d_model"], k["num_k_heads"], k["num_v_heads"]) for k in kernels} == {
                    (phase, 4096, heads, heads)
                }
                gemms = [child["Gemm"] for child in measured if "Gemm" in child]
                assert {g["quant_mode"] for g in gemms} == {"bfloat16"}
                assert len(gemms) == (4 if backend == "vllm" else 9)
                inputs = [g for g in gemms if g["k"] == 4096 and not g["name"].endswith("_o_proj")]
                assert sum(g["n"] for g in inputs) == 3 * p + heads + 256
            elif kind == "Glm53Mhc":
                # Read directly from mhc_module_perf by role in Rust.
                assert "measured" not in op
            elif kind == "Glm53Ffn":
                gemms = [child["Gemm"] for child in measured if "Gemm" in child]
                if op["is_dense"]:
                    assert kinds == {"Gemm": 2, "Elementwise": 1}
                    assert {g["quant_mode"] for g in gemms} == {"nvfp4" if nvfp4 else "fp8_block"}
                else:
                    assert kinds == {"Gemm": 3, "Elementwise": 1, "Moe": 1}
                    router = [g for g in gemms if g["n"] == 288]
                    assert len(router) == 1 and router[0]["quant_mode"] == "bfloat16"
                    moe = next(child["Moe"] for child in measured if "Moe" in child)
                    assert (moe["num_experts"], moe["topk"], moe["inter_size"], moe["moe_tp_size"]) == (
                        288,
                        8,
                        2048,
                        tp,
                    )
            elif kind == "Glm53Primitive" and op["role"] == "allreduce":
                assert kinds == {"CustomAllReduce": 1}
                assert measured[0]["CustomAllReduce"]["tp_size"] == tp
            else:
                assert measured and op["role"] in {"embedding", "final_norm", "logits"}


@pytest.mark.parametrize(
    ("backend", "prefill_ms", "decode_ms"),
    [("vllm", 334.87059419292984, 13.066380602915983), ("sglang", 350.4071011642707, 13.012015546915993)],
)
def test_sol_is_unchanged_by_measured_composition(backend, prefill_ms, decode_ms):
    # Pinned from the SOL-only graph at PR #323 f317d2bc: GLM boundaries keep
    # their analytical formulas in SOL mode, so op-level SOL (and the FPM
    # roofline built from it) is bit-identical after adding generic children.
    model = RustForwardPassPerfModel.best_available(_config("SOL", backend))
    try:
        assert model.static_phase_latency(batch_size=4, input_tokens=8192, output_tokens=2, prefill=True) == prefill_ms
        assert model.static_phase_latency(batch_size=32, input_tokens=4096, output_tokens=2, prefill=False) == decode_ms
    finally:
        model.close()


def test_kda_kernels_match_collected_kda_rows():
    # Ops W2 collection contract: merged-qkv conv on both backends; SGLang
    # decode avoids the packed kernel because GLM sets a gate lower bound.
    from aisimulate_core.sdk.models.glm53flash import KDA_KERNELS

    assert KDA_KERNELS == {
        ("vllm", "context"): ("causal_conv1d_fn", "flashkda_fwd"),
        ("vllm", "generation"): ("causal_conv1d_update", "fused_recurrent_kda"),
        ("sglang", "context"): ("causal_conv1d_fn", "chunk_kda"),
        ("sglang", "generation"): ("causal_conv1d_update", "fused_sigmoid_gating_delta_rule_update"),
    }
