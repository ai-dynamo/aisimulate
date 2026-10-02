# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# Dense composition and exclusion fixtures adapt SGLang's behavior:
# https://gitlab-master.nvidia.com/dl/sglang/sglang/-/tree/02c5a855aceb968c310e6fbc6632270e26edc84b/python/sglang/srt

"""Checkpoint layer composition must survive transfer to the native engine."""

import json
from copy import deepcopy

import pytest

from aisimulate.sdk import common, config, engine
from aisimulate.sdk.models import get_model
from aisimulate.sdk.models.deepseek_v32 import DeepSeekV32Model
from aisimulate.sdk.utils import get_model_config_from_model_path

pytestmark = pytest.mark.unit


def _native_ops(model, phase):
    # This is the same Rust-backed serialization used by build_engine_spec_json.
    return json.loads(engine._ops_json(getattr(model, f"{phase}_ops")))


def _flatten(specs):
    result = {}
    for spec in specs:
        tag, fields = next(iter(spec.items()))
        if tag == "Overlap":
            result.update(_flatten(fields["group_a"]))
            result.update(_flatten(fields["group_b"]))
        else:
            result[fields["name"]] = (tag, fields)
    return result


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_glm52_checkpoint_dense_prefix_reaches_native_ops(phase):
    model = get_model(
        "nvidia/GLM-5.2-NVFP4",
        config.ModelConfig(tp_size=4, moe_tp_size=4, moe_ep_size=1),
        backend_name="sglang",
    )
    native = _flatten(_native_ops(model, phase))

    # The packaged checkpoint declares 78 attention layers, a three-layer dense
    # prefix, dense intermediate width 12,288, and routed width 2,048. Its
    # exclusions keep layers 0..2 in BF16; routed experts remain NVFP4.
    assert native[f"{phase}_attention"][1]["scale_factor"] == 78
    assert native[f"{phase}_moe"][1]["scale_factor"] == 75
    assert native[f"{phase}_moe"][1]["quant_mode"] == "nvfp4"
    for suffix in ("shared_gate_up_gemm", "shared_act_gate", "shared_ffn2_gemm", "router_gemm"):
        assert native[f"{phase}_{suffix}"][1]["scale_factor"] == 75

    gate = native[f"{phase}_dense_gate_up_gemm"][1]
    down = native[f"{phase}_dense_down_gemm"][1]
    assert (gate["scale_factor"], gate["n"], gate["k"], gate["quant_mode"]) == (3, 6144, 6144, "bfloat16")
    assert (down["scale_factor"], down["n"], down["k"], down["quant_mode"]) == (3, 6144, 3072, "bfloat16")
    assert native[f"{phase}_dense_act_gate"][1]["scale_factor"] == 3
    reduction = native[f"{phase}_dense_ffn_ar"][1]
    assert reduction["scale_factor"] == 3
    assert reduction["tp_size"] == 4


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_dense_split_preserves_attention_and_ffn_all_reduces(phase):
    model = get_model(
        "nvidia/GLM-5.2-NVFP4",
        config.ModelConfig(tp_size=4, moe_tp_size=4, moe_ep_size=1),
        "sglang",
    )
    native = _flatten(_native_ops(model, phase))
    attention = ffn = 0
    for tag, fields in native.values():
        if tag == "MoeDispatch":
            # In ordinary SGLang TP, pre/post dispatch are the attention/FFN
            # all-reduce proxies. The DSA attention table excludes prepare_mlp.
            assert fields["backend"] == "sglang"
            assert fields["flavor"] == "CustomAllReduce"
            if fields["pre_dispatch"]:
                attention += fields["scale_factor"]
            else:
                ffn += fields["scale_factor"]
        elif tag == "CustomAllReduce":
            assert fields["tp_size"] == 4
            assert fields["hidden_size"] == 6144
            if fields["name"].endswith("dense_attn_ar"):
                attention += fields["scale_factor"]
            elif fields["name"].endswith("dense_ffn_ar"):
                ffn += fields["scale_factor"]
    # All 78 layers reduce both the attention and FFN tensor-parallel outputs,
    # regardless of whether their FFN is dense or MoE: 156 reductions in total.
    assert attention == ffn == 78


def _from_checkpoint(raw_updates=None, **config_updates):
    info = deepcopy(get_model_config_from_model_path("nvidia/GLM-5.2-NVFP4"))
    info.update(model_path="nvidia/GLM-5.2-NVFP4", model_family="DEEPSEEKV32")
    info["raw_config"].update(raw_updates or {})
    cfg = config.ModelConfig(
        tp_size=4,
        moe_tp_size=4,
        moe_ep_size=1,
        gemm_quant_mode=common.GEMMQuantMode.nvfp4,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        **config_updates,
    )
    return DeepSeekV32Model.create(info, cfg, "sglang")


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_no_dense_prefix_preserves_all_moe_graph(phase):
    native = _flatten(_native_ops(_from_checkpoint({"first_k_dense_replace": 0}), phase))
    assert not any("dense" in name for name in native)
    assert native[f"{phase}_moe"][1]["scale_factor"] == 78
    assert native[f"{phase}_shared_gate_up_gemm"][1]["scale_factor"] == 78


@pytest.mark.parametrize("source", ["quantization_config", "hf_quant_config"])
def test_dense_projection_exclusions_preserve_layer_and_projection_precision(source):
    excludes = ["model.layers.0*", "model.layers.1.mlp.down_proj", "model.layers.2.mlp.down_proj"]
    raw = {"quantization_config": {}, "hf_quant_config": {}}
    raw[source] = (
        {"ignore": excludes} if source == "quantization_config" else {"quantization": {"exclude_modules": excludes}}
    )
    native = _flatten(_native_ops(_from_checkpoint(raw), "generation"))
    # Layer 0 is wholly BF16; layers 1 and 2 keep NVFP4 gate/up and BF16 down.
    for name, count, quant in [
        ("generation_dense_0_gate_up_gemm", 1, "bfloat16"),
        ("generation_dense_0_down_gemm", 1, "bfloat16"),
        ("generation_dense_1_gate_up_gemm", 2, "nvfp4"),
        ("generation_dense_1_down_gemm", 2, "bfloat16"),
    ]:
        assert native[name][1]["scale_factor"] == count
        assert native[name][1]["quant_mode"] == quant


@pytest.mark.parametrize("exclusion", ["*.mlp*", "gate_up_proj"])
def test_dense_quantization_honors_globs_and_packed_projection_names(exclusion):
    raw = {"quantization_config": {"ignore": [exclusion]}, "hf_quant_config": {}}
    native = _flatten(_native_ops(_from_checkpoint(raw), "context"))
    assert native["context_dense_gate_up_gemm"][1]["quant_mode"] == "bfloat16"
    assert native["context_dense_down_gemm"][1]["quant_mode"] == ("bfloat16" if exclusion == "*.mlp*" else "nvfp4")


def test_dense_fused_gate_up_rejects_conflicting_shard_precision():
    raw = {"quantization_config": {"ignore": ["model.layers.0.mlp.gate_proj"]}, "hf_quant_config": {}}
    with pytest.raises(ValueError, match="gate_proj and up_proj"):
        _from_checkpoint(raw)


@pytest.mark.parametrize(
    ("exclusions", "expected"),
    [
        (["model.layers.0"], [(1, "bfloat16", "bfloat16"), (2, "nvfp4", "nvfp4")]),
        (["layers.0.mlp"], [(1, "bfloat16", "bfloat16"), (2, "nvfp4", "nvfp4")]),
        (["model.layers.0."], [(1, "bfloat16", "bfloat16"), (2, "nvfp4", "nvfp4")]),
        (["*.gate_proj", "*.up_proj"], [(3, "nvfp4", "nvfp4")]),
        (["*.gate_proj"], [(3, "nvfp4", "nvfp4")]),
        (["model.layers.0.mlp.gate_up_proj"], [(1, "bfloat16", "nvfp4"), (2, "nvfp4", "nvfp4")]),
        (
            ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"],
            [(1, "bfloat16", "nvfp4"), (2, "nvfp4", "nvfp4")],
        ),
        (["mlp.gate"], [(3, "nvfp4", "nvfp4")]),
    ],
)
@pytest.mark.parametrize("phase", ["context", "generation"])
def test_dense_packed_projection_selection_matches_runtime(exclusions, expected, phase):
    raw = {"quantization_config": {"ignore": exclusions}, "hf_quant_config": {}}
    native = _flatten(_native_ops(_from_checkpoint(raw), phase))
    groups = []
    for name, (_, fields) in native.items():
        if "dense" in name and name.endswith("gate_up_gemm"):
            down = native[name.removesuffix("gate_up_gemm") + "down_gemm"][1]
            groups.append((fields["scale_factor"], fields["quant_mode"], down["quant_mode"]))
    assert groups == expected


def test_dense_literal_shard_disagreement_precedes_packed_glob():
    raw = {
        "quantization_config": {"ignore": ["model.layers.0.mlp.gate_proj", "*.gate_up_proj"]},
        "hf_quant_config": {},
    }
    with pytest.raises(ValueError, match="gate_proj and up_proj"):
        _from_checkpoint(raw)


@pytest.mark.parametrize("count", [-1, 79, 1.5, True, False, 0.0, "", [], {}])
def test_dense_prefix_rejects_invalid_layer_counts(count):
    with pytest.raises(ValueError, match="first_k_dense_replace"):
        _from_checkpoint({"first_k_dense_replace": count})


def test_none_dense_prefix_defaults_to_zero():
    model = _from_checkpoint({"first_k_dense_replace": None})
    native = _flatten(_native_ops(model, "context"))
    assert not any("dense" in name for name in native)
    assert native["context_moe"][1]["scale_factor"] == 78


@pytest.mark.parametrize(
    ("updates", "message"),
    [({"moe_layer_freq": 2}, "moe_layer_freq"), ({"hidden_act": "gelu"}, "silu")],
)
def test_dense_prefix_rejects_unmodeled_architecture(updates, message):
    with pytest.raises(ValueError, match=message):
        _from_checkpoint(updates)


def test_dense_prefix_rejects_nondivisible_tensor_shard():
    info = deepcopy(get_model_config_from_model_path("nvidia/GLM-5.2-NVFP4"))
    info.update(model_path="nvidia/GLM-5.2-NVFP4", model_family="DEEPSEEKV32")
    info["inter_size"] = 12289
    with pytest.raises(ValueError, match="intermediate_size"):
        DeepSeekV32Model.create(info, config.ModelConfig(tp_size=4, moe_tp_size=4, moe_ep_size=1), "sglang")


def test_all_dense_checkpoint_omits_routed_and_shared_experts():
    model = _from_checkpoint({"first_k_dense_replace": 78, "quantization_config": {}, "hf_quant_config": {}})
    for phase in ("context", "generation"):
        native = _flatten(_native_ops(model, phase))
        assert not any("moe" in name or "shared" in name or "router" in name for name in native)
        assert native[f"{phase}_dense_gate_up_gemm"][1]["scale_factor"] == 78


def test_layer_count_override_truncates_the_dense_prefix():
    model = get_model(
        "nvidia/GLM-5.2-NVFP4",
        config.ModelConfig(tp_size=4, moe_tp_size=4, moe_ep_size=1, overwrite_num_layers=2),
        "sglang",
    )
    for phase in ("context", "generation"):
        native = _flatten(_native_ops(model, phase))
        assert native[f"{phase}_attention"][1]["scale_factor"] == 2
        assert native[f"{phase}_dense_gate_up_gemm"][1]["scale_factor"] == 2
        assert f"{phase}_moe" not in native


@pytest.mark.parametrize(
    ("backend", "cfg_updates"),
    [
        ("vllm", {}),
        ("trtllm", {}),
        ("sglang", {"pp_size": 2}),
        ("sglang", {"tp_size": 1, "cp_size": 2, "moe_tp_size": 1, "moe_ep_size": 2}),
        ("sglang", {"attention_dp_size": 2, "moe_ep_size": 2}),
        ("sglang", {"nextn": 2}),
        (
            "sglang",
            {
                "moe_comm_backend": {"context": "deepep_ht", "generation": "deepep_ll"},
                "num_gpus_per_node": 4,
            },
        ),
    ],
)
def test_other_regimes_preserve_existing_approximation(backend, cfg_updates):
    kwargs = {"tp_size": 4, "moe_tp_size": 4, "moe_ep_size": 1, **cfg_updates}
    model = get_model("nvidia/GLM-5.2-NVFP4", config.ModelConfig(**kwargs), backend)
    for phase in ("context", "generation"):
        native = _flatten(_native_ops(model, phase))
        assert not any("dense" in name for name in native)
        # No new claim about CP/ADP/PP/large-EP or draft-layer precision: retain
        # the original approximation. MTP still adds two all-MoE draft layers.
        expected = 80 if phase == "generation" and cfg_updates.get("nextn") else 78
        assert native[f"{phase}_moe"][1]["scale_factor"] == expected
