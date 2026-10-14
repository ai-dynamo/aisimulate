# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD layout oracles from checked-in model metadata (no performance queries)."""

import copy
import dataclasses
import json

import pytest

from aisimulate_core.sdk import common
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.models import afd
from aisimulate_core.sdk.models.helpers import _get_model_info
from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig

pytestmark = pytest.mark.unit


def config(tp, role):
    return ModelConfig(
        tp_size=tp if role == "a" else 8, moe_tp_size=tp if role == "a" else 1, moe_ep_size=1 if role == "a" else 8
    )


def specs(model, phase):
    return [json.loads(op._spec_json()) for op in getattr(model, phase + "_ops")]


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_vocabulary_sharded_embedding_keeps_its_one_time_reduction(tp):
    model = afd.build_afd_model("Qwen/Qwen3-30B-A3B", config(tp, "a"), "sglang", "a")
    for phase in ("context", "generation"):
        emitted = specs(model, phase)
        prefix = [emitted[i] for i in model.afd_plan[phase]["prefix"]]
        assert prefix[0]["Embedding"]["vocab_size"] == 151936 // tp
        if tp == 1:
            assert len(prefix) == 1
        else:
            assert len(prefix) == 2
            reduction = prefix[1]["CustomAllReduce"]
            assert reduction["name"] == f"{phase}_embedding_ar"
            assert reduction["tp_size"] == tp
            assert reduction["hidden_size"] == 2048
            assert reduction["scale_factor"] == 1
            # It belongs only to the prefix, never to a repeated layer.
            assert all(model.afd_plan[phase]["prefix"][1] not in t["ops"] for t in model.afd_plan[phase]["templates"])


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("path", ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3.2"])
def test_deepseek_shared_copy_dense_tp_and_source_order(tp, path):
    model = afd.build_afd_model(path, config(tp, "a"), "sglang", "a")
    plan = model.afd_plan
    # V3 and V3.2: 3 dense layers, then 58 routed layers. One shared expert
    # has intermediate 2048; dense intermediate is 18432.
    assert [plan["context"]["templates"][i]["routed"] for i in plan["layers"]] == [False] * 3 + [True] * 58
    assert plan["shared_inter_size"] == 2048
    for phase in ("context", "generation"):
        emitted = specs(model, phase)
        shared = [op["Gemm"] for op in emitted if "Gemm" in op and "shared_gate_up" in op["Gemm"]["name"]]
        assert shared and all(op["n"] == 4096 and op["k"] == 7168 and op["scale_num_tokens"] == 1 for op in shared)
        dense = [op["Gemm"] for op in emitted if "Gemm" in op and "afd_dense_gate_up" in op["Gemm"]["name"]]
        assert dense and all(op["n"] == 36864 // tp for op in dense)
        assert not any("Moe" in op or "Overlap" in op for op in emitted)
        reductions = [op for op in emitted if "CustomAllReduce" in op]
        assert bool(reductions) == (tp > 1)
        if tp > 1:
            for template in plan[phase]["templates"]:
                names = [getattr(model, phase + "_ops")[i]._name for i in template["ops"]]
                reduction = next(i for i, name in enumerate(names) if "afd_attn_ar" in name)
                assert all(i > reduction for i, name in enumerate(names) if "add_norm_2" in name)
        assert len(plan[phase]["prefix"]) == len(plan[phase]["suffix"]) == 1


@pytest.mark.parametrize("path", ["Qwen/Qwen3-30B-A3B", "deepseek-ai/DeepSeek-V3.2", "XiaomiMiMo/MiMo-V2-Flash"])
def test_f_pool_emits_only_routed_compute_and_shares_identical_templates(path):
    model = afd.build_afd_model(path, config(4, "f"), "sglang", "f")
    for phase in ("context", "generation"):
        emitted = specs(model, phase)
        assert emitted and all(set(op) == {"Moe"} for op in emitted)
        assert all(op["Moe"]["moe_tp_size"] == 1 and op["Moe"]["moe_ep_size"] == 8 for op in emitted)
        assert all(op["Moe"]["workload_distribution"].startswith("power_law_") for op in emitted)
        for t in model.afd_plan[phase]["templates"]:
            assert not t["rank_ops"]
            assert bool(t["ops"]) == t["routed"]
    assert len(model.context_ops) == len(model.generation_ops) == 1


def test_mimo_nonuniform_attention_matches_source_patterns():
    path = "XiaomiMiMo/MiMo-V2-Flash"
    info = _get_model_info(path)
    model = afd.build_afd_model(path, config(2, "a"), "sglang", "a")
    extra = info["extra_params"]
    assert isinstance(extra, common.HybridMoEConfig)
    for i, template in enumerate(model.afd_plan["layers"]):
        t = model.afd_plan["context"]["templates"][template]
        assert t["routed"] == bool(extra.moe_layer_freq[i])
        assert t["attention_kind"].startswith("global" if extra.attn_layer_pattern[i] else "swa")
        attention = [
            json.loads(model.context_ops[j]._spec_json())["ContextAttention"]
            for j in t["ops"]
            if "ContextAttention" in json.loads(model.context_ops[j]._spec_json())
        ]
        assert len(attention) == 1
        assert attention[0]["window_size"] == (0 if extra.attn_layer_pattern[i] else extra.sliding_window_size)


@pytest.mark.parametrize("path", ["openai/gpt-oss-120b", "google/gemma-4-26B-A4B"])
def test_grouped_attention_uses_exact_window_and_one_layer_scaling(path):
    model = afd.build_afd_model(path, config(2, "a"), "sglang", "a")
    for phase in ("context", "generation"):
        variant = "ContextAttention" if phase == "context" else "GenerationAttention"
        emitted = specs(model, phase)
        for template in model.afd_plan[phase]["templates"]:
            attention = [emitted[i][variant] for i in template["ops"] if variant in emitted[i]]
            assert len(attention) == 1
            assert bool(attention[0]["window_size"]) == template["attention_kind"].startswith("swa")
            assert attention[0]["scale_factor"] == 1
            assert all(next(iter(emitted[i].values()))["scale_factor"] == 1 for i in template["ops"])
    if "gemma" in path:
        assert model.afd_plan["shared_inter_size"] == 2112
        assert _get_model_info(path)["num_shared_experts"] == 1
        shared = [
            op["Gemm"] for op in specs(model, "context") if "Gemm" in op and "shared_mlp_gate_up" in op["Gemm"]["name"]
        ]
        assert shared and all(op["n"] == 4224 for op in shared)


def test_dsa_reuse_preserves_source_layer_positions_in_both_phases():
    model = afd.build_afd_model("zai-org/GLM-5.2", config(4, "a"), "sglang", "a")
    # Source: 78 layers, frequency 4, offset 3. The first three layers are
    # full, then layers 6/10/.../74 compute the indexer (21 full, 57 reuse).
    full_layers = {0, 1, 2, *range(6, 78, 4)}
    assert len(full_layers) == 21
    for phase, variant in [("context", "DsaContext"), ("generation", "DsaGeneration")]:
        emitted = specs(model, phase)
        for layer, ti in enumerate(model.afd_plan["layers"]):
            template = model.afd_plan[phase]["templates"][ti]
            modules = [emitted[i][variant] for i in template["ops"] if variant in emitted[i]]
            assert len(modules) == 1
            assert modules[0]["full_frac"] == float(layer in full_layers)


def test_shared_actual_width_is_not_rounded_to_routed_expert_count(monkeypatch):
    info = copy.deepcopy(_get_model_info("deepseek-ai/DeepSeek-V3"))
    # A source can declare a shared width that is not num_shared * routed width.
    info["raw_config"]["shared_expert_intermediate_size"] = 3072
    monkeypatch.setattr(afd, "_get_model_info", lambda _: info)
    model = afd.build_afd_model("deepseek-ai/DeepSeek-V3", config(4, "a"), "sglang", "a")
    assert model.afd_plan["shared_inter_size"] == 3072
    up = next(op["Gemm"] for op in specs(model, "context") if "Gemm" in op and "shared_gate_up" in op["Gemm"]["name"])
    assert up["n"] == 6144


def test_dense_projection_quant_groups_remain_separate(monkeypatch):
    info = copy.deepcopy(_get_model_info("deepseek-ai/DeepSeek-V3.2"))
    info["raw_config"]["quantization_config"]["exclude_modules"] = [
        "model.layers.1.mlp.gate_proj",
        "model.layers.1.mlp.up_proj",
    ]
    monkeypatch.setattr(afd, "_get_model_info", lambda _: info)
    model = afd.build_afd_model("deepseek-ai/DeepSeek-V3.2", config(2, "a"), "sglang", "a")
    ids = model.afd_plan["layers"][:3]
    assert ids[0] != ids[1] and ids[1] != ids[2]
    assert model.afd_plan["context"]["templates"][ids[1]]["routed"] is False


def test_routed_precision_follows_the_agg_operator_override(monkeypatch):
    import aisimulate_core.sdk.models as models

    original = models.get_model

    def source_override(*args, **kwargs):
        model = original(*args, **kwargs)
        # Model-specific checkpoint exclusions can leave a routed op at a
        # different precision than the global mode. Change only that global
        # mode after construction to exercise the consumer contract.
        model.config = dataclasses.replace(model.config, moe_quant_mode=common.MoEQuantMode.bfloat16)
        return model

    monkeypatch.setattr(models, "get_model", source_override)
    model = afd.build_afd_model("deepseek-ai/DeepSeek-V3.2", config(4, "f"), "sglang", "f")
    for phase in ("context", "generation"):
        assert all(op["Moe"]["quant_mode"] == "fp8_block" for op in specs(model, phase))


def test_invalid_patterns_and_unsupported_attention_fail_explicitly():
    info = copy.deepcopy(_get_model_info("deepseek-ai/DeepSeek-V3"))
    info["raw_config"]["moe_layer_freq"] = [1]
    with pytest.raises(ValueError, match="MoE layer pattern"):
        afd.layer_layout(info)
    with pytest.raises(NotImplementedError, match="attention layer pattern"):
        info["raw_config"]["moe_layer_freq"] = 1
        info["raw_config"]["layer_types"] = ["linear_attention"] * 61
        afd.layer_layout(info)


@pytest.mark.parametrize("role", ["a", "f"])
def test_interleaved_dense_maverick_rejects_before_model_construction(role, monkeypatch):
    import aisimulate_core.sdk.models as models

    def unexpected_construction(*args, **kwargs):
        pytest.fail("unsupported layer order must fail before constructing an aggregate model")

    monkeypatch.setattr(models, "get_model", unexpected_construction)
    with pytest.raises(NotImplementedError, match="dense layer 2 follows routed layer 1"):
        afd.build_afd_model("meta-llama/Llama-4-Maverick-17B-128E-Instruct", config(2, role), "sglang", role)


def test_model_preflight_rejects_interleaved_dense_before_database_preparation():
    from aisimulate.sweeper.afd_parallel import validate_afd_model

    with pytest.raises(NotImplementedError, match="dense layer 2 follows routed layer 1"):
        validate_afd_model("meta-llama/Llama-4-Maverick-17B-128E-Instruct")


@pytest.mark.parametrize("fallback_policy", ["deny", "allow", "regression"])
def test_canonical_pool_constructor_cannot_fallback_from_nonprefix_dense(fallback_policy):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    request = ForwardPassPerfModelConfig(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        system="h200_sxm",
        backend="sglang",
        backend_version="0.5.14",
        worker_type="aggregated",
        afd_pool_role="a",
        afd_routing_location="a",
        fallback_policy=fallback_policy,
    )
    with pytest.raises((NotImplementedError, RuntimeError, ValueError), match="dense layer 2 follows routed layer 1"):
        RustForwardPassPerfModel.best_available(request)


@pytest.mark.parametrize("pattern", [[1, 0], [0, 1, 0], [0, 1, 0, 1]])
def test_nonprefix_dense_patterns_reject_by_layout_not_model_name(pattern, monkeypatch):
    info = copy.deepcopy(_get_model_info("deepseek-ai/DeepSeek-V3"))
    info["layers"] = len(pattern)
    info["raw_config"]["num_hidden_layers"] = len(pattern)
    info["raw_config"]["moe_layer_freq"] = pattern
    info["num_moe_layers"] = sum(pattern)
    monkeypatch.setattr(afd, "_get_model_info", lambda _: info)
    with pytest.raises(NotImplementedError, match="dense prefix followed by routed MoE"):
        afd.build_afd_model("synthetic/nonprefix", config(1, "a"), "sglang", "a")


def test_pool_role_identity_round_trip_ordinary_agg_omits_it():
    base = dict(model="Qwen/Qwen3-30B-A3B", system="h200_sxm", backend="sglang", worker_type="aggregated")
    assert "afd_pool_role" not in ForwardPassPerfModelConfig(**base).to_dict()
    from aisimulate_core._native import RustForwardPassPerfModel

    for role in ("a", "f"):
        cfg = ForwardPassPerfModelConfig(**base, afd_pool_role=role, afd_routing_location="a")
        normalized = json.loads(RustForwardPassPerfModel.normalize_config(json.dumps(cfg.to_dict())))
        assert normalized["afd_pool_role"] == role
        assert ForwardPassPerfModelConfig(**normalized).to_dict()["afd_pool_role"] == role


def test_pure_tp1_and_wide_f_pool_compile_source_plans():
    # EP64 divides 128 experts; F does not execute the model's 32-head attention.
    for role, tp, ep in [("a", 1, 1), ("f", 64, 64)]:
        cfg = ModelConfig(tp_size=tp, moe_tp_size=1, moe_ep_size=ep)
        model = afd.build_afd_model("Qwen/Qwen3-30B-A3B", cfg, "sglang", role)
        plan = model.afd_plan
        assert len(plan["layers"]) == 48
        assert plan["layers"] == [0] * 48
        for phase in ("context", "generation"):
            assert len(plan[phase]["templates"]) == 1
            if role == "f":
                assert len(specs(model, phase)) == 1
                assert specs(model, phase)[0]["Moe"]["moe_ep_size"] == 64
            else:
                assert len(plan[phase]["prefix"]) == len(plan[phase]["suffix"]) == 1
