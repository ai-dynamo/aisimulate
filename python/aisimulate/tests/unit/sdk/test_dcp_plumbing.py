# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode-context-parallel (DCP) field plumbing.

DCP (vLLM ``-dcp`` / SGLang ``--dcp-size``) is a per-phase knob orthogonal to
prefill CP (``cp_size``): it stripes the DECODE KV cache across ranks that
already belong to the attention group. This file pins the first-version
contract:

* the field exists on ``ModelConfig`` and does NOT widen the attention side;
* it travels through the shared config builder and the engine identity;
* it shards the per-rank persistent KV (prefill CP never did);
* ``get_model`` gates it on a model-level capability flag.

Deployment policy (whether one worker may combine prefill CP with DCP) is
deliberately NOT a model-building concern; it lives in the topology layer and
is covered by the compiler / task_v2 tests.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aisimulate_core.sdk import config
from aisimulate_core.sdk.config_builders import build_model_config
from aisimulate_core.sdk.models.base import BaseModel

pytestmark = pytest.mark.unit


def test_model_config_dcp_defaults_to_one_and_does_not_widen_attention():
    cfg = config.ModelConfig(
        tp_size=2,
        attention_dp_size=2,
        cp_size=2,
        dcp_size=4,
        moe_tp_size=1,
        moe_ep_size=8,
    )
    assert cfg.dcp_size == 4
    # Prefill CP folds into the attention width; decode CP reuses those ranks.
    assert cfg.attn_width == 8
    assert cfg.total_gpus_per_worker == 8
    assert cfg.resolve_moe_parallelism() == (1, 8)
    assert config.ModelConfig().dcp_size == 1


@pytest.mark.parametrize("bad", [0, -2, 1.5, True])
def test_model_config_rejects_non_positive_dcp(bad):
    with pytest.raises(ValueError, match="dcp_size"):
        config.ModelConfig(dcp_size=bad)


@pytest.mark.parametrize("bad", [0, -2, 1.5, True])
def test_model_config_rejects_non_positive_or_fractional_cp(bad):
    with pytest.raises(ValueError, match="cp_size must be a positive integer"):
        config.ModelConfig(cp_size=bad)


def test_build_model_config_carries_both_context_parallel_knobs():
    cfg = build_model_config(
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        cp_size=2,
        dcp_size=8,
    )
    assert (cfg.cp_size, cfg.dcp_size) == (2, 8)
    defaults = build_model_config(tp_size=1, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=1)
    assert (defaults.cp_size, defaults.dcp_size) == (1, 1)


def test_base_model_does_not_claim_dcp_support_by_default():
    for backend in ("vllm", "sglang", "trtllm"):
        assert BaseModel.supports_dcp(backend) is False


def test_persistent_kv_divisor_is_dcp_not_prefill_cp():
    shell = SimpleNamespace(config=SimpleNamespace(cp_size=8, dcp_size=1))
    assert BaseModel._cp_kv_memory_divisor(shell) == 1
    shell.config.dcp_size = 4
    assert BaseModel._cp_kv_memory_divisor(shell) == 4


def test_get_model_gates_dcp_on_model_capability():
    from aisimulate_core.sdk.models import get_model

    # Dense GQA DCP is only modeled on vLLM (SGLang's GQA DCP lives in its
    # Triton backend), so the sglang request must fail loud.
    model_config = config.ModelConfig(tp_size=16, moe_tp_size=16, moe_ep_size=1, dcp_size=2)
    with pytest.raises(NotImplementedError, match="dcp_size=2"):
        get_model("meta-llama/Meta-Llama-3.1-70B", model_config, "sglang")


# --------------------------------------------------------------------------
# Model-level rewrite: sharded decode attention + merge collectives
# --------------------------------------------------------------------------


def _generation_op_names(model) -> list[str]:
    return [op._name for op in model.generation_ops]


def _decode_attention_ops(model):
    """``[(top-level container name, decode attention op)]`` incl. FallbackOp interiors."""
    # Core classes: FallbackOp interiors come back as bare core instances.
    import aisimulate_core._native as core

    kinds = (core.GenerationAttention, core.GenerationMLA, core.WideEPGenerationMLA, core.GenerationDSAModule)

    def walk(container_name, op):
        if isinstance(op, core.FallbackOp):
            yield from walk(container_name, op._primary)
            for inner in op._fallback:
                yield from walk(container_name, inner)
        elif isinstance(op, kinds) or (isinstance(op, core.MLAModule) and not op._is_context):
            yield container_name, op

    return [pair for top in model.generation_ops if not top._name.startswith("draft_") for pair in walk(top._name, top)]


def _context_attention_ops(model):
    import aisimulate_core._native as core

    kinds = (core.ContextAttention, core.ContextMLA, core.ContextDSAModule)

    def walk(op):
        if isinstance(op, core.FallbackOp):
            yield from walk(op._primary)
            for inner in op._fallback:
                yield from walk(inner)
        elif isinstance(op, kinds) or (isinstance(op, core.MLAModule) and op._is_context):
            yield op

    return [op for top in model.context_ops if not top._name.startswith("draft_") for op in walk(top)]


@pytest.mark.parametrize(
    ("backend", "merge_suffix"),
    [("sglang", "_dcp_out_all_to_all"), ("vllm", "_dcp_out_reduce_scatter")],
)
def test_deepseek_dcp_rewrites_decode_attention_and_adds_merge_collectives(backend, merge_suffix):
    from aisimulate_core.sdk.models import get_model

    model_config = config.ModelConfig(tp_size=8, moe_tp_size=8, moe_ep_size=1, dcp_size=8)
    model = get_model("deepseek-ai/DeepSeek-V3", model_config, backend)

    attention = _decode_attention_ops(model)
    assert attention, "DeepSeek decode graph must expose a decode attention op"
    # Every decode attention op -- the fused MLAModule primary AND its granular
    # GenerationMLA / GenerationAttention fallback -- is striped.
    assert all(op._dcp_size == 8 for _, op in attention)
    names = _generation_op_names(model)
    for container, _ in {container: None for container, _ in attention}.items():
        index = names.index(container)
        # The merge collectives ride directly behind the block they serve.
        assert names[index + 1] == f"{container}_dcp_q_all_gather"
        assert names[index + 2] == f"{container}{merge_suffix}"
    # The prefill graph gains no collectives of its own ...
    assert not any("_dcp_" in op._name for op in model.context_ops)
    # ... but its attention ops are marked so cached-context prefill pays the
    # stripe gather (aggregated serving on the same engine).
    context_attention = _context_attention_ops(model)
    assert context_attention
    assert all(op._dcp_size == 8 for op in context_attention)


def test_dcp_comm_override_selects_the_merge_collective():
    from aisimulate_core.sdk.models import get_model

    model_config = config.ModelConfig(tp_size=8, moe_tp_size=8, moe_ep_size=1, dcp_size=4, dcp_comm="a2a")
    model = get_model("deepseek-ai/DeepSeek-V3", model_config, "vllm")
    names = _generation_op_names(model)
    assert any(name.endswith("_dcp_out_all_to_all") for name in names)
    assert not any(name.endswith("_dcp_out_reduce_scatter") for name in names)


def test_dsa_dcp_rewrites_the_sparse_decode_module():
    from aisimulate_core.sdk.models import get_model

    model_config = config.ModelConfig(tp_size=8, moe_tp_size=8, moe_ep_size=1, dcp_size=4)
    model = get_model("deepseek-ai/DeepSeek-V3.2", model_config, "sglang")
    attention = _decode_attention_ops(model)
    assert attention
    assert all(op._dcp_size == 4 for _, op in attention)
    assert any(name.endswith("_dcp_q_all_gather") for name in _generation_op_names(model))


def test_gqa_dcp_is_bounded_by_kv_head_replication():
    from aisimulate_core.sdk.models import get_model

    # Llama-3.1-70B: 8 kv heads. tp=16 replicates each kv head twice -> dcp<=2.
    ok = get_model(
        "meta-llama/Meta-Llama-3.1-70B",
        config.ModelConfig(tp_size=16, moe_tp_size=16, moe_ep_size=1, dcp_size=2),
        "vllm",
    )
    assert all(op._dcp_size == 2 for _, op in _decode_attention_ops(ok))
    with pytest.raises(ValueError, match="KV-head replication"):
        get_model(
            "meta-llama/Meta-Llama-3.1-70B",
            config.ModelConfig(tp_size=16, moe_tp_size=16, moe_ep_size=1, dcp_size=4),
            "vllm",
        )
    with pytest.raises(ValueError, match="must divide the attention TP size"):
        get_model(
            "meta-llama/Meta-Llama-3.1-70B",
            config.ModelConfig(tp_size=16, moe_tp_size=16, moe_ep_size=1, dcp_size=3),
            "vllm",
        )


def test_fpm_forward_model_refuses_dcp_until_tables_carry_it():
    from aisimulate_core.sdk.models import get_model

    model_config = config.ModelConfig(tp_size=8, moe_tp_size=8, moe_ep_size=1, dcp_size=8, forward_model="fpm")
    with pytest.raises(NotImplementedError, match="forward_model='fpm' has no decode-context-parallel cells"):
        get_model("deepseek-ai/DeepSeek-V3", model_config, "vllm")


def test_get_model_records_the_backend_for_families_that_do_not():
    # The DCP merge collective defaults per backend (a2a on sglang); dense
    # families do not store backend_name themselves, so get_model must.
    from aisimulate_core.sdk.models import get_model

    for backend in ("vllm", "sglang"):
        model = get_model("meta-llama/Meta-Llama-3.1-70B", config.ModelConfig(tp_size=8), backend)
        assert model._backend_name == backend
        assert model._dcp_comm_style() == ("a2a" if backend == "sglang" else "ag_rs")


def test_dcp_one_leaves_the_decode_graph_untouched():
    from aisimulate_core.sdk.models import get_model

    model_config = config.ModelConfig(tp_size=8, moe_tp_size=8, moe_ep_size=1)
    model = get_model("deepseek-ai/DeepSeek-V3", model_config, "sglang")
    assert all(op._dcp_size == 1 for _, op in _decode_attention_ops(model))
    assert not any("_dcp_" in name for name in _generation_op_names(model))


def test_engine_identity_includes_dcp_size():
    from aisimulate_core.sdk.rust_engine_step import _engine_config_json

    def make(dcp):
        cfg = SimpleNamespace(
            tp_size=4,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=4,
            attention_dp_size=1,
            cp_size=None,
            dcp_size=dcp,
            gemm_quant_mode=SimpleNamespace(name="fp8", value=None),
            moe_quant_mode=SimpleNamespace(name="nvfp4", value=None),
            fmha_quant_mode=SimpleNamespace(name="bfloat16", value=None),
            comm_quant_mode=SimpleNamespace(name="half", value=None),
            kvcache_quant_mode=SimpleNamespace(name="fp8", value=None),
        )
        model = SimpleNamespace(
            config=cfg,
            model_path="org/model-a",
            architecture="X",
            forward_model="op_level",
            _nextn=None,
            _nextn_accepted=None,
        )
        database = SimpleNamespace(system="b200_sxm", backend="vllm", version="0.25.1", systems_root="/tmp/x")
        return _engine_config_json(model, database)

    assert make(1) != make(8)
    assert json.loads(make(8))["dcp_size"] == 8
    assert json.loads(make(None))["dcp_size"] is None
