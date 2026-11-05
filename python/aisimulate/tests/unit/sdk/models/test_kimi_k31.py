# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K3.1's sparse attention must retain the K3 backbone without borrowing dense MLA."""

import json
from copy import deepcopy

import pytest

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, common, config, engine
from aisimulate_core.sdk.config_builders import resolve_dspark_nextn, resolve_nextn_auto
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.models.helpers import attention_op_keys, resolve_kimi_k3_moe_arch_mode
from aisimulate_core.sdk.speculation.base import SpeculationConfig
from aisimulate_core.sdk.utils import _parse_hf_config_json, get_model_config_from_model_path

pytestmark = pytest.mark.unit

ARCHITECTURE = "KimiK31ForConditionalGeneration"


@pytest.fixture
def k31_config():
    # Synthetic extension of the already bundled K3 fixture. This is not a
    # copied K3.1 checkpoint config; the indexer fields define the tested contract.
    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K3")["raw_config"])
    raw["architectures"] = [ARCHITECTURE]
    raw["model_type"] = "kimi_k31"
    raw["text_config"]["dsa_config"] = {
        "dsa_layers": list(raw["text_config"]["linear_attn_config"]["full_attn_layers"]),
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_num_heads": 32,
        "index_rope_dim": 0,
        "index_k_norm_type": "layernorm",
        "index_topk_freq": 1,
        "index_skip_topk_offset": 1,
    }
    return raw


def _write_checkpoint(tmp_path, raw):
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return str(tmp_path)


def _model_config(**overrides):
    return config.ModelConfig(tp_size=8, moe_tp_size=1, moe_ep_size=8, **overrides)


def _specs(model, phase):
    return json.loads(engine._ops_json(getattr(model, f"{phase}_ops")))


def test_k31_parser_preserves_backbone_and_marks_sparse_layers(k31_config):
    parsed = _parse_hf_config_json(k31_config)
    baseline = get_model_config_from_model_path("moonshotai/Kimi-K3")
    extra = parsed["extra_params"]
    assert parsed["architecture"] == ARCHITECTURE
    assert extra.layer_types.count("linear_attention") == 69
    assert extra.layer_types.count("sparse_attention") == 24
    assert "full_attention" not in extra.layer_types
    assert extra.vision_config == baseline["extra_params"].vision_config
    assert extra.dsa_config.index_num_heads == 32
    assert extra.dsa_config.index_topk == 2048
    assert baseline["extra_params"].dsa_config is None
    assert baseline["extra_params"].layer_types.count("full_attention") == 24
    assert common.ARCHITECTURE_TO_MODEL_FAMILY[ARCHITECTURE] == "KIMIK3"


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_k31_graph_replaces_only_attention_without_double_counting(tmp_path, k31_config, phase, backend):
    model = get_model(_write_checkpoint(tmp_path, k31_config), _model_config(), backend)
    baseline = get_model("moonshotai/Kimi-K3", _model_config(), backend)
    specs = _specs(model, phase)
    dsa = [spec[f"Dsa{phase.capitalize()}"] for spec in specs if f"Dsa{phase.capitalize()}" in spec]
    assert len(dsa) == 1
    assert dsa[0]["architecture"] == ARCHITECTURE
    assert dsa[0]["scale_factor"] == 24
    assert dsa[0]["num_heads"] == 12
    assert dsa[0]["index_topk"] == 2048
    assert dsa[0]["full_frac"] == 1.0
    assert not any("mla" in op._name or "draft" in op._name for op in getattr(model, f"{phase}_ops"))
    assert not any("dsa_gate" in op._name for op in getattr(model, f"{phase}_ops"))
    # The native DSA module includes the output gate. Hand-counted at TP8:
    # q_a=11,010,048; q_b=3,538,944; kv_a=4,128,768; kv_b=1,572,864;
    # o=11,010,048; index q=6,291,456, score=229,376, k=917,504;
    # output gate=11,010,048. 49,709,056 BF16 params/layer * 24 layers.
    module = next(op for op in getattr(model, f"{phase}_ops") if op._name == f"{phase}_dsa_module")
    assert module.get_weights() == 2_386_034_688
    unchanged = lambda op: not any(part in op._name for part in ("_mla_", "_dsa_", "_attention", "_bmm_"))
    assert [op._spec_json() for op in getattr(model, f"{phase}_ops") if unchanged(op)] == [
        op._spec_json() for op in getattr(baseline, f"{phase}_ops") if unchanged(op)
    ]
    assert [op._spec_json() for op in model.encoder_ops] == [op._spec_json() for op in baseline.encoder_ops]


@pytest.mark.parametrize(
    "kv_quant,backend,extra_bytes_per_token",
    [
        (common.KVCacheQuantMode.bfloat16, None, 3168),
        (common.KVCacheQuantMode.fp8, "trtllm_mla", 3168),
        (common.KVCacheQuantMode.fp8, "flashmla", 5088),
    ],
)
def test_k31_cache_retains_history_and_adds_indexer(tmp_path, k31_config, kv_quant, backend, extra_bytes_per_token):
    model = get_model(
        _write_checkpoint(tmp_path, k31_config),
        _model_config(kvcache_quant_mode=kv_quant, attention_backend=backend),
        "sglang",
    )
    baseline = get_model("moonshotai/Kimi-K3", _model_config(kvcache_quant_mode=kv_quant), "sglang")
    assert model.get_kvcache_elements_per_token() == 24 * 576
    assert model.get_kvcache_bytes_per_sequence(0) == baseline.get_kvcache_bytes_per_sequence(0)
    # 24 indexer caches * (128 FP8 key bytes + one FP32 scale). FlashMLA
    # additionally keeps the 64-wide tail BF16 and four FP32 latent scales:
    # 24 * (64 + 16) = 1920 bytes more than K3's raw-FP8 latent entry.
    for length in (2048, 2049, 16384):
        assert (
            model.get_kvcache_bytes_per_sequence(length) - baseline.get_kvcache_bytes_per_sequence(length)
            == extra_bytes_per_token * length
        )
        budget = model.get_kvcache_bytes_per_sequence(length)
        assert model.get_kvcache_rank_bytes_per_sequence(length) == budget
        assert model.get_kvcache_max_tokens(budget) == length
        assert model.get_kvcache_rank_batch_capacity(budget, 1) == length


@pytest.mark.parametrize("backend", ["flashmla", "trtllm_mla"])
def test_k31_explicit_fp8_layout_reaches_both_native_ops_and_deepcopy(tmp_path, k31_config, backend):
    model = get_model(
        _write_checkpoint(tmp_path, k31_config),
        _model_config(kvcache_quant_mode=common.KVCacheQuantMode.fp8, attention_backend=backend),
        "sglang",
    )
    for phase in ("context", "generation"):
        module = next(op for op in getattr(model, f"{phase}_ops") if op._name == f"{phase}_dsa_module")
        assert json.loads(module._spec_json())[f"Dsa{phase.capitalize()}"]["attention_backend"] == backend
        assert deepcopy(module)._spec_json() == module._spec_json()


@pytest.mark.parametrize("backend", [None, "default", "fa3", "flashinfer"])
def test_k31_fp8_requires_explicit_physical_cache_layout(tmp_path, k31_config, backend):
    with pytest.raises(ValueError, match="Kimi K3.1.*attention_backend"):
        get_model(
            _write_checkpoint(tmp_path, k31_config),
            _model_config(kvcache_quant_mode=common.KVCacheQuantMode.fp8, attention_backend=backend),
            "sglang",
        )


def test_k31_rejects_unqualified_int8_kv(tmp_path, k31_config):
    with pytest.raises(ValueError, match="supports only bfloat16 or fp8"):
        get_model(
            _write_checkpoint(tmp_path, k31_config),
            _model_config(kvcache_quant_mode=common.KVCacheQuantMode.int8, attention_backend="flashmla"),
            "sglang",
        )


@pytest.mark.parametrize("value", [None, [], {}, False])
def test_k31_requires_dsa_definition(k31_config, value):
    k31_config["text_config"]["dsa_config"] = value
    with pytest.raises(ValueError, match="Kimi K3.1"):
        _parse_hf_config_json(k31_config)


@pytest.mark.parametrize("value", [[4, 4], [0], [94], [True], [4.0], [], "4", [1, 4], [4]])
def test_k31_rejects_invalid_sparse_layer_partition(k31_config, value):
    k31_config["text_config"]["dsa_config"]["dsa_layers"] = value
    with pytest.raises(ValueError, match="Kimi K3.1"):
        _parse_hf_config_json(k31_config)


@pytest.mark.parametrize(
    "name,value",
    [
        ("index_topk", 1024),
        ("index_head_dim", 64),
        ("index_num_heads", 64),
        ("index_rope_dim", 64),
        ("index_rope_dim", False),
        ("index_k_norm_type", "rmsnorm"),
        ("index_topk_freq", 2),
        ("index_skip_topk_offset", 0),
    ],
)
def test_k31_rejects_unmodeled_indexer_geometry(k31_config, name, value):
    k31_config["text_config"]["dsa_config"][name] = value
    with pytest.raises(ValueError, match=name):
        _parse_hf_config_json(k31_config)


@pytest.mark.parametrize("name,value", [("hidden_size", 4096), ("mla_use_nope", False), ("mla_use_output_gate", False)])
def test_k31_rejects_unmodeled_attention_geometry(k31_config, name, value):
    k31_config["text_config"][name] = value
    with pytest.raises(ValueError, match=name):
        _parse_hf_config_json(k31_config)


def test_k3_cannot_silently_discard_a_sparse_definition(k31_config):
    k31_config["architectures"] = ["KimiK3ForConditionalGeneration"]
    with pytest.raises(ValueError, match="requires architecture KimiK31"):
        _parse_hf_config_json(k31_config)


@pytest.mark.parametrize(
    "overrides",
    [
        {"nextn": 7},
        {"speculation": SpeculationConfig(kind="mtp", params={"depth": 1})},
        {"speculation": SpeculationConfig(kind="dspark", params={"block_size": 7})},
    ],
)
def test_k31_rejects_unqualified_speculative_decoding(tmp_path, k31_config, overrides):
    with pytest.raises(ValueError, match="non-speculative decoding only"):
        get_model(_write_checkpoint(tmp_path, k31_config), _model_config(**overrides), "sglang")


def test_k31_does_not_inherit_k3_dspark_or_decode_cp(tmp_path, k31_config):
    path = _write_checkpoint(tmp_path, k31_config)
    assert resolve_nextn_auto(path) == 0
    assert ARCHITECTURE not in common.DSPARK_ARCHITECTURES
    assert resolve_dspark_nextn(path) is None
    with pytest.raises(ValueError, match="decode context parallelism"):
        get_model(path, _model_config(dcp_size=2), "sglang")


def test_k31_does_not_borrow_k3_physical_state_layout(tmp_path, k31_config):
    from aisimulate_core.sdk.memory.state import estimate_state_cache

    with pytest.raises(ValueError, match="Kimi K3.1 automatic vLLM physical-state sizing is not qualified"):
        estimate_state_cache(_write_checkpoint(tmp_path, k31_config), tp_size=8)


@pytest.mark.parametrize("backend", ["sglang", "vllm", "trtllm"])
def test_k31_attention_support_uses_dsa_keys(backend):
    assert attention_op_keys("KIMIK3", backend, architecture=ARCHITECTURE) == (
        "dsa_context_module",
        "dsa_generation_module",
    )


def test_k31_blackwell_moe_quantization_matches_k3():
    assert resolve_kimi_k3_moe_arch_mode("moonshotai/Kimi-K3.1-Pro", "b200_sxm", "sglang") == (
        common.MoEQuantMode.w4a8_mxfp4_mxfp8
    )
    assert resolve_kimi_k3_moe_arch_mode("moonshotai/Kimi-K3.1-Pro", "h200_sxm", "sglang") is None


def test_k31_canonical_sol_constructor_and_saved_identity(tmp_path, k31_config):
    path = _write_checkpoint(tmp_path, k31_config)
    options = ForwardPassPerfModelConfig(
        model=path,
        system="gb300",
        backend="sglang",
        worker_type="aggregated",
        tp=8,
        moe_tp_size=1,
        moe_ep_size=8,
        database_mode="SOL",
        estimation_mode="op_level",
        fallback_policy="deny",
    )
    predictor = RustForwardPassPerfModel.best_available(options)
    try:
        resolved = predictor.diagnostics()["provenance"]["config"]
        assert resolved["model"] == path
        saved = ForwardPassPerfModelConfig(**resolved).to_dict()
        saved.pop("cp_size", None)
        assert saved == resolved
        for prefill in (True, False):
            assert (
                predictor.static_phase_latency(batch_size=1, input_tokens=16384, output_tokens=2, prefill=prefill) > 0
            )
    finally:
        predictor.close()
