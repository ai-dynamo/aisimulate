# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the estimate-path data-driven context-FMHA guard.

NVBug 6401867: the single-point ``cli estimate`` / AFD path resolved fp8 FMHA
for DeepSeek-V3 context MLA (no perf data) and crashed with a
PerfDataNotAvailableError traceback. ``resolve_context_fmha_by_data`` mirrors
the resolve-time data fallback that ``task_v2`` (the sweep path) applies: the
perf DB's fmha-keyed context table decides, not a hand-written model list.
"""

import json
from types import SimpleNamespace

import pytest

import aiconfigurator.sdk.models.helpers as helpers
from aiconfigurator.sdk import common, config, inference_session, models, pareto_analysis, sweep
from aiconfigurator.sdk.models import resolve_context_fmha_by_data
from aiconfigurator_core.sdk import models as canonical_models

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("explicit", [None, common.FMHAQuantMode.fp8])
@pytest.mark.parametrize(
    "entry",
    [
        "agg",
        "agg_pareto",
        "prefill",
        "decode",
        "static_ctx",
        "static_gen",
        "run_disagg",
        "afd_prefill",
        "afd_decode",
        "afd_both",
    ],
)
def test_direct_context_construction_uses_runtime_precision(monkeypatch, entry, explicit):
    # Stop after real model construction: this checks the final native op
    # identity without depending on performance tables or a mocked builder.
    class ModelBuilt(BaseException):
        pass

    captured = []
    real_get_model = models.get_model

    def capture(model_path, model_config, backend_name):
        captured.append(real_get_model(model_path, model_config, backend_name))
        if (entry != "run_disagg" and not entry.startswith("afd_")) or len(captured) == 2:
            raise ModelBuilt
        return captured[-1]

    monkeypatch.setattr(sweep, "get_model", capture)
    monkeypatch.setattr(pareto_analysis, "get_model", capture)
    monkeypatch.setattr(models, "get_model", capture)
    db = SimpleNamespace(version="0.5.14", system_spec={"gpu": {"sm_version": 90}})
    backend = SimpleNamespace(name=SimpleNamespace(value="sglang"))
    mc = config.ModelConfig(tp_size=8, moe_tp_size=8, fmha_quant_mode=explicit)
    rt = config.RuntimeConfig(isl=1024, osl=1, ttft=1000, tpot=50)
    path = "deepseek-ai/DeepSeek-V3"
    parallel = [(8, 1, 1, 8, 1, 1)]
    with pytest.raises(ModelBuilt):
        if entry == "agg_pareto":
            pareto_analysis.agg_pareto(path, rt, db, "sglang", mc, parallel)
        elif entry == "agg":
            sweep.sweep_agg(
                model_path=path,
                model_config=mc,
                runtime_config=rt,
                database=db,
                backend_name="sglang",
                parallel_config_list=parallel,
                max_batch_size=1,
            )
        elif entry in {"prefill", "decode"}:
            sweep._get_disagg_worker_candidates(
                model_path=path,
                model_config=mc,
                runtime_config=rt,
                database=db,
                backend_name="sglang",
                parallel_config_list=parallel,
                b_list=[1],
                role=entry,
                latency_correction=1.0,
            )
        elif entry.startswith("afd_"):
            session = inference_session.AFDInferenceSession(
                path, mc, mc, db, backend, config.AFDConfig(gpus_per_node=8)
            )
            session.run_afd(rt, phase=entry.removeprefix("afd_"), free_gpu_memory_fraction=0.9)
        else:
            session = inference_session.DisaggInferenceSession(db, backend, db, backend)
            if entry == "run_disagg":
                # Shared input must not let prefill resolution alter decode.
                session.run_disagg(path, rt, mc, 1, 1, mc, 1, 1)
            else:
                session.get_worker_candidates(path, mc, parallel, [1], rt, entry)

    expected = explicit or (
        common.FMHAQuantMode.fp8 if entry in {"decode", "static_gen", "afd_decode"} else common.FMHAQuantMode.bfloat16
    )
    block = next(op for op in captured[0].context_ops if op._name == "context_mla_block")
    specs = json.loads(block._spec_json())["Fallback"]
    assert specs["primary"]["MlaModuleContext"]["fmha_quant_mode"] == expected.name
    attention = next(spec["ContextMla"] for spec in specs["fallback"] if "ContextMla" in spec)
    assert attention["fmha_quant_mode"] == expected.name
    if entry == "run_disagg":
        assert captured[1].config.fmha_quant_mode == common.FMHAQuantMode.fp8
        assert mc.fmha_quant_mode == common.FMHAQuantMode.fp8
    else:
        assert mc.fmha_quant_mode == explicit
    if entry.startswith("afd_"):
        assert captured[1].config.fmha_quant_mode == expected


def test_mla_precision_resolver_export_identity():
    assert "resolve_sglang_mla_compute" in canonical_models.__all__
    assert canonical_models.resolve_sglang_mla_compute is helpers.resolve_sglang_mla_compute
    assert models.resolve_sglang_mla_compute is canonical_models.resolve_sglang_mla_compute


# DeepSeek-V3 ships fp8_block weights → inference resolves FMHA to fp8.
_V3_FP8_RAW = {"quant_algo": "fp8_block"}
_V3_BF16_RAW = {"quant_algo": None}


@pytest.fixture
def fake_model_info(monkeypatch):
    """Patch _get_model_info so the helper resolves a chosen (arch, raw_config)."""

    def _install(architecture, raw_config):
        monkeypatch.setattr(
            helpers,
            "_get_model_info",
            lambda _model_path: {"architecture": architecture, "raw_config": raw_config},
        )

    return _install


def _mc(fmha=None, *, forward_model="op_level"):
    return config.ModelConfig(fmha_quant_mode=fmha, forward_model=forward_model)


def _db(**supported):
    """Database stub exposing only supported_quant_mode."""
    return SimpleNamespace(supported_quant_mode=supported)


# V3 on trtllm consults the "context_mla" op key.
_BF16_ONLY_DB = _db(context_mla=["bfloat16"])
_FP8_DB = _db(context_mla=["bfloat16", "fp8"])
_NO_INFO_DB = _db()


@pytest.mark.parametrize("kv", [common.KVCacheQuantMode.bfloat16, common.KVCacheQuantMode.fp8])
def test_hopper_mla_compute_is_independent_of_table_coverage(fake_model_info, kv, caplog):
    fake_model_info(
        "DeepseekV3ForCausalLM",
        {**_V3_FP8_RAW, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "torch_dtype": "bfloat16"},
    )
    mc = _mc()
    mc.kvcache_quant_mode = kv
    db = _db(context_mla=["fp8", "bfloat16"])
    db.version = "0.5.14"
    db.system_spec = {"gpu": {"sm_version": 90}}
    with caplog.at_level("INFO"):
        resolve_context_fmha_by_data(mc, "local-r1", db, "sglang", is_context_role=True)
    assert mc.fmha_quant_mode == common.FMHAQuantMode.bfloat16
    assert mc.kvcache_quant_mode == kv
    assert "FA3 execution dtype" in caplog.text
    assert "falling back" not in caplog.text


@pytest.mark.parametrize(
    "override",
    [
        {"fmha": common.FMHAQuantMode.fp8},
        {"forward_model": "fpm"},
        {"sm": 100},
        {"sm": 103},
        {"version": "0.5.13"},
        {"version": "0.5.15"},
        {"backend": "trtllm"},
        {"attention_backend": "flashinfer"},
        {"moe_comm_backend": {"context": "deepep_ht"}},
        {"architecture": "DeepseekV32ForCausalLM"},
        {"raw": {"kv_lora_rank": 256}},
        {"raw": {"torch_dtype": "float16"}},
    ],
)
def test_hopper_mla_mapping_preserves_explicit_modes_and_unaudited_paths(fake_model_info, override):
    raw = {**_V3_FP8_RAW, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "torch_dtype": "bfloat16"}
    raw.update(override.get("raw", {}))
    fake_model_info(override.get("architecture", "DeepseekV3ForCausalLM"), raw)
    mc = _mc(override.get("fmha"), forward_model=override.get("forward_model", "op_level"))
    mc.attention_backend = override.get("attention_backend")
    mc.moe_comm_backend = override.get("moe_comm_backend")
    helpers.resolve_sglang_mla_compute(
        mc,
        "local-r1",
        override.get("backend", "sglang"),
        override.get("version", "0.5.14"),
        {"gpu": {"sm_version": override.get("sm", 90)}},
    )
    assert mc.fmha_quant_mode == override.get("fmha")


def test_context_role_inferred_fp8_downgrades_to_bf16(fake_model_info, caplog):
    """Auto-inferred fp8 FMHA with a bf16-only context table falls back to bf16."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_FP8_RAW)
    mc = _mc(fmha=None)
    with caplog.at_level("WARNING"):
        resolve_context_fmha_by_data(mc, "deepseek-ai/DeepSeek-V3", _BF16_ONLY_DB, "trtllm", is_context_role=True)
    assert mc.fmha_quant_mode == common.FMHAQuantMode.bfloat16
    assert any("falling back to bfloat16" in r.message for r in caplog.records)


def test_context_role_inferred_fp8_kept_when_data_exists(fake_model_info):
    """With an fp8 slice in the context table, the inference survives (left to get_model)."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_FP8_RAW)
    mc = _mc(fmha=None)
    resolve_context_fmha_by_data(mc, "deepseek-ai/DeepSeek-V3", _FP8_DB, "trtllm", is_context_role=True)
    assert mc.fmha_quant_mode is None


def test_context_role_explicit_fp8_raises(fake_model_info):
    """Explicit fp8 FMHA with no fp8 slice raises a concise error, no traceback."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_FP8_RAW)
    mc = _mc(fmha=common.FMHAQuantMode.fp8)
    with pytest.raises(ValueError, match="has no 'context_mla' perf data"):
        resolve_context_fmha_by_data(mc, "deepseek-ai/DeepSeek-V3", _BF16_ONLY_DB, "trtllm", is_context_role=True)


def test_generation_role_keeps_fp8(fake_model_info):
    """Generation-only roles (static_gen / decode) keep fp8 — no downgrade, no error."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_FP8_RAW)
    # Explicit fp8 must NOT raise for a gen role.
    mc = _mc(fmha=common.FMHAQuantMode.fp8)
    resolve_context_fmha_by_data(mc, "deepseek-ai/DeepSeek-V3", _BF16_ONLY_DB, "trtllm", is_context_role=False)
    assert mc.fmha_quant_mode == common.FMHAQuantMode.fp8
    # Auto-inferred case: helper leaves it for get_model to resolve to fp8.
    mc_auto = _mc(fmha=None)
    resolve_context_fmha_by_data(mc_auto, "deepseek-ai/DeepSeek-V3", _BF16_ONLY_DB, "trtllm", is_context_role=False)
    assert mc_auto.fmha_quant_mode is None


def test_context_role_explicit_bf16_is_untouched(fake_model_info):
    """An explicit non-fp8 request is respected (no error, no change)."""
    fake_model_info("KimiK25ForConditionalGeneration", _V3_FP8_RAW)
    mc = _mc(fmha=common.FMHAQuantMode.bfloat16)
    resolve_context_fmha_by_data(mc, "moonshotai/Kimi-K2.5", _BF16_ONLY_DB, "trtllm", is_context_role=True)
    assert mc.fmha_quant_mode == common.FMHAQuantMode.bfloat16


def test_no_db_information_is_untouched(fake_model_info):
    """Without table info (synthetic what-if system), the inference is kept."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_FP8_RAW)
    mc = _mc(fmha=None)
    resolve_context_fmha_by_data(mc, "deepseek-ai/DeepSeek-V3", _NO_INFO_DB, "trtllm", is_context_role=True)
    assert mc.fmha_quant_mode is None
    # Explicit fp8 is also left for the query path to surface (no early raise).
    mc_explicit = _mc(fmha=common.FMHAQuantMode.fp8)
    resolve_context_fmha_by_data(mc_explicit, "deepseek-ai/DeepSeek-V3", _NO_INFO_DB, "trtllm", is_context_role=True)
    assert mc_explicit.fmha_quant_mode == common.FMHAQuantMode.fp8


def test_bf16_v3_checkpoint_needs_no_downgrade(fake_model_info):
    """A bf16 V3 checkpoint infers no fp8, so the helper is a no-op."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_BF16_RAW)
    mc = _mc(fmha=None)
    resolve_context_fmha_by_data(mc, "deepseek-ai/DeepSeek-V3-bf16", _BF16_ONLY_DB, "trtllm", is_context_role=True)
    assert mc.fmha_quant_mode is None


def test_generic_arch_consults_context_attention(fake_model_info, caplog):
    """Non-MLA families consult the generic context_attention table."""
    fake_model_info("Qwen3MoeForCausalLM", {"quant_algo": "fp8"})
    db = _db(context_attention=["bfloat16"])
    mc = _mc(fmha=None)
    with caplog.at_level("WARNING"):
        resolve_context_fmha_by_data(mc, "Qwen/Qwen3-235B", db, "sglang", is_context_role=True)
    assert mc.fmha_quant_mode == common.FMHAQuantMode.bfloat16
    assert any("context_attention" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ("fmha", "expected"),
    [
        (None, common.FMHAQuantMode.fp8),
        (common.FMHAQuantMode.fp8, common.FMHAQuantMode.fp8),
    ],
)
def test_fpm_does_not_apply_op_level_context_fmha_guard(fake_model_info, fmha, expected):
    """Whole-model FPM identity must not be rewritten from op-level table coverage."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_FP8_RAW)
    mc = _mc(fmha=fmha, forward_model="fpm")

    resolve_context_fmha_by_data(
        mc,
        "deepseek-ai/DeepSeek-V3",
        _BF16_ONLY_DB,
        "trtllm",
        is_context_role=True,
    )

    assert mc.fmha_quant_mode == expected
