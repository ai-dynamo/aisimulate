# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the current NVIDIA NVFP4 support-matrix checkpoints."""

import contextlib
import dataclasses
import hashlib
from pathlib import Path

import pytest

import aisimulate_core
from aisimulate.sdk import common, config
from aisimulate.sdk.models import get_model
from aisimulate.sdk.operations import OverlapOp
from aisimulate.sdk.task_v2 import Task
from aisimulate.sdk.utils import get_model_config_from_model_path
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

pytestmark = pytest.mark.unit


NVFP4_VARIANTS = (
    "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "nvidia/Qwen3.6-27B-NVFP4",
    "nvidia/Qwen3.5-397B-A17B-NVFP4",
    "nvidia/Qwen3.5-397B-A17B-NVFP4-V2",
    "nvidia/Qwen3.5-122B-A10B-NVFP4",
    "nvidia/Gemma-4-31B-IT-NVFP4",
    "nvidia/Gemma-4-26B-A4B-NVFP4",
    "nvidia/Kimi-K2.6-NVFP4",
    "nvidia/Kimi-K2.7-Code-NVFP4",
    "nvidia/DeepSeek-V4-Flash-NVFP4",
    "nvidia/DeepSeek-V4-Pro-NVFP4",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
    "nvidia/MiniMax-M3-NVFP4",
)


def _model_config():
    return config.ModelConfig(
        tp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
    )


@pytest.mark.parametrize("hf_id", NVFP4_VARIANTS)
def test_nvfp4_variant_is_registered_for_default_matrix(hf_id):
    assert hf_id in common.DefaultHFModels
    assert hf_id in common.SupportMatrixHFModels


@pytest.mark.parametrize("hf_id", NVFP4_VARIANTS)
def test_nvfp4_variant_loads_offline_with_quant_metadata(hf_id, monkeypatch):
    import aisimulate.sdk.utils as sdk_utils

    def _no_network(*args, **kwargs):
        raise AssertionError("network path reached")

    monkeypatch.setattr(sdk_utils, "_download_hf_config", _no_network, raising=False)
    sdk_utils._load_model_config_from_model_path.cache_clear()
    loaded = get_model_config_from_model_path(hf_id)

    assert loaded["architecture"] in common.ARCHITECTURE_TO_MODEL_FAMILY
    raw_config = loaded["raw_config"]
    assert raw_config.get("hf_quant_config") or raw_config.get("quantization_config")
    sdk_utils._load_model_config_from_model_path.cache_clear()


def test_lightning_nvfp4_loads_bundled_quant_config_offline(monkeypatch):
    import aisimulate.sdk.utils as sdk_utils
    from aisimulate_core.sdk.models.helpers import _get_model_info

    hf_id = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"

    def _no_network(*args, **kwargs):
        raise AssertionError("network path reached")

    monkeypatch.setattr(sdk_utils, "_download_hf_config", _no_network)
    monkeypatch.setattr(sdk_utils, "_download_hf_json", _no_network)
    sdk_utils.get_model_config_from_model_path.cache_clear()
    sdk_utils._load_model_config_from_model_path.cache_clear()
    _get_model_info.cache_clear()
    try:
        model_config = _model_config()
        model = get_model(hf_id, model_config, backend_name="vllm")

        assert model.model_family == "NEMOTRONH"
        assert model_config.gemm_quant_mode == common.GEMMQuantMode.fp8_static
        assert model_config.moe_quant_mode == common.MoEQuantMode.nvfp4
        assert model_config.kvcache_quant_mode == common.KVCacheQuantMode.fp8
    finally:
        sdk_utils.get_model_config_from_model_path.cache_clear()
        sdk_utils._load_model_config_from_model_path.cache_clear()
        _get_model_info.cache_clear()


@pytest.mark.parametrize(
    "hf_id,gemm_mode,moe_mode,kv_mode,fmha_mode",
    [
        (
            "nvidia/Qwen3.6-35B-A3B-NVFP4",
            common.GEMMQuantMode.fp8_static,
            common.MoEQuantMode.w4a16_nvfp4,
            common.KVCacheQuantMode.fp8,
            common.FMHAQuantMode.fp8,
        ),
        (
            "nvidia/Qwen3.6-27B-NVFP4",
            common.GEMMQuantMode.fp8_static,
            common.MoEQuantMode.bfloat16,
            common.KVCacheQuantMode.fp8,
            common.FMHAQuantMode.fp8,
        ),
        *[
            (
                hf_id,
                common.GEMMQuantMode.nvfp4,
                common.MoEQuantMode.nvfp4,
                common.KVCacheQuantMode.fp8,
                common.FMHAQuantMode.fp8,
            )
            for hf_id in (
                "nvidia/Qwen3.5-397B-A17B-NVFP4",
                "nvidia/Qwen3.5-122B-A10B-NVFP4",
                "nvidia/Gemma-4-31B-IT-NVFP4",
                "nvidia/Gemma-4-26B-A4B-NVFP4",
                "nvidia/Kimi-K2.6-NVFP4",
                "nvidia/Kimi-K2.7-Code-NVFP4",
                "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            )
        ],
        (
            "nvidia/Qwen3.5-397B-A17B-NVFP4-V2",
            common.GEMMQuantMode.fp8_static,
            common.MoEQuantMode.nvfp4,
            common.KVCacheQuantMode.fp8,
            common.FMHAQuantMode.fp8,
        ),
        *[
            (
                hf_id,
                common.GEMMQuantMode.fp8_block,
                common.MoEQuantMode.nvfp4,
                common.KVCacheQuantMode.fp8,
                common.FMHAQuantMode.bfloat16,
            )
            for hf_id in (
                "nvidia/DeepSeek-V4-Flash-NVFP4",
                "nvidia/DeepSeek-V4-Pro-NVFP4",
            )
        ],
        (
            "nvidia/MiniMax-M3-NVFP4",
            # MXFP8 prices as fp8_block (owner decision 2026-08-09; DSV4
            # precedent, and the shipped M3 MSA tables carry the fp8_block
            # gemm tier this lane resolves to — see helpers.py).
            common.GEMMQuantMode.fp8_block,
            common.MoEQuantMode.nvfp4,
            common.KVCacheQuantMode.bfloat16,
            common.FMHAQuantMode.bfloat16,
        ),
    ],
)
def test_nvfp4_variant_infers_checkpoint_quant_modes(hf_id, gemm_mode, moe_mode, kv_mode, fmha_mode):
    model_config = _model_config()

    get_model(hf_id, model_config, backend_name="trtllm")

    assert model_config.gemm_quant_mode == gemm_mode
    assert model_config.moe_quant_mode == moe_mode
    assert model_config.kvcache_quant_mode == kv_mode
    assert model_config.fmha_quant_mode == fmha_mode


@pytest.mark.parametrize(
    "hf_id,ffn_names",
    [
        (
            "nvidia/Qwen3.6-35B-A3B-NVFP4",
            ("context_gdn_shared_gate_up_gemm", "generation_full_shared_down_gemm"),
        ),
        (
            "nvidia/Qwen3.6-27B-NVFP4",
            ("context_gdn_gate_ffn1_gemm", "generation_full_ffn2_gemm"),
        ),
    ],
)
def test_qwen36_uses_fp8_projections_and_w4a16_nvfp4_ffn(hf_id, ffn_names):
    model = get_model(hf_id, _model_config(), backend_name="trtllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    for name in ("context_gdn_in_proj_gemm", "context_qkv_gemm", "generation_proj_gemm"):
        assert by_name[name]._quant_mode == common.GEMMQuantMode.fp8_static
    for name in ffn_names:
        assert by_name[name]._quant_mode == common.GEMMQuantMode.w4a16_nvfp4
    if hf_id == "nvidia/Qwen3.6-35B-A3B-NVFP4":
        assert by_name["context_gdn_moe"]._quant_mode == common.MoEQuantMode.w4a16_nvfp4


def test_qwen36_task_preserves_inferred_provenance_for_mixed_precision_split():
    task = Task(
        model_path="nvidia/Qwen3.6-27B-NVFP4",
        system_name="gb300",
        backend_name="sglang",
        backend_version="next",
        total_gpus=32,
    )
    model_config = task.build_model_config(role="agg")
    model_config = dataclasses.replace(model_config, tp_size=1)
    model = get_model(task.model_path, model_config, task.backend_name)
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    assert model_config._gemm_quant_mode_is_explicit is False
    assert by_name["context_gdn_in_proj_gemm"]._quant_mode == common.GEMMQuantMode.fp8_static
    assert by_name["context_gdn_gate_ffn1_gemm"]._quant_mode == common.GEMMQuantMode.w4a16_nvfp4


def test_model_config_provenance_does_not_shift_positional_quant_modes():
    model_config = config.ModelConfig(
        2,
        1,
        common.GEMMQuantMode.fp8,
        common.MoEQuantMode.nvfp4,
        common.KVCacheQuantMode.fp8,
        common.FMHAQuantMode.fp8,
    )

    assert model_config.gemm_quant_mode == common.GEMMQuantMode.fp8
    assert model_config.moe_quant_mode == common.MoEQuantMode.nvfp4
    assert model_config.kvcache_quant_mode == common.KVCacheQuantMode.fp8
    assert model_config.fmha_quant_mode == common.FMHAQuantMode.fp8
    assert model_config._gemm_quant_mode_is_explicit is None


def test_qwen36_task_preserves_explicit_gemm_override():
    task = Task(
        model_path="nvidia/Qwen3.6-27B-NVFP4",
        system_name="gb300",
        backend_name="sglang",
        backend_version="next",
        total_gpus=32,
        gemm_quant_mode=common.GEMMQuantMode.fp8_static,
    )
    model_config = task.build_model_config(role="agg")
    model_config = dataclasses.replace(model_config, tp_size=1)
    model = get_model(task.model_path, model_config, task.backend_name)
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    assert model_config._gemm_quant_mode_is_explicit is True
    assert by_name["context_gdn_gate_ffn1_gemm"]._quant_mode == common.GEMMQuantMode.fp8_static


@pytest.mark.parametrize(
    "hf_id,ffn_name",
    [
        ("nvidia/Qwen3.6-27B-NVFP4", "generation_full_ffn2_gemm"),
        ("nvidia/Qwen3.6-35B-A3B-NVFP4", "generation_full_shared_down_gemm"),
    ],
)
@pytest.mark.parametrize("gemm_mode", [common.GEMMQuantMode.bfloat16, common.GEMMQuantMode.fp8_static])
def test_qwen36_preserves_explicit_global_gemm_override(hf_id, ffn_name, gemm_mode):
    model_config = _model_config()
    model_config.gemm_quant_mode = gemm_mode

    model = get_model(hf_id, model_config, backend_name="trtllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    for name in ("context_gdn_in_proj_gemm", "context_qkv_gemm", ffn_name):
        assert by_name[name]._quant_mode == gemm_mode


@pytest.mark.parametrize(
    ("hf_id", "backend", "moe_mode"),
    [
        # vLLM has no weight-only NVFP4 MoE lane: the compressed-tensors
        # dispatch quantizes activations at run time and serves the w4a4
        # FP4-tensor-core kernels (collected kernel_source
        # vllm_compressedtensorsw4a4nvfp4moe_*), so the W4A16_NVFP4 storage
        # label resolves to the nvfp4 execution mode there (AIC-1748).
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "vllm", common.MoEQuantMode.nvfp4),
        # Qwen3.6 stays on the weight-only profile even on vLLM: it is served
        # through HYBRID's calibrated XPROFILE relation by design (the remap
        # is scoped to architectures with PROVEN w4a4 vLLM dispatch).
        ("nvidia/Qwen3.6-35B-A3B-NVFP4", "vllm", common.MoEQuantMode.w4a16_nvfp4),
        # trtllm/sglang keep the label mode: their dequant-to-BF16 weight-only
        # MoE path is the real dispatch (the Qwen3.6 profile above).
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "trtllm", common.MoEQuantMode.w4a16_nvfp4),
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "sglang", common.MoEQuantMode.w4a16_nvfp4),
        ("nvidia/Qwen3.6-35B-A3B-NVFP4", "trtllm", common.MoEQuantMode.w4a16_nvfp4),
    ],
)
def test_w4a16_nvfp4_label_resolves_to_the_backend_execution_lane(hf_id, backend, moe_mode):
    model_config = _model_config()
    get_model(hf_id, model_config, backend_name=backend)
    assert model_config.moe_quant_mode == moe_mode


def test_explicit_w4a16_nvfp4_moe_mode_wins_over_the_vllm_remap():
    model_config = _model_config()
    model_config.moe_quant_mode = common.MoEQuantMode.w4a16_nvfp4
    get_model("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", model_config, backend_name="vllm")
    assert model_config.moe_quant_mode == common.MoEQuantMode.w4a16_nvfp4


def test_w4a16_nvfp4_uses_scale_aware_weight_only_profile():
    for mode in (common.GEMMQuantMode.w4a16_nvfp4, common.MoEQuantMode.w4a16_nvfp4):
        assert mode.value.memory == 9 / 16
        assert mode.value.compute == 1
        assert mode.value.compute_dtype == "bfloat16"


def test_qwen35_nvfp4_exclusions_keep_attention_and_shared_experts_bf16():
    model = get_model("nvidia/Qwen3.5-122B-A10B-NVFP4", _model_config(), backend_name="trtllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    assert by_name["context_gdn_in_proj_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
    assert by_name["context_qkv_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
    assert by_name["context_gdn_shared_gate_up_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
    assert by_name["context_gdn_moe"]._quant_mode == common.MoEQuantMode.nvfp4


QWEN35_NVFP4_V2 = "nvidia/Qwen3.5-397B-A17B-NVFP4-V2"
# nvidia/Qwen3.5-397B-A17B-NVFP4-V2@8f590eae8f10bf55d9a46f79ea0280bde435c9f8, byte for byte
# (model_configs/QWEN35_397B_NVFP4_V2_PROVENANCE.md).
QWEN35_NVFP4_V2_SHA256 = {
    "config": "9408a9e559cc2f05f0b357738213666353e6651160ce8ff477b1c26982bc4f63",
    "hf_quant_config": "2be0a371edb1983ace9b776b12ff3a19e2b4841efc15c69df8d9f4aa5ec1c56c",
}


@pytest.mark.parametrize("suffix", list(QWEN35_NVFP4_V2_SHA256))
def test_qwen35_nvfp4_v2_bundle_is_the_pinned_upstream_bytes(suffix):
    configs = Path(aisimulate_core.__file__).parent / "model_configs"
    data = (configs / f"nvidia--Qwen3.5-397B-A17B-NVFP4-V2_{suffix}.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == QWEN35_NVFP4_V2_SHA256[suffix]


@contextlib.contextmanager
def _offline_bundle(monkeypatch):
    """Patch the hub download helpers out and clear the model-config caches around a bundled-config load."""
    import aisimulate.sdk.utils as sdk_utils
    from aisimulate_core.sdk.models.helpers import _get_model_info

    def _no_network(*args, **kwargs):
        raise AssertionError("network path reached")

    monkeypatch.setattr(sdk_utils, "_download_hf_config", _no_network)
    monkeypatch.setattr(sdk_utils, "_download_hf_json", _no_network)
    caches = (sdk_utils.get_model_config_from_model_path, sdk_utils._load_model_config_from_model_path, _get_model_info)
    for cached in caches:
        cached.cache_clear()
    try:
        yield
    finally:
        for cached in caches:
            cached.cache_clear()


def _get_qwen35_nvfp4_v2(monkeypatch, backend, nextn=0):
    """Build the V2 model from the bundled config with the hub download helpers patched out."""
    model_config = _model_config()
    model_config.nextn = nextn
    with _offline_bundle(monkeypatch):
        return get_model(QWEN35_NVFP4_V2, model_config, backend_name=backend)


def _flatten_ops(phase_ops):
    """Yield leaf ops; vLLM and SGLang decode nest the routed and shared experts in an OverlapOp."""
    for op in phase_ops:
        if isinstance(op, OverlapOp):
            yield from op._group_a
            yield from op._group_b
        else:
            yield op


@pytest.mark.parametrize("backend", ["sglang", "vllm", "trtllm"])
def test_qwen35_nvfp4_v2_uses_fp8_static_projections_and_nvfp4_experts(backend, monkeypatch):
    model = _get_qwen35_nvfp4_v2(monkeypatch, backend)
    by_name = {op._name: op for op in _flatten_ops(model.context_ops + model.generation_ops)}

    # quantized_layers lists the GDN in_proj_qkv, in_proj_z and out_proj, the full-attention
    # q/k/v/o and the shared-expert gate/up/down projections as static per-tensor FP8. The
    # export keeps in_proj_a and in_proj_b in BF16; pricing their GEMM (in_proj_ba) with the
    # other GDN projections is an approximation that docs/perf-model/support-matrix.md states.
    for name in (
        "context_gdn_in_proj_gemm",
        "context_gdn_in_proj_ba_gemm",
        "context_gdn_out_proj_gemm",
        "context_qkv_gemm",
        "context_proj_gemm",
        "context_gdn_shared_gate_up_gemm",
        "context_gdn_shared_down_gemm",
        "context_full_shared_gate_up_gemm",
        "context_full_shared_down_gemm",
        "generation_gdn_in_proj_gemm",
        "generation_gdn_in_proj_ba_gemm",
        "generation_gdn_out_proj_gemm",
        "generation_qkv_gemm",
        "generation_proj_gemm",
        "generation_gdn_shared_gate_up_gemm",
        "generation_gdn_shared_down_gemm",
        "generation_full_shared_gate_up_gemm",
        "generation_full_shared_down_gemm",
    ):
        assert by_name[name]._quant_mode == common.GEMMQuantMode.fp8_static, name
    # The routed experts are NVFP4.
    for name in ("context_gdn_moe", "context_full_moe", "generation_gdn_moe", "generation_full_moe"):
        assert by_name[name]._quant_mode == common.MoEQuantMode.nvfp4, name
    # The router (mlp.gate), the scalar shared-expert gate and lm_head are not quantized. SGLang
    # fuses the scalar gate into the merge kernel, so only vLLM and TRT-LLM emit its GEMM.
    bf16_names = (
        "context_gdn_router_gemm",
        "context_full_router_gemm",
        "context_logits_gemm",
        "generation_gdn_router_gemm",
        "generation_full_router_gemm",
        "generation_logits_gemm",
    )
    if backend != "sglang":
        bf16_names += (
            "context_gdn_shared_expert_gate_gemm",
            "context_full_shared_expert_gate_gemm",
            "generation_gdn_shared_expert_gate_gemm",
            "generation_full_shared_expert_gate_gemm",
        )
    for name in bf16_names:
        assert by_name[name]._quant_mode == common.GEMMQuantMode.bfloat16, name


@pytest.mark.parametrize("backend", ["sglang", "vllm", "trtllm"])
@pytest.mark.parametrize("nextn", [1, 3])
def test_qwen35_nvfp4_v2_mtp_scales_only_generation_ops(backend, nextn, monkeypatch):
    base = _get_qwen35_nvfp4_v2(monkeypatch, backend)
    mtp = _get_qwen35_nvfp4_v2(monkeypatch, backend, nextn=nextn)
    # The decode graph scales by (nextn + L) / L for L = 60 decoder layers (45 GDN, 15 full
    # attention), so the BF16 MTP layer is priced with the decoder-layer lanes. Context does not scale.
    num_layers = 60
    scale = (nextn + num_layers) / num_layers

    assert mtp._mtp_scale_factor == pytest.approx(scale)
    for base_ops, mtp_ops, expected in (
        (base.context_ops, mtp.context_ops, 1.0),
        (base.generation_ops, mtp.generation_ops, scale),
    ):
        base_by_name = {op._name: op for op in _flatten_ops(base_ops)}
        mtp_by_name = {op._name: op for op in _flatten_ops(mtp_ops)}
        assert mtp_by_name.keys() == base_by_name.keys()
        for name, op in mtp_by_name.items():
            assert op._scale_factor == pytest.approx(base_by_name[name]._scale_factor * expected), name


def _qwen35_nvfp4_v2_static_phase(monkeypatch, *, prefill, batch_size, nextn=0):
    """Per-op diagnostics of one static phase, from the V2 engine compiled on the b200_sxm SGLang 0.5.14 tables.

    Those packaged tables hold NVFP4 MoE rows for 512 experts at top-10, per-tensor FP8 and BF16 GEMM rows, and
    the FP8 compute-scale rows that the fp8_static path subtracts. The shared layer is off, so no row comes from
    another version. In SILICON mode a missing row raises, and op_level with fallback denied permits no other
    estimator.
    """
    with _offline_bundle(monkeypatch):
        model = RustForwardPassPerfModel.best_available(
            ForwardPassPerfModelConfig(
                model=QWEN35_NVFP4_V2,
                system="b200_sxm",
                backend="sglang",
                backend_version="0.5.14",
                worker_type="aggregated",
                tp=4,
                moe_tp_size=4,
                moe_ep_size=1,
                nextn=nextn,
                database_mode="SILICON",
                estimation_mode="op_level",
                fallback_policy="deny",
                enable_shared_layer=False,
            )
        )
        try:
            ops = model.static_phase_diagnostics(prefill=prefill, batch_size=batch_size, context_length=1024)
        finally:
            model.close()
    return {op["name"]: op for op in ops}


def test_qwen35_nvfp4_v2_engine_prices_each_lane_from_the_packaged_tables(monkeypatch):
    prefill = _qwen35_nvfp4_v2_static_phase(monkeypatch, prefill=True, batch_size=1)
    decode = _qwen35_nvfp4_v2_static_phase(monkeypatch, prefill=False, batch_size=8)

    # The GDN, full-attention and shared-expert projections take the fp8_static path: per-tensor FP8 GEMM rows
    # minus the compute-scale rows. The engine labels every fp8_static result "estimated" by construction
    # (GemmOp::query in crates/core/src/perfmodel/operators/gemm.rs).
    projections = ("gdn_in_proj_gemm", "gdn_in_proj_ba_gemm", "gdn_out_proj_gemm", "qkv_gemm", "proj_gemm")
    shared_experts = (
        "gdn_shared_gate_up_gemm",
        "gdn_shared_down_gemm",
        "full_shared_gate_up_gemm",
        "full_shared_down_gemm",
    )
    for name in projections + shared_experts:
        assert prefill[f"context_{name}"]["source"] == "estimated", name
    for name in projections:
        assert decode[f"generation_{name}"]["source"] == "estimated", name
    # The routed experts read measured NVFP4 MoE rows, and the router and lm_head GEMMs read measured BF16 GEMM
    # rows; the graph test above pins their quant modes.
    for name in ("gdn_moe", "full_moe", "gdn_router_gemm", "full_router_gemm", "logits_gemm"):
        assert prefill[f"context_{name}"]["source"] == "silicon", name
    assert decode["generation_logits_gemm"]["source"] == "silicon"
    # SGLang decode nests the router, the routed experts and the shared experts in one overlap op per layer type.
    # The engine reports that op under one label merged from its members, so prefill pins their lanes.
    for name in ("generation_gdn_moe_overlap", "generation_full_moe_overlap"):
        assert decode[name]["source"] == "mixed", name


def test_qwen35_nvfp4_v2_engine_scales_each_decode_op_by_the_mtp_factor(monkeypatch):
    # The engine multiplies the decode batch by (nextn + 1), so nextn=1 at batch 4 and nextn=0 at batch 8 query
    # every decode op at the same shape. The nextn=1 estimate then differs only by the (nextn + L) / L decode
    # scale for L = 60 decoder layers.
    num_layers = 60
    scale = (1 + num_layers) / num_layers
    base = _qwen35_nvfp4_v2_static_phase(monkeypatch, prefill=False, batch_size=8)
    mtp = _qwen35_nvfp4_v2_static_phase(monkeypatch, prefill=False, batch_size=4, nextn=1)

    assert mtp.keys() == base.keys()
    # With pp=1 only the P2P op prices at zero, so every other op's comparison below has a nonzero base.
    assert {name for name, op in base.items() if op["latency_ms"] == 0} == {"generation_p2p"}
    for name, op in mtp.items():
        assert op["latency_ms"] == pytest.approx(base[name]["latency_ms"] * scale, rel=1e-9), name


def test_gemma4_exclusions_keep_attention_and_mlp_bf16():
    model = get_model("nvidia/Gemma-4-26B-A4B-NVFP4", _model_config(), backend_name="trtllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    assert by_name["context_swa_qkv_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
    assert by_name["context_swa_shared_mlp_gate_up_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
    assert by_name["context_swa_moe"]._quant_mode == common.MoEQuantMode.bfloat16


def test_kimi_k26_exclusions_keep_attention_and_shared_experts_bf16():
    model = get_model("nvidia/Kimi-K2.6-NVFP4", _model_config(), backend_name="vllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}
    attention_block = by_name["context_mla_block"]

    assert attention_block._primary._gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert all(
        op._quant_mode == common.GEMMQuantMode.bfloat16
        for op in attention_block._fallback
        if hasattr(op, "_quant_mode")
    )
    assert by_name["context_shared_gate_up_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
    assert by_name["context_moe"]._quant_mode == common.MoEQuantMode.nvfp4


def test_minimax_m3_uses_mxfp8_lane_for_non_experts_and_nvfp4_for_experts():
    model = get_model("nvidia/MiniMax-M3-NVFP4", _model_config(), backend_name="trtllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    # MXFP8 non-expert lane prices as fp8_block (owner decision 2026-08-09).
    assert by_name["context_shared_gate_up_gemm"]._quant_mode == common.GEMMQuantMode.fp8_block
    assert by_name["context_moe"]._quant_mode == common.MoEQuantMode.nvfp4


@pytest.mark.parametrize(
    "hf_id",
    ("nvidia/DeepSeek-V4-Flash-NVFP4", "nvidia/DeepSeek-V4-Pro-NVFP4"),
)
def test_dsv4_nvfp4_experts_skip_native_mxfp4_backend_remap(hf_id):
    from aisimulate.sdk.models.helpers import resolve_dsv4_moe_arch_mode

    assert resolve_dsv4_moe_arch_mode(hf_id, "b200_sxm", "sglang") is None
    assert resolve_dsv4_moe_arch_mode(hf_id, "h200_sxm", "sglang") is None


@pytest.mark.parametrize(
    "hf_id",
    ("nvidia/DeepSeek-V4-Flash-NVFP4", "nvidia/DeepSeek-V4-Pro-NVFP4"),
)
def test_dsv4_nvfp4_experts_preserve_fp8_block_nonexpert_lane(hf_id):
    model_config = _model_config()
    model = get_model(hf_id, model_config, backend_name="trtllm")

    assert model_config.gemm_quant_mode == common.GEMMQuantMode.fp8_block
    assert model_config.moe_quant_mode == common.MoEQuantMode.nvfp4
    module_modes = {
        op._gemm_quant_mode for op in model.context_ops + model.generation_ops if hasattr(op, "_gemm_quant_mode")
    }
    assert module_modes == {common.GEMMQuantMode.fp8_block}


@pytest.mark.parametrize("backend", ["vllm", "trtllm", "sglang"])
@pytest.mark.parametrize("explicit", [False, True])
def test_nemotron_super_omni_preserves_backend_and_explicit_moe_modes(backend, explicit):
    from aisimulate_core.sdk.models.helpers import _apply_model_quant_defaults
    from aisimulate_core.sdk.utils import _infer_quantization_fields

    model_config = _model_config()
    if explicit:
        model_config.moe_quant_mode = common.MoEQuantMode.w4a16_nvfp4
    raw_config = {
        "architectures": ["NemotronH_Omni_Reasoning_V3"],
        "llm_config": {
            "quantization_config": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {"backbone.layers.1.mixer.experts.0.up_proj": {"quant_algo": "W4A16_NVFP4"}},
            }
        },
    }
    raw_config.update(_infer_quantization_fields(raw_config))
    _apply_model_quant_defaults(model_config, raw_config, "NemotronH_Omni_Reasoning_V3", backend)
    expected = common.MoEQuantMode.nvfp4 if backend == "vllm" and not explicit else common.MoEQuantMode.w4a16_nvfp4
    assert model_config.moe_quant_mode == expected
