# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM MoE collector for the torch-flow create_moe API.

One case builds one routed-expert layer the way serving builds it: the
backend serving's AUTO resolution names for the model's architecture and
quant mode, the architecture's routing method and activation, the framework's
own implementation resolution, post_load_weights, and serving's warmup
autotune. It then times the case's token sweep under CUDA graphs. Shared MoE
shapes come from YAML; this file owns the TRT-LLM construction.

Line citations name tensorrt_llm/_torch/ files at v1.4.0rc2.
"""

__compat__ = "trtllm==1.4.0rc2"

import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import tensorrt_llm
import torch
from collector.case_generator import get_common_moe_test_cases, get_moe_quantization_module_config
from collector.helper import (
    EXIT_CODE_RESTART,
    FIRST_POINT_WARMUP_SECONDS,
    balanced_logits,
    benchmark_with_power,
    get_sm_version,
    log_perf,
    power_law_logits_v3,
)
from collector.op_plans import plan_cases
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate
from tensorrt_llm._torch.moe.fused_moe import (
    DEFAULT_MOE_ACTIVATION,
    ConfigurableMoE,
    DeepSeekV3MoeRoutingMethod,
    Llama4RenormalizeMoeRoutingMethod,
    MiniMaxM2MoeRoutingMethod,
    RenormalizeMoeRoutingMethod,
    SimpleActivation,
    SiTuActivation,
    SwigluActivation,
    SwigluBiasActivation,
    TritonFusedMoE,
    create_moe,
)
from tensorrt_llm._torch.pyexecutor.model_loader import initialize_dummy_weights
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

_MXFP4_MOE_TYPES = {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}

aic_debug = int(os.getenv("aic_moe_debug", "0"))  # noqa: SIM112

moe_tune_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_tuned_cache_path")
_TRTLLM_VERSION = tensorrt_llm.__version__
_MODEL_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "src/aisimulate_core/model_configs"

# Serving's max_num_tokens default (BaseLlmArgs, llmapi/llm_args.py:5932-5933).
# The autotuner warmup forwards that many tokens, then one generation token
# (pyexecutor/engine/runners/decoder/warmup.py:1059-1068, 1099-1104, 1158), and
# both MoE tuning bounds follow it (_run_moe_case).
_SERVING_MAX_NUM_TOKENS = 8192

# kernel_source per (provider, technique) of the implementation identity the
# framework resolved (moe/fused_moe/impl_identity.py MoEImplId; cutlass/identity.py,
# trtllm_gen/identity.py). A layer is one implementation: its autotuner tactics are
# configurations of that implementation's kernels, so the label never depends on them.
_MOE_KERNEL_SOURCES = {
    ("trtllm", "cutlass"): "moe_torch_flow_cutlass",
    # TrtllmTritonFp8BlockScalesImpl: fp8_block on SM120 runs the whole MoE in
    # run_triton_fp8_block_scale_moe (cutlass/fp8_block_scales_triton.py:39-44, 109).
    ("trtllm", "triton"): "moe_torch_flow_triton_fp8_block",
    ("trtllm", "trtllm_gen"): "moe_torch_flow_trtllm_gen",
}


def gc_collect():
    """Run GC and clear CUDA cache to reduce fragmentation between runs."""
    for _ in range(2):
        gc.collect()
        torch.cuda.empty_cache()


def get_moe_test_cases():
    return plan_cases("moe", "trtllm", get_sm_version())


def _declared_moe_routing(model_name: str):
    """The model's row in cases/models/*_cases.yaml (architecture and routing constants).

    The routing constants (sglang_moe_num_expert_group / _topk_group /
    _routed_scaling_factor / _swiglu_limit) are framework-neutral model facts
    despite the sglang_ prefix: TensorRT-LLM reads the same numbers from the
    checkpoint config (n_group, topk_group, routed_scaling_factor, swiglu_limit).
    """
    for case in get_common_moe_test_cases():
        if case.model_name == model_name:
            return case
    return None


def _packaged_model_config(model_name: str) -> dict:
    """The checked-in HF config of ``model_name`` (its text sub-config for VLM wrappers)."""
    config_path = _MODEL_CONFIG_ROOT / f"{model_name.replace('/', '--')}_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing packaged model config for TRT-LLM MoE case {model_name!r}: {config_path}")
    with config_path.open() as config_file:
        config = json.load(config_file)
    return config.get("text_config", config)


def _zero_routing_bias(gate) -> None:
    """Serving gates allocate e_score_correction_bias with torch.empty and only
    load_weights fills it (DeepseekV3Gate, models/modeling_deepseekv3.py:899-901;
    DeepseekV4Gate and MiniMaxM3Gate likewise). Their top-k ranks
    score + bias, and the collector's balanced / power-law logits separate the
    intended experts by ~1e-3 after the sigmoid, so whatever that memory holds
    would pick the experts instead of the row's distribution. Zero it, as
    vllm/collect_moe.py and wideep/trtllm/collect_moe_a2a.py do."""
    with torch.no_grad():
        gate.e_score_correction_bias.zero_()


def _serving_moe_gate(declared, hidden_size, num_experts, topk, dtype, moe_backend):
    """The gate serving builds for a DeepSeek-style architecture, with a zeroed bias.

    DeepSeek-V3-family / GLM / Kimi-K2.x / Nemotron-H: DeepseekV3Gate
    (models/modeling_deepseekv3.py:866-911, built at :1002-1014;
    models/modeling_nemotron_h.py:245-256). DeepSeek-V4: DeepseekV4Gate, the
    scored (non-hashed) layer (models/modeling_deepseekv4.py:1508, built at
    :1648-1662); the first n_hash_layers route by token id (tid2eid) and have no
    logit distribution to label. MiniMax-M3: MiniMaxM3Gate
    (models/modeling_minimaxm3.py:595-625). moe_backend is the resolved
    upper-case name serving passes; DeepseekV3Gate keeps a bf16 bias only for
    'TRTLLM' (modeling_deepseekv3.py:895-898).
    """
    n_group = int(declared.sglang_moe_num_expert_group or 1)
    topk_group = int(declared.sglang_moe_topk_group or 1)
    routed_scaling_factor = float(declared.sglang_moe_routed_scaling_factor or 1.0)
    backend = str(moe_backend).upper()
    if declared.architecture == "DeepseekV4ForCausalLM":
        from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Gate

        gate = DeepseekV4Gate(
            hidden_size,
            num_experts,
            top_k=topk,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            is_hashed=False,
            dtype=dtype,
            moe_backend=backend,
        )
    elif declared.architecture in _MINIMAX_M3_ARCHITECTURES:
        from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Gate

        gate = MiniMaxM3Gate(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=topk,
            routed_scaling_factor=routed_scaling_factor,
        )
    else:
        gate = DeepseekV3Gate(
            hidden_size,
            num_experts,
            top_k=topk,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            dtype=dtype,
            moe_backend=backend,
        )
    _zero_routing_bias(gate)
    return gate


# Serving's routed-expert layer per HF architecture. TensorRT-LLM picks the
# modeling file by architecture, and that file fixes the routing method, the
# activation, the expert bias and apply_router_weight_on_input it hands to
# create_moe. These sets name the architectures that share one construction.
_DEEPSEEK_V3_GATE_ARCHITECTURES = frozenset(
    {
        "DeepseekV3ForCausalLM",
        # Subclasses DeepseekV3ForCausalLM unchanged (modeling_deepseekv3.py:2159-2160).
        "GlmMoeDsaForCausalLM",
        # Text model is DeepseekV3ForCausalLM (models/modeling_kimi_k25.py:1660).
        "KimiK25ForConditionalGeneration",
        "NemotronHForCausalLM",
    }
)
# MiniMaxM3ForCausalLM is the bf16 checkpoint's architecture string; TRT-LLM
# registers the Sparse names (modeling_minimaxm3.py:2948, :3105) and builds one MoE.
_MINIMAX_M3_ARCHITECTURES = frozenset(
    {"MiniMaxM3ForCausalLM", "MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"}
)
_KIMI_K3_ARCHITECTURES = frozenset({"KimiK3ForConditionalGeneration", "KimiLinearForCausalLM"})
_RENORMALIZE_ARCHITECTURES = frozenset(
    {
        # models/modeling_mixtral.py:64-75.
        "MixtralForCausalLM",
        # Qwen3Gate, routing_method_type=Renormalize (models/modeling_qwen3_moe.py:120-127);
        # Qwen3-VL-MoE maps to Qwen3MoeForCausalLM (models/modeling_qwen3vl.py:1508).
        "Qwen3MoeForCausalLM",
        "Qwen3VLMoeForConditionalGeneration",
        # Qwen3NextGate, routing_method_type=Renormalize (models/modeling_qwen3_next.py:261-269),
        # via Qwen3_5Moe* (models/modeling_qwen3_5.py:670, :825).
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    }
)
# Qwen3MoE re-reads the routing output dtype from the bound backend after
# create_moe (models/modeling_qwen3_moe.py:146-156); Qwen3Next does not.
_ROUTING_SCALES_FROM_BACKEND_ARCHITECTURES = frozenset({"Qwen3MoeForCausalLM", "Qwen3VLMoeForConditionalGeneration"})


def _serving_moe_layer(declared, model_name, hidden_size, num_experts, topk, dtype, moe_backend):
    """What serving passes to create_moe for this architecture, besides shapes and quant.

    Returns routing_method, activation, bias, apply_router_weight_on_input, the
    dtype of the router logits serving feeds the layer, and whether the routing
    output dtype is re-read from the bound backend. Routing biases and scales
    are neutral values (zero bias, unit per-expert scale): they change which
    experts the logits pick, not which kernels run, and the case's logits pick
    the experts. Activation constants are serving's: they select the kernel's
    epilogue and its eligibility, not only its values.
    """
    architecture = declared.architecture
    layer = SimpleNamespace(
        activation=DEFAULT_MOE_ACTIVATION,
        bias=False,
        apply_router_weight_on_input=False,
        router_logits_dtype=torch.bfloat16,
        routing_scales_from_backend=architecture in _ROUTING_SCALES_FROM_BACKEND_ARCHITECTURES,
        gate=None,
    )
    if architecture in _DEEPSEEK_V3_GATE_ARCHITECTURES or architecture == "DeepseekV4ForCausalLM":
        # DeepseekV3Gate / DeepseekV4Gate emit fp32 logits (dsv3_router_gemm_op,
        # modeling_deepseekv3.py:913-936, modeling_deepseekv4.py:1586-1595).
        layer.gate = _serving_moe_gate(declared, hidden_size, num_experts, topk, dtype, moe_backend)
        layer.routing_method = layer.gate.routing_method
        layer.router_logits_dtype = torch.float32
        if architecture == "NemotronHForCausalLM":
            # Relu2 for every Nemotron-H checkpoint, FP8 included: the file has
            # no other branch (models/modeling_nemotron_h.py:189, create_moe :264-278).
            layer.activation = SimpleActivation(kind=ActivationType.Relu2)
        elif architecture == "DeepseekV4ForCausalLM" and declared.sglang_moe_swiglu_limit is not None:
            # SwiGLU clamped before the SiLU (models/modeling_deepseekv4.py:1670-1675).
            layer.activation = SwigluActivation(clamp=float(declared.sglang_moe_swiglu_limit))
    elif architecture == "Glm5NextForConditionalGeneration":
        # Glm5NextMoE (models/modeling_glm5_next.py:1552-1616): Glm5NextGate, a
        # DeepseekV3Gate built from the checkpoint config that keeps an fp32
        # correction bias and fp32 logits whatever the backend (:1520-1549), its
        # routing method, and SwiGLU clamped at the config's swiglu_limit
        # (:1573, :1613); no expert bias, weights applied after FC2.
        from tensorrt_llm._torch.models.modeling_glm5_next import Glm5NextGate

        config = _packaged_model_config(model_name)
        layer.gate = Glm5NextGate(
            SimpleNamespace(
                hidden_size=hidden_size,
                n_routed_experts=num_experts,
                num_experts_per_tok=topk,
                n_group=config.get("n_group"),
                topk_group=config.get("topk_group"),
                routed_scaling_factor=config["routed_scaling_factor"],
                scoring_func=config.get("scoring_func", "sigmoid"),
                norm_topk_prob=config["norm_topk_prob"],
            ),
            moe_backend=str(moe_backend),
        )
        _zero_routing_bias(layer.gate)
        layer.routing_method = layer.gate.routing_method
        layer.router_logits_dtype = torch.float32
        layer.activation = SwigluActivation(clamp=float(config["swiglu_limit"]))
    elif architecture in _MINIMAX_M3_ARCHITECTURES:
        layer.gate = _serving_moe_gate(declared, hidden_size, num_experts, topk, dtype, moe_backend)
        layer.routing_method = layer.gate.routing_method
        layer.router_logits_dtype = torch.float32
        # swiglu_alpha / swiglu_limit defaults are the checkpoint's 1.702 / 7.0
        # (models/modeling_minimaxm3.py:768-779).
        layer.activation = SwigluBiasActivation(gate_sigmoid_scale=1.702, linear_offset=1.0, clamp=7.0)
    elif architecture in _KIMI_K3_ARCHITECTURES:
        # KimiK3MoEGate.routing_method with its fp32 bias and fp32 logits
        # (models/modeling_kimi_linear.py:163, :223-230), and the SiTu experts
        # (:1279-1282) with the checkpoint's betas (_resolve_kimi_situ_betas, :255-269).
        bias = torch.zeros(num_experts, dtype=torch.float32)
        layer.routing_method = DeepSeekV3MoeRoutingMethod(
            top_k=topk,
            n_group=int(declared.sglang_moe_num_expert_group or 1),
            topk_group=int(declared.sglang_moe_topk_group or 1),
            routed_scaling_factor=float(declared.sglang_moe_routed_scaling_factor or 1.0),
            callable_e_score_correction_bias=lambda: bias,
            is_fused=True,
        )
        layer.router_logits_dtype = torch.float32
        config = _packaged_model_config(model_name)
        situ_beta = config.get("activation_situ_beta")
        layer.activation = SiTuActivation(
            gate_softcap=1.0 if situ_beta is None else float(situ_beta),
            linear_softcap=float(config["activation_situ_linear_beta"]),
        )
    elif architecture == "MiniMaxM2ForCausalLM":
        # fp32 gate and fp32 bias holder (models/modeling_minimaxm2.py:51, :87-107).
        bias = torch.zeros(num_experts, dtype=torch.float32)
        layer.routing_method = MiniMaxM2MoeRoutingMethod(
            top_k=topk, num_experts=num_experts, callable_e_score_correction_bias=lambda: bias
        )
        layer.router_logits_dtype = torch.float32
    elif architecture in _RENORMALIZE_ARCHITECTURES:
        layer.routing_method = RenormalizeMoeRoutingMethod(top_k=topk)
    elif architecture == "Llama4ForConditionalGeneration":
        # models/modeling_llama.py:326-338.
        layer.routing_method = Llama4RenormalizeMoeRoutingMethod(topk)
        layer.apply_router_weight_on_input = True
    elif architecture == "Gemma4ForConditionalGeneration":
        # Gemma4MoeRoutingMethod (models/modeling_gemma4.py:684-716) and GeGLU
        # experts (:741-756); per_expert_scale is a torch_dtype parameter (:732-735).
        from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4MoeRoutingMethod

        per_expert_scale = torch.ones(num_experts, dtype=dtype)
        layer.routing_method = Gemma4MoeRoutingMethod(top_k=topk, callable_per_expert_scale=lambda: per_expert_scale)
        layer.activation = SimpleActivation(kind=ActivationType.Geglu)
    elif architecture == "GptOssForCausalLM":
        # models/modeling_gpt_oss.py:173-200: bf16 routing weights for TRTLLM,
        # fp32 otherwise; SwiGLU-bias constants; expert bias.
        layer.routing_method = RenormalizeMoeRoutingMethod(
            top_k=topk, output_dtype=torch.bfloat16 if moe_backend == "TRTLLM" else torch.float32
        )
        layer.activation = SwigluBiasActivation(gate_sigmoid_scale=1.702, linear_offset=1.0, clamp=7.0)
        layer.bias = True
    else:
        # Step3p7FlashForCausalLM, for one: no TRT-LLM model registers it (the
        # Step3p7 text model registers as "Step3p5ForCausalLM",
        # models/modeling_step3p7.py:1557), so serving cannot load it.
        raise ValueError(
            f"TensorRT-LLM {_TRTLLM_VERSION} builds no MoE layer for architecture {architecture!r} "
            f"({model_name}); add its serving construction to _serving_moe_layer"
        )
    return layer


def _moe_quant_config(moe_type: str, model_name: str) -> QuantConfig:
    """The experts' QuantConfig for ``moe_type``; the activation dtype stays bf16."""
    quant_group_size = 128
    quant_algo = None
    # int4_wo: collector/skips.py skips it on every SM (no TRT-LLM implementation), so it is never queued.
    if moe_type == "fp8_block":
        quant_algo = QuantAlgo.FP8_BLOCK_SCALES
    elif moe_type == "w4afp8":
        quant_algo = QuantAlgo.W4A8_AWQ
    elif moe_type == "fp8":
        quant_algo = QuantAlgo.FP8
    elif moe_type == "int4_wo":
        quant_algo = QuantAlgo.W4A16
        int4_config = get_moe_quantization_module_config("trtllm", moe_type, model_name=model_name)
        quant_group_size = int(int4_config.get("group_size", 128))
    elif moe_type == "nvfp4":
        quant_algo = QuantAlgo.NVFP4
        quant_group_size = 16
    elif moe_type == "w4a16_mxfp4":
        quant_algo = QuantAlgo.W4A16_MXFP4
        quant_group_size = 32
    elif moe_type == "w4a8_mxfp4_mxfp8":
        quant_algo = QuantAlgo.W4A8_MXFP4_MXFP8
        quant_group_size = 32
    return QuantConfig(
        quant_algo=quant_algo,
        kv_cache_quant_algo=None,
        group_size=quant_group_size,
        smoothquant_val=0.5,
        clamp_val=None,
        use_meta_recipe=False,
        has_zero_point=False,
        pre_quant_scale=False,
        exclude_modules=None,
    )


def _refuse_moe_weight_capacity(
    hidden_size, inter_size, num_experts, moe_tp_size, moe_ep_size, weight_bits, is_gated, capacity_bytes
):
    """Reject only a proven lower bound: local resident expert matrices, no workspace estimate."""
    # create_moe's gate/up projection has 2*I rows for gated activations and I for Relu2;
    # the down projection adds H*I. Mapping shards experts by EP and I by TP.
    matrices = 3 if is_gated else 2
    local_elements = (num_experts // moe_ep_size) * hidden_size * (inter_size // moe_tp_size) * matrices
    minimum_bytes = (local_elements * weight_bits + 7) // 8
    if minimum_bytes >= capacity_bytes:
        raise RuntimeError(
            f"FIXME(kernel-limit): does not fit: resident expert weights need at least {minimum_bytes} bytes, "
            f"device capacity is {capacity_bytes} bytes; excludes scales, inputs, outputs and workspace"
        )


def _post_load_weights(moe) -> None:
    """Serving's post-load hook on every module of the layer, once its weights are in.

    ModelLoader._walk_full_post_load (pyexecutor/model_loader.py:1898-1915)
    calls post_load_weights on every module not marked _weights_removed;
    ConfigurableMoE marks itself so only its execution unit runs it
    (moe/fused_moe/configurable_moe.py:271-276). The unit's quant method then
    applies its load-time transforms (MoEImplBlocks.post_load_weights,
    moe/fused_moe/impl_blocks.py:213-215): CUTLASS FP8 pads the expert weights
    to the GEMM alignment (moe/fused_moe/quantization.py:1187-1222), DeepGEMM
    re-lays FP8 block scales as UE8M0 (:1508-1527), and every method rebuilds
    its quant-scale tuple. All of it runs here, before any timing.
    """
    for module in moe.modules():
        hook = getattr(module, "post_load_weights", None)
        if hook is not None and not getattr(module, "_weights_removed", False):
            hook()


def _moe_kernel_source(moe, min_latency_mode: bool) -> str:
    """The kernel_source of the implementation the framework built for this layer.

    create_moe resolves the requested backend to one implementation class
    (moe/fused_moe/create_moe.py:344-362, moe_resolution.py:502-658),
    degrading to the CUTLASS leaves when the requested family declines the
    layer, as serving does; ConfigurableMoE then runs that class
    (configurable_moe.py:389). TritonFusedMoE is a layer of its own
    (create_moe.py:389-410). A class with no label fails the case before timing.
    """
    if isinstance(moe, TritonFusedMoE):
        return "moe_torch_flow_triton"
    descriptor = getattr(getattr(moe, "backend", None), "descriptor", None)
    if descriptor is None:
        raise RuntimeError(f"cannot name the MoE implementation TRT-LLM built: {type(moe).__name__}")
    identity = descriptor.identity
    if min_latency_mode:
        # create_moe(allow_backend_degradation=False) already refused any other family.
        if identity.technique != "trtllm_gen":
            raise RuntimeError(f"min-latency row resolved to {identity.canonical()}, not a TRTLLM-Gen implementation")
        return "moe_torch_flow_min_latency"
    if identity.provider == "deepgemm":
        return "deepgemm"
    source = _MOE_KERNEL_SOURCES.get((identity.provider, identity.technique))
    if source is None:
        raise RuntimeError(f"no kernel_source label for MoE implementation {identity.canonical()}")
    return source


def _tuning_cache_path(sm_version, moe_type, hidden_size, inter_size, topk, num_experts, moe_tp_size, moe_ep_size):
    """Profiling-cache file shared by the workers that tune one layer shape.

    autotune(cache_path=...) loads it, profiles only what it lacks and merges
    its entries back under a file lock (autotuner.py:281-370, 606-760), the way
    serving shares TLLM_AUTOTUNER_CACHE_PATH (warmup.py:1157-1158). Each entry
    is keyed by the exact runner and input buckets (AutoTunerProfilingCache.
    get_cache_key, autotuner.py:536-555), so the file name only groups layers;
    it carries the SM and the TRT-LLM version because tactic ids index a
    per-arch, per-release config table.
    """
    return (
        f"{moe_tune_path}/sm{sm_version}_trtllm{_TRTLLM_VERSION}_{moe_type}_h{hidden_size}"
        f"_i{inter_size // moe_tp_size}_e{num_experts // moe_ep_size}_k{topk}.json"
    )


def _run_moe_case(
    moe_type,
    num_tokens_lists,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    min_latency_mode,
    model_name,
    distributed="power_law",
    power_law_alpha=0.0,
    *,
    perf_filename,
    device="cuda:0",
):
    """Run one MoE case (its token sweep) and log latency/power rows."""
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    gc_collect()

    if aic_debug == 1:
        print("MOE Allocated GDRAM:", torch.cuda.memory_allocated(device.index) / 1024**2, "MB")
        print("MOE Reserved GDRAM:", torch.cuda.memory_reserved(device) / 1024**2, "MB")
    # ``dtype`` is the MODEL/activation dtype serving hands to create_moe
    # (models/modeling_deepseekv3.py:1028-1045 ``dtype=dtype`` = config.torch_dtype).
    # Quantization travels in quant_config only; activations are bf16 and are
    # quantized inside the kernel path.
    dtype = torch.bfloat16
    quant_config = _moe_quant_config(moe_type, model_name)

    if power_law_alpha - 0.0 < 1e-6:
        distributed = "balanced"

    declared = _declared_moe_routing(model_name)
    if declared is None:
        raise ValueError(f"no cases/models row declares MoE model {model_name!r}")

    sm_version = get_sm_version()

    # FIXME(kernel-limit): fp8_block weight scales are 128x128 blocks and no
    # fp8_block path pads, so hidden_size and the TP-sharded intermediate size
    # must be 128-aligned: SM100/103 TRTLLM-Gen checks it (TORCH_CHECK in
    # cpp/tensorrt_llm/thop/moe/trtllmGen/fp8BlockScaleMoe.cpp:338), SM90 runs
    # DeepGEMM's 1x128/128x128 layout (cutlass/fp8_block_scales_hopper.py) and
    # SM120 the Triton block-scale kernel at the same granularity
    # (cutlass/fp8_block_scales_triton.py). Raised before the build; re-verify
    # the per-SM paths on the next framework version bump.
    if moe_type == "fp8_block" and (hidden_size % 128 != 0 or (inter_size // moe_tp_size) % 128 != 0):
        raise ValueError(
            f"FIXME(kernel-limit): fp8_block MoE requires 128-aligned hidden_size and TP-sharded intermediate "
            f"size (128x128-blocked weight scales; trtllmGen/fp8BlockScaleMoe.cpp:338 on SM100/103, "
            f"DeepGEMM on SM90, Triton block-scale on SM120); got hidden_size={hidden_size}, "
            f"inter_size={inter_size} / moe_tp={moe_tp_size} = {inter_size // moe_tp_size}"
        )

    if moe_type == "w4afp8" and (inter_size // moe_tp_size) % 128 != 0:
        raise ValueError(
            f"FIXME(kernel-limit): w4afp8 MoE requires a 128-aligned TP-sharded intermediate size (grouped-GEMM "
            f"k alignment); got inter_size={inter_size} / moe_tp={moe_tp_size} = "
            f"{inter_size // moe_tp_size}"
        )

    # Serving's backend: ModelConfig.resolve_moe_backend turns the default AUTO
    # (MoeConfig.backend, llmapi/llm_args.py:1958-1964) into a backend name from
    # the architecture, the quant algo and the SM (model_config.py:517-599): TRTLLM
    # for Kimi-K3, for DeepSeek-V4 on SM100-119, for fp8_block on SM100/103 and
    # for GPT-OSS on Blackwell; TRITON for GPT-OSS on Hopper; CUTLASS otherwise.
    # The min-latency row times the unfinalized TRTLLM-Gen path serving takes only
    # with moe_backend TRTLLM (models/modeling_deepseekv3.py:1606-1610,
    # models/modeling_qwen3_moe.py:293-297).
    if min_latency_mode:
        moe_backend = "TRTLLM"
    else:
        moe_backend = ModelConfig.resolve_moe_backend("AUTO", declared.architecture, quant_config)

    # The layer is tuned as a default deployment tunes it: max_num_tokens 8192,
    # which bounds both MoE tunings. CUTLASS tunes power-of-2 buckets up to
    # min(moe_max_num_tokens, 16384 * slots / top_k, max_num_tokens * dp)
    # (moe/fused_moe/cutlass/base.py:228-241, custom_ops/torch_custom_ops.py:
    # 97-102, 327) and TRTLLM-Gen a power-of-2 ladder to 8192, plus
    # max_num_tokens only above it (trtllm_gen/base.py:225;
    # custom_ops/trtllm_gen_custom_ops.py:546-640). A larger M maps at run time
    # to the 8192 tactic: the lookup takes the bucket's last power of 2 and caps
    # it at the config's tune_max_num_tokens (autotuner.py:1807-1849; CUTLASS
    # last_positive_power_of_2, TRTLLM-Gen round_rule), so the sweep times it on
    # that tactic as serving runs it. moe_max_num_tokens stays at the sweep's
    # largest M, which keeps every token count in one chunk
    # (configurable_moe.py:244-258; it only enters CUTLASS's bound through the
    # min above).
    max_tokens = max(num_tokens_lists)
    max_num_tokens = _SERVING_MAX_NUM_TOKENS

    mapping = Mapping()
    mapping.moe_ep_size = moe_ep_size
    mapping.moe_tp_size = moe_tp_size
    # create_moe and CommunicationFactory read the shapes from pretrained_config.
    pretrained_config = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=inter_size,
        num_experts=num_experts,
        torch_dtype=dtype,
    )
    model_config = ModelConfig(pretrained_config=pretrained_config)
    model_config.mapping = mapping
    model_config.quant_config = quant_config
    model_config.max_num_tokens = max_num_tokens
    model_config.moe_max_num_tokens = max(max_num_tokens, max_tokens)
    model_config.moe_backend = moe_backend

    layer = _serving_moe_layer(declared, model_name, hidden_size, num_experts, topk, dtype, moe_backend)
    weight_bits = {
        "bfloat16": 16,
        "fp8": 8,
        "fp8_block": 8,
        "int4_wo": 4,
        "w4a16_mxfp4": 4,
        "w4a8_mxfp4_mxfp8": 4,
        "w4afp8": 4,
        "nvfp4": 4,
    }[moe_type]
    _refuse_moe_weight_capacity(
        hidden_size,
        inter_size,
        num_experts,
        moe_tp_size,
        moe_ep_size,
        weight_bits,
        ActivationType(layer.activation.kind) is not ActivationType.Relu2,
        torch.cuda.get_device_properties(device).total_memory,
    )
    # create_moe resolves the backend to one implementation, wraps it in
    # ConfigurableMoE and creates its weights (moe/fused_moe/create_moe.py:264-386).
    # Serving lets a declined backend degrade to the CUTLASS leaves; the
    # min-latency row must run TRTLLM-Gen, so it refuses degradation.
    moe = create_moe(
        routing_method=layer.routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=inter_size,
        dtype=dtype,
        reduce_results=False,
        model_config=model_config,
        bias=layer.bias,
        apply_router_weight_on_input=layer.apply_router_weight_on_input,
        activation=layer.activation,
        allow_backend_degradation=not min_latency_mode,
    )
    moe.to(device)
    if layer.routing_scales_from_backend and isinstance(moe, ConfigurableMoE):
        scales_dtype = moe.backend.input_requirement.routing_scales_dtype
        if scales_dtype is not None:
            layer.routing_method.output_dtype = scales_dtype
    kernel_source = _moe_kernel_source(moe, min_latency_mode)
    built = type(getattr(moe, "backend", moe)).__name__
    print(f"trtllm.moe: {model_name} {moe_type}: requested {moe_backend}, built {built} ({kernel_source})")

    # Both MXFP4 formats load real MXFP4 weights through load_weights() so the
    # TRTLLM-Gen shuffle/permutation (torch.ops.trtllm.shuffle_matrix) the
    # kernel's memory access expects is applied, as serving's loader does.
    # Every other quant mode fills create_weights' tensors as serving's dummy
    # load does, before its post-load walk: initialize_dummy_weights
    # (pyexecutor/model_loader.py:357-437, called for LoadFormat.DUMMY at
    # :1520-1523) draws FP8/INT8 from U(-3, 3), packed FP4 bytes from [0, 64),
    # other floats from U(-1e-3, 1e-3) and keeps the calibration scalars. The
    # weights must be dense: on create_weights' uninitialized memory the B200
    # FP8 block-scale layer timed 9-13% faster from 1024 tokens than on these.
    # Deviation note: load_weights is skipped for these modes, so a layout
    # permutation it would apply is skipped too. The kernels and their shapes
    # are serving's; only the data order differs, which their timing does not
    # depend on.
    if moe_type not in _MXFP4_MOE_TYPES:
        initialize_dummy_weights(moe)
    else:
        local_num_experts = num_experts // moe_ep_size

        from triton_kernels.numerics_details.mxfp import downcast_to_mxfp_torch

        def fp32_to_mxfp4(tensor):
            tensor = tensor.transpose(1, 2).contiguous()
            tensor_fp4, tensor_scales = downcast_to_mxfp_torch(tensor, torch.uint8, axis=1)
            tensor_fp4 = tensor_fp4.transpose(1, 2).contiguous()
            tensor_scales = tensor_scales.transpose(1, 2).contiguous()
            return tensor_fp4, tensor_scales

        def random_mxfp4(rows, cols):
            # Random weights for every local expert, converted about 1 GiB of experts at a
            # time: the downcast holds several fp32 copies of its input, and a whole Kimi-K3
            # layer at TP1/EP1 (896 experts) did not fit a 180 GB GPU that way (CUDA OOM in
            # downcast_to_mxfp_torch). Serving loads already-quantized checkpoint weights.
            step = max(1, (1 << 30) // (rows * cols * 4))
            fp4, scales = [], []
            for start in range(0, local_num_experts, step):
                weight = torch.randn((min(step, local_num_experts - start), rows, cols), dtype=dtype, device=device)
                q, s = fp32_to_mxfp4(weight)
                del weight
                fp4.append(q)
                scales.append(s)
            out = (torch.cat(fp4), torch.cat(scales))
            del fp4, scales
            torch.cuda.empty_cache()
            return out

        w1_weight_fp4, w1_weight_scale = random_mxfp4(inter_size, hidden_size)
        w2_weight_fp4, w2_weight_scale = random_mxfp4(hidden_size, inter_size)
        w3_weight_fp4, w3_weight_scale = random_mxfp4(inter_size, hidden_size)

        # The E8M0 block scales go under both names the MXFP4 loaders read: weight_scale
        # (MXFP4WeightFusedMoEMethod and its CUTLASS/TRTLLM-Gen subclasses) and
        # weight_scale_inv (WFP4A16FusedMoEMethod.load_quant_scales,
        # moe/fused_moe/quantization.py:2279-2345, the Hopper CUTLASS W4A16 MXFP4
        # leaf Kimi-K3 builds). Both entries hold the same tensor.
        weights = {}
        for expert_id in range(local_num_experts):
            weights[f"{expert_id}.w1.weight"] = w1_weight_fp4[expert_id]
            weights[f"{expert_id}.w2.weight"] = w2_weight_fp4[expert_id]
            weights[f"{expert_id}.w3.weight"] = w3_weight_fp4[expert_id]
            for proj, scale in (("w1", w1_weight_scale), ("w2", w2_weight_scale), ("w3", w3_weight_scale)):
                weights[f"{expert_id}.{proj}.weight_scale"] = scale[expert_id]
                weights[f"{expert_id}.{proj}.weight_scale_inv"] = scale[expert_id]
        if layer.bias:
            # Only GPT-OSS checkpoints carry expert biases (bias=True,
            # models/modeling_gpt_oss.py:196).
            w1_bias = torch.randn((local_num_experts, inter_size), dtype=dtype, device=device)
            w2_bias = torch.randn((local_num_experts, hidden_size), dtype=dtype, device=device)
            w3_bias = torch.randn((local_num_experts, inter_size), dtype=dtype, device=device)
            for expert_id in range(local_num_experts):
                weights[f"{expert_id}.w1.bias"] = w1_bias[expert_id]
                weights[f"{expert_id}.w2.bias"] = w2_bias[expert_id]
                weights[f"{expert_id}.w3.bias"] = w3_bias[expert_id]
        moe.load_weights([weights])
        del weights
    _post_load_weights(moe)

    torch.cuda.synchronize()
    hidden_states_max_tokens = torch.randn([max_num_tokens, hidden_size]).bfloat16().to(torch.device(device))
    logits_max_tokens = balanced_logits(max_num_tokens, num_experts, topk).to(layer.router_logits_dtype)

    # Serving's warmup autotune (warmup.py:1099-1104, 1106-1159): one forward of
    # max_num_tokens, then one generation token, under autotune(). A pass tunes
    # every bucket the op's TuningConfig derives, not only the pass's own M: the
    # CUTLASS MoE runner derives power-of-2 buckets from tune_max_num_tokens
    # (custom_ops/torch_custom_ops.py:97-102, 327; autotuner.py:1746-1769), and
    # TRTLLM-Gen a fixed power-of-2 ladder to 8192 plus max_num_tokens above it
    # (trtllm_gen_custom_ops.py:549-571). The first forward is also the dry run
    # that moves lazy initialization and JIT out of the timed region; larger
    # token counts get theirs in benchmark_with_power's eager warm-ups.
    tuning_start = time.perf_counter()
    with (
        torch.inference_mode(),
        autotune(
            cache_path=_tuning_cache_path(
                sm_version, moe_type, hidden_size, inter_size, topk, num_experts, moe_tp_size, moe_ep_size
            )
        ),
    ):
        for warmup_tokens in (max_num_tokens, 1):
            moe.forward(
                hidden_states_max_tokens[:warmup_tokens],
                logits_max_tokens[:warmup_tokens],
                do_finalize=not min_latency_mode,
            )
    torch.cuda.synchronize()
    print(f"trtllm.moe: autotune max_num_tokens={max_num_tokens}: {time.perf_counter() - tuning_start:.1f} s")

    del hidden_states_max_tokens, logits_max_tokens

    for point, num_tokens in enumerate(num_tokens_lists):
        hidden_states = torch.randn([num_tokens, hidden_size]).bfloat16().to(torch.device(device))
        num_iter = 5 if distributed == "power_law" else 1
        if distributed == "power_law":
            actual_logits_list = [
                power_law_logits_v3(num_tokens, num_experts, topk, moe_ep_size, power_law_alpha)
                .to(layer.router_logits_dtype)
                .to(device)
                for _ in range(num_iter)
            ]
        elif distributed == "balanced":
            actual_logits = balanced_logits(num_tokens, num_experts, topk).to(
                device=torch.device(device), dtype=layer.router_logits_dtype
            )
        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")

        def run_forward_pass():
            """Execute one forward pass through MOE, handling both power_law and balanced modes."""
            if distributed == "power_law":
                for logits in actual_logits_list:
                    moe.forward(hidden_states, logits, do_finalize=not min_latency_mode)  # noqa: F821
            else:
                moe.forward(hidden_states, actual_logits, do_finalize=not min_latency_mode)  # noqa: F821

        num_warmups = 1 if distributed == "power_law" else 3
        num_runs = 1 if distributed == "power_law" else 6

        # CUDA-graph timing is mandatory: a failed capture raises
        # (benchmark_with_power default allow_graph_fail=False) and the case
        # fails classified. The moe row has no column that could record an
        # eager-timed measurement, and the former eager fallback
        # shipped the Python launch-loop floor as kernel latency.
        with benchmark_with_power(
            device=device,
            kernel_func=run_forward_pass,
            num_warmups=num_warmups,
            num_runs=num_runs,
            repeat_n=1,
            min_warmup_seconds=FIRST_POINT_WARMUP_SECONDS if point == 0 else 0.0,
        ) as results:
            # Calculate per-iteration latency (accounting for internal iterations)
            latency = results["latency_ms"] / num_iter
            power_stats = results["power_stats"]

        log_perf(
            item_list=[
                {
                    "moe_dtype": moe_type,
                    "num_tokens": num_tokens,
                    "hidden_size": hidden_size,
                    "inter_size": inter_size,
                    "topk": topk,
                    "num_experts": num_experts,
                    "moe_tp_size": moe_tp_size,
                    "moe_ep_size": moe_ep_size,
                    "distribution": "power_law_" + str(power_law_alpha) if distributed == "power_law" else distributed,
                    "latency": latency,
                }
            ],
            framework="TRTLLM",
            version=tensorrt_llm.__version__,
            device_name=torch.cuda.get_device_name(device),
            op_name="moe",
            kernel_source=kernel_source,
            perf_filename=perf_filename,
            power_stats=power_stats,
        )
        if distributed == "power_law":
            del actual_logits_list
        else:
            del actual_logits
        del hidden_states


# ── Worker recycling ─────────────────────────────────────────────────────────
# AIConfigurator's collector (40bc989a9) exited the worker after EVERY case so the
# OS reclaimed all GPU memory. collect.py spawns its workers, so each recycle
# re-imports tensorrt_llm and re-creates the CUDA context (10-15 s); on the b300
# base run (job 471159002) that was most of a 14-20 GPU-s case of 27 token
# points. The policy is now:
#   TRTLLM_MOE_RESTART_EVERY=N    recycle after N completed cases on this worker
#                                 (default 16; 1 = the legacy per-case recycle;
#                                 0 = never recycle on the count)
#   TRTLLM_MOE_RESTART_LEAK_MB=M  recycle at once when more than M MiB are still
#                                 allocated after the case's own cleanup (default
#                                 1024; 0 disables the guard) — the residue the
#                                 per-case exit used to hide
#   TRTLLM_MOE_RESTART_WORKER=0   legacy switch: never recycle (overrides both)
# A recycle is always announced on one greppable line so a run log shows how
# often, and why, workers were replaced.
_cases_since_worker_start = 0


def _worker_restart_policy() -> tuple[int, int] | None:
    """(cases per worker, leak guard MiB), or None when recycling is disabled."""
    if os.getenv("TRTLLM_MOE_RESTART_WORKER", "1") == "0":
        return None
    every = int(os.getenv("TRTLLM_MOE_RESTART_EVERY", "16"))
    leak_mb = int(os.getenv("TRTLLM_MOE_RESTART_LEAK_MB", "1024"))
    if every < 0 or leak_mb < 0:
        raise ValueError(
            f"TRTLLM_MOE_RESTART_EVERY and TRTLLM_MOE_RESTART_LEAK_MB must be >= 0; got {every} and {leak_mb}"
        )
    return every, leak_mb


def _release_trtllm_memory_buffers() -> int:
    """Drop TRT-LLM's workspace caches between cases; returns MiB released from the buffer pool.

    b200 probe (job 514855922, TRTLLM_MOE_LEAK_DEBUG=1): everything still
    allocated after a case was `BufferBlock` tensors of
    tensorrt_llm/_torch/memory_buffer_utils.py — the module-level ``_buffer = Buffers()``
    pool (get_memory_buffers, memory_buffer_utils.py:125-130; DeepGemmFusedMoE.buffers,
    moe/fused_moe/fused_moe_deepgemm.py:853) the fused-MoE kernels draw their
    workspaces from (2.7-86 GiB, the two largest blocks in the 2:1 shape of
    w3_w1 / w2). It is a reuse pool, not a leak: the next case replaces it, so
    memory did not grow across 117 cases. Serving keeps it alive for the model's
    lifetime; between collector cases nothing is live, so every block (reserved
    ones included — ``clear_memory_buffers`` keeps those) is released here and
    the leak guard measures real residue again.

    The CUTLASS MoE runners are cached per dtype combination across layers
    (MoERunner.runner_dict, custom_ops/torch_custom_ops.py:82-95, 154-161) and each
    keeps a per-stream scratch tensor sized for the largest token count it has run
    (FusedMoeRunner::getWorkspaceInfo, cpp/tensorrt_llm/thop/moe/cutlass/moeOp.cpp:1448-1468),
    so a 65536-token case leaves it allocated into the next case.
    MoERunner.clear_all_workspaces releases it (moeOp.cpp:515-522); the next run
    re-allocates it outside the timed region (the warmup forward runs first).
    """
    MoERunner.clear_all_workspaces()
    pool = get_memory_buffers()
    released = 0
    for blocks in list(pool.buffers.values()):
        for block in blocks:
            tensor = getattr(block, "buffer", None)
            if tensor is not None:
                released += tensor.numel() * tensor.element_size()
            block.buffer = None
    pool.buffers.clear()
    return released // 2**20


def _print_live_cuda_tensors(device, top: int = 8) -> None:
    """TRTLLM_MOE_LEAK_DEBUG=1: name what still pins CUDA memory after a case.

    b200 probe (jobs 514778748/514778867): 2.7-86 GiB stayed allocated after
    every DeepSeek-V3 fp8_block case, so the leak guard recycled the worker each time and
    the policy bought nothing. Lists the largest live CUDA tensors with the types of the
    objects that reference them (gc view; a traceback-free best effort).
    """
    sizes = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda and obj.device == device:
                sizes.append((obj.numel() * obj.element_size(), obj))
        except Exception:  # gc.get_objects can hand back half-built objects
            continue
    sizes.sort(key=lambda item: -item[0])
    total = sum(size for size, _ in sizes)
    print(f"trtllm.moe: leak debug: {len(sizes)} live CUDA tensors, {total / 2**20:.0f} MiB; largest:")
    for size, tensor in sizes[:top]:
        referrers = [type(r).__name__ for r in gc.get_referrers(tensor) if r is not sizes][:4]
        print(
            f"  {size / 2**20:8.0f} MiB {tuple(tensor.shape)} {tensor.dtype} "
            f"param={isinstance(tensor, torch.nn.Parameter)} referrers={referrers}"
        )


def _recycle_worker_if_due(device) -> None:
    """Exit this worker with EXIT_CODE_RESTART when the recycling policy says so.

    Runs after the case's rows are persisted; collect.py marks the task done and
    starts a fresh worker on the same device (worker(): SystemExit handling).
    """
    global _cases_since_worker_start
    policy = _worker_restart_policy()
    if policy is None:
        return
    every, leak_mb = policy
    _cases_since_worker_start += 1
    workspace_mb = _release_trtllm_memory_buffers()
    gc_collect()
    allocated_mb = torch.cuda.memory_allocated(device) / 2**20
    reserved_mb = torch.cuda.memory_reserved(device) / 2**20
    print(
        f"trtllm.moe: after case {_cases_since_worker_start} on this worker: "
        f"{allocated_mb:.0f} MiB allocated, {reserved_mb:.0f} MiB reserved "
        f"(released {workspace_mb} MiB of TRT-LLM workspace buffers)"
    )
    if os.getenv("TRTLLM_MOE_LEAK_DEBUG", "0") == "1" and allocated_mb > 256:
        _print_live_cuda_tensors(device)
    due_count = every > 0 and _cases_since_worker_start >= every
    due_leak = leak_mb > 0 and allocated_mb > leak_mb
    if not (due_count or due_leak):
        return
    reason = "leak guard" if due_leak else f"every {every}"
    print(
        f"trtllm.moe: recycling worker after {_cases_since_worker_start} case(s) "
        f"({allocated_mb:.0f} MiB still allocated, {reason})"
    )
    sys.exit(EXIT_CODE_RESTART)


def run_moe_torch(
    moe_type,
    num_tokens_lists,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    min_latency_mode,
    model_name,
    distributed="power_law",
    power_law_alpha=0.0,
    *,
    perf_filename,
    device="cuda:0",
):
    """Run one MoE case, then recycle the worker when the policy above is due."""
    _run_moe_case(
        moe_type,
        num_tokens_lists,
        hidden_size,
        inter_size,
        topk,
        num_experts,
        moe_tp_size,
        moe_ep_size,
        min_latency_mode,
        model_name,
        distributed=distributed,
        power_law_alpha=power_law_alpha,
        perf_filename=perf_filename,
        device=device,
    )
    _recycle_worker_if_due(torch.device(device))


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_moe_test_cases()
    for test_case in test_cases:
        run_moe_torch(*test_case, perf_filename=PerfFile.MOE)
