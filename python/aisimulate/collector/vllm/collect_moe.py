# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM production-path MoE collector.

Shared MoE cases come from YAML. Every quantization mode is built through
``FusedMoEFactory`` so vLLM owns routing, TP/EP sharding, weight
post-processing, and backend selection exactly as it does for model execution.
The router gate GEMM, shared experts and latent projections are separate ops
and are not part of these rows.
"""

# Audited against vLLM v0.32.0 (from v0.31.0); line numbers below are at that
# tag. Earlier audits are git history.
# Construction: FusedMoEFactory (fused_moe/layer.py:88-141) keeps every kwarg
#   passed here; v0.32.0 only adds the defaulted shared_expert_prefix. New in
#   v0.32.0, RoutedExperts rejects a non-gated activation whose ckpt_names still
#   name an up projection (routed_experts.py:118-124), so Nemotron-H's relu2
#   layer passes ckpt_names=("up_proj", "down_proj", None) as serving does
#   (model_executor/models/nemotron_h.py:238). Probed attributes:
#   routed_experts/router/moe_config (runner/moe_runner.py:265-272),
#   quant_method (routed_experts.py:159), moe_kernel.fused_experts
#   (modular_kernel.py:1664), FallbackExperts._select_experts_impl
#   (experts/fallback.py:126), init_workspace_manager (v1/worker/workspace.py:
#   307, new param defaulted), moe_layer_index / set_forward_context
#   (forward_context.py:195, :278). Backend attributes: unquantized_backend
#   (unquantized_fused_moe_method.py:54), fp8_backend (fp8.py:518,
#   modelopt.py:497), nvfp4_backend (modelopt.py:870,
#   compressed_tensors_moe_w4a4_nvfp4.py:60), mxfp4_backend (mxfp4.py:158,
#   :576), wna16_backend (compressed_tensors_moe_wna16.py:102).
# Kernel selection is the framework's own oracle with moe_backend "auto". What
#   v0.32.0 changes underneath an unchanged call site:
#   - prioritize_humming (quantization/utils/humming/priority.py:11,26-45) moves
#     Humming directly ahead of Marlin on SM90 only. int4_wo on SM90 therefore
#     leaves MarlinExperts for a Humming experts class (oracle/int_wna16.py:
#     126-143, :72-83). No other SM90 lane reaches an oracle's Marlin entry;
#     Kimi-K3's compressed-tensors method binds MarlinExperts directly.
#   - TrtLlmBf16ExpertsMonolithic adds the MiniMax2 routing method (sigmoid,
#     bias, ungrouped; experts/trtllm_bf16_moe.py:246-259): bf16 Step-3.7-Flash
#     can move to the monolithic kernel on SM100/SM103. MiniMax-M3 bf16 runs
#     swigluoai, which demotes FlashInfer (oracle/unquantized.py:88-94).
#   - gpt-oss mxfp4: OAI Triton now runs on SM12x but is ordered after Marlin
#     there (oracle/mxfp4.py:389-422, experts/gpt_oss_triton_kernels_moe.py:
#     41-53), so SM120 stays on Marlin; SM90 Triton, SM100/SM103 FlashInfer
#     TRT-LLM BF16 as before.
#   - Kimi-K3 compressed-tensors mxfp4: CompressedTensorsW4A4Mxfp4MoEMethod
#     picks CutlassExpertsMxfp4 (W4A4) wherever the CUDA quant helper exists, SM
#     100-119 (compressed_tensors_moe_w4a4_mxfp4.py:51-78,
#     csrc/quantization/fp4/mxfp4_experts_quant.cu:383-389), and MarlinExperts
#     elsewhere. Both run SiTU through apply_moe_activation
#     (fused_moe/activation.py:123-157), and CUTLASS fp4/mxfp4 experts now
#     accept EP through an expert-map remap (experts/cutlass_moe.py:1057-1067).
#   - DeepGEMM fp8_block experts use the psum layout on SM90/SM100
#     (experts/deep_gemm_moe.py:361-373) and DeepGEMM moves e1f418c2 ->
#     1e1842a8 (cmake/external_projects/deepgemm.cmake:37), which replaces
#     the SM120 device layer. FlashInfer 0.7.0.post1 is unchanged;
#     nvidia-cutlass-dsl 4.7.1 -> 4.8.0 and humming-kernels 0.1.16 -> 0.1.17
#     (requirements/cuda.txt).
#   The fp8, fp8_block, nvfp4 and DeepSeek-V4 mxfp4 x mxfp8 priority lists are
#   otherwise unchanged (oracle/fp8.py:77-143, oracle/nvfp4.py:191-213,
#   oracle/mxfp4.py:425-449,742-813). kernel_source records whatever ran.
# Timed closure: routing (router; GroupedTopk.forward_cuda is forward_native,
#   fused_moe/router/grouped_topk_router.py:210-218), activation quant through
#   direct _C ops (fused_moe/utils.py:323-345, quantization/utils/fp8_utils.py:
#   703-716), permute, experts, finalize and the routed output scale: the
#   per-forward work of MoERunner. Weight re-layout and scale swizzles run in
#   process_weights_after_loading, outside timing. The MoE path calls no
#   QuantFP8 custom op, so the VllmConfig quant_config / +quant_fp8 setting
#   (config/vllm.py:1733-1748) does not change what these rows time.
__compat__ = "vllm==0.32.0"

import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from collector.case_generator import get_moe_quantization_module_config
from collector.helper import (
    FIRST_POINT_WARMUP_SECONDS,
    MOE_WEIGHT_MATRIX_SUFFIXES,
    balanced_logits,
    benchmark_with_power,
    fill_dense_weight_,
    flashinfer_autotuned,
    flashinfer_tuning_bucket,
    get_sm_version,
    log_perf,
    power_law_logits_v3,
)
from collector.op_plans import plan_cases
from vllm.version import __version__ as vllm_version

aic_debug = int(os.getenv("aic_moe_debug", "0"))  # noqa: SIM112
_MODEL_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "src/aisimulate_core/model_configs"


def _load_model_moe_config(model_name: str) -> dict:
    """Load the checked-in HF config used to derive vLLM's FusedMoE args."""
    config_path = _MODEL_CONFIG_ROOT / f"{model_name.replace('/', '--')}_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing packaged model config for vLLM MoE case {model_name!r}: {config_path}")

    with config_path.open() as config_file:
        config = json.load(config_file)
    return config.get("text_config", config)


def _load_checkpoint_quantization_config(model_name: str) -> dict:
    """Return the checked-in checkpoint's top-level ``quantization_config``.

    ``_load_model_moe_config`` returns the text sub-config, whose
    ``quantization_config`` is null for multimodal wrappers such as
    GLM-5.3-Flash; vLLM reads the top-level one (weight_utils.py:224).
    """
    config_path = _MODEL_CONFIG_ROOT / f"{model_name.replace('/', '--')}_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing packaged model config for vLLM MoE case {model_name!r}: {config_path}")
    with config_path.open() as config_file:
        return json.load(config_file).get("quantization_config") or {}


def _uses_modelopt_nvfp4_checkpoint(model_name: str) -> bool:
    """True when serving loads this model's NVFP4 experts via ModelOptNvFp4Config.

    Scoped to GLM-5.3-Flash so every other nvfp4 model keeps its existing
    CompressedTensors construction (and kernel_source label). vLLM resolves
    quant_method "modelopt" + quant_algo "NVFP4" to modelopt_fp4
    (config/model.py:1349-1388, quantization/modelopt.py:756-763).
    """
    if _load_model_moe_config(model_name).get("model_type") != "glm5_next_text":
        return False
    checkpoint_qc = _load_checkpoint_quantization_config(model_name)
    return (
        str(checkpoint_qc.get("quant_method", "")).lower() == "modelopt"
        and str(checkpoint_qc.get("quant_algo", "")).upper() == "NVFP4"
    )


def _resolve_moe_runtime_config(model_name: str, module_config: dict) -> dict:
    """Resolve the FusedMoEFactory arguments the model's vLLM code passes."""
    model_config = _load_model_moe_config(model_name)
    model_type = str(model_config.get("model_type", ""))
    activation = str(
        module_config.get("activation")
        or model_config.get("hidden_act")
        or model_config.get("hidden_activation")
        or "silu"
    )

    use_grouped_topk = model_type in {
        "deepseek_v3",
        "kimi_k2",
        # Kimi-K3 (kimi_linear): config declares use_grouped_topk=true with
        # one expert group (num_expert_group=1, topk_group=1, noaux_tc), and
        # KimiMoE passes the flag through (models/kimi_k3/nvidia/model.py:
        # 727-729).
        "kimi_linear",
        "mimo_v2_flash",
        "glm_moe_dsa",
        # GLM-5.3-Flash: Glm5NextMoE hard-codes use_grouped_topk=True with
        # num_expert_group=config.n_group (1) and topk_group
        # (models/glm5next/common/model.py:297-299).
        "glm5_next_text",
        "nemotron_h",
    }
    declares_grouped_routing = (
        model_config.get("n_group") is not None
        or model_config.get("topk_group") is not None
        or model_config.get("topk_method") == "noaux_tc"
    )
    # DeepSeek V4 declares topk_method=noaux_tc but routes UNGROUPED: its
    # FusedMoEFactory call passes no expert grouping, only the noaux_tc
    # e_score_correction_bias with sqrtsoftplus scoring and float32 router
    # logits (models/deepseek_v4/nvidia/model.py:873-896). The bias/scoring
    # fields below already carry that. Both V4 model_types share that model
    # code: "deepseek_ref" (the sgl-project FP8 requants) and "deepseek_v4"
    # (the native expert_dtype=fp4 checkpoints).
    ungrouped_noaux_tc = model_type in ("deepseek_ref", "deepseek_v4")
    if declares_grouped_routing and not use_grouped_topk and not ungrouped_noaux_tc:
        raise ValueError(
            f"vLLM MoE model {model_name!r} (model_type={model_type!r}) declares grouped/noaux_tc "
            "routing fields but is not a recognized grouped-topk model type; verify how vLLM "
            "routes this model and extend the mapping instead of silently benchmarking non-grouped routing"
        )
    # Some upstream configs spell the routing contract differently (Step3p5/3p7
    # use "use_moe_router_bias"/"moe_router_scaling_factor"). Read both spellings:
    # a curated config that drops the vendor key silently benchmarks unbiased
    # routing while vLLM serves biased routing, which the guard above cannot see
    # because the vendor key is not one of the grouped/noaux_tc fields it checks.
    declared_routing_bias = model_config.get("use_routing_bias")
    if declared_routing_bias is None:
        declared_routing_bias = model_config.get("use_moe_router_bias")
    use_routing_bias = (
        model_config.get("topk_method") == "noaux_tc"
        or bool(declared_routing_bias)
        or model_type in {"mimo_v2_flash", "glm_moe_dsa", "nemotron_h"}
    )
    # Resolve on declaration, not truthiness: a canonical routed_scaling_factor
    # of 0.0 is a real value (it zeroes the routed contribution) and must not
    # fall through to the vendor key or the 1.0 default.
    declared_routed_scale = model_config.get("routed_scaling_factor")
    if declared_routed_scale is None:
        declared_routed_scale = model_config.get("moe_router_scaling_factor")

    scoring_func = str(model_config.get("scoring_func") or "softmax")
    renormalize = bool(model_config.get("norm_topk_prob", True))
    min_moe_intermediate_per_partition = None
    if model_type == "nemotron_h":
        scoring_func = "sigmoid"
    elif model_type == "kimi_linear":
        # KimiMoE routes with config.moe_router_activation_func (sigmoid for
        # Kimi-K3; the config has no scoring_func) and renormalizes with
        # config.moe_renormalize (models/kimi_k3/nvidia/model.py:725,731).
        scoring_func = str(model_config["moe_router_activation_func"])
        renormalize = bool(model_config.get("moe_renormalize", True))
        # Pure-TP KimiMoE pads each rank's expert intermediate up to this floor
        # (models/kimi_k3/nvidia/model.py:595-604, getattr default 256).
        min_moe_intermediate_per_partition = int(model_config.get("min_moe_intermediate_per_partition") or 256)

    custom_routing = None
    apply_router_weight_on_input = False
    if model_type == "llama4_text":
        custom_routing = "llama4"
        apply_router_weight_on_input = True
        renormalize = False
    elif model_type == "gemma4_text":
        custom_routing = "gemma4"
        activation = "gelu_tanh"
    elif model_type == "nemotron_h":
        activation = str(model_config["mlp_hidden_act"])
        activation = {
            "gelu_pytorch_tanh": "gelu_tanh",
        }.get(activation, activation)
        activation = f"{activation}_no_mul"

    if use_grouped_topk:
        # DeepSeek-family configs spell the group count `n_group`; Kimi-K3
        # (kimi_linear) spells it `num_expert_group`, which is the attribute
        # KimiMoE reads (models/kimi_k3/nvidia/model.py:728).
        raw_n_group = model_config.get("n_group", model_config.get("num_expert_group"))
        missing_fields = [
            field
            for field, value in (("n_group", raw_n_group), ("topk_group", model_config.get("topk_group")))
            if value is None
        ]
        if missing_fields:
            raise ValueError(f"vLLM grouped-topk model {model_name!r} is missing {', '.join(missing_fields)}")
        try:
            num_expert_group = int(raw_n_group)
            topk_group = int(model_config["topk_group"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"vLLM grouped-topk model {model_name!r} requires integer group fields") from error
        if num_expert_group <= 0 or topk_group <= 0 or topk_group > num_expert_group:
            raise ValueError(
                f"vLLM grouped-topk model {model_name!r} has invalid "
                f"n_group={num_expert_group}, topk_group={topk_group}"
            )
    else:
        num_expert_group = None
        topk_group = None

    return {
        "renormalize": renormalize,
        "scoring_func": scoring_func,
        "activation": activation,
        # SiTU (Kimi-K3) is parameterized: FusedMoEFactory takes the betas
        # (fused_moe/layer.py:111-112) and KimiMoE passes the config values
        # (models/kimi_k3/nvidia/model.py:605-611,723-724).
        "activation_situ_beta": (model_config.get("activation_situ_beta") if activation == "situ" else None),
        "activation_situ_linear_beta": (
            model_config.get("activation_situ_linear_beta") if activation == "situ" else None
        ),
        "routed_scaling_factor": float(declared_routed_scale if declared_routed_scale is not None else 1.0),
        # Glm5NextMoE also passes config.swiglu_limit to FusedMoEFactory
        # (models/glm5next/common/model.py:266,314); on SM10x it filters the
        # NVFP4 backends (oracle/nvfp4.py:215-230) and gates the TRT-LLM FP8
        # experts (trtllm_fp8_moe.py:86-112).
        "swiglu_limit": (
            model_config.get("swiglu_limit")
            if model_type in ("deepseek_v4", "deepseek_ref", "glm5_next_text")
            else None
        ),
        "use_grouped_topk": use_grouped_topk,
        "num_expert_group": num_expert_group,
        "topk_group": topk_group,
        "apply_routed_scale_to_output": model_type in {"deepseek_v3", "kimi_k2", "glm_moe_dsa", "nemotron_h"},
        "use_routing_bias": use_routing_bias,
        "router_logits_float32": use_routing_bias or scoring_func in {"sigmoid", "sqrtsoftplus"},
        "custom_routing": custom_routing,
        "apply_router_weight_on_input": apply_router_weight_on_input,
        "has_bias": bool(module_config.get("has_bias", False)),
        "min_moe_intermediate_per_partition": min_moe_intermediate_per_partition,
    }


def _moe_intermediate_size(inter_size: int, moe_tp_size: int, moe_ep_size: int, min_per_partition: int | None) -> int:
    """The intermediate size the model passes to FusedMoEFactory for this layout.

    Kimi-K3 pads a pure-TP shard below ``min_per_partition`` to that floor
    (models/kimi_k3/nvidia/model.py:595-604); every other model passes its
    config value unchanged.
    """
    if min_per_partition is None or moe_tp_size <= 1 or moe_ep_size > 1:
        return inter_size
    if inter_size // moe_tp_size < min_per_partition:
        return min_per_partition * moe_tp_size
    return inter_size


# FlashInfer's MoE ops map a runtime token count to a tuned bucket no larger than
# the tune_max_num_tokens their caller hands them (helper.flashinfer_tuning_bucket),
# so any m at or above it runs that bucket's tactic. What each experts class hands
# FlashInfer at v0.32.0:
# - nothing, so FlashInfer's default 8192 (flashinfer 0.7.0.post1 fused_moe/core.py:
#   1290 cutlass_fused_moe, :4916 trtllm_bf16_routed_moe): FlashInferExperts
#   (experts/flashinfer_cutlass_moe.py:365-389), TrtLlmBf16ExpertsModular
#   (experts/trtllm_bf16_moe.py:209-229); TrtLlmMxint4ExpertsMonolithic passes
#   8192 itself (quantization/utils/flashinfer_mxint4_moe.py:267);
# - fi_moe_largest_bucket(moe_config) = max(max_num_tokens * dp_size, 8192)
#   (fused_moe/utils.py:557-572): TrtLlmBf16ExpertsMonolithic (trtllm_bf16_moe.py:
#   303), TrtLlmFp8ExpertsModular / Monolithic (trtllm_fp8_moe.py:332, 516, 575),
#   TrtLlmMxfp4ExpertsMonolithic / Modular (trtllm_mxfp4_moe.py:268, 402),
#   TrtLlmNvFp4ExpertsMonolithic (trtllm_nvfp4_moe.py:583);
# - that bound capped at its own launch chunk: TrtLlmNvFp4ExpertsModular
#   (trtllm_nvfp4_moe.py:385-387);
# - no cap: FlashInferCuteDSLExperts, whose runner maps with
#   map_to_hybrid_bucket_uncapped (flashinfer fused_moe/cute_dsl/tuner.py:545-556).
_FLASHINFER_DEFAULT_TUNE_MAX_NUM_TOKENS = 8192
_FI_DEFAULT_TUNE_MAX_EXPERTS = frozenset(
    {"FlashInferExperts", "TrtLlmBf16ExpertsModular", "TrtLlmMxint4ExpertsMonolithic"}
)
_FI_LARGEST_BUCKET_EXPERTS = frozenset(
    {
        "TrtLlmBf16ExpertsMonolithic",
        "TrtLlmFp8ExpertsModular",
        "TrtLlmFp8ExpertsMonolithic",
        "TrtLlmMxfp4ExpertsMonolithic",
        "TrtLlmMxfp4ExpertsModular",
        "TrtLlmNvFp4ExpertsMonolithic",
    }
)
_FI_UNCAPPED_EXPERTS = frozenset({"FlashInferCuteDSLExperts"})


def _flashinfer_moe_tune_max_num_tokens(leaf_experts) -> int | None:
    """The tune_max_num_tokens ``leaf_experts`` hands its FlashInfer MoE op.

    ``None`` for an experts class whose FlashInfer op maps uncapped and for
    kernels FlashInfer's tuner does not own (Triton, Marlin, Humming, DeepGEMM,
    vLLM CUTLASS). A FlashInfer or TRT-LLM-gen experts class not listed above
    raises: its bound was never traced.
    """
    name = type(leaf_experts).__name__
    if name in _FI_DEFAULT_TUNE_MAX_EXPERTS:
        return _FLASHINFER_DEFAULT_TUNE_MAX_NUM_TOKENS
    if name in _FI_LARGEST_BUCKET_EXPERTS or name == "TrtLlmNvFp4ExpertsModular":
        from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket

        largest = fi_moe_largest_bucket(leaf_experts.moe_config)
        if name == "TrtLlmNvFp4ExpertsModular":
            return min(largest, leaf_experts._get_chunk_size())
        return largest
    if name.startswith(("FlashInfer", "TrtLlm")) and name not in _FI_UNCAPPED_EXPERTS:
        raise RuntimeError(
            f"vLLM MoE experts {name} run a FlashInfer op whose tune_max_num_tokens the collector has not "
            "traced; add it to collector/vllm/collect_moe.py before timing it"
        )
    return None


def get_moe_test_cases():
    return plan_cases("moe", "vllm", get_sm_version())


def run_moe_torch(
    moe_type,
    num_tokens_lists,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    distributed="power_law",
    power_law_alpha=0.0,
    *,
    perf_filename,
    device="cuda:0",
):
    """Benchmark the vLLM model-execution MoE path."""
    from collector.vllm.utils import setup_distributed
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.fallback import FallbackExperts
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
    from vllm.platforms import current_platform
    from vllm.utils.flashinfer import has_flashinfer
    from vllm.v1.worker.workspace import init_workspace_manager

    if moe_tp_size > 1 and moe_ep_size > 1:
        raise ValueError("vLLM MoE collector does not combine logical TP and EP")

    setup_distributed(device)
    torch.cuda.set_device(device)
    init_workspace_manager(torch.device(device))
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)

    module_config = get_moe_quantization_module_config("vllm", moe_type, model_name=model_name)
    runtime_config = _resolve_moe_runtime_config(model_name, module_config)
    if runtime_config["use_grouped_topk"]:
        num_expert_group = runtime_config["num_expert_group"]
        topk_group = runtime_config["topk_group"]
        if num_experts <= num_expert_group or num_experts % num_expert_group != 0:
            raise ValueError(
                f"vLLM grouped-topk model {model_name!r} cannot divide {num_experts} experts "
                f"into {num_expert_group} groups"
            )
        experts_per_group = num_experts // num_expert_group
        if topk > topk_group * experts_per_group:
            raise ValueError(
                f"vLLM grouped-topk model {model_name!r} requests topk={topk}, but "
                f"topk_group={topk_group} exposes only {topk_group * experts_per_group} experts"
            )
    e_score_correction_bias = (
        torch.zeros(num_experts, dtype=torch.float32, device=device) if runtime_config["use_routing_bias"] else None
    )
    router_logits_dtype = torch.float32 if runtime_config["router_logits_float32"] else None
    custom_routing_function = None
    if runtime_config["custom_routing"] == "llama4":
        from vllm.model_executor.models.llama4 import Llama4MoE

        # Llama4MoE passes this routing with apply_router_weight_on_input and
        # renormalize=False (model_executor/models/llama4.py:75,137-140).
        custom_routing_function = Llama4MoE.custom_routing_function
    elif runtime_config["custom_routing"] == "gemma4":
        from vllm.model_executor.models.gemma4 import gemma4_fused_routing_kernel_triton

        per_expert_scale = torch.ones(num_experts, device=device)

        def gemma4_routing_function(hidden_states, gating_output, topk, renormalize):
            return gemma4_fused_routing_kernel_triton(gating_output, topk, per_expert_scale)

        custom_routing_function = gemma4_routing_function

    if moe_type == "bfloat16":
        quant_config = None
    elif moe_type == "fp8":
        # Per-tensor "fp8" rows are anchored on ModelOpt static checkpoints
        # (e.g. Nemotron Ultra FP8). Serving loads those via
        # ModelOptFp8MoEMethod: static/static quant keys
        # (quantization/modelopt.py:477,497-501) and non-gated-aware weight
        # sizing (w13_num_shards = 2 if is_act_and_mul else 1, modelopt.py:523).
        # Fp8Config/Fp8MoEMethod would diverge on both: activation_scheme
        # "dynamic" resolves kFp8DynamicTensorSym and selects a different
        # backend than serving (fp8.py:464,518), and its create_weights sizes a
        # gated 2x w13, which breaks non-gated models like Nemotron.
        quant_config = ModelOptFp8Config(
            quant_method="FP8",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=None,
            exclude_modules=[],
        )
    elif moe_type == "fp8_block":
        # Block-FP8 serving (DeepSeek-style checkpoints) is per-128-block
        # weights with dynamic per-group activations; Fp8Config rejects
        # static for block quant (fp8.py:128-133).
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
    elif moe_type == "w4a16_mxfp4":
        checkpoint_qc = _load_model_moe_config(model_name).get("quantization_config") or {}
        is_ct_mxfp4 = checkpoint_qc.get("quant_method") == "compressed-tensors" and "mxfp4" in str(
            checkpoint_qc.get("format", "")
        )
        if is_ct_mxfp4:
            # Kimi-K3-style compressed-tensors mxfp4-pack checkpoints (weights
            # only, input_activations null): vLLM resolves the descriptor to
            # CompressedTensorsW4A4Mxfp4MoEMethod (compressed_tensors_moe.py:
            # 76-81), which runs CutlassExpertsMxfp4 on SM100-SM119 and
            # MarlinExperts elsewhere (compressed_tensors_moe_w4a4_mxfp4.py:
            # 51-78). Constructing the checkpoint's own config (instead of the
            # OCP Mxfp4Config) lets that selection run exactly as serving; OCP
            # mxfp4 checkpoints (gpt-oss family) keep the Mxfp4Config path below.
            quant_config = CompressedTensorsConfig.from_config(checkpoint_qc)
        else:
            # OCP "mxfp4" checkpoints of model_type gpt_oss are OVERRIDDEN by
            # vLLM to the gpt_oss_mxfp4 method (GptOssMxfp4Config.
            # override_quantization_method, quantization/mxfp4.py:116-150).
            # Its GptOssMxfp4MoEMethod selects through the gpt-oss priority
            # list (fused_moe/oracle/mxfp4.py:389-422, mxfp4.py:158), while
            # the generic Mxfp4Config resolves to Mxfp4MoEMethod's list
            # (oracle/mxfp4.py:425-449). Mirror the framework's own override.
            model_cfg = _load_model_moe_config(model_name)
            quant_config = Mxfp4Config()
            if (
                checkpoint_qc.get("quant_method") in ("mxfp4", "gpt_oss_mxfp4")
                and model_cfg.get("model_type") == "gpt_oss"
            ):
                from vllm.model_executor.layers.quantization.mxfp4 import GptOssMxfp4Config

                quant_config = GptOssMxfp4Config()
    elif moe_type == "w4a8_mxfp4_mxfp8":
        # Native DeepSeek-V4 (expert_dtype=fp4) serving path: vLLM overrides
        # the checkpoint's fp8 quant_method to DeepseekV4FP8Config
        # (models/deepseek_v4/quant_config.py:140-157), which returns
        # Mxfp4MoEMethod for the routed experts (quant_config.py:178-199);
        # backend selection then runs select_deepseek_v4_mxfp4_moe_backend
        # exactly as in serving. Constructor args mirror the checkpoints'
        # quantization_config (fp8-serialized, dynamic, 128x128 blocks).
        from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

        expert_dtype = _load_model_moe_config(model_name).get("expert_dtype")
        if expert_dtype != "fp4":
            raise ValueError(
                f"vLLM MoE case {model_name!r} requests w4a8_mxfp4_mxfp8 but its packaged "
                f"config declares expert_dtype={expert_dtype!r}; only the native fp4-expert "
                "DeepSeek-V4 checkpoints run this path"
            )
        quant_config = DeepseekV4FP8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
    elif moe_type == "nvfp4" and _uses_modelopt_nvfp4_checkpoint(model_name):
        # Serving loads this checkpoint with ModelOptNvFp4Config
        # (ModelOptNvFp4FusedMoE, modelopt.py:836-870). It shares the
        # CT method's backend selector and kernel builder, but constructing
        # the checkpoint's own config keeps the method identity exact.
        from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

        quant_config = ModelOptNvFp4Config.from_config(_load_checkpoint_quantization_config(model_name))
    elif moe_type == "nvfp4":
        nvfp4_args = {
            "num_bits": 4,
            "type": "float",
            "strategy": "tensor_group",
            "group_size": 16,
            "symmetric": True,
            "dynamic": False,
        }
        quant_config = CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "config_groups": {
                    "group_0": {
                        "format": "nvfp4-pack-quantized",
                        "weights": nvfp4_args,
                        "input_activations": nvfp4_args,
                        "output_activations": None,
                        "targets": ["Linear"],
                    }
                },
            }
        )
    elif moe_type == "int4_wo":
        group_size = int(module_config.get("group_size", 128))
        quant_config = CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "format": "pack-quantized",
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "strategy": "group",
                            "group_size": group_size,
                            "symmetric": True,
                            "dynamic": False,
                        },
                        "input_activations": None,
                        "output_activations": None,
                        "targets": ["Linear"],
                    }
                },
            }
        )
    else:
        raise ValueError(f"Unsupported vLLM MoE quantization mode: {moe_type}")

    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_text_config=SimpleNamespace(model_type=""),
        model="collector_dummy",
        quantization_config=None,
    )
    if moe_type == "w4a8_mxfp4_mxfp8":
        # DeepseekV4FP8Config resolves expert_dtype lazily from the current
        # vllm_config's hf_config (models/deepseek_v4/quant_config.py:57-78);
        # mirror the native checkpoint field so the RoutedExperts
        # dispatch resolves to the MXFP4 expert path deterministically.
        vllm_config.model_config.hf_config = SimpleNamespace(
            model_type="deepseek_v4",
            expert_dtype="fp4",
        )
    vllm_config.scheduler_config.max_num_batched_tokens = max(num_tokens_lists)
    vllm_config.parallel_config.enable_expert_parallel = moe_ep_size > 1

    # vLLM represents EP by flattening the requested TP world and then
    # converting it to EP when enable_expert_parallel is set. The collector is
    # one physical rank, but requested logical sizes determine local weights
    # and the rank-0 expert map exactly as in a multi-rank model.
    logical_parallel_size = moe_ep_size if moe_ep_size > 1 else moe_tp_size
    moe_intermediate_size = _moe_intermediate_size(
        inter_size, moe_tp_size, moe_ep_size, runtime_config["min_moe_intermediate_per_partition"]
    )

    # A non-gated activation has no up projection to load: RoutedExperts
    # rejects an up name next to it (routed_experts.py:118-124), and
    # Nemotron-H passes these names (model_executor/models/nemotron_h.py:238).
    ckpt_names = {}
    if not MoEActivation.from_str(runtime_config["activation"]).is_gated:
        ckpt_names = {"ckpt_names": ("up_proj", "down_proj", None)}

    with set_current_vllm_config(vllm_config):
        moe_module = FusedMoEFactory(
            num_experts=num_experts,
            top_k=topk,
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
            params_dtype=torch.bfloat16,
            renormalize=runtime_config["renormalize"],
            use_grouped_topk=runtime_config["use_grouped_topk"],
            num_expert_group=runtime_config["num_expert_group"],
            topk_group=runtime_config["topk_group"],
            quant_config=quant_config,
            tp_size=logical_parallel_size,
            dp_size=1,
            pcp_size=1,
            prefix="collector_moe",
            custom_routing_function=custom_routing_function,
            scoring_func=runtime_config["scoring_func"],
            routed_scaling_factor=runtime_config["routed_scaling_factor"],
            swiglu_limit=runtime_config["swiglu_limit"],
            activation_situ_beta=runtime_config["activation_situ_beta"],
            activation_situ_linear_beta=runtime_config["activation_situ_linear_beta"],
            e_score_correction_bias=e_score_correction_bias,
            apply_router_weight_on_input=runtime_config["apply_router_weight_on_input"],
            has_bias=runtime_config["has_bias"],
            activation=runtime_config["activation"],
            router_logits_dtype=router_logits_dtype,
            apply_routed_scale_to_output=runtime_config["apply_routed_scale_to_output"],
            **ckpt_names,
        )
        if moe_intermediate_size != inter_size:
            # KimiMoE records the unpadded shard after construction
            # (models/kimi_k3/nvidia/model.py:744-757).
            moe_module.moe_config.intermediate_size_per_partition_unpadded = inter_size // moe_tp_size
        moe_module.to(device)
        moe_module.eval()
        moe_module.requires_grad_(False)

        # Populate the checkpoint-native tensors created by the selected
        # quantization method, then let that method convert them to the exact
        # runtime layout and construct the production kernel. Expert weight
        # matrices get dense, checkpoint-like bits (helper.fill_dense_weight_ says
        # why and how); the other tensors keep values every layout accepts.
        routed_experts = moe_module.routed_experts
        with torch.no_grad():
            for name, parameter in routed_experts.named_parameters(recurse=False):
                if "scale" not in name and name.endswith(MOE_WEIGHT_MATRIX_SUFFIXES):
                    fill_dense_weight_(parameter.data)
                elif parameter.dtype == torch.uint8:
                    if "scale" in name:
                        parameter.fill_(127)
                    else:
                        parameter.random_(0, 256)
                elif parameter.dtype == torch.int32:
                    parameter.random_(0, 16)
                elif parameter.is_floating_point():
                    parameter.fill_(1.0 if "scale" in name else 0.01)
                else:
                    parameter.zero_()

        quant_method = routed_experts.quant_method
        quant_method.process_weights_after_loading(routed_experts)

        selected_backend = None
        for backend_attr in (
            "unquantized_backend",
            "fp8_backend",
            "nvfp4_backend",
            "mxfp4_backend",
            "wna16_backend",
        ):
            if hasattr(quant_method, backend_attr):
                selected_backend = getattr(quant_method, backend_attr)
                break
        backend_name = getattr(selected_backend, "value", str(selected_backend))
        if getattr(quant_method, "use_cutlass_mxfp4", False):
            # CompressedTensorsW4A4Mxfp4MoEMethod binds CutlassExpertsMxfp4
            # directly and leaves mxfp4_backend at its MARLIN default
            # (compressed_tensors_moe_w4a4_mxfp4.py:54,61,67-69). Label the
            # oracle backend of that experts class (oracle/mxfp4.py:143,
            # 235-240) so the row does not claim Marlin.
            from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

            backend_name = Mxfp4MoeBackend.CUTLASS_MXFP4_MXFP4.value
        moe_kernel = getattr(quant_method, "moe_kernel", None)
        experts_impl = getattr(moe_kernel, "fused_experts", None)
        wrapper_name = type(experts_impl).__name__ if experts_impl is not None else "direct"
        print(
            "vLLM MoE path:",
            type(quant_method).__name__,
            f"backend={backend_name}",
            f"tp={moe_module.moe_config.tp_size}",
            f"ep={moe_module.moe_config.ep_size}",
            f"routing={moe_module.router.routing_method_type}",
            f"activation={runtime_config['activation']}",
            f"intermediate={moe_intermediate_size}",
        )

        method_name = type(quant_method).__name__.removesuffix("Method")

        # FlashInfer kernels run on the tactic serving's warmup tunes. vLLM
        # autotunes on SM90 and newer when FlashInfer is installed
        # (model_executor/warmup/kernel_warmup.py:261-268;
        # enable_flashinfer_autotune is on from O1, config/vllm.py:265-328,
        # default O2 :427), skipping only fp4_gemm under the cute-dsl NVFP4
        # linear kernel (:295-311), which no MoE op uses. It profiles the hybrid
        # bucket list up to max_num_batched_tokens with round_up=True
        # (:334-374). After warmup no override is active, so every FlashInfer
        # MoE op maps a runtime count up to the next bucket of its own list,
        # clamped to the tune_max_num_tokens its experts class passes
        # (_flashinfer_moe_tune_max_num_tokens): a token count off that list runs
        # the tactic of the next bucket up, and one past the clamp the tactic of
        # the clamp; tune at that bucket. Kernels the FlashInfer tuner does not
        # own (Triton, Marlin, Humming, DeepGEMM, vLLM CUTLASS) are unaffected.
        serving_autotunes = has_flashinfer() and current_platform.has_device_capability(90)
        max_tuned_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        # Performance testing for each token count.
        for point, num_tokens in enumerate(num_tokens_lists):
            print("num_tokens", num_tokens)
            print("topk", topk)
            hidden_states = torch.randn([num_tokens, hidden_size], dtype=torch.bfloat16, device=device)
            if isinstance(experts_impl, FallbackExperts):
                leaf_experts = experts_impl._select_experts_impl(
                    hidden_states,
                    routed_experts.w13_weight,
                    routed_experts.w2_weight,
                )
                if leaf_experts is None:
                    raise RuntimeError(f"vLLM MoE wrapper {wrapper_name} did not select an experts implementation")
            else:
                leaf_experts = experts_impl
            experts_name = type(leaf_experts).__name__ if leaf_experts is not None else "direct"
            print(f"vLLM MoE experts: wrapper={wrapper_name} leaf={experts_name}")
            source = f"vllm_{method_name}_{backend_name}_{experts_name}".lower()
            source = source.replace(" ", "_").replace("-", "_")

            num_iter = 5 if distributed == "power_law" else 1
            logits_dtype = router_logits_dtype or torch.bfloat16
            if distributed == "power_law":
                # The workload generator's smoothing Conv1d consumes FP32.
                # Module construction uses a BF16 default dtype, so restore
                # FP32 only while generating logits and then put the collector
                # default back exactly as it was.
                previous_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
                try:
                    router_logits_list = [
                        power_law_logits_v3(num_tokens, num_experts, topk, moe_ep_size, power_law_alpha)
                        .to(logits_dtype)
                        .to(device)
                        for _ in range(num_iter)
                    ]
                finally:
                    torch.set_default_dtype(previous_dtype)
            elif distributed == "balanced":
                router_logits_list = [balanced_logits(num_tokens, num_experts, topk).to(logits_dtype).to(device)]
            else:
                raise ValueError(f"Unsupported distributed mode: {distributed}")
            num_warmups = 1 if distributed == "power_law" else 3
            num_runs = 1 if distributed == "power_law" else 6

            def run_single_iteration():
                for router_logits in router_logits_list:
                    forward_context = get_forward_context()
                    forward_context.moe_layer_index = 0
                    moe_module(hidden_states, router_logits)

            tune_tokens = (
                flashinfer_tuning_bucket(
                    num_tokens,
                    max_tuned_tokens,
                    tune_max_num_tokens=_flashinfer_moe_tune_max_num_tokens(leaf_experts),
                )
                if serving_autotunes
                else None
            )
            if tune_tokens is not None:
                autotuned = flashinfer_autotuned(tune_tokens, run_single_iteration, label=source)
            else:
                autotuned = contextlib.nullcontext()
            with (
                set_forward_context({}, vllm_config),
                autotuned,
                benchmark_with_power(
                    device=device,
                    kernel_func=run_single_iteration,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                    min_warmup_seconds=FIRST_POINT_WARMUP_SECONDS if point == 0 else 0.0,
                ) as results,
            ):
                pass

            latency = results["latency_ms"] / num_iter
            power_stats = results["power_stats"]
            print(f"moe latency: {latency}")

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
                        "distribution": (
                            "power_law_" + str(power_law_alpha) if distributed == "power_law" else distributed
                        ),
                        "latency": latency,
                    }
                ],
                framework="VLLM",
                version=vllm_version,
                device_name=torch.cuda.get_device_name(device),
                op_name="moe",
                kernel_source=source,
                perf_filename=perf_filename,
                power_stats=power_stats,
            )


if __name__ == "__main__":
    import traceback

    from collector.registry_types import PerfFile

    test_cases = get_moe_test_cases()
    print(f"Total test cases: {len(test_cases)}")

    # Standalone debug entrypoint: report each failing case and keep going,
    # like the collect.py worker path does, instead of aborting the sweep.
    for test_case in test_cases[:4]:
        print(f"Running test case: {test_case}")
        try:
            run_moe_torch(*test_case, perf_filename=PerfFile.MOE)
        except Exception as error:
            traceback.print_exc()
            print(f"Test case failed ({type(error).__name__}): {test_case}")
