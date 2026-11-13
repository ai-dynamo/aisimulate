# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang MoE collector.

Times SGLang's FusedMoE layer behind its TopK router, built and bound the way a
model builds them, for the shared MoE case plan (``op_plans.OP_PLANS["moe"]``):
BF16, FP8 block, NVFP4, INT4 weight-only and MXFP4 experts. Each case pins the
MoE runner its plan dispatch names, lets SGLang pick the quant method and
runner from it, records the runner FusedMoE resolved, and times every token
point of its sweep on one layer build.
"""

# Source-checked against the sglang v0.5.23 tag (flashinfer-python 0.7.2,
# sglang-kernel 0.4.10). Citations below are python/sglang/... paths at that
# tag. One release per file; earlier code is `git log -- <this file>` away.
#
# Runner selection. The scheduler publishes its ServerArgs and seeds the MoE
# flags from --moe-runner-backend (initialize_moe_config,
# srt/layers/moe/utils.py:479-514); FusedMoE then lets the quant method
# resolve its runner (srt/layers/moe/fused_moe_triton/layer.py:576-578). This
# collector publishes a default ServerArgs and pins the flag to the backend the
# case's dispatch names (cases/base_ops/moe.yaml moe_sglang.backends, a model
# row's sglang_moe_backends), which is the value serving's arg resolution
# writes for that model, quant and SM: e.g. flashinfer_trtllm for the
# DeepSeek family on SM100/103 (srt/arg_groups/overrides.py:763-823), and for
# modelopt_fp4 under auto (srt/layers/quantization/modelopt_quant.py:3117-3123),
# flashinfer_cutlass for modelopt_fp4 on SM120
# (srt/arg_groups/overrides.py:1513-1521), Triton for Fp8MoEMethod and
# UnquantizedFusedMoEMethod under auto without an all-to-all backend
# (srt/layers/quantization/fp8.py:2962-2972, unquant.py:1169-1193). The op
# plan's dispatch also follows a model override that depends on EP size:
# GPT-OSS at EP1 on SM89/90 serves triton_kernel (op_plans._sglang_moe_runner).
#
# Routing. TopK is built as the model builds it: from the row's routing fields,
# or with the model's own routing function for Llama-4 and Gemma-4.
#
# Where this table times a different path than SGLang serving (the direction
# of each error is unknown unless stated):
# - Deferred finalize: DeepSeek-family, MiMo-V2, MiniMax-M3, Kimi-K3 and
#   Qwen3.5-FP8 layers with a separate shared expert call
#   FusedMoE.forward_deferred_finalize on flashinfer_trtllm with bypassed TopK
#   (models/deepseek_v2.py:1206-1224, models/qwen2_moe.py:700-717;
#   SGLANG_ENABLE_MOE_DEFERRED_FINALIZE defaults on, srt/environ.py:1454), so
#   the MoE kernel skips its finalize and a later fused op adds it to the
#   shared-expert output. The table times FusedMoE.forward, finalize
#   included: it overstates the routed-MoE time by that finalize. Retire it by
#   timing forward_deferred_finalize plus the fused finalize as one op.
# - MiniMax-M3: serving hands the routed scaling factor to TopK only, applied
#   to the top-k weights (apply_routed_scaling_factor_on_output=True), and
#   builds FusedMoE without it and with gate_up_interleaved=False
#   (models/minimax_m3.py:421-452). The row has one scaling field, which this
#   collector passes to both, and no interleave field: the table times the
#   factor where the runner applies it and the interleaved SwiGLU-alpha
#   epilogue (on Triton, swiglu_gpt_oss_sigmoid_alpha instead of
#   swiglu_no_interleaved_with_alpha_and_limit, moe_runner/triton_utils/
#   fused_moe.py:741-756). Both are elementwise work on the MoE output or the
#   intermediate; the GEMMs are serving's. Retire it with row fields for the
#   scaling placement and the gate/up layout.
__compat__ = "sglang==0.5.23"

import contextlib
import gc
import os
import tempfile
from collections.abc import Callable
from importlib.metadata import version as _dist_version
from pathlib import Path
from types import SimpleNamespace

import torch
from collector.case_generator import get_moe_quantization_module_config, moe_model_layer
from collector.helper import (
    FIRST_POINT_WARMUP_SECONDS,
    MOE_WEIGHT_MATRIX_SUFFIXES,
    WORKER_RESTART,
    balanced_logits,
    benchmark_with_power,
    fill_dense_weight_,
    flashinfer_autotuned,
    flashinfer_tuning_bucket,
    get_sm_version,
    log_perf,
    power_law_logits_v3,
)
from collector.op_plans import SGLANG_GEMMA4_ROUTING_ARCHITECTURES, plan_cases
from collector.sglang.moe_config_observation import append_record as append_triton_config_record
from collector.sglang.moe_config_observation import observe_triton_config_selection
from collector.sglang.runtime_limits import serving_memory_defaults
from sglang.srt.runtime_context import get_exec, get_flags, get_server_args, publish
from sglang.srt.server_args import ServerArgs


def _publish_serving_args() -> None:
    """Publish the ServerArgs record the scheduler publishes before building a model.

    FusedMoE and TopK read the resolved config bags (get_exec(), get_model())
    while they are built (srt/layers/moe/fused_moe_triton/layer.py:423,
    488-512; srt/layers/moe/topk.py:724), and get_server_args raises until a
    record is published (srt/runtime_context.py:1022-1029). A record already
    published in this process is kept, as publish would replace it
    (:1489-1515). The four fields equal their declared defaults; they are
    pinned so a default change cannot move the table: no deterministic
    inference (which also disables serving's FlashInfer autotune,
    srt/model_executor/runner/flashinfer_autotune.py:91), no fused MoE sum
    all-reduce (a TP collective), no KTransformers wrapper and the default
    MXFP4 activation precision.
    """
    try:
        get_server_args()
    except ValueError:
        publish(
            ServerArgs(
                model_path="dummy",
                enable_deterministic_inference=False,
                enable_fused_moe_sum_all_reduce=False,
                kt_weight_path=None,
                flashinfer_mxfp4_moe_precision="default",
            ),
            role="scheduler",
        )


_publish_serving_args()

import sglang.kernels.ops.moe.fused_moe_triton_kernels as _fmoe_kernels_mod
import sglang.srt.layers.moe.fused_moe_triton.layer as _moe_layer_mod
import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as _fi_cutlass_runner_mod
import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm as _fi_trtllm_runner_mod
import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as _triton_fused_moe_mod
import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config as _triton_config_mod
import sglang.srt.layers.moe.token_dispatcher.standard as _std_dispatch_mod
import sglang.srt.layers.moe.topk as _topk_mod
import sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_mxint4_moe as _mxint4_mod
import sglang.srt.layers.quantization.fp8 as _fp8_mod
import sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe as _mxfp4_trtllm_mod
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import MoeRunnerBackend, RoutingMethodType
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config

# The MoE modules that read the parallel topology through a module-level
# get_parallel / is_allocation_symmetric binding on the planned paths;
# _patch_framework_moe_parallel replaces both in each.
_PARALLEL_READERS = (
    _moe_layer_mod,
    _std_dispatch_mod,
    _topk_mod,
    _triton_fused_moe_mod,
    _mxint4_mod,
    _fp8_mod,
    _mxfp4_trtllm_mod,
    _fi_trtllm_runner_mod,
    _fi_cutlass_runner_mod,
)

# Serving runs FlashInfer's autotuner at warm-up only when the MoE runner is one
# of these (should_run_flashinfer_autotune,
# srt/model_executor/runner/flashinfer_autotune.py:117-124) and only on SM90 or
# newer (:155-156).
_SERVING_AUTOTUNED_MOE_RUNNERS = frozenset(
    {
        "flashinfer_trtllm",
        "flashinfer_trtllm_routed",
        "flashinfer_mxfp4",
        "flashinfer_cutedsl",
        "flashinfer_cutlass",
        "flashinfer_megamoe",
    }
)

_KERNEL_SOURCE_BY_RUNNER = {
    "triton": "sglang_fused_moe_triton",
    "triton_kernel": "sglang_triton_kernel_moe",
    "marlin": "sglang_marlin_moe",
    "flashinfer_trtllm": "sglang_flashinfer_trtllm_moe",
    # The routed TRT-LLM-gen runner: the router's top-k is materialized and the
    # kernel takes precomputed ids (moe_runner/flashinfer_trtllm.py:1845-1876);
    # NVFP4 shares the trtllm weight prep (modelopt_quant.py:2389-2392, 3237-3243).
    "flashinfer_trtllm_routed": "sglang_flashinfer_trtllm_routed_moe",
    "flashinfer_cutlass": "sglang_flashinfer_cutlass_moe",
}

# FlashInfer's MoE ops profile every bucket of their own list,
# get_hybrid_num_tokens_buckets(tune_max_num_tokens), in one tuning call whatever
# the call's m: the list is a static tuple (flashinfer 0.7.2
# fused_moe/shared/tuning.py:303-309, fused_moe/core.py:970-981), which the tuner
# takes as is and profiles in full (autotuner/autotuner.py:3282-3287, 2331-2332).
# A lookup maps m to the next bucket up, capped at the tune_max_num_tokens of
# that call (helper.flashinfer_tuning_bucket). SGLang's default warm-up calls each
# op in one decode dummy forward (runtime_limits.decode_autotune_max_tokens), so
# the tuned list depends on what the runner passes as tune_max_num_tokens:
# - the TRT-LLM-gen runners (flashinfer_trtllm and _routed, the SM100 MXFP4 leaf,
#   the DeepSeek-V4 FP4 TRT-LLM method, MXINT4) pass next_power_of_2(m)
#   (moe_runner/flashinfer_trtllm.py:821, 1007, 1067, 1247, 1298, 1333, 1588,
#   1642, 1780, 1808; mxfp4_flashinfer_trtllm_moe.py:521;
#   compressed_tensors_w4a4_mxint4_moe.py:361). Warm-up tunes the buckets up to
#   next_power_of_2(batch size); a run-time m maps to its own bucket (its cap
#   next_power_of_2(m) never binds), and past that list runs FlashInfer's
#   fallback tactic. GLM-5.3-Flash on these runners also tunes its experts at the
#   prefill chunk (models/glm5_next.py:1503-1531; flashinfer_autotune.py:412-443).
# - The CUTLASS runner (flashinfer_cutlass, the SM90 / SM120 MXFP4 leaves, the
#   DeepSeek-V4 FP4 CUTLASS method) passes min(chunked prefill size, 8192) on
#   every call (moe_runner/flashinfer_cutlass.py:168-181, 388, 610): warm-up
#   tunes every bucket up to it, and m at or above it runs its tactic.
_CUTLASS_MOE_DEFAULT_TUNE_MAX_NUM_TOKENS = 8192
_PREFILL_AUTOTUNED_MOE_ARCHITECTURES = frozenset({"Glm5NextForConditionalGeneration"})
_PREFILL_AUTOTUNED_MOE_RUNNERS = frozenset({"flashinfer_trtllm", "flashinfer_trtllm_routed"})


def _flashinfer_moe_tuning_bucket(
    num_tokens: int, *, kernel_source: str, moe_backend: str, tp_size: int, architecture: str | None
) -> int | None:
    """The bucket serving's warm-up tuned the tactic of an m-token MoE call for, or None (untuned).

    ``tp_size`` is the serving world (moe_tp x moe_ep), which sets the default
    decode batch size; ``kernel_source`` names the runner family (CUTLASS or
    TRT-LLM-gen) the built layer runs.
    """
    chunked_prefill_size, decode_batch_size = serving_memory_defaults(tp_size)
    if "cutlass" in kernel_source:
        tune_max = min(chunked_prefill_size, _CUTLASS_MOE_DEFAULT_TUNE_MAX_NUM_TOKENS)
        return flashinfer_tuning_bucket(num_tokens, tune_max, tune_max_num_tokens=tune_max)
    warmup_tokens = decode_batch_size
    if architecture in _PREFILL_AUTOTUNED_MOE_ARCHITECTURES and moe_backend in _PREFILL_AUTOTUNED_MOE_RUNNERS:
        warmup_tokens = max(warmup_tokens, chunked_prefill_size)
    return flashinfer_tuning_bucket(num_tokens, 1 << (warmup_tokens - 1).bit_length())


# Mxfp4MoEMethod runs flashinfer_mxfp4 on one of three FlashInfer leaves by GPU
# (srt/layers/quantization/mxfp4.py:420-437): trtllm-gen on SM100/103, the
# CUTLASS mixed-input MoE on SM90 and SM120.
_KERNEL_SOURCE_BY_MXFP4_LEAF = {
    "trtllm_sm100": "sglang_mxfp4_flashinfer_trtllm_moe",
    "cutlass_sm90": "sglang_flashinfer_cutlass_moe",
    "cutlass_sm120": "sglang_flashinfer_cutlass_moe",
}

# DeepSeek-V4 FP4 experts under flashinfer_mxfp4: Fp8Config builds the CUTLASS
# method on SM90/SM120 and the trtllm-gen method on SM100/103
# (srt/layers/quantization/fp8.py:459-472).
_KERNEL_SOURCE_BY_FP4_EXPERT_METHOD = {
    "Mxfp4FlashinferCutlassMoEMethod": "sglang_flashinfer_cutlass_moe",
    "Mxfp4FlashinferTrtllmMoEMethod": "sglang_mxfp4_flashinfer_trtllm_moe",
}


def _mxfp4_activation_precision(moe_type: str) -> str:
    """Map the persisted quant label to SGLang's explicit activation mode."""

    return "bf16" if moe_type == "w4a16_mxfp4" else "default"


def _ensure_writable_flashinfer_cubin_dir() -> None:
    """Overlay packaged cubins when FlashInfer needs to create JIT symlinks."""
    from flashinfer.jit import cubin_loader
    from flashinfer.jit import env as jit_env

    source = Path(jit_env.FLASHINFER_CUBIN_DIR)
    if os.access(source, os.W_OK):
        return
    configured = os.environ.get("FLASHINFER_CUBIN_DIR")
    target = Path(configured) if configured and Path(configured) != source else None
    if target is None:
        version = str(jit_env.flashinfer_version).replace("+", "_")
        target = Path(tempfile.gettempdir()) / f"aic_flashinfer_cubins_{os.getuid()}_{version}"
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    for artifact_root in source.iterdir():
        overlay_entry = target / artifact_root.name
        try:
            overlay_entry.symlink_to(artifact_root, target_is_directory=artifact_root.is_dir())
        except FileExistsError:
            pass
    jit_env.FLASHINFER_CUBIN_DIR = target
    cubin_loader.FLASHINFER_CUBIN_DIR = target
    os.environ["FLASHINFER_CUBIN_DIR"] = str(target)


def get_moe_test_cases():
    return plan_cases("moe", "sglang", get_sm_version())


@contextlib.contextmanager
def _patch_framework_moe_parallel(*, moe_tp_size: int, moe_ep_size: int):
    """Give the MoE modules the parallel topology of rank 0 of a moe_tp x moe_ep group.

    FusedMoE, its dispatcher, TopK and the runners read sizes, ranks and the TP
    group through their own module-level ``get_parallel`` binding and gate
    symmetric-memory allocation on ``is_allocation_symmetric``. Each such
    binding in ``_PARALLEL_READERS`` is replaced for the ``with`` block. No
    process group exists: the ``tp_group`` they pass to ``use_symmetric_memory``
    is ``None``, which the disabled context never touches
    (srt/layers/moe/topk.py:938-940).
    """
    parallel = SimpleNamespace(
        tp_size=moe_tp_size,
        tp_rank=0,
        tp_group=None,
        moe_tp_size=moe_tp_size,
        moe_tp_rank=0,
        moe_ep_size=moe_ep_size,
        moe_ep_rank=0,
    )
    originals = []
    try:
        for module in _PARALLEL_READERS:
            for name, value in (("get_parallel", lambda: parallel), ("is_allocation_symmetric", lambda: False)):
                if hasattr(module, name):
                    originals.append((module, name, getattr(module, name)))
                    setattr(module, name, value)
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


@contextlib.contextmanager
def _pin_moe_runner_backend(backend: MoeRunnerBackend):
    """Pin the MoE runner for the ``with`` block as --moe-runner-backend does.

    ``initialize_moe_config`` stores the parsed backend in
    ``get_flags().moe.runner_backend`` (srt/layers/moe/utils.py:491-492), the
    value ``get_moe_runner_backend`` returns (:524-528) to FusedMoE, the quant
    methods and the dispatcher. ``MoeFlags.override`` is SGLang's scoped,
    validated writer for it (srt/runtime_context.py:523-541, 557-579).
    """
    with get_flags().moe.override(runner_backend=backend):
        yield


@contextlib.contextmanager
def _pin_flashinfer_mxfp4_moe_precision(value: str | None):
    """Pin ``flashinfer_mxfp4_moe_precision`` for the ``with`` block; ``None`` leaves it.

    The MXFP4 methods read it from ``get_exec().moe`` when they are built
    (srt/layers/quantization/mxfp4.py:408-410,
    mxfp4_flashinfer_trtllm_moe.py:145-147), a published config bag whose
    scoped writer is ``override`` (srt/runtime_context.py:848-864).
    """
    if value is None:
        yield
        return
    with get_exec().moe.override(flashinfer_mxfp4_moe_precision=value):
        yield


def _quant_config(moe_type: str, *, is_fp4_experts: bool, int4_group_size: int):
    """The quant config a checkpoint of this label publishes for its experts."""
    if moe_type == "bfloat16":
        return None
    if moe_type == "int4_wo":
        return CompressedTensorsConfig.from_config(
            {
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": True,
                            "strategy": "group",
                            "group_size": int4_group_size,
                            "dynamic": False,
                        },
                        "input_activations": None,
                        "output_activations": None,
                    }
                },
                "format": "pack-quantized",
                "ignore": [],
            }
        )
    if moe_type == "fp8_block":
        return Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[128, 128])
    if moe_type == "nvfp4":
        return ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16)
    if is_fp4_experts:
        return Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
            is_fp4_experts=True,
        )
    if moe_type in {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}:
        return Mxfp4Config(is_checkpoint_mxfp4_serialized=True)
    raise ValueError(f"Unsupported SGLang MoE quant mode {moe_type!r}")


def _kernel_source(moe_layer, *, moe_type: str, moe_backend: str, is_fp4_experts: bool) -> str:
    """Name the kernel the built layer runs, after checking it is the requested runner.

    ``FusedMoE.runner_backend`` is the runner the quant method resolved
    (srt/layers/moe/fused_moe_triton/layer.py:576-578); the quant method and,
    for INT4, the compressed-tensors scheme pin the kernel family.
    """
    quant_method = moe_layer.quant_method
    method = type(quant_method).__name__
    runner = getattr(moe_layer.runner_backend, "value", None)
    if is_fp4_experts:
        expected_methods = _KERNEL_SOURCE_BY_FP4_EXPERT_METHOD
    elif moe_type == "int4_wo":
        expected_methods = {"CompressedTensorsFusedMoEMethod": None}
    elif moe_type == "nvfp4":
        expected_methods = {"ModelOptNvFp4FusedMoEMethod": None}
    elif moe_type in {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}:
        expected_methods = {"Mxfp4MoEMethod": None}
    elif moe_type == "fp8_block":
        expected_methods = {"Fp8MoEMethod": None}
    else:
        expected_methods = {"UnquantizedFusedMoEMethod": None}
    if method not in expected_methods or runner != moe_backend:
        raise RuntimeError(
            f"SGLang {moe_type} MoE did not build the requested path: requested_backend={moe_backend}, "
            f"method={method}, runner_backend={runner}"
        )
    if moe_type == "int4_wo":
        # get_moe_scheme picks MXINT4 for an explicit flashinfer_trtllm runner and
        # Marlin WNA16 for every other CUDA runner except an SM100/103 auto or
        # explicit triton (srt/layers/quantization/compressed_tensors/compressed_tensors.py:911-951).
        expected_scheme = {"flashinfer_trtllm": "CompressedTensorsMxInt4MoE", "marlin": "CompressedTensorsWNA16MoE"}
        scheme = type(getattr(moe_layer, "scheme", None)).__name__
        if scheme != expected_scheme[moe_backend]:
            raise RuntimeError(f"SGLang INT4-WO built scheme {scheme} on {moe_backend}")
    if moe_backend == "flashinfer_mxfp4":
        precision = getattr(quant_method, "flashinfer_mxfp4_moe_precision", None)
        if precision != _mxfp4_activation_precision(moe_type):
            raise RuntimeError(
                f"SGLang {moe_type} MXFP4 activation precision is {precision!r}, "
                f"expected {_mxfp4_activation_precision(moe_type)!r}"
            )
        if is_fp4_experts:
            return _KERNEL_SOURCE_BY_FP4_EXPERT_METHOD[method]
        leaf = getattr(quant_method, "_fi_kernel", None)
        if leaf not in _KERNEL_SOURCE_BY_MXFP4_LEAF:
            raise RuntimeError(f"SGLang {moe_type} has unexpected FlashInfer MXFP4 leaf {leaf!r}")
        return _KERNEL_SOURCE_BY_MXFP4_LEAF[leaf]
    return _KERNEL_SOURCE_BY_RUNNER[moe_backend]


def _gemma4_routing_function(num_experts: int, device: str) -> Callable:
    """Gemma4MoE's custom routing function, as SGLang serving builds it.

    Serving's TopK routes Gemma-4 through a closure Gemma4MoE.__init__ builds
    (srt/models/gemma4_causal.py:251-277), which select_experts calls in place
    of its own top-k (srt/layers/moe/topk.py:3092-3103). On CUDA 2-D float
    router logits, which this collector always passes, the closure is
    gemma4_fused_routing (gemma4_causal.py:259-265): one _gemma4_routing_kernel
    launch that takes the top-k of all logits, softmaxes them and multiplies
    by per_expert_scale (kernels/ops/moe/gemma4_routing.py:75-186).
    per_expert_scale is a parameter of ones in the model dtype
    (gemma4_causal.py:246, created under model_loader/loader.py
    set_default_torch_dtype); the checkpoint's values do not change the work.
    """
    from sglang.kernels.ops.moe.gemma4_routing import gemma4_fused_routing

    per_expert_scale = torch.ones(num_experts, dtype=torch.bfloat16, device=device)

    def routing_function(hidden_states, gating_output, topk, renormalize):
        return gemma4_fused_routing(gating_output, per_expert_scale, topk)

    return routing_function


def _benchmark_moe_layer(
    *,
    moe_type: str,
    moe_backend: str,
    num_tokens_list: list[int],
    on_result: Callable[[int, float, dict, str], None],
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    moe_tp_size: int,
    moe_ep_size: int,
    distributed: str,
    power_law_alpha: float,
    activation: str,
    is_gated: bool,
    has_bias: bool,
    gemm1_alpha: float | None,
    gemm1_clamp_limit: float | None,
    swiglu_limit: float | None,
    scoring_func: str,
    routing_method_type: str | None,
    routed_scaling_factor: float | None,
    renormalize: bool,
    has_correction_bias: bool,
    num_expert_group: int | None,
    topk_group: int | None,
    apply_router_weight_on_input: bool,
    is_fp4_experts: bool,
    int4_group_size: int,
    architecture: str | None,
    device: str,
) -> str:
    """Build one SGLang FusedMoE + TopK and time every token point on it.

    Each measured row is handed to ``on_result(num_tokens, latency_ms,
    power_stats, kernel_source)`` as soon as it is timed, so a crash later in
    the sweep keeps the rows already persisted. Returns the kernel source.
    """
    if moe_backend.startswith("flashinfer"):
        _ensure_writable_flashinfer_cubin_dir()

    quant_config = _quant_config(moe_type, is_fp4_experts=is_fp4_experts, int4_group_size=int4_group_size)
    routing_method = None if routing_method_type is None else RoutingMethodType[routing_method_type]
    serving_autotunes = (
        moe_backend in _SERVING_AUTOTUNED_MOE_RUNNERS and torch.cuda.get_device_capability(device)[0] >= 9
    )
    # Kimi-K3's SiTU experts load w1/w3 non-interleaved and its TopK materializes
    # the top-k on the FlashInfer runners (models/kimi_k3.py:505-551); every other
    # row keeps FusedMoE's interleaved default and the format bind_runner_backend picks.
    situ = activation == "situ"
    moe_layer = None
    try:
        with (
            _pin_moe_runner_backend(MoeRunnerBackend(moe_backend)),
            _pin_flashinfer_mxfp4_moe_precision(
                _mxfp4_activation_precision(moe_type) if moe_backend == "flashinfer_mxfp4" else None
            ),
            _patch_framework_moe_parallel(moe_tp_size=moe_tp_size, moe_ep_size=moe_ep_size),
        ):
            moe_layer = FusedMoE(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=inter_size,
                layer_id=0,
                top_k=topk,
                params_dtype=torch.bfloat16,
                reduce_results=False,
                quant_config=quant_config,
                prefix="aic_sglang_moe",
                activation=activation,
                is_gated=is_gated,
                with_bias=has_bias,
                gemm1_alpha=gemm1_alpha,
                gemm1_clamp_limit=gemm1_clamp_limit,
                swiglu_limit=swiglu_limit,
                routing_method_type=routing_method,
                routed_scaling_factor=routed_scaling_factor,
                apply_router_weight_on_input=apply_router_weight_on_input,
                gate_up_interleaved=not situ,
            ).to(device)
            # Expert weight matrices get dense, checkpoint-like bits (helper.fill_dense_weight_
            # says why and how); scales and alphas 1, everything else (biases, shapes, g_idx) 0.
            with torch.no_grad():
                for name, parameter in moe_layer.named_parameters():
                    if "scale" in name or "alpha" in name:
                        parameter.fill_(1)
                    elif name.endswith(MOE_WEIGHT_MATRIX_SUFFIXES):
                        fill_dense_weight_(parameter.data)
                    else:
                        parameter.zero_()
            kernel_source = _kernel_source(
                moe_layer, moe_type=moe_type, moe_backend=moe_backend, is_fp4_experts=is_fp4_experts
            )
            # Load-time weight work (shuffles, scale interleave, padding) runs here, untimed.
            moe_layer.quant_method.process_weights_after_loading(moe_layer)

            correction_bias = (
                torch.zeros(num_experts, dtype=torch.float32, device=device) if has_correction_bias else None
            )
            if routing_method_type == "Llama4":
                # Serving builds the Llama-4 router with a custom routing function
                # (fast_topk on the raw logits, then sigmoid; renormalize=False),
                # models/llama4.py:76-91 and :113-118.
                from sglang.srt.models.llama4 import Llama4MoE

                topk_layer = TopK(
                    top_k=topk,
                    layer_id=0,
                    renormalize=False,
                    custom_routing_function=Llama4MoE.custom_routing_function,
                )
            elif architecture in SGLANG_GEMMA4_ROUTING_ARCHITECTURES:
                # Gemma4MoE builds its TopK from the custom routing function
                # alone (models/gemma4_causal.py:279-283).
                topk_layer = TopK(
                    top_k=topk,
                    layer_id=0,
                    custom_routing_function=_gemma4_routing_function(num_experts, device),
                )
            else:
                topk_layer = TopK(
                    top_k=topk,
                    layer_id=0,
                    use_grouped_topk=num_expert_group is not None and topk_group is not None,
                    num_expert_group=num_expert_group,
                    topk_group=topk_group,
                    renormalize=renormalize,
                    scoring_func=scoring_func,
                    correction_bias=correction_bias,
                    routed_scaling_factor=routed_scaling_factor,
                    apply_routed_scaling_factor_on_output=(
                        routed_scaling_factor is not None and moe_layer.should_fuse_routed_scaling_factor_in_topk
                    ),
                    is_fp4_experts=is_fp4_experts,
                    output_format=(
                        TopKOutputFormat.STANDARD
                        if situ and moe_backend in ("flashinfer_mxfp4", "flashinfer_trtllm")
                        else None
                    ),
                )
            # A model binds its router to the built experts, which picks the TopK
            # output format: kernel-side routing (bypassed) only where the
            # runner's routing can reproduce the router, else a materialized
            # top-k (srt/layers/moe/topk.py:613-650, 769-805). Models that pass
            # a routing method to FusedMoE bind with the experts' resolved one
            # (models/deepseek_v2.py:829-832, qwen2_moe.py:343-346,
            # llama4.py:131-134, nemotron_h.py:303-306); models that pass none
            # bind without it (models/mixtral.py:110, gemma4_causal.py:298,
            # gpt_oss.py:255).
            topk_layer.bind_runner_backend(
                moe_layer.runner_backend,
                routing_method_type=None if routing_method is None else moe_layer.routing_method_type,
            )

            for point, num_tokens in enumerate(num_tokens_list):
                print(f"num_tokens {num_tokens} ({kernel_source})")
                hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
                # The vLLM and TRT-LLM MoE collectors' protocol: balanced routing is one
                # deterministic router input timed after 3 warm-ups over 6 runs; a random
                # routing averages 5 draws in one run after one warm-up.
                if distributed == "balanced":
                    logits = [balanced_logits(num_tokens, num_experts, topk).to(device=device, dtype=torch.float32)]
                elif distributed == "power_law":
                    logits = [
                        power_law_logits_v3(num_tokens, num_experts, topk, moe_ep_size, power_law_alpha).to(
                            device=device, dtype=torch.float32
                        )
                        for _ in range(5)
                    ]
                else:
                    logits = [
                        torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device) for _ in range(5)
                    ]

                def kernel_func(hidden_states=hidden_states, logits=logits):
                    # One forward per router input: TopK, then FusedMoE (dispatch,
                    # runner, combine), as a model's MoE block runs them.
                    for router_logits in logits:
                        moe_layer(hidden_states, topk_layer(hidden_states, router_logits))

                tune = contextlib.nullcontext()
                tune_tokens = (
                    _flashinfer_moe_tuning_bucket(
                        num_tokens,
                        kernel_source=kernel_source,
                        moe_backend=moe_backend,
                        tp_size=moe_tp_size * moe_ep_size,
                        architecture=architecture,
                    )
                    if serving_autotunes
                    else None
                )
                if tune_tokens is not None:
                    # Serving tunes FlashInfer ops under autotune(True) plus
                    # autotune_v2(mode="tune", persistent_cache=False), where
                    # FlashInfer's default tactic races the candidates
                    # (flashinfer_autotune.py:354-362). The pass profiles the
                    # bucket serving tuned this point's tactic at
                    # (_flashinfer_moe_tuning_bucket), then the timed runs look
                    # it up (helper.flashinfer_autotuned). A point past the
                    # warm-up's list runs untuned, on FlashInfer's fallback
                    # tactic, as serving does.
                    def tune_fn(tune_tokens=tune_tokens, kernel_func=kernel_func):
                        from flashinfer import autotune_v2

                        with autotune_v2(mode="tune", persistent_cache=False, tuning_buckets=(tune_tokens,)):
                            kernel_func()

                    tune = flashinfer_autotuned(tune_tokens, tune_fn, label=kernel_source)
                with (
                    tune,
                    benchmark_with_power(
                        device=device,
                        kernel_func=kernel_func,
                        num_warmups=3 if len(logits) == 1 else 1,
                        num_runs=6 if len(logits) == 1 else 1,
                        repeat_n=1,
                        min_warmup_seconds=FIRST_POINT_WARMUP_SECONDS if point == 0 else 0.0,
                    ) as results,
                ):
                    pass
                on_result(num_tokens, results["latency_ms"] / len(logits), results["power_stats"], kernel_source)
            return kernel_source
    finally:
        if moe_layer is not None:
            for parameter in moe_layer.parameters():
                parameter.__dict__.pop("weight_loader", None)
            parameter = None
            moe_layer = None
        _fmoe_kernels_mod._B_DESC_CACHE.clear()
        gc.collect()
        torch.cuda.empty_cache()


# The SMs on which each version-sensitive lane's declared runner was traced to
# what SGLang v0.5.23 serves; any other SM raises before building.
# - int4_wo (Kimi-K2.5 compressed-tensors W4A16 group 32): SM100/103 serve
#   flashinfer_trtllm (srt/arg_groups/overrides.py:763-823) with the MXINT4
#   scheme; every other CUDA SM keeps auto and builds the Marlin WNA16 scheme
#   (srt/layers/quantization/compressed_tensors/compressed_tensors.py:911-951).
# - w4a16_mxfp4: GPT-OSS on SM100/103 serves flashinfer_mxfp4
#   (srt/arg_groups/model_overrides/gpt_oss.py:64-68) on the trtllm-gen leaf.
#   On SM89/90 an EP>1 GPT-OSS server keeps auto, which Mxfp4MoEMethod resolves
#   to Triton (srt/layers/quantization/mxfp4.py:1443-1450) on experts upcast to
#   BF16 at load (:1115-1133); an EP1 server runs triton_kernel (gpt_oss.py:
#   102-120) on the MXFP4 experts swizzled for triton_kernels (mxfp4.py:
#   1046-1076), because sglang-kernel installs the triton_kernels package
#   (kernels/aot/CMakeLists.txt:548-553; is_triton_kernels_available,
#   srt/utils/common.py:4814-4815). The op plan's dispatch follows that EP
#   split (op_plans._sglang_moe_runner). Kimi-K3 keeps
#   auto off SM100/103/107 (model_overrides/kimi_k3.py:172-190), so SM89, SM90
#   and SM120 serve it on the same Triton path, SiTU applied from gemm1_alpha /
#   gemm1_clamp_limit (moe_runner/triton.py:162-170, 208-215;
#   triton_utils/fused_moe.py:825-834). SM120 GPT-OSS runs the MXFP8 x MXFP4
#   lane below (collector/skips.py).
# - w4a8_mxfp4_mxfp8: the trtllm-gen MXFP4 x MXFP8 leaf on SM100/103 and the
#   CUTLASS MXFP8 x MXFP4 kernel on SM120, which flashinfer_mxfp4 serves there for
#   GPT-OSS (gpt_oss.py:69-74; Mxfp4MoEMethod's cutlass_sm120 leaf, mxfp4.py:
#   426-427, quantizes activations to MXFP8, moe_runner/flashinfer_cutlass.py:
#   541-548) and for the DeepSeek-V4 FP4 experts
#   (model_overrides/deepseek_v4.py:75-88; Mxfp4FlashinferCutlassMoEMethod
#   scales activations to MXFP8 on SM120, mxfp4_flashinfer_cutlass_moe.py:36-44).
_VERIFIED_LANE_SMS = {
    "int4_wo": frozenset({89, 90, 100, 103, 120}),
    "w4a16_mxfp4": frozenset({89, 90, 100, 103, 120}),
    "w4a8_mxfp4_mxfp8": frozenset({100, 103, 120}),
}


def _raise_if_unverified_moe_lane(moe_type: str) -> str:
    """Refuse a version-sensitive MoE lane on an SM its runner was not traced on.

    Returns the installed SGLang version for the perf row.
    """
    installed_version = _dist_version("sglang")
    verified_sms = _VERIFIED_LANE_SMS.get(moe_type)
    if verified_sms is not None and get_sm_version() not in verified_sms:
        raise RuntimeError(
            f"SGLang {moe_type} MoE runner is traced to serving only on SM{sorted(verified_sms)}; "
            f"installed: {installed_version}, SM{get_sm_version()}; "
            "re-verify framework dispatch before extending this guard."
        )
    return installed_version


def _triton_config_namespaces() -> list:
    """SGLang's Triton MoE config module, whose functions try_get_optimal_moe_config calls."""
    return [_triton_config_mod]


def run_moe_torch(
    moe_type,
    num_tokens_list,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    distributed="power_law",
    power_law_alpha=0,
    swiglu_limit=None,
    moe_backend="triton",
    activation="silu",
    is_gated=True,
    has_bias=False,
    gemm1_alpha=None,
    gemm1_clamp_limit=None,
    scoring_func="softmax",
    routing_method_type=None,
    routed_scaling_factor=None,
    renormalize=True,
    has_correction_bias=False,
    num_expert_group=None,
    topk_group=None,
    apply_router_weight_on_input=False,
    is_fp4_experts=False,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    # One case carries the whole token sweep; a single int is accepted for fills and
    # direct callers that re-measure one point.
    num_tokens_list = [int(num_tokens_list)] if isinstance(num_tokens_list, int) else [int(n) for n in num_tokens_list]
    if not num_tokens_list:
        raise ValueError("SGLang MoE case has an empty num_tokens_list")

    assert moe_type in [
        "fp8_block",
        "bfloat16",
        "nvfp4",
        "w4a8_mxfp4_mxfp8",
        "int4_wo",
        "w4a16_mxfp4",
    ], "only support moe type = fp8_block, bfloat16, nvfp4, int4_wo, w4a16_mxfp4, or w4a8_mxfp4_mxfp8"
    installed_sglang_version = _raise_if_unverified_moe_lane(moe_type)
    if moe_backend not in {
        "triton",
        "triton_kernel",
        "marlin",
        "flashinfer_trtllm",
        "flashinfer_trtllm_routed",
        "flashinfer_cutlass",
        "flashinfer_mxfp4",
    }:
        raise ValueError(f"Unsupported SGLang MoE backend: {moe_backend}")
    # Marlin is a weight-only (bf16-activation) runner: a valid identity for
    # INT4-WO and MXFP4 W4A16 only. NVFP4 and MXFP8-activation modes measured
    # through a Marlin repack would be mislabeled rows.
    if moe_backend == "marlin" and moe_type not in ("int4_wo", "w4a16_mxfp4"):
        raise ValueError(
            f"SGLang Marlin is only valid for the weight-only modes int4_wo/w4a16_mxfp4, got moe_type={moe_type!r}"
        )
    assert inter_size % moe_tp_size == 0, "inter_size % moe_tp_size must be 0"
    assert num_experts % moe_ep_size == 0, "num_experts must be divisible by moe_ep_size"

    local_inter_size = inter_size // moe_tp_size
    sm_version = get_sm_version()
    if (
        moe_type == "bfloat16"
        and moe_backend == "flashinfer_cutlass"
        and (hidden_size % 8 != 0 or local_inter_size % 8 != 0)
    ):
        # FlashInfer 0.7.2 (pinned by SGLang 0.5.23, python/pyproject.toml:41)
        # flashinfer_cutlass_fused_moe_binding.cu:430-436: min_alignment =
        # 128 / (8 * min(act_bytes, weight_bytes)) = 8 for BF16, checked on both
        # hidden_size and inter_size; SGLang does not pad this path
        # (srt/layers/quantization/unquant.py:1289-1305).
        raise ValueError(
            "FIXME(kernel-limit): SGLang FlashInfer CUTLASS BF16 MoE requires hidden_size and local_inter_size "
            "to be divisible by 8 (flashinfer 0.7.2 flashinfer_cutlass_fused_moe_binding.cu:430-436), "
            f"got hidden_size={hidden_size}, local_inter_size={local_inter_size}"
        )
    activation_vector_size = 16 if sm_version >= 100 else 8
    if (
        moe_backend == "triton"
        and is_gated
        and activation in ("gelu", "silu")
        and gemm1_alpha is None
        and gemm1_clamp_limit is None
        and local_inter_size % activation_vector_size != 0
    ):
        # The Triton runner's gated activation is the JIT act_and_mul kernel
        # (srt/layers/moe/moe_runner/triton_utils/fused_moe.py:61-66), which checks
        # hidden_size % kVecSize == 0 (kernels/jit/csrc/elementwise/activation.cuh:219),
        # kVecSize = kMaxVecBytes / sizeof(bf16) with kMaxVecBytes 32 on Blackwell
        # and 16 below (kernels/jit/include/sgl_kernel/utils.cuh:140). A SwiGLU with
        # gemm1_alpha or gemm1_clamp_limit (GPT-OSS) never reaches it: the runner runs
        # the torch.compile'd swiglu_gpt_oss_sigmoid_alpha /
        # swiglu_no_interleaved_with_alpha_and_limit / _swiglu_silu_clamp_mul
        # (fused_moe.py:410-434, 741-760); gelu asserts both are None (:837-838).
        # local_inter_size is the unpadded width: serving pads it for
        # flashinfer_trtllm (srt/layers/moe/fused_moe_triton/layer.py:513-524), and
        # Mxfp4MoEMethod pads its buffers to 64 on the Triton paths
        # (srt/layers/quantization/mxfp4.py:548-551).
        raise ValueError(
            "FIXME(kernel-limit): SGLang Triton gated BF16 GELU/SiLU (JIT act_and_mul, "
            f"activation.cuh:219) requires local_inter_size to be divisible by {activation_vector_size}, "
            f"got local_inter_size={local_inter_size} (activation={activation})"
        )
    if (
        moe_type == "w4a16_mxfp4"
        and is_fp4_experts
        and moe_backend == "flashinfer_mxfp4"
        and sm_version == 90
        and (hidden_size % 128 != 0 or local_inter_size % 128 != 0)
    ):
        # SM90 FP4 experts build Mxfp4FlashinferCutlassMoEMethod
        # (srt/layers/quantization/fp8.py:459-466), whose create_weights raises
        # unless both widths are multiples of 128 (mxfp4_flashinfer_cutlass_moe.py:67-75).
        raise ValueError(
            "FIXME(kernel-limit): SGLang SM90 DeepSeek-V4 W4A16 FP4 experts require hidden_size and local_inter_size "
            "to be divisible by 128 (mxfp4_flashinfer_cutlass_moe.py:67-75), "
            f"got hidden_size={hidden_size}, local_inter_size={local_inter_size}"
        )
    int4_group_size = 128
    if moe_type == "int4_wo":
        if moe_backend not in ("marlin", "flashinfer_trtllm"):
            raise ValueError(f"SGLang int4_wo requires the Marlin or flashinfer_trtllm backend, got {moe_backend}")
        int4_config = get_moe_quantization_module_config("sglang", moe_type, model_name=model_name)
        int4_group_size = int(int4_config.get("group_size", 128))
        if hidden_size % int4_group_size != 0 or local_inter_size % int4_group_size != 0:
            # FIXME(re-verify): no source line rejects K % group_size. The kernels'
            # own limits are stricter and fire on their own: SM100/103 MXINT4 packs
            # block_k=128 bytes (compressed_tensors_w4a4_mxint4_moe.py:251-257 ->
            # flashinfer 0.7.2 fused_moe/core.py:623), Marlin tiles
            # (moe_wna16_marlin.cuh). Marlin floor-divides the groups
            # (compressed_tensors_wNa16_moe.py:158-159), so an unaligned K may run
            # with a different group size than the row says; this stays an
            # unclassified failure until a run decides it.
            raise ValueError(
                "SGLang INT4-WO group quantization requires hidden_size and local_inter_size "
                f"to be divisible by group_size={int4_group_size}, got "
                f"hidden_size={hidden_size}, local_inter_size={local_inter_size}"
            )
    if moe_type == "fp8_block" and moe_backend == "triton" and (hidden_size % 128 != 0 or local_inter_size % 128 != 0):
        # Fp8MoEMethod.create_weights rejects a local intermediate width that is not
        # a multiple of the 128 block (srt/layers/quantization/fp8.py:1588-1601),
        # and the Triton kernel quantizes each GEMM input per 128-token group,
        # asserting K % 128 == 0 (kernels/ops/moe/fused_moe_triton_kernels.py:913 ->
        # kernels/ops/quantization/fp8_kernel.py:502): K = hidden_size for the
        # first GEMM, local_inter_size for the second.
        raise ValueError(
            "FIXME(kernel-limit): SGLang Triton fp8_block requires hidden_size and local_inter_size "
            "to be divisible by 128 (fp8.py:1588-1601, fp8_kernel.py:502), "
            f"got hidden_size={hidden_size}, local_inter_size={local_inter_size}"
        )

    distribution = "power_law_" + str(power_law_alpha) if distributed == "power_law" else distributed

    # Record (never choose) whether each timed Triton call used a tuned JSON or
    # SGLang's default config. One task times a whole num_tokens series, so the
    # counters cover one point at a time: _persist hands them to the sidecar with
    # that point's row and resets them.
    with observe_triton_config_selection(_triton_config_namespaces()) as triton_config_observed:

        def _persist(num_tokens: int, latency: float, power_stats: dict, kernel_source: str) -> None:
            persisted = log_perf(
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
                        "distribution": distribution,
                        "latency": latency,
                    }
                ],
                framework="SGLang",
                version=installed_sglang_version,
                device_name=torch.cuda.get_device_name(device),
                op_name="moe",
                kernel_source=kernel_source,
                perf_filename=perf_filename,
                power_stats=power_stats,
            )
            if not persisted:
                raise RuntimeError("Failed to persist SGLang MoE performance row")
            point_observed = dict(triton_config_observed)
            for key in triton_config_observed:
                triton_config_observed[key] = 0
            if moe_backend == "triton":
                append_triton_config_record(
                    perf_filename,
                    {
                        "moe_dtype": moe_type,
                        "num_tokens": num_tokens,
                        "hidden_size": hidden_size,
                        "inter_size": inter_size,
                        "topk": topk,
                        "num_experts": num_experts,
                        "moe_tp_size": moe_tp_size,
                        "moe_ep_size": moe_ep_size,
                        "device_name": torch.cuda.get_device_name(device),
                    },
                    kernel_source,
                    point_observed,
                )

        _benchmark_moe_layer(
            moe_type=moe_type,
            moe_backend=moe_backend,
            num_tokens_list=num_tokens_list,
            on_result=_persist,
            hidden_size=hidden_size,
            inter_size=inter_size,
            topk=topk,
            num_experts=num_experts,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            distributed=distributed,
            power_law_alpha=power_law_alpha,
            activation=activation,
            is_gated=is_gated,
            has_bias=has_bias,
            gemm1_alpha=gemm1_alpha,
            gemm1_clamp_limit=gemm1_clamp_limit,
            swiglu_limit=swiglu_limit,
            scoring_func=scoring_func,
            routing_method_type=routing_method_type,
            routed_scaling_factor=routed_scaling_factor,
            renormalize=renormalize,
            has_correction_bias=has_correction_bias,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            apply_router_weight_on_input=apply_router_weight_on_input,
            is_fp4_experts=is_fp4_experts,
            int4_group_size=int4_group_size,
            architecture=moe_model_layer(model_name, hidden_size, inter_size, topk, num_experts).architecture,
            device=device,
        )

    restart_worker = torch.cuda.memory_allocated(device) > torch.cuda.get_device_properties(device).total_memory // 4
    if restart_worker:
        return WORKER_RESTART


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_moe_test_cases()
    for test_case in test_cases:
        print(test_case)
        run_moe_torch(*test_case, perf_filename=PerfFile.MOE)
