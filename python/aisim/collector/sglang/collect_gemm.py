# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang GEMM collector.

Benchmarks the SGLang linear-layer GEMM paths for shared GEMM case specs: BF16,
FP8 (per-channel weight, dynamic per-token activation), FP8 block and NVFP4.
Each dtype runs through the function SGLang's linear method calls in serving,
after the backend configuration the scheduler installs, and records the kernel
that function ran. The case plan is shared with the other backends
(``op_plans.OP_PLANS["gemm"]``).
"""

# Source-checked against the sglang v0.5.23 tag (sglang-kernel 0.4.10,
# sgl-deep-gemm 0.2.3.post2, flashinfer-python 0.7.2, torch 2.14.1). Citations
# below are python/sglang/... paths at that tag. One release per file; earlier
# code is `git log -- <this file>` away.
#
# Where this table times a different path than SGLang serving (the direction of
# each error is unknown):
# - fp8 on SM120: the table times the per-channel layout serving builds when it
#   quantizes a BF16 checkpoint online (srt/layers/quantization/fp8.py:1151-1175)
#   or loads a per-channel FP8 checkpoint. An FP8-serialized per-tensor
#   Fp8Config checkpoint is requantized to one per-tensor scale on SM12x
#   (fp8.py:1194-1253), and apply_fp8_linear then runs torch._scaled_mm: with
#   per-tensor scales for a static activation scale
#   (srt/layers/quantization/fp8_utils.py:2754-2766), unscaled with an unfused
#   dequant for a dynamic one (:2782-2791, _apply_fallback_scaled_mm :2442-2472).
#   Retire it by giving per-tensor checkpoints their own gemm dtype.
# - fp8_block on SM90 for checkpoints whose quantization_config has
#   scale_fmt "ue8m0" (DeepSeek-V3.2 and the [128, 128]-block DeepSeek-V4
#   checkpoints): Fp8LinearMethod dispatches with act_scale_ue8m0 from the
#   checkpoint config (fp8.py:548-552, fp8_utils.py:748-749), and the DeepGEMM
#   runner then quantizes activations with row-major power-of-two fp32 scales
#   (fp8_utils.py:1559-1564), which runs the per_token_group_quant_8bit_v2
#   kernel (kernels/ops/quantization/fp8_kernel.py:435-452) and leaves DeepGEMM
#   to re-lay the scales for TMA. The table dispatches as a checkpoint without
#   ue8m0 scales (column-major TMA-aligned fp32 scales, fp8_utils.py:1565-1572).
#   The gemm key has no checkpoint dimension, so only a scale-format key could
#   retire it.
__compat__ = "sglang==0.5.23"

import contextlib
import os

# SGLang reads this flag while importing ``deep_gemm_wrapper.compile_utils``
# (srt/layers/deep_gemm_wrapper/compile_utils.py:43; srt/environ.py:1369). Set it
# before any SGLang imports so a task compiles only its requested M.
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")

# setuptools/pkg_resources is absent from recent framework images.
from importlib.metadata import version as _dist_version

import torch
from collector.op_plans import plan_cases

try:
    from flashinfer import autotune_v2 as flashinfer_autotune_v2
    from flashinfer import fp4_quantize as flashinfer_fp4_quantize
    from flashinfer import mm_fp4 as flashinfer_mm_fp4

    HAS_FLASHINFER_FP4 = True
except ImportError:
    HAS_FLASHINFER_FP4 = False

from collector.helper import (
    benchmark_with_power,
    flashinfer_autotuned,
    flashinfer_tuning_bucket,
    get_sm_version,
    log_perf,
)
from collector.sglang.runtime_limits import decode_autotune_max_tokens
from sglang.kernels.fused_op import (
    clear_fused_op_trace,
    disable_fused_op_trace,
    enable_fused_op_trace,
    get_fused_op_trace,
)
from sglang.kernels.ops.gemm import try_qwen3x_nvfp4_gemm
from sglang.srt.layers.quantization import unquant
from sglang.srt.layers.quantization.fp4_utils import fp4_quantize as sglang_fp4_quantize
from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend, initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import (
    _deepgemm_block_fp8_supported,
    apply_fp8_linear,
    cutlass_fp8_supported,
    cutlass_w8a8_block_fp8_linear_with_fallback,
    deepgemm_w8a8_block_fp8_linear_with_fallback,
    dispatch_w8a8_block_fp8_linear,
    initialize_fp8_gemm_config,
    requant_block_scale_ue8m0_for_deepgemm,
    triton_w8a8_block_fp8_linear,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight,
    slice_nvfp4_output,
)

_BLOCK_SIZE = [128, 128]
_SERVING_GEMM_CONFIG_READY = False


def get_gemm_test_cases():
    return plan_cases("gemm", "sglang", get_sm_version())


def _ensure_serving_gemm_config() -> None:
    """Install the GEMM backend configuration the scheduler installs before loading a model.

    Scheduler.__init__ publishes its ServerArgs and then calls
    initialize_fp8_gemm_config, initialize_fp4_gemm_config and
    initialize_bf16_gemm_config (srt/managers/scheduler.py:1010-1013;
    bench_one_batch does the same, benchmark/one_batch.py:119-129). A default
    ServerArgs leaves --fp8-gemm-backend, --fp4-gemm-backend and
    --bf16-gemm-backend at "auto" (srt/arg_groups/fields/exec_.py:273-298), so
    each initializer resolves the backend serving resolves on this SM. A record
    already published in this process is kept (get_server_args raises until one
    is, srt/runtime_context.py:1023-1029, 1289-1290), as publish would replace
    it (:1489-1515).
    """
    global _SERVING_GEMM_CONFIG_READY
    if _SERVING_GEMM_CONFIG_READY:
        return
    from sglang.srt.runtime_context import get_server_args, publish
    from sglang.srt.server_args import ServerArgs

    try:
        get_server_args()
    except ValueError:
        publish(ServerArgs(model_path="dummy"), role="scheduler")
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    unquant.initialize_bf16_gemm_config()
    _SERVING_GEMM_CONFIG_READY = True


def _decode_autotune_max_tokens() -> int:
    """The largest token count SGLang's default warm-up tunes FlashInfer GEMMs at.

    The decode warm-up dummy's batch size (runtime_limits.decode_autotune_max_tokens)
    for the tp_size 1 of the ServerArgs _ensure_serving_gemm_config publishes.
    FlashInfer profiles mm_fp4 at the hybrid buckets up to that dummy's m
    (flashinfer 0.7.2 autotuner/autotuner.py:3282-3285, gemm/gemm_base.py:9073-9101).
    """
    return decode_autotune_max_tokens(tp_size=1)


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def scale_shape(shape, group_shape):
    assert len(shape) == len(group_shape)
    return tuple(cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


def _prepare_fp8_block_weights(N, K, device, linear):  # noqa: N803
    """Build the fp8_block weight and its block scale once, as serving loads them.

    The [N, K] FP8 weight and its fp32 [N/128, K/128] block scale are what a
    checkpoint without ue8m0 scales loads. They are then repacked the way
    ``Fp8LinearMethod.process_weights_after_loading_block_quant`` does
    (srt/layers/quantization/fp8.py:743-747, 835-844):
    ``requant_block_scale_ue8m0_for_deepgemm`` (fp8_utils.py:2147-2181)
    requantizes them to UE8M0 only when the dispatched linear is the DeepGEMM
    runner and DeepGEMM casts scales and can run this shape
    (``should_deepgemm_weight_requant_ue8m0``, model_loader/utils.py:266-292);
    otherwise the fp32 scales stay for the Triton or CUTLASS kernel. This runs
    at setup, never inside the timed call.
    """
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    b_fp32 = (torch.rand(N, K, device=device) - 0.5) * 2 * fp8_info.max
    b_fp8 = b_fp32.clamp(min=fp8_info.min, max=fp8_info.max).to(torch.float8_e4m3fn)
    del b_fp32
    scale_b = torch.randn(scale_shape((N, K), _BLOCK_SIZE), device=device, dtype=torch.float32)
    weight = torch.nn.Parameter(b_fp8, requires_grad=False)
    weight_scale_inv = torch.nn.Parameter(scale_b, requires_grad=False)
    requant_block_scale_ue8m0_for_deepgemm(
        weight,
        weight_scale_inv,
        _BLOCK_SIZE,
        use_deepgemm_runner=getattr(linear, "func", linear) is deepgemm_w8a8_block_fp8_linear_with_fallback,
        output_dtype=torch.bfloat16,
        weight_shape=weight.shape,
    )
    return weight, weight_scale_inv


def _fp8_block_kernel_source(linear, n: int, k: int) -> str:
    """Name the kernel the dispatched block-FP8 linear runs for an [n, k] weight.

    The DeepGEMM runner runs DeepGEMM when ``_deepgemm_block_fp8_supported``
    holds for the weight and Triton otherwise (fp8_utils.py:1491-1494,
    1517-1538); the CUTLASS runner (SM120) runs CUTLASS for N and K multiples
    of 128 and Triton otherwise (fp8_utils.py:1389, 1410-1414).
    """
    runner = getattr(linear, "func", linear)
    if runner is deepgemm_w8a8_block_fp8_linear_with_fallback:
        if _deepgemm_block_fp8_supported((n, k), _BLOCK_SIZE, torch.bfloat16):
            return "sglang_deepgemm_gemm_nt_f8f8bf16"
        return "sglang_triton_w8a8_block_fp8"
    if runner is cutlass_w8a8_block_fp8_linear_with_fallback:
        if n % 128 == 0 and k % 128 == 0:
            return "sglang_cutlass_w8a8_block_fp8"
        return "sglang_triton_w8a8_block_fp8"
    if runner is triton_w8a8_block_fp8_linear:
        return "sglang_triton_w8a8_block_fp8"
    raise RuntimeError(f"SGLang dispatched an unexpected block-FP8 linear {runner!r}")


def _bf16_kernel_source(m: int, n: int, k: int) -> str:
    """Name the kernel UnquantizedLinearMethod.apply runs for an [m, k] x [n, k] GEMM.

    Mirrors its branch order: F.linear unless the bf16 backend is cutedsl or
    gemv (srt/layers/quantization/unquant.py:597-614); in
    ``_bf16_gemm_dispatch_impl`` (unquant.py:421-463) the tuned split-K table
    first, as warp split-K, direct or split-K (``_bf16_splitk_gemm_out``,
    unquant.py:384-409), then the CuTe DSL TGV kernel where
    ``use_cutedsl_bf16_gemm`` picks it (kernels/ops/gemm/cutedsl_bf16_gemm.py:1376-1419),
    else F.linear. The predicates are the module state
    ``initialize_bf16_gemm_config`` installed (unquant.py:298-375).
    """
    backend = unquant.get_bf16_gemm_backend()
    if backend.is_gemv():
        raise RuntimeError("SGLang bf16 GEMM backend 'gemv' is not the serving default this table times")
    if not backend.is_cutedsl():
        return "sglang_torch_linear"
    if unquant._enable_bf16_splitk_gemm and unquant.use_bf16_splitk_gemm(m, n, k):
        if (m, n, k) in unquant._BF16_WARP_SPLITK_TUNED_TACTICS:
            return "sglang_flashinfer_bf16_warp_splitk"
        if unquant._prefer_direct(m, n, k):
            return "sglang_flashinfer_bf16_direct"
        return "sglang_flashinfer_bf16_splitk"
    if unquant._use_cutedsl_bf16_gemm is not None and unquant._use_cutedsl_bf16_gemm(m, n, k):
        return "sglang_cutedsl_tgv_bf16"
    return "sglang_torch_linear"


def _fp8_linear_kernel_source(gemm_op) -> str:
    """Name the GEMM apply_fp8_linear runs for this case, from one eager call.

    With a per-channel weight scale, apply_fp8_linear (fp8_utils.py:2641-2683)
    runs either the ``gemm.fp8_scaled_mm`` fused op or ``triton_scaled_mm``
    (a weight CUTLASS cannot take, N or K not a multiple of 16, or a shape with
    a tuned per-GPU Triton config, kernels/ops/gemm/configs/*channelwise*).
    The fused op runs sgl-kernel CUTLASS (AOT) unless, on SM90, Torch's NVJet
    kernel is preferred for the shape (``_prefer_torch_rowwise_fp8``,
    kernels/ops/gemm/__init__.py:30-75, 78-107). Its call trace
    (kernels/fused_op.py:252-272, 570-585) names the backend that ran; a call
    without a trace record ran triton_scaled_mm.
    """
    clear_fused_op_trace()
    enable_fused_op_trace()
    try:
        gemm_op()
        backends = {record.backend for record in get_fused_op_trace() if record.op == "gemm.fp8_scaled_mm"}
    finally:
        disable_fused_op_trace()
        clear_fused_op_trace()
    if not backends:
        return "sglang_triton_scaled_mm"
    if backends == {"aot"}:
        return "sglang_sgl_kernel_fp8_scaled_mm"
    if backends == {"torch"}:
        return "sglang_torch_scaled_mm"
    raise RuntimeError(f"SGLang fp8_scaled_mm ran on unexpected backends {sorted(backends)}")


def run_gemm(gemm_type, batch_size, N, K, *, perf_filename, device="cuda:0"):  # noqa: N803
    assert gemm_type in [
        "fp8_block",
        "fp8",
        "bfloat16",
        "nvfp4",
    ], "not support gemm type"
    torch.cuda.set_device(device)
    _ensure_serving_gemm_config()
    M = batch_size  # noqa: N806
    sm_version = get_sm_version()
    fp4_backend = None
    block_linear = None
    # The NVFP4 GEMM each eager call of gemm_op ran (graph replays run no Python).
    nvfp4_ran = set()
    if gemm_type == "fp8" and not cutlass_fp8_supported():
        # Fp8LinearMethod keeps a per-channel weight only where CUTLASS FP8 is
        # supported (fp8.py:1151-1175, fp8_utils.py:397-406); elsewhere it
        # quantizes per tensor, a layout this table does not time.
        raise RuntimeError(f"SGLang per-channel FP8 linear needs CUTLASS FP8 support, absent on SM{sm_version}")
    if gemm_type == "fp8_block":
        # Fp8LinearMethod.__init__ picks the linear once from the block size and
        # the checkpoint's scale format (fp8.py:548-552 -> fp8_utils.py:704-750,
        # under the --fp8-gemm-backend _ensure_serving_gemm_config resolved).
        block_linear = dispatch_w8a8_block_fp8_linear(weight_block_size=_BLOCK_SIZE, act_scale_ue8m0=False)
    if gemm_type == "nvfp4":
        if not HAS_FLASHINFER_FP4 or sglang_fp4_quantize is None:
            raise RuntimeError("SGLang NVFP4 GEMM requires FlashInfer FP4 quantization support")
        # The FP4 runner initialize_fp4_gemm_config resolved (fp4_utils.py:149-170:
        # flashinfer_cutedsl on SM100/103, marlin on SM80-90, flashinfer_cutlass
        # otherwise) under the mm_fp4 backend name serving passes
        # (get_flashinfer_backend, fp4_utils.py:128-143).
        runner = get_fp4_gemm_runner_backend()
        if not (runner.is_flashinfer_cutedsl() or runner.is_flashinfer_cutlass()):
            raise ValueError(f"SGLang serves NVFP4 dense GEMMs on SM{sm_version} with {runner.value}, not FP4 mm_fp4")
        fp4_backend = runner.get_flashinfer_backend()

    def create_gemm():
        dtype = torch.bfloat16
        if gemm_type == "nvfp4":
            # Prepare source data: Activation A [M, K] in BF16
            a_bf16 = torch.randn((M, K), device=device, dtype=dtype)

            # Prepare Weight B [N, K] and its dummy scale
            b_bf16_dummy = torch.randn((N, K), device=device, dtype=dtype)

            # Global scales
            a_global_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
            b_global_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
            alpha = 1.0 / (a_global_scale * b_global_scale)

            # Weight preparation (setup only, untimed). The CuTe DSL and CUTLASS
            # backends both consume the modelopt swizzled scale layout; the
            # shuffle_matrix_a/sf_a layout belongs to the TRTLLM backend. Serving
            # hands the GEMM float8_e4m3fn weight scales
            # (srt/layers/quantization/modelopt_quant.py:2117).
            b_fp4, b_sf = flashinfer_fp4_quantize(
                b_bf16_dummy,
                b_global_scale,
                is_sf_swizzled_layout=True,
            )
            del b_bf16_dummy
            # ModelOptFp4LinearMethod.process_weights_after_loading pads the packed
            # weight's N and K to FP4_GEMM_ALIGNMENT (32) for these backends
            # (modelopt_quant.py:1991-1993, pad_nvfp4_weight :205-252); the swizzled
            # scale is already 128 x 4 padded, as serving's interleave pads it
            # (:2000-2027). apply() then pads the activation by the same K columns
            # and slices the N padding off the output (:2120-2140).
            b_fp4, weights_padding_cols = pad_nvfp4_weight(b_fp4)
            w_n = b_fp4.shape[0]
            b_fp4 = b_fp4.t()
            b_sf = b_sf.view(torch.float8_e4m3fn).t()

            def gemm_op():
                # ModelOptFp4LinearMethod.apply (modelopt_quant.py:2076-2160):
                # serving's fp4_quantize quantizes the activation with FlashInfer's
                # CuTe DSL backend on SM100/103 and its CUDA backend elsewhere
                # (fp4_utils.py:22-90), then fp4_gemm (modelopt_quant.py:136-185)
                # tries the shape-specialized SM120 KDA kernel before mm_fp4
                # (kernels/kda_kernels/qwen3x_nvfp4_gemm.py:13-22, 94-110; it
                # returns None off SM120 and for other shapes).
                a_fp4_dynamic, a_sf_dynamic = sglang_fp4_quantize(a_bf16, a_global_scale)
                a_fp4_dynamic = pad_nvfp4_activation_for_cutlass(a_fp4_dynamic, weights_padding_cols)
                kda_out = try_qwen3x_nvfp4_gemm(a_fp4_dynamic, b_fp4, a_sf_dynamic, b_sf, alpha, dtype, w_n)
                if kda_out is not None:
                    nvfp4_ran.add("kda")
                    return slice_nvfp4_output(kda_out, N)
                nvfp4_ran.add(fp4_backend)
                # fp4_gemm (modelopt_quant.py:136-185) calls mm_fp4 without an output buffer
                out = flashinfer_mm_fp4(a_fp4_dynamic, b_fp4, a_sf_dynamic, b_sf, alpha, dtype, backend=fp4_backend)
                return slice_nvfp4_output(out, N)

            return gemm_op

        elif gemm_type == "fp8_block":
            a_bf16 = torch.randn(M, K, dtype=dtype, device=device)
            b_fp8, scale_b = _prepare_fp8_block_weights(N, K, device, block_linear)

            def gemm_op():
                # Fp8LinearMethod.apply for an unquantized activation
                # (fp8.py:1427-1435): the dispatched linear quantizes the
                # activation and runs its GEMM.
                return block_linear(
                    input=a_bf16,
                    weight=b_fp8,
                    block_size=_BLOCK_SIZE,
                    weight_scale=scale_b,
                    input_scale=None,
                    bias=None,
                )

            return gemm_op

        elif gemm_type == "fp8":
            # Fp8LinearMethod with a dynamic activation scheme and a per-channel
            # weight: layer.weight is the [K, N] column-major view of the [N, K]
            # FP8 weight and weight_scale is [N, 1] fp32 (fp8.py:1151-1175;
            # convert_to_channelwise, srt/layers/quantization/utils.py:141-161).
            # apply() hands both to apply_fp8_linear (fp8.py:1466-1474), which
            # quantizes per token (JIT per_token_quant_fp8, fp8_utils.py:2625) and
            # picks the GEMM per shape (see _fp8_linear_kernel_source).
            fp8_info = torch.finfo(torch.float8_e4m3fn)
            a_bf16 = torch.randn(M, K, dtype=dtype, device=device)
            b_fp32 = (torch.rand(N, K, device=device) - 0.5) * 2 * fp8_info.max
            b_fp8 = b_fp32.clamp(min=fp8_info.min, max=fp8_info.max).to(torch.float8_e4m3fn).t()
            del b_fp32
            scale_b = torch.randn((N, 1), device=device, dtype=torch.float32)

            def gemm_op():
                return apply_fp8_linear(
                    input=a_bf16,
                    weight=b_fp8,
                    weight_scale=scale_b,
                    input_scale=None,
                    bias=None,
                    # run_gemm checked cutlass_fp8_supported(), which
                    # Fp8LinearMethod passes here (fp8.py:510, 1472).
                    cutlass_fp8_supported=True,
                    use_per_token_if_dynamic=False,
                )

            return gemm_op

        elif gemm_type == "bfloat16":
            # UnquantizedLinearMethod holds the [N, K] bf16 weight as a frozen
            # Parameter (create_weights, unquant.py:546-566) and apply() routes
            # it by the bf16 backend (see _bf16_kernel_source).
            a_bfloat16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            layer = torch.nn.Module()
            layer.register_parameter(
                "weight",
                torch.nn.Parameter(torch.randn(N, K, dtype=torch.bfloat16, device=device), requires_grad=False),
            )
            method = unquant.UnquantizedLinearMethod()

            def gemm_op():
                return method.apply(layer, a_bfloat16)

            return gemm_op

    op_list = []
    try:
        # Scale loop count down for large matrices to avoid OOM. FP8 setup briefly
        # materializes FP32 tensors before retaining quantized weights, so account
        # for both persistent and transient pressure instead of only steady state.
        _gpu_mem = torch.cuda.get_device_properties(device).total_memory
        _persistent_bytes = M * K * 2 + M * N * 2
        _transient_bytes = 0
        if gemm_type == "bfloat16":
            _persistent_bytes += N * K * 2
        elif gemm_type == "fp8":
            _persistent_bytes += N * K + M * K + N * 4 + M * 4
            _transient_bytes = N * K * 4
        elif gemm_type == "fp8_block":
            _persistent_bytes += N * K + cdiv(N, 128) * cdiv(K, 128) * 4
            _transient_bytes = N * K * 4
        elif gemm_type == "nvfp4":
            _persistent_bytes += int(N * K * 0.75)
            _transient_bytes = N * K * 2

        _budget = int(_gpu_mem * 0.45)
        if _persistent_bytes + _transient_bytes >= _budget:
            outside_loop_count = 1
        else:
            outside_loop_count = max(1, min(6, (_budget - _transient_bytes) // max(_persistent_bytes, 1)))

        for _ in range(outside_loop_count):
            op_list.append(create_gemm())

        kernel_source = None
        if gemm_type == "fp8":
            kernel_source = _fp8_linear_kernel_source(op_list[0])
        elif gemm_type == "fp8_block":
            kernel_source = _fp8_block_kernel_source(block_linear, N, K)
        elif gemm_type == "bfloat16":
            kernel_source = _bf16_kernel_source(M, N, K)

        def kernel_func():
            for op in op_list:
                op()

        tune = contextlib.nullcontext()
        if gemm_type == "nvfp4":
            # Serving autotunes the dense FP4 GEMM of modelopt_fp4 checkpoints on
            # the FlashInfer cutlass and CuTe DSL runners at warm-up
            # (should_run_flashinfer_autotune,
            # srt/model_executor/runner/flashinfer_autotune.py:77-161, fp4 gate at
            # :135-138; base_runner.py:279-285). No op is skipped by default:
            # FLASHINFER_AUTOTUNE_WORKAROUND_SKIPS is empty (:50) and
            # --flashinfer-autotune-skip-ops defaults to None
            # (srt/arg_groups/fields/exec_.py:398-406). It tunes under
            # autotune(True) plus autotune_v2(mode="tune", persistent_cache=False)
            # (flashinfer_autotune.py:354-362), where FlashInfer's default tactic
            # races the candidates; the tuning pass here enters the same v2
            # context. mm_fp4 is the only tuner-owned op in gemm_op: fp4_quantize
            # and the KDA kernel run unchanged. The warm-up dummy covers the
            # hybrid buckets up to the decode batch size
            # (_decode_autotune_max_tokens), and mm_fp4 maps a runtime M up to
            # the next bucket (flashinfer 0.7.2 gemm/gemm_base.py:9073-9101,
            # fused_moe/utils.py:309-324), so M is timed on that
            # bucket's tactic. Past the last tuned bucket serving has no entry
            # and runs mm_fp4's fallback tactic, so those rows are timed untuned.
            tune_m = flashinfer_tuning_bucket(M, _decode_autotune_max_tokens())
            if tune_m is not None:

                def tune_fn():
                    with flashinfer_autotune_v2(mode="tune", persistent_cache=False, tuning_buckets=(tune_m,)):
                        kernel_func()

                tune = flashinfer_autotuned(tune_m, tune_fn, label=f"gemm nvfp4 n={N} k={K}")

        # Use benchmark_with_power context manager
        nvtx_tag = f"{gemm_type}_m{M}_n{N}_k{K}"
        torch.cuda.nvtx.range_push(nvtx_tag)

        try:
            with (
                tune,
                benchmark_with_power(
                    device=device,
                    kernel_func=kernel_func,
                    num_warmups=3,
                    num_runs=6,
                    repeat_n=1,
                ) as results,
            ):
                pass
        finally:
            torch.cuda.nvtx.range_pop()

        if gemm_type == "nvfp4":
            if nvfp4_ran == {"kda"}:
                kernel_source = "sglang_kda_qwen3x_nvfp4"
            elif nvfp4_ran == {"cute-dsl"}:
                kernel_source = "sglang_flashinfer_cutedsl_nvfp4"
            elif nvfp4_ran == {"cutlass"}:
                kernel_source = "sglang_flashinfer_cutlass_nvfp4"
            else:
                raise RuntimeError(f"SGLang NVFP4 GEMM ran on unexpected kernels {sorted(nvfp4_ran)}")
        if not log_perf(
            item_list=[
                {"gemm_dtype": gemm_type, "m": M, "n": N, "k": K, "latency": results["latency_ms"] / len(op_list)}
            ],
            framework="SGLang",
            version=_dist_version("sglang"),
            device_name=torch.cuda.get_device_name(device),
            op_name="gemm",
            kernel_source=kernel_source,
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        ):
            raise RuntimeError(f"Failed to persist SGLang GEMM performance row to {perf_filename}")
    finally:
        op_list.clear()
        torch.cuda.empty_cache()
