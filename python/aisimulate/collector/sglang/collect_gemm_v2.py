# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang 0.5.20 GEMM collector (serving-built linear layers).

At SGLang 0.5.20 the same raw kernel calls that collect_gemm_v1.py times no
longer match serving for several dtypes, so this version fork builds each
GEMM the way serving does -- a ``ReplicatedLinear`` whose quant method is
constructed from the checkpoint-style quant config -- and times
``layer.quant_method.apply`` after ``process_weights_after_loading``. The
framework therefore owns weight repacking, padding, fallbacks and kernel
selection; kernel_source records the branch the framework took.
"""

# Audited 2026-09-30 against sglang v0.5.20 (tag commit
# 94602c9c2b7cbdb8efd5c52802dac6a1c180089e, source clone) vs v0.5.17
# (29481685); runtime smoke on GB300 in the GLM-5.3-Flash W1 campaign.
# Dependency pins moved sglang-kernel 0.4.5->0.4.7, flashinfer
# 0.6.15.post1->0.6.18, sgl-deep-gemm 0.1.5.post1->0.2.0, torch 2.11->2.13,
# so rows are not cross-version comparable with collect_gemm_v1.py's.
# Serving initialization: the scheduler calls initialize_fp8_gemm_config,
# initialize_fp4_gemm_config and initialize_bf16_gemm_config
# (managers/scheduler.py:999-1001); _ensure_serving_gemm_config() does the
# same after publishing ServerArgs.
# bf16: UnquantizedLinearMethod.apply (quantization/unquant.py:460-499)
#   routes bf16 through _bf16_gemm_dispatch_impl when the bf16 backend is
#   cutedsl, which "auto" resolves to on SM100/103 (unquant.py:215-220).
#   The dispatch order is FlashInfer split-K/direct for the tuned (m,n,k)
#   table (:106-181,199-201,283-313, SGLANG_ENABLE_BF16_SPLITK_GEMM),
#   CuTe DSL TGV when use_cutedsl_bf16_gemm(m,n,k) (cutedsl_bf16_gemm.py:
#   1357-1387), else F.linear (unquant.py:315-350). collect_gemm_v1.py times
#   F.linear unconditionally, which is not serving truth on SM10x.
# fp8 (per-channel weight, dynamic per-token activation): apply_fp8_linear
#   (quantization/fp8_utils.py:1888-2060) quantizes with the 0.5.20 JIT
#   per_token_quant_fp8 kernel (kernels/ops/quantization/__init__.py:21-34,
#   76-84) and runs Fp8ScaledMMOp (kernels/ops/gemm/__init__.py:27-162: AOT
#   sgl_kernel, or torch._scaled_mm on SM90 for large M), a tuned Triton
#   tile when get_w8a8_channelwise_fp8_config has one, or triton_scaled_mm
#   when N or K is not a multiple of 16. The fused-op trace records which
#   backend ran.
# fp8_block: Fp8LinearMethod (quantization/fp8.py) with the auto runner
#   deepgemm_w8a8_block_fp8_linear_with_fallback when JIT DeepGEMM is on
#   (fp8_utils.py:799-810); shapes with N%64 or K%128 fall back to Triton
#   with fp32 scales (fp8_utils.py:1094-1107) and their scales are not
#   requantized to UE8M0 (fp8.py:758-770, model_loader/utils.py:265-291).
# nvfp4: ModelOptFp4LinearMethod (quantization/modelopt_quant.py:1706-2075)
#   pads N/K to 32 (pad_nvfp4_weight, :195-250), quantizes activations with
#   fp4_utils.fp4_quantize (FlashInfer "cute-dsl" backend on SM10x,
#   fp4_utils.py:26-49) and runs mm_fp4 with the auto fp4 runner
#   (fp4_utils.py:145-158: cute-dsl on SM100/103).
# The collect_gemm_v1.py FIXME(kernel-limit) "n<128 or k<128" is false at this
# version (fp8_block falls back to Triton; nvfp4 pads to 32), so this fork
# queues those shapes and lets the framework handle them.
__compat__ = "sglang==0.5.20"

import os
import random

# SGLang reads this flag while importing ``deep_gemm_wrapper.compile_utils``.
# Set it before any SGLang imports so a task compiles only its requested M.
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")

import pkg_resources
import torch
from collector.case_generator import get_gemm_case_specs
from collector.helper import benchmark_with_power, get_sm_version, log_perf

_SERVING_GEMM_CONFIG_READY = False


def _ensure_serving_gemm_config() -> None:
    """Publish dummy ServerArgs, then run serving's GEMM backend initializers."""
    global _SERVING_GEMM_CONFIG_READY
    if _SERVING_GEMM_CONFIG_READY:
        return
    from sglang.srt.runtime_context import get_server_args
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    try:
        get_server_args()  # raises ValueError when nothing is published (runtime_context.py:1031-1037)
    except ValueError:
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config

    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    initialize_bf16_gemm_config()
    _SERVING_GEMM_CONFIG_READY = True


def get_gemm_test_cases():
    sm_version = get_sm_version()
    if sm_version < 89:
        gemm_list = ["bfloat16"]
    elif sm_version < 90:
        gemm_list = ["bfloat16", "fp8"]
    elif sm_version < 100:
        gemm_list = ["fp8_block", "bfloat16", "fp8"]
    elif sm_version < 110:
        gemm_list = ["fp8_block", "bfloat16", "fp8", "nvfp4"]
    else:
        # SM120 dense fp8_block serving is CUTLASS (fp8_utils.py:832-833);
        # not collected here, unchanged from collect_gemm_v1.py.
        gemm_list = ["bfloat16", "fp8", "nvfp4"]

    requested_gemm_types = os.environ.get("AIC_COLLECT_GEMM_TYPES")
    if requested_gemm_types:
        requested = {item.strip() for item in requested_gemm_types.split(",") if item.strip()}
        gemm_list = [gemm_type for gemm_type in gemm_list if gemm_type in requested]

    test_cases = [[gemm_type, case.x, case.n, case.k] for case in get_gemm_case_specs() for gemm_type in gemm_list]
    # Group DeepGEMM JIT cache hits the same way collect_gemm_v1.py does.
    random.seed(42)
    random.shuffle(test_cases)
    return test_cases


def _quant_config(gemm_type: str):
    if gemm_type == "bfloat16":
        return None
    if gemm_type == "fp8_block":
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        return Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[128, 128])
    if gemm_type == "nvfp4":
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config

        return ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16, exclude_modules=[])
    raise ValueError(f"no serving quant config for {gemm_type!r}")


def _fill_checkpoint_tensors(layer: "torch.nn.Module", gemm_type: str) -> None:
    """Populate the checkpoint-native parameters create_weights allocated."""
    with torch.no_grad():
        for name, param in layer.named_parameters(recurse=False):
            data = param.data
            if data.dtype == torch.uint8:
                data.random_(0, 256)  # packed e2m1 nibbles: no NaN/Inf encodings
            elif data.dtype == torch.float8_e4m3fn:
                if "scale" in name:
                    data.copy_(torch.ones(data.shape, device=data.device).to(data.dtype))
                else:
                    data.copy_(((torch.rand(data.shape, device=data.device) - 0.5) * 2).to(data.dtype))
            elif data.is_floating_point():
                if "scale" in name:
                    data.copy_(torch.rand(data.shape, device=data.device) * 0.5 + 0.5)
                else:
                    data.copy_(torch.randn(data.shape, device=data.device) * 0.02)


def _build_linear(gemm_type: str, n: int, k: int, device: str):
    from sglang.srt.layers.linear import ReplicatedLinear

    layer = ReplicatedLinear(
        k, n, bias=False, params_dtype=torch.bfloat16, quant_config=_quant_config(gemm_type), prefix="collector_gemm"
    ).to(device)
    _fill_checkpoint_tensors(layer, gemm_type)
    layer.quant_method.process_weights_after_loading(layer)
    layer.requires_grad_(False)
    return layer


def _bf16_kernel_source(m: int, n: int, k: int) -> str:
    """Mirror the branch unquant.UnquantizedLinearMethod.apply takes (unquant.py:460-499)."""
    from sglang.srt.layers.quantization import unquant

    if not unquant.get_bf16_gemm_backend().is_cutedsl():
        return "sglang_torch_linear"
    if unquant._enable_bf16_splitk_gemm and unquant.use_bf16_splitk_gemm(m, n, k):
        return "sglang_flashinfer_bf16_direct" if unquant._prefer_direct(m, n, k) else "sglang_flashinfer_bf16_splitk"
    if unquant._use_cutedsl_bf16_gemm is not None and unquant._use_cutedsl_bf16_gemm(m, n, k):
        return "sglang_cutedsl_tgv_bf16"
    return "sglang_torch_linear"


def _fp8_block_kernel_source(layer, n: int, k: int) -> str:
    from sglang.srt.layers.quantization.fp8_utils import deepgemm_w8a8_block_fp8_linear_with_fallback

    runner = layer.quant_method.w8a8_block_fp8_linear
    if runner is deepgemm_w8a8_block_fp8_linear_with_fallback:
        # fp8_utils.py:1094-1107: DeepGEMM only for N%64==0 and K%128==0.
        if n % 64 == 0 and k % 128 == 0:
            return "sglang_deepgemm_gemm_nt_f8f8bf16"
        return "sglang_triton_w8a8_block_fp8"
    return f"sglang_{getattr(runner, '__name__', type(runner).__name__)}"


def _nvfp4_kernel_source() -> str:
    from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend

    backend = get_fp4_gemm_runner_backend().get_flashinfer_backend()
    return f"sglang_flashinfer_{str(backend).replace('-', '')}_nvfp4"


_FP8_SCALED_MM_TRACE_LABELS = {
    # Fp8ScaledMMOp backends (kernels/ops/gemm/__init__.py:75-162).
    "gemm.fp8_scaled_mm:aot": "sglang_sgl_kernel_fp8_scaled_mm",
    "gemm.fp8_scaled_mm:torch": "sglang_torch_scaled_mm",
}


def _fp8_kernel_source(traced: str) -> str:
    """Label apply_fp8_linear's GEMM from the fused-op trace.

    Fp8ScaledMMOp is the only fused op on this path; when it is absent the
    GEMM ran through triton_scaled_mm (misaligned N/K, or a tuned Triton tile,
    fp8_utils.py:2025-2053), which is not a fused op.
    """
    if not traced:
        return "sglang_triton_scaled_mm"
    if traced not in _FP8_SCALED_MM_TRACE_LABELS:
        raise RuntimeError(f"SGLang fp8 GEMM ran unexpected fused ops: {traced}")
    return _FP8_SCALED_MM_TRACE_LABELS[traced]


def _traced_fused_op_backends(func) -> str:
    """Run ``func`` once eagerly and report the fused-op backends that executed."""
    from sglang.kernels import fused_op

    fused_op.clear_fused_op_trace()
    fused_op.enable_fused_op_trace()
    try:
        func()
        torch.cuda.synchronize()
    finally:
        fused_op.disable_fused_op_trace()
    records = sorted({f"{record.op}:{record.backend}" for record in fused_op.get_fused_op_trace()})
    fused_op.clear_fused_op_trace()
    return "+".join(records)


def run_gemm(gemm_type, batch_size, N, K, *, perf_filename, device="cuda:0"):  # noqa: N803
    if gemm_type not in {"fp8_block", "fp8", "bfloat16", "nvfp4"}:
        raise ValueError(f"unsupported SGLang GEMM type {gemm_type!r}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    _ensure_serving_gemm_config()
    M = batch_size  # noqa: N806
    dtype = torch.bfloat16

    def create_gemm():
        x = torch.randn((M, K), dtype=dtype, device=device)
        if gemm_type == "fp8":
            from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear, cutlass_fp8_supported

            fp8_info = torch.finfo(torch.float8_e4m3fn)
            weight = (
                ((torch.rand(N, K, device=device) - 0.5) * 2 * fp8_info.max)
                .clamp(min=fp8_info.min, max=fp8_info.max)
                .to(torch.float8_e4m3fn)
                .t()
            )
            weight_scale = torch.rand((N, 1), device=device, dtype=torch.float32) + 0.5
            use_cutlass = cutlass_fp8_supported()

            def gemm_op():
                return apply_fp8_linear(x, weight, weight_scale, cutlass_fp8_supported=use_cutlass)

            return gemm_op
        layer = _build_linear(gemm_type, N, K, device)

        def gemm_op():
            return layer.quant_method.apply(layer, x)

        gemm_op.layer = layer
        return gemm_op

    op_list = []
    try:
        gpu_mem = torch.cuda.get_device_properties(device).total_memory
        persistent_bytes = M * K * 2 + M * N * 2
        transient_bytes = 0
        if gemm_type == "bfloat16":
            persistent_bytes += N * K * 2
        elif gemm_type in {"fp8", "fp8_block"}:
            persistent_bytes += N * K + M * K + N * 4 + M * 4
            transient_bytes = N * K * 4
        elif gemm_type == "nvfp4":
            persistent_bytes += int(N * K * 0.75)
            transient_bytes = N * K * 2
        budget = int(gpu_mem * 0.45)
        if persistent_bytes + transient_bytes >= budget:
            outside_loop_count = 1
        else:
            outside_loop_count = max(1, min(6, (budget - transient_bytes) // max(persistent_bytes, 1)))

        for _ in range(outside_loop_count):
            op_list.append(create_gemm())

        # Eager dry run: JIT-compiles outside graph capture and records the
        # fused-op backends that actually executed.
        traced = _traced_fused_op_backends(op_list[0])
        if gemm_type == "bfloat16":
            kernel_source = _bf16_kernel_source(M, N, K)
        elif gemm_type == "fp8_block":
            kernel_source = _fp8_block_kernel_source(op_list[0].layer, N, K)
        elif gemm_type == "nvfp4":
            kernel_source = _nvfp4_kernel_source()
        else:
            kernel_source = _fp8_kernel_source(traced)

        def kernel_func():
            for op in op_list:
                op()

        nvtx_tag = f"{gemm_type}_m{M}_n{N}_k{K}"
        torch.cuda.nvtx.range_push(nvtx_tag)
        try:
            with benchmark_with_power(
                device=device,
                kernel_func=kernel_func,
                num_warmups=3,
                num_runs=6,
                repeat_n=1,
            ) as results:
                pass
        finally:
            torch.cuda.nvtx.range_pop()

        if not log_perf(
            item_list=[
                {"gemm_dtype": gemm_type, "m": M, "n": N, "k": K, "latency": results["latency_ms"] / len(op_list)}
            ],
            framework="SGLang",
            version=pkg_resources.get_distribution("sglang").version,
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
