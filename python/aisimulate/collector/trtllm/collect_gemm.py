# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM GEMM collector.

Expands shared GEMM case specs into TRT-LLM Linear-layer benchmarks across BF16,
FP8, FP8 block, and related quantization modes. Shape policy comes from
`case_generator.py`/YAML; this file owns TRT-LLM layer construction, padding,
quantized-weight setup, and perf logging.
"""

__compat__ = "trtllm==1.4.0rc2"

import ctypes
import math
import time

import tensorrt_llm
import torch
import torch.nn.functional as F
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def pad_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _compute_fp8_block_weight_scales(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Compute per-block scales from weight tensor (shape: [n, k])."""
    n, k = weight.shape
    n_blocks = math.ceil(n / group_size)
    k_blocks = math.ceil(k / group_size)
    pad_n = n_blocks * group_size - n
    pad_k = k_blocks * group_size - k

    weight_abs = weight.abs()
    if pad_n or pad_k:
        weight_abs = F.pad(weight_abs, (0, pad_k, 0, pad_n))

    block_max = weight_abs.view(n_blocks, group_size, k_blocks, group_size).amax(dim=(1, 3))
    return (block_max / 448.0).clamp_min(1e-6).to(dtype=torch.float32)


# Per-worker-process weight cache for large shapes (outside_loop_count == 1).
# Keyed by (gemm_type, n, k).  Only the most recently used entry is kept so
# that GPU memory is not accumulated across (n, k) groups.
# One worker builds the Linear modules (and their weights) once per
# (gemm_type, n, k) and reuses them for every m that follows on this worker:
# the modules do not depend on m, and the per-case rebuild was the shard's
# cost — the sub-L2 shapes build up to 5 copies per case, and nvfp4 then
# unswizzled its block scales on the CPU (b300 base run, job 471159002:
# gemm/nvfp4 1.3 GPU-s/row and gemm/fp8_block 1.7 GPU-s/row against 0.04 for bf16).
# The shared GEMM plan (op_plans.OP_PLANS["gemm"]) orders cases by (n, k), then
# gemm_type, then m, so a worker sees runs of equal keys. The cache holds one
# key at a time, so peak memory is unchanged.
_gemm_op_cache: dict = {}

# Serving tunes every tunable op in one autotune() pass at warmup: a context
# forward of curr_max_num_tokens tokens (max_num_tokens, BaseLlmArgs default
# 8192 at llmapi/llm_args.py:5932-5933, unless the KV pool holds fewer), then a
# one-token generation forward (pyexecutor/engine/runners/decoder/warmup.py:
# 1050-1068, 1098-1104, 1106-1159). The collector runs the same two forwards
# under autotune() once per module key, on the first copy, and times every case
# outside it, as serving runs.
#
# One pass tunes every bucket an op's TuningConfig derives from the pass's M,
# not only the bucket of M itself: AutoTuner._optimization_profiles takes
# gen_tuning_buckets(M) plus M (autotuner.py:1746-1769), and choose_one
# profiles each one not cached yet (autotuner.py:1208-1234). For nvfp4 that is
# every power of 2 up to M (NVFP4GemmUnifiedRunner.tuning_config,
# custom_ops/torch_custom_ops.py:1286-1292; get_last_power_of_2_num_tokens_buckets,
# _torch/utils.py:411-418), across the allowed backends cutlass, cublaslt,
# cutedsl (SM100/103) and cuda_core (M <= 8) (modules/linear.py:4027-4030;
# torch_custom_ops.py:1355-1456); a case reads bucket
# last_positive_power_of_2(M) (utils.py:388-393, autotuner.py:1838-1843).
# fp8_block's quant-kernel choice on SM100/103 (Fp8QuantKernelRunner, no
# buckets, torch_custom_ops.py:2126-2137) is tuned at the two warmup M only
# and takes its CUDA fallback elsewhere, as in serving; its DeepGEMM and
# block-scaling runners have one tactic and tune only to drive JIT over
# deep_gemm_jit_warmup_buckets (torch_custom_ops.py:2156-2211, 2368-2392).
# bf16 and fp8 call no tunable op (F.linear, cublas_scaled_mm, cuda_scaled_mm).
# Serving runs an M above its warmup M only with max_num_tokens >= M, so such
# a case first gets one more pass at its own M, which tunes the buckets up to
# it as that warmup would.
_SERVING_WARMUP_NUM_TOKENS: tuple[int, ...] = (8192, 1)
# The largest M tuned for the cached key.
_gemm_tuned_num_tokens: dict = {}
# nvfp4 Linear's GEMM in the autotuner (nvfp4_gemm, torch_custom_ops.py:1588-1598).
_NVFP4_GEMM_OP: str = "trtllm::nvfp4_gemm::gemm"


def _autotune(op, k: int, num_tokens: tuple[int, ...], *, label: str, device, dtype) -> None:
    """Forward ``op`` at each M of ``num_tokens`` under autotune(), as serving's warmup pass does."""
    start = time.perf_counter()
    with autotune():
        for tokens in num_tokens:
            op.forward(torch.randn((tokens, k), dtype=dtype, device=device))
    torch.cuda.synchronize()
    print(f"TRT-LLM autotune {label} num_tokens={list(num_tokens)}: {time.perf_counter() - start:.1f} s", flush=True)


def _nvfp4_kernel_source(captured) -> str:
    """``nvfp4_<backend>`` for the backend the tuned nvfp4_gemm dispatch ran in ``captured``.

    AutoTuner.capture() records each choose_one call's op, runners, tuning
    config and inputs and otherwise dispatches as usual (autotuner.py:1136-1145,
    1981-2012). Outside tuning, choose_one runs the cached (runner, tactic) for
    the call's input sizes (autotuner.py:1147, 1163-1187); the same lookup
    (AutoTunerProfilingCache.search_cache, autotuner.py:504-534) names it.
    NVFP4GemmUnifiedRunner's tactics are (backend, sub_tactic) pairs
    (torch_custom_ops.py:1355-1510). A miss means the case would time the
    cache-miss fallback, so it raises instead of naming one.
    """
    tuner = AutoTuner.get()
    backends = set()
    for call in captured._captured_contexts:
        if call["custom_op"] != _NVFP4_GEMM_OP:
            continue
        sizes = tuple(tuner._get_input_sizes(call["inputs"]))
        hit, _, tactic, _ = tuner.profiling_cache.search_cache(
            call["custom_op"], call["runners"], sizes, call["tuning_config"]
        )
        if not hit:
            raise RuntimeError(f"{_NVFP4_GEMM_OP} has no tuned tactic for input sizes {sizes}")
        backends.add(tactic[0])
    if len(backends) != 1:
        raise RuntimeError(f"expected one tuned {_NVFP4_GEMM_OP} backend for the case, found {sorted(backends)}")
    return f"nvfp4_{backends.pop()}"


def _gemm_ops(gemm_type: str, n: int, k: int, *, quant_config, group_size, outside_loop_count: int, device, dtype, x):
    """The ``outside_loop_count`` Linear copies for (gemm_type, n, k), built and tuned on first use."""
    key = (gemm_type, n, k, outside_loop_count)
    label = f"gemm {gemm_type} n={n} k={k}"
    op_list = _gemm_op_cache.get(key)
    if op_list is None:
        _gemm_op_cache.clear()  # release the previous key's modules before building
        _gemm_tuned_num_tokens.clear()
        torch.cuda.empty_cache()
        op_list = []
        for _i in range(outside_loop_count):
            # nvfp4's per-tensor activation scale is a calibration constant in
            # serving; the first x of this key provides it, and the kernel's
            # timing does not depend on its value.
            weights = _build_weights(gemm_type, n, k, device, dtype, group_size, x)
            gemm = Linear(
                k,
                n,
                bias=False,
                dtype=dtype,
                quant_config=quant_config,
                force_dynamic_quantization=False,
            )
            gemm.load_weights([weights])
            # Serving runs post_load_weights on every module once its weights are
            # loaded (ModelLoader._walk_full_post_load, pyexecutor/model_loader.py:1898-1915).
            # It is a no-op for bf16 and fp8; fp8_block re-lays its weight scales for
            # DeepGEMM on SM100/103 and SM120 (FP8BlockScalesLinearMethod.transform_weights,
            # modules/linear.py:1517-1578), and nvfp4 pads N and K to multiples of 32
            # (NVFP4LinearMethod.transform_weights, modules/linear.py:2186-2229), so its
            # N < 32 cases run the padded GEMM that serving runs.
            gemm.post_load_weights()
            gemm.to(torch.device(device))
            op_list.append(gemm)
        # The copies share shapes, so the autotuner's cache entries cover them all.
        _autotune(op_list[0], k, _SERVING_WARMUP_NUM_TOKENS, label=label, device=device, dtype=dtype)
        _gemm_op_cache[key] = op_list
        _gemm_tuned_num_tokens[key] = max(_SERVING_WARMUP_NUM_TOKENS)
    m = x.shape[0]
    if m > _gemm_tuned_num_tokens[key]:
        _autotune(op_list[0], k, (m,), label=label, device=device, dtype=dtype)
        _gemm_tuned_num_tokens[key] = m
    return op_list


def _get_l2_cache_bytes(device_id: int = 0) -> int:
    """Query the GPU L2 cache size via the CUDA runtime API."""
    cuda_dev_attr_l2_cache_size = 38
    libcudart = ctypes.CDLL("libcudart.so")
    value = ctypes.c_int()
    ret = libcudart.cudaDeviceGetAttribute(ctypes.byref(value), cuda_dev_attr_l2_cache_size, device_id)
    if ret != 0 or value.value <= 0:
        raise RuntimeError(f"Failed to query L2 cache size (cudaError={ret}, value={value.value})")
    return value.value


def _build_weights(gemm_type: str, n: int, k: int, device, dtype, group_size, x) -> dict:
    """Allocate and return the weight tensors for one GEMM copy."""
    if gemm_type == "fp8":
        return {
            "weight": torch.randn((n, k), dtype=torch.bfloat16, device=device).to(dtype=torch.float8_e4m3fn),
            "weight_scale": torch.randn(1, dtype=torch.float32, device=device),
        }
    elif gemm_type == "fp8_block":
        fp_weight = torch.randn((n, k), dtype=torch.bfloat16, device=device)
        return {
            "weight": fp_weight.to(dtype=torch.float8_e4m3fn),
            "weight_scale": _compute_fp8_block_weight_scales(fp_weight, group_size),
        }
    elif gemm_type == "nvfp4":
        w = torch.randn((n, k), dtype=torch.bfloat16, device=device)
        w_sf_global = (448 * 6) / w.abs().max().float()
        w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global, 16, False)
        # load_weights takes a checkpoint's row-major block scales and swizzles
        # them itself (NVFP4LinearMethod.load_weight_scales, modules/linear.py:1847-1856),
        # while fp4_quantize returns them swizzled and padded to
        # [pad_up(n, 128), pad_up(k / 16, 4)], so undo the swizzle. The op runs on
        # the device; its CPU kernel is the same permutation (BlockScaleInterleaveReverse,
        # thop/fp4Op.cpp:305).
        w_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(w_sf_block.view(pad_up(n, 128), -1))
        x_sf_global = (448 * 6) / x.abs().max().float()
        return {
            "weight": w_fp4.cpu(),
            "weight_scale": w_sf_block_unswizzled.view(torch.float8_e4m3fn),
            "weight_scale_2": 1.0 / w_sf_global.cpu(),
            "input_scale": 1.0 / x_sf_global.cpu(),
        }
    else:  # bfloat16
        return {"weight": torch.randn((n, k), dtype=torch.bfloat16, device=device)}


def get_gemm_test_cases():
    return plan_cases("gemm", "trtllm", get_sm_version())


def run_gemm(gemm_type, m, n, k, *, perf_filename, device="cuda:0"):
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    sm_version = get_sm_version()

    # FIXME(kernel-limit): fp8_block weight scales are 128-column blocks, and the
    # CUTLASS block-scale GEMM needs K in whole blocks. fp8_block_scaling_gemm
    # checks it on SM89 and SM120 (TORCH_CHECK k % 128,
    # cpp/tensorrt_llm/thop/fp8BlockScalingGemm.cpp:99, 138) but only k % 16 on
    # SM90 (:170), where such a case then faults the worker and costs a worker
    # restart (about a minute) instead of a classified failure. Raised before
    # the build; the SM100/103 DeepGEMM path takes these shapes.
    if gemm_type == "fp8_block" and sm_version == 90 and k % 128 != 0:
        raise ValueError(
            f"FIXME(kernel-limit): fp8_block GEMM on SM90 needs K in whole 128-column scale blocks; got K={k}"
        )

    dtype = torch.bfloat16
    x = torch.randn((m, k), dtype=dtype).to(torch.device(device))

    if gemm_type == "fp8":
        group_size = None
        qc = QuantConfig(quant_algo=QuantAlgo.FP8)
    elif gemm_type == "fp8_block":
        group_size = 128
        qc = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=group_size)
    elif gemm_type == "nvfp4":
        # NVFP4 scale blocks are 16 elements (the weights above are quantized with
        # fp4_quantize(..., 16, ...)); NVFP4LinearMethod.resolve_scaling_vector_size
        # (modules/linear.py:1603-1611) rejects any other declared group_size.
        group_size = 16
        qc = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=group_size)
    else:
        qc = None
        group_size = None

    _l2_cache_bytes = _get_l2_cache_bytes(device.index or 0)
    _bytes_per_elem = {"bfloat16": 2, "fp8": 1, "fp8_block": 1, "nvfp4": 0.5}
    _weight_bytes = int(n * k * _bytes_per_elem[gemm_type])
    outside_loop_count = max(1, min(5, math.ceil(_l2_cache_bytes / _weight_bytes)))
    op_list = _gemm_ops(
        gemm_type,
        n,
        k,
        quant_config=qc,
        group_size=group_size,
        outside_loop_count=outside_loop_count,
        device=device,
        dtype=dtype,
        x=x,
    )
    # Deviation note (bf16 on SM100/103): the rows time F.linear, serving's default.
    # With pipeline_parallel_size > 1 serving switches bf16 Linears to the CuTe DSL
    # persistent GEMM (TorchLlmArgs.validate_cute_dsl_bf16, llmapi/llm_args.py:7996-8003;
    # UnquantizedLinearMethod.apply, modules/linear.py:631-652); the GEMM key has
    # no PP field, so PP > 1 deployments read the F.linear rows.
    #
    # Dry run at this m (lazy init / JIT) before graph capture. It is captured so
    # the nvfp4 label names the backend the tuned dispatch ran; nothing tunes
    # between it and the timed forwards, so they take the same cached tactic.
    with AutoTuner.get().capture() as captured:
        for op in op_list:
            op.forward(x)

    # fp8_block dispatch (FP8BlockScalesLinearMethod.apply, modules/linear.py:1351-1399)
    # is is_sm_100f-scoped, not sm>=100: SM100/103 run DeepGEMM through
    # fp8_swap_ab_gemm unless the Linear opts into CuTe DSL or out of DeepGEMM
    # (use_cute_dsl_blockscaling_mm and disable_deep_gemm, both off by default in
    # Linear and TorchLlmArgs); SM120 runs per_token_quant_and_transform +
    # fp8_block_scaling_gemm, and SM89/90 fp8_quantize_1x128 + fp8_block_scaling_gemm,
    # both labeled torch_flow. Resolved before timing so a miss fails fast.
    if gemm_type == "nvfp4":
        kernel_source = _nvfp4_kernel_source(captured)
    elif gemm_type == "fp8_block" and is_sm_100f(sm_version):
        kernel_source = "deepgemm"
    else:
        kernel_source = "torch_flow"

    def kernel_func():
        for op in op_list:
            op.forward(x)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        repeat_n=1,  # Already repeating inside kernel_func via op_list
    ) as results:
        pass

    log_perf(
        item_list=[
            {"gemm_dtype": gemm_type, "m": m, "n": n, "k": k, "latency": results["latency_ms"] / outside_loop_count}
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name="gemm",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_gemm_test_cases():
        run_gemm(*test_case, perf_filename=PerfFile.GEMM)
