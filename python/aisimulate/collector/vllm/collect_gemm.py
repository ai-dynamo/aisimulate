# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# vLLM owns block-FP8 dispatch. Its FlashInfer/DeepGEMM dynamic wrapper
# (FlashInfer for m < 32, DeepGEMM above) is selectable on SM90 only
# (has_flashinfer_fp8_blockscale_gemm, utils/flashinfer.py:1177-1193
# @v0.30.0); SM10x selects DeepGEMM or CUTLASS directly (see the
# __compat__ comment below). Keep every grid M/N/K shape observable instead
# of hiding a presumed M-divisibility restriction in population logic.

"""vLLM GEMM collector for CUDA backends.

Builds vLLM RowParallelLinear layers with synthetic weights to benchmark BF16,
FP8, FP8 block, and FP4-style paths where available. Shared GEMM shapes come
from `case_generator.py`; this file handles vLLM config contexts, distributed setup,
quantized-weight preparation, and selected-kernel reporting.
"""

# Verified 2026-08-21 against vLLM v0.24.0 and v0.27.1 tags (source clones,
# no runtime GPU available in this environment) for AIC-1782 Task V1. Every
# framework surface this collector touches is either byte-identical or a
# pure line-number shift between the two tags -- no version branching is
# needed, unlike the sglang gemm precedent (AIC-1762 Task 4c/4d), where the
# same call shape silently dispatched to a different kernel at the newer
# pin. Checked: vllm.envs.VLLM_BATCH_INVARIANT (unchanged flag; its two
# reroute sites shifted line-only, see the guard's comment below);
# vllm._custom_ops.scaled_fp4_quant (op registration unchanged);
# vllm.config.{VllmConfig,set_current_vllm_config} (already a package with
# these two names re-exported at both versions); FlashInferFp8DeepGEMM
# DynamicBlockScaledKernel (scaled_mm/flashinfer.py -- file byte-identical,
# same class, same m>=32 DeepGEMM/FlashInfer .fallback/.base split);
# RowParallelLinear (byte-identical class body, same __init__ kwargs);
# CompressedTensorsConfig.from_config's nvfp4-pack-quantized parsing path
# and compressed_tensors_w4a4_nvfp4.py's process_weights_after_loading
# (byte-identical file); Fp8Config and its fp8/fp8_block kernel-candidate
# lists (model_executor/kernels/linear/__init__.py) -- 0.27.1 appends a new
# "Humming" kernel family to the END of every list this collector's lanes
# consult (_POSSIBLE_FP8_KERNELS[CUDA] and _POSSIBLE_FP8_BLOCK_KERNELS[CUDA]
# both keep their pre-existing entries, including
# FlashInferFp8DeepGEMMDynamicBlockScaledKernel, ahead of the Humming
# addition -- selection is first-match by is_supported(), so today's
# selected kernels for fp8/fp8_block are unaffected); vllm.utils.deep_gemm
# (per_block_cast_to_fp8 -- file byte-identical in its entirety, both
# clones). The fp8_block SM120 FIXME(kernel-limit) below still holds: its
# vllm-side citations (platforms/cuda.py's support_deep_gemm,
# utils/deep_gemm.py's should_use_deepgemm_for_fp8_linear) are unchanged in
# content at 0.27.1 (only support_deep_gemm's def line shifted 663->665);
# the DeepGEMM/CUTLASS-side citations (csrc/apis/layout.hpp,
# cutlass_gemm_caller.cuh) live outside this vllm clone and were not
# re-derived, but nothing on the vllm side suggests the upstream gap closed.
# 0.25.0 collection audit against the installed official runtime, vLLM
# dd10e03f95f94edbea1975c67ace3a35ec9a8a40.
# RowParallelLinear's class body is AST-identical to 0.24.0; Fp8LinearMethod
# changes only type annotations. scaled_mm/flashinfer.py is byte-identical,
# preserving its m>=32 leaf dispatch. The CUDA fp8/block-fp8 selector lists
# append Humming after the existing candidates (kernels/linear/__init__.py:
# 322-363); NVFP4 still uses CT's factory and records the selected kernel
# (schemes/compressed_tensors_w4a4_nvfp4.py:29-31,95-141). This adds the
# exact 0.25.0 release to the existing lane, not 0.25.1/0.26.0/0.27.0.
# 0.30.0 audit (2026-09-30, source-only: tag v0.30.0 == ced6857afa0e vs
# v0.27.1; runtime smoke on GB300 recorded in the GLM-5.3-Flash W1 campaign).
# Intermediate 0.28.0/0.29.0 were NOT audited and stay excluded.
# bf16: UnquantizedLinearMethod now binds its GEMM at __init__ from
#   kernel_config.linear_backend (layers/linear.py:169-174, apply :234-238);
#   dispatch_unquantized_gemm (layers/utils.py:584-616) returns
#   default_unquantized_gemm = F.linear (:84-90) unless the opt-in
#   linear_backend "flashinfer_cutedsl" is set (default "auto",
#   config/kernel.py:300). run_gemm asserts that binding below.
# fp8: per-token dynamic activation key when cutlass is supported
#   (fp8.py:289-297). _POSSIBLE_FP8_KERNELS[CUDA]
#   (kernels/linear/__init__.py:421-430) is now FlashInfer, Cutlass,
#   B12xTensor (SM12x, static only), Torch x2, Marlin, Humming; FlashInfer
#   rejects per-token activations (scaled_mm/flashinfer.py:56-65), so SM89+
#   still selects CutlassFP8ScaledMMLinearKernel (cutlass.py:163-173).
# fp8_block: _POSSIBLE_FP8_BLOCK_KERNELS[CUDA] (__init__.py:457-466) is
#   FlashInferDyn (SM90 only), DeepGemm, Cutlass, B12x (SM12x only,
#   scaled_mm/b12x.py:53-66), Marlin, Humming, Triton, BlockWiseTorch. SM10x:
#   DeepGemm when N%64==0 and K%128==0 (scaled_mm/deep_gemm.py:46-82,
#   utils/deep_gemm.py:775-794), else Cutlass (cutlass.py:287-310). SM89:
#   Marlin lost its VLLM_TEST_FORCE_FP8_MARLIN guard (scaled_mm/marlin.py:
#   35-46), so Marlin (W8A16) now replaces Triton there -- kernel_source
#   records it. The FlashInferDyn m<32 leaf split is unchanged
#   (scaled_mm/flashinfer.py:301-316, file byte-identical).
# nvfp4: CT from_config (compressed_tensors.py:231) only adds an actorder
#   rejection; compressed_tensors_w4a4_nvfp4.py is byte-identical.
#   _POSSIBLE_NVFP4_KERNELS[CUDA] (__init__.py:551-567) inserts
#   FlashInferB12x (SM120+, nvfp4/flashinfer.py:383-393): SM10x still
#   selects FlashInferCuteDsl (:115-123), SM12x FlashInferCutlass (:183-197).
# Byte-identical helpers: scaled_fp4_quant (_custom_ops.py:1525-1600),
#   per_block_cast_to_fp8 (utils/deep_gemm.py:737-756).
# glm53tail overlay (0.30.0+glm53tail.eb4704514fdf): kv_cache_interface.py
#   is imported at module scope via quantization/kv_cache.py:14 but only
#   called by attention KV-cache methods; sparse_attn_indexer_kpool.py is
#   imported only by models/glm5next/nvidia/attention.py:26. Neither is on
#   a GEMM path, so rows are identical with or without the overlay.
__compat__ = "vllm>=0.24.0,<=0.30.0,!=0.25.1,!=0.26.0,!=0.27.0,!=0.28.0,!=0.29.0"

from types import SimpleNamespace

import torch
import vllm.envs as envs
from collector.case_generator import get_gemm_case_specs
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.vllm.utils import setup_distributed, with_exit_stack
from vllm._custom_ops import scaled_fp4_quant as _scaled_fp4_quant
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig as _CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.utils.deep_gemm import per_block_cast_to_fp8
from vllm.version import __version__ as vllm_version

FP8_BLOCK_SHAPE = (128, 128)

# NVFP4 is source-supported by vLLM 0.24.0 on datacenter and RTX Blackwell
# (SM100+). The SM100/SM103/SM120 paths remain hardware-unvalidated in this
# effort.

_NVFP4_QUANT_ARGS = {
    "num_bits": 4,
    "type": "float",
    "strategy": "tensor_group",
    "group_size": 16,
    "symmetric": True,
    "dynamic": False,
}


def get_gemm_test_cases():
    sm = get_sm_version()

    # Open floors matching cases/capabilities.yaml (fp8: 89, fp8_block: 89,
    # nvfp4: 100) — a closed SM whitelist here would silently drop unlisted
    # SMs (e.g. SM101/SM121) with no logged reason.
    gemm_list = ["bfloat16"]
    if sm >= 89:
        gemm_list += ["fp8"]
    # Blockwise FP8 runs on fp8 hardware from SM89 (Ada): below SM90 the
    # DeepGEMM/cutlass tiers of vLLM's block-scale dispatch are unavailable
    # and _POSSIBLE_FP8_BLOCK_KERNELS falls through to the Marlin/Triton
    # tiers (model_executor/kernels/linear/__init__.py:319-330 @0.24.0;
    # Marlin is selected first on SM89 from 0.30.0, __init__.py:457-466);
    # verified end-to-end on L40S (SM89) at 0.24.0, with kernel_source
    # recording the actually-selected kernel per row.
    if sm >= 89:
        gemm_list += ["fp8_block"]

    if sm >= 100:
        gemm_list += ["nvfp4"]

    test_cases = []

    for gemm_common_testcase in get_gemm_case_specs():
        x = gemm_common_testcase.x
        n = gemm_common_testcase.n
        k = gemm_common_testcase.k
        for gemm_type in gemm_list:
            test_cases.append([gemm_type, x, n, k])

    return test_cases


@with_exit_stack
def run_gemm(exit_stack, gemm_type, m, n, k, *, perf_filename, device="cuda:0"):
    setup_distributed(device)

    if envs.VLLM_BATCH_INVARIANT:
        # Batch-invariant mode reroutes bf16 linears to linear_batch_invariant
        # (vllm/model_executor/layers/linear.py:223-224 @0.24.0, :228-229
        # @0.27.1 -- byte-identical body, 5-line shift) and per-tensor fp8 to
        # a BF16-dequant F.linear path (fp8.py:452-486 @0.24.0, :442-476
        # @0.27.1 -- byte-identical body, 10-line shift; re-verified
        # 2026-08-21). At 0.30.0 the bf16 reroute is linear.py:234-237 and
        # the fp8 one fp8.py:444-477 (re-verified 2026-09-30). Either way
        # the kernel_source values recorded below would not be ground truth.
        raise RuntimeError("VLLM_BATCH_INVARIANT is set; gemm kernel_source recording assumes default dispatch")

    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    x = torch.randn((m, k), dtype=dtype, device=torch.device(device))

    if gemm_type == "fp8":
        qc = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            ignored_layers=None,
            weight_block_size=None,
        )
    elif gemm_type == "fp8_block":
        # FIXME(kernel-limit): on SM120, vLLM's default block-fp8 linear
        # dispatch is broken end to end: support_deep_gemm claims the 12x
        # family (platforms/cuda.py:663-669 @0.24.0, :665-671 @0.27.1 --
        # body byte-identical, only the def line shifted) so DeepGEMM takes
        # shapes with N%64==0 and K%128==0 (should_use_deepgemm_for_fp8_linear,
        # utils/deep_gemm.py:700-720, unchanged at 0.27.1 -- the whole file
        # is byte-identical between the two pinned tags) and asserts "Unknown
        # SF transformation" (deepgemm csrc/apis/layout.hpp:59, outside this
        # repo, not re-verified at this bump); the remaining shapes go to
        # cutlass c3x which fails "Invalid status" (cutlass_gemm_caller.cuh
        # :51, also outside this repo). 29/30 sampled fp8_block shapes failed
        # on RTX PRO 6000 Blackwell at 0.24.0; module collectors with
        # fp8_block linears (mla/dsa/dsv4/moe) inherit the same failure at
        # build time. Serving fails identically. Upstream: vllm#47436/#47130
        # (open, same assertion), DeepGEMM#318 (SM120 support PR, open); the
        # Triton block-fp8 kernel works on SM120 (verified) and
        # vllm#40929/#41834 move DSV4-on-SM120 onto that fallback. Re-verified
        # 2026-08-21 against the v0.27.1 source clone: nothing on the vllm
        # side (the two citations above) suggests this gap closed since
        # 0.24.0 -- re-verify again on the next vLLM/DeepGEMM bump.
        # 2026-09-30 v0.30.0 re-verification: support_deep_gemm body unchanged
        # (platforms/cuda.py:721-727), should_use_deepgemm_for_fp8_linear
        # unchanged (utils/deep_gemm.py:775-794); cutlass_gemm_caller.cuh's
        # check is at :52 and the SM120 blockwise swap_ab/CTA-swizzle dispatch
        # changed (#55180), so which shapes hit "Invalid status" may differ.
        # The new B12x block kernel sits after DeepGemm/Cutlass and never
        # auto-selects. Claim still unverified on SM120 at 0.30.0.
        qc = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=list(FP8_BLOCK_SHAPE),
        )
    elif gemm_type == "nvfp4":
        qc = _CompressedTensorsConfig.from_config(
            {
                "quant_type": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "global_compression_ratio": 1.0,
                "config_groups": {
                    "group_0": {
                        "weights": _NVFP4_QUANT_ARGS,
                        "input_activations": _NVFP4_QUANT_ARGS,
                        "targets": ["Linear"],
                        "output_activations": None,
                    }
                },
            }
        )
    else:
        qc = None

    def create_gemm():
        gemm = RowParallelLinear(
            input_size=k,
            output_size=n,
            bias=False,
            skip_bias_add=True,
            params_dtype=dtype,
            quant_config=qc,
            prefix="",
            return_bias=True,
            disable_tp=True,
        )
        # TODO, to evaluate random weights impact
        gemm.to(torch.device(device))

        if gemm_type == "fp8":
            with torch.no_grad():
                gemm.weight.fill_(0.01)
                gemm.weight_scale.fill_(1.0)
            gemm.quant_method.process_weights_after_loading(gemm)
        elif gemm_type == "fp8_block":
            block_n, block_k = FP8_BLOCK_SHAPE
            with torch.no_grad():
                # Blockwise quantize a random weight to provide valid scales.
                raw_weight = torch.randn((n, k), dtype=torch.float32, device=device)
                q_weight, weight_scale = per_block_cast_to_fp8(raw_weight, [block_n, block_k], use_ue8m0=False)
                gemm.weight.copy_(q_weight)
                gemm.weight_scale_inv.copy_(weight_scale.contiguous().to(torch.float32))
                gemm.quant_method.process_weights_after_loading(gemm)
        elif gemm_type == "nvfp4":
            with torch.no_grad():
                weight_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device)
                w_gscale_val = float(weight_bf16.abs().max()) / 6.0
                weight_fp4, weight_scale_fp8 = _scaled_fp4_quant(
                    weight_bf16,
                    torch.tensor(1.0 / w_gscale_val, dtype=torch.float32, device=device),
                    is_sf_swizzled_layout=False,
                )
                in_gscale_val = float(x.abs().max()) / 6.0
                gemm.weight_packed.data.copy_(weight_fp4)
                gemm.weight_scale.data.copy_(weight_scale_fp8.to(torch.float8_e4m3fn))
                # CT convention: global_scale parameters store 1/actual_scale.
                gemm.weight_global_scale.data.fill_(1.0 / w_gscale_val)
                gemm.input_global_scale.data.fill_(1.0 / in_gscale_val)
            gemm.scheme.process_weights_after_loading(gemm)

        gemm.forward(x)  # dry run to init

        return gemm

    vllm_config = VllmConfig()
    if vllm_config.model_config is None:
        vllm_config.model_config = SimpleNamespace(
            dtype=dtype,
            hf_text_config=SimpleNamespace(model_type=""),
            model="collector_dummy",
        )
    exit_stack.enter_context(set_current_vllm_config(vllm_config))

    outside_loop_count = 1 if gemm_type in ("fp8_block", "nvfp4") else 6
    op_list = []
    for i in range(outside_loop_count):
        op_list.append(create_gemm())

    if gemm_type in {"fp8", "fp8_block"}:
        kernel_sources = set()
        for op in op_list:
            selected_kernel = op.quant_method.fp8_linear
            if isinstance(selected_kernel, FlashInferFp8DeepGEMMDynamicBlockScaledKernel):
                # SM90-only wrapper (see header). vLLM's custom op selects the
                # same two leaf objects at scaled_mm/flashinfer.py:301-316
                # (byte-identical 0.24.0-0.30.0): DeepGEMM for m >= 32,
                # FlashInfer swap-AB below. Its batch-invariant branch is
                # unreachable here — run_gemm raises on VLLM_BATCH_INVARIANT
                # at entry — so the label depends on m alone.
                selected_kernel = selected_kernel.fallback if m >= 32 else selected_kernel.base
            kernel_sources.add(type(selected_kernel).__name__)
    elif gemm_type == "nvfp4":
        kernel_sources = {type(op.scheme.kernel).__name__ for op in op_list}
    else:
        # From 0.30.0 UnquantizedLinearMethod binds its GEMM at __init__
        # (layers/linear.py:169-174 -> layers/utils.py:584-616); earlier
        # releases have no _gemm_impl and always call F.linear. Assert the
        # binding instead of assuming the label.
        from vllm.model_executor.layers.utils import default_unquantized_gemm

        bound = {getattr(op.quant_method, "_gemm_impl", default_unquantized_gemm) for op in op_list}
        if bound != {default_unquantized_gemm}:
            raise RuntimeError(f"vLLM bf16 linear is not bound to F.linear: {sorted(map(repr, bound))}")
        kernel_sources = {"torch.nn.functional.linear"}
    if len(kernel_sources) != 1:
        raise RuntimeError(f"vLLM selected inconsistent GEMM kernels: {sorted(kernel_sources)}")
    kernel_source = kernel_sources.pop()

    def kernel_func():
        for op in op_list:
            op.forward(x)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=6,
        repeat_n=1,
        # These rows model GPU execution, including graph-replayed decode.
        # Eager fp8_block timing includes host launch gaps and is not a valid
        # substitute for this contract. Capture failures must remain failures.
        allow_graph_fail=False,
        use_cuda_graph=True,
    ) as results:
        pass
    if results.get("used_cuda_graph") is not True:
        raise RuntimeError("vLLM GEMM collection requires CUDA Graph replay; refusing to publish eager timing")

    log_perf(
        item_list=[
            {
                "gemm_dtype": gemm_type,
                "m": m,
                "n": n,
                "k": k,
                "latency": results["latency_ms"] / outside_loop_count,
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="gemm",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_gemm_test_cases()
    for test_case in test_cases[:10]:
        run_gemm(*test_case, perf_filename=PerfFile.GEMM)
