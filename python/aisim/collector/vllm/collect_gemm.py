# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# vLLM owns block-FP8 dispatch. Its FlashInfer/DeepGEMM dynamic wrapper
# (FlashInfer for m < 32, DeepGEMM above) is selectable on SM90 only
# (has_flashinfer_fp8_blockscale_gemm in utils/flashinfer.py); SM10x and SM120
# select DeepGEMM or CUTLASS directly (see the __compat__ comment below). Keep
# every grid M/N/K shape observable instead of hiding a presumed M-divisibility
# restriction in population logic.

"""vLLM GEMM collector for CUDA backends.

Builds vLLM RowParallelLinear layers with synthetic weights to benchmark BF16,
FP8, FP8 block, and FP4-style paths where available. Shared GEMM shapes come
from `case_generator.py`; this file handles vLLM config contexts, distributed setup,
quantized-weight preparation, and selected-kernel reporting.
"""

# Audited against vLLM v0.32.0 (from v0.31.0): every import, constructor
# argument and config field used below is unchanged, and so is the Python
# kernel selection of every lane. Earlier audits are git history. Selection is
# first-match over the CUDA candidate lists in model_executor/kernels/linear/
# __init__.py with linear_backend "auto" (config/kernel.py:350); the new
# prioritize_humming (:411) only moves Humming ahead of Marlin, and only on
# SM90 (quantization/utils/humming/priority.py:11), where no lane reaches
# either.
# bf16, every SM: UnquantizedLinearMethod.__init__ binds
#   dispatch_unquantized_gemm(linear_backend) (layers/linear.py:170-175, apply
#   :229-239), which returns default_unquantized_gemm = F.linear
#   (layers/utils.py:620-652, :83-89). run_gemm asserts that binding below.
# fp8, SM89-SM120: dynamic activations take the per-token key when CUTLASS FP8
#   is supported (fp8.py:272-280). FlashInferFP8ScaledMMLinearKernel, first in
#   _POSSIBLE_FP8_KERNELS (__init__.py:444-452), needs per-tensor activations
#   (scaled_mm/flashinfer.py:56-65), so CutlassFP8ScaledMMLinearKernel runs
#   (scaled_mm/cutlass.py:158-175).
# fp8_block (_POSSIBLE_FP8_BLOCK_KERNELS, __init__.py:481-490):
#   SM89: no DeepGEMM (platforms/cuda.py:689-695) or CUTLASS block-FP8
#     (cutlass_scaled_mm_supports_block_fp8 needs SM90+) and B12x is SM12x only
#     (scaled_mm/b12x.py:53-67), so MarlinFP8ScaledMMLinearKernel (W8A16,
#     scaled_mm/marlin.py:29-49).
#   SM90: FlashInferFp8DeepGEMMDynamicBlockScaledKernel when N%64==0 and
#     K%128==0 (scaled_mm/flashinfer.py:88-195, utils/flashinfer.py:1160-1199),
#     else CutlassFp8BlockScaledMMKernel (scaled_mm/cutlass.py:277-312).
#   SM100, SM103, SM120: DeepGemmFp8BlockScaledMMKernel when N%64==0 and
#     K%128==0 (scaled_mm/deep_gemm.py:47-82, utils/deep_gemm.py:111-117 and
#     :1021-1040; DeepGEMM covers families 100 and 120), else Cutlass.
# nvfp4: CT from_config (compressed_tensors.py:235) maps the config below to
#   CompressedTensorsW4A4Fp4 (:804), whose init_nvfp4_linear_kernel
#   (__init__.py:1128-1246) walks _POSSIBLE_NVFP4_KERNELS (:575-591): SM100 and
#   SM103 FlashInferCuteDslNvFp4LinearKernel (nvfp4/flashinfer.py:112-126),
#   SM120 FlashInferCutlassNvFp4LinearKernel (:185-206). The new
#   B12xNvFp4W4A16LinearKernel is weight-only and listed after both.
# Kernel-side changes since v0.31.0: DeepGEMM moves from vllm-project/DeepGEMM
#   e1f418c2 to 1e1842a8 (cmake/external_projects/deepgemm.cmake:37), a
#   binding migration that leaves the SM90/SM100 FP8 GEMM kernels and
#   heuristics as they were but replaces the SM120 device layer, so SM120
#   DeepGEMM rows change. nvidia-cutlass-dsl, the compiler of FlashInfer's
#   cute-dsl NVFP4 GEMM, moves 4.7.1 -> 4.8.0 (requirements/cuda.txt).
#   FlashInfer 0.7.0.post1, CUTLASS v4.7.1, torch 2.13.0 and CUDA 13.0.3 are
#   unchanged; vLLM's own CUTLASS scaled-mm, FP8/FP4 quant and Marlin sources
#   moved out of csrc/libtorch_stable with include-only edits.
# Helpers with identical bodies: scaled_fp4_quant (_custom_ops.py:1513-1588),
#   per_block_cast_to_fp8 (utils/deep_gemm.py:984-1003).
# One release per file; earlier code is `git log -- <this file>` away.
__compat__ = "vllm==0.32.0"

import contextlib
from types import SimpleNamespace

import torch
import vllm.envs as envs
from collector.helper import (
    benchmark_with_power,
    flashinfer_autotuned,
    flashinfer_tuning_bucket,
    get_sm_version,
    log_perf,
)
from collector.op_plans import plan_cases
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
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import per_block_cast_to_fp8
from vllm.utils.flashinfer import has_flashinfer
from vllm.version import __version__ as vllm_version

FP8_BLOCK_SHAPE = (128, 128)

# Weights and input activations of an nvfp4-pack-quantized compressed-tensors
# checkpoint (W4A4, 16-element groups); the header lists the kernel per SM.
_NVFP4_QUANT_ARGS = {
    "num_bits": 4,
    "type": "float",
    "strategy": "tensor_group",
    "group_size": 16,
    "symmetric": True,
    "dynamic": False,
}


def get_gemm_test_cases():
    return plan_cases("gemm", "vllm", get_sm_version())


@with_exit_stack
def run_gemm(exit_stack, gemm_type, m, n, k, *, perf_filename, device="cuda:0"):
    setup_distributed(device)

    if envs.VLLM_BATCH_INVARIANT:
        # Batch-invariant mode reroutes bf16 linears to linear_batch_invariant
        # (layers/linear.py:235-238), per-tensor fp8 to a BF16-dequant F.linear
        # path unless fp8_scaled_mm_is_batch_invariant accepts the kernel
        # (fp8.py:426-459, kernels/linear/__init__.py:815-825), and NVFP4 to
        # CUTLASS or emulation (__init__.py:1145-1173), so the kernel_source
        # values recorded below would not be ground truth.
        raise RuntimeError("VLLM_BATCH_INVARIANT is set; gemm kernel_source recording assumes default dispatch")

    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    x = torch.randn((m, k), dtype=dtype, device=torch.device(device))

    if gemm_type == "fp8":
        # This row times dynamic per-token activations on per-tensor weights, and
        # the perf model derives fp8_static from it plus compute_scale, assuming
        # both schemes run one kernel. On compute capability 10.0 and newer, vLLM
        # serves static per-tensor checkpoints with FlashInferFP8ScaledMMLinearKernel,
        # first in _POSSIBLE_FP8_KERNELS (kernels/linear/__init__.py:444-452) and
        # limited to SM100+ with per-tensor activations (scaled_mm/flashinfer.py:
        # 39-65), while this row runs Cutlass, so fp8_static predictions there are
        # approximate, in an unknown direction (seen on SM120:
        # benchmarks/evidence/collector/op-probe-2026-10/findings.yaml,
        # sm120_collector_gemm_fp8_static_row). Retire it by also timing
        # activation_scheme="static" on those systems and preferring that row, or
        # by dropping fp8_static for ModelOpt FP8 checkpoints there.
        qc = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            ignored_layers=None,
            weight_block_size=None,
        )
    elif gemm_type == "fp8_block":
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

    # Serving's VllmConfig carries the checkpoint's quant config
    # (config/vllm.py:1507-1510). For block FP8 that turns on the CUDA
    # quant_fp8 custom op (has_blocked_weights, :1733-1748), so the block
    # kernels' QuantFP8 runs forward_cuda (per-token-group quant, with packed
    # UE8M0 scales for DeepGEMM on SM100/SM103/SM120; input_quant_fp8.py:83-133)
    # instead of the torch.compile'd forward_native that the custom_ops "none"
    # of a bare VllmConfig() selects (model_executor/custom_op.py:172-192). The
    # fp8 and nvfp4 configs leave custom ops unchanged, as in serving.
    vllm_config = VllmConfig(quant_config=qc)
    if vllm_config.model_config is None:
        # Fp8LinearMethod reads model_config.dtype (fp8.py:245), and
        # DeepGemmFp8BlockScaledMMKernel.can_implement reads
        # hf_text_config.model_type (scaled_mm/deep_gemm.py:69-76).
        # Deviation (fp8_block on SM100, SM103 and SM120): serving disables
        # DeepGEMM for model types qwen3_5_text and qwen3_5_moe_text
        # (utils/deep_gemm.py:27-52) and runs CutlassFp8BlockScaledMMKernel on
        # every shape, but this shape-keyed table times DeepGEMM wherever
        # N%64==0 and K%128==0. Block-FP8 GEMMs of those models are predicted on
        # the wrong kernel, direction unknown. Retire it with a per-model
        # dispatch for the excluded model types.
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
                # same two leaf objects at scaled_mm/flashinfer.py:298-313:
                # DeepGEMM for m >= 32, FlashInfer swap-AB below. Its
                # batch-invariant branch is unreachable here — run_gemm raises
                # on VLLM_BATCH_INVARIANT at entry — so the label depends on m
                # alone.
                selected_kernel = selected_kernel.fallback if m >= 32 else selected_kernel.base
            kernel_sources.add(type(selected_kernel).__name__)
    elif gemm_type == "nvfp4":
        kernel_sources = {type(op.scheme.kernel).__name__ for op in op_list}
    else:
        # UnquantizedLinearMethod binds its GEMM at __init__ (layers/linear.py:
        # 170-175 -> layers/utils.py:620-652). Assert the binding instead of
        # assuming the label.
        from vllm.model_executor.layers.utils import default_unquantized_gemm

        bound = {getattr(op.quant_method, "_gemm_impl", default_unquantized_gemm) for op in op_list}
        if bound != {default_unquantized_gemm}:
            raise RuntimeError(f"vLLM bf16 linear is not bound to F.linear: {sorted(map(repr, bound))}")
        kernel_sources = {"torch.nn.functional.linear"}
    if len(kernel_sources) != 1:
        raise RuntimeError(f"vLLM selected inconsistent GEMM kernels: {sorted(kernel_sources)}")
    kernel_source = kernel_sources.pop()

    # Deviation (fp8, fp8_block and nvfp4, every SM): each row times the
    # activation quantization the linear runs on its own input. Under serving's
    # default compilation (O2, config/vllm.py:289-293) that quantization is
    # mostly folded into the op producing the input. FP8 per-token stays a
    # native op that Inductor compiles together with the RMSNorm or
    # SiLU-and-mul in front of it. Block FP8 is fused by RMSNormQuantFusionPass
    # and ActivationQuantFusionPass (compilation/passes/fusion/
    # rms_quant_fusion.py:772-837, act_quant_fusion.py:33-44). NVFP4 is fused
    # into SiLU-and-mul only (act_quant_fusion.py:36-40; models/llama.py:120
    # also fuses it by hand through maybe_fused_act_quant, which reaches the
    # SM100/SM103 cute-dsl kernel since it declares kNvfp4Dynamic input,
    # nvfp4/flashinfer.py:115-116). The quantization stays standalone for
    # projections fed by attention (o_proj), for NVFP4 qkv/gate_up, and on
    # SM90 inside the FlashInfer/DeepGEMM wrapper's opaque custom op
    # (scaled_mm/flashinfer.py:228-296). Elsewhere rows over-predict by about
    # one quantization kernel, most visibly at small m. Retire it by timing the
    # quantization apart from the GEMM and billing it only where serving does
    # not fuse it, as compute_scale does for FP8 per-tensor.

    def kernel_func():
        for op in op_list:
            op.forward(x)

    # FlashInfer kernels run on the tactic serving tunes: kernel warmup autotunes
    # every FlashInfer op on SM90 and newer (warmup/kernel_warmup.py:261-268;
    # enable_flashinfer_autotune is on from O1, config/vllm.py:286-328) and skips
    # fp4_gemm only when the SM100/SM103 cute-dsl NVFP4 kernel is selected, whose
    # fallback is already the heuristic (:295-311). The only tuner-owned op these
    # rows reach is that fp4_gemm under the SM120 FlashInferCutlassNvFp4LinearKernel
    # (nvfp4/flashinfer.py:185-260 -> flashinfer mm_fp4, backend "cutlass" with the
    # 128x4 scale layout, utils/flashinfer.py:1079-1115); the SM90 FlashInfer
    # block-FP8 swap-AB GEMM runs a fixed heuristic (flashinfer 0.7.0.post1
    # gemm/gemm_base.py:10005-10219), and F.linear, vLLM CUTLASS, DeepGEMM and
    # Marlin are not FlashInfer's. Warmup profiles the hybrid token buckets up to
    # max_num_batched_tokens (kernel_warmup.py:334-374) and mm_fp4 maps a runtime
    # m up to the next bucket (gemm_base.py:7887-7915), so tune at that bucket
    # and time m on its tactic. Serving never runs a single linear above
    # max_num_batched_tokens, so every m of the grid has a tuned bucket.
    tune_m = None
    if (
        kernel_source != "FlashInferCuteDslNvFp4LinearKernel"
        and has_flashinfer()
        and current_platform.has_device_capability(90)
    ):
        tune_m = flashinfer_tuning_bucket(m)
    if tune_m is None:
        autotuned = contextlib.nullcontext()
    else:
        autotuned = flashinfer_autotuned(tune_m, kernel_func, label=kernel_source)

    with (
        autotuned,
        benchmark_with_power(
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
        ) as results,
    ):
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
