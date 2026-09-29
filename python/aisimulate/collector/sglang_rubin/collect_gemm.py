# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 linear collector for the GLM-5.2 NVFP4 Rubin image pilot.

The checkpoint leaves attention, shared experts and the first three dense MLPs
unquantized. This collector deliberately exposes only that BF16 lane. Timing and
row construction follow the same-repository ``sglang/collect_gemm.py``.
Framework references below are relative to python/sglang at immutable commit
02c5a855aceb968c310e6fbc6632270e26edc84b in dl/sglang/sglang.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from collector.sglang_rubin import runtime

# Observed in the pinned image on Hecate, JET job 443897220. This is the
# installed distribution version; the SGLANG_VERSION build label differs.
SGLANG_DISTRIBUTION_VERSION = "0.5.18+nvinternal.rubin.0.8full.66997102"
__compat__ = f"sglang=={SGLANG_DISTRIBUTION_VERSION}"
PILOT_MODEL = "nvidia/GLM-5.2-NVFP4"
# Only this checkpoint's replicated router has this (N, K). Its TP4 dense,
# shared, attention and vocabulary projections have different physical widths.
# Native MoEGate construction: srt/models/deepseek_v2.py:458-495, :620-629.
_ROUTER_NK = (256, 6144)
_MODEL_CONFIG = (
    Path(__file__).resolve().parents[2] / "src/aisimulate_core/model_configs/nvidia--GLM-5.2-NVFP4_config.json"
)


def _require_model_scope():
    requested = os.environ.get("COLLECTOR_MODEL_PATH", "").strip()
    if requested != PILOT_MODEL:
        raise ValueError(f"Rubin pilot requires COLLECTOR_MODEL_PATH={PILOT_MODEL!r}, got {requested!r}")


def _require_runtime():
    # This inventory does not import SGLang. In particular, require the pilot's
    # serving environment before callers load framework modules; never repair it
    # after SGLang has resolved module or layer state.
    inventory = runtime.collect_inventory()
    errors = runtime.validate_runtime(inventory)
    installed = inventory.get("observed", {}).get("package_versions", {}).get("sglang", {}).get("version")
    # Shared version routing normalizes away local versions. Check the full
    # installed value here so a generic 0.5.18 wheel cannot enter this fork.
    if installed != SGLANG_DISTRIBUTION_VERSION:
        errors.append(f"Expected SGLang distribution {SGLANG_DISTRIBUTION_VERSION!r}, observed {installed!r}")
    if errors:
        raise RuntimeError("Rubin runtime preflight failed: " + "; ".join(errors))


def get_gemm_test_cases():
    """Keep the shared physical shape grid, in the declared BF16 pilot lane."""
    from collector.case_generator import get_gemm_case_specs

    _require_model_scope()
    requested = os.environ.get("AIC_COLLECT_GEMM_TYPES")
    if requested is not None and {item.strip() for item in requested.split(",")} != {"bfloat16"}:
        raise ValueError("Rubin GLM-5.2 NVFP4 pilot GEMM supports only bfloat16")
    return [["bfloat16", case.x, case.n, case.k] for case in get_gemm_case_specs(backend="sglang")]


def _kernel_source(unquant, m, n, k):
    """Use the very predicate consumed by UnquantizedLinearMethod.apply.

    srt/layers/quantization/unquant.py:89-109 resolves auto to CuteDSL on
    SM10x; :238-265 chooses CuteDSL or torch per shape for BF16 CUDA weights
    with requires_grad=False. All those input conditions hold below.
    """
    backend = unquant.get_bf16_gemm_backend()
    if backend.is_cutedsl():
        if unquant._use_cutedsl_bf16_gemm(m, n, k):
            return "sglang_cutedsl_bf16_gemm"
        return "sglang_torch_linear"
    if backend.value == "torch":
        return "sglang_torch_linear"
    raise RuntimeError(f"Unresolved Rubin BF16 GEMM backend: {backend.value!r}")


def _make_router():
    """Construct the checkpoint's native gate; do not reproduce its dispatch."""
    import torch
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models.deepseek_v2 import MoEGate
    from sglang.srt.runtime_context import get_exec

    from collector.sglang_rubin.registry import CHECKPOINT_METADATA_SHA256

    payload = _MODEL_CONFIG.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CHECKPOINT_METADATA_SHA256["config.json"]:
        raise RuntimeError("Rubin router requires the frozen GLM-5.2 NVFP4 checkpoint metadata")
    config = SimpleNamespace(**json.loads(payload))
    if get_exec().deterministic.enable_deterministic_inference:
        raise RuntimeError("Rubin router pilot requires deterministic inference disabled")
    # Every config field comes unchanged from the hash-checked checkpoint.
    # modelopt_quant.py:1449-1461 prefers its embedded quantization_config;
    # model_loader/loader.py:806-816 constructs under the model's default dtype.
    quant = ModelOptFp4Config.from_config(config.quantization_config)
    with set_default_torch_dtype(torch.bfloat16):
        # MoEGate defaults match deepseek_v2.py:620-629 for ordinary GLM layers:
        # no next-n/hash/V4/CP. forward_normal (:959-960) passes hidden states
        # without a ForwardBatch; no TP communication or top-k runs here.
        layer = MoEGate(config=config, quant_config=quant)
    layer.requires_grad_(False)
    # The gate owns BF16 [n_routed_experts, hidden_size] weights and FP32
    # correction bias (:472-491). Synthetic values retain that native storage;
    # forward does not consume correction bias (it belongs to the top-k op).
    with torch.no_grad():
        layer.weight.normal_()
        layer.e_score_correction_bias.zero_()
    return layer


def _observe_router_source(kernel_func, shape):
    """Observe one untimed native call, then remove profiling before timing."""
    import torch

    router = importlib.import_module("sglang.kernels.ops.gemm.dsv3_router_gemm")
    linear = importlib.import_module("sglang.kernels.ops.attention.dsv4.gemm")
    # The JIT implementation calls the registered custom op; the cuBLAS leaf
    # calls torch.mm(..., out_dtype=float32). References: kernels/ops/gemm/
    # dsv3_router_gemm.py:65-97 and ops/attention/dsv4/gemm.py:119-122.
    leaves = {
        inspect.unwrap(router.dsv3_router_gemm).__code__: "sglang_dsv3_router_gemm",
        linear._linear_bf16_fp32_cublas.__code__: "sglang_linear_bf16_fp32_cublas",
    }
    observed = []

    def observe(frame, event, _arg):
        if event == "return" and frame.f_code in leaves:
            observed.append(leaves[frame.f_code])

    if sys.getprofile() is not None:
        raise RuntimeError("Rubin router dispatch observation requires no existing Python profiler")
    try:
        sys.setprofile(observe)
        with torch.no_grad():
            output = kernel_func()
    finally:
        sys.setprofile(None)
    if len(observed) != 1 or output.dtype != torch.float32 or tuple(output.shape) != shape:
        raise RuntimeError(
            f"Unexpected Rubin router execution: leaves={observed}, output={output.dtype}/{output.shape}"
        )
    return observed[0]


def run_gemm(gemm_type, M, N, K, *, perf_filename, device="cuda:0"):  # noqa: N803
    if gemm_type != "bfloat16":
        raise ValueError(f"Rubin pilot GEMM supports only bfloat16, got {gemm_type!r}")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (M, N, K)):
        raise ValueError("GEMM dimensions must be positive integers")
    _require_model_scope()
    _require_runtime()

    import torch
    from sglang.srt.layers.quantization import unquant
    from sglang.srt.runtime_context import get_context

    from collector.helper import benchmark_with_power, log_perf

    torch.cuda.set_device(device)
    layer = None
    x = None
    try:
        with get_context().override_server_args(bf16_gemm_backend="auto"), torch.device(device):
            # Use the framework initializer and method; plain F.linear misses
            # this image's default CuteDSL path (citation in _kernel_source).
            unquant.initialize_bf16_gemm_config(get_context().server_args)
            if (N, K) == _ROUTER_NK:
                layer = _make_router()
                x = torch.randn(M, K, dtype=torch.bfloat16, device=device)

                def kernel_func():
                    return layer(x)

                kernel_source = _observe_router_source(kernel_func, (M, N))
            else:
                method = unquant.UnquantizedLinearMethod()
                layer = torch.nn.Module()
                method.create_weights(layer, K, [N], K, N, torch.bfloat16)
                with torch.no_grad():
                    layer.weight.normal_()
                method.process_weights_after_loading(layer)
                x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
                kernel_source = _kernel_source(unquant, M, N, K)

                def kernel_func():
                    return method.apply(layer, x, None)

            with benchmark_with_power(
                device=device, kernel_func=kernel_func, num_warmups=3, num_runs=6, repeat_n=1
            ) as results:
                pass
            if not log_perf(
                item_list=[{"gemm_dtype": gemm_type, "m": M, "n": N, "k": K, "latency": results["latency_ms"]}],
                framework="SGLang",
                version=importlib.metadata.version("sglang"),
                device_name=torch.cuda.get_device_name(device),
                op_name="gemm",
                kernel_source=kernel_source,
                perf_filename=perf_filename,
                power_stats=results["power_stats"],
            ):
                raise RuntimeError(f"Failed to persist Rubin GEMM performance row to {perf_filename}")
    finally:
        layer = x = None
        torch.cuda.empty_cache()
