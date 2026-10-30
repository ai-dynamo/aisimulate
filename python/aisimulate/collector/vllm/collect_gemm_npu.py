# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM GEMM collector for Huawei Ascend NPU devices (vllm-ascend).

The NPU counterpart to the XPU vLLM GEMM collector: it builds
RowParallelLinear layers, expands YAML-backed matrix shapes and logs perf
rows through the device-agnostic helpers. MVP scope: bf16 only, eager only,
no power sampling.

The pinned version is the vllm-ascend release, not base vLLM: vllm-ascend
ships the Ascend kernels, so it is the kernel-source identity of these rows
and the version axis of the parquet directory (manifest key ``vllm_npu``).
"""

__compat__ = "vllm_ascend==0.30.0rc2"

import os

import torch
from collector.case_generator import get_gemm_case_specs, get_gemm_type_specs
from collector.helper import benchmark_with_power, get_device_module, log_perf

# The stock vLLM harness utilities reused here are device-neutral:
# create_vllm_config builds a plain VllmConfig from a local fake HF model and
# setup_distributed does a single-rank FileStore init whose process-group
# backend comes from vLLM's own platform plugin (HCCL under vllm-ascend).
# The XPU fork (utils_xpu) stays off-limits to non-XPU collectors
# (tests/unit/collector/test_vllm_xpu_utils.py), and its env://+port rendezvous
# is the shape the precedent survey recommends avoiding for this lane.
# TODO: on-device campaign — confirm that the single-rank init selects the HCCL PG.
from collector.vllm.utils import create_vllm_config, setup_distributed, with_exit_stack
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm_ascend import __version__ as vllm_ascend_version

# Max device-memory fraction for the packed op copies.
_LOOP_MEM_BUDGET_FRACTION = 0.45

# Upper bound on GEMM ops packed into one measurement loop. Small GEMMs are
# launch-overhead dominated, so packing more per measurement amortizes it and
# stabilizes the per-op latency (total/count). The memory budget drops this
# lower for large shapes.
_MAX_OPS_PER_LOOP = 64


# Per-worker cache: VllmConfig is identical for every gemm case, but building
# it re-scans package entry_points (slow). Build once, reuse.
_VLLM_CONFIG_CACHE = {}


def _get_cached_vllm_config(dtype):
    key = str(dtype)
    if key not in _VLLM_CONFIG_CACHE:
        model = os.path.join(os.path.dirname(__file__), "fake_hf_model")
        _VLLM_CONFIG_CACHE[key] = create_vllm_config(model_name=model, dtype=dtype)
    return _VLLM_CONFIG_CACHE[key]


def _gemm_peak_footprint_bytes(m: int, n: int, k: int, copies: int = 1) -> int:
    """Peak bf16 GEMM case footprint (bytes): input + copies*(weight+output).

    ``copies=0`` returns just the copy-independent portion (the input).
    """
    input_bytes = m * k * 2
    per_copy = (n * k + m * n) * 2
    return input_bytes + per_copy * max(copies, 0)


def get_gemm_test_cases():
    gemm_list = get_gemm_type_specs("vllm_npu")
    if not gemm_list:
        raise RuntimeError("collector/cases/base_ops/gemm.yaml must define vllm_npu gemm_types")

    test_cases = []

    for gemm_common_testcase in get_gemm_case_specs("vllm_npu"):
        x = gemm_common_testcase.x
        n = gemm_common_testcase.n
        k = gemm_common_testcase.k
        for gemm_type in gemm_list:
            test_cases.append([gemm_type, x, n, k])

    return test_cases


@with_exit_stack
def run_gemm(exit_stack, gemm_type, m, n, k, *, perf_filename, device="npu:0"):
    if gemm_type != "bfloat16":
        # The YAML partition declares bf16 only; anything else would be
        # benchmarked as bf16 and recorded under the wrong dtype label.
        raise RuntimeError(f"vllm_npu gemm collects bfloat16 only, got {gemm_type!r}")

    setup_distributed(device)

    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    get_device_module(device).set_device(device)
    # TODO: on-device campaign — NPU visibility is governed by
    # ASCEND_RT_VISIBLE_DEVICES (CUDA_VISIBLE_DEVICES has no effect);
    # multi-card parallel collection must pin its workers with the former.

    x = torch.randn((m, k), dtype=dtype, device=torch.device(device))

    def create_gemm():
        gemm = RowParallelLinear(
            input_size=k,
            output_size=n,
            bias=False,
            skip_bias_add=True,
            params_dtype=dtype,
            quant_config=None,
            prefix="",
            return_bias=True,
            disable_tp=True,
        )
        # vLLM >=0.16 may create layers on the meta device; use to_empty()
        # then fill with random data.
        try:
            gemm.to(torch.device(device))
        except NotImplementedError:
            gemm = gemm.to_empty(device=torch.device(device))
            with torch.no_grad():
                for param in gemm.parameters():
                    if param.dtype.is_floating_point:
                        param.normal_()
                    else:
                        param.zero_()

        gemm.forward(x)  # noqa: F821  # dry run to init

        return gemm

    # VllmConfig is identical across gemm cases; build once per worker and
    # reuse (construction re-scans package entry_points, which is slow).
    vllm_config = _get_cached_vllm_config(dtype)
    exit_stack.enter_context(set_current_vllm_config(vllm_config))

    # Ops per measurement: pack as many as the memory budget allows, capped so
    # tiny shapes don't create an absurd number. Large shapes -> few
    # (memory-bound); tiny shapes -> the cap, diluting fixed launch overhead.
    total_mem = get_device_module(device).get_device_properties(device).total_memory
    per_copy = (n * k + m * n) * 2
    fixed_bytes = _gemm_peak_footprint_bytes(m, n, k, copies=0)
    budget = int(total_mem * _LOOP_MEM_BUDGET_FRACTION)
    mem_cap = (budget - fixed_bytes) // max(per_copy, 1)
    outside_loop_count = max(1, min(_MAX_OPS_PER_LOOP, mem_cap))

    op_list = []
    try:
        for i in range(outside_loop_count):
            op_list.append(create_gemm())

        def kernel_func():
            for op in op_list:  # noqa: F821
                op.forward(x)  # noqa: F821

        with benchmark_with_power(
            device=device,
            kernel_func=kernel_func,
            num_warmups=3,
            num_runs=6,
            repeat_n=1,
            # MVP is eager: ACL graph capture is a follow-up, and
            # benchmark_with_power has no NPU graph arm (helper.py use_graph is
            # CUDA/XPU only), so `used_cuda_graph` is recorded False below.
            use_cuda_graph=False,
            allow_graph_fail=False,
        ) as results:
            pass
    finally:
        # Free device weights after each case; a packed op_list left cached
        # OOMs later shapes.
        del op_list
        del x
        import gc

        gc.collect()
        get_device_module(device).empty_cache()

    # torch_npu 2.10.0.post7 API notes (read from the pinned image's own
    # wheel): get_device_name validates the ordinal but returns the global SoC
    # name (no per-card distinction) — fine for this informational column,
    # same behavior as the XPU lane.
    # get_device_capability is never called: it returns None unless
    # TORCH_NPU_DEVICE_CAPABILITY is set, and then a configured fake value.
    log_perf(
        item_list=[
            {
                "gemm_dtype": gemm_type,
                "m": m,
                "n": n,
                "k": k,
                "latency": results["latency_ms"] / outside_loop_count,
                # Same columns as the XPU gemm table; always False while the
                # lane is eager-only.
                "used_cuda_graph": results["used_cuda_graph"],
            }
        ],
        framework="VLLM",
        version=vllm_ascend_version,
        device_name=get_device_module(device).get_device_name(device),
        op_name="gemm",
        # The vllm-ascend dispatch for an unquantized linear is not traced yet
        # (same posture as the XPU lane); op_backend_facts.yaml records the
        # slice as backend "unverified".
        kernel_source="vllm_default",
        perf_filename=perf_filename,
        # No NVML-equivalent on CANN: power is never sampled on this lane.
        power_stats=None,
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_gemm_test_cases()
    for test_case in test_cases[:10]:
        run_gemm(*test_case, perf_filename=PerfFile.GEMM)
