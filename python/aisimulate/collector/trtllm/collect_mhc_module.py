# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 mHC pre/post module collector for TensorRT-LLM.

TRT-LLM serves DeepSeek-V4's manifold-constrained hyper-connections through
``tensorrt_llm._torch.modules.mhc``: every ``DeepseekV4DecoderLayer`` builds two
``mHC`` sites, ``hc_attn`` and ``hc_ffn``, through its ``_make_mhc`` hook
(models/modeling_deepseekv4.py:1923-1937, 1965, 1979@v1.4.0rc2), whose
``pre_mapping`` / ``post_mapping`` / ``fused_hc`` run the
``torch.ops.trtllm.mhc_*`` CUDA kernels (modules/mhc/hyper_connection.py,
modules/mhc/mhc_cuda.py). ``pre_mapping`` takes its GEMM+sqrsum backend
(CUDA-core FMA, or DeepGEMM TF32 split-K / no-split) and BigFuse block size
from the AutoTuner (mhc_cuda.py:263-343, 468-524). This collector builds the
sites through that hook and drives the module API without the attention and
MoE blocks between them.

Serving's mHC calls per forward of L layers (models/modeling_deepseekv4.py
@v1.4.0rc2). fused_hc is on by default on every SM (``_resolve_enable_fused_hc``
:332-337 reads TRTLLM_MHC_ENABLE_FUSED_HC, else the config's enable_fused_hc,
which no DeepSeek-V4 checkpoint config sets), and then:
  - layer 0 entry: ``hc_attn.pre_mapping`` and a standalone input_layernorm
    (``_entry_boundary`` :2283-2293);
  - every mid-layer boundary: ``hc_ffn.fused_hc`` with no norm folded, the
    post_attention_layernorm standalone (:2201-2207, forward_MoE :2336-2338);
  - every later layer entry: ``hc_attn.fused_hc`` with input_layernorm folded
    (:2273-2281; the previous layer defers its post_mapping,
    post_load_weights :3120-3142);
  - last layer: ``hc_ffn.post_mapping`` (:2236-2241), then the model's HCHead
    (DeepseekV4Model.hc_head, :2579, 2633-2634).
With fused_hc off, every site runs pre_mapping and post_mapping (:2208-2216,
:2236-2241, :2291): the path these rows time.

Deviation (every SM): the rows time ``pre_mapping`` with no norm folded and
``post_mapping`` for both sites of a layer, the decomposition the SDK's
DeepSeekV4 model bills (mhc_pre + attn_norm (ElementWise) + mhc_post,
src/aisimulate_core/sdk/models/deepseek_v4.py); folding the norm here would
double-count it. Serving runs 2L-1 of the 2L boundaries as fused_hc instead.
On SM100/103, fused_hc has tcgen05 TF32 MMA tactics for hidden 4096/5120/7168
(mhc_cuda.py:37-51, 589, 929-953): the rows over-charge the boundary. The
delta is unmeasured; mHC is 9-14% of the DSV4 per-layer cost on b200, an upper
bound on the error. Off the SM100 family (SM90, SM120, SM89), fused_hc has FMA
tactics only (mhc_cuda.py:916-927, 955-960) while pre_mapping keeps its
DeepGEMM tactics. On H20-3e (SM90, hidden 4096/7168, measured at 1.3.0rc20)
fused_hc took 0.80x of pre+post at M=16 but 3.8-6.5x at M=1024/8192, so there
the rows over-charge decode and under-charge prefill-sized boundaries. Retire
it by collecting DeepSeek-V4 fused_hc rows (the ``fused_post_pre`` op that
this table's GLM-5.3-Flash rows already carry) and switching the SDK consumer
(https://github.com/ai-dynamo/aiconfigurator/issues/1580).
"""

from __future__ import annotations

__compat__ = "trtllm==1.4.0rc2"

import argparse
import os
import sys
from collections.abc import Sequence
from types import SimpleNamespace

import torch

try:
    from case_generator import get_common_mhc_test_cases
    from registry_types import PerfFile

    from helper import benchmark_with_power, get_sm_version, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import get_common_mhc_test_cases
    from registry_types import PerfFile

    from helper import benchmark_with_power, get_sm_version, log_perf

from collector.op_plans import plan_cases

ARCHITECTURE = "DeepseekV4ForCausalLM"
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_HC_MULT = 4
# A decoder layer holds two sites, hc_attn and hc_ffn (models/modeling_deepseekv4.py:1965,
# 1979@v1.4.0rc2); every row times both.
MHC_NUM_SITES = 2
# hc_sinkhorn_iters of every DeepSeek-V4 checkpoint config
# (src/aisimulate_core/model_configs/deepseek-ai--DeepSeek-V4-{Flash,Pro}_config.json),
# which _make_mhc passes to mHC.
MHC_SINKHORN_ITERS = 20


def _parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _resolve_perf_path(output_path: str | None, filename: str | None) -> str:
    if filename is None:
        raise ValueError("filename is required")
    if not output_path:
        return filename
    if output_path.endswith(".txt"):
        return output_path
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, filename)


def get_mhc_module_test_cases() -> list[dict]:
    return plan_cases("mhc_module", "trtllm", get_sm_version())


def _default_num_tokens() -> list[int]:
    cases = get_common_mhc_test_cases()
    if not cases:
        raise RuntimeError("get_common_mhc_test_cases() returned no cases")
    return cases[0].num_tokens_list


def _make_mhc_site(hidden_size: int, hc_mult: int, *, device: str):
    """One mHC site, built by serving's own hook.

    ``DeepseekV4DecoderLayer._make_mhc`` (models/modeling_deepseekv4.py:1923-1937@v1.4.0rc2)
    reads only ``self.config`` (hc_mult, hidden_size, hc_sinkhorn_iters), so it runs on a
    stand-in that carries the checkpoint's values and sets the rest as serving does
    (post_mult_value=2.0, the module's default eps / norm_eps / sinkhorn_eps).
    ``mHC.__init__`` allocates ``fn`` [(2+mult)*mult, mult*hidden], ``base`` and ``scale``
    in fp32 (modules/mhc/hyper_connection.py:111-121). Random values stand in for the
    checkpoint weights: the kernels and the AutoTuner tactics key on shapes, never values.
    """
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4DecoderLayer

    config = SimpleNamespace(hc_mult=hc_mult, hidden_size=hidden_size, hc_sinkhorn_iters=MHC_SINKHORN_ITERS)
    site = DeepseekV4DecoderLayer._make_mhc(SimpleNamespace(config=config))
    with torch.no_grad():
        site.fn.normal_()
        site.base.normal_()
        site.scale.fill_(1.0)
    return site.to(device)


def _make_residual(num_tokens: int, hidden_size: int, hc_mult: int, *, device: str) -> torch.Tensor:
    """Residual stream in the exact ``pre_mapping`` contract
    (modules/mhc/hyper_connection.py:123-155@v1.4.0rc2): bfloat16 [..., mult, hidden],
    asserted at :130-132 and flattened to [num_tokens, mult, hidden] at :134, so
    collapsing batch*seq into ``num_tokens`` up front preserves the call.
    ``post_mapping`` takes the same layout (:482-507).
    """
    return torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)


def _autotune_pre(site_inputs) -> None:
    """Tune pre_mapping at the case's token count, as serving's warmup does.

    Serving runs its warmup forwards under autotune() (_run_autotuner_warmup,
    pyexecutor/engine/runners/decoder/warmup.py:1050-1104@v1.4.0rc2), where
    ``mhc_pre_mapping_fused`` (mhc_cuda.py:468-524) profiles every MhcPreMappingRunner
    tactic (FMA tile variants, plus DeepGEMM split-K / no-split when tf32_hc_prenorm_gemm
    imports, :322-343) at each token bucket up to the warmup's count. A call of M tokens
    then runs the winner of M's bucket (:241-249: powers of two up to 128, 128-token steps
    up to 8192, 1024-token steps above). ``skip_dynamic_tuning_buckets`` profiles only
    that bucket (and M itself): the same winner for M, without first profiling every
    bucket below it, which above 8192 tokens is up to ~500 buckets at near-M sizes.
    """
    from tensorrt_llm._torch.autotuner import autotune

    with torch.inference_mode(), autotune(skip_dynamic_tuning_buckets=True):
        for site, residual in site_inputs:
            site.pre_mapping(residual)
    torch.cuda.synchronize()


def _observe_pre_backend(site, residual) -> str:
    """Record which GEMM+sqrsum entry the tuned pre_mapping invokes.

    ``MhcPreMappingRunner.forward`` runs the tactic's backend through one of two
    module-level entries (mhc_cuda.py:363-373@v1.4.0rc2): ``mhc_gemm_rms_dg_cuda``
    (``dg_splitk`` with num_splits=16 or ``dg_nosplit`` with num_splits=1, DeepGEMM
    TF32) or ``mhc_gemm_rms_fma_cuda`` (``fma``, CUDA-core FMA). Observing the call
    is ground truth for ``kernel_source``, with no assumption about the AutoTuner's
    cache.
    """
    from tensorrt_llm._torch.modules.mhc import mhc_cuda

    calls: list[str] = []
    orig_fma = mhc_cuda.mhc_gemm_rms_fma_cuda
    orig_dg = mhc_cuda.mhc_gemm_rms_dg_cuda

    def _fma(*args, **kwargs):
        calls.append("fma")
        return orig_fma(*args, **kwargs)

    def _dg(*args, **kwargs):
        # Both DeepGEMM tactics pass num_splits by keyword (mhc_cuda.py:365-369).
        calls.append("dg_splitk" if int(kwargs["num_splits"]) > 1 else "dg_nosplit")
        return orig_dg(*args, **kwargs)

    mhc_cuda.mhc_gemm_rms_fma_cuda = _fma
    mhc_cuda.mhc_gemm_rms_dg_cuda = _dg
    try:
        with torch.no_grad():
            site.pre_mapping(residual)
    finally:
        mhc_cuda.mhc_gemm_rms_fma_cuda = orig_fma
        mhc_cuda.mhc_gemm_rms_dg_cuda = orig_dg

    backends = sorted(set(calls))
    if not backends:
        raise RuntimeError("pre_mapping dispatched no known GEMM+sqrsum entry point")
    return "+".join(backends)


def run_mhc_module(
    *,
    ops: Sequence[str],
    num_tokens_cases: Sequence[int] | None = None,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    hc_mult: int = DEFAULT_HC_MULT,
    device: str = "cuda:0",
    output_path: str | None = None,
    perf_filename: str | None = None,
    num_warmup: int = 5,
    num_iterations: int = 10,
) -> list[dict]:
    # TRT-LLM imports stay lazy in this module so that a mismatched install
    # fails inside collect.py's per-op error handling (after the __compat__
    # gate can label it) rather than at module import.
    import tensorrt_llm

    torch.cuda.set_device(device)
    hidden_size = int(hidden_size)
    hc_mult = int(hc_mult)
    token_cases = list(num_tokens_cases or _default_num_tokens())
    if "--smoke" in sys.argv and num_tokens_cases is None:
        token_cases = [16]

    results = []
    for op in ops:
        if op not in {"pre", "post"}:
            raise ValueError(f"unsupported mHC op: {op}")
        for num_tokens in token_cases:
            site_inputs = [
                (
                    _make_mhc_site(hidden_size, hc_mult, device=device),
                    _make_residual(num_tokens, hidden_size, hc_mult, device=device),
                )
                for _ in range(MHC_NUM_SITES)
            ]

            if op == "pre":
                _autotune_pre(site_inputs)
                kernel_source = f"trtllm_mhc_pre_{_observe_pre_backend(*site_inputs[0])}"

                def kernel_func(site_inputs=site_inputs):
                    with torch.no_grad():
                        return [site.pre_mapping(residual) for site, residual in site_inputs]

            else:
                # One CUDA kernel with no tactic axis (mhc_cuda.py:532-570@v1.4.0rc2,
                # mhc_post_mapping_cuda -> torch.ops.trtllm.mhc_post_mapping), so no
                # autotune pass; the pre_mapping below only builds inputs, and its tactic
                # cannot change post_mapping latency. Its layer_input stands in for the
                # attention / MoE output serving passes as ``x``: the same [M, hidden]
                # bfloat16 (models/modeling_deepseekv4.py:2210-2215, 2236-2241).
                kernel_source = "trtllm_mhc_post_mapping"
                with torch.no_grad():
                    post_inputs = [(site, residual, site.pre_mapping(residual)) for site, residual in site_inputs]
                torch.cuda.synchronize()

                def kernel_func(post_inputs=post_inputs):
                    with torch.no_grad():
                        return [
                            site.post_mapping(layer_input, residual, post_mix, comb_mix)
                            for site, residual, (post_mix, comb_mix, layer_input) in post_inputs
                        ]

            with benchmark_with_power(
                device=torch.device(device),
                kernel_func=kernel_func,
                num_warmups=num_warmup,
                num_runs=num_iterations,
                repeat_n=1,
                allow_graph_fail=False,
                use_cuda_graph=True,
            ) as result:
                pass
            latency = float(result["latency_ms"])
            log_perf(
                item_list=[
                    {
                        "architecture": ARCHITECTURE,
                        "num_tokens": num_tokens,
                        "num_sites": MHC_NUM_SITES,
                        "hc_mult": hc_mult,
                        "hidden_size": hidden_size,
                        "sinkhorn_iters": MHC_SINKHORN_ITERS,
                        "latency": f"{latency:.4f}",
                    }
                ],
                framework="TRTLLM",
                version=tensorrt_llm.__version__,
                device_name=torch.cuda.get_device_name(device),
                op_name=op,
                kernel_source=kernel_source,
                perf_filename=_resolve_perf_path(output_path, perf_filename or PerfFile.MHC_MODULE.value),
                power_stats=result.get("power_stats"),
            )
            print(f"[trtllm-mhc] op={op} tokens={num_tokens} kernel={kernel_source} latency={latency:.4f} ms")
            results.append({"op": op, "num_tokens": num_tokens, "kernel_source": kernel_source, "latency": latency})
            # kernel_func retains site_inputs/post_inputs through its default
            # argument; drop every reference before empty_cache() so the next
            # token case doesn't allocate on top of the previous one.
            del kernel_func
            if op == "post":
                del post_inputs
            del site_inputs
            torch.cuda.empty_cache()
    return results


def run_mhc_module_worker(
    op: str,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    output_path = os.path.dirname(perf_filename) or os.getcwd()
    run_mhc_module(
        ops=[op],
        num_tokens_cases=[num_tokens],
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        device=device,
        output_path=output_path,
        perf_filename=os.path.basename(perf_filename),
        num_warmup=3 if "--smoke" in sys.argv else 5,
        num_iterations=3 if "--smoke" in sys.argv else 10,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect TRT-LLM DeepSeek-V4 mHC module latency.")
    parser.add_argument("--op", choices=["pre", "post", "all"], default="all")
    parser.add_argument("--num-tokens", default="16")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--hc-mult", type=int, default=DEFAULT_HC_MULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)

    args = parser.parse_args()
    run_mhc_module(
        ops=["pre", "post"] if args.op == "all" else [args.op],
        num_tokens_cases=_parse_int_list(args.num_tokens),
        hidden_size=args.hidden_size,
        hc_mult=args.hc_mult,
        device=args.device,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
