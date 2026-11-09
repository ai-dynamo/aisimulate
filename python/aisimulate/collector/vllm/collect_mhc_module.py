# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mHC module collector for vLLM GLM-5.3-Flash (Glm5Next)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from importlib.metadata import version as get_version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases
from collector.registry_types import PerfFile
from collector.version_resolver import _check_compat

# The mhc family runs on 0.32.0 (framework_manifest.yaml family pin).
# Each architecture is additionally gated by
# ``_ARCHITECTURE_COMPAT`` below; an architecture that is not audited on the
# installed release raises instead of running.
# One release per file; earlier code is `git log -- <this file>` away.
__compat__ = "vllm==0.32.0"

# vLLM imports stay lazy in this module so that a mismatched install fails
# inside collect.py's per-op error handling (after the __compat__ gate can
# label it) rather than at module import.

DSV4_ARCHITECTURE = "DeepseekV4ForCausalLM"
GLM5NEXT_ARCHITECTURE = "Glm5NextForConditionalGeneration"

# Per-architecture audited vLLM releases. A case whose architecture is not
# audited on the installed release raises MhcRuntimeNotAuditedError (a
# classified failure) instead of running an unverified dispatch.
# DeepSeek-V4 is not audited here (collector/skips.py leaves its vLLM rows
# out). mhc_module_perf's architecture-free view (MhcKey = op_name, hc_mult,
# hidden_size; the DeepSeek-V4 consumer) would merge GLM-5.3-Flash's
# fused-site rows at (4, 4096) into one table, so DeepSeek-V4 rows stay on the
# 0.24.0-0.30.0 tables and their collector route is `git log -- <this file>`
# away. Re-admitting it needs an architecture-keyed DeepSeek-V4 lookup first
# (consumer change).
_ARCHITECTURE_COMPAT = {
    # Stock vllm 0.32.0 only (no overlay) — see the GLM audit below.
    GLM5NEXT_ARCHITECTURE: "vllm==0.32.0",
}

# GLM-5.3-Flash mHC call sites in stock vLLM 0.32.0 serving. vLLM paths below
# are @v0.32.0 (5df63a72) unless marked; "model.py" is
# vllm/models/glm5next/common/model.py. The registry maps
# Glm5NextForConditionalGeneration (and Glm5NextForCausalLM) to
# vllm.models.glm5next (vllm/model_executor/models/registry.py:121, 423-426),
# whose __init__ re-exports model.py (Glm5NextDecoderLayer.forward L518-623,
# Glm5NextModel.forward layer loop L823-833):
#   layer 0 attn:   hc_expand (L559-560) + standalone hc_pre with the fused
#                   input_layernorm (L561-569)
#   every other attn site and every ffn site: hc_fused_post_pre with the fused
#                   input/post_attention layernorm (L570-581, L596-607) — the
#                   previous sublayer's hc_post is deferred into this kernel
#   last layer:     standalone hc_post (L619) + hc_contract (L620)
# i.e. per forward with L=45 layers: pre=1, fused_post_pre=2L-1=89, post=1,
# expand=1, contract=1. With EAGLE3 speculative decoding (SupportsEagle3), each
# aux hidden-state layer adds one hc_post (Glm5NextModel._aux_post_op) and one
# hc_contract (L748, L773-791, L822-849); the counts above are for serving
# without EAGLE3 (aux_hidden_state_layers defaults to (),
# vllm/model_executor/models/interfaces.py:1571).
# MHCPreOp/MHCPostOp/MHCFusedPostPreOp report enabled() == True, so they take
# forward_cuda whatever the compilation config says, and it dispatches
# unconditionally to torch.ops.vllm.mhc_{pre,post,fused_post_pre}_tilelang
# (vllm/model_executor/layers/mhc.py L118-150, L573-586, L778-818);
# hc_expand/hc_contract are plain torch (mhc.py L1013-1020). Glm5NextModel has
# no @support_torch_compile, so serving runs both as standalone eager torch
# ops, as here (graph mode: see the note in run_glm5next_mhc_module).
#
# mHC is replicated per rank: the decoder layer runs it on the rank's full
# token batch unless sequence-parallel MoE is on (model.py L372, L583-594,
# Glm5NextModel L761-763/L819-820), and ParallelConfig.use_sequence_parallel_moe
# requires a listed all2all backend, EP, TP>1 and DP>1
# (vllm/config/parallel.py:763-781). Rows carry no TP dimension; pure-TP
# serving reads the same num_tokens rows.
#
# 0.32.0 audit (source-only: v0.32.0 == 5df63a72 vs v0.31.0 db9527a4):
# - vllm/transformers_utils/configs/glm5_next.py is gone. Serving loads
#   transformers' Glm5NextConfig through AutoConfig and reads its text config
#   (model.py:9, L694; requirements/common.txt pins transformers >= 5.16.1,
#   < 5.20.0). The mHC fields keep the checkpoint's names: hc_mult (L444,
#   L450), hc_eps (L446), hc_sinkhorn_iters (L447). The post multiplier is the
#   module constant _MHC_POST_MULT_VALUE (L108, L448), no longer a config
#   field; tau (_MHC_TAU) feeds no kernel.
# - hc_pre / hc_post / hc_fused_post_pre (L625-687) and every call site above
#   are unchanged apart from line moves; the EAGLE3 aux path is new.
# - layers/mhc.py changes only the ROCm AITER path (forward_hip);
#   kernels/mhc/tilelang.py and tilelang_kernels.py are byte-identical, and
#   requirements/cuda.txt pins tilelang==0.1.12 at both tags.
#   utils/deep_gemm.py is_deep_gemm_supported (L110-116) and
#   CudaPlatform.support_deep_gemm (SM90, SM100 family, SM120 family;
#   vllm/platforms/cuda.py:689-695) are unchanged.
# - DeepGEMM moves e1f418c2 -> 1e1842a8 (cmake/external_projects/
#   deepgemm.cmake:37): the binding moves from pybind11 to TORCH_LIBRARY
#   (csrc/apis/hyperconnection.hpp). The SM90 and SM100 tf32_hc_prenorm_gemm
#   kernels change only their host includes; the SM120 kernel adds a
#   fence_view_async_shared() before each stage release
#   (deep_gemm/include/deep_gemm/impls/sm120_tf32_hc_prenorm_gemm.cuh:231-232
#   @1e1842a8).
#
# Row convention (shared with the DeepSeek-V4 rows of this table): pre, post
# and fused_post_pre rows time BOTH per-layer sites (attention params +
# input_layernorm, FFN params + post_attention_layernorm) in one graph, so
# num_sites=2; expand/contract exist once per forward, so num_sites=1. The
# consumer scales by the per-forward call counts above.
GLM5NEXT_VLLM_OPS = ("pre", "post", "fused_post_pre", "expand", "contract")
_GLM5NEXT_NUM_SITES = {"pre": 2, "post": 2, "fused_post_pre": 2, "expand": 1, "contract": 1}
# GLM-5.3-Flash geometry (both packaged configs); the CLI defaults.
_GLM5NEXT_HIDDEN_SIZE = 4096
_GLM5NEXT_HC_MULT = 4

_MODEL_CONFIG_DIR = REPO_ROOT / "src" / "aisimulate_core" / "model_configs"


class MhcRuntimeNotAuditedError(RuntimeError):
    """The installed vLLM release is not audited for this architecture's mHC dispatch."""


def _require_audited_runtime(architecture: str, runtime_version: str) -> None:
    compat = _ARCHITECTURE_COMPAT.get(architecture)
    if compat is None:
        raise MhcRuntimeNotAuditedError(f"no audited vLLM mHC dispatch for architecture {architecture!r}")
    if not _check_compat(compat, runtime_version):
        raise MhcRuntimeNotAuditedError(
            f"vLLM {runtime_version} is not an audited mHC runtime for {architecture} (audited: {compat})"
        )


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


def _init_cuda(device: str) -> None:
    from collector.vllm.utils import setup_distributed
    from vllm.v1.worker.workspace import init_workspace_manager

    setup_distributed(device)
    torch.cuda.set_device(device)
    init_workspace_manager(torch.device(device))


def get_mhc_module_test_cases() -> list[dict]:
    return plan_cases("mhc_module", "vllm", get_sm_version())


def _read_packaged_model_config(model_path: str) -> dict:
    """Load AIC's packaged ``model_configs/<id>_config.json`` (no HF download)."""
    config_file = _MODEL_CONFIG_DIR / (model_path.replace("/", "--") + "_config.json")
    if not config_file.is_file():
        raise FileNotFoundError(f"AIC packaged config not found for model_path={model_path!r}: expected {config_file}")
    return json.loads(config_file.read_text(encoding="utf-8"))


def _glm5next_text_config(model_path: str):
    """Resolve the text config the way serving does.

    vLLM registers no config class for model_type glm5_next, so
    HFConfigParser.parse loads transformers' Glm5NextConfig through
    AutoConfig.from_pretrained (vllm/transformers_utils/config.py:399), and the
    decoder reads ``model_config.hf_text_config``, i.e. ``get_text_config()``
    (get_hf_text_config, config.py:1195-1209; model.py:694).
    """
    from transformers import Glm5NextConfig

    raw = _read_packaged_model_config(model_path)
    if raw.get("architectures") != [GLM5NEXT_ARCHITECTURE]:
        raise ValueError(f"{model_path!r} is not a {GLM5NEXT_ARCHITECTURE} config: {raw.get('architectures')!r}")
    text_config = Glm5NextConfig.from_dict(raw).get_text_config()
    # Glm5NextDecoderLayer.__init__ reads the checkpoint's own `mhc` flag (model.py:369).
    if not text_config.mhc:
        raise ValueError(f"{model_path!r} text_config has mhc disabled")
    return text_config


def _build_glm5next_sites(text_config, *, device: str):
    """Build the mHC state of one Glm5NextDecoderLayer without its attention/MLP.

    The returned module binds the serving decoder layer's own ``hc_pre`` /
    ``hc_post`` / ``hc_fused_post_pre`` methods (model.py L625-687), so every
    kernel argument (eps values, post multiplier, Sinkhorn repeat, n_splits,
    tile_n, fused-norm weight) is the one serving passes. The attributes those
    methods read mirror ``Glm5NextDecoderLayer.__init__`` (model.py L366,
    L442-472): fp32 hc fn/base/scale and the two RMSNorms (L429, L438-440)
    whose weights are fused into the pre kernels.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp, MHCPostOp, MHCPreOp
    from vllm.models.glm5next.common.model import _MHC_POST_MULT_VALUE, Glm5NextDecoderLayer

    class _Glm5NextMhcSites(torch.nn.Module):
        hc_pre = Glm5NextDecoderLayer.hc_pre
        hc_post = Glm5NextDecoderLayer.hc_post
        hc_fused_post_pre = Glm5NextDecoderLayer.hc_fused_post_pre

        def __init__(self) -> None:
            super().__init__()
            hidden_size = int(text_config.hidden_size)
            self.rms_norm_eps = text_config.rms_norm_eps
            self.hc_eps = text_config.hc_eps
            self.mhc_sinkhorn_iterations = text_config.hc_sinkhorn_iters
            self.mhc_post_mult_value = _MHC_POST_MULT_VALUE
            self.n = int(text_config.hc_mult)
            mix_hc = (2 + self.n) * self.n
            d_model = self.n * hidden_size
            for site in ("attn", "ffn"):
                setattr(self, f"hc_{site}_fn", torch.nn.Parameter(torch.randn(mix_hc, d_model, dtype=torch.float32)))
                setattr(self, f"hc_{site}_base", torch.nn.Parameter(torch.randn(mix_hc, dtype=torch.float32)))
                setattr(self, f"hc_{site}_scale", torch.nn.Parameter(torch.ones(3, dtype=torch.float32)))
            # Serving builds these under the model dtype (bfloat16 for both GLM
            # checkpoints), so the fused-norm weight needs no per-call cast.
            self.input_layernorm = RMSNorm(hidden_size, eps=text_config.rms_norm_eps, dtype=torch.bfloat16)
            self.post_attention_layernorm = RMSNorm(hidden_size, eps=text_config.rms_norm_eps, dtype=torch.bfloat16)
            self.mhc_pre_op = MHCPreOp()
            self.mhc_post_op = MHCPostOp()
            self.mhc_fused_post_pre_op = MHCFusedPostPreOp()

    with set_current_vllm_config(VllmConfig()):
        sites = _Glm5NextMhcSites().to(device)
    for custom_op in (sites.mhc_pre_op, sites.mhc_post_op, sites.mhc_fused_post_pre_op):
        # CustomOp dispatch is resolved at construction; record/verify it.
        if custom_op._forward_method.__func__ is not type(custom_op).forward_cuda:
            raise RuntimeError(f"{type(custom_op).__name__} did not dispatch to forward_cuda")
    return sites


def _glm5next_site_params(sites):
    return (
        (sites.hc_attn_fn, sites.hc_attn_scale, sites.hc_attn_base, sites.input_layernorm),
        (sites.hc_ffn_fn, sites.hc_ffn_scale, sites.hc_ffn_base, sites.post_attention_layernorm),
    )


def _glm5next_pre_outputs(sites, residuals):
    return [
        sites.hc_pre(
            residual,
            fn,
            scale,
            base,
            norm_weight=norm.weight.data,
            norm_eps=norm.variance_epsilon,
        )
        for residual, (fn, scale, base, norm) in zip(residuals, _glm5next_site_params(sites), strict=True)
    ]


def _glm5next_kernel(sites, op: str, num_tokens: int, *, device: str):
    """Return a zero-arg callable that runs ``op`` exactly as serving calls it."""
    from vllm.model_executor.layers.mhc import hc_contract, hc_expand

    hidden_size = sites.input_layernorm.weight.shape[0]
    n = sites.n

    def residual():
        return torch.randn(num_tokens, n, hidden_size, dtype=torch.bfloat16, device=device)

    def layer_out():
        return torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    if op == "expand":
        # model.py L559-560: embedding output [T, H] -> [T, n, H].
        x = layer_out()
        return lambda: hc_expand(x, n)
    if op == "contract":
        # model.py L620: last layer's post output [T, n, H] -> [T, H].
        x = residual()
        return lambda: hc_contract(x, n)

    residuals = [residual() for _ in range(_GLM5NEXT_NUM_SITES[op])]
    if op == "pre":
        return lambda: _glm5next_pre_outputs(sites, residuals)

    with torch.no_grad():
        state = [
            (post, comb, res)
            for (post, comb, _x), res in zip(_glm5next_pre_outputs(sites, residuals), residuals, strict=True)
        ]
    outs = [layer_out() for _ in state]
    torch.cuda.synchronize()
    if op == "post":
        # model.py L619: hc_post(x, residual, post, comb).
        return lambda: [sites.hc_post(x, res, post, comb) for x, (post, comb, res) in zip(outs, state, strict=True)]
    if op == "fused_post_pre":
        # model.py L570-581 (attention site) and L596-607 (FFN site).
        params = _glm5next_site_params(sites)
        return lambda: [
            sites.hc_fused_post_pre(
                x,
                res,
                post,
                comb,
                fn,
                scale,
                base,
                norm_weight=norm.weight.data,
                norm_eps=norm.variance_epsilon,
            )
            for x, (post, comb, res), (fn, scale, base, norm) in zip(outs, state, params, strict=True)
        ]
    raise ValueError(f"unsupported GLM mHC op: {op}")


def _glm5next_kernel_source(op: str, num_tokens: int, hidden_size: int, hc_mult: int) -> str:
    """Name what actually runs, including the internal GEMM/fusion selection."""
    if op == "expand":
        return "vllm.model_executor.layers.mhc.hc_expand"
    if op == "contract":
        return "vllm.model_executor.layers.mhc.hc_contract"
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    # _hc_prenorm_gemm_outputs (kernels/mhc/tilelang.py L22-71) uses
    # DeepGEMM tf32_hc_prenorm_gemm when is_deep_gemm_supported() (L38), else
    # the TileLang prenorm GEMM.
    gemm = "deepgemm" if is_deep_gemm_supported() else "tilelang"
    if op == "pre":
        return f"vllm.mhc_pre_tilelang[prenorm_gemm={gemm},norm=fused]"
    if op == "post":
        return "vllm.mhc_post_tilelang"
    if op == "fused_post_pre":
        from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_fused_post_pre_split_config

        # mhc_fused_post_pre_tilelang (tilelang.py L814-983; branch
        # L893/L914-975) picks the fused post+GEMM kernel when a split config
        # exists (tilelang_kernels.py L67-95: num_tokens <= 32 and
        # hidden_size % 128 == 0), else post + prenorm GEMM.
        if mhc_fused_post_pre_split_config(num_tokens, hidden_size, hc_mult) is not None:
            return "vllm.mhc_fused_post_pre_tilelang[fused_post_gemm,norm=fused]"
        return f"vllm.mhc_fused_post_pre_tilelang[post+prenorm_gemm={gemm},norm=fused]"
    raise ValueError(f"unsupported GLM mHC op: {op}")


def run_glm5next_mhc_module(
    *,
    ops: Sequence[str],
    num_tokens_cases: Sequence[int],
    model_path: str,
    hidden_size: int,
    hc_mult: int,
    device: str = "cuda:0",
    output_path: str | None = None,
    perf_filename: str | None = None,
    num_warmup: int = 5,
    num_iterations: int = 10,
) -> list[dict]:
    version = get_version("vllm")
    _require_audited_runtime(GLM5NEXT_ARCHITECTURE, version)
    text_config = _glm5next_text_config(model_path)
    config_shape = (int(text_config.hidden_size), int(text_config.hc_mult))
    if config_shape != (int(hidden_size), int(hc_mult)):
        raise ValueError(f"case shape {(hidden_size, hc_mult)} does not match {model_path} config {config_shape}")
    _init_cuda(device)
    sites = _build_glm5next_sites(text_config, device=device)
    # The repeat count the kernels receive (hc_pre / hc_fused_post_pre).
    sinkhorn_iters = int(sites.mhc_sinkhorn_iterations)

    results = []
    for op in ops:
        if op not in GLM5NEXT_VLLM_OPS:
            raise ValueError(f"unsupported GLM mHC op: {op}")
        for num_tokens in num_tokens_cases:
            kernel = _glm5next_kernel(sites, op, int(num_tokens), device=device)

            def kernel_func(kernel=kernel):
                with torch.no_grad():
                    return kernel()

            # Deviation (graph mode): serving runs this architecture on
            # breakable CUDA graphs with torch.compile off (config/vllm.py:74-121
            # lists it on CUDA; L812-840 enables them and sets
            # CompilationMode.NONE). A step of up to the capture ceiling
            # (min(2 * max_num_seqs * decode_query_len, 1024 on the SM100
            # family else 512), config/vllm.py:2357-2364) replays graph
            # segments, and mHC replays inside them: no mHC op carries
            # @eager_break_during_capture (the breaks sit in attention: the KDA
            # core glm5next/common/kda.py:501, the MLA attention op
            # model_executor/layers/attention/mla_attention.py:1462 and the
            # sparse indexer glm5next/nvidia/sparse_indexer.py:100). A larger
            # step runs mHC eagerly. Every row here is one CUDA graph, so rows above the
            # ceiling leave out mHC's host launch gaps (the Python wrappers and
            # 1-3 kernel launches per site): lower than serving by that
            # overhead whenever the host, not the GPU, bounds the step. The
            # kernels and their arguments match (no capture-dependent branch in
            # the mHC path). Retired by the pending measurement-method decision.
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
            if not result.get("used_cuda_graph", False):
                raise RuntimeError("benchmark_with_power did not use CUDA Graph")
            latency = float(result["latency_ms"])
            if not log_perf(
                item_list=[
                    {
                        "architecture": GLM5NEXT_ARCHITECTURE,
                        "num_tokens": num_tokens,
                        "num_sites": _GLM5NEXT_NUM_SITES[op],
                        "hc_mult": hc_mult,
                        "hidden_size": hidden_size,
                        "sinkhorn_iters": sinkhorn_iters,
                        "latency": f"{latency:.4f}",
                    }
                ],
                framework="VLLM",
                version=version,
                device_name=torch.cuda.get_device_name(device),
                op_name=op,
                kernel_source=_glm5next_kernel_source(op, int(num_tokens), hidden_size, hc_mult),
                perf_filename=_resolve_perf_path(output_path, perf_filename or PerfFile.MHC_MODULE.value),
                power_stats=result.get("power_stats"),
            ):
                raise RuntimeError("Failed to persist vLLM GLM mHC performance row")
            print(f"[vllm-mhc] arch={GLM5NEXT_ARCHITECTURE} op={op} tokens={num_tokens} latency={latency:.4f} ms")
            results.append({"op": op, "num_tokens": num_tokens, "latency": latency})
            del kernel, kernel_func
            torch.cuda.empty_cache()
    return results


def run_mhc_module_worker(
    op: str,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    architecture: str = DSV4_ARCHITECTURE,
    model_path: str | None = None,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    # The op plan gives DeepSeek-V4 rows no architecture (the default above);
    # they, and any other architecture without a route here, raise.
    if architecture != GLM5NEXT_ARCHITECTURE:
        raise MhcRuntimeNotAuditedError(f"no audited vLLM mHC dispatch for architecture {architecture!r}")
    if not model_path:
        raise ValueError("GLM mHC cases require a model_path")
    output_path = os.path.dirname(perf_filename) or os.getcwd()
    num_warmup = 3 if "--smoke" in sys.argv else 5
    num_iterations = 3 if "--smoke" in sys.argv else 10
    run_glm5next_mhc_module(
        ops=[op],
        num_tokens_cases=[num_tokens],
        model_path=model_path,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        device=device,
        output_path=output_path,
        perf_filename=os.path.basename(perf_filename),
        num_warmup=num_warmup,
        num_iterations=num_iterations,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect vLLM GLM-5.3-Flash mHC module latency.")
    parser.add_argument("--model-path", required=True, help="Packaged AIC model id")
    parser.add_argument("--op", choices=[*GLM5NEXT_VLLM_OPS, "all"], default="all")
    parser.add_argument("--num-tokens", default="16")
    parser.add_argument("--hidden-size", type=int, default=_GLM5NEXT_HIDDEN_SIZE)
    parser.add_argument("--hc-mult", type=int, default=_GLM5NEXT_HC_MULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)

    args = parser.parse_args()
    run_glm5next_mhc_module(
        ops=list(GLM5NEXT_VLLM_OPS) if args.op == "all" else [args.op],
        num_tokens_cases=_parse_int_list(args.num_tokens),
        model_path=args.model_path,
        hidden_size=args.hidden_size,
        hc_mult=args.hc_mult,
        device=args.device,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
