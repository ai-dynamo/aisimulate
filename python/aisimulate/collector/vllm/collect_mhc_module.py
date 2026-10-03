# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mHC module collector for vLLM DeepSeek-V4 and GLM-5.3-Flash (Glm5Next)."""

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
from collector.case_generator import get_common_mhc_test_cases
from collector.helper import benchmark_with_power, log_perf
from collector.registry_types import PerfFile
from collector.version_resolver import _check_compat

# B200 0.25.0 qualification (installed vLLM dd10e03f9), job 1968046:
# mHC: 4/4 representative shape/operation cases. The native framework
# builders/selectors remain authoritative; no kernel fallback is introduced.
# The campaign manifest still selects one exact release per run.
#
# The file-level range spans the two audited routes; each architecture is
# additionally gated to its own audited release(s) by ``_ARCHITECTURE_COMPAT``
# below, so DeepSeek-V4 behaviour on 0.24.0-0.25.0 is unchanged and neither
# architecture runs on an unaudited release in between (it raises instead).
__compat__ = "vllm>=0.24.0,<=0.30.0"

# vLLM imports stay lazy in this module so that a mismatched install fails
# inside collect.py's per-op error handling (after the __compat__ gate can
# label it) rather than at module import.
_MHC_TILELANG_KERNELS: tuple | None = None


def _mhc_tilelang_kernels():
    global _MHC_TILELANG_KERNELS
    if _MHC_TILELANG_KERNELS is None:
        from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang, mhc_pre_tilelang

        _MHC_TILELANG_KERNELS = (mhc_pre_tilelang, mhc_post_tilelang)
    return _MHC_TILELANG_KERNELS


DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_HC_MULT = 4
ARCHITECTURE = "DeepseekV4ForCausalLM"
MHC_NUM_SITES = 2
MHC_SINKHORN_ITERS = 20
MHC_EPS = 1.0e-6

GLM5NEXT_ARCHITECTURE = "Glm5NextForConditionalGeneration"

# Per-architecture audited vLLM releases. A case whose architecture is not
# audited on the installed release raises MhcRuntimeNotAuditedError (a
# classified failure) instead of running an unverified dispatch.
_ARCHITECTURE_COMPAT = {
    ARCHITECTURE: "vllm>=0.24.0,<=0.25.0",
    # vllm 0.30.0 (+glm53tail overlay, which patches only the KPool indexer and
    # kv_cache_interface; neither is on the mHC path) — see the GLM audit below.
    GLM5NEXT_ARCHITECTURE: "vllm==0.30.0",
}

# GLM-5.3-Flash mHC call sites in vLLM 0.30.0 serving. The registry maps
# Glm5NextForConditionalGeneration to vllm.models.glm5next
# (vllm/model_executor/models/registry.py:429), which re-exports
# vllm/models/glm5next/nvidia/model.py (sha256 d7353ea0..., installed
# 0.30.0+glm53tail.eb4704514fdf; Glm5NextDecoderLayer.forward L450-555,
# Glm5NextModel.forward layer loop L729):
#   layer 0 attn:   hc_expand (L492) + standalone hc_pre with the fused
#                   input_layernorm (L494-501)
#   every other attn site and every ffn site: hc_fused_post_pre with the fused
#                   input/post_attention layernorm (L503-513, L529-539) — the
#                   previous sublayer's hc_post is deferred into this kernel
#   last layer:     standalone hc_post (L551) + hc_contract (L552)
# i.e. per forward with L=45 layers: pre=1, fused_post_pre=2L-1=89, post=1,
# expand=1, contract=1. MHCPreOp/MHCPostOp/MHCFusedPostPreOp.forward_cuda
# dispatch unconditionally to torch.ops.vllm.mhc_{pre,post,fused_post_pre}_tilelang
# (vllm/model_executor/layers/mhc.py sha256 923828e7..., L110-138, L533-542,
# L738-774); hc_expand/hc_contract are plain torch (mhc.py L969-976).
#
# Row convention (shared with the DeepSeek-V4 rows of this table): pre, post
# and fused_post_pre rows time BOTH per-layer sites (attention params +
# input_layernorm, FFN params + post_attention_layernorm) in one graph, so
# num_sites=2; expand/contract exist once per forward, so num_sites=1. The
# consumer scales by the per-forward call counts above.
GLM5NEXT_VLLM_OPS = ("pre", "post", "fused_post_pre", "expand", "contract")
_GLM5NEXT_NUM_SITES = {"pre": 2, "post": 2, "fused_post_pre": 2, "expand": 1, "contract": 1}

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


def _active_mhc_common_cases():
    # Architecture is part of the invocation identity: DeepSeek-V4-Flash and
    # GLM-5.3-Flash share (hidden_size, hc_mult) but run different call sites.
    seen: set[tuple[str, str, int, int]] = set()
    for case in get_common_mhc_test_cases():
        key = (case.architecture, case.phase, case.hidden_size, case.hc_mult)
        if key in seen:
            continue
        seen.add(key)
        yield case


def get_mhc_module_test_cases() -> list[dict]:
    cases: list[dict] = []
    glm5next_profiles: dict[tuple[int, int], tuple[str, list[int]]] = {}
    for case in _active_mhc_common_cases():
        num_tokens_list = [16] if "--smoke" in sys.argv else case.num_tokens_list
        if case.architecture == GLM5NEXT_ARCHITECTURE:
            # GLM call sites are a framework-dispatch fact (see the audit
            # above), not the generator's pre/post phase pair: collect the
            # model profile once and expand every serving op below.
            glm5next_profiles.setdefault((case.hidden_size, case.hc_mult), (case.model_name, num_tokens_list))
            continue
        for num_tokens in num_tokens_list:
            cases.append(
                {
                    "id": f"mhc_{case.phase}_hs{case.hidden_size}_hcm{case.hc_mult}_{num_tokens}",
                    "params": [case.phase, num_tokens, case.hidden_size, case.hc_mult],
                }
            )
    for (hidden_size, hc_mult), (model_path, num_tokens_list) in glm5next_profiles.items():
        for op in GLM5NEXT_VLLM_OPS:
            for num_tokens in num_tokens_list:
                cases.append(
                    {
                        "id": f"mhc_glm5next_{op}_hs{hidden_size}_hcm{hc_mult}_{num_tokens}",
                        "params": [op, num_tokens, hidden_size, hc_mult, GLM5NEXT_ARCHITECTURE, model_path],
                    }
                )
    return cases


def _default_num_tokens() -> list[int]:
    cases = get_common_mhc_test_cases()
    if not cases:
        raise RuntimeError("get_common_mhc_test_cases() returned no cases")
    return cases[0].num_tokens_list


def _make_mhc_tensors(num_tokens: int, hidden_size: int, hc_mult: int, *, device: str):
    mix_hc = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * hidden_size
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    fn = torch.randn(mix_hc, hc_dim, dtype=torch.float32, device=device)
    base = torch.randn(mix_hc, dtype=torch.float32, device=device)
    scale = torch.ones(3, dtype=torch.float32, device=device)
    return residual, fn, base, scale


def _mhc_pre(residual, fn, base, scale):
    # KNOWN GAP vs vLLM 0.24 serving: the NVIDIA DeepSeek-V4 model calls
    # mhc_pre_tilelang standalone only on the FIRST layer and always with
    # norm_weight=attn_norm.weight (fused-RMSNorm big_fuse variant), and every
    # subsequent layer boundary runs the fused mhc_fused_post_pre_tilelang
    # (vllm/models/deepseek_v4/nvidia/model.py:854-890 @0.24.0). This
    # collector measures the norm_weight=None variant because the SDK's
    # DeepSeekV4 model composes mhc_pre + attn_norm (ElementWise) + mhc_post
    # as separate per-layer ops (src/aisimulate/sdk/models/deepseek_v4.py)
    # — fusing the norm here would double-count it downstream.
    # Measured impact (H20, hc_mult=4, hidden=4096, T=1k/8k, 2026-07):
    # fused(post+pre+norm) matches pre(no-norm)+post within 1-2%, and the
    # fused norm adds only 2-3% to pre — so this decomposition tracks the
    # fused serving path closely; the SDK's separately-billed attn_norm is
    # the only (small) over-count. Aligning row semantics with the fused
    # serving path is a coordinated producer+consumer contract change; do
    # not switch variants unilaterally.
    mhc_pre_tilelang, _ = _mhc_tilelang_kernels()
    post, comb, layer_input = mhc_pre_tilelang(
        residual,
        fn,
        scale,
        base,
        MHC_EPS,
        MHC_EPS,
        MHC_EPS,
        2.0,
        MHC_SINKHORN_ITERS,
    )
    return layer_input, post, comb


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
    _require_audited_runtime(ARCHITECTURE, get_version("vllm"))
    _init_cuda(device)
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
                _make_mhc_tensors(num_tokens, hidden_size, hc_mult, device=device) for _ in range(MHC_NUM_SITES)
            ]
            if op == "pre":

                def kernel_func(site_inputs=site_inputs):
                    with torch.no_grad():
                        return [_mhc_pre(residual, fn, base, scale) for residual, fn, base, scale in site_inputs]

            else:
                _, mhc_post_tilelang = _mhc_tilelang_kernels()
                with torch.no_grad():
                    post_inputs = [
                        (_mhc_pre(residual, fn, base, scale), residual) for residual, fn, base, scale in site_inputs
                    ]
                torch.cuda.synchronize()

                def kernel_func(post_inputs=post_inputs, mhc_post_tilelang=mhc_post_tilelang):
                    with torch.no_grad():
                        return [mhc_post_tilelang(x, residual, post, comb) for (x, post, comb), residual in post_inputs]

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
                framework="VLLM",
                version=get_version("vllm"),
                device_name=torch.cuda.get_device_name(device),
                op_name=op,
                kernel_source=f"vllm.model_executor.kernels.mhc.tilelang.mhc_{op}_tilelang",
                perf_filename=_resolve_perf_path(output_path, perf_filename or PerfFile.MHC_MODULE.value),
                power_stats=result.get("power_stats"),
            )
            print(f"[vllm-mhc] op={op} tokens={num_tokens} latency={latency:.4f} ms")
            results.append({"op": op, "num_tokens": num_tokens, "latency": latency})
            del site_inputs
            torch.cuda.empty_cache()
    return results


def _read_packaged_model_config(model_path: str) -> dict:
    """Load AIC's packaged ``model_configs/<id>_config.json`` (no HF download)."""
    config_file = _MODEL_CONFIG_DIR / (model_path.replace("/", "--") + "_config.json")
    if not config_file.is_file():
        raise FileNotFoundError(f"AIC packaged config not found for model_path={model_path!r}: expected {config_file}")
    return json.loads(config_file.read_text(encoding="utf-8"))


def _glm5next_text_config(model_path: str):
    """Resolve the text config through vLLM's own Glm5NextConfig (HF field aliases included)."""
    from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

    raw = _read_packaged_model_config(model_path)
    if raw.get("architectures") != [GLM5NEXT_ARCHITECTURE]:
        raise ValueError(f"{model_path!r} is not a {GLM5NEXT_ARCHITECTURE} config: {raw.get('architectures')!r}")
    text_config = Glm5NextConfig(**raw).text_config
    if not text_config.mhc:
        raise ValueError(f"{model_path!r} text_config has mhc disabled")
    return text_config


def _build_glm5next_sites(text_config, *, device: str):
    """Build the mHC state of one Glm5NextDecoderLayer without its attention/MLP.

    The returned module binds the serving decoder layer's own ``hc_pre`` /
    ``hc_post`` / ``hc_fused_post_pre`` methods (model.py L557-616), so every
    kernel argument (eps values, post multiplier, Sinkhorn repeat, n_splits,
    tile_n, fused-norm weight) is the one serving passes. Parameters mirror
    ``Glm5NextDecoderLayer.__init__`` (model.py L364-404): fp32 hc fn/base/scale
    and the two RMSNorms whose weights are fused into the pre kernels.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp, MHCPostOp, MHCPreOp
    from vllm.models.glm5next.nvidia.model import Glm5NextDecoderLayer

    class _Glm5NextMhcSites(torch.nn.Module):
        hc_pre = Glm5NextDecoderLayer.hc_pre
        hc_post = Glm5NextDecoderLayer.hc_post
        hc_fused_post_pre = Glm5NextDecoderLayer.hc_fused_post_pre

        def __init__(self) -> None:
            super().__init__()
            hidden_size = int(text_config.hidden_size)
            self.rms_norm_eps = text_config.rms_norm_eps
            self.hc_eps = text_config.hc_eps
            self.mhc_sinkhorn_iterations = text_config.mhc_sinkhorn_iterations
            self.mhc_post_mult_value = text_config.mhc_post_mult_value
            self.n = int(text_config.mhc_num_residual_streams)
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
        # model.py L492: embedding output [T, H] -> [T, n, H].
        x = layer_out()
        return lambda: hc_expand(x, n)
    if op == "contract":
        # model.py L552: last layer's post output [T, n, H] -> [T, H].
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
        # model.py L551: hc_post(x, residual, post, comb).
        return lambda: [sites.hc_post(x, res, post, comb) for x, (post, comb, res) in zip(outs, state, strict=True)]
    if op == "fused_post_pre":
        # model.py L503-513 (attention site) and L529-539 (FFN site).
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

    # _hc_prenorm_gemm_outputs (kernels/mhc/tilelang.py L22-60) uses DeepGEMM
    # tf32_hc_prenorm_gemm when supported, else the TileLang prenorm GEMM.
    gemm = "deepgemm" if is_deep_gemm_supported() else "tilelang"
    if op == "pre":
        return f"vllm.mhc_pre_tilelang[prenorm_gemm={gemm},norm=fused]"
    if op == "post":
        return "vllm.mhc_post_tilelang"
    if op == "fused_post_pre":
        from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_fused_post_pre_split_config

        # mhc_fused_post_pre_tilelang (tilelang.py L812-981) picks the fused
        # post+GEMM kernel when a split config exists, else post + prenorm GEMM.
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
    config_shape = (int(text_config.hidden_size), int(text_config.mhc_num_residual_streams))
    if config_shape != (int(hidden_size), int(hc_mult)):
        raise ValueError(f"case shape {(hidden_size, hc_mult)} does not match {model_path} config {config_shape}")
    _init_cuda(device)
    sites = _build_glm5next_sites(text_config, device=device)
    sinkhorn_iters = int(text_config.mhc_sinkhorn_iterations)

    results = []
    for op in ops:
        if op not in GLM5NEXT_VLLM_OPS:
            raise ValueError(f"unsupported GLM mHC op: {op}")
        for num_tokens in num_tokens_cases:
            kernel = _glm5next_kernel(sites, op, int(num_tokens), device=device)

            def kernel_func(kernel=kernel):
                with torch.no_grad():
                    return kernel()

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
    architecture: str = ARCHITECTURE,
    model_path: str | None = None,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    output_path = os.path.dirname(perf_filename) or os.getcwd()
    num_warmup = 3 if "--smoke" in sys.argv else 5
    num_iterations = 3 if "--smoke" in sys.argv else 10
    if architecture == GLM5NEXT_ARCHITECTURE:
        if not model_path:
            raise ValueError("GLM mHC cases require a model_path")
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
        return
    if architecture != ARCHITECTURE:
        raise MhcRuntimeNotAuditedError(f"no audited vLLM mHC dispatch for architecture {architecture!r}")
    run_mhc_module(
        ops=[op],
        num_tokens_cases=[num_tokens],
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        device=device,
        output_path=output_path,
        perf_filename=os.path.basename(perf_filename),
        num_warmup=num_warmup,
        num_iterations=num_iterations,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect vLLM DeepSeek-V4 / GLM-5.3-Flash mHC module latency.")
    parser.add_argument("--architecture", choices=[ARCHITECTURE, GLM5NEXT_ARCHITECTURE], default=ARCHITECTURE)
    parser.add_argument("--model-path", default=None, help="Packaged AIC model id (required for GLM)")
    parser.add_argument("--op", choices=["pre", "post", "fused_post_pre", "expand", "contract", "all"], default="all")
    parser.add_argument("--num-tokens", default="16")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--hc-mult", type=int, default=DEFAULT_HC_MULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)

    args = parser.parse_args()
    if args.architecture == GLM5NEXT_ARCHITECTURE:
        if not args.model_path:
            parser.error("--model-path is required for GLM")
        run_glm5next_mhc_module(
            ops=list(GLM5NEXT_VLLM_OPS) if args.op == "all" else [args.op],
            num_tokens_cases=_parse_int_list(args.num_tokens),
            model_path=args.model_path,
            hidden_size=args.hidden_size,
            hc_mult=args.hc_mult,
            device=args.device,
            output_path=args.output_path,
        )
        return
    if args.op not in {"pre", "post", "all"}:
        parser.error(f"--op {args.op} is GLM-only")
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
