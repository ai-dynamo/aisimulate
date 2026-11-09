# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 sparse-attention kernel-level collector for SGLang.

Collects the DeepSeek-V4 NSA sparse-attention sub-kernels at the kernel level.
DeepSeek-V4's primary AIC path uses prefix-aware full CSA/HCA attention-module
rows; paged_mqa_logits here is supporting kernel-level data for the SDK CP
chunk walk, while the topk_calib DELTA is actively consumed by
perf_database's topK correction.

The two sub-kernels form ONE sparse-op family — the CSA sparse path in
order — all collected, none modeled analytically:

    1. ``deep_gemm.fp8_paged_mqa_logits``      CSA indexer scoring
    2. ``topk_transform`` (topk_512/_1024)     CSA selection — phase-specific
                                               v1/v2 flat-vs-top_last DELTA calib

The standalone FMLA sidecars (``flash_mla_with_kvcache`` as csa_attn / hca_attn)
were removed their SDK readers were retired design placeholders with
no query path (csa_attn never shipped data) and, as registry-only ops, they
failed every shard plan generated from the registry.

Inputs are the op plans' rows (collector/op_plans.py): paged_mqa_logits takes
the dsv4 block's sparse grid, the one vLLM times, and topk takes the CSA module
plan's points. They are benched at the upstream layout (DeepGEMM
``test_attention.py`` for the indexer GEMM).

CSV schema matches existing aic dsv4 module CSVs (so loaders can be shared):
``isl`` carries M, ``step`` carries past_kv, ``compress_ratio`` carries the CSA
c4 path (=4).
"""

# Requires stock SGLang 0.5.21 with its matching ``sgl-kernel`` package.
from __future__ import annotations

# 0.5.21 (H20/sm90 collector port: op_smoke + path gates in the v0.5.21 image;
# findings hopper_sglang_collector_port_0514_to_0521_2026_10_01).
# One release per file; earlier code is `git log -- <this file>` away.
__compat__ = "sglang==0.5.23"

import functools
import json
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    from collector.sglang.helper import benchmark_with_power, get_sm_version, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import benchmark_with_power, get_sm_version, log_perf

try:
    from collector.case_generator import _DSV4_DEFAULT_MODELS
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import _DSV4_DEFAULT_MODELS
from collector.op_plans import plan_cases


def get_dsv4_paged_mqa_logits_test_cases():
    return plan_cases("dsv4_paged_mqa_logits_module", "sglang", get_sm_version())


def get_dsv4_topk_calib_test_cases():
    return plan_cases("dsv4_csa_topk_calib", "sglang", get_sm_version())


__all__ = [
    "DEFAULT_MODEL",
    "get_dsv4_paged_mqa_logits_test_cases",
    "get_dsv4_topk_calib_test_cases",
    "run_dsv4_sparse_kernel_worker",
]


DEFAULT_MODEL = _DSV4_DEFAULT_MODELS[0]
MODEL_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "src" / "aisimulate_core" / "model_configs"

# ═══════════════════════════════════════════════════════════════════════
# Model config (single source for all DSV4 model-specific shapes)
# ═══════════════════════════════════════════════════════════════════════


def _dsv4_model_config(model_path: str) -> dict:
    """Load a DSV4 model config json — the single source of model shapes.

    Accepts a local model directory (``<dir>/config.json``) or an HF-style id
    resolved under ``MODEL_CONFIGS_DIR``. Hard error if absent; these values
    MUST NOT be guessed/defaulted.
    """
    if model_path and os.path.isdir(model_path):
        path = Path(model_path) / "config.json"
    else:
        path = MODEL_CONFIGS_DIR / ((model_path or "").replace("/", "--") + "_config.json")
    if not path.is_file():
        raise FileNotFoundError(f"DSV4 model config not found at {path} for {model_path!r}")
    return json.loads(path.read_text())


def _dsv4_cfg_int(cfg: dict, field: str) -> int:
    """Required int field from a loaded config (hard error if missing/empty)."""
    if cfg.get(field) in (None, ""):
        raise KeyError(f"{field} missing in DSV4 model config")
    return int(cfg[field])


@functools.cache
def _dsv4_sparse_config(model_path: str) -> SimpleNamespace:
    """All sparse-attention kernel shapes, extracted from the model config the
    SAME way sglang's MQALayer / C4Indexer ``__init__`` read them
    (models/deepseek_v4.py) — the single source for every sparse-kernel shape,
    no scattered reads or magic numbers.

    NSA compress-ratio layer types are exactly {0, 4, 128} (sglang asserts this
    and selects the CSA/indexer path with the literal ``compress_ratio == 4``);
    the topk / paged-mqa indexer is that c4 path, so ``csa_compress_ratio`` is 4
    by architecture (validated against the config, not min()/heuristic).
    ``page_size`` and the fp8 quant tile are kernel-layout invariants (not in the
    config) and remain module constants below.
    """
    cfg = _dsv4_model_config(model_path)
    compress = {int(r) for r in (cfg.get("compress_ratios") or [])}
    if not compress <= {0, 4, 128}:
        raise ValueError(
            f"unexpected compress_ratios {sorted(compress)} for {model_path!r}; "
            "DSV4 NSA expects a subset of {0, 4, 128}"
        )
    if 4 not in compress:
        raise KeyError(f"no CSA layer (compress_ratio=4) in config for {model_path!r}; got {sorted(compress)}")
    head_dim = _dsv4_cfg_int(cfg, "head_dim")
    rope = _dsv4_cfg_int(cfg, "qk_rope_head_dim")
    return SimpleNamespace(
        head_dim=head_dim,  # FlashMLA d_qk / V_HEAD_DIM (512)
        d_rope=rope,  # FlashMLA d_rope (64)
        d_nope=head_dim - rope,  # FlashMLA d_nope (448)
        num_attention_heads=_dsv4_cfg_int(cfg, "num_attention_heads"),
        index_n_heads=_dsv4_cfg_int(cfg, "index_n_heads"),
        index_head_dim=_dsv4_cfg_int(cfg, "index_head_dim"),
        index_topk=_dsv4_cfg_int(cfg, "index_topk"),  # V4-Pro=1024, V4-Flash=512
        csa_compress_ratio=4,  # sglang: compress_ratio == 4 is the CSA/indexer path
        sliding_window=_dsv4_cfg_int(cfg, "sliding_window"),  # HCA SWA window (128)
    )


DEFAULT_ARCHITECTURE = "DeepseekV4ForCausalLM"


@functools.lru_cache(maxsize=1)
def _dsv4_kv_page_size() -> int:
    """DSV4 KV-pool paged block size — single source from sglang.

    The deep_gemm indexer (paged_mqa_logits block_kv) and FlashMLA paged
    attention share ONE KV-pool page size: sglang's DSA indexer asserts
    get_token_to_kv_pool().page_size == 64 and FlashMLA uses the same value
    (flashmla_backend.PAGE_SIZE). Imported lazily — flashmla_backend pulls in
    the whole MLA attention backend, which must not run at collector
    module-import time.
    """
    from sglang.srt.layers.attention.flashmla_backend import PAGE_SIZE

    return int(PAGE_SIZE)


def _device_num_sms(device: str | torch.device) -> int:
    """Return the actual SM count of ``device`` (the kernel sizes
    ``schedule_meta`` from this and asserts ``_schedule_meta_size == num_sms + 1``)."""
    return torch.cuda.get_device_properties(device).multi_processor_count


# The DSV4 sparse-op family — both collected by ONE worker
# (run_dsv4_sparse_kernel_worker), all driven by the KERNEL_* maps below, all
# timing the points of their op plan rows, all writing the SAME row schema
# (one ``latency`` per row):
#   - paged_mqa_logits: CSA indexer scoring (deep_gemm)            -> 1 row/shape
#   - topk:           CSA selection — context runs v1 and generation runs v2,
#                     each under two score distributions, emitted as TWO
#                     rows/shape (score_mode=v1_flat | v1_top_last or
#                     v2_flat | v2_top_last); perf_database takes the matching
#                     DELTA = flat.latency - top_last.latency -> 2 rows/shape
KERNEL_TO_OP_NAME = {
    "paged_mqa_logits": "dsv4_paged_mqa_logits_module",
    "topk": "dsv4_csa_topk_calib",
}

KERNEL_TO_KERNEL_SOURCE = {
    "paged_mqa_logits": "deep_gemm.fp8_paged_mqa_logits",
}

# compress_ratio: 4 for the CSA c4 path (indexer/topk).
KERNEL_TO_COMPRESS_RATIO = {
    "paged_mqa_logits": 4,
    "topk": 4,
}

# Both sparse kernels are TP-independent — the indexer latency does not
# change with tp (the module's per-tp rows differ only by the attention module's
# comms/sharding, not the sparse sub-kernel). So each is benched ONCE per point
# and written one row per point at the row's tp_size (1), matching how the SDK
# consumes them (keyed by shape, not tp). (topk additionally emits two
# phase-qualified score_mode rows per point.)


# ═══════════════════════════════════════════════════════════════════════
# Bench helper
# ═══════════════════════════════════════════════════════════════════════


def _bench_cuda_graph(
    kernel_fn: Callable[[], None],
    *,
    num_warmup: int = 5,
    num_iterations: int = 20,
    graph_repeat: int = 4,
    allow_graph_fail: bool = False,
    device: str = "cuda:0",
) -> dict:
    """Benchmark a kernel via AIC's benchmark_with_power helper (the single
    bench path for all DSV4 sparse kernels).

    benchmark_with_power handles warmup, CUDA-Graph capture/replay, optional
    power sampling, and graph-private-pool teardown. With ``allow_graph_fail``
    False, CUDA-graph capture is mandatory (used_cuda_graph is asserted).
    Returns ``{"latency_ms", "power_stats"}``.
    """
    if num_iterations < 3:
        raise ValueError("num_iterations must be at least 3")
    if graph_repeat < 1:
        raise ValueError("graph_repeat must be at least 1")

    def timed_kernel():
        with torch.no_grad():
            return kernel_fn()

    with benchmark_with_power(
        device=torch.device(device),
        kernel_func=timed_kernel,
        num_warmups=num_warmup,
        num_runs=num_iterations,
        repeat_n=graph_repeat,
        allow_graph_fail=allow_graph_fail,
    ) as result:
        pass

    if not allow_graph_fail and not result.get("used_cuda_graph", False):
        raise RuntimeError("benchmark_with_power did not use CUDA Graph")

    return {
        "latency_ms": float(result["latency_ms"]),
        "power_stats": result.get("power_stats"),
    }


# ═══════════════════════════════════════════════════════════════════════
# Cache packing helpers (mirror DeepGEMM test code)
# ═══════════════════════════════════════════════════════════════════════


def _kv_cache_cast_to_fp8_indexer(x: torch.Tensor) -> torch.Tensor:
    """DeepGEMM test_attention's kv_cache_cast_to_fp8.

    x: (num_blocks, block_size, 1, head_dim=128) bf16
    out: (num_blocks, block_size, 1, head_dim+4=132) packed fp8 + per-token fp32 sf
    """
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    out = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
    out[:, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(torch.uint8)
    out[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(torch.uint8)
    return out.view(num_blocks, block_size, 1, head_dim + 4)


def _expand_block_table(num_blocks: int, batch_size: int, device) -> torch.Tensor:
    """Block list [0..num_blocks) replicated to (batch_size, num_blocks) int32.
    All requests share the same physical blocks (KV cache is shared in-bench)."""
    row = torch.arange(num_blocks, dtype=torch.int32, device=device)
    return row.unsqueeze(0).expand(batch_size, num_blocks).contiguous()


# ═══════════════════════════════════════════════════════════════════════
# Kernel 1: deep_gemm.fp8_paged_mqa_logits
# ═══════════════════════════════════════════════════════════════════════


def _bench_paged_mqa_logits(
    M: int,  # noqa: N803
    past_kv: int,
    *,
    index_n_heads: int,
    index_head_dim: int,
    batch_size: int = 1,
    device: str = "cuda:0",
) -> float:
    """Benchmark paged_mqa_logits.

    The kernel imposes ``next_n ≤ 2`` (smem capacity), so prefill tokens map
    to ``batch_dim`` with ``next_n=1``. Per-request causal lengths remain
    distinct even though those token rows share one physical KV cache.
    """
    from deep_gemm import fp8_paged_mqa_logits

    # Serving builds the paged-MQA schedule with SGLang's own JIT kernel
    # (sglang::paged_mqa_metadata_tiny_kernel) whenever SGLANG_OPT_USE_JIT_INDEXER_METADATA
    # is set (default True, srt/environ.py:1829) or the query rows exceed 11673
    # (srt/layers/attention/dsv4/metadata.py:66, :212-219), for one row per request
    # (rows_per_request, :188) and without force_deep_gemm_metadata (:184, set only for
    # the SM120 FP4 indexer); the DeepGEMM sched kernel is the opt-out path. Same
    # (seq_lens, page_size, num_sm) contract (kernels/ops/attention/dsv4/attn.py:55).
    # Citations are v0.5.23. The H20 serving decode trace carries the tiny kernel, never
    # deep_gemm::sched::sm90_paged_mqa_logits_metadata.
    from sglang.kernels.ops.attention.dsv4.attn import get_paged_mqa_logits_metadata

    if batch_size <= 0 or M % batch_size:
        raise ValueError(f"M={M} must be divisible by positive batch_size={batch_size}")
    isl = M // batch_size
    full_s = isl + past_kv
    full_c4 = max(1, full_s // 4)
    block_kv = _dsv4_kv_page_size()

    b = M
    next_n = 1

    # Q: (b, 1, index_n_heads, index_head_dim) → fp8
    q_bf16 = torch.randn(b, next_n, index_n_heads, index_head_dim, dtype=torch.bfloat16, device=device)
    q = q_bf16.to(torch.float8_e4m3fn)

    # KV cache: SHARED across all b "fake-batch" entries (avoid b-fold blowup
    # at long past_kv).  Different entries' block_tables all point at the
    # same physical blocks — kernel just reads the same KV M times.
    blocks_per_req = (full_c4 + block_kv - 1) // block_kv
    kv_bf16 = torch.randn(blocks_per_req, block_kv, 1, index_head_dim, dtype=torch.bfloat16, device=device)
    kv_in = _kv_cache_cast_to_fp8_indexer(kv_bf16)

    weights = torch.randn(b * next_n, index_n_heads, dtype=torch.float32, device=device)

    # Compute each request's causal span before flattening the token rows.
    causal_seq = torch.arange(past_kv + 1, past_kv + isl + 1, dtype=torch.int32, device=device).repeat(batch_size)
    causal_c4 = (causal_seq // 4).clamp(min=1)  # min=1 to avoid empty scans
    context_lens = causal_c4.view(b, next_n)

    # All requests reuse the same block list [0..blocks_per_req-1]
    block_table = _expand_block_table(blocks_per_req, b, device)

    schedule_meta = get_paged_mqa_logits_metadata(context_lens, block_kv, _device_num_sms(device))

    def kernel_fn():
        return fp8_paged_mqa_logits(q, kv_in, weights, context_lens, block_table, schedule_meta, int(full_c4), False)

    return _bench_cuda_graph(kernel_fn, allow_graph_fail=False, device=device)


# ═══════════════════════════════════════════════════════════════════════
# CSV write helper
# ═══════════════════════════════════════════════════════════════════════


def _make_perf_filename(kernel: str, output_path: str) -> str:
    # Default filename derives directly from the op name (op_name + "_perf.txt").
    if os.path.isdir(output_path) or not output_path.endswith(".txt"):
        return os.path.join(output_path, f"{KERNEL_TO_OP_NAME[kernel]}_perf.txt")
    return output_path


def _write_row(
    perf_filename: str,
    *,
    kernel: str,
    bs: int,
    isl: int,
    past_kv: int,
    tp_size: int,
    native_heads: int,
    latency_ms: float,
    device_name: str,
    model_path: str = DEFAULT_MODEL,
    architecture: str = DEFAULT_ARCHITECTURE,
    score_mode: str | None = None,
    power_stats: dict | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(perf_filename)) or ".", exist_ok=True)

    mla_dtype = "fp8_e4m3"
    kv_cache_dtype = "fp8_e4m3"
    gemm_type = "fp8_block"

    item = {
        "model": model_path,
        "architecture": architecture,
        "mla_dtype": mla_dtype,
        "kv_cache_dtype": kv_cache_dtype,
        "gemm_type": gemm_type,
        "num_heads": native_heads,
        "batch_size": bs,
        "isl": isl,
        "tp_size": tp_size,
        "step": past_kv,
        "compress_ratio": KERNEL_TO_COMPRESS_RATIO[kernel],
        "latency": f"{latency_ms:.6f}",
    }
    # topk emits two variant-qualified rows per shape (flat vs top_last); the
    # discriminator column is absent from single-latency kernel rows.
    if score_mode is not None:
        item["score_mode"] = score_mode

    if kernel == "topk":
        if score_mode not in ("v1_flat", "v1_top_last", "v2_flat", "v2_top_last"):
            raise ValueError(f"topk row requires a v1/v2 score_mode, got {score_mode!r}")
        kernel_source = f"topk_transform_{score_mode.split('_', 1)[0]}"
    else:
        kernel_source = KERNEL_TO_KERNEL_SOURCE[kernel]

    if not log_perf(
        item_list=[item],
        framework="SGLang",
        version="kernel-level",
        device_name=device_name,
        op_name=KERNEL_TO_OP_NAME[kernel],
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=power_stats,
    ):
        raise RuntimeError(f"failed to persist DeepSeek-V4 sparse row to {perf_filename}")


# ═══════════════════════════════════════════════════════════════════════
# Worker (invoked by collect.py via the registry; test-case generators live in
# case_generator.py so all collectors share one sweep-grid definition).
# ═══════════════════════════════════════════════════════════════════════


def _guarded_bench(bench_fn: Callable[[], object], label: str):
    """Run one shape and return ``(result, exception)`` for final reporting."""
    try:
        return bench_fn(), None
    except torch.cuda.OutOfMemoryError as exc:
        print(f"  OOM at {label}; recording failure")
        torch.cuda.empty_cache()
        return None, exc
    except Exception as exc:
        traceback.print_exc()
        print(f"  failed at {label}; recording failure")
        torch.cuda.empty_cache()
        return None, exc


def _bench_sparse_kernel_shape(kernel, prefix, isl, bs, sc, device, *, topk_variant=None):
    """Bench one module ``(prefix, isl, bs)`` shape; return a list of
    ``(score_mode, latency_ms)`` rows. TP-independent (paged_mqa maps tokens to
    b=M while preserving request-local causal lengths), so this is computed
    once per shape.

    Single-latency kernels return one ``(None, latency)``; topk returns the
    phase-qualified flat-vs-representative pair whose matching-variant DELTA
    perf_database consumes (flat.latency - top_last.latency)."""
    M = bs * isl  # noqa: N806
    if kernel == "topk":
        if topk_variant not in ("v1", "v2"):
            raise ValueError(f"topk requires variant v1 or v2, got {topk_variant!r}")
        return _bench_topk_shape(prefix, isl, bs, sc, device, variant=topk_variant)
    # paged_mqa_logits: tokens map to b=M, next_n=1, while causal lengths repeat
    # per request.
    lat = _bench_paged_mqa_logits(
        M,
        prefix,
        index_n_heads=sc.index_n_heads,
        index_head_dim=sc.index_head_dim,
        batch_size=bs,
        device=device,
    )["latency_ms"]
    return [(None, lat)]


# Derived once per process by _sglang_chunked_prefill_size.
_CHUNKED_PREFILL = None


def _sglang_chunked_prefill_size():
    """The chunked_prefill_size a server launched without the flag derives on this device.

    The csa/hca modules (collect_dsv4_attn) and the DSA modules launch SGLang with
    chunked_prefill_size=None, which SGLang derives from device memory alone
    (_handle_gpu_memory_settings, srt/arg_groups/memory_hook.py:89-176 at v0.5.23):
    2048 below 20 GiB, 4096 below 60 GiB (and without a memory reading), 8192 below
    160 GiB, 16384 above. runtime_limits.serving_memory_defaults holds that table;
    the chunk does not depend on the TP world. The earlier local copy put 20-35 GiB
    devices at 2048 and raised without a memory reading.
    """
    global _CHUNKED_PREFILL
    if _CHUNKED_PREFILL is None:
        from collector.sglang.runtime_limits import serving_memory_defaults

        _CHUNKED_PREFILL = int(serving_memory_defaults(tp_size=1)[0])
    return _CHUNKED_PREFILL


def run_dsv4_sparse_kernel_worker(
    model_path: str,
    kernel: str,
    batch_size: int,
    tp_size: int,
    points,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """Single worker for the whole DSV4 sparse-op family: one (model, TP, batch) row of
    its op plan (op_plans._sglang_dsv4_kernel_rows).

    ``points`` are the row's ``[prefix, isl, topk_variant]`` points, each benched
    once. ``_bench_sparse_kernel_shape`` returns one or more ``(score_mode,
    latency)`` rows per point — one for paged_mqa_logits, two for topk
    (variant-qualified flat + top_last) — each written as one row."""
    if kernel not in KERNEL_TO_OP_NAME:
        raise ValueError(f"unknown kernel={kernel}; expected one of {list(KERNEL_TO_OP_NAME)}")
    bs_only = int(batch_size)
    shapes = [(int(prefix), int(isl), bs_only, variant) for prefix, isl, variant in points]
    if not shapes:
        raise RuntimeError(f"dsv4-sparse {kernel} bs={bs_only}: the row lists no points")
    sc = _dsv4_sparse_config(model_path)
    output_dir = os.path.dirname(perf_filename) or os.getcwd()
    perf_path = _make_perf_filename(kernel, output_dir)

    device_name = torch.cuda.get_device_name(device)
    print(f"[dsv4-sparse {kernel} bs={bs_only}] {len(shapes)} shapes -> {perf_path}")
    n_ok = 0
    failures = []
    for prefix, isl, bs, topk_variant in shapes:
        # Every point is benched; tiny shapes run fine (K/window clamp to
        # full_s); failed attempts are reported together.
        label = f"bs={bs} isl={isl} past_kv={prefix}" + (f" topk_variant={topk_variant}" if topk_variant else "")
        results, error = _guarded_bench(
            lambda: _bench_sparse_kernel_shape(
                kernel,
                prefix,
                isl,
                bs,
                sc,
                device,
                topk_variant=topk_variant,
            ),
            label,
        )
        if error is not None:
            failures.append(f"{label}: {type(error).__name__}: {error}")
            continue
        expected_modes = {f"{topk_variant}_flat", f"{topk_variant}_top_last"} if kernel == "topk" else {None}
        if len(results) != len(expected_modes) or {score_mode for score_mode, _ in results} != expected_modes:
            message = f"expected score modes {expected_modes}, got {results}"
            print(f"  incomplete result at {label}: {message}")
            failures.append(f"{label}: RuntimeError: {message}")
            continue
        for score_mode, latency_ms in results:
            _write_row(
                perf_path,
                kernel=kernel,
                bs=bs,
                isl=isl,
                past_kv=prefix,
                tp_size=tp_size,
                native_heads=sc.num_attention_heads,
                latency_ms=latency_ms,
                device_name=device_name,
                model_path=model_path,
                score_mode=score_mode,
            )
        n_ok += 1
    error_count = len(failures)
    summary = f"ok={n_ok} error={error_count} skip=0 total={len(shapes)}"
    print(f"  {kernel}: {summary}")
    if not n_ok or failures:
        details = "\n- ".join(failures) if failures else "no inner shapes produced a row"
        raise RuntimeError(f"dsv4-sparse {kernel} bs={bs_only}: {summary}; failures:\n- {details}")


# ═══════════════════════════════════════════════════════════════════════
# topk_512 indexer DELTA calibration (kernels, fed by the shared worker above)
# ═══════════════════════════════════════════════════════════════════════
# The CSA indexer's topk kernel needs a CALIBRATION rather than a single latency:
# the CSA module CSV already contains the topK time measured on DEGENERATE scores
# (dummy weights -> near-constant logits -> the small O(n^2) tie-break path).
# ``_bench_topk_shape`` benches the production phase's topk_transform under
# FLAT (worst-case degenerate) and TOP_LAST (representative: largest scores at
# the causal tail) distributions and returns phase-qualified score_mode rows.
# The SDK applies only the matching v1/v2 DELTA = flat.latency -
# top_last.latency to swap the degenerate cost for a representative one.
#
# topK is CSA-only (compress_ratio=4) and, like the other sparse kernels,
# TP-independent (per-token causal scan): one variant-qualified pair per shape.


def _make_topk_scores(
    mode: str,
    seq_lens: torch.Tensor,
    seq: int,
    device: str,
    generator,
    topk_k: int,
) -> torch.Tensor:
    # score_stride must be a multiple of 4 (kernel TMA 16B alignment); allocate
    # padded width and let the kernel read [:, :seq].
    rows = seq_lens.numel()
    pad = ((seq + 3) // 4) * 4
    if mode == "flat":
        return torch.zeros(rows, pad, device=device)
    if mode == "top_last":
        s = -5.0 + 0.05 * torch.randn(rows, pad, device=device, generator=generator)
        counts = seq_lens.to(torch.int64).clamp(min=0, max=min(topk_k, seq))
        offsets = torch.arange(topk_k, device=device)
        columns = seq_lens.to(torch.int64).unsqueeze(1) - counts.unsqueeze(1) + offsets.unsqueeze(0)
        valid = offsets.unsqueeze(0) < counts.unsqueeze(1)
        row_ids = torch.arange(rows, device=device).unsqueeze(1).expand_as(columns)
        selected = int(valid.sum().item())
        s[row_ids[valid], columns[valid]] = 5.0 + torch.randn(selected, device=device, generator=generator)
        return s.contiguous()
    raise ValueError(f"unknown topk score mode: {mode}")


def _bench_topk_512(
    seq_lens: torch.Tensor,
    mode: str,
    device: str,
    topk_k: int,
    variant: str,
) -> float:
    """Time one v1/v2 topk_transform shape via the shared bench path.

    v1 is topk_transform_paged (kernels/ops/attention/dsv4/topk.py:106-125, the JIT
    _jit_topk_v1_module on CUDA) and v2 topk_transform_paged_v2 (:209-268) on its
    plan_topk_v2 routing plan (:134-156). Both execute inside production CUDA graphs,
    so capture is mandatory here too. Citations are v0.5.23.

    Deviation (the v1 lane on SM90/SM100/SM103): the op plan times context points on v1
    (op_plans.Dsv4TopkCase), but SGLang's C4 indexer runs v2 in both phases there:
    run_topk_transform takes v2 whenever DSATopKBackend.should_use_topk_v2() (sgl-kernel
    backend and SGLANG_OPT_USE_TOPK_V2, default True; srt/layers/attention/dsv4/
    indexer.py:948-998, dsa/dsa_topk_backend.py:55-56, srt/environ.py:1486), and a
    prefill passes the raw-index output too (deepseek_v4_backend.py:897-900). Only
    SM120 serving takes v1 (arg_groups/model_hook.py:487-488 sets the flag False), and
    there the v2 decode lane is the mismatch; both lanes are parked on SM120
    (registry unverified_sms). The context v1_* rows therefore calibrate a kernel
    serving does not launch on SM90/SM10x; the error direction is unmeasured. Retire
    by planning context points on v2 with the raw-index output and moving the
    consumer's context lookup to v2_*.
    """
    from sglang.kernels.ops.attention.dsv4.topk import (
        plan_topk_v2,
    )
    from sglang.kernels.ops.attention.dsv4.topk import (
        topk_transform_paged as topk_transform_512,
    )
    from sglang.kernels.ops.attention.dsv4.topk import (
        topk_transform_paged_v2 as topk_transform_512_v2,
    )

    if variant not in ("v1", "v2"):
        raise ValueError(f"unknown topk variant: {variant}")

    generator = torch.Generator(device=device)
    generator.manual_seed(1234)
    rows = seq_lens.numel()
    c4_len = int(seq_lens.max().item())
    # Serving plans with the kernel's own threshold heuristic (static_threshold=-1):
    # PagedIndexerMetadata (srt/layers/attention/dsv4/metadata.py:278) and the per-step
    # refresh (deepseek_v4_backend.py:1188) call plan_topk_v2(compressed_seq_lens).
    # A static threshold of 0 sent every row past the 32768 floor to the persistent
    # cluster pool, which the heuristic declines for long-KV batches above its per-arch
    # caps (kernels/jit/csrc/deepseek_v4/topk_v2.cuh:655-745 at v0.5.23).
    meta = plan_topk_v2(seq_lens) if variant == "v2" else None
    torch.cuda.synchronize(device)
    pages = (c4_len + _dsv4_kv_page_size() - 1) // _dsv4_kv_page_size()
    page_table = torch.arange(pages, dtype=torch.int32, device=device).unsqueeze(0).repeat(rows, 1)
    scores = _make_topk_scores(mode, seq_lens, c4_len, device, generator, topk_k=topk_k)
    out = torch.empty((rows, topk_k), dtype=torch.int32, device=device)
    raw_indices = torch.empty_like(out) if variant == "v1" else None

    def kernel_func():
        if variant == "v1":
            topk_transform_512(
                scores,
                seq_lens,
                page_table,
                out,
                _dsv4_kv_page_size(),
                raw_indices,
            )
        else:
            topk_transform_512_v2(
                scores,
                seq_lens,
                page_table,
                out,
                _dsv4_kv_page_size(),
                meta,
            )

    return _bench_cuda_graph(kernel_func, allow_graph_fail=False, device=device)["latency_ms"]


def _bench_topk_shape(prefix: int, isl: int, bs: int, sc, device: str, *, variant: str) -> list:
    """topk DELTA calibration for one CSA ``(prefix, isl, bs)`` shape: bench
    topk_transform under FLAT (degenerate worst-case) and TOP_LAST
    (representative) score distributions and return them as two
    phase-qualified rows (for example, ``v1_flat`` and ``v1_top_last``).

    Trivial when c4 <= K (no select stage, no degenerate tie-break) -> both 0
    (DELTA 0). ``rows`` = bs*isl is the per-token causal scan count."""
    ratio = sc.csa_compress_ratio  # CSA compress ratio (=4)
    topk_k = sc.index_topk  # V4-Pro=1024, V4-Flash=512
    causal_seq = torch.arange(prefix + 1, prefix + isl + 1, dtype=torch.int32, device=device)
    seq_lens = (causal_seq // ratio).clamp(min=1).repeat(bs)
    c4_len = int(seq_lens.max().item())
    if c4_len <= topk_k:
        return [(f"{variant}_flat", 0.0), (f"{variant}_top_last", 0.0)]
    return [
        (
            f"{variant}_{mode}",
            round(_bench_topk_512(seq_lens, mode, device, topk_k=topk_k, variant=variant), 6),
        )
        for mode in ("flat", "top_last")
    ]
