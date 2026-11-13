# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4-Flash module-level attention collector for SGLang.

ONE file containing both:

  1. The bench engine — builds an sglang ``ModelRunner`` for a single
     attn_kind (CSA / HCA) layer and times CUDA-Graph replay of
     ``layer.self_attn(...)`` (Q/KV proj + norm/rope + cache store +
     compressor + C4 indexer/topk for CSA + final FlashMLA).
  2. The registry-facing entrypoints — ``run_dsv4_attn_worker``
     (one (kind, tp, gemm, bs) row of the op plan) which spawns a
     subprocess that measures the row's (prefix, seq_len) points.

Test cases are the op plans' rows (collector/op_plans.py). The manual CLI
runs the plan's rows too::

    python collect_dsv4_attn.py --mode generation --attn-kind csa
    python collect_dsv4_attn.py --mode context --attn-kind hca \
        --batch-sizes 1,4 --tp-sizes 1
"""

# Requires stock SGLang 0.5.21 with its matching ``sgl-kernel`` package.
from __future__ import annotations

# 0.5.21 (H20/sm90 collector port: op_smoke + path gates in the v0.5.21 image;
# findings hopper_sglang_collector_port_0514_to_0521_2026_10_01).
# One release per file; earlier code is `git log -- <this file>` away.
__compat__ = "sglang==0.5.23"

import argparse
import contextlib
import copy
import gc
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Iterable
from fractions import Fraction
from importlib.metadata import version as get_version

import torch

# DSV4 local forks default to replacing small patched configs with packaged
# config_backup_small.json.  Suppress so collector's per-kind 1-layer config
# isn't overwritten.
os.environ.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")
# Hard-disable DeepGEMM bulk pre-compile.  Each test case touches only a
# few shapes which the bench's own warmup JIT-compiles on first use.
os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
try:
    from helper import (
        PeriodicGarbageCollector,
        PhaseClock,
        _resolve_local_model_path,
        benchmark_with_power,
        get_sm_version,
        log_perf,
        render_phase_totals,
        worker_heartbeat,
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import (
        PeriodicGarbageCollector,
        PhaseClock,
        _resolve_local_model_path,
        benchmark_with_power,
        get_sm_version,
        log_perf,
        render_phase_totals,
        worker_heartbeat,
    )

try:
    from collector.sglang.runtime_limits import (
        KERNEL_LIMIT,
        kernel_limit_error,
        over_chunk,
        over_kv_pool,
        refusal_summary,
        refuse,
        required_kv_alloc_tokens,
        required_prefill_extend_tokens,
        required_swa_kv_alloc_tokens,
    )
    from collector.sglang.runtime_limits import (
        alloc_prefix_indices as _alloc_prefix_indices,
    )
    from collector.sglang.runtime_limits import (
        kv_pool_capacity_tokens as _kv_pool_capacity_tokens,
    )
    from collector.sglang.runtime_limits import (
        kv_pool_page_size as _kv_pool_page_size,
    )
    from collector.sglang.runtime_limits import (
        runtime_chunk_size as _runtime_chunk_size,
    )
    from collector.sglang.runtime_limits import (
        swa_kv_pool_capacity_tokens as _swa_kv_pool_capacity_tokens,
    )
    from collector.sglang.runtime_limits import (
        temporarily_chunked_alloc_extend as _temporarily_chunked_alloc_extend,
    )
except ModuleNotFoundError:
    from runtime_limits import (
        KERNEL_LIMIT,
        kernel_limit_error,
        over_chunk,
        over_kv_pool,
        refusal_summary,
        refuse,
        required_kv_alloc_tokens,
        required_prefill_extend_tokens,
        required_swa_kv_alloc_tokens,
    )
    from runtime_limits import (
        alloc_prefix_indices as _alloc_prefix_indices,
    )
    from runtime_limits import (
        kv_pool_capacity_tokens as _kv_pool_capacity_tokens,
    )
    from runtime_limits import (
        kv_pool_page_size as _kv_pool_page_size,
    )
    from runtime_limits import (
        runtime_chunk_size as _runtime_chunk_size,
    )
    from runtime_limits import (
        swa_kv_pool_capacity_tokens as _swa_kv_pool_capacity_tokens,
    )
    from runtime_limits import (
        temporarily_chunked_alloc_extend as _temporarily_chunked_alloc_extend,
    )


try:
    from collector.case_generator import _DSV4_MODULE_TP_SIZES as _TP_SIZES
    from collector.case_generator import DSV4_ATTN_KINDS as ATTN_KINDS
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import _DSV4_MODULE_TP_SIZES as _TP_SIZES
    from case_generator import DSV4_ATTN_KINDS as ATTN_KINDS
from collector.op_plans import plan, plan_cases


def get_dsv4_csa_context_test_cases():
    return plan_cases("dsv4_csa_context_module", "sglang", get_sm_version())


def get_dsv4_csa_generation_test_cases():
    return plan_cases("dsv4_csa_generation_module", "sglang", get_sm_version())


def get_dsv4_hca_context_test_cases():
    return plan_cases("dsv4_hca_context_module", "sglang", get_sm_version())


def get_dsv4_hca_generation_test_cases():
    return plan_cases("dsv4_hca_generation_module", "sglang", get_sm_version())


_DSV4_CUDA_PAGE_SIZE = 256
_DSV4_SWA_WINDOW_SIZE = 128
_DSV4_SWA_TO_FULL_SCALE = 10

ATTN_KIND_TO_COMPRESS_RATIO = {
    "csa": 4,
    "hca": 128,
}


CLI_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


_PORTS_PER_GPU = 1000
_DSV4_PORT_RETRIES = 5


def _canonical_dsv4_model_id(model_path: str) -> str:
    normalized = str(model_path).rstrip("/")
    basename = normalized.rsplit("/", 1)[-1]
    if basename == "DeepSeek-V4-Pro":
        return "deepseek-ai/DeepSeek-V4-Pro"
    if basename == "DeepSeek-V4-Flash":
        return "deepseek-ai/DeepSeek-V4-Flash"
    return str(model_path)


def _port_is_available(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # TCPStore may listen on any local interface; check the same address
        # family instead of only 127.0.0.1.
        s.bind(("0.0.0.0", port))
        s.listen(1)
        return True
    except OSError:
        return False
    finally:
        s.close()


def _nccl_port_for_attempt(gpu_id: int, attempt: int) -> int:
    """Use a deterministic, GPU-scoped TCPStore port."""
    return 40000 + gpu_id * _PORTS_PER_GPU + attempt


def _pick_free_port(gpu_id: int) -> int:
    """Return a free TCP port from a ``gpu_id``-scoped 1000-port range.

    Used as ``nccl_port`` for the per-subprocess torch.distributed
    rendezvous.  Up to 8 collector workers run in parallel, each pinned
    to one GPU.  Partitioning the port space by ``gpu_id`` makes
    cross-worker collision impossible: worker N's candidate set is
    [40000 + N*1000, 40000 + N*1000 + 999].  Kept as a fallback for
    direct/manual use; normal collect.py entrypoints pass
    ``AIC_DSV4_NCCL_PORT`` explicitly.
    """
    base = _nccl_port_for_attempt(gpu_id, 0)
    for offset in range(_PORTS_PER_GPU):
        port = _nccl_port_for_attempt(gpu_id, offset)
        if _port_is_available(port):
            return port
    raise RuntimeError(f"no free port in [{base}, {base + 999}] for gpu_id={gpu_id}")


def _kv_dtype_db_to_sglang(kv_dtype_db: str) -> str:
    """Map perf-database kv dtype string to SGLang's ServerArgs value."""
    return {"bfloat16": "bfloat16", "fp8": "fp8_e4m3"}[kv_dtype_db]


# ═══════════════════════════════════════════════════════════════════════
# Bench engine — model load, forward batch, CUDA-graph timing, perf log
# ═══════════════════════════════════════════════════════════════════════


def _resolve_perf_path(output_path: str | None, default_name: str) -> str:
    if not output_path:
        return default_name
    if output_path.endswith(".txt"):
        return output_path
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, default_name)


def _copy_non_weight_files(src_dir: str, dst_dir: str) -> None:
    """Mirror model assets into the patched-config temp dir.

    - Non-weight files (tokenizer, generation_config, etc.) are copied.
    - Weight files (``.safetensors`` etc.) are *symlinked* so that
      ``load_format=auto`` can read real weights from the original model dir
      while the temp dir's patched ``config.json`` controls the architecture.
      This is required to reproduce production score distributions in the
      indexer's ``topk_512_transform`` (dummy weights produce uniformly
      random logits which take a different radix path and clock in higher
      than the structured logits a trained checkpoint produces).
    - ``config.json`` is intentionally skipped here; the caller writes the
      patched config in its place.
    """
    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)
        if not os.path.isfile(src_path):
            continue
        if fname == "config.json":
            continue
        dst_path = os.path.join(dst_dir, fname)
        if os.path.exists(dst_path) or os.path.islink(dst_path):
            continue
        if fname.endswith(_WEIGHT_SUFFIXES) or fname.endswith(".safetensors.index.json"):
            os.symlink(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def _download_non_weight_model_files(model_id: str) -> tuple[str, dict]:
    from huggingface_hub import hf_hub_download, list_repo_files

    try:
        files = list_repo_files(model_id)
    except Exception:
        files = ["config.json"]

    config_file = None
    for fname in files:
        if fname.endswith(_WEIGHT_SUFFIXES):
            continue
        try:
            path = hf_hub_download(model_id, fname)
            if fname == "config.json":
                config_file = path
        except Exception:
            continue

    if config_file is None:
        config_file = hf_hub_download(model_id, "config.json")

    with open(config_file) as f:
        config = json.load(f)
    return os.path.dirname(config_file), config


def _resolve_model_path(
    model_path: str,
    *,
    attn_kind: str,
    num_layers: int,
    shrink_unused_moe: bool,
    disable_weight_quant: bool,
    strip_auto_map: bool = True,
    gemm_type: str = "bfloat16",
    load_format: str = "dummy",
) -> str:
    """Create a local config dir patched for a single DSV4 attention kind.

    ``gemm_type`` controls which GEMM path the projection layers take:
        - ``"bfloat16"`` (default): drops fp8 ``quantization_config`` so weights
          load as bf16 and projections dispatch to cuBLASLt nvjet kernels.  This
          matches the historical collector behavior and is fast/light to load.
        - ``"fp8_block"``: keeps the upstream V4-Flash fp8 block-quantized
          ``quantization_config``.  Combined with ServerArgs ``quantization="fp8"``
          this routes projection GEMMs through DeepGEMM's
          ``sm90_fp8_gemm_1d2d_impl`` kernel — the same path the production
          server uses, so kernel-by-kernel the latency lines up with a real run.

    TP is not emulated at this layer (do not patch num_attention_heads): that
    would bypass the attention-TP query padding to h_q=64 and trip FlashMLA's
    "Unsupported h_q: 8" template guard. See ``_refuse_tp_emulation``.
    """

    if os.path.isdir(model_path):
        src_dir = model_path
        with open(os.path.join(src_dir, "config.json")) as f:
            config = json.load(f)
    elif load_format == "dummy":
        src_dir = _resolve_local_model_path(model_path)
        with open(os.path.join(src_dir, "config.json")) as f:
            config = json.load(f)
    else:
        src_dir, config = _download_non_weight_model_files(model_path)

    config = copy.deepcopy(config)
    if strip_auto_map:
        config.pop("auto_map", None)

    compress_ratio = ATTN_KIND_TO_COMPRESS_RATIO[attn_kind]
    config["num_hidden_layers"] = num_layers
    config["num_key_value_heads"] = 1
    if config.get("architectures") != ["DeepseekV4ForCausalLM"]:
        config["architectures"] = ["DeepseekV4ForCausalLM"]

    # transformers has no "deepseek_v4" entry in CONFIG_MAPPING.  The V4
    # sglang fork's get_config only triggers its V4 loader when AutoConfig
    # fails with "deepseek_ref" or "deepseek_v32" in the message.  Rewriting
    # model_type to "deepseek_v3" mirrors what sglang's
    # _load_deepseek_temp_model produces internally, so AutoConfig succeeds
    # and the V4 model class is still selected via the architectures field.
    try:
        # sglang>=0.5.21 ships a native DeepseekV4Config (configs/deepseek_v4.py, model_type
        # "deepseek_v4") and its DSV4 model reads V4-only fields from it
        # (hc_pre_from_prev_sublayer, models/deepseek_v4.py:2679); the v3 rewrite below
        # would hand the model a DeepseekV3Config and fail there.
        import sglang.srt.configs.deepseek_v4  # noqa: F401

        config["model_type"] = "deepseek_v4"
    except ImportError:
        config["model_type"] = "deepseek_v3"

    # gemm_type "fp8_block" overrides disable_weight_quant: we MUST keep the
    # fp8 quantization_config so sglang dispatches projections to DeepGEMM.
    drop_quant = disable_weight_quant and gemm_type != "fp8_block"
    if drop_quant:
        config.pop("quantization_config", None)
        config.pop("compression_config", None)

    old_ratios = config.get("compress_ratios") or []
    if old_ratios:
        config["compress_ratios"] = [compress_ratio] * num_layers
    else:
        config["compress_ratios"] = [compress_ratio] * num_layers

    if shrink_unused_moe:
        # V4 DeepseekV4DecoderLayer always constructs ``self.mlp = DeepseekV2MoE``
        # (no dense-MLP fallback like V2's ``first_k_dense_replace`` toggles), so
        # the MoE weights *are* allocated even though forward only calls
        # ``layer.self_attn``.  Shrink only the count of experts; keep the per-
        # expert intermediate dim and the shared-experts count at production
        # values, because:
        #   - ``moe_intermediate_size`` shows up as the ``output_size`` of
        #     ColumnParallelLinear in fp8 block-quant; per-partition size must
        #     be divisible by ``block_n=128`` (``fp8.py:validate_block_quant_shapes``).
        #     Production 2048 / TP=8 = 256 (ok); shrinking to 256 would give
        #     32 at TP=8 and trigger a quantization shape error.
        #   - Setting ``n_shared_experts=0`` makes ``DeepseekV2MoE`` build a
        #     shared expert with intermediate=0, which divides-by-zero in
        #     ``validate_block_quant_shapes``.
        # 8 routed experts x 2048 inter x 7168 hidden x 1 byte fp8 ≈ 230 MB
        # per layer, comfortable on one H20.
        config["n_routed_experts"] = min(int(config.get("n_routed_experts", 8)), 8)
        config["num_experts_per_tok"] = min(int(config.get("num_experts_per_tok", 2)), 2)

    tmp_dir = os.path.join(
        tempfile.gettempdir(),
        f"aic_dsv4_{attn_kind}_{model_path.replace('/', '_')}_{os.getpid()}",
    )
    os.makedirs(tmp_dir, exist_ok=True)
    _copy_non_weight_files(src_dir, tmp_dir)
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f)
    return tmp_dir


def _refuse_tp_emulation(tp_size: int) -> None:
    """Refuse a TP>1 row before its model is built: this process cannot time one rank of it.

    SGLang shards DeepSeek-V4 attention by the published parallel context only.
    MQALayer reads get_parallel().attn_tp_size / attn_tp_rank for n_local_heads
    (srt/models/deepseek_v4.py:1024-1036 at v0.5.23; :766-780 at v0.5.21), and its
    wq_b / wo_a / wo_b bind get_parallel().attn_tp_group (parallel_group="attn_tp",
    deepseek_v4.py:1118-1166; srt/layers/linear.py:111-137). The previous load patch
    rewrote parallel_state._TP.world_size and dp_attention._ATTN_TP_SIZE, which no
    longer exist or are not read, so every TP=N row timed the TP=1 module under a
    rank-local num_heads label: the 0.5.21 SGLang tables are flat across tp_size
    (H200 CSA decode, batch 64, step 1: 0.1253/0.1251/0.1253/0.1256 ms at TP 1/2/4/8)
    where the 0.5.14 ones scale (0.1248/0.0961/0.0824/0.0743 ms).

    Retire by building the model under a scoped get_parallel().override(...) of the
    TP and attention-TP widths, as parallel_state.patch_tensor_parallel_group(...,
    owns_attention=True) does (srt/distributed/parallel_state.py:3114-3171), with the
    projections on a logical group (linear.py:88-91, :111-137); validate on a GPU that
    n_local_heads and the projection shards match one rank before re-enabling.
    """
    if tp_size > 1:
        raise RuntimeError(
            f"dsv4 module tp_size={tp_size}: single-process TP emulation does not reach SGLang's "
            "parallel context (get_parallel().attn_tp_size), so the module would run all heads "
            "unsharded; refusing to record TP=1 timings under a TP>1 label"
        )


def _effective_prefill_chunk_size(model_runner) -> int:
    """Return the query-token limit enforced by this collector path.

    This is SGLang's own serving chunk (``chunked_prefill_size``), nothing
    less. The retired extra bound ``(shared_memory_per_block_optin//4 - 1)//5``
    modeled the deep_gemm sched-meta kernel's SMEM limit, but SGLang does not
    expose prefill to that kernel: ``PagedIndexerMetadata`` builds its metadata
    through the JIT path whenever ``SGLANG_OPT_USE_JIT_INDEXER_METADATA`` is set
    (default ``True``, ``srt/environ.py:1829`` at v0.5.23; only HIP disables it,
    ``arg_groups/model_hook.py:506``), and even with the env off it switches to
    the JIT path above ``_LARGE_INDEXER_QUERY_THRESHOLD`` = 11,673 query rows
    (``layers/attention/dsv4/metadata.py:66, :212-219``). On H20 the derived
    chunk (8,192) sat below the old bound (11,622), so removing it changes
    nothing on SM90; on B200 the derived chunk is 16,384 and a probe
    executed 8,192/12,288/16,384 fresh-token CSA
    context cells through the JIT metadata path with zero errors.
    """
    return _runtime_chunk_size(model_runner)


def _derive_csa_context_pool_cap(
    model_runner,
    shapes: Iterable[tuple[int, int, int]],
) -> tuple[int, int, int]:
    """Derive a DSV4 CSA-context pool cap while reserving its logits output."""
    from sglang.srt.environ import envs
    from sglang.srt.model_executor.pool_configurator import (
        DSV4PoolConfigurator,
        create_memory_pool_configurator,
    )

    sm_major, sm_minor = torch.cuda.get_device_capability(model_runner.device)
    sm_version = sm_major * 10 + sm_minor
    if sm_version == 120:
        # SGLang forces the Torch/TileLang logits leaves on SM120
        # (arg_groups/model_hook.py:499-502 at v0.5.23). Those leaves
        # materialize additional FP32 gather/BMM intermediates, so the
        # DeepGEMM output-workspace formula below is not their memory contract.
        raise RuntimeError(
            "DSV4 CSA context on SM120 uses the Torch indexer and needs a separately "
            "validated workspace policy; the SM90/100/103 DeepGEMM formula is disabled"
        )
    if sm_version not in {90, 100, 103}:
        raise RuntimeError(
            "DSV4 CSA context pool derivation supports the DeepGEMM indexer on "
            f"SM90/100/103; SM{sm_version} needs a separately validated workspace policy"
        )

    alternate_indexers = {
        "tilelang": envs.SGLANG_OPT_USE_TILELANG_INDEXER.get(),
        "aiter": envs.SGLANG_OPT_USE_AITER_INDEXER.get(),
        "torch": envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get(),
    }
    selected_alternates = [name for name, enabled in alternate_indexers.items() if enabled]
    if selected_alternates:
        raise RuntimeError(
            "DSV4 CSA context pool derivation is not validated for indexer leaf/leaves " + ",".join(selected_alternates)
        )
    if bool(getattr(model_runner.server_args, "enable_deepseek_v4_fp4_indexer", False)):
        raise RuntimeError("DSV4 CSA context pool derivation is not validated for the FP4 indexer")

    # Serving hands the pool configurator factory the KV-cache configurator, which
    # alloc_memory_pool builds first (model_runner.py:971-983, kv_cache_configurator.py:
    # 3187 at v0.5.23). The runner itself no longer carries the pp_size / attn_dp_size
    # the DSV4 configurator reads (pool_configurator.py:1119, :1140-1142), so build the same
    # object here, before the pool exists; alloc_memory_pool rebuilds it.
    model_runner.init_kv_cache_configurator()
    kv_cache_configurator = model_runner.kv_cache_configurator
    configurator = create_memory_pool_configurator(kv_cache_configurator)
    if not isinstance(configurator, DSV4PoolConfigurator):
        raise TypeError(f"expected SGLang's DSV4PoolConfigurator, got {type(configurator).__name__}")

    from collector.sglang.runtime_compat import resolved_arg

    page_size = int(
        resolved_arg(model_runner.server_args, "page_size")
    )  # 0.5.21: derived, read through the resolving view
    if page_size != _DSV4_CUDA_PAGE_SIZE:
        raise RuntimeError(f"expected DSV4 CUDA page_size={_DSV4_CUDA_PAGE_SIZE}, got {page_size}")
    window_size = int(model_runner.model_config.window_size)
    if window_size <= 0:
        raise RuntimeError(f"invalid DSV4 SWA window_size={window_size}")

    effective_chunk = _effective_prefill_chunk_size(model_runner)
    chunk_eligible_shapes = [
        (int(batch_size), int(seq_len), int(prefix_len))
        for batch_size, seq_len, prefix_len in shapes
        if required_prefill_extend_tokens(int(batch_size), int(seq_len)) <= effective_chunk
    ]
    if not chunk_eligible_shapes:
        raise RuntimeError(f"DSV4 CSA context pool derivation has no shape within effective_chunk={effective_chunk}")

    # The free-memory-minus-slack profile serving sizes the pool from
    # (kv_cache_configurator.py:2990 _profile_available_bytes at v0.5.23).
    profiled_bytes = int(kv_cache_configurator._profile_available_bytes(model_runner.pre_model_load_memory))
    profiled_config = configurator.calculate_pool_sizes(profiled_bytes, page_size)
    compress_ratio = ATTN_KIND_TO_COMPRESS_RATIO["csa"]
    if page_size % compress_ratio != 0:
        raise RuntimeError(f"page_size={page_size} is not divisible by CSA ratio={compress_ratio}")
    c4_page_size = page_size // compress_ratio
    # sgl-deep-gemm 0.1.3 aligns the FP8 logits stride to split_kv=256 on
    # architecture major 9/10 (csrc/apis/attention.hpp).
    logits_alignment = 256
    fp32_bytes = 4
    eligible_shapes = []
    source_capacity_excluded = 0
    workspace_shape = None
    workspace_bytes = 0
    required_full_tokens = 0
    required_swa_tokens = 0
    for batch_size, seq_len, prefix_len in chunk_eligible_shapes:
        full_tokens = required_kv_alloc_tokens(
            batch_size,
            seq_len,
            prefix_len,
            page_size,
            is_prefill=True,
        )
        swa_tokens = required_swa_kv_alloc_tokens(
            batch_size,
            seq_len,
            prefix_len,
            page_size,
            window_size,
            is_prefill=True,
        )
        if (
            full_tokens > profiled_config.full_max_total_num_tokens
            or swa_tokens > profiled_config.swa_max_total_num_tokens
        ):
            source_capacity_excluded += 1
            continue

        raw_c4_width = ((prefix_len + seq_len + page_size - 1) // page_size) * c4_page_size
        logits_stride = ((raw_c4_width + logits_alignment - 1) // logits_alignment) * logits_alignment
        shape_workspace_bytes = batch_size * seq_len * logits_stride * fp32_bytes
        if shape_workspace_bytes > workspace_bytes:
            workspace_shape = (batch_size, seq_len, prefix_len)
            workspace_bytes = shape_workspace_bytes
        eligible_shapes.append((batch_size, seq_len, prefix_len))
        required_full_tokens = max(required_full_tokens, full_tokens)
        required_swa_tokens = max(required_swa_tokens, swa_tokens)

    if not eligible_shapes:
        raise RuntimeError(
            "DSV4 CSA context pool derivation has no shape within the framework-profiled "
            f"full/SWA capacities {profiled_config.full_max_total_num_tokens}/"
            f"{profiled_config.swa_max_total_num_tokens}"
        )

    swa_ratio = Fraction(str(configurator.swa_ratio))
    if not 0 < swa_ratio <= 1:
        raise RuntimeError(f"invalid DSV4 swa_full_tokens_ratio={configurator.swa_ratio}")
    required_from_swa = (required_swa_tokens * swa_ratio.denominator + swa_ratio.numerator - 1) // swa_ratio.numerator
    required_cap = max(required_full_tokens, required_from_swa)
    required_cap = ((required_cap + page_size - 1) // page_size) * page_size
    required_config = configurator.calculate_pool_sizes_from_max_tokens(required_cap, page_size)
    if (
        required_config.full_max_total_num_tokens < required_full_tokens
        or required_config.swa_max_total_num_tokens < required_swa_tokens
    ):
        raise RuntimeError(
            "SGLang DSV4 configurator could not preserve the retained full/SWA requirements: "
            f"required_full={required_full_tokens}, configured_full="
            f"{required_config.full_max_total_num_tokens}, required_swa={required_swa_tokens}, "
            f"configured_swa={required_config.swa_max_total_num_tokens}"
        )

    if workspace_bytes >= profiled_bytes:
        raise RuntimeError(
            f"DSV4 CSA logits workspace={workspace_bytes} leaves no pool budget from "
            f"profiled_bytes={profiled_bytes}; shape={workspace_shape}"
        )
    workspace_config = configurator.calculate_pool_sizes(profiled_bytes - workspace_bytes, page_size)
    workspace_cap = int(workspace_config.max_total_num_tokens)
    if required_cap > workspace_cap:
        raise RuntimeError(
            "one DSV4 CSA context worker cannot preserve all framework-feasible cells and "
            f"reserve its maximum logits workspace: required_cap={required_cap}, "
            f"workspace_cap={workspace_cap}; split only the measured Pareto-distinct groups"
        )

    print(
        "[dsv4-collector] CSA context derived pool: "
        f"sm={sm_version}, indexer_leaf=deep_gemm, page_size={page_size}, "
        f"swa_ratio={float(swa_ratio)}, effective_chunk={effective_chunk}, "
        f"eligible_shapes={len(eligible_shapes)}, source_capacity_excluded={source_capacity_excluded}, "
        f"profiled_bytes={profiled_bytes}, "
        f"profiled_full={profiled_config.full_max_total_num_tokens}, "
        f"profiled_swa={profiled_config.swa_max_total_num_tokens}, "
        f"workspace_bytes={workspace_bytes}, workspace_shape={workspace_shape}, "
        f"bytes_per_full_token={configurator.bytes_per_full_token:.6f}, "
        f"workspace_cap={workspace_cap}, required_full={required_full_tokens}, "
        f"required_swa={required_swa_tokens}, final_cap={required_cap}"
    )
    return required_cap, required_full_tokens, required_swa_tokens


def _load_model_runner(
    model_path: str,
    *,
    attn_kind: str,
    num_layers: int,
    kv_cache_dtype: str,
    device: str,
    shrink_unused_moe: bool,
    disable_weight_quant: bool,
    gemm_type: str = "bfloat16",
    tp_size: int = 1,
    max_total_tokens: int | None = None,
    csa_context_shapes: Iterable[tuple[int, int, int]] | None = None,
):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()
    torch.cuda.set_device(device)

    load_format = os.environ.get("SGLANG_LOAD_FORMAT", "dummy")
    local_model_path = _resolve_model_path(
        model_path,
        attn_kind=attn_kind,
        num_layers=num_layers,
        shrink_unused_moe=shrink_unused_moe,
        disable_weight_quant=disable_weight_quant,
        gemm_type=gemm_type,
        load_format=load_format,
    )
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    # CUDA_VISIBLE_DEVICES remaps every child to cuda:0; keep the physical GPU
    # id for NCCL port sharding so parallel workers do not collide.
    port_shard = int(os.environ.get("AIC_DSV4_PORT_SHARD", gpu_id))
    nccl_port = int(os.environ.get("AIC_DSV4_NCCL_PORT") or _pick_free_port(port_shard))
    # Mirror collect_mla_module.py's "grab sglang's own defaults, only override
    # what is strictly needed" approach: construct ServerArgs and let its
    # ``__post_init__`` derive the runtime-sizing knobs from sglang's own logic.
    # Do NOT pass collector-derived values for mem_fraction_static,
    # chunked_prefill_size, or max_prefill_tokens -- leaving them unset means:
    #   * mem_fraction_static=None -> derived from GPU memory in __post_init__
    #   * chunked_prefill_size=None -> derived from GPU memory
    #   * max_prefill_tokens         -> sglang default (16384)
    # An explicit ``max_total_tokens`` is reserved for direct diagnostics.  The
    # normal CSA-context path starts with None and, after model load, derives a
    # workspace-aware bound through SGLang's live DSV4PoolConfigurator.  HCA and
    # generation keep the framework-derived None path.

    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format=load_format,
        tp_size=1,
        trust_remote_code=True,
        disable_radix_cache=True,
        # The module benchmark below captures its own CUDA Graph and fails if
        # capture is not possible.  Keep SGLang's serving-level graph runner off
        # so it does not add unrelated full-model graph state to this collector.
        disable_decode_cuda_graph=True,
        disable_prefill_cuda_graph=True,
        kv_cache_dtype=kv_cache_dtype,
        # The bench sweep includes batch_size up to 1024 (collector's
        # ``_BATCH_SIZES``).  sglang's ``alloc_req_slots`` exposes
        # ``available_size = max_running_requests - 1`` (one slot is
        # reserved internally), so a bs=1024 cell with
        # ``max_running_requests=1024`` raises
        # ``alloc_req_slots runs out of memory: available=1023, num_reqs=1024``.
        # Bump to 1100 for headroom over the largest tested bs.  This is the
        # only sizing knob the collector must override (per-cell request slots
        # for the bs sweep); all memory/token knobs come from sglang.
        max_running_requests=1100,
        # None -> SGLang default sizing.  mem_fraction_static and
        # chunked_prefill_size also stay unset and are derived by
        # ServerArgs.__post_init__.
        max_total_tokens=max_total_tokens,
    )
    # gemm_type controls projection GEMM dispatch.  "fp8_block" → DeepGEMM
    # (matches production V4-Flash-FP8); anything else → cuBLASLt bf16.
    server_args.quantization = "fp8" if gemm_type == "fp8_block" else None
    server_args.attention_backend = "dsv4"

    print(
        f"[dsv4-collector] model_path {model_path} -> {local_model_path}; "
        f"attn_kind={attn_kind}, backend=dsv4, kv_cache_dtype={kv_cache_dtype}, "
        f"mem_fraction_static={server_args.mem_fraction_static} (sglang-derived), "
        f"chunked_prefill_size={server_args.chunked_prefill_size} (sglang-derived), "
        f"max_prefill_tokens={server_args.max_prefill_tokens} (sglang-derived), "
        f"max_total_tokens_input={server_args.max_total_tokens}, "
        f"shrink_unused_moe={shrink_unused_moe}, "
        f"disable_weight_quant={disable_weight_quant}, gemm_type={gemm_type}, "
        f"quantization={server_args.quantization}, tp_size={tp_size}, nccl_port={nccl_port}"
    )

    _set_envs_and_config(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    from collector.sglang.runtime_compat import init_runtime_config

    _runner_parallel_kwargs = init_runtime_config(server_args, gpu_id, nccl_port=nccl_port, model_config=model_config)
    model_runner = ModelRunner(
        model_config=model_config,
        # Use sglang's own __post_init__-derived value, not a collector knob.
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        nccl_port=nccl_port,
        server_args=server_args,
        **_runner_parallel_kwargs,
    )
    derived_requirements = None
    if csa_context_shapes is not None:
        if max_total_tokens is not None:
            raise RuntimeError("CSA context live pool derivation cannot be combined with an explicit token cap")
        derived_requirements = _derive_csa_context_pool_cap(
            model_runner,
            csa_context_shapes,
        )
        # A resolved ServerArgs is read-only (server_args.py:474-490 at v0.5.23);
        # post-publish config changes go through the context's config bags
        # (runtime_context.py:1103 override(source, **fields)). The configurator
        # reads the cap from get_schedule().max_total_tokens
        # (kv_cache_configurator.py:3090-3102 _apply_token_constraints).
        from sglang.srt.runtime_context import get_context

        get_context().override("collector-dsv4-csa-context", max_total_tokens=derived_requirements[0])
    # Model construction is separate from serving-state setup.
    model_runner.alloc_memory_pool()
    if derived_requirements is not None:
        _, required_full_tokens, required_swa_tokens = derived_requirements
        actual_full_tokens = int(model_runner.full_max_total_num_tokens)
        actual_swa_tokens = int(model_runner.swa_max_total_num_tokens)
        if actual_full_tokens < required_full_tokens or actual_swa_tokens < required_swa_tokens:
            raise RuntimeError(
                "SGLang DSV4 pool shrank below the derived retained-case requirements: "
                f"actual_full={actual_full_tokens}, required_full={required_full_tokens}, "
                f"actual_swa={actual_swa_tokens}, required_swa={required_swa_tokens}"
            )
    model_runner.init_attention_backends()

    allocator = model_runner.token_to_kv_pool_allocator
    pool_parts = []
    for name in (
        "max_total_num_tokens",
        "full_max_total_num_tokens",
        "swa_max_total_num_tokens",
        "c4_max_total_num_tokens",
        "c128_max_total_num_tokens",
        "c4_state_pool_size",
        "c128_state_pool_size",
    ):
        if hasattr(model_runner, name):
            pool_parts.append(f"{name}={getattr(model_runner, name)}")
    if hasattr(allocator, "debug_print"):
        pool_parts.append(allocator.debug_print().strip())
    elif hasattr(allocator, "available_size"):
        pool_parts.append(f"available_size={allocator.available_size()}")
    print("[dsv4-collector] pool " + ", ".join(pool_parts))
    return model_runner


def _make_reqs(batch_size: int, seq_len: int, *, decode: bool, prefix_len: int = 0):
    """Requests that extend fill ids [prefix_len, prefix_len + seq_len).

    SGLang 0.5.23 tracks prefill progress as Req.prefix_len and the end of this
    round's extend as Req.extend_end (managers/schedule_batch.py:1103-1106, :1218-1220;
    sglang #42824, #42822); the KV of [0, prefix_len) sits in the request's row
    (``_hold_prefix_rows``), not on the request (#42825). The fields are set as SGLang's
    offline batch builder sets them (benchmark/one_batch.py:450-476).
    """
    from array import array

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    full_len = prefix_len + seq_len
    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=str(i),
            origin_input_text="",
            origin_input_ids=array("q", torch.randint(0, 10000, (full_len,)).tolist()),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
        req.logprob_start_len = 0
        req.prefix_len = prefix_len
        req.extend_end = full_len
        if decode:
            req.cached_tokens = 0
            req.already_computed = 0
        reqs.append(req)
    return reqs


def _hold_prefix_rows(model_runner, reqs, prefix_len: int) -> None:
    """Give every request the row an earlier prefill leaves: KV for [0, prefix_len) written.

    alloc_for_extend reads a held row's [0, prefix_len) as the prefix
    (mem_cache/allocation.py:363-371 at v0.5.23). The rows are installed as SGLang
    installs a prefilled request outside a prefix tree (PD decode,
    disaggregation/decode.py:2371-2386, 2408-2422): ReqToTokenPool.alloc binds a row
    (mem_cache/memory_pool.py:419-435), the prefix slots are written into it, and
    on a hybrid-SWA pool only the window tail holds SWA slots, recorded as the SWA
    eviction cursor (ReqKvInfo.set_evicted_seqlen, managers/schedule_batch.py:1021).
    The prefix slots come from runtime_limits.alloc_prefix_indices, which allocates
    that SWA tail the same way (alloc_extend_swa_tail, mem_cache/allocator/swa.py:356).
    """
    if prefix_len <= 0:
        return
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    prefix_indices = _alloc_prefix_indices(model_runner, len(reqs), prefix_len)
    pool = model_runner.req_to_token_pool
    if pool.alloc(reqs) is None:
        raise RuntimeError(f"no free request rows for {len(reqs)} prefix rows; available={pool.available_size()}")
    allocator = model_runner.token_to_kv_pool_allocator
    window_size = getattr(model_runner.model_config, "window_size", None)
    page_size = getattr(allocator, "page_size", None)
    swa_evicted_seqlen = 0
    if (
        callable(getattr(allocator, "alloc_extend_swa_tail", None))
        and isinstance(window_size, int)
        and window_size > 0
        and isinstance(page_size, int)
        and page_size > 1
    ):
        # alloc_prefix_indices' page-aligned window start: SWA KV exists from here on.
        swa_evicted_seqlen = (max(0, prefix_len - window_size) // page_size) * page_size
    for req, indices in zip(reqs, prefix_indices, strict=True):
        pool.write((req.kv.req_pool_idx, slice(0, prefix_len)), indices)
        req.kv.kv_allocated_len = prefix_len
        req.kv.kv_committed_len = prefix_len
        if swa_evicted_seqlen:
            req.kv.set_evicted_seqlen(ComponentType.SWA, swa_evicted_seqlen)


# Distinct history keys: a longer history repeats them, so writing it costs one store call
# per this many keys (16 MiB of activations) and no further sampling. A key then recurs a
# handful of times in the longest histories, far below kMaxNumTie = 2048, the number of
# threshold-bin candidates the top-k stages before it re-derives them
# (kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh:178 at v0.5.23).
_HISTORY_ROWS = 1 << 16


def _write_index_history(model_runner, attention_module, forward_batch, history_length: int) -> None:
    """Write the index keys of the cached history the C4 indexer scores and selects from.

    _hold_prefix_rows only binds slots. The indexer pool starts as zeros
    (DeepSeekV4IndexerPool._create_buffer, mem_cache/deepseek_v4_memory_pool.py:594-632)
    and one ModelRunner serves every point of a row, so the history held zeros or an
    earlier point's keys. A zero key carries a zero scale: every such candidate scored
    exactly 0.0 and the top-k was timed on ties, a case serving does not have.

    Serving writes key j of a request when it computes token 4j+3, at pool slot
    raw_out_loc // 4 (_init_compressed_attn_metadata_kernel,
    kernels/ops/attention/dsv4/metadata_kernel.py:101), as 128 FP8 values and one fp32
    scale (kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh:197-199, "matching the
    indexer cache layout (fused_store_indexer_cache)"). The indexer reads that buffer
    through the request's page table (forward_c4_indexer, layers/attention/dsv4/
    indexer.py:1074-1101). Seeded N(0, 1) keys, the magnitude of a key normalized with a
    unit gain, go to the same slots through the pool's writer for that layout,
    DeepSeekV4TokenToKVPool.set_index_k_fused (deepseek_v4_memory_pool.py:2275-2284 ->
    fused_store_cache type="indexer", kernels/ops/attention/dsv4/attn.py:75 ->
    fused_store_indexer_cache, kernels/jit/csrc/deepseek_v4/store.cuh:147), outside the
    measurement. They stand for no particular checkpoint distribution. Same treatment as
    the DSA modules (collect_mla_module._initialize_dsa_history). Paths are relative to
    python/sglang/srt unless they start with kernels/; citations are v0.5.23.
    """
    indexer = getattr(attention_module, "indexer", None)
    keys_per_request = history_length // 4
    if indexer is None or keys_per_request <= 0:
        return
    if indexer.use_fp4_indexer:
        # set_index_k_fused writes the FP8 layout; an FP4 pool has 68-byte keys.
        raise RuntimeError("DSV4 index history is written in the FP8 indexer layout; the FP4 indexer is enabled")
    # The slot of each group's fourth token, as serving derives the key's slot from it.
    token_slots = model_runner.req_to_token_pool.req_to_token[
        forward_batch.req_pool_indices.long(), 3 : 4 * keys_per_request : 4
    ]
    slots = (token_slots.long() // 4).flatten().contiguous()
    generator = torch.Generator(device=slots.device)
    generator.manual_seed(0)
    keys = torch.randn(
        (min(slots.numel(), _HISTORY_ROWS), indexer.head_dim),
        generator=generator,
        device=slots.device,
        dtype=torch.bfloat16,
    )
    pool = model_runner.token_to_kv_pool
    for start in range(0, slots.numel(), _HISTORY_ROWS):
        chunk = slots[start : start + _HISTORY_ROWS]
        pool.set_index_k_fused(layer_id=attention_module.layer_id, loc=chunk, cache_k=keys[: chunk.numel()])


def _give_index_scores_magnitude(model_runner, attention_module) -> None:
    """Bring the C4 index scores of a dummy-weight module into fp16's normal range.

    A score is sum_h relu(q_h . k) * w_h * k_scale (fp8_paged_mqa_logits; the torch
    reference is fp8_paged_mqa_logits_torch, layers/attention/dsv4/indexer.py:82-145), with
    q = indexer.wq_b(q_norm(wq_a(x))) (MqaAttentionBase._normalize_q_lora, models/
    deepseek_v4.py:1736-1773; C4Indexer.compute_q, indexer.py:1220-1244),
    w = indexer.weights_proj(x) * weight_scale (compute_weights, indexer.py:1246-1250) and
    k = indexer.compressor.norm(...) (Compressor, layers/attention/dsv4/compressor.py:76-78).
    initialize_dummy_weights draws every float parameter from U(-1e-3, 1e-3), the two
    normalization gains included (model_loader/weight_utils.py:1740-1776), so q_norm's
    output, the keys and w were each about a thousand times smaller than a unit-gain
    module's and the scores came out near 1e-9. The top-k bins scores as fp16 first
    (extract_coarse_bin, __float2half_rn; kernels/jit/include/sgl_kernel/deepseek_v4/
    topk_impl.cuh:87-93). Scores below fp16's smallest subnormal (6e-8) all round to zero
    and share its bin, and a threshold bin holding more than kMaxNumTie nonzero
    candidates is re-derived in exact-key radix rounds (refine_ties, topk_impl.cuh:571-574,
    taken at :766-768), work that scores spread over the fp16 bins do not need.

    Unit gains on q_norm and the indexer compressor's norm, and U(-1, 1) on weights_proj
    through SGLang's own initializer, give scores a standard deviation of a few tenths.
    Those three are plain bf16/fp32 parameters read at every forward. wq_b is left alone:
    under FP8 block quantization its weight is stored against scales SGLang repacks at
    load (Fp8LinearMethod.process_weights_after_loading_block_quant, layers/quantization/
    fp8.py:743-850), which new values would not match. Values only: no kernel in the
    module is chosen by them. A loaded checkpoint keeps its own weights. Citations are
    v0.5.23, paths relative to python/sglang/srt unless they start with kernels/.
    """
    indexer = getattr(attention_module, "indexer", None)
    if indexer is None or model_runner.server_args.load_format != "dummy":
        return
    from sglang.srt.model_loader.weight_utils import initialize_dummy_weights

    attention_module.q_norm.weight.data.fill_(1.0)
    indexer.compressor.norm.weight.data.fill_(1.0)
    initialize_dummy_weights(indexer.weights_proj, low=-1.0, high=1.0)


def _serving_tree_cache(model_runner):
    """The prefix cache a radix-disabled server builds over this runner's pools, built once.

    The scheduler builds it in kv_cache_builder.build_kv_cache
    (mem_cache/kv_cache_builder.py:330-461 at v0.5.23): with --disable-radix-cache,
    registry.default_radix_cache_factory returns UnifiedRadixCache in its disabled
    mode for every layout but pure SWA (mem_cache/registry.py:80-96). ChunkCache,
    which earlier collectors used, is gone (sglang #43201). A disabled tree shares
    no prefix, so it frees out-of-window SWA slots of a held row on its extend
    (ScheduleBatch.maybe_evict_swa, managers/schedule_batch.py:4068-4129), as serving does.
    """
    tree_cache = getattr(model_runner, "_aic_tree_cache", None)
    if tree_cache is not None:
        return tree_cache
    from sglang.srt.configs.model_config import is_deepseek_dsa
    from sglang.srt.mem_cache.allocator.swa import is_swa_req_ring
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.kv_cache_builder import uses_ssm_state
    from sglang.srt.mem_cache.registry import TreeCacheBuildContext, create_tree_cache
    from sglang.srt.runtime_context import get_schedule

    pool = model_runner.token_to_kv_pool
    allocator = model_runner.token_to_kv_pool_allocator
    dsv4_pool = isinstance(pool, DeepSeekV4TokenToKVPool)
    # kv_cache_builder.py:332-335, :349-360.
    is_hybrid_swa = model_runner.is_hybrid_swa and (not dsv4_pool or pool.needs_paged_swa_allocator)
    params = CacheInitParams(
        disable=True,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=model_runner.page_size,
        chunked_prefill_size=get_schedule().chunked_prefill_size,
        sliding_window_size=model_runner.sliding_window_size if is_hybrid_swa else None,
        swa_gates_prefix_match=not (is_swa_req_ring(allocator) or (dsv4_pool and pool.request_window is not None)),
    )
    tree_cache = create_tree_cache(
        TreeCacheBuildContext(
            server_args=model_runner.server_args,
            params=params,
            is_hybrid_swa=is_hybrid_swa,
            full_tokens_per_layer=model_runner.full_max_total_num_tokens if is_hybrid_swa else None,
            is_hybrid_ssm=uses_ssm_state(model_runner.model_config),
            is_dsa=is_deepseek_dsa(model_runner.model_config.hf_config),
            enable_hierarchical_cache=False,
            disable_radix_cache=True,
            tp_worker=None,  # read only to attach HiCache or a cache linker, both off here
            model_config=model_runner.model_config,
            tp_size=1,
            tp_rank=0,
            tp_group=None,
        )
    )
    model_runner._aic_tree_cache = tree_cache
    return tree_cache


def _build_forward_batch(
    model_runner,
    batch_size: int,
    seq_len: int,
    *,
    is_prefill: bool,
    prefix_len: int = 0,
):
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    cached_prefix_len = prefix_len if is_prefill else seq_len
    extend_seq_len = seq_len if is_prefill else 0
    reqs = _make_reqs(batch_size, extend_seq_len, decode=not is_prefill, prefix_len=cached_prefix_len)
    _hold_prefix_rows(model_runner, reqs, cached_prefix_len)
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=_serving_tree_cache(model_runner),
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )

    with _temporarily_chunked_alloc_extend(model_runner, batch_size * extend_seq_len):
        batch.prepare_for_extend()
    if is_prefill:
        # prepare_for_extend stages the ids on the host (schedule_batch.py:3069-3070);
        # the forward stream receives them as SGLang's offline builder hands them over
        # (benchmark/one_batch.py:516-523), so the ForwardBatch counts its real tokens
        # (forward_batch_info.py:1364-1382).
        batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        batch.prefill_input_ids_cpu = None
    else:
        for req in batch.reqs:
            req.output_ids.append(0)
        # The sampled token of the previous step (benchmark/one_batch.py:536-538).
        batch.input_ids = torch.zeros(batch_size, dtype=torch.int64, device=batch.device)
        batch.prepare_for_decode()

    # forward_batch_info.py:958-1142 at v0.5.23 also binds the KV location plan.
    forward_batch = ForwardBatch.init_new(batch, model_runner, return_hidden_states_before_norm=False)
    model_runner.attn_backend.init_forward_metadata(forward_batch)
    return forward_batch


def _make_inputs(
    model_runner,
    *,
    batch_size: int,
    seq_len: int,
    is_prefill: bool,
    device: str,
    prefix_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = model_runner.model.config.hidden_size
    max_pos = getattr(model_runner.model_config.hf_config, "max_position_embeddings", None)
    if is_prefill:
        full_len = prefix_len + seq_len
        if max_pos is not None and full_len > max_pos:
            raise ValueError(
                f"context full_len={full_len} exceeds max_position_embeddings={max_pos} "
                f"(seq_len={seq_len}, prefix_len={prefix_len})"
            )
        n_tokens = batch_size * seq_len
        positions = (
            torch.arange(prefix_len, prefix_len + seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
            .flatten()
        )
    else:
        if max_pos is not None and seq_len >= max_pos:
            raise ValueError(
                f"decode seq_len={seq_len} >= max_position_embeddings={max_pos}; "
                f"max valid decode seq_len is {max_pos - 1}"
            )
        n_tokens = batch_size
        positions = torch.full((batch_size,), seq_len, dtype=torch.int64, device=device)

    hidden_states = torch.randn(
        n_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    return hidden_states, positions


def _bench_cuda_events(
    kernel_func,
    num_warmup: int,
    num_iterations: int,
    graph_repeat: int = 1,
    device: str = "cuda:0",
) -> dict[str, float]:
    """Benchmark through AIC's benchmark_with_power helper.

    benchmark_with_power handles warmup, CUDA Graph capture/replay, optional
    power sampling, and graph-private-pool teardown.  Capture failure is a hard
    error: allow_graph_fail=False and used_cuda_graph is checked explicitly.
    """

    if num_iterations < 3:
        raise ValueError("num_iterations must be at least 3")
    if graph_repeat < 1:
        raise ValueError("graph_repeat must be at least 1")

    def timed_kernel():
        with torch.no_grad():
            return kernel_func()

    with benchmark_with_power(
        device=torch.device(device),
        kernel_func=timed_kernel,
        num_warmups=num_warmup,
        num_runs=num_iterations,
        repeat_n=graph_repeat,
        allow_graph_fail=False,
    ) as result:
        pass

    if not result.get("used_cuda_graph", False):
        raise RuntimeError("benchmark_with_power did not use CUDA Graph")

    latency_ms = float(result["latency_ms"])
    return {
        "mean_ms": latency_ms,
        "median_ms": latency_ms,
        "min_ms": latency_ms,
        "max_ms": latency_ms,
        "std_ms": 0.0,
        "n": int(result.get("num_runs_executed", num_iterations)),
        "used_cuda_graph": True,
        "power_stats": result.get("power_stats"),
        "throttled": bool(result.get("throttled", False)),
        "phase_seconds": result.get("phase_seconds"),
    }


def _resolve_local_heads(*, native_heads: int, module_heads: int | None, tp_size: int) -> int:
    """Resolve the rank-LOCAL head count persisted as ``num_heads``.

    The unified convention persists the head count this rank's module actually
    runs with; consumers derive the model-native count as
    ``num_heads * tp_size``.  Local is derived from the HF config's
    ``num_attention_heads``, which no TP layout rewrites.
    ``attention_module.n_heads`` may report the native or the rank-local count
    depending on the sglang version, so it is only cross-checked, never
    persisted — a value matching neither is a geometry surprise worth failing
    the case over. The heads the module actually runs (``n_local_heads``) are
    checked against this value by ``_check_module_local_heads``.
    """
    local_heads = max(1, int(native_heads) // tp_size)
    if module_heads is not None and int(module_heads) not in (int(native_heads), local_heads):
        raise RuntimeError(
            f"DSV4 attention head geometry mismatch: module n_heads={int(module_heads)} is neither "
            f"config-native {int(native_heads)} nor rank-local {local_heads} (tp_size={tp_size})"
        )
    return local_heads


def _check_module_local_heads(attention_module, local_heads: int, tp_size: int) -> None:
    """Fail a row whose module runs a different head count than its persisted ``num_heads``.

    MQALayer runs n_local_heads = n_heads // attn_tp_size query heads
    (srt/models/deepseek_v4.py:1035 at v0.5.23); a row labeled with
    ``local_heads`` must time exactly that many.
    """
    module_local_heads = getattr(attention_module, "n_local_heads", None)
    if module_local_heads is not None and int(module_local_heads) != int(local_heads):
        raise RuntimeError(
            f"DSV4 attention runs n_local_heads={int(module_local_heads)}, but the row would persist "
            f"num_heads={int(local_heads)} at tp_size={tp_size}"
        )


def _log_result(
    *,
    output_path: str | None,
    model_path: str,
    mode: str,
    attn_kind: str,
    compress_ratio: int,
    batch_size: int,
    seq_len: int,
    kv_cache_dtype: str,
    latency_ms: float,
    version: str,
    device_name: str,
    power_stats: dict | None = None,
    perf_filename_prefix: str = "dsv4",
    gemm_type: str = "bfloat16",
    tp_size: int = 1,
    step: int | None = None,
    num_heads: int,
) -> None:
    # V4-Flash output layout: ONE CSV per (attn_kind, mode) — 3 kinds x 2
    # modes = 6 files total, regardless of how many (tp_size, gemm_type)
    # subprocesses run.  Within each file, rows are disambiguated by the
    # ``tp_size``, ``gemm_type``, ``batch_size``, ``isl`` columns.
    # ``log_perf`` is file-locked so concurrent appends from different
    # subprocesses to the same kind+mode file are safe.
    # Non-V4-Flash callers (legacy ``dsv4`` MLA module) still use the old
    # per-(prefix, kind) filename layout to avoid behavior breaks.
    if perf_filename_prefix.startswith("dsv4"):
        consolidated_filename = f"dsv4_{attn_kind}_{mode}_module_perf.txt"
    else:
        consolidated_filename = f"{perf_filename_prefix}_{attn_kind}_{mode}_module_perf.txt"
    perf_filename = _resolve_perf_path(output_path, consolidated_filename)
    is_prefill = mode == "context"
    step_value = step if step is not None else (0 if is_prefill else seq_len)
    if not log_perf(
        item_list=[
            {
                "model": _canonical_dsv4_model_id(model_path),
                "architecture": "DeepseekV4ForCausalLM",
                "mla_dtype": "bfloat16",
                "kv_cache_dtype": kv_cache_dtype,
                "gemm_type": gemm_type,
                # Rank-LOCAL by the unified convention (native = num_heads * tp_size).
                "num_heads": num_heads,
                "batch_size": batch_size,
                "isl": seq_len if is_prefill else 1,
                "tp_size": tp_size,
                "step": step_value,
                "compress_ratio": compress_ratio,
                "latency": f"{latency_ms:.4f}",
            }
        ],
        framework="SGLang",
        version=version,
        device_name=device_name,
        # op_name still encodes the run config so a single-CSV view can group
        # by op_name when needed (e.g. for plotting per-(kind, tp, gemm)).
        op_name=f"{perf_filename_prefix}_{attn_kind}_{mode}_module",
        kernel_source="compressed_flashmla",
        perf_filename=perf_filename,
        power_stats=power_stats,
    ):
        raise RuntimeError(f"failed to persist DeepSeek-V4 attention row to {perf_filename}")


def run_dsv4_mla_module(
    *,
    model_path: str,
    mode: str,
    attn_kind: str,
    batch_size: int,
    seq_lens_by_prefix: dict[int, list[int]],
    layer_id: int = 0,
    num_layers: int = 1,
    kv_cache_dtype: str = "fp8_e4m3",
    num_warmup: int = 5,
    num_iterations: int = 20,
    graph_repeat: int = 1,
    device: str = "cuda:0",
    output_path: str | None = None,
    shrink_unused_moe: bool = True,
    disable_weight_quant: bool = True,
    perf_filename_prefix: str = "dsv4",
    gemm_type: str = "bfloat16",
    tp_size: int = 1,
    max_total_tokens: int | None = None,
) -> list[dict[str, float]]:
    """Measure one batch's (prefix, seq_len) points on one ModelRunner.

    A point the device cannot run is refused with its reason before anything is
    allocated for it; the refusals end the run on one classified error once the
    other points are measured.
    """
    is_prefill = mode == "context"
    if not is_prefill and set(seq_lens_by_prefix) != {0}:
        raise ValueError("prefix_len is only supported for context/extend collection")
    case_shapes = [
        (batch_size, seq_len, prefix_len) for prefix_len, seq_lens in seq_lens_by_prefix.items() for seq_len in seq_lens
    ]
    if not case_shapes:
        raise RuntimeError(f"dsv4-flash mode={mode} bs={batch_size}: the row lists no points")
    phase_totals: dict[str, float] = {}
    # gc.collect() after every point cost 0.2-0.4 s (half of a context sweep); collect periodically instead.
    point_gc = PeriodicGarbageCollector()

    refused: dict[tuple, str] = {}
    if is_prefill:
        # SGLang derives its prefill chunk from device memory (ServerArgs), and
        # serving chunks a longer prefill.
        try:
            from collector.sglang.deepseekv4_sparse_modules import _sglang_chunked_prefill_size
        except ModuleNotFoundError:
            from deepseekv4_sparse_modules import _sglang_chunked_prefill_size
        chunk = _sglang_chunked_prefill_size()
        refuse(case_shapes, refused, lambda shape: over_chunk(shape, chunk))
    runnable = [shape for shape in case_shapes if shape not in refused]
    if not runnable:
        raise RuntimeError(refusal_summary(refused, len(case_shapes)))

    # SGLang derives compressed-pool sizes from max_total_tokens. Direct
    # probes with an explicit value must account for both the full pool and
    # the smaller page-rounded SWA tail pool before ModelRunner construction.
    #
    # HCA always takes this path since sglang 0.5.21: the ratio-128 stage prices a
    # full token at ~1/128 of a paged slot (pool_configurator.py:1264-1283
    # _compressed_bytes_per_full_token / _get_bytes_per_full_token), so the
    # framework-derived pool on an otherwise empty 140 GB device reaches ~2e9 full
    # tokens, and the hybrid allocator's int64 full_to_swa_index_mapping
    # (mem_cache/allocator/swa.py:99-108, size + page_size entries, not priced by the
    # configurator) then needs ~15 GiB that the sizing already handed to the pools
    # (OOM in SWATokenToKVPoolAllocator.__init__, kv_cache_configurator.py:2080,
    # on H20). Serving never meets this corner because weights leave far
    # less free memory; the collector sizes the pool to what the planned shapes
    # allocate instead. sglang treats the value as an upper bound only
    # (kv_cache_configurator.py:2269-2277 _apply_token_constraints: min(profiled,
    # user cap)), so no cell gains capacity it would not have in serving.
    if max_total_tokens is None and attn_kind == "hca":
        max_total_tokens = 0
    if max_total_tokens is not None:
        max_full_alloc = 0
        max_swa_alloc = 0
        for batch_size, seq_len, cur_prefix in runnable:
            max_full_alloc = max(
                max_full_alloc,
                required_kv_alloc_tokens(
                    batch_size,
                    seq_len,
                    cur_prefix,
                    _DSV4_CUDA_PAGE_SIZE,
                    is_prefill=is_prefill,
                ),
            )
            max_swa_alloc = max(
                max_swa_alloc,
                required_swa_kv_alloc_tokens(
                    batch_size,
                    seq_len,
                    cur_prefix,
                    _DSV4_CUDA_PAGE_SIZE,
                    _DSV4_SWA_WINDOW_SIZE,
                    is_prefill=is_prefill,
                ),
            )
        max_total_tokens = max(max_total_tokens, max_full_alloc, max_swa_alloc * _DSV4_SWA_TO_FULL_SCALE)
        max_total_tokens = (
            (max_total_tokens + _DSV4_CUDA_PAGE_SIZE - 1) // _DSV4_CUDA_PAGE_SIZE
        ) * _DSV4_CUDA_PAGE_SIZE

    compress_ratio = ATTN_KIND_TO_COMPRESS_RATIO[attn_kind]
    if tp_size not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"tp_size must be a power of 2 in [1, 32]; got {tp_size}")
    _refuse_tp_emulation(tp_size)
    model_runner = _load_model_runner(
        model_path,
        attn_kind=attn_kind,
        num_layers=max(num_layers, layer_id + 1),
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        shrink_unused_moe=shrink_unused_moe,
        disable_weight_quant=disable_weight_quant,
        gemm_type=gemm_type,
        tp_size=tp_size,
        max_total_tokens=max_total_tokens,
        csa_context_shapes=(runnable if is_prefill and attn_kind == "csa" and max_total_tokens is None else None),
    )

    attention_module = model_runner.model.model.layers[layer_id].self_attn
    actual_ratio = getattr(attention_module, "compress_ratio", None)
    if actual_ratio != compress_ratio:
        raise RuntimeError(f"target layer compress_ratio mismatch: expected {compress_ratio}, got {actual_ratio}")
    local_attention_heads = _resolve_local_heads(
        native_heads=int(model_runner.model_config.hf_config.num_attention_heads),
        module_heads=getattr(attention_module, "n_heads", None),
        tp_size=tp_size,
    )
    _check_module_local_heads(attention_module, local_attention_heads, tp_size)
    _give_index_scores_magnitude(model_runner, attention_module)

    print(
        f"[dsv4-collector] layer={layer_id}, attn_kind={attn_kind}, "
        f"compress_ratio={actual_ratio}, mode={mode}, prefix_lens={sorted(seq_lens_by_prefix)}, "
        f"n_local_heads={getattr(attention_module, 'n_local_heads', None)}, "
        f"wo_a_fp8={getattr(attention_module, 'wo_a_fp8', None)}"
    )

    version = get_version("sglang")
    device_name = torch.cuda.get_device_name(device)
    results = []
    failures: list[tuple[int, int, str]] = []
    sweep_label = f"kind={attn_kind} mode={mode} tp={tp_size} gemm={gemm_type} bs={batch_size}"
    try:
        kv_capacity = _kv_pool_capacity_tokens(model_runner)
        kv_page_size = _kv_pool_page_size(model_runner)
        swa_capacity = _swa_kv_pool_capacity_tokens(model_runner)
        swa_window_size = getattr(getattr(model_runner, "model_config", None), "window_size", None)
        # Refuse what the built runner cannot run: a prefill beyond the chunk it
        # resolved (the same live value sized the CSA pool), or a point beyond its
        # full or sliding-window KV pool. Paged alloc_extend rounds each request's KV
        # span up to a page and reserves one more, so the pools are checked in pages.
        if is_prefill:
            runtime_chunk = _effective_prefill_chunk_size(model_runner)
            refuse(case_shapes, refused, lambda shape: over_chunk(shape, runtime_chunk))
        if kv_capacity is not None:
            refuse(case_shapes, refused, lambda shape: over_kv_pool(shape, kv_page_size, kv_capacity, is_prefill))
        if swa_capacity is not None and isinstance(swa_window_size, int) and swa_window_size > 0:
            refuse(
                case_shapes,
                refused,
                lambda shape: _over_swa_pool(shape, kv_page_size, swa_window_size, swa_capacity, is_prefill),
            )
        runnable = [shape for shape in case_shapes if shape not in refused]
        if not runnable:
            raise RuntimeError(refusal_summary(refused, len(case_shapes)))
        for _batch, seq_len, cur_prefix in runnable:
            print(f"\n{mode}: batch_size={batch_size}, seq_len={seq_len}, prefix_len={cur_prefix}")
            point_clock = PhaseClock()
            try:
                forward_batch = _build_forward_batch(
                    model_runner,
                    batch_size,
                    seq_len,
                    is_prefill=is_prefill,
                    prefix_len=cur_prefix,
                )
                point_clock.mark("forward_batch")
                # Context history is the prefix; generation history is the whole past KV.
                _write_index_history(
                    model_runner, attention_module, forward_batch, cur_prefix if is_prefill else seq_len
                )
                point_clock.mark("index_history")
                hidden_states, positions = _make_inputs(
                    model_runner,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    is_prefill=is_prefill,
                    device=device,
                    prefix_len=cur_prefix,
                )
                point_clock.mark("inputs")

                def kernel_func(model_runner=model_runner):
                    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
                    from sglang.srt.model_executor.runner import model_capture_mode

                    # Production decode graphs enter model_capture_mode,
                    # which selects DSV4's multi-stream overlap path.
                    capture_context = model_capture_mode() if not is_prefill else contextlib.nullcontext()
                    with (
                        capture_context,
                        forward_context(ForwardContext(attn_backend=model_runner.attn_backend)),
                    ):
                        return attention_module(
                            x=hidden_states,
                            positions=positions,
                            forward_batch=forward_batch,
                        )

                stats = _bench_cuda_events(
                    kernel_func,
                    num_warmup=num_warmup,
                    num_iterations=num_iterations,
                    graph_repeat=graph_repeat,
                    device=device,
                )
                point_clock.mark("bench")
                point_clock.absorb(stats.get("phase_seconds"), prefix="bench_")
                print(
                    f"  latency mean={stats['mean_ms']:.4f} ms, "
                    f"median={stats['median_ms']:.4f} ms, "
                    f"min={stats['min_ms']:.4f} ms, max={stats['max_ms']:.4f} ms, "
                    f"std={stats['std_ms']:.4f} ms, n={stats['n']}"
                )
                _log_result(
                    output_path=output_path,
                    model_path=model_path,
                    mode=mode,
                    attn_kind=attn_kind,
                    compress_ratio=compress_ratio,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    kv_cache_dtype=kv_cache_dtype,
                    latency_ms=stats["mean_ms"],
                    version=version,
                    device_name=device_name,
                    power_stats=stats.get("power_stats"),
                    perf_filename_prefix=perf_filename_prefix,
                    gemm_type=gemm_type,
                    tp_size=tp_size,
                    step=cur_prefix if is_prefill else None,
                    num_heads=local_attention_heads,
                )
                stats.update(
                    {
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "compress_ratio": compress_ratio,
                        "prefix_len": cur_prefix,
                    }
                )
                results.append(stats)
            except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
                print(
                    f"[WARN] dsv4-flash {sweep_label} bs={batch_size} sl={seq_len} "
                    f"prefix={cur_prefix}: OOM; skipping this shape"
                )
                failures.append((seq_len, cur_prefix, "OOM"))
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception as exc:
                traceback.print_exc()
                print(
                    f"[WARN] dsv4-flash {sweep_label} bs={batch_size} sl={seq_len} "
                    f"prefix={cur_prefix}: {type(exc).__name__}; skipping this shape"
                )
                failures.append((seq_len, cur_prefix, type(exc).__name__))
            finally:
                cleanup_errors = []
                for _cleanup_label, _cleanup_step in (
                    ("req_to_token_pool.clear", model_runner.req_to_token_pool.clear),
                    ("token_to_kv_pool_allocator.clear", model_runner.token_to_kv_pool_allocator.clear),
                    ("torch.cuda.empty_cache", torch.cuda.empty_cache),
                    ("gc.collect", point_gc.maybe_collect),
                ):
                    try:
                        _cleanup_step()
                    except Exception as _cleanup_exc:
                        cleanup_errors.append(f"{_cleanup_label}: {type(_cleanup_exc).__name__}: {_cleanup_exc}")
                if cleanup_errors:
                    raise RuntimeError(
                        f"dsv4-flash {sweep_label} bs={batch_size} sl={seq_len} "
                        f"prefix={cur_prefix}: cleanup failed: {'; '.join(cleanup_errors)}"
                    )
                point_clock.mark("cleanup")
                point_clock.add_to(phase_totals)
                print(
                    f"  [dsv4 point timing] {sweep_label} bs={batch_size} sl={seq_len} prefix={cur_prefix}: "
                    + point_clock.render()
                )
    finally:
        final_cleanup_errors = []
        del model_runner
        for cleanup_name, cleanup_fn in (
            ("torch.cuda.empty_cache", torch.cuda.empty_cache),
            ("gc.collect", gc.collect),
        ):
            try:
                cleanup_fn()
            except Exception as cleanup_exc:
                final_cleanup_errors.append(f"{cleanup_name}: {type(cleanup_exc).__name__}: {cleanup_exc}")
        if final_cleanup_errors:
            raise RuntimeError(f"dsv4-flash {sweep_label}: final cleanup failed: {'; '.join(final_cleanup_errors)}")
    summary = f"ok={len(results)} error={len(failures)} refused={len(refused)} total={len(case_shapes)}"
    if failures:
        errored = ", ".join(f"(bs={batch_size},sl={s},prefix={p},reason={r})" for s, p, r in failures)
        print(f"[WARN] dsv4-flash {sweep_label}: errored shapes: {errored}")
    print(f"[dsv4 timing] {sweep_label}: {render_phase_totals(phase_totals)}")
    print(f"[dsv4-collector] {sweep_label}: {summary}")
    if not results or failures:
        raise RuntimeError(f"dsv4-flash {sweep_label}: {summary}")
    if refused:
        raise RuntimeError(refusal_summary(refused, len(case_shapes)))
    return results


def _over_swa_pool(shape: tuple, page_size: int, window_size: int, capacity: int, is_prefill: bool) -> str | None:
    """Why this point's sliding-window KV cannot fit SGLang's SWA KV pool on this device, or None."""
    needed = required_swa_kv_alloc_tokens(*shape, page_size, window_size, is_prefill=is_prefill)
    if needed <= capacity:
        return None
    return f"{KERNEL_LIMIT}: does not fit: the point needs {needed} SWA KV tokens, the SWA KV pool holds {capacity}"


# ═══════════════════════════════════════════════════════════════════════
# Subprocess-isolated worker (registry path)
# ═══════════════════════════════════════════════════════════════════════


def _run_subprocess(
    *,
    mode: str,
    attn_kind: str,
    model_path: str,
    kv_cache_dtype_sglang: str,
    batch_size: int,
    output_path: str,
    gpu_id: int,
    inner_shapes: Iterable[tuple[int, Iterable[int]]],
    gemm_type: str = "bfloat16",
    tp_size: int = 1,
):
    """Run one (attn_kind, tp, gemm, bs) row in a subprocess: its (prefix, seq_lens) points.

    Builds one ``ModelRunner`` for the batch and measures every point.  Per-point
    crash isolation is handled by ``run_dsv4_mla_module``'s try/except per forward.
    """
    # Before the child pays for an SGLang import.
    _refuse_tp_emulation(tp_size)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["AIC_DSV4_PORT_SHARD"] = str(gpu_id)
    env.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")
    env.setdefault("SGLANG_LOAD_FORMAT", "dummy")
    # Hard-disable DeepGEMM bulk pre-compile.  First sl in this sweep
    # triggers runtime lazy JIT for the (M, N, K) shapes it needs;
    # subsequent sl within the same subprocess hit in-memory cache.
    env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
    # Leave the FP8 wo_a GEMM to SGLang, as a default server does: it stays on
    # SM100 (UE8M0 scales) and is turned off on SM90 unless the variable is set
    # explicitly (arg_groups/serving_hook.py:471-492, layers/deep_gemm_wrapper/
    # configurer.py:58-60) and on SM120 for DeepSeek-V4 (arg_groups/model_hook.py:
    # 485-486); citations are v0.5.23, and v0.5.21 is the same. Setting it to 1
    # for fp8_block timed the FP8 wo_a einsum on Hopper and SM120, where serving
    # dequantizes wo_a to BF16 at load (models/deepseek_v4.py:276-287).
    # A bfloat16 row drops the checkpoint's quantization_config, so its wo_a has
    # no Fp8Config and stays BF16 either way.
    env.pop("SGLANG_OPT_FP8_WO_A_GEMM", None)

    inner_shapes_arg = tuple(
        (int(prefix), tuple(int(seq_len) for seq_len in seq_lens)) for prefix, seq_lens in inner_shapes
    )
    code = (
        f'import sys; sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")\n'
        f"from collect_dsv4_attn import _subprocess_entry\n"
        f"_subprocess_entry(\n"
        f'    mode="{mode}",\n'
        f'    attn_kind="{attn_kind}",\n'
        f'    model_path="{model_path}",\n'
        f'    kv_cache_dtype="{kv_cache_dtype_sglang}",\n'
        f"    batch_size={batch_size},\n"
        f'    output_path="{output_path}",\n'
        f'    gemm_type="{gemm_type}",\n'
        f"    tp_size={tp_size!r},\n"
        f"    inner_shapes={inner_shapes_arg!r},\n"
        f")\n"
    )

    # Persist subprocess output to a per-task log so we can inspect failures
    # even when the child dies before stdout is streamed (e.g. OOM kill).
    log_dir = os.path.join(tempfile.gettempdir(), "dsv4_subproc_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{attn_kind}_{mode}_bs{batch_size}_tp{tp_size}_{gemm_type}_gpu{gpu_id}.log")

    def _run_once(nccl_port: int) -> tuple[int, str]:
        attempt_env = env.copy()
        attempt_env["AIC_DSV4_NCCL_PORT"] = str(nccl_port)
        with open(log_path, "wb") as logf:
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                env=attempt_env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            # Up to 1 hour per (kind, tp, gemm, bs). The child reports no liveness of its
            # own, so beat while waiting: the executor's stall watchdog
            # (STALL_TIMEOUT_SEC, 1800 s default) would otherwise fail a healthy sweep.
            deadline = time.monotonic() + 3600
            while True:
                try:
                    proc.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        proc.kill()
                        proc.wait()
                        break
                    worker_heartbeat()

        try:
            with open(log_path, encoding="utf-8", errors="replace") as logf:
                log_text = logf.read()
        except OSError:
            log_text = ""

        # Echo the log so it shows up in the parent's collector log.
        if log_text:
            print(log_text)
        return proc.returncode, log_text

    max_attempts = _DSV4_PORT_RETRIES
    for attempt in range(max_attempts):
        nccl_port = _nccl_port_for_attempt(gpu_id, attempt)
        returncode, log_text = _run_once(nccl_port)

        if returncode == 0:
            return

        is_port_race = "EADDRINUSE" in log_text or "address already in use" in log_text
        if is_port_race and attempt + 1 < max_attempts:
            print(
                f"[dsv4-collector] retrying after NCCL/TCPStore port collision "
                f"on nccl_port={nccl_port} ({attempt + 1}/{max_attempts}); log: {log_path}"
            )
            continue

        # Keep the kernel-limit classification of points the device refused.
        error = kernel_limit_error(log_text.splitlines())
        if error is not None:
            raise RuntimeError(error)
        raise RuntimeError(
            f"dsv4_{attn_kind}_{mode} subprocess failed for "
            f"(bs={batch_size}, tp={tp_size}, gemm={gemm_type}); exit={returncode}; log: {log_path}"
        )


def _subprocess_entry(
    *,
    mode: str,
    attn_kind: str,
    model_path: str,
    kv_cache_dtype: str,
    batch_size: int,
    output_path: str,
    inner_shapes: Iterable[tuple[int, Iterable[int]]],
    gemm_type: str = "bfloat16",
    tp_size: int = 1,
):
    """In-subprocess runner: build the model once for the batch and measure the row's points.

    SGLang derives its full/SWA/DSV4 auxiliary pools; the live allocator capacities
    and page size then refuse the points beyond them.
    """
    run_dsv4_mla_module(
        model_path=model_path,
        mode=mode,
        attn_kind=attn_kind,
        batch_size=batch_size,
        seq_lens_by_prefix={int(prefix): [int(seq_len) for seq_len in seq_lens] for prefix, seq_lens in inner_shapes},
        kv_cache_dtype=kv_cache_dtype,
        device="cuda:0",
        output_path=output_path,
        perf_filename_prefix="dsv4",
        gemm_type=gemm_type,
        tp_size=tp_size,
    )


def run_dsv4_attn_worker(
    seq_len: int,
    batch_size: int,
    tp_size: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_kind: str,
    attention_backend: str | None = None,
    inner_shapes: Iterable[tuple[int, Iterable[int]]] | None = None,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """collect.py-compatible worker — runs ONE (kind, tp, gemm, bs) row of the op plan.

    The tenth element is the row's points as ``(prefix, sequence_lengths)`` groups
    (op_plans._sglang_dsv4_rows); the subprocess measures them, or refuses one with
    its reason, and never re-derives the grid. ``perf_filename`` is bound by
    collect.py via OpEntry, not in the tuple.

    A row with ``tp_size`` > 1 is refused before the subprocess starts
    (``_refuse_tp_emulation``): this process cannot build one rank of a TP=N module.
    """
    del seq_len, attention_backend  # the row's points replace seq_len; SGLang picks the backend.

    if attn_kind not in ATTN_KINDS:
        raise ValueError(f"unknown attn_kind={attn_kind}; expected one of {ATTN_KINDS}")
    if tp_size not in _TP_SIZES:
        raise ValueError(f"unsupported tp_size={tp_size}; expected one of {_TP_SIZES}")

    mode = "context" if "context" in perf_filename else "generation"
    if not inner_shapes:
        raise RuntimeError("DSV4 worker requires the plan row's (prefix, sequence_lengths) points")
    inner_shapes = tuple(
        (int(prefix), tuple(int(seq_len) for seq_len in seq_lens)) for prefix, seq_lens in inner_shapes
    )

    device_str = str(device)
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    print(
        f"[dsv4-flash {mode}] kind={attn_kind} tp={tp_size} gemm={gemm_type} "
        f"bs={batch_size} prefix_lens={[prefix for prefix, _ in inner_shapes]} GPU={gpu_id}"
    )

    output_path = os.path.dirname(perf_filename) or os.getcwd()
    _run_subprocess(
        mode=mode,
        attn_kind=attn_kind,
        model_path=model_path,
        kv_cache_dtype_sglang=_kv_dtype_db_to_sglang(kv_cache_dtype),
        batch_size=batch_size,
        output_path=output_path,
        gpu_id=gpu_id,
        inner_shapes=inner_shapes,
        gemm_type=gemm_type,
        tp_size=tp_size,
    )


# ═══════════════════════════════════════════════════════════════════════
# CLI (manual / smoke test)
# ═══════════════════════════════════════════════════════════════════════


def _parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect DeepSeek-V4 HCA/CSA attention-module latency on SGLang: the op plan's rows."
    )
    parser.add_argument(
        "--model-path", default=None, help="Run this checkpoint's rows (COLLECTOR_MODEL_PATH); default: the plan's."
    )
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument(
        "--attn-kind",
        choices=ATTN_KINDS,
        default=None,
        help="If unset, runs csa and hca in turn.",
    )
    parser.add_argument("--batch-sizes", default=None, help="Comma-separated batch sizes; default: every row's.")
    parser.add_argument("--tp-sizes", default=None, help="Comma-separated TP sizes; default: every row's.")
    parser.add_argument("--gemm-type", choices=["bfloat16", "fp8_block"], default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.model_path:
        os.environ["COLLECTOR_MODEL_PATH"] = args.model_path
    batch_sizes = set(_parse_int_list(args.batch_sizes)) if args.batch_sizes else None
    tp_sizes = set(_parse_int_list(args.tp_sizes)) if args.tp_sizes else None
    device_str = str(args.device)
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
    output_path = args.output_path or os.getcwd()
    # The plan's rows, so a CLI reproduction runs the exact planned points.
    for kind in [args.attn_kind] if args.attn_kind else list(ATTN_KINDS):
        rows = plan(f"dsv4_{kind}_{args.mode}_module", "sglang", get_sm_version()).rows
        for _0, batch, tp_size, kv_dtype, _compute, gemm, model, _kind, _backend, points in rows:
            if batch_sizes is not None and batch not in batch_sizes:
                continue
            if tp_sizes is not None and tp_size not in tp_sizes:
                continue
            if args.gemm_type is not None and gemm != args.gemm_type:
                continue
            try:
                _run_subprocess(
                    mode=args.mode,
                    attn_kind=kind,
                    model_path=model,
                    kv_cache_dtype_sglang=_kv_dtype_db_to_sglang(kv_dtype),
                    batch_size=batch,
                    output_path=output_path,
                    gpu_id=gpu_id,
                    inner_shapes=points,
                    gemm_type=gemm,
                    tp_size=tp_size,
                )
            except Exception:
                traceback.print_exc()
                print(f"[dsv4-flash] FAILED kind={kind} tp={tp_size} gemm={gemm} bs={batch}; continuing")


if __name__ == "__main__":
    main()
