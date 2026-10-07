# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contract and timing for the GLM-5.3-Flash KDA state checkpoint copy.

Backend collectors (``collector/sglang/collect_glm53_mamba_state_copy.py`` and
``collector/vllm/collect_glm53_mamba_state_copy.py``) own the framework calls;
this module owns what both must agree on:

- the persisted row contract of ``glm53_mamba_state_copy_perf`` (column names
  and order, op name, per-variant ``kernel_source`` and ``graph_mode``);
- the audited-release gate (any other release raises
  ``MambaStateCopyRuntimeNotAuditedError``, a classified failure);
- the measurement method:

  * every timed replay is preceded, OUTSIDE the timed region, by an L2 flush
    (a write over a buffer of ``max(4 x L2, 256 MiB)``) so neither source nor
    destination state starts L2-resident: the large copies (up to ~9 GiB of
    state at TP1 B32) are far beyond one L2 anyway, but the smallest ones
    (34 x 0.5 MiB at TP8 B1) would otherwise be timed from cache;
  * device time (``median_device_ms``): CUDA events bracket the measured
    launch. A ``torch.cuda._sleep`` spin is enqueued after the flush and
    before the start event, so the stream is still busy when the host
    enqueues the measured launch and the events bracket device execution
    only (no host launch latency inside the interval);
  * host wall time (``median_host_ms``): ``time.perf_counter`` between a
    ``torch.cuda.synchronize`` before the sequence and one after it, so the
    value includes every launch, D2H/H2D copy and sync the sequence performs;
  * both report the median of ``TIMED_REPLAYS`` (30) individually timed
    replays after ``WARMUP_REPLAYS`` (5) untimed ones.

Importable without torch: torch and the framework are imported by the
backend run functions only.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

try:
    from collector.version_resolver import _check_compat
except ModuleNotFoundError:  # collect.py's sys.path fallback (collector dir on path)
    from version_resolver import _check_compat

OP_NAME = "glm53_mamba_state_checkpoint_copy"

# Persisted row contract (log_perf item keys, in this order). The base columns
# framework/version/device/op_name/kernel_source are written by log_perf.
ROW_COLUMNS = (
    "variant",
    "tp_size",
    "num_layers",
    "num_heads",
    "head_dim",
    "conv_dim",
    "conv_width",
    "ssm_bytes_per_req_layer",
    "conv_bytes_per_req_layer",
    "batch_size",
    "num_copy_requests",
    "graph_mode",
    "latency",
    "gpu_time",
)

# variant -> (framework key, kernel_source, graph_mode)
VARIANTS = {
    "sglang_decode": ("sglang", "track_mamba_states_all_layers_kernel", "cuda_graph"),
    "sglang_prefill": ("sglang", "index_gather_put_per_layer", "eager"),
    "vllm_precopy": ("vllm", "precopy_mamba_align_fused_kernel", "eager"),
}

# Releases whose serving source was audited for these call sites (file:line
# citations in the backend modules). GLM-5.3-Flash serves on stock vLLM
# 0.31.0 (no overlay); 0.30.0 (formerly audited with the
# 0.30.0+glm53tail.eb4704514fdf overlay) is no longer a GLM runtime and is
# not admitted. Local build metadata is ignored by _check_compat.
AUDITED_RUNTIMES = {
    "sglang": "sglang==0.5.20",
    "vllm": "vllm==0.31.0",
}

WARMUP_REPLAYS = 5
TIMED_REPLAYS = 30
# ~25 us of device spin at ~2 GHz; keeps the stream busy across the host
# enqueue of the measured launch (see module docstring).
LAUNCH_CUSHION_CYCLES = 50_000
MIN_FLUSH_BYTES = 256 << 20


class MambaStateCopyRuntimeNotAuditedError(RuntimeError):
    """The installed framework release is not audited for this op's call sites."""


def require_audited_runtime(framework: str, runtime_version: str) -> None:
    compat = AUDITED_RUNTIMES[framework]
    if not _check_compat(compat, runtime_version):
        raise MambaStateCopyRuntimeNotAuditedError(
            f"{framework} {runtime_version} is not an audited runtime for {OP_NAME} (audited: {compat})"
        )


def validate_case(
    framework: str,
    variant: str,
    tp_size: int,
    num_heads: int,
    batch_size: int,
    num_copy_requests: int,
) -> None:
    if variant not in VARIANTS or VARIANTS[variant][0] != framework:
        raise ValueError(f"{OP_NAME}: variant {variant!r} is not a {framework} variant")
    if tp_size < 1 or num_heads % tp_size:
        raise ValueError(f"{OP_NAME}: {num_heads} heads cannot shard over tp={tp_size}")
    if batch_size < 1 or not 0 <= num_copy_requests <= batch_size:
        raise ValueError(f"{OP_NAME}: need 0 <= num_copy_requests ({num_copy_requests}) <= batch_size ({batch_size})")


def build_row(
    *,
    variant: str,
    tp_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    conv_dim: int,
    conv_width: int,
    ssm_bytes_per_req_layer: int,
    conv_bytes_per_req_layer: int,
    batch_size: int,
    num_copy_requests: int,
    latency: float,
    gpu_time: float,
) -> dict:
    row = {
        "variant": variant,
        "tp_size": int(tp_size),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "conv_dim": int(conv_dim),
        "conv_width": int(conv_width),
        "ssm_bytes_per_req_layer": int(ssm_bytes_per_req_layer),
        "conv_bytes_per_req_layer": int(conv_bytes_per_req_layer),
        "batch_size": int(batch_size),
        "num_copy_requests": int(num_copy_requests),
        "graph_mode": VARIANTS[variant][2],
        "latency": float(latency),
        "gpu_time": float(gpu_time),
    }
    assert tuple(row) == ROW_COLUMNS
    return row


def check_local_geometry(
    *,
    num_heads: int,
    tp_size: int,
    head_dim: int,
    conv_kernel_size: int,
    local_heads: int,
    local_head_dim: int,
    conv_width: int,
    conv_dim: int,
) -> None:
    """Fail loudly when the framework-built state disagrees with the case row."""
    expected = (num_heads // tp_size, head_dim, conv_kernel_size - 1, 3 * num_heads * head_dim // tp_size)
    observed = (local_heads, local_head_dim, conv_width, conv_dim)
    if expected != observed:
        raise RuntimeError(
            f"{OP_NAME}: framework state geometry (heads, head_dim, conv_width, conv_dim)={observed} "
            f"!= case geometry {expected}"
        )


def persist_row(row: dict, *, framework_label: str, version: str, device_name: str, perf_filename: str, log_perf):
    variant = row["variant"]
    if not log_perf(
        item_list=[row],
        framework=framework_label,
        version=version,
        device_name=device_name,
        op_name=OP_NAME,
        kernel_source=VARIANTS[variant][1],
        perf_filename=perf_filename,
    ):
        raise RuntimeError(f"failed to persist {OP_NAME} row to {perf_filename}")


class L2Flusher:
    """Evicts L2 by writing a buffer several times the L2 size."""

    def __init__(self, torch, device):
        props = torch.cuda.get_device_properties(device)
        l2_bytes = int(getattr(props, "L2_cache_size", 0) or 0)
        if l2_bytes <= 0:
            raise RuntimeError(f"{OP_NAME}: cannot read the L2 size of {props.name}; refusing to time without a flush")
        self.nbytes = max(4 * l2_bytes, MIN_FLUSH_BYTES)
        self._buf = torch.empty(self.nbytes // 4, dtype=torch.int32, device=device)
        self._value = 0

    def flush(self) -> None:
        self._value ^= 1
        self._buf.fill_(self._value)


def median_device_ms(
    torch,
    launch: Callable[[], None],
    flusher: L2Flusher,
    *,
    warmup: int = WARMUP_REPLAYS,
    reps: int = TIMED_REPLAYS,
) -> float:
    """Median device time (ms) of ``launch`` over individually timed replays."""
    for _ in range(warmup):
        flusher.flush()
        launch()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for start, end in zip(starts, ends, strict=True):
        flusher.flush()
        torch.cuda._sleep(LAUNCH_CUSHION_CYCLES)
        start.record()
        launch()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))


def median_host_ms(
    torch,
    sequence: Callable[[], None],
    flusher: L2Flusher,
    *,
    warmup: int = WARMUP_REPLAYS,
    reps: int = TIMED_REPLAYS,
) -> float:
    """Median host wall time (ms) of ``sequence`` between device synchronizations."""
    samples = []
    for index in range(warmup + reps):
        flusher.flush()
        torch.cuda.synchronize()
        start = time.perf_counter()
        sequence()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1e3
        if index >= warmup:
            samples.append(elapsed)
    return statistics.median(samples)
