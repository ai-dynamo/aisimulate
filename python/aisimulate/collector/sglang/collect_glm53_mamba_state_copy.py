# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash prefix-cache KDA state checkpoint copy — SGLang 0.5.20.

With the radix cache on, SGLang serves GLM-5.3-Flash with MambaRadixCache
``extra_buffer`` (``mamba_radix_cache_strategy='extra_buffer'``, page 64,
``mamba_track_interval`` 256): the KDA conv/recurrent state of a request is
snapshotted into a ping-pong "track" slot of the mamba pool. Two serving
paths do this, one row variant each (op ``glm53_mamba_state_checkpoint_copy``,
table ``glm53_mamba_state_copy_perf``):

``sglang_decode`` (kernel_source ``track_mamba_states_all_layers_kernel``,
graph_mode ``cuda_graph``)
    At the last KDA layer of a decode step, ``_track_mamba_state_decode``
    (srt/layers/attention/hybrid_linear_attn_backend.py:867-909) launches ONE
    ``track_mamba_states_all_layers`` (kernels/ops/mamba/
    mamba_state_scatter_triton.py:859-895, kernel :801-857) over the full
    ``[34, slots, ...]`` conv/temporal pools with grid ``34 x bs``, inside the
    captured decode graph. Rows whose ``mamba_track_mask`` is False (seq_len
    not on the 256 grid, schedule_batch.py:3496-3516) return at once; the
    first ``num_copy_requests`` rows here have the mask set and copy every
    layer's conv + recurrent state from their working slot to their track
    slot. The collector calls the stock function, captures it in a CUDA graph
    exactly as the decode graph holds it, and times graph replays
    (``latency == gpu_time``).

``sglang_prefill`` (kernel_source ``index_gather_put_per_layer``,
graph_mode ``eager``)
    Every extend request with ``extend_range.length >= checkpoint_grid``
    (lcm(mamba_cache_chunk_size 64, page 64) = 64; schedule_batch.py:
    2922-2935) is tracked. Per forward the backend then runs, eagerly:
    ``bool(mamba_track_mask.any())`` (hybrid_linear_attn_backend.py:149-152,
    a D2H sync), ``_init_track_conv_indices`` (:349-372) and
    ``_init_track_ssm_indices`` (:374-455; .cpu() syncs, CPU masking, H2D
    copies), and KDA's ``init_forward_metadata`` mask ``nonzero`` + gather
    (linear/kda_backend.py:533-543); then per KDA layer (x34) the conv
    window snapshot (kda_backend.py:831-840) and ``_track_mamba_state_extend``
    (hybrid_linear_attn_backend.py:911-954; GLM's 128-token extends are
    chunk-aligned, so it takes the ``ssm_states[final_dst] =
    ssm_states[final_src]`` branch at :951-954). The stock methods run on a
    synthetic backend/forward batch (fields and citations below); the two
    inline statements are replicated verbatim with citations. ``latency`` is
    the host wall time of the whole per-forward sequence (launch/sync bound,
    syncs included); ``gpu_time`` is the device time of the 34-layer copy
    statements alone, captured into a CUDA graph and replayed. A batch with no
    tracked request (``num_copy_requests == 0``) runs only the ``any()`` sync
    in serving (every other statement is gated on ``has_mamba_track_mask``),
    so its ``gpu_time`` is 0.

Request geometry: tracked requests extend 128 tokens from prefix 0 (the
aligned final-state path); untracked requests in a mixed batch extend 32
tokens (< 64, so serving would not track them) from prefix 0.

State pools are built from SGLang's own GLM shape/dtype calculators
(configs/glm5_next.py:264-272 -> configs/mamba_utils.py:325-366
KimiLinearStateShape.create; default dtypes mamba2_state_dtype(None) ->
conv bf16 / temporal fp32, mamba_utils.py:47-107, GLM's config has no
mamba_ssm_dtype) in the default per-layer layout of mem_cache/
memory_pool.py:597-628 (``[num_layers, size + 1, *shape]``; page-major and
unified pools are opt-in, arg_groups/fields/memory.py:72-89). Timing method:
see collector/glm53_mamba_state_copy_common.py; in addition the worker binds
itself to its GPU's NUMA node exactly when SGLang serving would
(_bind_numa_like_serving), because the prefill latency is host bound.
"""

# Audited serving source: lmsysorg/sglang v0.5.20 (tag commit 94602c9c,
# GB300 arm64 image b0d8718a). Any other release raises
# MambaStateCopyRuntimeNotAuditedError inside the run function as well.
__compat__ = "sglang==0.5.20"

import gc
import os

try:
    from collector.case_generator import get_common_glm53_mamba_state_copy_test_cases
    from collector.glm53_mamba_state_copy_common import (
        L2Flusher,
        build_row,
        check_local_geometry,
        median_device_ms,
        median_host_ms,
        persist_row,
        require_audited_runtime,
        validate_case,
    )
    from collector.helper import log_perf
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import get_common_glm53_mamba_state_copy_test_cases
    from glm53_mamba_state_copy_common import (
        L2Flusher,
        build_row,
        check_local_geometry,
        median_device_ms,
        median_host_ms,
        persist_row,
        require_audited_runtime,
        validate_case,
    )

    from helper import log_perf

SGLANG_VARIANTS = ("sglang_decode", "sglang_prefill")

# Extend lengths of the synthetic prefill batch (see module docstring).
TRACKED_EXTEND_LEN = 128
UNTRACKED_EXTEND_LEN = 32
# mamba_cache_chunk_size() = max(model mamba_chunk_size, page_size)
# (arg_groups/overrides.py:1896-1923): GLM's text config has no
# mamba_chunk_size, so the FLA CHUNK_SIZE 64 (kernels/ops/attention/fla/
# chunk_delta_h.py:23) applies, and the serving page size is 64.
MAMBA_CACHE_CHUNK_SIZE = 64


def get_glm53_mamba_state_copy_test_cases():
    """[variant, tp_size, num_layers, num_heads, head_dim, conv_kernel_size,
    batch_size, num_copy_requests, model_name] per SGLang variant."""
    common = get_common_glm53_mamba_state_copy_test_cases()
    return [
        [
            variant,
            case.tp_size,
            case.num_layers,
            case.num_heads,
            case.head_dim,
            case.conv_kernel_size,
            case.batch_size,
            case.num_copy_requests,
            case.model_name,
        ]
        for variant in SGLANG_VARIANTS
        for case in common
    ]


class _ExtendTrackBatch:
    """The ForwardBatch fields the extend track path reads.

    Population sites (sglang 0.5.20):
    - ``mamba_track_mask`` bool / ``mamba_track_indices`` int64 /
      ``mamba_track_seqlens`` int64, device tensors: schedule_batch.py:2862-2879
      (per-request values from _mamba_radix_cache_v2_req_prepare_for_extend,
      :2894-2990: mask = extend length >= 64, seqlen = prefix + extend length
      when tracked else -1, index = the request's ping-pong track slot).
    - ``extend_seq_lens`` / ``extend_prefix_lens`` int32 device tensors:
      forward_batch_info.py:938-943.
    """

    def __init__(self, torch, device, extend_lens, track_mask, track_indices):
        self.mamba_track_mask = torch.tensor(track_mask, dtype=torch.bool, device=device)
        self.mamba_track_indices = torch.tensor(track_indices, dtype=torch.int64, device=device)
        self.mamba_track_seqlens = torch.tensor(
            [length if tracked else -1 for length, tracked in zip(extend_lens, track_mask, strict=True)],
            dtype=torch.int64,
            device=device,
        )
        self.extend_seq_lens = torch.tensor(extend_lens, dtype=torch.int32, device=device)
        self.extend_prefix_lens = torch.zeros(len(extend_lens), dtype=torch.int32, device=device)

    def mamba_track_aligned_lens(self):
        # Verbatim body of ForwardBatch.mamba_track_aligned_lens
        # (model_executor/forward_batch_info.py:1079-1088) with
        # mamba_cache_chunk_size() resolved to MAMBA_CACHE_CHUNK_SIZE.
        if self.mamba_track_mask is None or self.mamba_track_seqlens is None or self.extend_prefix_lens is None:
            return None
        chunk_size = MAMBA_CACHE_CHUNK_SIZE
        lens_to_track = self.mamba_track_seqlens - self.extend_prefix_lens
        return (lens_to_track // chunk_size) * chunk_size


_NUMA_BINDING: dict[int, object] = {}


def _bind_numa_like_serving(torch, device) -> object:
    """Bind this worker to its GPU's NUMA node exactly when serving would.

    SGLang 0.5.20 auto-binds every scheduler process to the NUMA node of its
    GPU: by default it launches the scheduler subprocess under numactl
    (srt/utils/numa_utils.py:28-77, SGLANG_NUMA_BIND_V2 / SGLANG_AUTO_NUMA_BIND
    default True, srt/environ.py:1442-1443); the V1 path binds in-process
    with numa_bind_to_node (srt/managers/scheduler.py:5782-5785). Both take
    the node from the stock get_numa_node_if_available (numa_utils.py:125-152;
    no --numa-node override here), which returns None when serving would not
    bind (no NUMA, no numactl, no permission). The collector applies the
    in-process binding (numa_utils.py:169-188) to the same node. The sglang_prefill latency is host bound
    (launches + D2H syncs), so an unbound thread migrating across Grace
    sockets would time a placement serving never runs. Returns the bound
    node, or None when serving would not bind either (no NUMA / numactl /
    permission).
    """
    from types import SimpleNamespace

    from sglang.srt.utils.numa_utils import get_numa_node_if_available, numa_bind_to_node

    index = torch.cuda.current_device()
    if index not in _NUMA_BINDING:
        node = get_numa_node_if_available(SimpleNamespace(numa_node=None), index)
        if node is not None:
            numa_bind_to_node(node)
        _NUMA_BINDING[index] = node
        print(f"glm53_mamba_state_checkpoint_copy: cuda:{index} NUMA binding -> {node}", flush=True)
    return _NUMA_BINDING[index]


def _state_pools(torch, device, tp_size, num_layers, num_heads, head_dim, conv_kernel_size, num_slots):
    from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape

    shape = KimiLinearStateShape.create(
        tp_world_size=tp_size,
        num_heads=num_heads,
        head_dim=head_dim,
        conv_kernel_size=conv_kernel_size,
    )
    params = KimiLinearCacheParams(shape=shape, layers=list(range(num_layers)))
    (conv_shape,) = shape.conv
    # memory_pool.py:597-628: per-layer [num_layers, size + 1, *shape] zeros.
    conv_pool = torch.zeros((num_layers, num_slots, *conv_shape), dtype=params.dtype.conv, device=device)
    ssm_pool = torch.zeros((num_layers, num_slots, *shape.temporal), dtype=params.dtype.temporal, device=device)
    return conv_pool, ssm_pool


def _geometry(conv_pool, ssm_pool):
    conv_width, conv_dim = conv_pool.shape[-2:]
    local_heads, local_head_dim = ssm_pool.shape[-3], ssm_pool.shape[-2]
    return {
        "local_heads": int(local_heads),
        "local_head_dim": int(local_head_dim),
        "conv_width": int(conv_width),
        "conv_dim": int(conv_dim),
        "ssm_bytes": int(ssm_pool[0, 0].numel() * ssm_pool.element_size()),
        "conv_bytes": int(conv_pool[0, 0].numel() * conv_pool.element_size()),
    }


def _time_decode(torch, device, conv_pool, ssm_pool, batch_size, num_copy_requests, flusher):
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import track_mamba_states_all_layers

    # Working slot of request i = 1 + i, its track slot = 1 + batch_size + i.
    # cache_indices: the decode graph's static int32 state_indices_list
    # buffer (hybrid_linear_attn_backend.py:518-521); track destinations: the
    # static int64 mamba_track_indices_buf (:514-516); mask: bool, True on the
    # rows whose seq_len is on the track grid (schedule_batch.py:3496-3516).
    cache_indices = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
    track_indices = torch.arange(batch_size + 1, 2 * batch_size + 1, dtype=torch.int64, device=device)
    track_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    track_mask[:num_copy_requests] = True

    def launch_track():
        # hybrid_linear_attn_backend.py:891-899; check_freed_slots =
        # enable_unified_memory, default False (arg_groups/fields/memory.py:80-89).
        track_mamba_states_all_layers(
            conv_pool,
            ssm_pool,
            cache_indices,
            track_mask,
            track_indices,
            batch_size,
            check_freed_slots=False,
        )

    # JIT-compile outside the capture, then capture as the decode graph holds it.
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        launch_track()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_track()
    try:
        gpu_ms = median_device_ms(torch, graph.replay, flusher)
    finally:
        del graph
    return gpu_ms, gpu_ms


def _prefill_backend(torch, device, conv_pool):
    from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend

    # Only the attributes the extend-track methods read are set (the full
    # __init__ needs a ModelRunner):
    backend = object.__new__(KDAAttnBackend)
    # MambaAttnBackendBase.__init__ (hybrid_linear_attn_backend.py:66-75):
    # device = model_runner.device; _mamba_chunk_size feeds the
    # mamba_chunk_size property (:107-113), whose default 64 applies to GLM
    # (no mamba_chunk_size in its text config).
    backend.device = device
    backend._mamba_chunk_size = 64
    # kda_backend.py:403-414: the KDA pool stores conv as [kernel-1, dim], so
    # the backend exposes the transposed pool shape; [-1] is the conv window.
    backend.conv_states_shape = conv_pool.transpose(-1, -2).shape
    return backend


def _time_prefill(torch, device, conv_pool, ssm_pool, batch_size, num_copy_requests, flusher):
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata

    backend = _prefill_backend(torch, device, conv_pool)
    num_layers = conv_pool.shape[0]
    tracked = [i < num_copy_requests for i in range(batch_size)]
    extend_lens = [TRACKED_EXTEND_LEN if t else UNTRACKED_EXTEND_LEN for t in tracked]
    # Working slot of request i = 1 + i, its track slot = 1 + batch_size + i.
    batch = _ExtendTrackBatch(
        torch,
        device,
        extend_lens,
        tracked,
        [batch_size + 1 + i for i in range(batch_size)],
    )
    # Baseline per-forward metadata built whether or not tracking is on, so
    # it stays outside the timed region: mamba_cache_indices is the int32
    # req_index_to_mamba_index_mapping gather (memory_pool.py:1370-1372,1479-1480)
    # and query_start_loc the int32 extend prefix sum
    # (hybrid_linear_attn_backend.py:253-260, extend_start_loc from
    # forward_batch_info.py:1938-1940).
    mamba_cache_indices = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
    query_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.cumsum(batch.extend_seq_lens, dim=0)
    total_tokens = sum(extend_lens)
    # Per-layer pre-conv mixed_qkv [tokens, conv_dim] (kda_backend.py:831-840
    # indexes its rows); every layer reads the same synthetic activations.
    mixed_qkv = torch.randn(total_tokens, conv_pool.shape[-1], dtype=conv_pool.dtype, device=device)

    state = {}

    def per_forward_metadata():
        # hybrid_linear_attn_backend.py:149-152
        has_mamba_track_mask = bool(batch.mamba_track_mask is not None and batch.mamba_track_mask.any())
        metadata = ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            mamba_track_indices=batch.mamba_track_indices,
            has_mamba_track_mask=has_mamba_track_mask,
        )
        if has_mamba_track_mask:
            # hybrid_linear_attn_backend.py:261-276
            metadata.track_conv_indices = backend._init_track_conv_indices(query_start_loc, batch)
            (
                metadata.track_chunk_idx,
                metadata.track_ssm_h_src,
                metadata.track_ssm_h_dst,
                metadata.track_ssm_h_batch_src,
                metadata.track_ssm_final_src,
                metadata.track_ssm_final_dst,
                metadata.track_ssm_seq_idx,
                metadata.track_ssm_end_locs,
                metadata.track_ssm_recompute_dst,
            ) = backend._init_track_ssm_indices(mamba_cache_indices, batch)
            # kda_backend.py:536-543 (KDAAttnBackend.init_forward_metadata)
            metadata.mamba_track_mask_indices = batch.mamba_track_mask.nonzero(as_tuple=True)[0]
            metadata.conv_states_mask_indices = batch.mamba_track_indices[metadata.mamba_track_mask_indices]
        state["metadata"] = metadata
        return metadata

    def per_layer_copies(metadata):
        if not metadata.has_mamba_track_mask:
            return
        for layer in range(num_layers):
            # kda_backend.py:831-839 (conv window snapshot; the KDA pool rows
            # take raw [tokens, dim] pre-conv input directly)
            conv_pool[layer][metadata.conv_states_mask_indices] = mixed_qkv[metadata.track_conv_indices]
            # kda_backend.py:923-931 -> hybrid_linear_attn_backend.py:911-954.
            # h is unused on the aligned path; h_track_buf stays None because
            # track_ssm_h_src is empty for aligned extends (kda_backend.py:864-876).
            backend._track_mamba_state_extend(batch, None, ssm_pool[layer], metadata)

    def sequence():
        per_layer_copies(per_forward_metadata())

    latency_ms = median_host_ms(torch, sequence, flusher)

    metadata = state["metadata"]
    if not metadata.has_mamba_track_mask:
        return latency_ms, 0.0
    # Device time of the 34-layer copy statements alone, with the index
    # tensors of the last measured forward.
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        per_layer_copies(metadata)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        per_layer_copies(metadata)
    try:
        gpu_ms = median_device_ms(torch, graph.replay, flusher)
    finally:
        del graph
    return latency_ms, gpu_ms


def run_glm53_mamba_state_copy(
    variant: str,
    tp_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    batch_size: int,
    num_copy_requests: int,
    model_name: str,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    from importlib.metadata import version as _get_version

    sglang_version = _get_version("sglang")
    require_audited_runtime("sglang", sglang_version)
    validate_case("sglang", variant, tp_size, num_heads, batch_size, num_copy_requests)

    import torch

    torch.cuda.set_device(device)
    _bind_numa_like_serving(torch, device)
    conv_pool = ssm_pool = flusher = None
    try:
        with torch.inference_mode():
            conv_pool, ssm_pool = _state_pools(
                torch,
                device,
                tp_size,
                num_layers,
                num_heads,
                head_dim,
                conv_kernel_size,
                num_slots=2 * batch_size + 1,
            )
            geometry = _geometry(conv_pool, ssm_pool)
            check_local_geometry(
                num_heads=num_heads,
                tp_size=tp_size,
                head_dim=head_dim,
                conv_kernel_size=conv_kernel_size,
                local_heads=geometry["local_heads"],
                local_head_dim=geometry["local_head_dim"],
                conv_width=geometry["conv_width"],
                conv_dim=geometry["conv_dim"],
            )
            flusher = L2Flusher(torch, device)
            if variant == "sglang_decode":
                latency_ms, gpu_ms = _time_decode(
                    torch, device, conv_pool, ssm_pool, batch_size, num_copy_requests, flusher
                )
            else:
                latency_ms, gpu_ms = _time_prefill(
                    torch, device, conv_pool, ssm_pool, batch_size, num_copy_requests, flusher
                )
        row = build_row(
            variant=variant,
            tp_size=tp_size,
            num_layers=num_layers,
            num_heads=geometry["local_heads"],
            head_dim=geometry["local_head_dim"],
            conv_dim=geometry["conv_dim"],
            conv_width=geometry["conv_width"],
            ssm_bytes_per_req_layer=geometry["ssm_bytes"],
            conv_bytes_per_req_layer=geometry["conv_bytes"],
            batch_size=batch_size,
            num_copy_requests=num_copy_requests,
            latency=latency_ms,
            gpu_time=gpu_ms,
        )
        persist_row(
            row,
            framework_label="SGLang",
            version=sglang_version,
            device_name=torch.cuda.get_device_name(device),
            perf_filename=perf_filename,
            log_perf=log_perf,
        )
    finally:
        del conv_pool, ssm_pool, flusher
        gc.collect()
        torch.cuda.empty_cache()
