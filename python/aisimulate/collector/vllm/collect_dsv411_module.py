# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dsv411 producer for vLLM v0.30.0 (torchrun, pure TP, dummy weights).

Measures the ``dsv411_module_perf`` components on vLLM's native DeepSeek-V4.1 modules:

* ``attention_core`` / ``indexer``: forty ``DeepseekV4FlashMLAAttention`` modules built the way
  ``DeepseekV4Model`` builds them (shared top-k/candidate buffers, aux streams), native caches
  bound through each layer's ``get_kv_cache_spec``/``bind_kv_cache`` and metadata builders
  (collector/vllm DSV4 helpers), real KV seeded by chunked prefills. The layer interval is
  ``attn.forward`` with the ``wo_b`` TP all-reduce moved after the end event. The ``indexer``
  row is the nested ``DeepseekV4Indexer.forward`` (index-query preparation, main stream) plus
  ``indexer.indexer_op`` (scoring + top-k) interval; ``attention_core`` = layer minus indexer.
  - context: eager, stream drained before every representative layer (``eager_drained``);
  - generation: serving splits every layer into CUDA-graph pieces around the attention custom
    ops (piecewise cudagraphs: ``_sparse_indexer_and_attn`` is the split point); the producer
    mirrors that exactly — one graph for the input projections / KV insertion / indexer
    preparation, the eager ``_sparse_indexer_and_attn`` region, one graph for ``_o_proj`` —
    and times the replayed sequence (``cuda_graph``).
* ``engram`` (``ParallelEngramEmbedding.lookup`` + ``Engram.forward`` with the TP all-gather
  outside), ``mhc`` (both ``mhc_shifted_post_pre`` sites of layer 2) and ``shared_linear``
  (``gate_up_proj``/``down_proj`` of layer 2): context eager, generation as one CUDA graph per
  sub-call (``runtime.GraphedCalls``).

Serving identity on sm90 at this pin: FP8 block[32,32] linears through
``ModelOptLinearMethod[MarlinMxfp8LinearKernel]`` (W8A16), ``wo_a`` BF16 emulation, KV
``fp8_ds_mla`` (584 B), indexer K fp8 (132 B), DeepGEMM fp8 index scoring. Framework
integration informed by vllm-project/vllm@ced6857a (Apache-2.0); see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from collector.dsv411 import contract
from collector.dsv411.runtime import (
    GraphedCalls,
    Intervals,
    RowStream,
    check_pins,
    completed_cases,
    device_witness,
    dispatch_label,
    failed_cases,
    finite,
    mark_case,
    new_receipt,
    prepare_private_caches,
    reduce_layer_intervals,
    source_hashes,
    write_receipt,
)
from collector.vllm.dsv41_isolated_runner import describe_module, initialize_dummy_weights

BACKEND = "vllm"
# vllm-project/vllm tag v0.30.0 (2026-09-21), image vllm/vllm-openai:v0.30.0
FRAMEWORK_COMMIT = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
FRAMEWORK_VERSION = "0.30.0"
EXPECTED_SM = {"H100": 90, "H200": 90, "H20": 90, "B200": 100, "GB200": 100}
WEIGHT_INITIALIZER = {
    "name": "vllm_dummy_loader_plus_packed_fill_v1",
    "float_low": -1e-3,
    "float_high": 1e-3,
    "e8m0_scale": 127,
    "packed_bytes": "uniform_nan_free",
}
REQUIRED_SOURCES = {
    "models/deepseek_v41/attention.py",
    "models/deepseek_v41/compressor.py",
    "models/deepseek_v41/sparse_mla.py",
    "models/deepseek_v41/quant_config.py",
    "models/deepseek_v41/common/engram.py",
    "models/deepseek_v41/nvidia/model.py",
    "models/deepseek_v41/nvidia/engram.py",
    "models/deepseek_v41/nvidia/flashmla.py",
    "models/deepseek_v41/nvidia/ops/mega_mhc.py",
    "models/deepseek_v4/nvidia/model.py",
    "models/deepseek_v4/nvidia/ops/o_proj.py",
    "model_executor/layers/sparse_attn_indexer.py",
    "model_executor/layers/fused_moe/router/gate_linear.py",
    "model_executor/layers/quantization/modelopt.py",
    "model_executor/layers/quantization/mxfp4.py",
    "model_executor/kernels/linear/mxfp8/marlin.py",
    "model_executor/kernels/mhc/tilelang.py",
    "model_executor/model_loader/weight_utils.py",
    "config/engram.py",
    "distributed/parallel_state.py",
    "v1/attention/backends/mla/sparse_swa.py",
    "v1/attention/backends/mla/indexer.py",
    "v1/attention/backends/utils.py",
    "v1/kv_cache_interface.py",
    "v1/worker/block_table.py",
    "forward_context.py",
    "compilation/breakable_cudagraph.py",
    "third_party/flashmla/flash_mla_interface.py",
}
HIDDEN = 5120
BLOCK = 64


def build_vllm_config(model_path, tp, pool):
    from vllm.config import CacheConfig, CompilationConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
    from vllm.config.engram import EngramConfig
    from vllm.config.load import LoadConfig

    max_len = pool["context_length"]
    model_config = ModelConfig(
        model=str(model_path),
        tokenizer=str(model_path),
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
        max_model_len=max_len,
    )
    cache_config = CacheConfig(
        block_size=BLOCK, cache_dtype="fp8"
    )  # sparse_mla.py: SM90 page 64; attention.py forces fp8_ds_mla
    cache_config.num_gpu_blocks, cache_config.num_cpu_blocks = 512, 0
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(tensor_parallel_size=tp, disable_custom_all_reduce=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=pool["max_requests"],
            max_num_batched_tokens=pool["max_new_tokens"],
            enable_chunked_prefill=True,
            max_model_len=max_len,
            is_encoder_decoder=False,
        ),
        load_config=LoadConfig(load_format="dummy"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
        engram_config=EngramConfig(cpu_offload=False),
    )


def layer_role(attn):
    if attn.compress_ratio == 0:
        return "swa"
    if attn.is_kv_source:
        return "full"
    return "reindex" if attn.is_index_source else "reuse"


def validate_attention_geometry(attns, manifest):
    for entry in (e for e in manifest["entries"] if e["phase"] == "context"):
        attn = attns[entry["layer"]]
        s = entry["structure"]
        if entry["component"] == "attention_core":
            actual = dict(
                role=layer_role(attn),
                compress_ratio=int(attn.compress_ratio),
                num_heads=int(attn.n_local_heads),
                head_dim=int(attn.head_dim),
                q_lora_rank=int(attn.q_lora_rank),
                o_lora_rank=int(attn.o_lora_rank),
                o_groups=int(attn.n_local_groups),
                window_size=int(attn.window_size),
            )
            if any(s[k] != v for k, v in actual.items()):
                raise RuntimeError(
                    f"native attention geometry differs from the SDK graph at layer {entry['layer']}: {actual}"
                )
        elif entry["component"] == "indexer":
            indexer = attn.indexer
            if indexer is None:
                raise RuntimeError(f"SDK graph expects an indexer at layer {entry['layer']}")
            uses = 0 <= attn.candidate_source_layer < entry["layer"] and attn.compress_ratio > 0
            actual = dict(
                compress_ratio=int(attn.compress_ratio),
                index_n_heads=int(indexer.n_head),
                index_head_dim=int(indexer.head_dim),
                index_topk=int(indexer.topk_tokens),
                is_candidate_source=int(entry["layer"] == attn.candidate_source_layer),
                candidate_limit=int(attn.candidate_topk_blocks * attn.candidate_block_size) if uses else 0,
            )
            if any(s[k] != v for k, v in actual.items()):
                raise RuntimeError(
                    f"native indexer geometry differs from the SDK graph at layer {entry['layer']}: {actual}"
                )
    for layer_id, attn in enumerate(attns):
        owns = any(e["layer"] == layer_id and e["component"] == "indexer" for e in manifest["entries"])
        if owns != (getattr(attn, "indexer", None) is not None):
            raise RuntimeError(f"indexer ownership differs from the SDK graph at layer {layer_id}")


class AttentionStack:
    """Forty native attention layers with their caches; collector-built metadata per forward."""

    def __init__(self, vllm_config, plan, device):
        import torch
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.models.deepseek_v41.nvidia.model import _select_dsv4_attn_cls

        self.torch, self.vllm_config, self.plan, self.device = torch, vllm_config, plan, device
        config = vllm_config.model_config.hf_config
        self.config = config
        if config.num_hidden_layers != 40 or config.model_type != "deepseek_v41":
            raise RuntimeError("unaltered native V4.1 config required")
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.topk_indices_buffer = torch.empty(max_tokens, config.index_topk, dtype=torch.int32, device=device)
        self.candidate_block_buffer = (
            torch.empty(max_tokens, config.candidate_topk_blocks, dtype=torch.int32, device=device)
            if getattr(config, "candidate_source_layer_id", -1) >= 0
            else None
        )
        self.aux_streams = [torch.cuda.Stream() for _ in range(3)]
        cls = _select_dsv4_attn_cls(vllm_config)
        self.attns = torch.nn.ModuleList(
            [
                cls(
                    vllm_config,
                    prefix=f"model.layers.{i}.attn",
                    topk_indices_buffer=self.topk_indices_buffer,
                    aux_stream_list=self.aux_streams,
                    candidate_block_buffer=self.candidate_block_buffer,
                )
                for i in range(40)
            ]
        )
        self.norms = torch.nn.ModuleList([RMSNorm(config.hidden_size, config.rms_norm_eps) for _ in range(40)])
        self.attention_class = cls.__name__
        generator = torch.Generator(device="cuda").manual_seed(plan["seed"])
        self.hidden_pool = torch.randn(
            max_tokens, config.hidden_size, generator=generator, dtype=torch.bfloat16, device=device
        )
        self.bound = None
        self.builders = {}

    def cache_layers(self, attn):
        layers = [attn, attn.swa_cache_layer]
        indexer = getattr(attn, "indexer", None)
        if indexer is not None and getattr(indexer, "k_cache", None) is not None:
            layers.append(indexer.k_cache)
        for owner in (
            getattr(attn, "compressor", None),
            getattr(indexer, "compressor", None) if indexer is not None else None,
        ):
            state = getattr(owner, "state_cache", None) if owner is not None else None
            if state is not None:
                layers.append(state)
        return layers

    # Sliding-window caches: serving keeps only window-sized page rings per request (the SWA block
    # pool evicts past the window), so these caches are allocated as per-request rings of
    # SWA_RING_BLOCKS pages (8192-token chunk + 128-token window + slack) and addressed modulo the ring;
    # a full-length SWA cache for all 40 layers at batch 4 x 524K tokens alone is 49 GB.
    SWA_RING_BLOCKS = 264

    @staticmethod
    def _is_swa(spec) -> bool:
        return "SlidingWindow" in type(spec).__name__

    def _ring_blocks(self, spec, ring_tokens=None):
        """Pages per request for caches serving addresses as per-request rings: the sliding-window
        caches (window pages) and the compressor state caches (CircularBufferSpec: ONE block whose
        block_size is the ring capacity; the compressor's slot kernel reads block_table[req, 0]).
        None = position-addressed cache sized by the sequence length. ``ring_tokens``: the most new
        tokens one forward of the case writes per request (seeding chunk or measured query); the SWA
        ring holds them plus the 128-token window - sizing it for the 8192-token chunk at batch 256
        (and the bound ring is what the metadata remap addresses modulo, see ``metadata``)."""
        if self._is_swa(spec):
            if ring_tokens is None:
                return self.SWA_RING_BLOCKS
            return min(self.SWA_RING_BLOCKS, -(-(ring_tokens + 128) // spec.block_size) + 1)
        if "CircularBuffer" in type(spec).__name__:
            return int(spec.max_num_blocks_per_req(self.vllm_config, self.vllm_config.model_config.max_model_len))
        return None

    def _allocate(self, backend, spec, num_blocks):
        """The DSV4 helper's layout (compute_layer_kv_cache_shape_bytes + padded page stride) with the
        stride probe on the meta device instead of a full-size scratch tensor."""
        import torch
        from vllm.v1.kv_cache_interface import compute_layer_kv_cache_shape_bytes

        shape_bytes = compute_layer_kv_cache_shape_bytes(spec, num_blocks)
        dtype_size = torch.empty((), dtype=spec.dtype).element_size()
        shape = (*shape_bytes[:-1], shape_bytes[-1] // dtype_size)
        if spec.page_size_padded is None:
            return torch.zeros(shape, dtype=spec.dtype, device=self.device)
        strides = list(torch.empty(shape, device="meta").stride())
        strides[0] = spec.page_size_bytes // dtype_size
        return torch.empty_strided(shape, tuple(strides), dtype=spec.dtype, device=self.device).zero_()

    def bind_caches(self, batch, seq_len, ring_tokens):
        import torch

        from collector.vllm.collect_dsv4_attn import _cache_blocks_for_block_size

        static_ctx = self.vllm_config.compilation_config.static_forward_context
        layers = [(layer, static_ctx[layer.prefix]) for attn in self.attns for layer in self.cache_layers(attn)]
        specs = {layer.prefix: registered.get_kv_cache_spec(self.vllm_config) for layer, registered in layers}
        # release the previous case's caches before allocating (peak = one case, not two)
        for layer, registered in layers:
            spec = specs[layer.prefix]
            if spec is not None:
                registered.bind_kv_cache(torch.zeros((1, 1, 1, 1), dtype=spec.dtype, device=self.device))
        gc.collect()
        torch.cuda.empty_cache()
        bound = {}
        for layer, registered in layers:
            spec = specs[layer.prefix]
            if spec is None:
                continue
            ring = self._ring_blocks(spec, ring_tokens)
            if ring:
                blocks = batch * ring + 64
            else:
                blocks = _cache_blocks_for_block_size(batch, seq_len, spec.block_size)
            registered.bind_kv_cache(self._allocate(registered.get_attn_backend(), spec, blocks))
            bound[layer.prefix] = dict(
                spec=type(spec).__name__,
                block_size=spec.block_size,
                blocks=blocks,
                ring=ring,
                backend=registered.get_attn_backend().get_name(),
            )
        self.bound = bound
        return bound

    def _remapped(self, common, block_size, ring_blocks=None):
        """Common metadata for a cache with another page size (SWA pages are 32, compressor states
        ``compress_ratio`` tokens): the DSV4 helper rebuilds the arange block table but assumes the
        tokens are the FIRST ``num_tokens`` positions; chunked seeding and cached prefills write at
        ``common.positions``, so the slot mapping is recomputed from those positions."""
        import torch

        from collector.vllm.collect_dsv4_attn import _remap_common_metadata

        remapped = _remap_common_metadata(common, block_size=block_size, device=str(self.device))
        positions = common.positions
        query_lens = torch.diff(common.query_start_loc)
        reqs = torch.repeat_interleave(torch.arange(query_lens.numel(), device=positions.device), query_lens)
        table = remapped.block_table_tensor
        if ring_blocks:
            # per-request page ring: logical block b of request r lives in physical block r*R + b % R
            width = table.shape[1]
            logical = torch.arange(width, device=table.device, dtype=table.dtype)
            base = (torch.arange(table.shape[0], device=table.device, dtype=table.dtype) * ring_blocks)[:, None]
            table = base + (logical % ring_blocks)[None, :]
            remapped.block_table_tensor = table.contiguous()
        remapped.slot_mapping = table[reqs, positions // block_size].long() * block_size + positions % block_size
        remapped.positions = positions
        return remapped

    def _builder(self, registered, spec, prefix, sub):
        """One metadata builder per (backend, spec kind, page size), shared by every layer of that
        kind exactly like serving's one builder per KV-cache group: the V4.1 indexer builder owns a
        16 GiB expanded block-table buffer, so a builder per layer per forward exhausted the GPU."""
        from collector.vllm.collect_dsv4_attn import _make_builder

        backend = registered.get_attn_backend()
        key = (backend.get_name(), type(spec).__name__, int(spec.block_size), getattr(spec, "cache_dtype_str", None))
        if key not in self.builders:
            width = None
            builder_cls = backend.get_builder_cls()
            if getattr(builder_cls, "requires_block_table_width", False):
                from vllm.v1.worker.block_table import get_block_table_width

                max_blocks = spec.max_num_blocks_per_req(self.vllm_config, self.vllm_config.model_config.max_model_len)
                width = get_block_table_width(max_blocks, spec.block_size)
            self.builders[key] = (
                _make_builder(backend, spec, prefix, self.vllm_config, sub, device=str(self.device)),
                width,
            )
        builder, width = self.builders[key]
        table = sub.block_table_tensor
        if width is not None and table.shape[1] < width:
            pad = self.torch.zeros((table.shape[0], width - table.shape[1]), dtype=table.dtype, device=table.device)
            sub.block_table_tensor = self.torch.cat([table, pad], dim=1)
        return key, builder

    def metadata(self, common):
        static_ctx = self.vllm_config.compilation_config.static_forward_context
        metadata, remapped, built = {}, {}, {}
        for attn in self.attns:
            for layer in self.cache_layers(attn):
                registered = static_ctx[layer.prefix]
                spec = registered.get_kv_cache_spec(self.vllm_config)
                if spec is None:
                    continue
                ring = self.bound[layer.prefix]["ring"]  # the ring this case's caches were bound with
                rkey = (spec.block_size, ring)
                if rkey not in remapped:
                    remapped[rkey] = (
                        common
                        if spec.block_size == self.vllm_config.cache_config.block_size and not ring
                        else self._remapped(common, spec.block_size, ring_blocks=ring)
                    )
                sub = remapped[rkey]
                key, builder = self._builder(registered, spec, layer.prefix, sub)
                if key not in built:
                    built[key] = builder.build(0, sub)
                metadata[layer.prefix] = built[key]
        return metadata

    def forward(self, common, *, layers=None, qualify=False):
        """One eager forward over the attention layers (all 40 unless ``layers`` restricts)."""
        import torch
        from vllm.forward_context import set_forward_context

        metadata = self.metadata(common)
        n = int(common.positions.numel())
        hidden = self.hidden_pool[:n]
        observations = []
        with set_forward_context(metadata, self.vllm_config), torch.inference_mode():
            for index, (attn, norm) in enumerate(zip(self.attns, self.norms, strict=True)):
                if layers is not None and index not in layers:
                    continue
                value = attn(common.positions, norm(hidden))
                if qualify:
                    observations.append((index, int(value.shape[0]), torch.isfinite(value).all(), value.ne(0).any()))
        if qualify:
            flags = torch.stack([f for _, _, a, b in observations for f in (a, b)]).cpu().tolist()
            bad = [i for k, (i, _, _, _) in enumerate(observations) if not (flags[2 * k] and flags[2 * k + 1])]
            if bad:
                raise RuntimeError(f"native attention produced non-finite or all-zero output at layers {bad[:5]}")
        return metadata


def common_metadata(batch, seq_len, query_len, device, block_table=None):
    import torch

    from collector.vllm.utils import BatchSpec, create_common_attn_metadata

    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[seq_len] * batch, query_lens=[query_len] * batch),
        block_size=BLOCK,
        device=torch.device(device),
        arange_block_indices=True,
    )
    if block_table is not None:
        width = common.block_table_tensor.shape[1]
        if block_table.shape[1] >= width:
            common.block_table_tensor = block_table[:, :width].contiguous()
        else:
            common.block_table_tensor = torch.cat(
                [
                    block_table,
                    torch.zeros(
                        (batch, width - block_table.shape[1]), dtype=block_table.dtype, device=block_table.device
                    ),
                ],
                dim=1,
            )
    if getattr(common, "_seq_lens_cpu", None) is not None:
        common.seq_lens_cpu_upper_bound = common._seq_lens_cpu
    start = seq_len - query_len
    common.positions = (start + torch.arange(query_len, device=device, dtype=torch.long)).repeat(batch)
    table = common.block_table_tensor.cpu()
    slots = [
        int(table[req, pos // BLOCK]) * BLOCK + pos % BLOCK for req in range(batch) for pos in range(start, seq_len)
    ]
    common.slot_mapping.copy_(torch.tensor(slots, dtype=torch.int64, device=device))
    return common


# --------------------------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------------------------
class Targets:
    def __init__(self, manifest, attns):
        self.phase = "context"
        self.sets = {}
        for phase in ("context", "generation"):
            reps = manifest["representatives"][phase]
            core = set(reps.get("attention_core", {}).values())
            index = set(reps.get("indexer", {}).values()) | {
                i for i in core if getattr(attns[i], "indexer", None) is not None
            }
            self.sets[phase] = dict(layer=core, indexer=index)

    def layers(self):
        return sorted(self.sets[self.phase]["layer"] | self.sets[self.phase]["indexer"])


def install_context_intervals(attns, manifest, targets, intervals):
    import torch.distributed as dist

    tp = manifest["tp_size"]
    all_reduce = (lambda t: (dist.all_reduce(t), t)[1]) if tp > 1 else None
    for index, attn in enumerate(attns):
        if tp > 1 and (attn.wo_b.reduce_results is not True or attn.wo_b.tp_size != tp):
            raise RuntimeError("attention requires unchanged pure-TP output reduction ownership")
        attn.wo_b.reduce_results = False  # o_proj.py returns wo_b(z); the same reduction runs after the end event
        intervals.wrap(
            attn,
            "forward",
            (lambda *a, _i=index, **k: ("layer", _i) if _i in targets.sets[targets.phase]["layer"] else None),
            drain=True,
            after=all_reduce,
        )
        indexer = getattr(attn, "indexer", None)
        if indexer is not None:
            resolve = (
                lambda *a, _i=index, **k: ("indexer", _i) if _i in targets.sets[targets.phase]["indexer"] else None
            )
            intervals.wrap(indexer, "forward", resolve)
            # indexer_op is a registered submodule: time its forward (nn.Module.__call__ -> forward)
            intervals.wrap(
                indexer.indexer_op,
                "forward",
                resolve,
                witness=dispatch_label(indexer.indexer_op, "forward") + "[sparse_attn_indexer]",
            )


class DecodeLayerGraphs:
    """Serving-faithful decode timing of one layer: graph(pre) → eager sparse indexer + MLA → graph(post)."""

    def __init__(self, attn, positions, value, index, witness_prefix):
        import torch

        self.torch, self.attn, self.index = torch, attn, index
        self.witness = {witness_prefix + ".forward[piecewise]"}
        saved = {}
        original_break, original_oproj = attn._sparse_indexer_and_attn, attn._o_proj
        indexer = getattr(attn, "indexer", None)
        # nested external events around the indexer preparation inside the pre graph
        self.prep_events = None
        if indexer is not None:
            original_prep = indexer.forward
            prep_events = []

            def prep(*args, **kwargs):
                if torch.cuda.is_current_stream_capturing():
                    start, end = (
                        torch.cuda.Event(enable_timing=True, external=True),
                        torch.cuda.Event(enable_timing=True, external=True),
                    )
                    start.record()
                    result = original_prep(*args, **kwargs)
                    end.record()
                    prep_events.append((start, end))
                    return result
                return original_prep(*args, **kwargs)

            indexer.forward = prep
            self.prep_events = prep_events
        try:
            # the class method, not the instance attribute: the context-regime interval wrapper (with its
            # trailing all-reduce) must not run inside the captured graphs
            forward = type(attn).forward
            for _ in range(2):
                forward(attn, positions, value)
            torch.cuda.synchronize()

            def record_break(*args, **kwargs):
                saved["break"] = (args, kwargs)

            def record_oproj(*args, **kwargs):
                saved["oproj"] = (args, kwargs)
                return None

            attn._sparse_indexer_and_attn, attn._o_proj = record_break, record_oproj
            self.pre = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.pre):
                forward(attn, positions, value)
            attn._sparse_indexer_and_attn, attn._o_proj = original_break, original_oproj
            torch.cuda.synchronize()
            if "break" not in saved or "oproj" not in saved:
                raise RuntimeError(
                    f"layer {index}: the attention forward did not reach the sparse-attention split point"
                )
            self.break_args = saved["break"]
            self.oproj_args = saved["oproj"]
            # eager region once (warm) then the post graph on its static output
            original_break(*self.break_args[0], **self.break_args[1])
            torch.cuda.synchronize()
            self.post = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.post):
                self.output = original_oproj(*self.oproj_args[0], **self.oproj_args[1])
            torch.cuda.synchronize()
        finally:
            attn._sparse_indexer_and_attn, attn._o_proj = original_break, original_oproj
            if indexer is not None:
                indexer.forward = original_prep
        self.eager_break = original_break
        self.indexer = indexer
        if indexer is not None:
            self.witness.add(dispatch_label(indexer, "forward"))
            self.witness.add(dispatch_label(indexer.indexer_op, "forward") + "[sparse_attn_indexer]")

    def replay(self):
        """(layer_ms, indexer_ms) of one replayed decode step of this layer."""
        torch = self.torch
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[0].record()
        self.pre.replay()
        ev[1].record()
        indexer_ms = 0.0
        if self.indexer is not None:
            op = self.indexer.indexer_op
            original_forward = op.forward
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            def timed_forward(*args, **kwargs):
                a.record()
                result = original_forward(*args, **kwargs)
                b.record()
                return result

            op.forward = timed_forward
            try:
                self.eager_break(*self.break_args[0], **self.break_args[1])
            finally:
                op.forward = original_forward
        else:
            self.eager_break(*self.break_args[0], **self.break_args[1])
        ev[2].record()
        self.post.replay()
        ev[3].record()
        torch.cuda.synchronize()
        if self.indexer is not None:
            indexer_ms = a.elapsed_time(b) + sum(s.elapsed_time(e) for s, e in self.prep_events)
        return ev[0].elapsed_time(ev[3]), indexer_ms


# --------------------------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------------------------
def seed_prefix(stack, batch, prefix, chunk, device, block_table):
    for start in range(0, prefix, chunk):
        end = min(prefix, start + chunk)
        stack.forward(common_metadata(batch, end, end - start, device, block_table=block_table))


def run_attention_case(stack, targets, intervals, case, plan, stream, manifest, receipt, device):
    import torch
    import torch.distributed as dist
    from vllm.forward_context import set_forward_context

    batch, query, prefix = case["batch_size"], case["query"], case["past_kv"]
    # one seeding forward carries batch*chunk tokens: cap it at the serving max_num_batched_tokens
    chunk = max(1, min(plan["chunk_prefill_size"], plan["pool"]["max_new_tokens"] // batch))
    targets.phase = case["phase"]
    intervals.active = False
    total = prefix + query if case["phase"] == "context" else prefix + 1
    new_tokens = query if case["phase"] == "context" else 1
    stack.bind_caches(batch, total, ring_tokens=max(new_tokens, chunk if prefix else 0))
    full = common_metadata(batch, total, query if case["phase"] == "context" else 1, device)
    if prefix:
        seed_prefix(stack, batch, prefix, chunk, device, full.block_table_tensor)
    entries, reps = manifest["entries"], manifest["representatives"]
    if case["phase"] == "context":
        stack.forward(full, qualify=True)
        for sample in range(plan["warmup"] + plan["iterations"]):
            intervals.active = sample >= plan["warmup"]
            intervals.buckets[None] = []
            stack.forward(full)
            if not intervals.active:
                continue
            records = intervals.take(None)
            if any(captured for *_, captured in records):
                raise RuntimeError("context intervals must be eager")
            for entry, ms, witnesses, _ in reduce_layer_intervals(records, reps, entries, "context").values():
                stream.write(
                    entry,
                    case,
                    latency=ms,
                    kernel_source="+".join(sorted(witnesses)),
                    used_cuda_graph=False,
                    sample=sample,
                )
            stream.flush()
        intervals.active = False
        return
    # generation: serving piecewise graphs per representative layer
    stack.forward(full, layers=set(targets.layers()), qualify=True)  # eager qualification of the decode step
    metadata = stack.metadata(full)
    hidden = stack.hidden_pool[: int(full.positions.numel())]
    graphs = {}
    tp = manifest["tp_size"]
    with set_forward_context(metadata, stack.vllm_config), torch.inference_mode():
        for index in targets.layers():
            attn = stack.attns[index]
            graphs[index] = DecodeLayerGraphs(
                attn,
                full.positions,
                stack.norms[index](hidden),
                index,
                type(attn).__module__ + "." + type(attn).__name__,
            )
        for sample in range(plan["warmup"] + plan["iterations"]):
            records = []
            for index, graph in graphs.items():
                layer_ms, indexer_ms = graph.replay()
                if tp > 1:
                    dist.all_reduce(graph.output)
                records.append((("layer", index), "+".join(sorted(graph.witness)), layer_ms, True))
                if graph.indexer is not None:
                    records.append(
                        (
                            ("indexer", index),
                            "+".join(sorted(w for w in graph.witness if "ndexer" in w)),
                            indexer_ms,
                            True,
                        )
                    )
            if sample < plan["warmup"]:
                continue
            for entry, ms, witnesses, _ in reduce_layer_intervals(records, reps, entries, "generation").values():
                stream.write(
                    entry,
                    case,
                    latency=ms,
                    kernel_source="+".join(sorted(witnesses)),
                    used_cuda_graph=True,
                    sample=sample,
                )
            stream.flush()
    del graphs
    gc.collect()
    torch.cuda.empty_cache()


def _measure_tokens(call, wraps, case, plan, stream, qualify, expected_calls=None):
    import torch

    expected_calls = len(wraps) if expected_calls is None else expected_calls
    if case["phase"] == "context":
        intervals = Intervals()
        try:
            for module, method, entry in wraps:
                intervals.wrap(module, method, (lambda *a, _e=entry, **k: _e), drain=True)
            for sample in range(plan["warmup"] + plan["iterations"]):
                intervals.active = True
                intervals.buckets[None] = []
                torch.cuda.synchronize()
                out = call()
                torch.cuda.synchronize()
                if sample == 0:
                    qualify(out)
                records = intervals.take(None)
                if len(records) != expected_calls:
                    raise RuntimeError(
                        f"{case['case_id']}: expected {expected_calls} native sub-calls, observed {len(records)}"
                    )
                if sample < plan["warmup"]:
                    continue
                grouped = {}
                for entry, witness, ms, _ in records:
                    slot = grouped.setdefault(entry["structure_key"], (entry, [0.0], set()))
                    slot[1][0] += ms
                    slot[2].add(witness)
                for entry, (ms,), witnesses in grouped.values():
                    stream.write(
                        entry,
                        case,
                        latency=ms,
                        kernel_source="+".join(sorted(witnesses)),
                        used_cuda_graph=False,
                        sample=sample,
                    )
            stream.flush()
        finally:
            intervals.restore()
        return
    graphed = GraphedCalls()
    try:
        for module, method, entry in wraps:
            graphed.wrap(module, method, entry)
        qualify(graphed.record(call))
        if len(graphed.calls) != expected_calls:
            raise RuntimeError(
                f"{case['case_id']}: expected {expected_calls} native sub-calls, observed {len(graphed.calls)}"
            )
        for entry, per_iteration, witnesses in graphed.measure(plan["warmup"], plan["iterations"]).values():
            for i, ms in enumerate(per_iteration):
                stream.write(
                    entry,
                    case,
                    latency=ms,
                    kernel_source="+".join(sorted(witnesses)),
                    used_cuda_graph=True,
                    sample=plan["warmup"] + i,
                )
        stream.flush()
    finally:
        graphed.restore()


def _entry(manifest, phase, component, name_part=None, layer=None):
    for e in manifest["entries"]:
        if (
            e["phase"] == phase
            and e["component"] == component
            and (layer is None or e["layer"] == layer)
            and (name_part is None or name_part in e["name"])
        ):
            return e
    raise RuntimeError(f"manifest lacks {component}/{name_part} in {phase}")


def _check_finite(label):
    def qualify(out):
        if not finite(out):
            raise RuntimeError(f"{label} output non-finite")

    return qualify


def collect_shared_linear(layer, manifest, plan, cases, stream, receipt):
    import torch

    shared = layer.ffn.shared_experts
    if shared is None or shared.down_proj.reduce_results:
        raise RuntimeError("native shared expert is not the local sharded consumer layout")
    for phase in ("context", "generation"):
        for module, part, actual in (
            (shared.gate_up_proj, "gate_up", (shared.gate_up_proj.output_size_per_partition, HIDDEN)),
            (shared.down_proj, "down", (HIDDEN, shared.down_proj.input_size_per_partition)),
        ):
            entry = _entry(manifest, phase, "shared_linear", part, layer=2)
            if (
                actual != (entry["structure"]["n"], entry["structure"]["k"])
                or entry["structure"]["quant_mode"] != "fp8_block"
            ):
                raise RuntimeError("native shared projection differs from the SDK structure")
    receipt["shared_linear_identity"] = dict(
        gate_up=dispatch_label(shared.gate_up_proj, "forward"), down=dispatch_label(shared.down_proj, "forward")
    )
    for case in cases:
        wraps = [
            (shared.gate_up_proj, "forward", _entry(manifest, case["phase"], "shared_linear", "gate_up", 2)),
            (shared.down_proj, "forward", _entry(manifest, case["phase"], "shared_linear", "down", 2)),
        ]
        generator = torch.Generator().manual_seed(plan["seed"] + case["tokens"])
        hidden = torch.randn(case["tokens"], HIDDEN, generator=generator, dtype=torch.bfloat16, device="cpu").cuda()
        _measure_tokens(lambda: shared(hidden), wraps, case, plan, stream, _check_finite("shared expert"))


def collect_mhc(layer, manifest, plan, cases, stream, receipt):
    import torch
    from vllm.models.deepseek_v41.nvidia.ops import mega_mhc as native_mhc

    for phase in ("context", "generation"):
        entry = _entry(manifest, phase, "mhc", layer=2)
        if entry["structure"] != dict(
            hidden_size=layer.hidden_size, hc_mult=layer.hc_mult, sinkhorn_iters=layer.hc_sinkhorn_iters
        ):
            raise RuntimeError("native mHC differs from the SDK structure")
    receipt["mhc_scope"] = dict(
        native_layer=2,
        sites="mhc_shifted_post_pre x2",
        norm_fused_into_combine=True,
        kernel_path="TileLang mhc_fused_post_pre on sm90 (ops/mega_mhc.py)",
    )
    hc, hidden_size = layer.hc_mult, layer.hidden_size
    for case in cases:
        entry = _entry(manifest, case["phase"], "mhc", layer=2)
        t = case["tokens"]
        generator = torch.Generator().manual_seed(plan["seed"] + t)
        residual = torch.randn(t, hc, hidden_size, generator=generator, dtype=torch.bfloat16, device="cpu").cuda()
        attn_output = torch.randn(t, hidden_size, generator=generator, dtype=torch.bfloat16, device="cpu").cuda()
        ffn_output = torch.randn(t, hidden_size, generator=generator, dtype=torch.bfloat16, device="cpu").cuda()
        post_mix = torch.ones(t, hc, 1, dtype=torch.float32, device="cuda")
        res_mix = torch.eye(hc, dtype=torch.float32, device="cuda").expand(t, hc, hc).contiguous()
        seed_pre = torch.full((t, hc), 1.0 / hc, dtype=torch.float32, device="cuda")
        *_, prev_pre, _ = native_mhc.mhc_shifted_post_pre(
            ffn_output,
            residual,
            post_mix,
            res_mix,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
            pre_mix=seed_pre,
            norm_weight=layer.ffn_norm.weight,
            norm_eps=layer.ffn_norm.variance_epsilon,
        )
        torch.cuda.synchronize()

        def both_sites(
            residual=residual,
            attn_output=attn_output,
            ffn_output=ffn_output,
            post_mix=post_mix,
            res_mix=res_mix,
            prev_pre=prev_pre,
        ):
            r1, p1, m1, _, attn_pre, _ = native_mhc.mhc_shifted_post_pre(
                attn_output,
                residual,
                post_mix,
                res_mix,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                layer.rms_norm_eps,
                layer.hc_eps,
                layer.hc_eps,
                layer.hc_post_alpha,
                layer.hc_sinkhorn_iters,
                pre_mix=prev_pre,
                norm_weight=layer.attn_norm.weight,
                norm_eps=layer.attn_norm.variance_epsilon,
            )
            r2, *_ = native_mhc.mhc_shifted_post_pre(
                ffn_output,
                r1,
                p1,
                m1,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                layer.rms_norm_eps,
                layer.hc_eps,
                layer.hc_eps,
                layer.hc_post_alpha,
                layer.hc_sinkhorn_iters,
                pre_mix=attn_pre,
                norm_weight=layer.ffn_norm.weight,
                norm_eps=layer.ffn_norm.variance_epsilon,
            )
            return r2

        _measure_tokens(
            both_sites,
            [(native_mhc, "mhc_shifted_post_pre", entry)],
            case,
            plan,
            stream,
            _check_finite("mHC"),
            expected_calls=2,
        )


def collect_engram(vllm_config, manifest, plan, cases, stream, token_ids, receipt):
    import torch
    from vllm.models.deepseek_v41.common.engram import EngramLayout, NgramHashState
    from vllm.models.deepseek_v41.nvidia.engram import Engram, ParallelEngramEmbedding

    config = vllm_config.model_config.hf_config
    layout = EngramLayout(config)
    hasher = NgramHashState(
        vllm_config, layout, SimpleNamespace(block_size=32, kv_cache=torch.empty(0, device="cuda"))
    ).cuda()
    device = torch.device("cuda", torch.cuda.current_device())
    for layer_id in sorted({e["layer"] for e in manifest["entries"] if e["component"] == "engram"}):
        index = layout.layer_ids.index(layer_id)
        entry_ctx = _entry(manifest, "context", "engram", layer=layer_id)
        with torch.device("cuda"):
            module = Engram(config, vllm_config.quant_config, layout, index, False, f"model.layers.{layer_id}.engram")
        receipt.setdefault("weight_initialization", {})[f"engram.{layer_id}"] = initialize_dummy_weights(
            module, plan, vllm_config, device, big_types=(ParallelEngramEmbedding,)
        )
        embed = module.embed_tokens
        if embed.cpu_offload or getattr(embed, "dp_size", 1) != 1 or embed.tp_size != plan["tp_size"]:
            raise RuntimeError("native Engram table is not the GPU-resident TP-sharded layout")
        if (
            layout.num_embeddings[index] != entry_ctx["structure"]["num_embeddings"]
            or layout.n_hash_cols != entry_ctx["structure"]["hash_columns"]
            or embed.dim != entry_ctx["structure"]["head_dim"]
            or entry_ctx["structure"]["sharding"] != "head"
        ):
            raise RuntimeError("native Engram layout differs from the SDK structure")
        receipt.setdefault("engram_identity", {})[str(layer_id)] = dict(
            sharding="tp_head_sharded_all_gather",
            heads_per_rank=int(embed.part_n_hash_cols),
            lookup=dispatch_label(embed, "lookup"),
            wkv=dispatch_label(module.wkv, "forward"),
        )
        for case in cases:
            t = case["tokens"]
            ids = torch.tensor(token_ids[:t], dtype=torch.int64, device="cuda")
            hashes = hasher(
                ids,
                torch.arange(t, device="cuda"),
                torch.tensor([0, t], dtype=torch.int32, device="cuda"),
                torch.zeros(t, dtype=torch.bool, device="cuda"),
                torch.full((1, layout.max_ngram_size - 1), -1, dtype=torch.int64, device="cuda"),
                torch.zeros(1, layout.max_ngram_size - 1, dtype=torch.bool, device="cuda"),
                None,
                None,
            )[:, index].contiguous()
            generator = torch.Generator().manual_seed(plan["seed"] + t)
            hidden = torch.randn(
                t, config.hc_mult, config.hidden_size, generator=generator, dtype=torch.bfloat16, device="cpu"
            ).cuda()
            entry = _entry(manifest, case["phase"], "engram", layer=layer_id)

            def call(module=module, hidden=hidden, hashes=hashes):
                # serving order: lookup into staging rows, all-gather the head shards (untimed), wkv + gate
                module.prepare_embeddings(hashes)
                rows = module.embed(hashes)
                module.embed = lambda _hash_ids, rows=rows: rows
                try:
                    return module(hidden, hashes)
                finally:
                    del module.embed

            _measure_tokens(
                call,
                [(embed, "lookup", entry), (module, "forward", entry)],
                case,
                plan,
                stream,
                _check_finite("Engram"),
            )
        del module
        gc.collect()
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------
# entry
# --------------------------------------------------------------------------------------------
def run(args, receipt):
    plan = json.loads(args.plan.read_text())
    manifest = json.loads(args.manifest.read_text())
    contract.validate_plan(
        plan,
        manifest,
        framework_commit=FRAMEWORK_COMMIT,
        framework_version=FRAMEWORK_VERSION,
        expected_sm=EXPECTED_SM,
        required_sources=REQUIRED_SOURCES,
    )
    rank, local_rank, tp = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if tp != plan["tp_size"] or local_rank != rank:
        raise RuntimeError("single-node torchrun world must match the plan TP")
    prepare_private_caches(rank, receipt, marker_prefix="vllm-dsv411")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
        raise RuntimeError(
            "the stateless engram hasher and the narrow attention split need the V2 model runner contract "
            "(VLLM_USE_V2_MODEL_RUNNER=1)"
        )
    import torch
    import torch.distributed as dist

    if torch.cuda.device_count() != tp:
        raise RuntimeError("visible CUDA device count differs from allocated TP")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(plan["seed"])
    receipt["allocated_device_witness"] = device_witness(local_rank, plan)
    import vllm
    from vllm.config import set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.v1.worker.workspace import init_workspace_manager

    package = Path(vllm.__file__).resolve().parent
    sources = source_hashes(package)
    receipt["source_hashes"] = sources
    check_pins(plan, sources, args.model_path, manifest, args.runtime_digest)
    receipt.update(
        framework_version=FRAMEWORK_VERSION,
        raw_package_version=importlib.metadata.version("vllm"),
        collector_revision=plan["collector_revision"],
        plan_sha256=contract.sha256_file(args.plan),
        manifest_sha256=contract.sha256_file(args.manifest),
        runtime_digest=args.runtime_digest,
        image_sha256=plan["image_sha256"],
        purpose=plan["purpose"],
        weight_initializer=WEIGHT_INITIALIZER,
        regimes=plan["regimes"],
    )
    vllm_config = build_vllm_config(args.model_path, tp, plan["pool"])
    init_distributed_environment(tp, rank, "env://", local_rank)
    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(tp, 1)
    device = torch.device("cuda", local_rank)
    init_workspace_manager(device)
    receipt["native_config"] = dict(
        quant_config=type(vllm_config.quant_config).__name__,
        kv_cache_dtype=vllm_config.cache_config.cache_dtype,
        block_size=vllm_config.cache_config.block_size,
        max_model_len=vllm_config.model_config.max_model_len,
        use_v2_model_runner=vllm_config.use_v2_model_runner,
        breakable_cudagraph=bool(os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH")),
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    token_ids = tokenizer.encode(args.prompt_file.read_text())
    longest = max(
        [c["past_kv"] + c["query"] for c in plan["cases"] if c["kind"] == "attention"]
        + [c["tokens"] for c in plan["cases"] if c["kind"] == "tokens"]
    )
    if len(token_ids) < longest or len(set(token_ids)) < 100:
        raise RuntimeError(
            f"tokenizer corpus too short ({len(token_ids)} tokens) or degenerate for the longest case ({longest})"
        )
    receipt["input_provenance"] = dict(
        text_sha256=contract.sha256_file(args.prompt_file),
        token_count=len(token_ids),
        unique_tokens=len(set(token_ids)),
        seed=plan["seed"],
    )
    provenance = dict(
        source_sha256=contract.sha256_json(sources),
        config_sha256=manifest["config_sha256"],
        runtime_digest=args.runtime_digest,
        case_plan_sha256=contract.sha256_file(args.plan),
    )
    done = completed_cases(args.output, plan) if args.resume else None
    stream = RowStream(args.output / f"rank-{rank}.jsonl", plan=plan, provenance=provenance, rank=rank, keep_cases=done)
    if done is not None:
        # cases that killed a preserved attempt are framework-side failures: recorded, skipped, never retried here
        receipt["failed_cases"] = failed_cases(args.output)
        done = done | set(receipt["failed_cases"])
        for marker in args.output.glob("current-rank-*.case"):
            marker.unlink()
        receipt["resume"] = dict(skipped_cases=len(done), kept_rows=stream.rows, failed=sorted(receipt["failed_cases"]))
    config = vllm_config.model_config.hf_config
    try:
        with set_current_vllm_config(vllm_config), torch.device("cuda"):
            torch.set_default_dtype(torch.bfloat16)
            # a resume that only has token components left must not build the attention stack at all:
            # its 40 layers, caches and decode graphs leave the allocator fragmented for the 262144-token
            # engram forward even after they are deleted
            attention_cases = [
                c for c in plan["cases"] if c["kind"] == "attention" and not (done and c["case_id"] in done)
            ]
            token_cases = [c for c in plan["cases"] if c["kind"] == "tokens"]
            if attention_cases:
                stack = AttentionStack(vllm_config, plan, device)
                receipt["weight_initialization"] = dict(
                    attention=initialize_dummy_weights(stack.attns, plan, vllm_config, device)
                )
                for norm in stack.norms:
                    norm.weight.data.fill_(1.0)
                validate_attention_geometry(list(stack.attns), manifest)
                receipt["native_pool"] = dict(
                    attention_class=stack.attention_class,
                    backend=stack.attns[2].get_attn_backend().get_name(),
                    kv_cache_dtype=stack.attns[2].kv_cache_dtype,
                    kv_bytes_per_token=int(stack.attns[2].kv_bytes_per_token),
                )
                receipt["loaded_modules"] = {
                    f"attn.{i}": describe_module(a) for i, a in enumerate(stack.attns) if i in (0, 2, 20, 24)
                }
                targets, intervals = Targets(manifest, stack.attns), Intervals()
                install_context_intervals(stack.attns, manifest, targets, intervals)
                progress = (args.output / f"progress-rank-{rank}.jsonl").open("a")
                try:
                    for case in attention_cases:
                        receipt["failed_case"] = case["case_id"]  # cleared when the case completes
                        mark_case(args.output, rank, case["case_id"])
                        started = time.monotonic()
                        run_attention_case(stack, targets, intervals, case, plan, stream, manifest, receipt, device)
                        progress.write(
                            json.dumps(
                                dict(case_id=case["case_id"], seconds=time.monotonic() - started, rows=stream.rows)
                            )
                            + "\n"
                        )
                        progress.flush()
                        receipt.pop("failed_case", None)
                        mark_case(args.output, rank, None)
                finally:
                    intervals.restore()
                receipt["bound_caches_last_case"] = stack.bound
                del stack
                # the attention layers registered themselves in the static forward context; the
                # token-component layer below is built under the same serving prefixes
                static_ctx = vllm_config.compilation_config.static_forward_context
                for name in [k for k in static_ctx if k.startswith("model.layers.")]:
                    del static_ctx[name]
                gc.collect()
                torch.cuda.empty_cache()
            if token_cases:
                if {"shared_linear", "mhc"} & set(plan["components"]):
                    from vllm.forward_context import set_forward_context
                    from vllm.models.deepseek_v41.nvidia.model import DeepseekV4DecoderLayer

                    topk_buffer = torch.empty(8192, config.index_topk, dtype=torch.int32)
                    streams = [torch.cuda.Stream() for _ in range(3)]
                    layer = DeepseekV4DecoderLayer(
                        vllm_config,
                        "model.layers.2",
                        topk_indices_buffer=topk_buffer,
                        aux_stream_list=streams,
                        candidate_block_buffer=None,
                    )
                    receipt.setdefault("weight_initialization", {})["layer2"] = initialize_dummy_weights(
                        layer, plan, vllm_config, device
                    )
                    with set_forward_context(None, vllm_config):
                        if "shared_linear" in plan["components"]:
                            collect_shared_linear(layer, manifest, plan, token_cases, stream, receipt)
                        if "mhc" in plan["components"]:
                            collect_mhc(layer, manifest, plan, token_cases, stream, receipt)
                    del layer
                    gc.collect()
                    torch.cuda.empty_cache()
                if "engram" in plan["components"]:
                    collect_engram(vllm_config, manifest, plan, token_cases, stream, token_ids, receipt)
    finally:
        stream.close()
    receipt.update(
        state="complete_pending_admission", memory_peak_allocated=torch.cuda.max_memory_allocated(), rows=stream.rows
    )
    dist.barrier()
    dist.destroy_process_group()


def admit(args):
    rows, _ = contract.aggregate_run(
        args.output,
        framework_commit=FRAMEWORK_COMMIT,
        framework_version=FRAMEWORK_VERSION,
        expected_sm=EXPECTED_SM,
        required_sources=REQUIRED_SOURCES,
    )
    with args.admit.open("xb") as destination:
        contract.write_parquet(rows, destination)
    print(f"admitted {len(rows)} dsv411 rows -> {args.admit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for option in ("plan", "manifest", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    for option in ("model-path", "prompt-file"):
        parser.add_argument("--" + option, type=Path)
    parser.add_argument("--runtime-digest")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a preserved run in --output: skip the attention cases every rank finished, re-measure the rest",
    )
    parser.add_argument("--admit", type=Path, help="CPU: admit a finished run directory into a parquet table")
    args = parser.parse_args()
    if args.admit is not None:
        admit(args)
        return
    if any(getattr(args, k) is None for k in ("model_path", "prompt_file", "runtime_digest")):
        parser.error("collection requires --model-path, --prompt-file and --runtime-digest")
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    path = args.output / f"rank-{rank}.json"
    if path.exists() and not args.resume:
        raise SystemExit("refusing to overwrite a prior run; use a fresh output directory (or --resume)")
    if args.resume:
        # a rank torchrun killed after another rank's failure may have left no receipt: its progress file is
        # the evidence of the interrupted attempt then
        if not path.exists() and not (args.output / f"progress-rank-{rank}.jsonl").exists():
            raise SystemExit("--resume needs a preserved (interrupted) run in --output")
        if path.exists():
            try:
                prior_state = json.loads(path.read_text()).get("state")
            except json.JSONDecodeError:
                prior_state = "truncated"  # killed while writing it; archived all the same
            if prior_state == "complete_pending_admission":
                raise SystemExit("--resume: the run already completed")
            # keep the interrupted attempt's receipt (with its traceback) as evidence
            attempts = len(list(args.output.glob(f"rank-{rank}.attempt-*.json")))
            path.rename(args.output / f"rank-{rank}.attempt-{attempts}.json")
    receipt = new_receipt(rank, BACKEND)
    try:
        run(args, receipt)
    except BaseException as error:
        receipt.update(
            state="failed_preserved", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
        )
        raise
    finally:
        write_receipt(path, receipt)


if __name__ == "__main__":
    main()
