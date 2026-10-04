# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dsv411 producer for SGLang v0.5.21 (torchrun, pure TP, dummy weights).

Measures the ``dsv411_module_perf`` components on the native DeepSeek-V4.1 modules:

* ``attention_core`` / ``indexer``: all forty ``MQALayer`` modules behind a facade model that owns
  the real ``DeepSeekV4TokenToKVPool`` and ``DeepseekV4AttnBackend`` through ``ModelRunner``
  (``benchmark/one_batch.py`` request lifecycle, chunked real-KV seeding). The layer interval is
  ``MQALayer.forward`` with the pure-TP ``wo_b`` all-reduce moved after the end event; the
  nested ``DeepseekV4AttnBackend._low_ratio_index_topk`` interval (index queries + scoring +
  top-k / candidate selection) is the ``indexer`` row and is subtracted from the layer for the
  ``attention_core`` row.
  - context: eager, stream drained before every representative layer (``eager_drained``);
  - generation: the serving decode CUDA graph captured by ``ModelRunner.init_cuda_graphs`` and
    replayed by ``decode``; intervals are *external* CUDA events recorded during capture and read
    after each replay (``cuda_graph``). An eager fallback is a recorded regime violation.
* ``engram`` (``_owned_rows`` + ``wkv`` + ``engram_gate``, the lookup all-reduce outside),
  ``mhc`` (both ``_hc_mix_and_combine``/``hc_post`` sites of layer 2) and ``shared_linear``
  (``gate_up_proj``/``down_proj`` of layer 2): context eager, generation as one CUDA graph per
  sub-call (``runtime.GraphedCalls``).

Framework integration informed by sgl-project/sglang@e00930c5 (Apache-2.0): benchmark/one_batch.py,
srt/models/deepseek_v4.py, srt/layers/attention/deepseek_v4_backend.py, srt/layers/engram.py,
srt/model_loader/loader.py. See THIRD_PARTY_NOTICES.md. Weights and hidden inputs are synthetic;
only native module intervals are data.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import itertools
import json
import os
import time
import traceback
from pathlib import Path

from collector.dsv411 import contract
from collector.dsv411.runtime import (
    GraphedCalls,
    Intervals,
    RowStream,
    check_pins,
    completed_cases,
    device_witness,
    failed_cases,
    finite,
    mark_case,
    new_receipt,
    prepare_private_caches,
    randomize_object_tensors,
    reduce_layer_intervals,
    source_hashes,
    write_receipt,
)

BACKEND = "sglang"
# Pinned SGLang: tag v0.5.21 (2026-09-30), image lmsysorg/sglang:v0.5.21.
FRAMEWORK_COMMIT = "e00930c5489053f26d86b179cee0d087f846acbb"
FRAMEWORK_VERSION = "0.5.21"
EXPECTED_SM = {"H100": 90, "H200": 90, "H20": 90, "B200": 100, "GB200": 100}
WEIGHT_INITIALIZER = {"name": "native_uniform_v1", "low": -1e-3, "high": 1e-3, "norm_weight": 1.0}
REQUIRED_SOURCES = {
    "benchmark/one_batch.py",
    "srt/distributed/bootstrap.py",
    "srt/model_loader/loader.py",
    "srt/model_loader/weight_utils.py",
    "srt/models/deepseek_v2.py",
    "srt/models/deepseek_v4.py",
    "srt/layers/engram.py",
    "srt/layers/quantization/fp8.py",
    "srt/model_executor/model_runner.py",
    "srt/model_executor/forward_batch_info.py",
    "srt/model_executor/runner/decode_cuda_graph_runner.py",
    "srt/model_executor/runner_utils/capture_mode.py",
    "srt/layers/attention/deepseek_v4_backend.py",
    "srt/mem_cache/deepseek_v4_memory_pool.py",
    "srt/layers/attention/dsv4/dsv41_sparse.py",
    "srt/layers/attention/dsv4/compressor.py",
}
HIDDEN = 5120


class State:
    """Per-rank measurement state shared between the facade forward and the case loop."""

    def __init__(self):
        self.phase = "context"
        self.qualifying = False
        self.observations = []
        self.eager_forwards = 0
        self.capturing = False
        self.gemm_token_limit = None  # set by build_attention_runner (FIXME(kernel-limit))


# --------------------------------------------------------------------------------------------
# geometry proof: the loaded native modules carry the SDK manifest's structure
# --------------------------------------------------------------------------------------------
def validate_attention_geometry(layers, manifest):
    for entry in (e for e in manifest["entries"] if e["phase"] == "context"):
        attn = layers[entry["layer"]].self_attn
        s = entry["structure"]
        if entry["component"] == "attention_core":
            actual = dict(
                role=manifest["layer_roles"][entry["layer"]],
                compress_ratio=int(attn.compress_ratio),
                num_heads=int(attn.n_local_heads),
                head_dim=int(attn.head_dim),
                q_lora_rank=int(attn.q_lora_rank),
                o_lora_rank=int(attn.o_lora_rank),
                o_groups=int(attn.n_local_groups),
            )
            if any(s[k] != v for k, v in actual.items()):
                raise RuntimeError(
                    f"native attention geometry differs from the SDK graph at layer {entry['layer']}: {actual}"
                )
        elif entry["component"] == "indexer":
            indexer = attn.indexer
            if indexer is None:
                raise RuntimeError(f"SDK graph expects an indexer at layer {entry['layer']}")
            actual = dict(
                compress_ratio=int(attn.compress_ratio),
                index_n_heads=int(indexer.n_local_heads),
                index_head_dim=int(indexer.index_head_dim),
                index_topk=int(indexer.index_topk),
                is_candidate_source=int(bool(indexer.is_candidate_source)),
                candidate_limit=int(indexer.candidate_topk_blocks * indexer.candidate_block_size)
                if indexer.uses_candidates
                else 0,
            )
            if any(s[k] != v for k, v in actual.items()):
                raise RuntimeError(
                    f"native indexer geometry differs from the SDK graph at layer {entry['layer']}: {actual}"
                )
    for layer_id, layer in enumerate(layers):
        owns = any(e["layer"] == layer_id and e["component"] == "indexer" for e in manifest["entries"])
        if owns != (layer.self_attn.indexer is not None):
            raise RuntimeError(f"indexer ownership differs from the SDK graph at layer {layer_id}")


# --------------------------------------------------------------------------------------------
# attention facade over the native runner
# --------------------------------------------------------------------------------------------
def build_attention_runner(bench, server, model_config, gpu_id, plan, manifest, receipt, state, intervals):
    import torch
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor import model_runner as native_runner
    from sglang.srt.model_executor.runner_utils import capture_mode
    from sglang.srt.model_loader.loader import DummyModelLoader, _get_quantization_config, post_load_weights
    from sglang.srt.model_loader.weight_utils import initialize_dummy_weights
    from sglang.srt.models import deepseek_v4 as native_model
    from torch import nn

    cfg = model_config.hf_text_config
    if cfg.num_hidden_layers != 40 or cfg.model_type != "deepseek_v41":
        raise RuntimeError("unaltered native V4.1 config required")
    # every forward slices this buffer: measured extends (batch*query) and seeding chunks (batch*chunk, capped
    # at the pool's per-forward token limit like serving chunked prefill)
    max_tokens = max(
        [c["batch_size"] * c["query"] for c in plan["cases"] if c["kind"] == "attention"]
        + [plan["pool"].get("max_new_tokens", 262144), 1]
    )

    class LayerFacade(nn.Module):
        def __init__(self, quant, index):
            super().__init__()
            self.self_attn = native_model.MQALayer(cfg, index, quant, prefix=f"model.layers.{index}.self_attn")
            self.input_layernorm = native_model.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.post_attention_layernorm = native_model.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        def refresh_mhc_norm_weight_cache(self):
            # deepseek_v4.py@v0.5.21 caches bf16 norm weights after load; this facade has no mHC.
            self._input_layernorm_weight_bf16 = self.input_layernorm.weight.data.bfloat16().contiguous()
            self._post_attention_layernorm_weight_bf16 = (
                self.post_attention_layernorm.weight.data.bfloat16().contiguous()
            )
            self._hc_attn_tf32_parts = self._hc_ffn_tf32_parts = None
            self._hc_attn_bf16_parts = self._hc_ffn_bf16_parts = None

    class AttentionStack(nn.Module):
        def __init__(self, quant):
            super().__init__()
            self.config, self.quant_config = cfg, quant
            self.start_layer, self.end_layer = 0, 40
            self.wo_a_fp8 = native_model.wo_a_fp8_gemm_enabled(quant)
            self.model = nn.Module()
            self.model.start_layer, self.model.end_layer = 0, 40
            self.model.layers = nn.ModuleList([LayerFacade(quant, i) for i in range(40)])
            native_model.get_attn_tp_context().init_context(cfg.q_lora_rank, is_dsa=True)
            self.late_layer_start = None
            # one seeded synthetic hidden buffer; sliced per forward (graph-capture safe: no RNG inside)
            generator = torch.Generator(device="cuda").manual_seed(plan["seed"])
            self.register_buffer(
                "hidden_pool",
                torch.randn((max_tokens, cfg.hidden_size), generator=generator, dtype=torch.bfloat16, device="cuda"),
                persistent=False,
            )

        def _setup_fp8_wo_a_scales(self, is_nextn):
            native_model.DeepseekV4ForCausalLM._setup_fp8_wo_a_scales(self, is_nextn)

        def post_load_weights(self):
            native_model.DeepseekV4ForCausalLM.post_load_weights(self)

        @torch.no_grad()
        def forward(self, input_ids, positions, forward_batch, **kwargs):
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing:
                intervals.bucket = (int(forward_batch.batch_size), capture_mode._capture_attention_variant)
            else:
                intervals.bucket = None
                state.eager_forwards += 1
            n = input_ids.numel()
            hidden = self.hidden_pool[:n]
            for index, layer in enumerate(self.model.layers):
                value = layer.input_layernorm(hidden)
                with layer.self_attn.maybe_use_decode_attn_tp(forward_batch):
                    value = layer.self_attn(x=value, positions=positions, forward_batch=forward_batch, x_quant=None)
                if state.qualifying and not capturing:
                    state.observations.append(
                        (index, int(value.shape[0]), torch.isfinite(value).all(), value.ne(0).any())
                    )
            return LogitsProcessorOutput(
                next_token_logits=torch.zeros(
                    (forward_batch.batch_size, cfg.vocab_size), dtype=torch.float32, device=input_ids.device
                )
            )

    class AttentionRunner(native_runner.ModelRunner):
        def load_model(self):
            begin = time.monotonic()
            before = native_runner.get_available_gpu_memory(self.device, self.gpu_id)
            self.load_config = LoadConfig(load_format="dummy")
            quant = _get_quantization_config(self.model_config, self.load_config)
            with torch.device(f"cuda:{gpu_id}"):
                self.model = AttentionStack(quant)
            initialize_dummy_weights(
                self.model, low=WEIGHT_INITIALIZER["low"], high=WEIGHT_INITIALIZER["high"], seed=plan["seed"]
            )
            with torch.no_grad():
                for layer in self.model.model.layers:
                    layer.input_layernorm.weight.fill_(WEIGHT_INITIALIZER["norm_weight"])
                    layer.post_attention_layernorm.weight.fill_(WEIGHT_INITIALIZER["norm_weight"])
            post_load_weights(self.model)
            for child in list(self.model.modules()):
                method = getattr(child, "quant_method", None)
                if method is not None:
                    method.process_weights_after_loading(child)
            self.model.eval()
            self.loader = DummyModelLoader(self.load_config)
            self.sliding_window_size = native_runner.resolve_sliding_window_size(self.model, self.model_config)
            self.prefill_aware_swa, self.dtype = False, self.model_config.dtype
            self.weight_load_mem_usage = before - native_runner.get_available_gpu_memory(self.device, self.gpu_id)
            self.weight_load_time = time.monotonic() - begin

    runner = AttentionRunner(
        model_config=model_config,
        mem_fraction_static=server.mem_fraction_static,
        gpu_id=gpu_id,
        nccl_port=int(os.environ["MASTER_PORT"]),
        server_args=server,
    )
    runner.alloc_memory_pool()
    runner.init_attention_backends()
    if contract.kv_seed_of(plan) == "random_kv":
        # random_kv: the pools hold bounded random contents once; seeding only does the allocation bookkeeping
        generator = torch.Generator(device="cuda").manual_seed(plan["seed"] + 7919 * (gpu_id + 1))
        receipt["kv_seed"] = dict(
            regime="random_kv",
            **randomize_object_tensors(runner.token_to_kv_pool, generator),
        )
        torch.cuda.synchronize()
    else:
        receipt["kv_seed"] = dict(regime="real_kv")
    layers = list(runner.model.model.layers)
    # FIXME(kernel-limit): sglang 0.5.21's Triton w8a8 block-fp8 GEMM (kernels/ops/gemm/fp8_kernel.py:99-140)
    # forms A/C offsets in int32 (offs_am * stride_am, offs_cm * stride_cm), and this checkpoint's 32-wide
    # weight blocks route EVERY fp8 linear onto it (layers/quantization/fp8_utils.py:582-595: only Triton
    # reads a non-128 K block). A forward whose token count x the largest per-rank weight dimension reaches
    # 2^31 faults (illegal memory access / CUBLAS_STATUS_EXECUTION_FAILED): 196608 tokens x 16384 (wq_b at
    # TP2) did, 131072 x 16384 = 2^31 - 1 max offset does not. Serving never prefills more than its chunk
    # budget per forward, so it never reaches the limit; the grid's batch x query does. Context cases beyond
    # it are recorded as kernel-limit failures (observed, not predicted) and seeding forwards stay below it.
    max_dim = max(max(param.shape) for layer in layers for param in layer.self_attn.parameters() if param.dim() == 2)
    state.gemm_token_limit = (1 << 31) // int(max_dim)
    receipt["kernel_limits"] = dict(
        triton_w8a8_block_fp8_int32_offsets=dict(
            max_weight_dim_per_rank=int(max_dim),
            max_tokens_per_forward=state.gemm_token_limit,
            source="sglang 0.5.21 kernels/ops/gemm/fp8_kernel.py:99-140; fp8_utils.py:582-595",
        )
    )
    validate_attention_geometry(layers, manifest)
    receipt["native_pool"] = dict(
        type=type(runner.token_to_kv_pool).__name__,
        backend=type(runner.attn_backend).__name__,
        max_tokens=int(runner.max_total_num_tokens),
        low_ratios=sorted(runner.token_to_kv_pool.kv_pools),
    )
    if receipt["native_pool"]["type"] != "DeepSeekV4TokenToKVPool":
        raise RuntimeError("native compressed KV pool ownership differs")
    install_attention_intervals(runner, layers, manifest, state, intervals)
    # Serving decode graphs: captured here (my wrappers record external events during capture).
    # init_cuda_graphs also installs the eager/prefill runners ModelRunner.forward dispatches through,
    # so it runs even when decode graphs are disabled (no generation cases).
    has_generation = any(c["kind"] == "attention" and c["phase"] == "generation" for c in plan["cases"])
    state.phase, state.capturing, intervals.active = "generation", has_generation, has_generation
    runner.init_cuda_graphs()
    state.capturing, intervals.active = False, False
    intervals.buckets[None] = []  # eager capture warmups are not measurements
    if has_generation:
        graph_runner = runner.decode_cuda_graph_runner
        captured = [k for k in intervals.buckets if k is not None]
        if graph_runner is None or not captured:
            raise RuntimeError(
                f"serving decode CUDA graphs were not captured (runner={type(graph_runner).__name__}, "
                f"buckets={captured}); the generation regime requires them"
            )
        receipt["decode_graphs"] = dict(
            capture_bs=list(getattr(graph_runner, "capture_bs", [])),
            buckets=[list(map(str, k)) for k in intervals.buckets if k is not None],
        )
    return bench._TorchBenchRunner(runner)


def install_attention_intervals(runner, layers, manifest, state, intervals):
    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    tp = manifest["tp_size"]
    targets = {phase: {} for phase in ("context", "generation")}
    for phase in targets:
        reps = manifest["representatives"][phase]
        core_layers = set(reps.get("attention_core", {}).values())
        index_layers = set(reps.get("indexer", {}).values()) | {
            i for i in core_layers if layers[i].self_attn.indexer is not None
        }
        targets[phase] = dict(layer=core_layers, indexer=index_layers)
    for index, layer in enumerate(layers):
        attn = layer.self_attn
        if tp > 1 and (attn.attn_tp_size != tp or attn.wo_b.reduce_results is not True):
            raise RuntimeError("attention requires unchanged pure-TP output reduction ownership")
        # deepseek_v4.py: wo_b reduces when reduce_results is True; the same reduction now runs after the end event.
        attn.wo_b.reduce_results = False
        intervals.wrap(
            attn,
            "forward",
            (lambda *a, _i=index, **k: ("layer", _i) if _i in targets[state.phase]["layer"] else None),
            drain=True,
            after=tensor_model_parallel_all_reduce if tp > 1 else None,
        )
    backend = runner.attn_backend
    intervals.wrap(
        backend,
        "_low_ratio_index_topk",
        (
            lambda layer, *a, **k: ("indexer", int(layer.layer_id))
            if int(layer.layer_id) in targets[state.phase]["indexer"]
            else None
        ),
    )


# --------------------------------------------------------------------------------------------
# request lifecycle (chunked real-KV seeding)
# --------------------------------------------------------------------------------------------
def _reset_fill(req, cut_len):
    # keep the native container type (array.array / list): prepare_for_extend reads it as a buffer
    del req.full_untruncated_fill_ids[cut_len:]


def slide_windows(runner, reqs, pre_len):
    """Free the SWA slots that fell out of each request's sliding window, as the serving scheduler does
    before it schedules the next chunk of a chunked prefill (sglang 0.5.21 ``ScheduleBatch.maybe_evict_swa``,
    managers/schedule_batch.py:3942-3954: chunk cache, no overlap -> ``evict_sliding_windows(req, prefix_len)``
    -> ``mem_cache/common.py:55 free_swa_out_of_window_slots``). The producer has no scheduler, so it slides
    the windows itself before every seeding chunk and before the measured extend; the full/indexer slots
    stay (serving keeps them too), only the 40-layer SWA copies behind ``pre_len - window`` are released.
    Without this the SWA pool would have to hold every seeded token (batch x past kv at 40 x 584 B each)."""
    from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
    from sglang.srt.runtime_context import get_schedule

    torch_runner = runner.torch_runner
    allocator = torch_runner.token_to_kv_pool_allocator
    allocator.free_group_begin()
    for req in reqs:
        free_swa_out_of_window_slots(
            req,
            pre_len,
            sliding_window_size=torch_runner.sliding_window_size,
            page_size=get_schedule().page_size,
            req_to_token_pool=torch_runner.req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            is_chunk_cache=True,
        )
    allocator.free_group_end()


LONG_KV_SEED_FLOOR = 131072


def seed_tokens_per_request(plan: dict, batch_size: int, done: int, token_limit: int | None = None) -> int:
    """New tokens per request of one seeding forward. Serving's PrefillAdder admits at most
    chunked_prefill_size new tokens per prefill batch (sglang 0.5.21 managers/schedule_policy.py
    ``rem_chunk_tokens``); while the kv is short the producer batches every request's chunk into one
    forward (bounded transient, far fewer forwards at batch 1024) and from LONG_KV_SEED_FLOOR on it keeps
    the serving budget: the indexer-prefill transient scales with new tokens x kv (65536 new tokens at
    ~1M kv asked the caching allocator for 6.4 GB blocks on H20, at the edge of the device)."""
    chunk = plan["chunk_prefill_size"]
    per_forward = chunk if done >= LONG_KV_SEED_FLOOR else min(chunk * batch_size, plan["pool"]["max_new_tokens"])
    if token_limit is not None:
        per_forward = min(per_forward, token_limit)  # FIXME(kernel-limit): see build_attention_runner
    return max(1, min(chunk, per_forward // batch_size))


def allocate_only(reqs, runner, bench):
    """The allocation half of one_batch.extend (sglang 0.5.21 benchmark/one_batch.py:486-505): the
    ScheduleBatch is built and ``prepare_for_extend`` allocates the chunk's slots into req_to_token and the
    pools exactly as serving does - the model forward is skipped (random_kv seeding)."""
    import torch

    torch_runner = runner.torch_runner
    tree_cache = bench.TreeCacheNamespace(
        page_size=bench.get_schedule().page_size,
        device=torch_runner.device,
        token_to_kv_pool_allocator=torch_runner.token_to_kv_pool_allocator,
    )
    batch = bench.ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=torch_runner.req_to_token_pool,
        token_to_kv_pool_allocator=torch_runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=torch_runner.model_config,
        enable_overlap=False,
        spec_algorithm=bench.SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    next_ids = torch.randint(1000, 100000, (len(reqs),), dtype=torch.int64, device=torch_runner.device)
    return next_ids, batch


def seed_prefix(runner, bench, token_ids, batch_size, prefix, plan, token_limit=None):
    """Seed ``prefix`` tokens per request in serving-sized chunks; returns (reqs, batch, next_ids).
    real_kv: every chunk is a real prefill of corpus tokens; random_kv: every chunk is allocated (serving
    bookkeeping, windows slid) but not computed - the pools were filled with random contents at startup."""
    extend = (lambda reqs: allocate_only(reqs, runner, bench)) if contract.kv_seed_of(plan) == "random_kv" else None
    first = min(prefix, seed_tokens_per_request(plan, batch_size, 0, token_limit))
    reqs = bench.prepare_synthetic_inputs_for_latency_test(
        batch_size, first, [token_ids[:first] for _ in range(batch_size)]
    )
    if extend is None:
        next_ids, _, batch = runner.extend(reqs)
    else:
        next_ids, batch = extend(reqs)
    done = first
    while done < prefix:
        end = min(prefix, done + seed_tokens_per_request(plan, batch_size, done, token_limit))
        for req in reqs:
            _reset_fill(req, done)
        bench.prepare_extend_inputs_for_correctness_test(
            argparse.Namespace(cut_len=done), [token_ids[:end] for _ in reqs], reqs, runner.torch_runner
        )
        slide_windows(runner, reqs, done)
        if extend is None:
            next_ids, _, batch = runner.extend(reqs)
        else:
            next_ids, batch = extend(reqs)
        done = end
    return reqs, batch, next_ids


def release_query_tokens(runner, reqs, prefix, query):
    """Return the measured extend's KV slots (full + SWA) so the seeded prefix serves the next sample and
    the next query length: ``alloc_for_extend`` allocates every extend's tokens afresh (sglang 0.5.21
    mem_cache/allocation.py:344-410), so without this they would accumulate. Pages are freed whole and
    the segment starts at the first page boundary at or after ``prefix``: the page before it is shared
    with the window tokens and its query slots are simply re-used by the next extend (``last_loc`` fills
    the open page first); ``free_segment`` requires that alignment (mem_cache/allocator/paged.py:282)."""
    from sglang.srt.runtime_context import get_schedule

    page = get_schedule().page_size
    start = -(-prefix // page) * page
    end = prefix + query
    if start >= end:
        return
    torch_runner = runner.torch_runner
    allocator = torch_runner.token_to_kv_pool_allocator
    allocator.free_group_begin()
    for req in reqs:
        slots = torch_runner.req_to_token_pool.req_to_token[req.kv.req_pool_idx, start:end]
        allocator.free_segment(slots, start_pos=start)
    allocator.free_group_end()


def run_generation_case(runner, bench, token_ids, case, plan, state, intervals, stream, manifest, receipt):
    batch_size, prefix = case["batch_size"], case["past_kv"]
    receipt["failed_case"] = case["case_id"]  # cleared when the case completes
    mark_case(stream.output, stream.rank, case["case_id"])
    import torch

    runner.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    state.phase = "generation"
    intervals.active = False
    entries = manifest["entries"]
    reps = manifest["representatives"]
    reqs, batch, next_ids = seed_prefix(runner, bench, token_ids, batch_size, prefix, plan, state.gemm_token_limit)
    graph_runner = runner.torch_runner.decode_cuda_graph_runner
    for sample in range(plan["warmup"] + plan["iterations"]):
        state.eager_forwards = 0
        next_ids, _ = runner.decode(next_ids, batch)
        if state.eager_forwards:
            message = (
                f"{case['case_id']}: decode ran eagerly ({state.eager_forwards} forwards); "
                "the generation regime requires the serving CUDA graph"
            )
            receipt["regime_violations"].append(message)
            raise RuntimeError(message)
        key = graph_runner._replay_graph_key
        bucket = (int(key.size), getattr(key, "attention_variant", None))
        if bucket not in intervals.buckets:
            raise RuntimeError(f"replayed graph {bucket} has no recorded intervals; captured {list(intervals.buckets)}")
        if sample < plan["warmup"]:
            continue
        records = intervals.take(bucket, clear=False)
        if not all(captured for *_, captured in records):
            raise RuntimeError("generation intervals must come from the captured graph")
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
    receipt.setdefault("decode_kv_drift_tokens", plan["warmup"] + plan["iterations"] - 1)
    runner.cleanup(batch)
    runner.clear()


def _capacity_failure(error: BaseException) -> str | None:
    """Classify a recoverable capacity miss: the plan's KV pools ran dry (sglang's paged allocators ask the
    tree cache to evict; the bench path's TreeCacheNamespace has no evict_for_alloc, or the SWA pool reports
    the shortfall) or the device ran out of memory in a forward (a Python-level OOM, the context is intact).
    Returns the record prefix, or None for anything else (re-raise)."""
    text = str(error)
    if "evict_for_alloc" in text or "eviction insufficient" in text:
        return "PoolCapacity"
    if type(error).__name__ == "OutOfMemoryError" or "CUDA out of memory" in text:
        return "MemoryCapacity"
    return None


def _pool_exhausted(error: BaseException) -> bool:
    return _capacity_failure(error) is not None


def run_context_group(runner, bench, token_ids, cases, plan, state, intervals, stream, manifest, receipt, progress):
    """Context cases sharing (batch, past kv): the prefix is seeded once and every query length is measured
    on it. Short prefixes (one seeding forward) are re-seeded per sample from a cleared pool; prefix-free
    cases reset the pool per sample (one_batch's cleanup is a no-op). After every measured extend the
    query tokens' pages are returned (release_query_tokens), so the pool holds prefix + one extend."""
    import torch

    batch_size, prefix = cases[0]["batch_size"], cases[0]["past_kv"]
    reseed = prefix <= seed_tokens_per_request(plan, batch_size, 0, state.gemm_token_limit)
    runner.clear()
    # hundreds of cases leave the caching allocator fragmented; a 262144-token eager prefill needs
    # multi-GB contiguous temporaries (one shard hit allocation retries, then an illegal access)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    state.phase = "context"
    intervals.active = False
    entries = manifest["entries"]
    reps = manifest["representatives"]
    reqs = None
    if prefix and not reseed:
        try:
            reqs, _, _ = seed_prefix(runner, bench, token_ids, batch_size, prefix, plan, state.gemm_token_limit)
        except (AttributeError, RuntimeError) as error:
            if not _pool_exhausted(error):
                raise
            # the shared prefix itself does not fit the plan's pools: every case of the group is a PoolCapacity record
            kind = _capacity_failure(error)
            for case in cases:
                receipt.setdefault("failed_cases", {})[case["case_id"]] = f"{kind} (prefix seeding): {error}"
            receipt.pop("failed_case", None)
            mark_case(stream.output, stream.rank, None)
            runner.clear()
            return
    for case in cases:
        started = time.monotonic()
        query = case["query"]
        if batch_size * query > state.gemm_token_limit:
            # FIXME(kernel-limit): see build_attention_runner; a classified failure record, not a crash
            receipt.setdefault("failed_cases", {})[case["case_id"]] = (
                f"KernelLimit: {batch_size * query} new tokens exceed the Triton w8a8 block-fp8 GEMM's "
                f"int32 offset range ({state.gemm_token_limit} tokens at this TP)"
            )
            continue
        receipt["failed_case"] = case["case_id"]  # cleared when the case completes
        mark_case(stream.output, stream.rank, case["case_id"])
        try:
            for sample in range(plan["warmup"] + plan["iterations"]):
                if prefix and reseed:
                    runner.clear()
                    reqs, _, _ = seed_prefix(runner, bench, token_ids, batch_size, prefix, plan, state.gemm_token_limit)
                # qualify the MEASURED forward only (seeding forwards would add their own 40 observations)
                state.qualifying = sample == 0
                state.observations = []
                if prefix:
                    for req in reqs:
                        _reset_fill(req, prefix)
                    bench.prepare_extend_inputs_for_correctness_test(
                        argparse.Namespace(cut_len=prefix),
                        [token_ids[: prefix + query] for _ in reqs],
                        reqs,
                        runner.torch_runner,
                    )
                    slide_windows(runner, reqs, prefix)
                else:
                    runner.clear()
                    reqs = bench.prepare_synthetic_inputs_for_latency_test(
                        batch_size, query, [token_ids[:query] for _ in range(batch_size)]
                    )
                intervals.active = sample >= plan["warmup"]
                intervals.buckets[None] = []
                _, _, measured = runner.extend(reqs)
                if state.qualifying:
                    flags = torch.stack([f for _, _, a, b in state.observations for f in (a, b)]).cpu().tolist()
                    bad = [
                        i
                        for k, (i, _, _, _) in enumerate(state.observations)
                        if not (flags[2 * k] and flags[2 * k + 1])
                    ]
                    if len(state.observations) != 40 or bad:
                        raise RuntimeError(
                            f"{case['case_id']}: attention outputs non-finite/zero or incomplete at layers {bad[:5]}"
                        )
                    state.qualifying = False
                if intervals.active:
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
                if prefix and not reseed:
                    release_query_tokens(runner, reqs, prefix, query)
        except (AttributeError, RuntimeError) as error:
            if not _pool_exhausted(error):
                raise
            # the bench path has no radix cache to evict into: the case does not fit the plan's pools.
            # A classified record, not a crash; the pools are reset and the group goes on.
            receipt.setdefault("failed_cases", {})[case["case_id"]] = f"{_capacity_failure(error)}: {error}"
            receipt.pop("failed_case", None)
            mark_case(stream.output, stream.rank, None)
            intervals.active = False
            runner.clear()
            torch.cuda.empty_cache()
            if prefix and not reseed:  # the shared seeded prefix went with the pools
                reqs, _, _ = seed_prefix(runner, bench, token_ids, batch_size, prefix, plan, state.gemm_token_limit)
            continue
        progress(case, started)
        receipt.pop("failed_case", None)
        mark_case(stream.output, stream.rank, None)
    runner.clear()


# --------------------------------------------------------------------------------------------
# token-only components (engram / mhc / shared_linear)
# --------------------------------------------------------------------------------------------
def _measure_tokens(call, wraps, case, plan, stream, qualify, expected_calls=None):
    """Context: eager drained intervals; generation: one CUDA graph per sub-call."""
    import torch

    expected_calls = len(wraps) if expected_calls is None else expected_calls
    mark_case(stream.output, stream.rank, case["case_id"])  # a token case that kills the attempt is recorded too
    _measure_tokens_inner(call, wraps, case, plan, stream, qualify, expected_calls, torch)
    mark_case(stream.output, stream.rank, None)


def _measure_tokens_inner(call, wraps, case, plan, stream, qualify, expected_calls, torch):
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


def _token_kernel_limit(receipt, case, component, width: int) -> bool:
    """FIXME(kernel-limit): see build_attention_runner - the same int32 offset range bounds every kernel that
    indexes tokens x ``width`` elements (sglang 0.5.21 Triton fp8 block GEMM, TileLang mHC fuse: a 131072-token
    mHC input of 4 x 5120 faulted). Records a component-scoped KernelLimit and tells the caller to skip."""
    limit = (1 << 31) // int(width)
    if case["tokens"] <= limit:
        return False
    receipt.setdefault("failed_components", {}).setdefault(case["case_id"], {})[component] = (
        f"KernelLimit: {case['tokens']} tokens x {width} elements exceed the int32 offset range (limit {limit} tokens)"
    )
    return True


def collect_shared_linear(moe, manifest, plan, cases, stream, receipt):
    import torch

    shared = moe.shared_experts
    if shared is None or getattr(moe, "_shared_expert_tp1", False) or shared.down_proj.reduce_results:
        raise RuntimeError("native shared expert is not the local sharded consumer layout")
    for phase in ("context", "generation"):
        for module, part, actual in (
            (shared.gate_up_proj, "gate_up", (shared.gate_up_proj.output_size_per_partition, HIDDEN)),
            (shared.down_proj, "down", (HIDDEN, shared.down_proj.input_size_per_partition)),
        ):
            entry = _entry(manifest, phase, "shared_linear", part, layer=2)
            if actual != (entry["structure"]["n"], entry["structure"]["k"]) or tuple(
                module.quant_method.quant_config.weight_block_size
            ) != (32, 32):
                raise RuntimeError("native shared projection differs from the SDK structure / FP8 block32")
    receipt["shared_linear_identity"] = dict(
        gate_up=type(shared.gate_up_proj.quant_method).__name__, down=type(shared.down_proj.quant_method).__name__
    )
    for case in cases:
        wraps = [
            (shared.gate_up_proj, "forward", _entry(manifest, case["phase"], "shared_linear", "gate_up", 2)),
            (shared.down_proj, "forward", _entry(manifest, case["phase"], "shared_linear", "down", 2)),
        ]
        width = max(HIDDEN, *(max(e["structure"]["n"], e["structure"]["k"]) for _, _, e in wraps))
        if _token_kernel_limit(receipt, case, "shared_linear", width):
            continue
        generator = torch.Generator().manual_seed(plan["seed"] + case["tokens"])
        hidden = torch.randn(case["tokens"], HIDDEN, generator=generator, dtype=torch.bfloat16).cuda()
        _measure_tokens(
            lambda: shared(hidden),
            wraps,
            case,
            plan,
            stream,
            lambda out: finite(out) or (_ for _ in ()).throw(RuntimeError("shared expert output non-finite")),
        )


def collect_mhc(config, quant, manifest, plan, cases, stream, receipt):
    import torch
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer

    from collector.sglang.dsv41_isolated_runner import initialize_dummy_weights as init_module_weights

    with torch.device("cuda"):
        layer = DeepseekV4DecoderLayer(config, 2, quant, prefix="model.layers.2", hc_stats_stream=None)
    if not layer.hc_pre_from_prev_sublayer or layer.use_fused_mhc_post_pre:
        raise RuntimeError("native mHC predecessor/serial dispatch differs")
    receipt.setdefault("weight_initialization", {})["mhc_layer2"] = init_module_weights(layer, plan)
    layer.refresh_mhc_norm_weight_cache()
    for phase in ("context", "generation"):
        entry = _entry(manifest, phase, "mhc", layer=2)
        if entry["structure"] != dict(
            hidden_size=layer.hidden_size, hc_mult=layer.hc_mult, sinkhorn_iters=layer.hc_sinkhorn_iters
        ):
            raise RuntimeError("native mHC differs from the SDK structure")
    receipt["mhc_scope"] = dict(
        native_layer=2, stats_stream=None, sites="_hc_mix_and_combine x2 + hc_post x2", norm_fused_into_combine=True
    )
    for case in cases:
        entry = _entry(manifest, case["phase"], "mhc", layer=2)
        t = case["tokens"]
        if _token_kernel_limit(receipt, case, "mhc", config.hc_mult * config.hidden_size):
            continue
        generator = torch.Generator().manual_seed(plan["seed"] + t)
        hidden = torch.randn(t, config.hc_mult, config.hidden_size, generator=generator, dtype=torch.bfloat16).cuda()
        attn_output = torch.randn(t, config.hidden_size, generator=generator, dtype=torch.bfloat16).cuda()
        ffn_output = torch.randn(t, config.hidden_size, generator=generator, dtype=torch.bfloat16).cuda()
        _, prev_pre, _, _ = layer._hc_mix_and_combine(
            hidden, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, None, layer.post_attention_layernorm
        )
        torch.cuda.synchronize()

        def both_sites(hidden=hidden, attn_output=attn_output, ffn_output=ffn_output, prev_pre=prev_pre):
            _, attn_pre, attn_post, attn_comb = layer._hc_mix_and_combine(
                hidden, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, prev_pre, layer.input_layernorm
            )
            residual = layer.hc_post(attn_output, hidden, attn_post, attn_comb)
            _, _, ffn_post, ffn_comb = layer._hc_mix_and_combine(
                residual,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                attn_pre,
                layer.post_attention_layernorm,
            )
            return layer.hc_post(ffn_output, residual, ffn_post, ffn_comb)

        wraps = [(layer, "_hc_mix_and_combine", entry), (layer, "hc_post", entry)]
        # both sites: each wrapped method runs twice per forward
        _measure_tokens(
            both_sites,
            wraps,
            case,
            plan,
            stream,
            lambda out: finite(out) or (_ for _ in ()).throw(RuntimeError("mHC output non-finite")),
            expected_calls=4,
        )


def collect_engram(config, quant, manifest, plan, cases, stream, token_ids, receipt):
    import torch
    from sglang.srt.layers import engram as native_engram
    from sglang.srt.layers.engram import Engram, EngramHasher, EngramLayout
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    from collector.sglang.dsv41_isolated_runner import initialize_dummy_weights as init_module_weights

    layout = EngramLayout.from_config(config)
    hasher = EngramHasher.from_config(config, layout).cuda()
    hasher.init_history(1, torch.device("cuda"))
    for layer_id in sorted({e["layer"] for e in manifest["entries"] if e["component"] == "engram"}):
        with torch.device("cuda"):
            module = Engram(config, layer_id, layout, quant, prefix=f"model.layers.{layer_id}.engram")
        entry_ctx = _entry(manifest, "context", "engram", layer=layer_id)
        if (
            module.embed.host_table is not None
            or module.embed._shared
            or module.embed.tp_size != manifest["tp_size"]
            or layout.num_embeddings[module.layer_hash_index] != entry_ctx["structure"]["num_embeddings"]
            or tuple(module.wkv.quant_method.quant_config.weight_block_size) != (32, 32)
        ):
            raise RuntimeError("native Engram GPU table/projection differs from the SDK structure")
        if entry_ctx["structure"]["sharding"] != "row":
            raise RuntimeError("SGLang Engram is row-sharded; the SDK structure says otherwise")
        receipt.setdefault("weight_initialization", {})[f"engram.{layer_id}"] = init_module_weights(module, plan)
        receipt.setdefault("engram_identity", {})[str(layer_id)] = dict(
            rows=int(module.embed.rows),
            row_start=int(module.embed.row_start),
            wkv=type(module.wkv.quant_method).__name__,
        )
        for case in cases:
            t = case["tokens"]
            if _token_kernel_limit(receipt, case, "engram", config.hc_mult * config.hidden_size):
                continue
            ids = torch.tensor(token_ids[:t], dtype=torch.int64, device="cuda")
            hasher.history.zero_()
            lengths = torch.tensor([t], dtype=torch.int32, device="cuda")
            batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=1,
                input_ids=ids,
                req_pool_indices=torch.tensor([0], dtype=torch.int64, device="cuda"),
                seq_lens=lengths.to(torch.int64),
                out_cache_loc=None,
                seq_lens_sum=t,
                positions=torch.arange(t, dtype=torch.int64, device="cuda"),
                extend_seq_lens=lengths,
                extend_start_loc=torch.tensor([0], dtype=torch.int32, device="cuda"),
            )
            hashes = hasher(ids, batch)[:, module.layer_hash_index, :].contiguous()
            generator = torch.Generator().manual_seed(plan["seed"] + t)
            hidden = torch.randn(
                t, config.hc_mult, config.hidden_size, generator=generator, dtype=torch.bfloat16
            ).cuda()
            entry = _entry(manifest, case["phase"], "engram", layer=layer_id)
            wraps = [
                (module.embed, "_owned_rows", entry),
                (module.wkv, "forward", entry),
                (native_engram, "engram_gate", entry),
            ]
            _measure_tokens(
                lambda module=module, hidden=hidden, hashes=hashes: module(hidden, hashes),
                wraps,
                case,
                plan,
                stream,
                lambda out: finite(out) or (_ for _ in ()).throw(RuntimeError("Engram output non-finite")),
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
        raise RuntimeError("single-node pure TP allocation differs from plan")
    prepare_private_caches(rank, receipt, marker_prefix="sglang-dsv411")
    if os.environ.get("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE") != "env://":
        raise RuntimeError("torchrun requires the native bootstrap env:// override")
    import torch
    import torch.distributed as dist

    if torch.cuda.device_count() != tp:
        raise RuntimeError("visible CUDA device count differs from allocated TP")
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(plan["seed"])
    receipt["allocated_device_witness"] = device_witness(local_rank, plan)
    from sglang.benchmark import one_batch as bench
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_loader.loader import _get_quantization_config
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    package = Path(bench.__file__).resolve().parents[1]
    sources = source_hashes(package)
    receipt["source_hashes"] = sources
    check_pins(plan, sources, args.model_path, manifest, args.runtime_digest)
    receipt.update(
        framework_version=FRAMEWORK_VERSION,
        raw_package_version=importlib.metadata.version("sglang"),
        collector_revision=plan["collector_revision"],
        plan_sha256=contract.sha256_file(args.plan),
        manifest_sha256=contract.sha256_file(args.manifest),
        runtime_digest=args.runtime_digest,
        image_sha256=plan["image_sha256"],
        purpose=plan["purpose"],
        weight_initializer=WEIGHT_INITIALIZER,
        regimes=plan["regimes"],
    )
    pool = plan["pool"]
    needs_attention = "attention_core" in plan["components"]
    server = bench.ServerArgs(
        model_path=str(args.model_path),
        trust_remote_code=True,
        tp_size=tp,
        moe_runner_backend=plan["moe_runner_backend"],
        moe_a2a_backend="none",
        disable_shared_experts_fusion=True,
        disable_custom_all_reduce=True,
        enforce_disable_flashinfer_allreduce_fusion=True,
        load_format="dummy",
        dtype="bfloat16",
        context_length=pool["context_length"],
        max_total_tokens=pool["max_total_tokens"],
        max_running_requests=pool["max_requests"],
        mem_fraction_static=pool["mem_fraction_static"],
        # The SWA pool only has to hold the sliding windows plus one extend's new tokens: the producer
        # slides the windows between chunks like the serving scheduler (slide_windows). The ratio and
        # the full pool size are plan inputs (collector/dsv411/plan.py DEFAULT_POOL) checked against the
        # grid by contract.validate_plan.
        swa_full_tokens_ratio=pool["swa_full_tokens_ratio"],
        cuda_graph_max_bs_decode=pool["max_requests"],
        cuda_graph_backend_prefill="disabled",
        **({} if needs_attention else {"cuda_graph_backend_decode": "disabled"}),
    )
    bench.publish(
        server,
        role="scheduler",
        ranks=bench.SpawnRanks(world_rank=bench.spawn_world_rank(server, tp_rank=rank, pp_rank=0), gpu_id=local_rank),
    )
    bench.initialize_moe_config()
    bench.initialize_fp8_gemm_config()
    bench.initialize_fp4_gemm_config()
    model_config = ModelConfig.from_server_args(server)
    bench.bootstrap.init_parallel_runtime(server_args=server, device="cuda", dist_port=int(os.environ["MASTER_PORT"]))
    bench.bootstrap.init_layer_runtime(model_config=model_config)
    tokenizer = get_tokenizer(str(args.model_path), trust_remote_code=True, tokenizer_backend="huggingface")
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
        own = args.output / f"current-rank-{rank}.case"
        if own.exists():  # archive (never delete): the other rank may not have read it yet
            n = len(list(args.output.glob(f"current-rank-{rank}.attempt-*.case")))
            own.rename(args.output / f"current-rank-{rank}.attempt-{n}.case")
        receipt["resume"] = dict(skipped_cases=len(done), kept_rows=stream.rows, failed=sorted(receipt["failed_cases"]))
    state, intervals = State(), Intervals()
    try:
        attention_cases = [c for c in plan["cases"] if c["kind"] == "attention" and not (done and c["case_id"] in done)]
        token_cases = [
            c
            for c in plan["cases"]
            if c["kind"] == "tokens" and c["case_id"] not in (receipt.get("failed_cases") or {})
        ]
        if attention_cases:
            runner = build_attention_runner(
                bench, server, model_config, local_rank, plan, manifest, receipt, state, intervals
            )
            progress_file = (args.output / f"progress-rank-{rank}.jsonl").open("a")

            def progress(case, started):
                progress_file.write(
                    json.dumps(dict(case_id=case["case_id"], seconds=time.monotonic() - started, rows=stream.rows))
                    + "\n"
                )
                progress_file.flush()

            # context groups (one seeded prefix, every query length) in seed order, then generation
            ordered = sorted(
                attention_cases,
                key=lambda c: (c["phase"] != "context", c["batch_size"], c["past_kv"], c["query"]),
            )
            for _, group in itertools.groupby(ordered, key=contract.seed_group):
                group = list(group)
                if group[0]["phase"] == "context":
                    run_context_group(
                        runner, bench, token_ids, group, plan, state, intervals, stream, manifest, receipt, progress
                    )
                    continue
                for case in group:
                    started = time.monotonic()
                    try:
                        run_generation_case(
                            runner, bench, token_ids, case, plan, state, intervals, stream, manifest, receipt
                        )
                    except (AttributeError, RuntimeError) as error:
                        if not _pool_exhausted(error):
                            raise
                        receipt.setdefault("failed_cases", {})[case["case_id"]] = f"{_capacity_failure(error)}: {error}"
                        receipt.pop("failed_case", None)
                        mark_case(stream.output, stream.rank, None)
                        intervals.active = False
                        runner.clear()
                        torch.cuda.empty_cache()
                        continue
                    progress(case, started)
                    receipt.pop("failed_case", None)
                    mark_case(stream.output, stream.rank, None)
            intervals.restore()
            del runner
            gc.collect()
            torch.cuda.empty_cache()
        if token_cases:
            quant = _get_quantization_config(model_config, LoadConfig(load_format="dummy"))
            config = model_config.hf_text_config
            if "shared_linear" in plan["components"]:
                from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

                from collector.sglang.dsv41_isolated_runner import initialize_dummy_weights as init_module_weights

                with torch.device("cuda"):
                    moe = DeepseekV2MoE(config, 2, quant, prefix="model.layers.2.mlp", is_deepseek_v4=True)
                receipt.setdefault("weight_initialization", {})["moe_layer2"] = init_module_weights(moe, plan)
                collect_shared_linear(moe, manifest, plan, token_cases, stream, receipt)
                del moe
                gc.collect()
                torch.cuda.empty_cache()
            if "mhc" in plan["components"]:
                collect_mhc(config, quant, manifest, plan, token_cases, stream, receipt)
            if "engram" in plan["components"]:
                collect_engram(config, quant, manifest, plan, token_cases, stream, token_ids, receipt)
    finally:
        stream.close()
    receipt.update(
        state="complete_pending_admission", memory_peak_allocated=torch.cuda.max_memory_allocated(), rows=stream.rows
    )
    dist.barrier()
    dist.destroy_process_group()


def admit(args):
    rows, provenance = contract.aggregate_run(
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
            state="failed_preserved",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        write_receipt(path, receipt)


if __name__ == "__main__":
    main()
