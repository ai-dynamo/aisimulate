# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time one real GLM-5.3-Flash sparse-MLA layer standalone on SGLang 0.5.20.

Framework integration: sgl-project/sglang v0.5.20
(94602c9c2b7cbdb8efd5c52802dac6a1c180089e), python/sglang/benchmark/one_batch.py
(Apache-2.0). This original adapter calls the upstream loader, request
allocation and forward helpers (``load_model``, ``prepare_synthetic_inputs_for_
latency_test``, ``extend``/``decode``) instead of copying metadata logic, like
the DeepSeek-V4.1 runner in this directory. See
``collector/README.glm53flash_attention.md``.

Boundary: ``DeepseekV2AttentionMLA.forward`` of layer ``--layer-id``. SGLang
builds it with ``reduce_results=False`` (models/glm5_next.py
Glm5NextDecoderLayer.__init__), and the attention-output all-reduce happens in
``MHCLayerCommunicator.prepare_mlp`` (layers/communicator_mhc.py), so the
measured call contains no collective. The lazily computed q/kv latent
(``AttentionInputs.fetch_qkv_latent``, layers/communicator.py) is recreated
before every repetition so the fused q_a/kv_a projection is always measured.

Graph-mode prefill (revision 2): serving captures prefill with SGLang's
breakable CUDA graph (``--cuda-graph-backend-prefill breakable``;
model_executor/runner/prefill_cuda_graph_runner.py with
runner_backend/breakable_cuda_graph_backend.py). In that mode the absorbed MLA
method is pinned (models/deepseek_common/attention_backend_handler.py), the
pooled-key indexer and the MLA attention core run as eager graph breaks that
read the live batch (layers/attention/dsa/kpool_prefill_cuda_graph.py,
models/deepseek_common/attention_forward_methods/forward_mla.py
``bcg_mla_bmm_then_unified_attention``) and every other kernel of the module
is replayed from captured segments. The probe records the module's
capture-time arguments for every prefill bucket while the framework captures,
witnesses the real step's framework replay, then captures the module alone
with the same BreakableCUDAGraphCapture under the step's live forward and
TcPiecewise contexts and replays it.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from collector.glm53flash_attention_contract import (
    GRAPH_PREFILL,
    PHASES,
    RUNTIME_VERSIONS,
    build_plan,
    geometry,
    representative_layer_is_uniform,
)
from collector.glm53flash_attention_runtime import (
    EventTimer,
    RawWriter,
    config_sha256,
    corpus_tokens,
    package_source_sha256,
    request_tokens,
    target_id,
)


def _quant_name(module) -> str:
    method = getattr(module, "quant_method", None)
    return "none" if method is None else type(method).__name__


class AttentionProbe:
    """Owns the timing wrapper around one loaded attention module."""

    def __init__(self, torch, attention, layer_id: int, options, writer: RawWriter):
        self.torch = torch
        self.attention = attention
        self.layer_id = layer_id
        self.options = options
        self.writer = writer
        self.original = attention.forward
        self.captured = {}
        self.prefill_captured = {}
        self.sentinel = self.sentinel_input = None
        self.replay_witness = None
        self.armed = None
        self.done = None
        self.pool = None
        attention.forward = self._forward

    # -- capture-time recording (framework decode CUDA graph capture) --------
    def _record_capture(self, args, kwargs):
        from sglang.srt.model_executor.forward_context import get_forward_context
        from sglang.srt.runtime_context import get_forward

        forward_batch = kwargs["forward_batch"]
        zero_allocator = kwargs["zero_allocator"]
        self.captured[int(forward_batch.batch_size)] = SimpleNamespace(
            args=args,
            kwargs=dict(kwargs, forward_batch=copy.copy(forward_batch)),
            attn_inputs=copy.copy(get_forward().attn_inputs),
            context=get_forward_context(),
            zero_allocator=zero_allocator,
            zero_pointer=zero_allocator._pointer,
        )

    def _record_prefill_capture(self, args, kwargs):
        from sglang.srt.model_executor.forward_context import get_forward_context
        from sglang.srt.runtime_context import get_forward

        forward_batch = kwargs["forward_batch"]
        if not forward_batch.forward_mode.is_extend():
            raise RuntimeError("breakable capture of a non-extend batch reached the probe")
        zero_allocator = kwargs["zero_allocator"]
        self.prefill_captured[int(kwargs["hidden_states"].shape[0])] = SimpleNamespace(
            args=args,
            kwargs=dict(kwargs, forward_batch=copy.copy(forward_batch)),
            attn_inputs=copy.copy(get_forward().attn_inputs),
            context=get_forward_context(),
            zero_allocator=zero_allocator,
            zero_pointer=zero_allocator._pointer,
        )

    def _fresh_inputs(self, attn_inputs, forward_batch, zero_allocator, zero_pointer):
        from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context

        zero_allocator._pointer = zero_pointer
        get_attn_tp_context().set_attn_inputs(
            AttentionInputs(
                attn_inputs.hidden_states_local,
                forward_batch,
                attn_inputs.qkv_latent_func,
                is_pre_gathered=attn_inputs.is_pre_gathered,
            )
        )

    def _source(self, forward_batch) -> str:
        attn = self.attention
        method = attn.dispatch_attn_forward_method(forward_batch)
        parts = [
            f"sglang{RUNTIME_VERSIONS['sglang']}",
            type(attn).__name__,
            type(attn.indexer).__name__,
            f"qkv_a={_quant_name(attn.fused_qkv_a_proj_with_mqa)}",
            f"q_b={_quant_name(attn.q_b_proj)}",
            f"kv_b={_quant_name(attn.kv_b_proj)}",
            f"o={_quant_name(attn.o_proj)}",
            f"attn_backend={attn.current_attention_backend}",
            f"method={getattr(method, 'name', method)}",
        ]
        return "|".join(parts)

    # -- the wrapped forward -------------------------------------------------
    def _forward(self, *args, **kwargs):
        torch = self.torch
        from sglang.srt.model_executor.runner import get_is_capture_mode
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
            _current_capture_var,
        )

        if _current_capture_var.get() is not None:
            # Framework breakable prefill capture of one token bucket.
            self._record_prefill_capture(args, kwargs)
            return self.original(*args, **kwargs)
        if get_is_capture_mode() and torch.cuda.is_current_stream_capturing():
            self._record_capture(args, kwargs)
            return self.original(*args, **kwargs)
        target = self.armed
        if target is None or target["phase"] != "context":
            return self.original(*args, **kwargs)
        forward_batch = kwargs["forward_batch"]
        observed = {
            "extend": bool(forward_batch.forward_mode.is_extend()),
            "batch": int(forward_batch.batch_size),
            "prefix": sorted(set(forward_batch.extend_prefix_lens_cpu)),
            "query": sorted(set(forward_batch.extend_seq_lens_cpu)),
        }
        expected = {
            "extend": True,
            "batch": target["batch_size"],
            "prefix": [target["prefix"]],
            "query": [target["x"]],
        }
        if observed != expected:
            raise RuntimeError(f"framework batch {observed} differs from planned target {expected}")
        from sglang.srt.runtime_context import get_forward

        attn_inputs = get_forward().attn_inputs
        zero_allocator = kwargs["zero_allocator"]
        pointer = zero_allocator._pointer
        # The dispatched forward method differs by regime (dense MHA vs sparse MLA).
        source = self._source(forward_batch)
        timer = EventTimer(torch)
        outputs = []
        for _ in range(self.options.warmup + self.options.iterations):
            self._fresh_inputs(attn_inputs, forward_batch, zero_allocator, pointer)
            result = timer(lambda: self.original(*args, **kwargs))
            outputs.append(result[0] if isinstance(result, tuple) else result)
        latencies = timer.read()
        host = [round(v, 4) for v in timer.host_ms[self.options.warmup :]]
        finite = all(bool(torch.isfinite(o).all().item()) for o in outputs)
        drift = float((outputs[-1].float() - outputs[0].float()).abs().max().item())
        self.writer.samples(
            target,
            latencies,
            self.options.warmup,
            source,
            {"finite": finite, "repeat_max_abs_diff": drift, "host_enqueue_ms": host},
        )
        if not finite:
            raise RuntimeError(f"nonfinite attention output for {target['target_id']}")
        self.done = target["target_id"]
        self._fresh_inputs(attn_inputs, forward_batch, zero_allocator, pointer)
        return self.original(*args, **kwargs)

    # -- graph-mode prefill: the framework's breakable capture, module only ---
    def measure_prefill(self, target: dict, witness) -> None:
        torch = self.torch
        from sglang.srt.model_executor.forward_context import forward_context
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
            enable_breakable_cuda_graph,
        )
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
            context_manager as tc_context,
        )

        tokens = target["batch_size"] * target["x"]
        record = self.prefill_captured.get(witness.size)
        if record is None or witness.size < tokens:
            raise RuntimeError(f"no framework prefill capture of bucket {witness.size} for {tokens} tokens")
        forward_batch = record.kwargs["forward_batch"]

        def call():
            self._fresh_inputs(record.attn_inputs, forward_batch, record.zero_allocator, record.zero_pointer)
            return self.original(*record.args, **record.kwargs)

        # One private pool for every prefill module graph. A permanent sentinel
        # graph holds the pool's use count above zero (a dead pool handle trips
        # the caching allocator's assertion), so the previous module graph can
        # be dropped before this capture and its blocks reused, instead of
        # holding two long-context graphs at once. (Releasing cached blocks
        # with empty_cache exposed an illegal address in a later framework BCG
        # replay, so memory is never returned to the device.)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
            self.sentinel_input = torch.zeros(1, device="cuda")
            self.sentinel = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.sentinel, pool=self.pool):
                self.sentinel_input.add_(1)
        pool = self.pool
        # BreakableCudaGraphBackend.replay_session/execute: the BCG flag, the
        # attention-backend forward context and the TcPiecewise context whose
        # forward_batch the eager breaks read (the real step's static batch).
        previous = tc_context._tc_piecewise_forward_context
        with enable_breakable_cuda_graph(), forward_context(witness.forward_context):
            tc_context._tc_piecewise_forward_context = witness.tc_context
            try:
                source = self._source(forward_batch)
                # BreakableCudaGraphBackend.capture_one: two eager warmups, then capture.
                for _ in range(2):
                    call()
                torch.cuda.synchronize()
                graph = BreakableCUDAGraph()
                with BreakableCUDAGraphCapture(cuda_graph=graph, pool=pool, stream=torch.cuda.Stream()):
                    output = call()
                torch.cuda.synchronize()
                timer = EventTimer(torch)
                for _ in range(self.options.warmup + self.options.iterations):
                    timer(graph.replay)
                latencies = timer.read()
            finally:
                tc_context._tc_piecewise_forward_context = previous
        host = [round(v, 4) for v in timer.host_ms[self.options.warmup :]]
        result = output[0] if isinstance(output, tuple) else output
        finite = bool(torch.isfinite(result[:tokens]).all().item())
        self.writer.samples(
            target,
            latencies,
            self.options.warmup,
            source,
            {
                "finite": finite,
                "host_enqueue_ms": host,
                "padded_tokens": witness.size,
                "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
                "segments": len(graph._segments),
                "eager_breaks": len(graph._break_fns),
            },
            timing_method=GRAPH_PREFILL,
        )
        del graph, output, result
        from sglang.srt.layers.communicator import get_attn_tp_context

        get_attn_tp_context().clear_attn_inputs()
        if not finite:
            raise RuntimeError(f"nonfinite prefill attention output for {target['target_id']}")

    # -- decode: replay a module graph built from the framework capture -------
    def measure_decode(self, target: dict) -> None:
        torch = self.torch
        from sglang.srt.model_executor.forward_context import forward_context
        from sglang.srt.model_executor.runner import model_capture_mode

        batch = target["batch_size"]
        record = self.captured.get(batch)
        if record is None:
            raise RuntimeError(f"framework captured no decode graph containing layer {self.layer_id} at bs={batch}")
        forward_batch = record.kwargs["forward_batch"]

        def call():
            self._fresh_inputs(record.attn_inputs, forward_batch, record.zero_allocator, record.zero_pointer)
            return self.original(*record.args, **record.kwargs)

        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        with forward_context(record.context), model_capture_mode():
            source = self._source(forward_batch)
            # The framework runs the captured callable eagerly before capture.
            for _ in range(2):
                call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                output = call()
        timer = EventTimer(torch)
        for _ in range(self.options.warmup + self.options.iterations):
            timer(graph.replay)
        latencies = timer.read()
        host = [round(v, 4) for v in timer.host_ms[self.options.warmup :]]
        result = output[0] if isinstance(output, tuple) else output
        finite = bool(torch.isfinite(result[:batch]).all().item())
        self.writer.samples(target, latencies, self.options.warmup, source, {"finite": finite, "host_enqueue_ms": host})
        del graph
        from sglang.srt.layers.communicator import get_attn_tp_context

        get_attn_tp_context().clear_attn_inputs()
        if not finite:
            raise RuntimeError(f"nonfinite decode attention output for {target['target_id']}")


def _extend_all(runner, bench, reqs, tokens, start: int, end: int):
    """Extend every request from ``start`` to ``end`` real tokens.

    Mirrors one_batch.prepare_extend_inputs_for_correctness_test: the prefix
    is the request's own allocated slots from req_to_token, then the native
    ``extend`` helper allocates and runs the continuation.
    """
    pool = runner.torch_runner.req_to_token_pool.req_to_token
    for req, request_tokens_ in zip(reqs, tokens, strict=True):
        if len(req.full_untruncated_fill_ids) != start:
            raise RuntimeError("request fill ids do not match the computed prefix")
        req.full_untruncated_fill_ids.extend(request_tokens_[start:end])
        req.prefix_indices = pool[req.kv.req_pool_idx, :start].to(req.prefix_indices.dtype)
        req.logprob_start_len = -1
        req.set_extend_range(start, end)
    return runner.extend(reqs)


def run_set(runner, bench, probe, request_set, tokens_all, options, progress, graph_prefill=False):
    torch = probe.torch
    batch = request_set["batch_size"]
    chunk = request_set["seed_chunk"]
    phase = request_set["phase"]
    last = request_set["targets"][-1] + (request_set["query"] if phase == "context" else 0)
    tokens = [request_tokens(tokens_all, r, last) for r in range(batch)]
    runner.clear()
    reqs = None
    computed = 0
    result = None

    def advance(end):
        nonlocal reqs, computed, result
        while computed < end:
            step = min(chunk, end - computed)
            if reqs is None:
                reqs = bench.prepare_synthetic_inputs_for_latency_test(batch, step, [t[:step] for t in tokens])
                result = runner.extend(reqs)
            else:
                result = _extend_all(runner, bench, reqs, tokens, computed, computed + step)
            computed += step

    for value in request_set["targets"]:
        started = time.monotonic()
        if phase == "context":
            query = request_set["query"]
            target = {
                "phase": "context",
                "batch_size": batch,
                "prefix": value,
                "x": query,
                "target_id": target_id("context", batch, value, query),
            }
            advance(value)
            if graph_prefill:
                prefill_runner = runner.torch_runner.prefill_cuda_graph_runner
                before = prefill_runner.w4_replays
                if reqs is None:
                    reqs = bench.prepare_synthetic_inputs_for_latency_test(batch, query, [t[:query] for t in tokens])
                    result = runner.extend(reqs)
                else:
                    result = _extend_all(runner, bench, reqs, tokens, computed, computed + query)
                computed += query
                if prefill_runner.w4_replays != before + 1:
                    raise RuntimeError(f"{target['target_id']} did not replay the framework prefill CUDA graph")
                witness = prefill_runner.w4_witness
                if witness.raw_tokens != batch * query:
                    raise RuntimeError(f"framework replayed {witness.raw_tokens} tokens, planned {batch * query}")
                probe.measure_prefill(target, witness)
                progress(
                    {
                        "set_id": request_set["set_id"],
                        "target_id": target["target_id"],
                        "elapsed_seconds": time.monotonic() - started,
                        "status": "passed",
                        "padded_tokens": witness.size,
                    }
                )
                continue
            probe.armed, probe.done = target, None
            try:
                if reqs is None:
                    reqs = bench.prepare_synthetic_inputs_for_latency_test(batch, query, [t[:query] for t in tokens])
                    result = runner.extend(reqs)
                else:
                    result = _extend_all(runner, bench, reqs, tokens, computed, computed + query)
            finally:
                probe.armed = None
            computed += query
            if probe.done != target["target_id"]:
                raise RuntimeError(f"layer {probe.layer_id} was not called for {target['target_id']}")
        else:
            target = {
                "phase": "generation",
                "batch_size": batch,
                "prefix": 0,
                "x": value,
                "target_id": target_id("generation", batch, 0, value),
            }
            if computed >= value:
                raise RuntimeError("decode targets must increase")
            advance(value - 1)
            next_ids, _, schedule_batch = result
            graph_runner = runner.torch_runner.decode_cuda_graph_runner
            before = graph_runner.w4_executions
            new_ids, _ = runner.decode(next_ids, schedule_batch)
            if graph_runner.w4_executions != before + 1 or graph_runner.w4_last_bs != batch:
                raise RuntimeError("real decode did not replay the framework decode CUDA graph at this batch")
            if [int(v) for v in schedule_batch.seq_lens.tolist()] != [value] * batch:
                raise RuntimeError("decode batch does not read the planned absolute sequence length")
            probe.measure_decode(target)
            # The decoded token stays part of each real sequence.
            decoded = next_ids.tolist()
            for req, request_tokens_, token in zip(reqs, tokens, decoded, strict=True):
                req.full_untruncated_fill_ids.append(int(token))
                request_tokens_[value - 1] = int(token)
            computed = value
            del new_ids
        progress(
            {
                "set_id": request_set["set_id"],
                "target_id": target["target_id"],
                "elapsed_seconds": time.monotonic() - started,
                "status": "passed",
            }
        )
    torch.cuda.synchronize()


def run_worker(server_args, port_args, bench_args, gpu_id, tp_rank):
    import torch
    from sglang.benchmark import one_batch as bench
    from sglang.srt.model_executor.model_runner import ModelRunner

    options = bench_args.glm53_options
    manifest = json.loads(Path(options.manifest).read_text())
    output = Path(options.output)
    bench.publish(server_args, role="scheduler")
    # one_batch.latency_test (python/sglang/benchmark/one_batch.py:902-906)
    # seeds these process-local dispatch configs in every spawned rank.
    bench.initialize_moe_config()
    bench.initialize_fp8_gemm_config()
    bench.initialize_fp4_gemm_config()
    bench.configure_logger(server_args, prefix=f" TP{tp_rank}")

    key_base = dict(manifest["geometry"])
    writer = RawWriter(output, tp_rank, key_base, {})
    graph_prefill = manifest.get("prefill_execution") == "framework_breakable_cuda_graph"
    phases = set(manifest.get("phases", PHASES))
    probes = {}
    original_init_graphs = ModelRunner.init_cuda_graphs

    def init_with_probe(model_runner, *args, **kwargs):
        # Install the probe before the framework captures its decode graphs so
        # the capture-time arguments of the measured module are recorded.
        layers = {
            int(m.layer_id): m for m in model_runner.model.modules() if type(m).__name__ == "Glm5NextDecoderLayer"
        }
        if len(layers) != 45:
            raise RuntimeError(f"expected 45 GLM decoder layers, found {len(layers)}")
        attention = layers[manifest["layer_id"]].self_attn
        _validate_attention(attention, manifest)
        probes["probe"] = AttentionProbe(torch, attention, manifest["layer_id"], options, writer)
        result = original_init_graphs(model_runner, *args, **kwargs)
        graph_runner = model_runner.decode_cuda_graph_runner
        if graph_runner is None:
            raise RuntimeError("serving decode CUDA graphs are disabled")
        graph_runner.w4_executions, graph_runner.w4_last_bs = 0, None
        execute = graph_runner.execute

        def counted(forward_batch, *a, **k):
            graph_runner.w4_executions += 1
            graph_runner.w4_last_bs = int(forward_batch.batch_size)
            return execute(forward_batch, *a, **k)

        graph_runner.execute = counted
        if graph_prefill:
            from sglang.srt.model_executor.forward_context import get_forward_context
            from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
                BreakableCudaGraphBackend,
            )
            from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
                get_tc_piecewise_forward_context,
            )

            prefill_runner = model_runner.prefill_cuda_graph_runner
            if prefill_runner is None or not isinstance(prefill_runner.backend, BreakableCudaGraphBackend):
                raise RuntimeError("serving breakable prefill CUDA graphs are not active")
            captured = sorted(probes["probe"].prefill_captured)
            if captured != sorted(prefill_runner.capture_num_tokens):
                raise RuntimeError(f"probe recorded prefill buckets {captured}")
            prefill_runner.w4_replays, prefill_runner.w4_witness = 0, None
            replay = prefill_runner.backend.replay

            def witnessed(shape_key, static_forward_batch, **k):
                # PrefillCudaGraphRunner.execute -> _execute_body_capture runs this
                # inside its live forward and TcPiecewise contexts.
                tc = get_tc_piecewise_forward_context()
                prefill_runner.w4_replays += 1
                prefill_runner.w4_witness = SimpleNamespace(
                    size=int(shape_key.size),
                    raw_tokens=int(tc.raw_num_tokens if tc.raw_num_tokens is not None else shape_key.size),
                    tc_context=tc,
                    forward_context=get_forward_context(),
                )
                return replay(shape_key, static_forward_batch, **k)

            prefill_runner.backend.replay = witnessed
        return result

    ModelRunner.init_cuda_graphs = init_with_probe
    try:
        runner, tokenizer = bench.load_model(server_args, port_args, gpu_id, tp_rank)
    finally:
        ModelRunner.init_cuda_graphs = original_init_graphs
    probe = probes["probe"]
    missing = sorted(set(range(1, 33)) - set(probe.captured))
    if missing:
        raise RuntimeError(f"decode capture did not record the probe for bs {missing}")

    import sglang

    source_sha, sources = package_source_sha256(Path(sglang.__file__).resolve().parent)
    writer.provenance = {
        "framework_version": manifest["framework_version"],
        "source_sha256": source_sha,
        "config_sha256": config_sha256(Path(server_args.model_path)),
        "checkpoint_revision": manifest["checkpoint_revision"],
        "runtime_digest": manifest["runtime_digest"],
        "layer_id": manifest["layer_id"],
    }
    plan = manifest["plan"]
    longest = max(s["targets"][-1] + (s["query"] if s["phase"] == "context" else 0) for s in plan["sets"])
    tokens, corpus = corpus_tokens(tokenizer, Path(options.corpus), longest + 32 * 4099)
    if tp_rank == 0:
        (output / "source_hashes.json").write_text(json.dumps(sources, sort_keys=True))
        (output / "input_provenance.json").write_text(json.dumps({**corpus, **writer.provenance}, sort_keys=True))
        (output / "resolved_server_args.json").write_text(
            json.dumps({k: str(v) for k, v in sorted(vars(server_args).items())})
        )

    def progress(payload):
        with (output / f"progress-rank-{tp_rank}.jsonl").open("a") as stream:
            stream.write(json.dumps({"tp_rank": tp_rank, **payload}) + "\n")

    for request_set in plan["sets"]:
        if options.only_sets and request_set["set_id"] not in options.only_sets:
            continue
        if request_set["phase"] not in phases:
            continue
        try:
            run_set(runner, bench, probe, request_set, tokens, plan, progress, graph_prefill=graph_prefill)
        except BaseException as error:
            progress(
                {
                    "set_id": request_set["set_id"],
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            raise
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
    if tp_rank == 0:
        (output / "COMPLETE").write_text("glm53flash attention collection completed\n")


def _validate_attention(attention, manifest) -> None:
    """Refuse to time a module whose loaded identity differs from the key."""
    expected = manifest["geometry"]
    indexer = attention.indexer
    checks = {
        "class": (type(attention).__name__, "DeepseekV2AttentionMLA"),
        "indexer": (type(indexer).__name__, "IndexerKPool"),
        "num_local_heads": (attention.num_local_heads, expected["num_heads"]),
        "qk_nope_head_dim": (attention.qk_nope_head_dim, expected["head_dim"]),
        "qk_rope_head_dim": (attention.qk_rope_head_dim, 0),
        "q_lora_rank": (attention.q_lora_rank, expected["q_lora_rank"]),
        "kv_lora_rank": (attention.kv_lora_rank, expected["kv_lora_rank"]),
        "v_head_dim": (attention.v_head_dim, expected["value_head_dim"]),
        "index_topk": (indexer.index_topk, expected["index_topk"]),
        "index_kpool": (indexer.index_kpool, expected["index_pool"]),
        "o_proj.reduce_results": (attention.o_proj.reduce_results, False),
        "skip_topk": (bool(attention.skip_topk), False),
    }
    for name, (actual, wanted) in checks.items():
        if actual != wanted:
            raise RuntimeError(f"loaded attention {name}={actual!r}, contract expects {wanted!r}")
    fp8 = expected["projection_quant_mode"] == "fp8_block"
    for name in ("fused_qkv_a_proj_with_mqa", "q_b_proj", "o_proj"):
        method = getattr(attention, name).quant_method
        is_fp8 = "Fp8" in type(method).__name__
        if is_fp8 != fp8:
            raise RuntimeError(
                f"{name} uses {type(method).__name__}, contract expects {expected['projection_quant_mode']}"
            )
        if fp8 and list(method.quant_config.weight_block_size) != [128, 128]:
            raise RuntimeError(f"{name} is not FP8 block-128")
    if "Unquantized" not in type(attention.kv_b_proj.quant_method).__name__:
        raise RuntimeError("kv_b_proj must stay BF16 in both GLM-5.3-Flash checkpoints")


def main():
    from sglang.benchmark import one_batch as bench

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--only-sets", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true", help="resolve everything, then exit before CUDA work")
    options, rest = parser.parse_known_args()
    manifest = json.loads(Path(options.manifest).read_text())
    plan = manifest["plan"]
    if plan != build_plan(manifest["sweep"]):
        raise ValueError("manifest plan differs from its frozen sweep")
    if manifest.get("only_sets") is not None:
        # A split attempt measures exactly the manifest's set selection.
        if options.only_sets and sorted(options.only_sets) != manifest["only_sets"]:
            raise ValueError("--only-sets differs from the manifest selection")
        options.only_sets = manifest["only_sets"]
    options.warmup, options.iterations = plan["warmup"], plan["iterations"]
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("rank-*.jsonl")) or (output / "COMPLETE").exists():
        raise RuntimeError("output has prior raw records; use a fresh attempt directory")
    native = argparse.ArgumentParser()
    bench.ServerArgs.add_cli_args(native)
    bench.BenchArgs.add_cli_args(native)
    args = native.parse_args(rest)
    server_args, bench_args = bench.ServerArgs.from_cli_args(args), bench.BenchArgs.from_cli_args(args)
    from importlib.metadata import version

    installed = version("sglang")
    if installed != RUNTIME_VERSIONS["sglang"] or manifest["framework_version"] != installed:
        raise RuntimeError(f"sglang {installed} is not the pinned {RUNTIME_VERSIONS['sglang']}")
    config = json.loads((Path(server_args.model_path) / "config.json").read_text())
    expected = geometry(config, "sglang", manifest["geometry"]["checkpoint_format"], manifest["geometry"]["tp_size"])
    if expected != manifest["geometry"] or server_args.tp_size != expected["tp_size"]:
        raise ValueError("checkpoint/TP geometry differs from the manifest")
    representative_layer_is_uniform(config, manifest["layer_id"], expected["checkpoint_format"])
    if manifest.get("prefill_execution") == "framework_breakable_cuda_graph":
        graph = manifest["serving_graph"]["sglang_args"].split()
        if any(flag not in rest for flag in graph[::2]) or [rest[rest.index(f) + 1] for f in graph[::2]] != graph[1::2]:
            raise ValueError(f"serving prefill graph arguments {graph} are missing from {rest}")
    if options.dry_run:
        sets = [
            s["set_id"]
            for s in plan["sets"]
            if s["phase"] in set(manifest.get("phases", PHASES))
            and (not options.only_sets or s["set_id"] in options.only_sets)
        ]
        print(
            json.dumps(
                {
                    "dry_run": "ok",
                    "framework": installed,
                    "geometry": expected,
                    "sets": len(sets),
                    "prefill_execution": manifest.get("prefill_execution", "eager"),
                    "native_cli_args": rest,
                    "cuda_graph_backend_prefill": getattr(server_args, "cuda_graph_backend_prefill", None),
                    "cuda_graph_max_bs_prefill": getattr(server_args, "cuda_graph_max_bs_prefill", None),
                },
                sort_keys=True,
            )
        )
        return
    # one_batch.main folds max(batch_size) into the decode graph max_bs; the
    # serving deployment captures bs 1..32.
    bench_args.batch_size = (32,)
    bench_args.glm53_options = options
    (output / "execution-contract.json").write_text(
        json.dumps({"native_cli_args": rest, "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True)
    )
    bench.latency_test = run_worker
    bench.main(server_args, bench_args)
    if not (output / "COMPLETE").exists():
        raise RuntimeError("native worker failed; inspect preserved rank logs")


if __name__ == "__main__":
    main()
