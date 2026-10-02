# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-side probe for the vLLM GLM-5.3-Flash attention collector.

Loaded in every vLLM worker through vLLM's own general-plugin entry point
(``vllm/plugins/__init__.py`` ``load_general_plugins``; called by
``v1/worker/worker_base.py`` before the model is built). It is inert unless
``GLM53_W4_MANIFEST`` is set by ``glm53flash_attention_runner``.

Boundary: ``Glm5NextMLAAttention.forward`` (models/glm5next/nvidia/attention.py)
of one layer, i.e. the MLA wrapper including the IndexPool indexer and
``o_proj``. vLLM's ``o_proj`` is a ``RowParallelLinear`` whose
``reduce_results`` flag owns the attention-output all-reduce; the model itself
turns that flag off for sequence parallelism (models/glm5next/nvidia/model.py,
Glm5NextDecoderLayer.__init__). Measured repetitions run with the flag off so
the collective is excluded; the real forward call restores it.

Graph-mode prefill (revision 2): serving runs prefill steps through vLLM's
breakable PIECEWISE graphs (v1/worker/gpu/cudagraph_utils.py run_pw_graph ->
compilation/breakable_cudagraph.py BreakableCUDAGraphWrapper). The custom ops
decorated with ``eager_break_during_capture`` -- the IndexPool indexer
(model_executor/layers/sparse_attn_indexer_kpool.py) and the MLA attention op
(model_executor/layers/attention/mla_attention.py) -- run eagerly against the
step's forward context; every other kernel of the module replays from captured
segments. The probe records the module's capture-time inputs per PIECEWISE
size, witnesses the real step's run_pw_graph, then captures the module alone
with the same BreakableCUDAGraphCapture under that step's forward context and
replays it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

STATE = SimpleNamespace(
    probe=None,
    writer=None,
    options=None,
    armed=None,
    done=None,
    full_replays=0,
    last_full_tokens=None,
    pw_replays=0,
    last_pw=None,
    error=None,
)


def register() -> None:
    """vLLM general-plugin entry point (runs in every vLLM process).

    vLLM 0.30.0 workers use the V2 model runner (v1/worker/gpu_worker.py
    ``use_v2_model_runner``; v1/worker/gpu/model_runner.py). Its
    ``capture_model`` drives ``ModelCudaGraphManager.capture`` (FULL decode
    graphs), so the probe is installed immediately before it.
    """
    if not os.environ.get("GLM53_W4_MANIFEST"):
        return
    from vllm.v1.worker.gpu import model_runner as runner_module

    runner_class = runner_module.GPUModelRunner
    if getattr(runner_class, "_glm53_w4_patched", False):
        return
    original_capture = runner_class.capture_model

    def capture_model(self, *args, **kwargs):
        if kwargs.get("profile_only"):
            return original_capture(self, *args, **kwargs)
        install_probe(self)
        return original_capture(self, *args, **kwargs)

    runner_class.capture_model = capture_model
    runner_class._glm53_w4_patched = True


def _layer(model_runner, layer_id: int):
    for module in model_runner.model.modules():
        if type(module).__name__ == "Glm5NextDecoderLayer" and module.layer_idx == layer_id:
            return module
    raise RuntimeError(f"GLM decoder layer {layer_id} is not loaded")


def install_probe(model_runner) -> None:
    import torch

    manifest = json.loads(Path(os.environ["GLM53_W4_MANIFEST"]).read_text())
    layer = _layer(model_runner, manifest["layer_id"])
    attention = layer.self_attn
    validate_attention(attention, manifest, model_runner)
    if STATE.probe is None:
        STATE.probe = Probe(torch, attention, model_runner, manifest)
        manager = model_runner.cudagraph_manager
        replay = manager.run_fullgraph

        def run_fullgraph(desc, *args, **kwargs):
            # Witness of the framework's own FULL decode replay.
            STATE.full_replays += 1
            STATE.last_full_tokens = int(desc.num_tokens)
            return replay(desc, *args, **kwargs)

        manager.run_fullgraph = run_fullgraph
        run_pw = manager.run_pw_graph

        def run_pw_graph(model, model_inputs):
            # Witness of the framework's PIECEWISE (breakable) replay, captured
            # inside execute_model's live set_forward_context.
            from vllm.forward_context import get_forward_context

            context = get_forward_context()
            STATE.pw_replays += 1
            STATE.last_pw = SimpleNamespace(
                tokens=int(context.batch_descriptor.num_tokens), mode=context.cudagraph_runtime_mode, context=context
            )
            return run_pw(model, model_inputs)

        manager.run_pw_graph = run_pw_graph


def validate_attention(attention, manifest, model_runner) -> None:
    expected = manifest["geometry"]
    indexer = attention.indexer
    checks = {
        "class": (type(attention).__name__, "Glm5NextMLAAttention"),
        "indexer": (type(indexer).__name__, "Indexer"),
        "num_local_heads": (attention.num_local_heads, expected["num_heads"]),
        "qk_nope_head_dim": (attention.qk_nope_head_dim, expected["head_dim"]),
        "qk_rope_head_dim": (attention.qk_rope_head_dim, 0),
        "q_lora_rank": (attention.q_lora_rank, expected["q_lora_rank"]),
        "kv_lora_rank": (attention.kv_lora_rank, expected["kv_lora_rank"]),
        "v_head_dim": (attention.v_head_dim, expected["value_head_dim"]),
        "index_n_heads": (indexer.n_head, expected["index_n_heads"]),
        "index_head_dim": (indexer.head_dim, expected["index_head_dim"]),
        "index_topk": (indexer.topk_tokens, expected["index_topk"]),
        "index_kpool": (indexer.index_kpool, expected["index_pool"]),
        "o_proj.reduce_results": (bool(attention.o_proj.reduce_results), expected["tp_size"] > 1),
        "cudagraph_mode": (
            model_runner.vllm_config.compilation_config.cudagraph_mode.name,
            "FULL_AND_PIECEWISE"
            if manifest.get("prefill_execution") == "framework_breakable_cuda_graph"
            else "FULL_DECODE_ONLY",
        ),
    }
    for name, (actual, wanted) in checks.items():
        if actual != wanted:
            raise RuntimeError(f"loaded attention {name}={actual!r}, contract expects {wanted!r}")
    # models/glm5next/nvidia/model.py builds MLA with quant_config=None.
    for name in ("fused_qkv_a_proj", "q_b_proj", "kv_b_proj", "o_proj"):
        method = type(getattr(attention, name).quant_method).__name__
        if method != "UnquantizedLinearMethod" or expected["projection_quant_mode"] != "bfloat16":
            raise RuntimeError(f"{name} uses {method}; vLLM GLM MLA projections are BF16")


def kernel_source(attention) -> str:
    impl = attention.mla_attn.mla_attn.impl if hasattr(attention.mla_attn, "mla_attn") else None
    return "|".join(
        [
            f"vllm{os.environ.get('GLM53_W4_FRAMEWORK_VERSION', '')}",
            type(attention).__name__,
            type(attention.indexer.indexer_op).__name__,
            f"projections={type(attention.o_proj.quant_method).__name__}",
            f"mla_impl={type(impl).__name__ if impl is not None else 'unknown'}",
        ]
    )


class Probe:
    def __init__(self, torch, attention, model_runner, manifest):
        self.torch = torch
        self.attention = attention
        self.model_runner = model_runner
        self.manifest = manifest
        self.original = attention.forward
        self.captured = {}
        self.pw_captured = {}
        self.pool = None
        self.source = kernel_source(attention)
        attention.forward = self.forward

    def _metadata(self):
        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        return attn_metadata[self.attention.indexer.k_cache.prefix]

    def forward(self, hidden_states, positions):
        torch = self.torch
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if BreakableCUDAGraphCapture.current() is not None and is_forward_context_available():
            # Framework PIECEWISE (breakable) capture of one token size.
            self.pw_captured[int(hidden_states.shape[0])] = SimpleNamespace(
                hidden_states=hidden_states, positions=positions, context=get_forward_context()
            )
            return self.original(hidden_states, positions)
        if torch.cuda.is_current_stream_capturing() and is_forward_context_available():
            # ModelCudaGraphManager.capture runs the FULL-graph forward under
            # torch.cuda.graph with the capture-time attention metadata.
            self.captured[int(hidden_states.shape[0])] = SimpleNamespace(
                hidden_states=hidden_states, positions=positions, context=get_forward_context()
            )
            return self.original(hidden_states, positions)
        target = STATE.armed
        if target is None or target["phase"] != "context" or not isinstance(get_forward_context().attn_metadata, dict):
            return self.original(hidden_states, positions)
        metadata = self._metadata()
        requests = metadata.num_decodes + metadata.num_prefills
        observed = {
            "requests": requests,
            "tokens": int(hidden_states.shape[0]),
            "seq_lens": sorted(set(int(v) for v in metadata.seq_lens[:requests].tolist())),
        }
        expected = {
            "requests": target["batch_size"],
            "tokens": target["batch_size"] * target["x"],
            "seq_lens": [target["prefix"] + target["x"]],
        }
        if observed != expected:
            STATE.error = f"framework batch {observed} differs from planned target {expected}"
            raise RuntimeError(STATE.error)
        # The indexer may classify short uniform queries as decode rows
        # (v1/attention/backends/mla/indexer.py decode_threshold); record it.
        classification = {"num_decodes": metadata.num_decodes, "num_prefills": metadata.num_prefills}
        from collector.glm53flash_attention_runtime import EventTimer

        options = STATE.options
        o_proj = self.attention.o_proj
        reduce = o_proj.reduce_results
        timer = EventTimer(torch)
        outputs = []
        o_proj.reduce_results = False
        try:
            for _ in range(options["warmup"] + options["iterations"]):
                outputs.append(timer(lambda: self.original(hidden_states, positions)))
        finally:
            o_proj.reduce_results = reduce
        latencies = timer.read()
        host = [round(v, 4) for v in timer.host_ms[STATE.options["warmup"] :]]
        finite = all(bool(torch.isfinite(o).all().item()) for o in outputs)
        drift = float((outputs[-1].float() - outputs[0].float()).abs().max().item())
        STATE.writer.samples(
            target,
            latencies,
            options["warmup"],
            self.source,
            {"finite": finite, "repeat_max_abs_diff": drift, "host_enqueue_ms": host, **classification},
        )
        if not finite:
            STATE.error = f"nonfinite attention output for {target['target_id']}"
            raise RuntimeError(STATE.error)
        STATE.done = target["target_id"]
        return self.original(hidden_states, positions)

    def measure_decode(self, target: dict, replays_before: int) -> dict:
        torch = self.torch
        from vllm.forward_context import override_forward_context

        batch = target["batch_size"]
        padded = STATE.last_full_tokens
        if STATE.full_replays != replays_before + 1 or padded is None or padded < batch:
            raise RuntimeError(
                f"real decode did not replay one FULL graph for batch {batch}: "
                f"{STATE.full_replays - replays_before} replays, last {padded} tokens"
            )
        record = self.captured.get(padded)
        if record is None:
            raise RuntimeError(f"framework captured no FULL decode graph for {padded} tokens")
        options = STATE.options
        o_proj = self.attention.o_proj
        reduce = o_proj.reduce_results
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        o_proj.reduce_results = False
        try:
            with override_forward_context(record.context):
                for _ in range(2):
                    self.original(record.hidden_states, record.positions)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool):
                    output = self.original(record.hidden_states, record.positions)
        finally:
            o_proj.reduce_results = reduce
        from collector.glm53flash_attention_runtime import EventTimer

        timer = EventTimer(torch)
        for _ in range(options["warmup"] + options["iterations"]):
            timer(graph.replay)
        latencies = timer.read()
        host = [round(v, 4) for v in timer.host_ms[STATE.options["warmup"] :]]
        finite = bool(torch.isfinite(output[:batch]).all().item())
        STATE.writer.samples(
            target,
            latencies,
            options["warmup"],
            self.source,
            {"finite": finite, "padded_tokens": padded, "host_enqueue_ms": host},
        )
        del graph
        if not finite:
            raise RuntimeError(f"nonfinite decode attention output for {target['target_id']}")
        return {"padded_tokens": padded}


def _measure_prefill(probe, target: dict, replays_before: int) -> dict:
    torch = probe.torch
    from collector.glm53flash_attention_contract import GRAPH_PREFILL
    from collector.glm53flash_attention_runtime import EventTimer
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import override_forward_context

    tokens = target["batch_size"] * target["x"]
    witness = STATE.last_pw
    if STATE.pw_replays != replays_before + 1 or witness is None or witness.mode != CUDAGraphMode.PIECEWISE:
        raise RuntimeError(f"{target['target_id']} did not replay one framework PIECEWISE prefill graph")
    record = probe.pw_captured.get(witness.tokens)
    if record is None or witness.tokens < tokens:
        raise RuntimeError(f"no framework PIECEWISE capture of {witness.tokens} tokens for {tokens}")
    options = STATE.options
    o_proj = probe.attention.o_proj
    reduce = o_proj.reduce_results
    # A private pool per measurement (a released module graph frees its pool).
    pool = torch.cuda.graph_pool_handle()
    o_proj.reduce_results = False
    try:
        with override_forward_context(witness.context):
            # Eager warmup (eager breaks run inline outside a capture), then the
            # framework's breakable capture of the module call on a capture stream.
            probe.original(record.hidden_states, record.positions)
            torch.cuda.synchronize()
            capture = BreakableCUDAGraphCapture(pool=pool)
            with torch.cuda.stream(torch.cuda.Stream()), capture:
                output = probe.original(record.hidden_states, record.positions)
            torch.cuda.synchronize()
            timer = EventTimer(torch)
            for _ in range(options["warmup"] + options["iterations"]):
                timer(capture.replay)
            latencies = timer.read()
    finally:
        o_proj.reduce_results = reduce
    host = [round(v, 4) for v in timer.host_ms[options["warmup"] :]]
    finite = bool(torch.isfinite(output[:tokens]).all().item())
    STATE.writer.samples(
        target,
        latencies,
        options["warmup"],
        probe.source,
        {
            "finite": finite,
            "padded_tokens": witness.tokens,
            "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
            "segments": capture.num_graphs,
            "eager_breaks": capture.num_eager_breaks,
            "host_enqueue_ms": host,
        },
        timing_method=GRAPH_PREFILL,
    )
    del capture, output
    import gc

    # Return the module graph's private pool; per-target pools would otherwise
    # accumulate as reserved memory beside the serving KV pool.
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if not finite:
        raise RuntimeError(f"nonfinite prefill attention output for {target['target_id']}")
    return {"padded_tokens": witness.tokens}


# --- collective_rpc entry points (first argument is the vLLM worker) --------
def rpc_setup(worker, output: str, key_base: dict, provenance: dict, options: dict) -> dict:
    from collector.glm53flash_attention_runtime import RawWriter
    from vllm.distributed import get_tensor_model_parallel_rank

    if STATE.probe is None:
        raise RuntimeError("the GLM attention probe was not installed before graph capture")
    rank = get_tensor_model_parallel_rank()
    STATE.writer = RawWriter(Path(output), rank, key_base, provenance)
    STATE.options = options
    return {
        "rank": rank,
        "captured": sorted(STATE.probe.captured),
        "pw_captured": sorted(STATE.probe.pw_captured),
        "source": STATE.probe.source,
    }


def rpc_arm(worker, target: dict | None) -> None:
    STATE.armed, STATE.done, STATE.error = target, None, None


def rpc_status(worker) -> dict:
    return {
        "done": STATE.done,
        "error": STATE.error,
        "full_replays": STATE.full_replays,
        "pw_replays": STATE.pw_replays,
    }


def rpc_measure_prefill(worker, target: dict, replays_before: int) -> dict:
    import torch

    with torch.inference_mode():
        return _measure_prefill(STATE.probe, target, replays_before)


def rpc_measure_decode(worker, target: dict, replays_before: int) -> dict:
    import torch

    # The framework executes and captures its model under inference mode.
    with torch.inference_mode():
        return STATE.probe.measure_decode(target, replays_before)
