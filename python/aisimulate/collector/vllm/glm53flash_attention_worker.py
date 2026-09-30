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
    executions=0,
    last_mode=None,
    last_descriptor=None,
    error=None,
)


def register() -> None:
    """vLLM general-plugin entry point (runs in every vLLM process)."""
    if not os.environ.get("GLM53_W4_MANIFEST"):
        return
    from vllm.v1.worker import gpu_model_runner as runner_module

    runner_class = runner_module.GPUModelRunner
    if getattr(runner_class, "_glm53_w4_patched", False):
        return
    original_capture = runner_class.capture_model

    def capture_model(self):
        # Install before the framework captures its decode graphs so the
        # module's capture-time arguments and forward context are recorded.
        install_probe(self)
        return original_capture(self)

    runner_class.capture_model = capture_model
    original_context = runner_module.set_forward_context

    def set_forward_context(*args, **kwargs):
        STATE.executions += 1
        STATE.last_mode = kwargs.get("cudagraph_runtime_mode")
        STATE.last_descriptor = kwargs.get("batch_descriptor")
        return original_context(*args, **kwargs)

    runner_module.set_forward_context = set_forward_context
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
    STATE.probe = Probe(torch, attention, model_runner, manifest)


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
            str(model_runner.compilation_config.cudagraph_mode),
            "CUDAGraphMode.FULL_DECODE_ONLY",
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
        self.pool = None
        self.source = kernel_source(attention)
        attention.forward = self.forward

    def _metadata(self):
        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        return attn_metadata[self.attention.indexer.k_cache.prefix]

    def forward(self, hidden_states, positions):
        torch = self.torch
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if torch.cuda.is_current_stream_capturing() and is_forward_context_available():
            context = get_forward_context()
            if context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
                self.captured[int(hidden_states.shape[0])] = SimpleNamespace(
                    hidden_states=hidden_states, positions=positions, context=context
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
        finite = all(bool(torch.isfinite(o).all().item()) for o in outputs)
        drift = float((outputs[-1].float() - outputs[0].float()).abs().max().item())
        STATE.writer.samples(
            target,
            latencies,
            options["warmup"],
            self.source,
            {"finite": finite, "repeat_max_abs_diff": drift, **classification},
        )
        if not finite:
            STATE.error = f"nonfinite attention output for {target['target_id']}"
            raise RuntimeError(STATE.error)
        STATE.done = target["target_id"]
        return self.original(hidden_states, positions)

    def measure_decode(self, target: dict) -> dict:
        torch = self.torch
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import override_forward_context

        batch = target["batch_size"]
        sizes = sorted(self.model_runner.compilation_config.cudagraph_capture_sizes)
        padded = min(size for size in sizes if size >= batch)
        descriptor = STATE.last_descriptor
        if (
            STATE.last_mode != CUDAGraphMode.FULL
            or descriptor is None
            or descriptor.num_tokens != padded
            or not descriptor.uniform
        ):
            raise RuntimeError(
                f"real decode ran {STATE.last_mode}/{descriptor}, not the FULL graph for {padded} tokens"
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
        finite = bool(torch.isfinite(output[:batch]).all().item())
        STATE.writer.samples(
            target, latencies, options["warmup"], self.source, {"finite": finite, "padded_tokens": padded}
        )
        del graph
        if not finite:
            raise RuntimeError(f"nonfinite decode attention output for {target['target_id']}")
        return {"padded_tokens": padded}


# --- collective_rpc entry points (first argument is the vLLM worker) --------
def rpc_setup(worker, output: str, key_base: dict, provenance: dict, options: dict) -> dict:
    from collector.glm53flash_attention_runtime import RawWriter
    from vllm.distributed import get_tensor_model_parallel_rank

    if STATE.probe is None:
        raise RuntimeError("the GLM attention probe was not installed before graph capture")
    rank = get_tensor_model_parallel_rank()
    STATE.writer = RawWriter(Path(output), rank, key_base, provenance)
    STATE.options = options
    return {"rank": rank, "captured": sorted(STATE.probe.captured), "source": STATE.probe.source}


def rpc_arm(worker, target: dict | None) -> None:
    STATE.armed, STATE.done, STATE.error = target, None, None


def rpc_status(worker) -> dict:
    return {"done": STATE.done, "error": STATE.error, "executions": STATE.executions}


def rpc_measure_decode(worker, target: dict) -> dict:
    return STATE.probe.measure_decode(target)
