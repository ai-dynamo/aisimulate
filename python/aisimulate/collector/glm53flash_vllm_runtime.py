# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe real vLLM V1 eager forwards after native hybrid metadata preparation.

Original wrappers over vllm-project/vllm@ced6857afa0ea7b2e3f0846a62e1394e90f15607,
vllm/v1/worker/gpu_model_runner.py and vllm/forward_context.py (Apache-2.0).
No native request, scheduling, cache or compute implementation is copied.
See THIRD_PARTY_NOTICES.md and README.glm53flash.md.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import threading
from pathlib import Path

from collector.glm53flash_sglang_runtime import match_frozen_requests


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def allocated_state_inventory(model, cache_dtype: str) -> dict:
    """Inspect allocated cache tensors; checkpoint precision is not cache precision."""
    groups = {key: [] for key in ("kda_conv", "kda_temporal", "mla_latent", "pooled_index_packed", "index_tail")}

    def add(kind, name, tensor):
        groups[kind].append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "device": str(tensor.device),
                "nbytes": int(tensor.numel() * tensor.element_size()),
            }
        )

    for name, layer in model.named_modules():
        if type(layer).__name__ != "Glm5NextDecoderLayer":
            continue
        attn = layer.self_attn
        if type(attn).__name__ == "Glm5NextLinearAttention":
            conv, recurrent = attn.kv_cache
            add("kda_conv", name, conv)
            add("kda_temporal", name, recurrent)
        else:
            add("mla_latent", name, attn.mla_attn.mla_attn.kv_cache)
            add("pooled_index_packed", name, attn.indexer.k_cache.kv_cache)
            add("index_tail", name, attn.indexer.tail_cache.kv_cache)
    counts = {"kda_conv": 34, "kda_temporal": 34, "mla_latent": 11, "pooled_index_packed": 11, "index_tail": 11}
    dtypes = {
        "kda_conv": {"torch.bfloat16"},
        "kda_temporal": {"torch.float32"},
        "mla_latent": {"torch.uint8", "torch.float8_e4m3fn"},
        "pooled_index_packed": {"torch.uint8"},
        "index_tail": {"torch.bfloat16"},
    }
    admitted = cache_dtype in ("fp8", "fp8_e4m3") and all(
        len(groups[key]) == count and all(row["dtype"] in dtypes[key] for row in groups[key])
        for key, count in counts.items()
    )
    return {
        "admitted": admitted,
        "groups": groups,
        "native_cache_dtype": cache_dtype,
        "pooled_index_layout": "packed_fp8_keys_and_fp32_scales",
        "index_tail_layout": "paged_bf16_key_and_gate_score",
    }


def native_coordinates(runner, scheduler_output) -> dict:
    batch = runner.input_batch
    ids = list(batch.req_ids)
    queries = [int(scheduler_output.num_scheduled_tokens[rid]) for rid in ids]
    prefixes = list(map(int, batch.num_computed_tokens_cpu[: len(ids)]))
    phases = {
        "context" if prefix < int(runner.requests[rid].num_prompt_tokens) else "generation"
        for rid, prefix in zip(ids, prefixes, strict=True)
    }
    if len(phases) != 1 or not ids or len(set(ids)) != len(ids):
        raise RuntimeError("native GLM Ops does not admit mixed phases or missing request identities")
    phase = phases.pop()
    if len(set(queries)) != 1 or len(set(prefixes)) != 1 or min(queries) < 1 or min(prefixes) < 0:
        raise RuntimeError("native GLM Ops requires homogeneous positive scheduled coordinates")
    if phase == "generation" and queries[0] != 1:
        raise RuntimeError("native GLM Ops excludes speculative decode")
    return {
        "phase": phase,
        "batch_size": len(ids),
        "request_ids": ids,
        "query_lengths": queries,
        "prefix_lengths": prefixes,
        "total_new_tokens": sum(queries),
        "total_past_kv_tokens": sum(prefixes),
    }


class _TraceState:
    def __init__(self, runner, output, provenance, manifest):
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        from collector.glm53flash_native_hooks import install_native_hooks
        from collector.glm53flash_observer import NativeOperationObserver

        self.torch = torch
        self.runner, self.output, self.provenance = runner, output, provenance
        self.rank = get_tensor_model_parallel_rank()
        self.previous, self.matched = {}, set()
        self.counter = 0
        self.observer = NativeOperationObserver(manifest, provenance, self.rank) if manifest is not None else None
        self.whole_events = None
        output.mkdir(parents=True, exist_ok=True)
        if self.observer is not None:
            inventory = install_native_hooks(runner.model, self.observer, "vllm")
            (output / f"inventory-rank-{self.rank}.json").write_text(json.dumps(inventory, indent=2))
        else:
            original_logits = runner.model.compute_logits

            @functools.wraps(original_logits)
            def compute_logits(*args, **kwargs):
                result = original_logits(*args, **kwargs)
                if self.whole_events is not None:
                    start, end, stream = self.whole_events
                    if self.whole_end_recorded or self.torch.cuda.current_stream() != stream:
                        raise RuntimeError("whole-forward logits receipt changed stream or repeated")
                    end.record(stream)
                    self.whole_end_recorded = True
                return result

            runner.model.compute_logits = compute_logits
        self.layout = allocated_state_inventory(runner.model, runner.cache_config.cache_dtype)
        self.layout_sha256 = _digest(self.layout)
        (output / f"state-layout-rank-{self.rank}.json").write_text(json.dumps(self.layout, indent=2))

    def append(self, name, value):
        path = self.output / (f"rank-{self.rank}.jsonl" if name == "ops" else f"{name}-rank-{self.rank}.jsonl")
        with path.open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")

    def before(self, scheduler_output, input_ids, forward_context):
        from collector.glm53flash_contract import validate_native_workload
        from collector.glm53flash_observer import NativeWorkload

        coords = native_coordinates(self.runner, scheduler_output)
        for prefix, query in zip(coords["prefix_lengths"], coords["query_lengths"], strict=True):
            validate_native_workload("vllm", coords["phase"], prefix, query)
        runtime_mode = forward_context.cudagraph_runtime_mode.name
        if runtime_mode != "NONE":
            raise RuntimeError("vLLM eager Ops policy encountered native CUDA graph dispatch")
        if input_ids is None:
            raise RuntimeError("initial GLM Ops requires native text token IDs")
        tokens = list(map(int, input_ids.detach().cpu().reshape(-1).tolist()))
        if len(tokens) != coords["total_new_tokens"]:
            raise RuntimeError("eager native input tensor differs from unpadded scheduled tokens")
        self.counter += 1
        invocation = self.counter
        records, completed, history = [], {}, []
        offset = 0
        for rid, query, prefix in zip(
            coords["request_ids"], coords["query_lengths"], coords["prefix_lengths"], strict=True
        ):
            request = self.runner.requests[rid]
            previous = self.previous.get(rid)
            real = previous is not None and previous["computed_tokens"] == prefix
            native_query = tokens[offset : offset + query]
            offset += query
            prompt = list(map(int, request.prompt_token_ids))
            full = (previous["tokens"] if real else prompt[:prefix]) + native_query
            if len(full) != prefix + query or full[: min(len(prompt), len(full))] != prompt[: len(full)]:
                raise RuntimeError("native query tokens disagree with real request prompt/history")
            history.append(previous["forward_id"] if real else "")
            completed[rid] = full
            records.append(
                {
                    "request_id": rid,
                    "prompt_token_ids": prompt,
                    "native_query_token_ids": native_query,
                    "computed_tokens_before": prefix,
                    "computed_tokens_after": prefix + query,
                    "previous_forward_id": history[-1] or None,
                    "same_request_real_prefix": not prefix or real,
                    "input_tokens_sha256": _digest(full),
                }
            )
        record = {
            **self.provenance,
            **coords,
            "requests": records,
            "invocation": invocation,
            "tp_rank": self.rank,
            "forward_id": f"rank-{self.rank}/forward-{invocation}",
            "runtime_mode": runtime_mode,
            "used_cuda_graph": False,
            "num_padded_tokens": len(tokens),
            "allocated_fake_tokens": 0,
            "state_layout_sha256": self.layout_sha256,
            "state_layout_admitted": self.layout["admitted"],
            "state_protocol": "glm53flash_same_request_real_hybrid_v1",
            "ops_instrumented": self.observer is not None,
        }
        mapping = json.loads(Path(os.environ["AISIM_GLM53_REQUEST_MANIFEST"]).read_text())
        record.update(match_frozen_requests(record, mapping))
        if record["stage"] == "measure":
            if not self.layout["admitted"]:
                raise RuntimeError("actual vLLM hybrid cache layout is outside admitted GLM identity")
            identity = (record["benchmark_id"], record["repetition"])
            if identity in self.matched:
                raise RuntimeError("frozen vLLM target matched more than one native forward")
            self.matched.add(identity)
            prefix = coords["prefix_lengths"][0]
            workload = NativeWorkload(
                coords["phase"],
                coords["batch_size"],
                coords["query_lengths"][0],
                prefix,
                "decode" if coords["phase"] == "generation" else "chunked_prefill" if prefix else "full_prefill",
                tuple(coords["request_ids"]),
                tuple(history) if prefix else (),
                record["repetition"],
                invocation,
            )
            if self.observer is not None:
                self.observer.begin(workload)
            else:
                stream = self.torch.cuda.current_stream()
                start = self.torch.cuda.Event(enable_timing=True)
                end = self.torch.cuda.Event(enable_timing=True)
                self.whole_events = start, end, stream
                self.whole_end_recorded = False
                start.record(stream)
        return record, completed

    def after(self, record, completed):
        if record["stage"] == "measure" and self.observer is not None:
            for row in self.observer.end():
                row.update(
                    {
                        key: record[key]
                        for key in (
                            "stage",
                            "benchmark_id",
                            "repetition",
                            "sampling_role",
                            "dataset_role",
                            "request_set",
                            "corpus_sha256",
                        )
                    }
                )
                self.append("ops", row)
        else:
            # Establish a completed real-prefix receipt; synchronization stays
            # outside every measured module interval and never populates state.
            self.torch.cuda.synchronize()
            if record["stage"] == "measure":
                if self.whole_events is None or not self.whole_end_recorded:
                    raise RuntimeError("whole-GPU forward lacks its native logits completion")
                start, end, _ = self.whole_events
                record["whole_forward_gpu_ms"] = start.elapsed_time(end)
                if record["whole_forward_gpu_ms"] <= 0:
                    raise RuntimeError("whole-GPU forward must have positive elapsed time")
                record["whole_forward_boundary"] = "embedding_to_logits_gpu_v1"
                self.whole_events = None
        for request in record["requests"]:
            rid = request["request_id"]
            self.previous[rid] = {
                "tokens": completed[rid],
                "computed_tokens": request["computed_tokens_after"],
                "forward_id": record["forward_id"],
            }
        record["gpu_completed"] = True
        self.append("forward", record)


def install():
    """Call in each worker before execution; model hooks install after native load."""
    from importlib.metadata import version

    if version("vllm") != "0.30.0":
        raise RuntimeError("GLM eager Ops requires pinned vLLM0.30.0")
    from vllm.forward_context import get_forward_context
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_aisim_glm53_ops_installed", False):
        return
    output = Path(os.environ["AISIM_GLM53_TRACE_DIR"])
    provenance = json.loads(Path(os.environ["AISIM_GLM53_PROVENANCE"]).read_text())
    manifest_path = os.environ.get("AISIM_GLM53_OPS_MANIFEST")
    purpose = os.environ.get("AISIM_GLM53_PURPOSE")
    if purpose not in ("ops", "ops_holdout") or bool(manifest_path) != (purpose == "ops"):
        raise RuntimeError("native worker instrumentation must match explicit Ops/heldout purpose")
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else None
    original_execute, original_forward = GPUModelRunner.execute_model, GPUModelRunner._model_forward
    current, states = threading.local(), {}

    @functools.wraps(original_execute)
    def execute(runner, scheduler_output, *args, **kwargs):
        if getattr(current, "scheduler_output", None) is not None:
            raise RuntimeError("nested vLLM execution cannot share a native Ops receipt")
        current.scheduler_output = scheduler_output
        current.forward_calls = 0
        current.forward_receipt = None
        try:
            result = original_execute(runner, scheduler_output, *args, **kwargs)
            if scheduler_output.total_num_scheduled_tokens and current.forward_calls != 1:
                raise RuntimeError("native scheduled tokens did not execute exactly one target model forward")
            if current.forward_receipt is not None:
                state, record, completed = current.forward_receipt
                state.after(record, completed)
            return result
        finally:
            current.scheduler_output = None

    @functools.wraps(original_forward)
    def forward(runner, *args, **kwargs):
        schedule = getattr(current, "scheduler_output", None)
        if schedule is None:
            return original_forward(runner, *args, **kwargs)
        current.forward_calls += 1
        state = states.get(id(runner))
        if state is None:
            state = states[id(runner)] = _TraceState(runner, output, provenance, manifest)
        record, completed = state.before(
            schedule, kwargs.get("input_ids", args[0] if args else None), get_forward_context()
        )
        try:
            result = original_forward(runner, *args, **kwargs)
            current.forward_receipt = (state, record, completed)
            return result
        except BaseException as error:
            state.append("failed", {**record, "error_type": type(error).__name__, "error": str(error)})
            raise

    GPUModelRunner.execute_model, GPUModelRunner._model_forward = execute, forward
    GPUModelRunner._aisim_glm53_ops_installed = True
