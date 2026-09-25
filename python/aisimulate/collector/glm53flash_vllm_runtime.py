# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe native vLLM eager and V2 graph forwards with their real hybrid state.

Original wrappers over vllm-project/vllm@ced6857afa0ea7b2e3f0846a62e1394e90f15607,
vllm/v1/worker/gpu_model_runner.py, vllm/v1/worker/gpu/model_runner.py,
vllm/compilation/breakable_cudagraph.py and vllm/forward_context.py (Apache-2.0).
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

from collector.glm53flash_protocol import (
    MAX_MEASURED_CONTEXT,
    VLLM_CONTEXT_POLICY_VERSION,
    native_gpu_identity,
    validate_gb300_identity,
    vllm_context_policy,
)
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


def native_context_receipt(runner) -> dict:
    policy = vllm_context_policy(int(os.environ.get("DYN_FPM_GLM53FLASH_MEASURED_CONTEXT", MAX_MEASURED_CONTEXT)))
    if runner.max_model_len != policy["runtime_context_length"]:
        raise RuntimeError("actual native vLLM worker context differs from measured context plus reserved headroom")
    return {
        "context_policy": policy,
        "context_policy_version": VLLM_CONTEXT_POLICY_VERSION,
        "native_max_model_len": int(runner.max_model_len),
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


def native_v2_coordinates(runner, scheduler_output, batch, *, graph_policy=None, native_descriptor=None) -> dict:
    """Read the actual V2 InputBatch; never reconstruct native hybrid metadata."""
    ids = list(batch.req_ids)
    queries = list(map(int, batch.num_scheduled_tokens))
    prefixes = list(map(int, batch.num_computed_tokens_np))
    gpu_lengths = list(map(int, batch.seq_lens[: len(ids)].detach().cpu().tolist()))
    phases = {"context" if flag else "generation" for flag in batch.is_prefilling_np}
    if (
        not ids
        or len(set(ids)) != len(ids)
        or len(phases) != 1
        or len(set(queries)) != 1
        or len(set(prefixes)) != 1
        or min(queries) < 1
        or min(prefixes) < 0
    ):
        raise RuntimeError("native V2 Ops requires homogeneous actual request coordinates")
    if (
        batch.num_tokens != sum(queries)
        or batch.num_draft_tokens
        or len(queries) != len(ids)
        or len(prefixes) != len(ids)
        or len(batch.is_prefilling_np) != len(ids)
        or gpu_lengths != [p + q for p, q in zip(prefixes, queries, strict=True)]
        or queries != [scheduler_output.num_scheduled_tokens[rid] for rid in ids]
        or list(map(int, batch.idx_mapping_np)) != [runner.req_states.req_id_to_index[rid] for rid in ids]
    ):
        raise RuntimeError("native V2 scheduled tokens/slots/padding differ from actual inputs")
    phase = phases.pop()
    if phase == "generation" and queries[0] != 1:
        raise RuntimeError("native V2 Ops excludes speculative decode")
    result = {
        "phase": phase,
        "batch_size": len(ids),
        "request_ids": ids,
        "query_lengths": queries,
        "prefix_lengths": prefixes,
        "total_new_tokens": sum(queries),
        "total_past_kv_tokens": sum(prefixes),
    }
    if graph_policy is None and native_descriptor is None:
        if batch.num_tokens_after_padding != batch.num_tokens:
            raise RuntimeError("native V2 eager inputs unexpectedly include graph padding")
        return result
    if graph_policy is None or native_descriptor is None:
        raise RuntimeError("native V2 graph padding needs both initialized policy and actual descriptor")
    from collector.glm53flash_vllm_graph_policy import descriptor, select_descriptor

    actual = descriptor(native_descriptor)
    expected = select_descriptor(graph_policy, batch=len(ids), query=queries[0], is_context=phase == "context")
    if actual != expected:
        raise RuntimeError("native V2 selected descriptor differs from its initialized source-bound policy")
    physical_requests = actual["num_reqs"] or len(ids)
    if batch.num_tokens_after_padding != actual["num_tokens"] or batch.num_reqs_after_padding != physical_requests:
        raise RuntimeError("native V2 physical padding differs from the actual selected descriptor")
    result["native_dispatch"] = {
        "descriptor": actual,
        "policy_sha256": _digest(graph_policy),
        "physical_tokens": int(batch.num_tokens_after_padding),
        "physical_requests": int(batch.num_reqs_after_padding),
    }
    return result


class _TraceState:
    def __init__(
        self,
        runner,
        output,
        provenance,
        manifest,
        *,
        graph_calibration=False,
        piecewise_replay=False,
        serving_none=False,
        serving_none_measured=False,
    ):
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        from collector.glm53flash_native_hooks import install_native_hooks
        from collector.glm53flash_observer import NativeOperationObserver

        self.torch = torch
        self.runner, self.output, self.provenance = runner, output, provenance
        self.rank = get_tensor_model_parallel_rank()
        self.previous, self.matched = {}, set()
        self.counter = 0
        self.serving_none = serving_none
        self.serving_none_measured = serving_none_measured
        self.none_witness, self.none_boundaries = None, None
        if serving_none:
            from collector.glm53flash_vllm_none import NativeNoneModelWitness

            self.none_witness = NativeNoneModelWitness(runner.model)
            self.none_identity_sha256 = _digest(self.none_witness.receipt)
        self.observer = NativeOperationObserver(manifest, provenance, self.rank) if manifest is not None else None
        self.none_execution = None
        if serving_none_measured and self.observer is not None:
            from collector.glm53flash_vllm_none_activity import NativeNoneExecution

            self.none_execution = NativeNoneExecution(self.observer, output)
        self.whole_events = None
        self.graph_execution = None
        self.piecewise_replay = piecewise_replay
        if graph_calibration:
            from collector.glm53flash_vllm_graph_ops import NativeVllmGraphExecution

            if manifest is not None:
                raise RuntimeError("native graph calibration cannot install eager operation intervals")
            self.graph_execution = NativeVllmGraphExecution(
                runner, output, self.rank, include_piecewise=piecewise_replay
            )
        output.mkdir(parents=True, exist_ok=True)
        if self.none_witness is not None:
            (output / f"serving-none-model-rank-{self.rank}.json").write_text(
                json.dumps(self.none_witness.receipt, indent=2)
            )
        if self.observer is not None:
            inventory = install_native_hooks(runner.model, self.observer, "vllm")
            (output / f"inventory-rank-{self.rank}.json").write_text(json.dumps(inventory, indent=2))
        original_logits = runner.model.compute_logits

        @functools.wraps(original_logits)
        def compute_logits(*args, **kwargs):
            if self.none_boundaries is not None:
                if len(self.none_boundaries) != 2:
                    raise RuntimeError("serving NONE logits preceded or repeated its original model return")
                self.none_witness.validate()
                self.mark_none_boundary()
                if self.none_execution is not None:
                    self.none_execution.boundary("logits")
            result = original_logits(*args, **kwargs)
            if self.whole_events is not None:
                start, end, stream = self.whole_events
                if self.whole_end_recorded or self.torch.cuda.current_stream() != stream:
                    raise RuntimeError("whole-forward logits receipt changed stream or repeated")
                end.record(stream)
                self.whole_end_recorded = True
                if self.graph_execution is not None:
                    self.graph_execution.end_logits()
                if self.none_execution is not None:
                    self.none_execution.end_logits()
            return result

        runner.model.compute_logits = compute_logits
        if self.none_witness is not None:
            self.none_witness.bind_observer_wrapper("compute_logits", compute_logits)
        self.layout = allocated_state_inventory(runner.model, runner.cache_config.cache_dtype)
        from collector.glm53flash_runtime_identity import observe_vllm_runtime_closure

        source_manifest = Path(__file__).parent / "fpm_forward/runtime/glm53flash/runtime-source-sha256.json"
        closure = observe_vllm_runtime_closure(provenance["backend_version"], source_manifest)
        if closure is not None:
            self.layout["runtime_closure"] = closure
        self.layout.update(tp_rank=self.rank, hardware=native_gpu_identity(torch))
        self.layout_sha256 = _digest(self.layout)
        (output / f"state-layout-rank-{self.rank}.json").write_text(json.dumps(self.layout, indent=2))
        validate_gb300_identity(self.layout["hardware"])

    def append(self, name, value):
        path = self.output / (f"rank-{self.rank}.jsonl" if name == "ops" else f"{name}-rank-{self.rank}.jsonl")
        with path.open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")

    def mark_none_boundary(self):
        stream = self.torch.cuda.current_stream()
        if self.whole_events is None or stream != self.whole_events[2]:
            raise RuntimeError("serving NONE runtime boundary changed the actual whole-forward stream")
        event = self.torch.cuda.Event(enable_timing=True)
        event.record(stream)
        self.none_boundaries.append(event)

    def before(
        self,
        scheduler_output,
        input_ids,
        forward_context,
        *,
        native_batch=None,
        graph_policy=None,
        native_descriptor=None,
    ):
        from collector.glm53flash_contract import validate_native_workload
        from collector.glm53flash_observer import NativeWorkload

        context_receipt = native_context_receipt(self.runner)
        coords = (
            native_coordinates(self.runner, scheduler_output)
            if native_batch is None
            else native_v2_coordinates(
                self.runner,
                scheduler_output,
                native_batch,
                graph_policy=graph_policy,
                native_descriptor=native_descriptor,
            )
        )
        for prefix, query in zip(coords["prefix_lengths"], coords["query_lengths"], strict=True):
            validate_native_workload("vllm", coords["phase"], prefix, query, self.provenance["backend_version"])
            if prefix + query > context_receipt["context_policy"]["measured_context_limit"]:
                raise RuntimeError("actual native forward exceeds frozen measured context")
        graph = graph_policy is not None
        runtime_mode = (
            coords["native_dispatch"]["descriptor"]["cg_mode"] if graph else forward_context.cudagraph_runtime_mode.name
        )
        if not graph and runtime_mode != "NONE":
            raise RuntimeError("vLLM eager Ops policy encountered native CUDA graph dispatch")
        if input_ids is None:
            raise RuntimeError("initial GLM Ops requires native text token IDs")
        tokens = list(map(int, input_ids.detach().cpu().reshape(-1).tolist()))
        physical_tokens = coords["native_dispatch"]["physical_tokens"] if graph else coords["total_new_tokens"]
        if len(tokens) != physical_tokens:
            raise RuntimeError("eager native input tensor differs from unpadded scheduled tokens")
        tokens = tokens[: coords["total_new_tokens"]]
        self.counter += 1
        invocation = self.counter
        records, completed, history = [], {}, []
        offset = 0
        for rid, query, prefix in zip(
            coords["request_ids"], coords["query_lengths"], coords["prefix_lengths"], strict=True
        ):
            previous = self.previous.get(rid)
            real = previous is not None and previous["computed_tokens"] == prefix
            native_query = tokens[offset : offset + query]
            offset += query
            if native_batch is None:
                prompt = list(map(int, self.runner.requests[rid].prompt_token_ids))
            else:
                index = self.runner.req_states.req_id_to_index[rid]
                count = int(self.runner.req_states.prompt_len.np[index])
                prompt = list(map(int, self.runner.req_states.all_token_ids.gpu[index, :count].detach().cpu().tolist()))
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
            **context_receipt,
            "requests": records,
            "invocation": invocation,
            "tp_rank": self.rank,
            "forward_id": f"rank-{self.rank}/forward-{invocation}",
            "runtime_mode": runtime_mode,
            "used_cuda_graph": runtime_mode != "NONE",
            "num_padded_tokens": physical_tokens,
            "allocated_fake_tokens": 0,
            "state_layout_sha256": self.layout_sha256,
            "state_layout_admitted": self.layout["admitted"],
            "state_protocol": "glm53flash_same_request_real_hybrid_v1",
            "ops_instrumented": self.observer is not None or getattr(self, "graph_execution", None) is not None,
        }
        mapping = json.loads(Path(os.environ["AISIM_GLM53_REQUEST_MANIFEST"]).read_text())
        if any(rid not in mapping.get("requests", {}) for rid in record["request_ids"]):
            raise RuntimeError("native serving request is absent from the frozen request manifest")
        record.update(match_frozen_requests(record, mapping))
        if record["stage"] == "measure":
            allowed_graph_target = (runtime_mode == "FULL" and coords["phase"] == "generation") or (
                runtime_mode == "PIECEWISE" and getattr(self, "piecewise_replay", False)
            )
            serving_none = getattr(self, "serving_none", False)
            if serving_none:
                from collector.glm53flash_vllm_none import validate_none_target

                validate_none_target(record)
                if getattr(self, "serving_none_measured", False):
                    from collector.glm53flash_vllm_none_activity import NONE_MEASUREMENT_CONTRACT

                    record["measurement_contract"] = NONE_MEASUREMENT_CONTRACT
                    record["profiled"] = False
                else:
                    record["measurement_admission"] = "DIAGNOSTIC_ONLY_NO_TABLE_EXPORT"
                record["serving_none_model_sha256"] = self.none_identity_sha256
            elif graph and (not allowed_graph_target or self.observer is not None):
                raise RuntimeError("native V2 graph target requires actual FULL decode or explicit PIECEWISE replay")
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
            graph_execution = getattr(self, "graph_execution", None)
            if graph_execution is not None:
                graph_execution.begin(record, native_descriptor)
            # Calibration also needs one coherent native rank timeline. The
            # window ends at logits, before sampling/readback, exactly as the
            # independent uninstrumented holdout window does.
            stream = self.torch.cuda.current_stream()
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            self.whole_events = start, end, stream
            self.whole_end_recorded = False
            self.whole_boundary = "native_metadata_to_logits_gpu_v1" if graph else "embedding_to_logits_gpu_v1"
            start.record(stream)
            if serving_none:
                self.none_boundaries = []
                if getattr(self, "none_execution", None) is not None:
                    self.none_execution.begin(record)
            if graph_execution is not None:
                graph_execution.start_range()
        return record, completed

    def after(self, record, completed):
        if record["stage"] == "measure" and self.observer is not None:
            if getattr(self, "serving_none", False):
                from collections import Counter

                record["native_operation_calls"] = dict(Counter(event["name"] for event in self.observer.events))
            rows = self.observer.end()
            if getattr(self, "serving_none", False):
                from collector.glm53flash_vllm_none import diagnostic_operation_rows, measured_operation_rows

                rows = (
                    measured_operation_rows(record, rows)
                    if getattr(self, "serving_none_measured", False)
                    else diagnostic_operation_rows(record, rows)
                )
            for row in rows:
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
                self.append(
                    "serving-none-measured-ops"
                    if getattr(self, "serving_none_measured", False)
                    else "serving-none-ops"
                    if getattr(self, "serving_none", False)
                    else "ops",
                    row,
                )
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
            record["whole_forward_boundary"] = getattr(self, "whole_boundary", "embedding_to_logits_gpu_v1")
            if getattr(self, "serving_none", False):
                if len(self.none_boundaries or ()) != 3 or not record.get("native_none_forward_completed"):
                    raise RuntimeError("serving NONE lacks actual model/logits runtime boundaries")
                model_start, model_end, logits_start = self.none_boundaries
                record["native_runtime_boundary_gpu_ms"] = {
                    "prepared_inputs_to_raw_model_entry": start.elapsed_time(model_start),
                    "raw_model_return_to_logits_entry": model_end.elapsed_time(logits_start),
                }
                if any(value < 0 for value in record["native_runtime_boundary_gpu_ms"].values()):
                    raise RuntimeError("serving NONE runtime boundary interval is negative")
                self.none_boundaries = None
                if getattr(self, "none_execution", None) is not None:
                    self.none_execution.complete(record)
            self.whole_events = None
        for request in record["requests"]:
            rid = request["request_id"]
            if record["stage"] == "measure":
                # The benchmark's target is the final requested forward for this
                # Req. Keep its complete history on disk; only a pending real
                # seed needs a host token chain for the next native forward.
                self.previous.pop(rid, None)
            else:
                self.previous[rid] = {
                    "tokens": completed[rid],
                    "computed_tokens": request["computed_tokens_after"],
                    "forward_id": record["forward_id"],
                }
        record["gpu_completed"] = True
        self.append("forward", record)
        graph_execution = getattr(self, "graph_execution", None)
        if record["stage"] == "measure" and graph_execution is not None:
            self.append("graph-forward", graph_execution.finish(record))


def install():
    """Call in each worker before execution; model hooks install after native load."""
    if os.environ.get("AISIM_GLM53_PURPOSE") in ("ops_graph", "ops_graph_holdout"):
        # V2 owns explicit native FULL replay; importing the legacy runner must
        # not install an eager observer on a graph-serving control.
        return
    from importlib.metadata import version

    from collector.glm53flash_runtime_identity import validate_backend_version

    backend_version = validate_backend_version("vllm", version("vllm"))
    from collector.glm53flash_contract import validate_run_identity

    provenance = json.loads(Path(os.environ["AISIM_GLM53_PROVENANCE"]).read_text())
    validate_run_identity(provenance, os.environ.get("FPM_RUN_ID"))
    from vllm.forward_context import get_forward_context
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_aisim_glm53_ops_installed", False):
        return
    output = Path(os.environ["AISIM_GLM53_TRACE_DIR"])
    if provenance.get("backend_version") != backend_version:
        raise RuntimeError("actual native worker package differs from frozen Ops provenance")
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


def install_worker_lifecycle():
    """Observe only after native compile/JIT warmup returns successfully.

    Pinned gpu_worker.py:773-816 calls warmup.py:355 with scheduler-realistic
    requests and dummy_run=False. The explicit native lifecycle, not missing
    files or a request-name heuristic, distinguishes those from serving.
    """
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(Worker, "_aisim_glm53_ops_lifecycle_installed", False):
        return
    original = Worker.compile_or_warm_up_model

    @functools.wraps(original)
    def compile_or_warm_up_model(worker, *args, **kwargs):
        runner = worker.model_runner
        if getattr(runner, "_aisim_glm53_ops_warming_up", False):
            raise RuntimeError("nested native worker warmup cannot define serving readiness")
        runner._aisim_glm53_ops_serving_ready = False
        runner._aisim_glm53_ops_warming_up = True
        try:
            result = original(worker, *args, **kwargs)
        finally:
            runner._aisim_glm53_ops_warming_up = False
        runner._aisim_glm53_ops_serving_ready = True
        return result

    Worker.compile_or_warm_up_model = compile_or_warm_up_model
    Worker._aisim_glm53_ops_lifecycle_installed = True


def _piecewise_capture_enabled(purpose):
    """Explicit capture-only qualification; never turn a control into profiling."""
    value = os.environ.get("AISIM_GLM53_PIECEWISE_CAPTURE_ONLY", "0")
    if value not in ("0", "1") or (value == "1" and purpose != "ops_graph"):
        raise RuntimeError("PIECEWISE capture-only observation requires explicit graph calibration purpose")
    return value == "1"


def _piecewise_replay_enabled(purpose):
    value = os.environ.get("AISIM_GLM53_PIECEWISE_REPLAY", "0")
    if value not in ("0", "1") or (
        value == "1"
        and (
            purpose not in ("ops_graph", "ops_graph_holdout")
            or os.environ.get("AISIM_GLM53_PIECEWISE_CAPTURE_ONLY", "0") != "0"
        )
    ):
        raise RuntimeError("PIECEWISE replay requires explicit graph purpose and cannot be capture-only")
    return value == "1"


def _serving_none_enabled(purpose):
    value = os.environ.get("AISIM_GLM53_SERVING_NONE_DIAGNOSTIC", "0")
    if value not in ("0", "1") or (
        value == "1"
        and (
            purpose not in ("ops_graph", "ops_graph_holdout")
            or os.environ.get("AISIM_GLM53_PIECEWISE_CAPTURE_ONLY", "0") != "0"
            or os.environ.get("AISIM_GLM53_PIECEWISE_REPLAY", "0") != "0"
            or os.environ.get("AISIM_GLM53_SERVING_NONE_MEASURED", "0") != "0"
        )
    ):
        raise RuntimeError("serving NONE diagnostic requires its own explicit graph-purpose observation")
    return value == "1"


def _serving_none_measured_enabled(purpose):
    value = os.environ.get("AISIM_GLM53_SERVING_NONE_MEASURED", "0")
    if value not in ("0", "1") or (
        value == "1"
        and (
            purpose not in ("ops_graph", "ops_graph_holdout")
            or any(
                os.environ.get(name, "0") != "0"
                for name in (
                    "AISIM_GLM53_SERVING_NONE_DIAGNOSTIC",
                    "AISIM_GLM53_PIECEWISE_CAPTURE_ONLY",
                    "AISIM_GLM53_PIECEWISE_REPLAY",
                )
            )
        )
    ):
        raise RuntimeError("serving NONE measured route requires its own explicit graph-purpose observation")
    return value == "1"


def install_v2():
    """Bind the pinned native V2 model forward and its later logits/sample step."""
    from importlib.metadata import version

    from collector.glm53flash_runtime_identity import validate_backend_version

    backend_version = validate_backend_version("vllm", version("vllm"))
    from collector.glm53flash_contract import validate_run_identity

    provenance = json.loads(Path(os.environ["AISIM_GLM53_PROVENANCE"]).read_text())
    validate_run_identity(provenance, os.environ.get("FPM_RUN_ID"))
    from vllm.forward_context import get_forward_context
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_aisim_glm53_ops_installed", False):
        return
    output = Path(os.environ["AISIM_GLM53_TRACE_DIR"])
    if provenance.get("backend_version") != backend_version:
        raise RuntimeError("actual native worker package differs from frozen Ops provenance")
    manifest_path = os.environ.get("AISIM_GLM53_OPS_MANIFEST")
    purpose = os.environ.get("AISIM_GLM53_PURPOSE")
    if purpose not in ("ops", "ops_holdout", "ops_graph", "ops_graph_holdout") or bool(manifest_path) != (
        purpose in ("ops", "ops_graph")
    ):
        raise RuntimeError("native V2 instrumentation must match explicit Ops/holdout purpose")
    graph_mode = purpose in ("ops_graph", "ops_graph_holdout")
    graph_calibration = purpose == "ops_graph"
    include_piecewise = _piecewise_capture_enabled(purpose)
    piecewise_replay = _piecewise_replay_enabled(purpose)
    serving_none_measured = _serving_none_measured_enabled(purpose)
    serving_none = _serving_none_enabled(purpose) or serving_none_measured
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else None
    if graph_mode:
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

        from collector.glm53flash_vllm_graph_ops import (
            calibration_policy,
            holdout_policy,
            install_holdout_capture,
        )
        from collector.glm53flash_vllm_graph_ops import (
            install as install_graph_capture,
        )

        if graph_calibration and not serving_none:
            install_graph_capture(manifest, provenance, output, include_piecewise=include_piecewise or piecewise_replay)
        else:
            install_holdout_capture(output, **({"include_piecewise": True} if piecewise_replay else {}))
        graph_policy_for = calibration_policy if graph_calibration and not serving_none else holdout_policy
        original_replay = ModelCudaGraphManager.run_fullgraph
        if piecewise_replay:
            import inspect

            from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

            from collector.glm53flash_vllm_graph_policy import SOURCE_PINS as POLICY_SOURCE_PINS

            original_piecewise_replay = BreakableCUDAGraphWrapper._replay
            path = Path(inspect.getfile(original_piecewise_replay)).resolve()
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != POLICY_SOURCE_PINS["compilation/breakable_cudagraph.py"]
            ):
                raise RuntimeError("native PIECEWISE replay method differs from the frozen source")
    original_execute = GPUModelRunner.execute_model
    original_prepare = GPUModelRunner.prepare_inputs
    original_sample = GPUModelRunner.sample
    original_sample_tokens = GPUModelRunner.sample_tokens
    current, bindings = threading.local(), {}

    def fail_observation(binding, error):
        if binding is None or binding["pending"] is None:
            return
        record = binding["pending"][0]
        if record.get("observation_failed"):
            return
        record["observation_failed"] = True
        state = binding["state"]
        cleanup_error = None
        if getattr(state, "none_execution", None) is not None:
            try:
                state.none_execution.abort(error)
            except BaseException as failure:
                cleanup_error = repr(failure)
        if getattr(state, "graph_execution", None) is not None:
            try:
                state.graph_execution.abort(error)
            except BaseException as failure:
                # Keep the original native exception even if trace flushing
                # also fails. Neither failure can become an accepted forward.
                cleanup_error = repr(failure)
        state.append(
            "failed",
            {**record, "error_type": type(error).__name__, "error": str(error), "trace_cleanup_error": cleanup_error},
        )

    @functools.wraps(original_prepare)
    def prepare(runner, scheduler_output, *args, **kwargs):
        batch = original_prepare(runner, scheduler_output, *args, **kwargs)
        if getattr(runner, "_aisim_glm53_ops_warming_up", False):
            return batch
        if getattr(current, "scheduler_output", None) is not scheduler_output:
            raise RuntimeError("native V2 InputBatch has no observed scheduling boundary")
        binding = bindings.get(id(runner))
        if binding is None:
            state = _TraceState(
                runner,
                output,
                provenance,
                manifest if serving_none or not graph_mode else None,
                graph_calibration=graph_calibration and not serving_none,
                **({"piecewise_replay": True} if piecewise_replay else {}),
                **({"serving_none": True} if serving_none else {}),
                **({"serving_none_measured": True} if serving_none_measured else {}),
            )
            binding = bindings[id(runner)] = {"state": state, "pending": None, "sampled": None}
            original_forward = runner.model.forward

            @functools.wraps(original_forward)
            def forward(*forward_args, **forward_kwargs):
                schedule = getattr(current, "scheduler_output", None)
                if schedule is None:
                    return original_forward(*forward_args, **forward_kwargs)
                if serving_none:
                    if binding["pending"] is None:
                        raise RuntimeError("serving NONE model call lacks its actual prepared request")
                    record = binding["pending"][0]
                    if record["runtime_mode"] != "NONE" or record.get("native_none_forward_completed"):
                        raise RuntimeError("serving NONE model call changed dispatch or executed twice")
                    state.none_witness.validate()
                    measured = record["stage"] == "measure"
                    if measured:
                        state.mark_none_boundary()
                        if serving_none_measured and state.none_execution is not None:
                            state.none_execution.boundary("model")
                    result = original_forward(*forward_args, **forward_kwargs)
                    if measured:
                        state.mark_none_boundary()
                        if serving_none_measured and state.none_execution is not None:
                            state.none_execution.boundary("before_logits")
                    state.none_witness.validate()
                    record["native_none_forward_completed"] = True
                    return result
                if binding["pending"] is not None:
                    raise RuntimeError("native V2 forward repeated before its logits/sample completion")
                ids = forward_kwargs.get("input_ids", forward_args[0] if forward_args else None)
                record, completed = state.before(schedule, ids, get_forward_context(), native_batch=binding["batch"])
                record["native_runner"] = "v2"
                try:
                    result = original_forward(*forward_args, **forward_kwargs)
                except BaseException as error:
                    state.append("failed", {**record, "error_type": type(error).__name__, "error": str(error)})
                    raise
                binding["pending"] = record, completed
                return result

            if not graph_mode or serving_none:
                runner.model.forward = forward
                if serving_none:
                    state.none_witness.bind_observer_wrapper("forward", forward)
        if binding["pending"] is not None:
            raise RuntimeError("native V2 previous forward did not complete its sampling step")
        binding["batch"] = batch
        if graph_mode:
            descriptor = kwargs.get("batch_desc", args[1] if len(args) > 1 else None)
            policy = graph_policy_for(runner.cudagraph_manager, runner.model)
            if policy["backend_version"] != provenance["backend_version"] or policy["tp_rank"] != binding["state"].rank:
                raise RuntimeError("native graph policy differs from actual worker version/rank")
            record, completed = binding["state"].before(
                scheduler_output,
                batch.input_ids,
                None,
                native_batch=batch,
                graph_policy=policy,
                native_descriptor=descriptor,
            )
            record.update(native_runner="v2", native_graph_replay_completed=False)
            if piecewise_replay and record["runtime_mode"] == "PIECEWISE":
                record["native_piecewise_replay_completed"] = False
            if serving_none and record["runtime_mode"] == "NONE":
                record["native_none_forward_completed"] = False
            binding["pending"] = record, completed
            binding["descriptor"] = descriptor
        return batch

    @functools.wraps(original_execute)
    def execute(runner, scheduler_output, *args, **kwargs):
        # Native positional argument order: intermediate_tensors, dummy_run.
        dummy = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
        if dummy or getattr(runner, "_aisim_glm53_ops_warming_up", False):
            return original_execute(runner, scheduler_output, *args, **kwargs)
        if getattr(runner, "_aisim_glm53_ops_serving_ready", False) is not True:
            raise RuntimeError("native V2 serving call precedes successful worker compile/warmup completion")
        if getattr(current, "scheduler_output", None) is not None:
            raise RuntimeError("nested native V2 execution cannot share an Ops receipt")
        current.scheduler_output = scheduler_output
        current.runner = runner
        try:
            result = original_execute(runner, scheduler_output, *args, **kwargs)
            if scheduler_output.total_num_scheduled_tokens:
                binding = bindings.get(id(runner))
                if binding is None or binding["pending"] is None:
                    raise RuntimeError("native V2 scheduled tokens bypassed the observed serving boundary")
                record = binding["pending"][0]
                if serving_none and record["runtime_mode"] == "NONE" and not record["native_none_forward_completed"]:
                    raise RuntimeError("serving NONE execution bypassed its original raw model forward")
                if graph_mode and record["runtime_mode"] == "FULL" and not record["native_graph_replay_completed"]:
                    raise RuntimeError("native V2 FULL forward omitted its actual registered graph replay")
                if (
                    piecewise_replay
                    and record["runtime_mode"] == "PIECEWISE"
                    and not record["native_piecewise_replay_completed"]
                ):
                    raise RuntimeError("native V2 PIECEWISE forward omitted its actual initialized entry replay")
            return result
        except BaseException as error:
            fail_observation(bindings.get(id(runner)), error)
            raise
        finally:
            current.scheduler_output = None
            current.runner = None

    if graph_mode:

        @functools.wraps(original_replay)
        def replay(manager, descriptor, *args, **kwargs):
            runner = getattr(current, "runner", None)
            if runner is None or getattr(runner, "_aisim_glm53_ops_warming_up", False):
                return original_replay(manager, descriptor, *args, **kwargs)
            binding = bindings.get(id(runner))
            if manager is not runner.cudagraph_manager or binding is None or binding["pending"] is None:
                raise RuntimeError("native FULL replay lacks its same-runner prepared request")
            record = binding["pending"][0]
            if descriptor != binding["descriptor"] or record["native_graph_replay_completed"]:
                raise RuntimeError("native FULL replay changed descriptor or executed twice")
            graph_policy_for(manager, runner.model)
            result = original_replay(manager, descriptor, *args, **kwargs)
            record["native_graph_replay_completed"] = True
            return result

        ModelCudaGraphManager.run_fullgraph = replay

    if piecewise_replay:

        @functools.wraps(original_piecewise_replay)
        def replay_piecewise(wrapper, entry, args, kwargs):
            runner = getattr(current, "runner", None)
            if runner is None or getattr(runner, "_aisim_glm53_ops_warming_up", False):
                return original_piecewise_replay(wrapper, entry, args, kwargs)
            binding = bindings.get(id(runner))
            manager = runner.cudagraph_manager
            if wrapper is not manager.breakable_cg_runner or binding is None or binding["pending"] is None:
                raise RuntimeError("native PIECEWISE replay lacks its same-runner prepared request")
            record = binding["pending"][0]
            if record["runtime_mode"] != "PIECEWISE" or record["native_graph_replay_completed"]:
                raise RuntimeError("native PIECEWISE replay changed mode or executed twice")

            def check_entry():
                graph_policy_for(manager, runner.model)
                if graph_calibration:
                    from collector.glm53flash_vllm_piecewise import (
                        captured_piecewise_registry,
                        piecewise_capture_for_descriptor,
                    )

                    expected = piecewise_capture_for_descriptor(manager, binding["descriptor"])
                    if (
                        wrapper.entries.get(entry.batch_descriptor) is not entry
                        or captured_piecewise_registry(wrapper, entry.batch_descriptor) is not expected
                    ):
                        raise RuntimeError("native PIECEWISE replay changed its observed initialized entry")
                else:
                    from collector.glm53flash_vllm_graph_ops import holdout_piecewise_entry

                    if holdout_piecewise_entry(manager, binding["descriptor"]) is not entry:
                        raise RuntimeError("native PIECEWISE replay changed its independent initialized entry")

            check_entry()
            result = original_piecewise_replay(wrapper, entry, args, kwargs)
            check_entry()
            import dataclasses

            record["native_piecewise_replay"] = {
                "source_sha256": POLICY_SOURCE_PINS["compilation/breakable_cudagraph.py"],
                "entry_descriptor": dataclasses.asdict(entry.batch_descriptor),
                "segment_count": len(entry.capture.segments),
            }
            record["native_graph_replay_completed"] = True
            record["native_piecewise_replay_completed"] = True
            return result

        BreakableCUDAGraphWrapper._replay = replay_piecewise

    @functools.wraps(original_sample)
    def sample(runner, *args, **kwargs):
        try:
            result = original_sample(runner, *args, **kwargs)
        except BaseException as error:
            fail_observation(bindings.get(id(runner)), error)
            raise
        binding = bindings.get(id(runner))
        if binding is not None and binding["pending"] is not None:
            if binding["sampled"] is not None:
                raise RuntimeError("native V2 sample repeated for one observed forward")
            # This copy follows native logits and sampling; it is outside every
            # measured module and the embedding-to-logits whole-GPU window.
            binding["sampled"] = result[0].sampled_token_ids.detach().cpu().tolist()
        return result

    @functools.wraps(original_sample_tokens)
    def sample_tokens(runner, *args, **kwargs):
        try:
            result = original_sample_tokens(runner, *args, **kwargs)
            binding = bindings.get(id(runner))
            if binding is not None and binding["pending"] is not None:
                record, completed = binding["pending"]
                sampled = binding["sampled"]
                if sampled is None or len(sampled) != len(record["requests"]) or any(len(row) != 1 for row in sampled):
                    raise RuntimeError("native V2 sampling receipt is not one token per actual request")
                for request, tokens in zip(record["requests"], sampled, strict=True):
                    request["sampled_token_id"] = int(tokens[0])
                binding["state"].after(record, completed)
                binding["pending"], binding["sampled"] = None, None
            return result
        except BaseException as error:
            fail_observation(bindings.get(id(runner)), error)
            raise

    GPUModelRunner.prepare_inputs = prepare
    GPUModelRunner.execute_model = execute
    GPUModelRunner.sample = sample
    GPUModelRunner.sample_tokens = sample_tokens
    GPUModelRunner._aisim_glm53_ops_installed = True
    output.mkdir(parents=True, exist_ok=True)
    (output / f"worker-activation-{os.getpid()}.json").write_text(
        json.dumps(
            {
                "native_runner": "v2",
                "class": GPUModelRunner.__module__ + "." + GPUModelRunner.__name__,
                "purpose": purpose,
                "piecewise_replay_enabled": piecewise_replay,
                "serving_none_diagnostic_enabled": serving_none and not serving_none_measured,
                "serving_none_measured_enabled": serving_none_measured,
                "pid": os.getpid(),
                "status": "wrappers_installed_no_gpu_execution_claim",
            }
        )
        + "\n"
    )
