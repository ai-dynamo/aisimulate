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


def native_v2_coordinates(runner, scheduler_output, batch) -> dict:
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
        or batch.num_tokens_after_padding != batch.num_tokens
        or batch.num_draft_tokens
        or gpu_lengths != [p + q for p, q in zip(prefixes, queries, strict=True)]
        or queries != [scheduler_output.num_scheduled_tokens[rid] for rid in ids]
        or list(map(int, batch.idx_mapping_np)) != [runner.req_states.req_id_to_index[rid] for rid in ids]
    ):
        raise RuntimeError("native V2 scheduled tokens/slots/padding differ from actual inputs")
    phase = phases.pop()
    if phase == "generation" and queries[0] != 1:
        raise RuntimeError("native V2 Ops excludes speculative decode")
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

    def before(self, scheduler_output, input_ids, forward_context, *, native_batch=None):
        from collector.glm53flash_contract import validate_native_workload
        from collector.glm53flash_observer import NativeWorkload

        context_receipt = native_context_receipt(self.runner)
        coords = (
            native_coordinates(self.runner, scheduler_output)
            if native_batch is None
            else native_v2_coordinates(self.runner, scheduler_output, native_batch)
        )
        for prefix, query in zip(coords["prefix_lengths"], coords["query_lengths"], strict=True):
            validate_native_workload("vllm", coords["phase"], prefix, query, self.provenance["backend_version"])
            if prefix + query > context_receipt["context_policy"]["measured_context_limit"]:
                raise RuntimeError("actual native forward exceeds frozen measured context")
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
            "used_cuda_graph": False,
            "num_padded_tokens": len(tokens),
            "allocated_fake_tokens": 0,
            "state_layout_sha256": self.layout_sha256,
            "state_layout_admitted": self.layout["admitted"],
            "state_protocol": "glm53flash_same_request_real_hybrid_v1",
            "ops_instrumented": self.observer is not None,
        }
        mapping = json.loads(Path(os.environ["AISIM_GLM53_REQUEST_MANIFEST"]).read_text())
        if any(rid not in mapping.get("requests", {}) for rid in record["request_ids"]):
            raise RuntimeError("native serving request is absent from the frozen request manifest")
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
            # Calibration also needs one coherent native rank timeline. The
            # window ends at logits, before sampling/readback, exactly as the
            # independent uninstrumented holdout window does.
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


def install():
    """Call in each worker before execution; model hooks install after native load."""
    from importlib.metadata import version

    from collector.glm53flash_runtime_identity import validate_backend_version

    backend_version = validate_backend_version("vllm", version("vllm"))
    from vllm.forward_context import get_forward_context
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_aisim_glm53_ops_installed", False):
        return
    output = Path(os.environ["AISIM_GLM53_TRACE_DIR"])
    provenance = json.loads(Path(os.environ["AISIM_GLM53_PROVENANCE"]).read_text())
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


def install_v2():
    """Bind the pinned native V2 model forward and its later logits/sample step."""
    from importlib.metadata import version

    from collector.glm53flash_runtime_identity import validate_backend_version

    backend_version = validate_backend_version("vllm", version("vllm"))
    from vllm.forward_context import get_forward_context
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_aisim_glm53_ops_installed", False):
        return
    output = Path(os.environ["AISIM_GLM53_TRACE_DIR"])
    provenance = json.loads(Path(os.environ["AISIM_GLM53_PROVENANCE"]).read_text())
    if provenance.get("backend_version") != backend_version:
        raise RuntimeError("actual native worker package differs from frozen Ops provenance")
    manifest_path = os.environ.get("AISIM_GLM53_OPS_MANIFEST")
    purpose = os.environ.get("AISIM_GLM53_PURPOSE")
    if purpose not in ("ops", "ops_holdout") or bool(manifest_path) != (purpose == "ops"):
        raise RuntimeError("native V2 instrumentation must match explicit Ops/holdout purpose")
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else None
    original_execute = GPUModelRunner.execute_model
    original_prepare = GPUModelRunner.prepare_inputs
    original_sample = GPUModelRunner.sample
    original_sample_tokens = GPUModelRunner.sample_tokens
    current, bindings = threading.local(), {}

    @functools.wraps(original_prepare)
    def prepare(runner, scheduler_output, *args, **kwargs):
        batch = original_prepare(runner, scheduler_output, *args, **kwargs)
        if getattr(runner, "_aisim_glm53_ops_warming_up", False):
            return batch
        if getattr(current, "scheduler_output", None) is not scheduler_output:
            raise RuntimeError("native V2 InputBatch has no observed scheduling boundary")
        binding = bindings.get(id(runner))
        if binding is None:
            state = _TraceState(runner, output, provenance, manifest)
            binding = bindings[id(runner)] = {"state": state, "pending": None, "sampled": None}
            original_forward = runner.model.forward

            @functools.wraps(original_forward)
            def forward(*forward_args, **forward_kwargs):
                schedule = getattr(current, "scheduler_output", None)
                if schedule is None:
                    return original_forward(*forward_args, **forward_kwargs)
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

            runner.model.forward = forward
        if binding["pending"] is not None:
            raise RuntimeError("native V2 previous forward did not complete its sampling step")
        binding["batch"] = batch
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
        try:
            result = original_execute(runner, scheduler_output, *args, **kwargs)
            if scheduler_output.total_num_scheduled_tokens:
                binding = bindings.get(id(runner))
                if binding is None or binding["pending"] is None:
                    raise RuntimeError("native V2 scheduled tokens bypassed the observed eager model call")
            return result
        finally:
            current.scheduler_output = None

    @functools.wraps(original_sample)
    def sample(runner, *args, **kwargs):
        result = original_sample(runner, *args, **kwargs)
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
                "pid": os.getpid(),
                "status": "wrappers_installed_no_gpu_execution_claim",
            }
        )
        + "\n"
    )
