# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-scheduler SGLang telemetry and optional eager GLM operation observation.

Integration: sgl-project/sglang@94602c9c2b7cbdb8efd5c52802dac6a1c180089e,
python/sglang/srt/{managers/tp_worker.py,model_executor/model_runner.py,
model_executor/forward_batch_info.py,utils/device_timer.py} (Apache-2.0).
Original wrappers invoke these native APIs without copying their implementation.
See README.glm53flash_sglang.md and THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import threading
from collections import deque
from pathlib import Path


def _tokens_digest(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def actual_coordinates(forward_batch) -> dict:
    """Read the native ForwardBatch after its scheduler-owned initialization."""
    mode = forward_batch.forward_mode
    count = int(forward_batch.batch_size)
    ids = list(forward_batch.rids or ())
    if len(ids) != count or len(set(ids)) != count:
        raise RuntimeError("native SGLang forward lacks exact request identities")
    if mode.is_decode():
        if forward_batch.seq_lens_cpu is None:
            raise RuntimeError("native decode lacks its CPU inclusive-sequence mirror")
        lengths = [int(value) for value in forward_batch.seq_lens_cpu.tolist()[:count]]
        queries, prefixes = [1] * count, [length - 1 for length in lengths]
        phase = "generation"
    elif mode.is_extend() and not mode.is_mixed():
        if forward_batch.extend_seq_lens_cpu is None or forward_batch.extend_prefix_lens_cpu is None:
            raise RuntimeError("native prefill lacks its scheduler-derived CPU coordinates")
        queries = list(map(int, forward_batch.extend_seq_lens_cpu[:count]))
        prefixes = list(map(int, forward_batch.extend_prefix_lens_cpu[:count]))
        lengths = [query + prefix for query, prefix in zip(queries, prefixes, strict=True)]
        phase = "context"
    else:
        raise RuntimeError(f"GLM native telemetry does not admit speculative/mixed/split phase {mode}")
    if (
        len(queries) != count
        or len(prefixes) != count
        or any(query <= 0 for query in queries)
        or any(prefix < 0 for prefix in prefixes)
    ):
        raise RuntimeError("invalid native scheduled token coordinates")
    return {
        "phase": phase,
        "native_forward_mode": mode.name,
        "batch_size": count,
        "request_ids": ids,
        "query_lengths": queries,
        "prefix_lengths": prefixes,
        "inclusive_sequence_lengths": lengths,
        "total_new_tokens": sum(queries),
        "total_past_kv_tokens": sum(prefixes),
    }


def allocated_state_inventory(runner) -> dict:
    """Record allocated tensor metadata, without reading cache contents."""
    pool = getattr(runner, "token_to_kv_pool", None)
    mamba = getattr(getattr(pool, "mamba_pool", None), "mamba_cache", None)
    full = getattr(pool, "full_kv_pool", None)
    if mamba is None or full is None:
        return {"admitted": False, "reason": "native hybrid cache objects unavailable"}

    def tensors(value):
        if value is None:
            return []
        values = value if isinstance(value, (list, tuple)) else [value]
        return [
            {
                "dtype": str(t.dtype),
                "shape": list(t.shape),
                "stride": list(t.stride()),
                "device": str(t.device),
                "nbytes": int(t.numel() * t.element_size()),
            }
            for t in values
        ]

    groups = {
        "kda_conv": tensors(mamba.conv),
        "kda_temporal": tensors(mamba.temporal),
        "mla_latent": tensors(full.kv_buffer),
        "pooled_index_packed": tensors(full.index_k_with_scale_buffer),
        "index_tail_key": tensors(getattr(full, "_compress_tail_k", None)),
        "index_tail_score": tensors(getattr(full, "_compress_tail_score", None)),
    }
    expected = {
        "kda_conv": "torch.bfloat16",
        "kda_temporal": "torch.float32",
        "pooled_index_packed": "torch.uint8",
        "index_tail_key": "torch.bfloat16",
        "index_tail_score": "torch.bfloat16",
    }
    admitted = all(groups[name] and all(t["dtype"] == dtype for t in groups[name]) for name, dtype in expected.items())
    logical_kv_dtype = str(full.dtype)
    admitted = admitted and logical_kv_dtype == "torch.float8_e4m3fn"
    return {
        "admitted": admitted,
        "groups": groups,
        "logical_kv_dtype": logical_kv_dtype,
        "physical_kv_dtype": str(full.store_dtype),
        "pooled_index_layout": "packed_fp8_keys_and_fp32_scales",
        "pool_class": f"{type(pool).__module__}.{type(pool).__name__}",
        "full_pool_class": f"{type(full).__module__}.{type(full).__name__}",
    }


class _TraceState:
    def __init__(self, runner, output: Path, provenance: dict, manifest: dict | None, request_manifest=None):
        from sglang.srt.utils.device_timer import DeviceTimer

        self.runner = runner
        self.rank = int(runner.ps.tp_rank)
        self.output = output
        self.provenance = provenance
        self.request_manifest = request_manifest
        self.matched = set()
        self.pending = deque()
        self.records = {}
        self.previous = {}
        self.sampled = {}
        self.counter = 0
        self.observer = None
        self.purpose = os.environ.get("AISIM_GLM53_PURPOSE", "fpm")
        self.whole_events = {}
        self.current_invocation = None
        self.prefill_receipt = None
        self.state_layout = allocated_state_inventory(runner)
        self.state_layout_sha256 = hashlib.sha256(json.dumps(self.state_layout, sort_keys=True).encode()).hexdigest()
        output.mkdir(parents=True, exist_ok=True)
        (output / f"state-layout-rank-{self.rank}.json").write_text(json.dumps(self.state_layout, indent=2))
        native_prefill = getattr(runner, "prefill_cuda_graph_runner", None)
        # Pinned setup aliases this slot to EagerRunner when prefill capture is
        # disabled. EagerRunner.load_batch has no graph backend/padding state.
        if native_prefill is not None and all(
            hasattr(native_prefill, attr) for attr in ("load_batch", "prefill_backend_name", "_is_full_backend")
        ):
            original_load = native_prefill.load_batch

            @functools.wraps(original_load)
            def load_batch(*args, **kwargs):
                result = original_load(*args, **kwargs)
                self.prefill_receipt = {
                    "num_padded_tokens": int(result.input_ids.shape[0]),
                    "native_prefill_backend": str(native_prefill.prefill_backend_name),
                    "runtime_mode": "FULL" if native_prefill._is_full_backend else "PIECEWISE",
                }
                return result

            native_prefill.load_batch = load_batch
        if runner.device_timer is None:
            runner.device_timer = DeviceTimer(self.on_timing)
        else:
            runner.device_timer.add_reporter(self.on_timing)
        self.timer = runner.device_timer
        if self.purpose == "ops_holdout":
            import torch

            if manifest is not None:
                raise RuntimeError("whole-forward holdout cannot install module observers")
            original_model = runner.model.forward

            @functools.wraps(original_model)
            def model_forward(*args, **kwargs):
                invocation = self.current_invocation
                if invocation is None or self.records[invocation]["stage"] != "measure":
                    return original_model(*args, **kwargs)
                if torch.cuda.is_current_stream_capturing() or invocation in self.whole_events:
                    raise RuntimeError("whole-forward eager holdout encountered capture or a repeated forward")
                stream = torch.cuda.current_stream()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(stream)
                result = original_model(*args, **kwargs)
                end.record(stream)
                if torch.cuda.current_stream() != stream:
                    raise RuntimeError("whole-forward native model changed current stream")
                self.whole_events[invocation] = start, end
                return result

            runner.model.forward = model_forward
        if manifest is not None:
            from collector.glm53flash_native_hooks import install_native_hooks
            from collector.glm53flash_observer import NativeOperationObserver

            self.observer = NativeOperationObserver(manifest, provenance, self.rank)
            inventory = install_native_hooks(runner.model, self.observer, "sglang")
            (output / f"inventory-rank-{self.rank}.json").write_text(json.dumps(inventory, indent=2))

    def append(self, name: str, value: dict) -> None:
        filename = f"rank-{self.rank}.jsonl" if name == "ops" else f"{name}-rank-{self.rank}.jsonl"
        with (self.output / filename).open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")

    def on_timing(self, *, t: float, category: str) -> None:
        if not self.pending:
            raise RuntimeError("native DeviceTimer emitted an uncorrelated forward segment")
        invocation = self.pending.popleft()
        record = self.records[invocation]
        expected = "decode" if record["phase"] == "generation" else "extend"
        if category != expected:
            raise RuntimeError(f"native timing category {category!r} disagrees with {expected!r}")
        if t <= 0:
            raise RuntimeError("native DeviceTimer returned a nonpositive observation")
        record.update(native_forward_ms=t * 1000, native_timer_category=category, gpu_completed=True)

    def before(self, forward_batch, requests) -> int:
        self.prefill_receipt = None
        coordinates = actual_coordinates(forward_batch)
        self.counter += 1
        invocation = self.counter
        self.current_invocation = invocation
        snapshots = []
        history_ids = []
        # The overlap scheduler can publish the next input through FutureMap
        # before Req.output_ids advances. Read the resolved native input tensor
        # rather than infer its contents from that lagging host request field.
        actual_inputs = [int(token) for token in forward_batch.input_ids.detach().cpu().reshape(-1).tolist()]
        if len(actual_inputs) != coordinates["total_new_tokens"]:
            raise RuntimeError("native unpadded input tensor differs from scheduled token count")
        input_offset = 0
        completed_tokens = {}
        for request, rid, prefix, query in zip(
            requests,
            coordinates["request_ids"],
            coordinates["prefix_lengths"],
            coordinates["query_lengths"],
            strict=True,
        ):
            if str(request.rid) != rid:
                raise RuntimeError("native ForwardBatch request order differs from scheduler")
            previous = self.previous.get(rid)
            real_history = previous is not None and previous["computed_tokens"] == prefix
            native_query = actual_inputs[input_offset : input_offset + query]
            input_offset += query
            prompt = [int(token) for token in request.origin_input_ids]
            if real_history:
                tokens = previous["tokens"] + native_query
            elif not prefix:
                tokens = native_query
            else:
                tokens = prompt[:prefix] + native_query
                if len(tokens) != prefix + query:
                    raise RuntimeError("unobserved native prefix cannot be reconstructed for admission")
            if tokens[: min(len(prompt), len(tokens))] != prompt[: len(tokens)]:
                raise RuntimeError("resolved native inputs differ from the actual request prompt")
            completed_tokens[rid] = tokens
            history_ids.append(previous["forward_id"] if real_history else "")
            snapshots.append(
                {
                    "request_id": rid,
                    "prompt_token_ids": prompt,
                    "output_token_ids_before": list(self.sampled.get(rid, ())),
                    "native_query_token_ids": native_query,
                    "computed_tokens_before": prefix,
                    "computed_tokens_after": prefix + query,
                    "input_tokens_sha256": _tokens_digest(tokens[: prefix + query]),
                    "previous_forward_id": previous["forward_id"] if real_history else None,
                    "same_request_real_prefix": not prefix or real_history,
                }
            )
        record = {
            **self.provenance,
            **coordinates,
            "invocation": invocation,
            "tp_rank": self.rank,
            "forward_id": f"rank-{self.rank}/forward-{invocation}",
            "requests": snapshots,
            "timing_boundary": "sglang_native_forward_device_timer",
            "ops_instrumented": self.observer is not None,
            "allocated_fake_tokens": 0,
            "state_protocol": "glm53flash_same_request_real_hybrid_v1",
            "state_layout_sha256": self.state_layout_sha256,
            "state_layout_admitted": self.state_layout["admitted"],
        }
        self.records[invocation] = record
        record["_completed_tokens"] = completed_tokens
        self.pending.append(invocation)
        record.update(match_frozen_requests(record, self.request_manifest))
        if record["stage"] == "measure":
            if not self.state_layout["admitted"]:
                raise RuntimeError("actual native hybrid cache dtype/layout is outside the admitted GLM identity")
            key = (record["benchmark_id"], record["repetition"])
            if key in self.matched:
                raise RuntimeError("frozen target matched more than one native forward")
            self.matched.add(key)
        if self.observer is not None and (self.request_manifest is None or record["stage"] == "measure"):
            from collector.glm53flash_observer import NativeWorkload

            if len(set(coordinates["query_lengths"])) != 1 or len(set(coordinates["prefix_lengths"])) != 1:
                raise RuntimeError("Ops physical keys require a homogeneous native batch")
            prefix, query = coordinates["prefix_lengths"][0], coordinates["query_lengths"][0]
            mode = "decode" if coordinates["phase"] == "generation" else "chunked_prefill" if prefix else "full_prefill"
            self.observer.begin(
                NativeWorkload(
                    coordinates["phase"],
                    coordinates["batch_size"],
                    query,
                    prefix,
                    mode,
                    tuple(coordinates["request_ids"]),
                    tuple(history_ids) if prefix else (),
                    record.get("repetition", 0),
                    invocation,
                )
            )
            record["ops_observed"] = True
        return invocation

    def after(self, invocation: int, result) -> None:
        record = self.records[invocation]
        graph = bool(result.can_run_graph)
        mode = "FULL" if graph and record["phase"] == "generation" else "PIECEWISE" if graph else "NONE"
        record.update(used_cuda_graph=graph, runtime_mode=mode)
        if graph and self.purpose == "ops_holdout":
            raise RuntimeError("whole-forward eager holdout encountered actual graph dispatch")
        if graph and record["phase"] == "generation":
            native_graph = self.runner.decode_cuda_graph_runner
            key = native_graph._replay_graph_key
            record["native_graph_key"] = str(key)
            record["num_padded_tokens"] = int(key.size) * int(native_graph.captured_req_width)
        elif not graph:
            record["num_padded_tokens"] = record["total_new_tokens"]
        else:
            if self.prefill_receipt is None:
                raise RuntimeError("native prefill graph dispatch lacks its actual load_batch padding receipt")
            record.update(self.prefill_receipt)
        if self.observer is not None and record.get("ops_observed"):
            if graph:
                raise RuntimeError("native eager Ops campaign actually replayed a CUDA graph")
            for row in self.observer.end():
                row.update(
                    {
                        key: record.get(key)
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
        for request in record["requests"]:
            self.previous[request["request_id"]] = {
                "forward_id": record["forward_id"],
                "computed_tokens": request["computed_tokens_after"],
                "tokens_sha256": request["input_tokens_sha256"],
                "tokens": record["_completed_tokens"][request["request_id"]],
            }

    def finish_worker(self, invocation: int, result) -> None:
        # Sampling is outside DeviceTimer's native forward interval. Reading
        # the actual sampled IDs also makes its GPU completion observable.
        sampled = getattr(result, "next_token_ids", None)
        if sampled is None or getattr(result, "delay_sample_func", None) is not None:
            raise RuntimeError("native telemetry requires completed normal sampling, not delayed/speculative sampling")
        ids = sampled.detach().cpu().reshape(-1).tolist()
        self.timer._report()
        record = self.records[invocation]
        if not record.get("gpu_completed"):
            raise RuntimeError("native forward completion did not produce its DeviceTimer receipt")
        if len(ids) != record["batch_size"]:
            raise RuntimeError("actual sampled IDs do not match the native batch")
        if record["stage"] == "measure" and record["num_padded_tokens"] is None:
            raise RuntimeError("actual piecewise graph padding is unknown; frozen target cannot be admitted")
        for request, token in zip(record["requests"], ids, strict=True):
            request["sampled_token_id"] = int(token)
            self.sampled.setdefault(request["request_id"], []).append(int(token))
        if self.purpose == "ops_holdout" and record["stage"] == "measure":
            if invocation not in self.whole_events:
                raise RuntimeError("whole-forward holdout lacks its native model interval")
            start, end = self.whole_events.pop(invocation)
            record["whole_forward_gpu_ms"] = start.elapsed_time(end)
            if record["whole_forward_gpu_ms"] <= 0:
                raise RuntimeError("whole-forward GPU interval must be positive")
            record["whole_forward_boundary"] = "embedding_to_logits_gpu_v1"
        record.pop("_completed_tokens")
        self.current_invocation = None
        self.append("forward", record)
        del self.records[invocation]


def match_frozen_requests(record: dict, manifest: dict | None) -> dict:
    """Label only exact native occurrences; every other forward remains seed evidence."""
    seed = {"stage": "seed", "sampling_role": "unclassified"}
    if manifest is None:
        return seed
    mappings = manifest.get("requests", {})
    entries = [mappings.get(rid) for rid in record["request_ids"]]
    if not entries or any(entry is None for entry in entries):
        return seed
    fields = (
        "benchmark_id",
        "repetition",
        "sampling_role",
        "target_phase",
        "target_query",
        "target_prefix",
        "target_batch_size",
    )
    if len({tuple(entry[field] for field in fields) for entry in entries}) != 1:
        return seed
    entry = entries[0]
    identity = {key: entry[key] for key in ("benchmark_id", "repetition", "sampling_role")}
    if entry["sampling_role"] not in ("warmup", "measurement"):
        raise RuntimeError("frozen request manifest has an invalid sampling role")
    if (
        record["phase"] != entry["target_phase"]
        or record["batch_size"] != entry["target_batch_size"]
        or record["query_lengths"] != [entry["target_query"]] * record["batch_size"]
        or record["prefix_lengths"] != [entry["target_prefix"]] * record["batch_size"]
        or not all(request["same_request_real_prefix"] for request in record["requests"])
    ):
        return {**seed, **identity}
    return {
        **identity,
        "stage": "measure",
        "request_set": manifest["request_set"],
        "dataset_role": manifest["dataset_role"],
        "corpus_sha256": manifest["corpus_sha256"],
    }


def install() -> None:
    """Install in each native worker before request execution, via sitecustomize."""
    from importlib.metadata import version

    if version("sglang") != "0.5.20":
        raise RuntimeError("GLM native serving observer requires SGLang 0.5.20")
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.model_runner import ModelRunner

    if getattr(ModelRunner, "_aisim_glm53_observer_installed", False):
        return
    output = Path(os.environ["AISIM_GLM53_TRACE_DIR"])
    provenance = json.loads(Path(os.environ["AISIM_GLM53_PROVENANCE"]).read_text())
    manifest_path = os.environ.get("AISIM_GLM53_OPS_MANIFEST")
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else None
    request_manifest_path = os.environ.get("AISIM_GLM53_REQUEST_MANIFEST")
    request_manifest = json.loads(Path(request_manifest_path).read_text()) if request_manifest_path else None
    current = threading.local()
    states = {}
    original_forward = ModelRunner.forward
    original_worker = TpModelWorker.forward_batch_generation

    @functools.wraps(original_forward)
    def forward(runner, forward_batch, *args, **kwargs):
        context = getattr(current, "context", None)
        if context is None or runner.is_draft_worker:
            return original_forward(runner, forward_batch, *args, **kwargs)
        state = states.get(id(runner))
        if state is None:
            state = states[id(runner)] = _TraceState(runner, output, provenance, manifest, request_manifest)
        invocation = state.before(forward_batch, context["requests"])
        context["calls"].append((state, invocation))
        try:
            result = original_forward(runner, forward_batch, *args, **kwargs)
            state.after(invocation, result)
            return result
        except BaseException as error:
            state.append(
                "failed", {**state.records[invocation], "error_type": type(error).__name__, "error": str(error)}
            )
            raise

    @functools.wraps(original_worker)
    def worker(worker_self, batch, *args, **kwargs):
        if batch is None:
            return original_worker(worker_self, batch, *args, **kwargs)
        if getattr(current, "context", None) is not None:
            raise RuntimeError("nested native worker forward is outside GLM's initial contract")
        context = {"requests": tuple(batch.reqs), "calls": []}
        current.context = context
        try:
            result = original_worker(worker_self, batch, *args, **kwargs)
            if len(context["calls"]) != 1:
                raise RuntimeError("normal native serving request did not execute exactly one target forward")
            state, invocation = context["calls"][0]
            state.finish_worker(invocation, result)
            return result
        finally:
            current.context = None

    ModelRunner.forward = forward
    TpModelWorker.forward_batch_generation = worker
    ModelRunner._aisim_glm53_observer_installed = True
