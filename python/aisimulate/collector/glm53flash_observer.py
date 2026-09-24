# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe unmodified native GLM operations during real serving requests.

The serving adapter owns request scheduling, hybrid state, and CUDA graph
policy. This observer never initializes a cache, substitutes tensor inputs,
changes dispatch, or changes a module's collective/stream settings. The first
integration is explicitly eager; a graph replay cannot masquerade as a fresh
Python module invocation and therefore fails complete-coverage admission.
"""

from __future__ import annotations

import functools
import json
import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass

from collector.glm53flash_contract import sha256_json, validate_native_workload, validate_row


@dataclass(frozen=True)
class NativeWorkload:
    phase: str
    batch_size: int
    query: int
    prefix: int
    state_mode: str
    request_ids: tuple[str, ...]
    history_ids: tuple[str, ...]
    sample: int
    invocation: int
    used_cuda_graph: bool = False

    def __post_init__(self):
        if self.phase not in ("context", "generation") or self.batch_size <= 0 or self.query <= 0 or self.prefix < 0:
            raise ValueError("invalid native workload coordinates")
        if self.prefix + self.query > 131072:
            raise ValueError("native workload exceeds inclusive 128K context")
        if self.phase == "generation" and self.query != 1:
            raise ValueError("initial decode contract excludes speculative/multiple-token steps")
        if len(self.request_ids) != self.batch_size or len(set(self.request_ids)) != self.batch_size:
            raise ValueError("every native request needs its own request identity")
        if (self.prefix or self.phase == "generation") and (
            len(self.history_ids) != self.batch_size or not all(self.history_ids)
        ):
            raise ValueError("cached/decode observations require actual prefix execution receipts")
        if self.used_cuda_graph:
            raise ValueError("eager observer cannot label a graph replay; capture-bound events are required")


def dispatch_identity(owner, method: str) -> str:
    """Identify the method actually called and its loaded quantization method."""
    native = getattr(owner, method)
    function = getattr(native, "__func__", native)
    identity = f"{function.__module__}.{function.__qualname__}"
    # These pinned native forwarding methods call one CustomOp, not their
    # owning DecoderLayer's unrelated attention/FFN descendants. Keep the
    # actually loaded forward implementation in the source witness.
    if identity in {
        f"vllm.models.glm5next.nvidia.model.Glm5NextDecoderLayer.{name}"
        for name in ("hc_pre", "hc_post", "hc_fused_post_pre")
    }:
        child_name = f"m{method}_op"
        selected = getattr(getattr(owner, child_name), "_forward_method", None)
        if selected is None or not hasattr(selected, "__qualname__"):
            raise RuntimeError("native vLLM mHC selected CustomOp forward is missing")
        return f"{identity}/{child_name}:forward={selected.__module__}.{selected.__qualname__}"
    quant = getattr(owner, "quant_method", None)
    if quant is not None:
        identity += f"/{type(quant).__module__}.{type(quant).__qualname__}"
    children = getattr(owner, "named_modules", None)
    if children is not None:
        dispatches = set()
        for name, module in children():
            for attribute in ("quant_method", "moe_runner", "moe_kernel"):
                selected = getattr(module, attribute, None)
                if selected is not None:
                    dispatches.add(f"{name}:{attribute}={type(selected).__module__}.{type(selected).__qualname__}")
            selected = getattr(module, "_forward_method", None)
            if selected is not None and hasattr(selected, "__qualname__"):
                dispatches.add(f"{name}:forward={selected.__module__}.{selected.__qualname__}")
        if dispatches:
            identity += "/" + ";".join(sorted(dispatches))
    return identity


class NativeOperationObserver:
    """CUDA-event intervals attached to native objects by a versioned adapter.

    Multiple disjoint parts may contribute to one graph occurrence, e.g. a
    sparse MLA latent projection hoisted into SGLang's communicator. Nested
    compute intervals are rejected rather than double-counted. A separately
    witnessed collective may be subtracted only when it executes synchronously
    on the enclosing interval's stream. Runtime overlap/fused collectives need
    their own measured contract, not an arithmetic correction here.
    """

    def __init__(self, manifest: dict, provenance: dict, tp_rank: int, *, torch_module=None):
        if torch_module is None:
            import torch as torch_module

        self.torch = torch_module
        self.manifest = manifest
        self.provenance = dict(provenance)
        self.tp_rank = tp_rank
        self.workload = None
        self.events = []
        self.active_interval = None
        self.restorations = []
        self.collective_calls = 0
        self.inside_collective = False
        self.profiler = None
        self.dispatches = {}
        self._entries = {
            phase: {entry["name"]: entry for entry in entries} for phase, entries in manifest["phases"].items()
        }

    def begin(self, workload: NativeWorkload) -> None:
        validate_native_workload(self.provenance["backend"], workload.phase, workload.prefix, workload.query)
        if self.workload is not None or self.events:
            raise RuntimeError("previous native invocation was not finalized")
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("eager collection cannot begin inside CUDA graph capture")
        self.workload = workload
        self.collective_calls = 0
        # The diagnostic CUPTI pass is the last excluded warmup. Timed retained
        # samples run without the profiler; missing native kernel attribution
        # disables interpolation rather than substituting method names.
        if os.environ.get("AISIM_GLM53_DISPATCH_PROFILING") == "1" and workload.sample == 4:
            self.profiler = self.torch.profiler.profile(
                activities=[
                    self.torch.profiler.ProfilerActivity.CPU,
                    self.torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            self.profiler.start()

    def _range(self, name):
        return (
            self.torch.profiler.record_function("aisim.glm53/" + name) if self.profiler is not None else nullcontext()
        )

    def _dispatch_key(self, name):
        w = self.workload
        return w.phase, w.batch_size, w.query, w.prefix, name

    def _finish_profile(self):
        if self.profiler is None:
            return
        profiler, self.profiler = self.profiler, None
        profiler.stop()
        kernels = defaultdict(list)
        for event in profiler.events():
            launched = getattr(event, "kernels", ())
            if not launched:
                continue
            parent = event
            while parent is not None and not parent.name.startswith("aisim.glm53/"):
                parent = getattr(parent, "cpu_parent", None)
            if parent is not None:
                kernels[parent.name.removeprefix("aisim.glm53/")].extend(kernel.name for kernel in launched)
        for name in self._entries[self.workload.phase]:
            names = sorted(kernels.get(name, []))
            self.dispatches[self._dispatch_key(name)] = names

    def wrap(
        self,
        owner,
        method: str,
        name: str | tuple[str, ...],
        *,
        validate_result=None,
        included_by_same_operation: bool = False,
    ) -> None:
        """Wrap an existing callable without altering its arguments or result."""
        names = (name,) if isinstance(name, str) else name
        if not names or any(item not in entries for item in names for entries in self._entries.values()):
            raise ValueError(f"native hook {name!r} is absent from a production phase")
        original = getattr(owner, method)
        witness = dispatch_identity(owner, method)
        last_workload, calls = None, 0

        @functools.wraps(original)
        def observed(*args, **kwargs):
            nonlocal last_workload, calls
            if self.workload is None:
                return original(*args, **kwargs)
            if last_workload is not self.workload:
                last_workload, calls = self.workload, 0
            if calls >= len(names):
                raise RuntimeError(f"native callable {witness} exceeded its declared graph occurrences")
            selected_name = names[calls]
            calls += 1
            if self.torch.cuda.is_current_stream_capturing():
                raise RuntimeError("eager operation observer encountered native graph capture")
            if self.active_interval is not None:
                if not included_by_same_operation or self.active_interval["name"] != selected_name:
                    raise RuntimeError("nested compute intervals cannot be summed as disjoint operations")
                # SGLang can evaluate the saved latent projection lazily inside
                # its own attention forward. The enclosing physical operation
                # already times this call; a second interval would double-count
                # it. Only this explicitly declared same-operation callback may
                # share the interval. Different operations still fail above.
                interval = self.active_interval
                result = original(*args, **kwargs)
                if self.torch.cuda.current_stream() != interval["stream"]:
                    raise RuntimeError("included native callback changed its current stream")
                if validate_result is not None:
                    validate_result(result)
                interval.setdefault("included_native_calls", []).append(witness)
                return result
            stream = self.torch.cuda.current_stream()
            start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
            interval = {
                "name": selected_name,
                "start": start,
                "end": end,
                "stream": stream,
                "collectives": [],
                "source": witness,
            }
            self.active_interval = interval
            start.record(stream)
            try:
                with self._range(selected_name):
                    result = original(*args, **kwargs)
                end.record(stream)
                if self.torch.cuda.current_stream() != stream:
                    raise RuntimeError("native operation changed its current stream")
                if validate_result is not None:
                    validate_result(result)
                self.events.append(interval)
                return result
            finally:
                self.active_interval = None

        setattr(owner, method, observed)
        self.restorations.append((owner, method, original))

    def wrap_collective(self, owner, method: str, names: tuple[str, ...] = ()) -> None:
        """Partition witnessed blocking collectives from native local modules."""
        if any(name not in entries for name in names for entries in self._entries.values()):
            raise ValueError("native collective is absent from the production graph")
        original = getattr(owner, method)
        witness = dispatch_identity(owner, method)

        @functools.wraps(original)
        def observed(*args, **kwargs):
            interval = self.active_interval
            if self.workload is None or self.inside_collective or (interval is None and not names):
                return original(*args, **kwargs)
            if kwargs.get("async_op", False):
                raise RuntimeError("asynchronous collectives cannot be partitioned from local compute")
            stream = self.torch.cuda.current_stream()
            if interval is not None and stream != interval["stream"]:
                raise RuntimeError("cross-stream collectives require a native fused timing contract")
            selected = None
            if names:
                if self.collective_calls >= len(names):
                    raise RuntimeError("native collective exceeded declared graph occurrences")
                selected = names[self.collective_calls]
                self.collective_calls += 1
            self.inside_collective = True
            start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
            start.record(stream)
            try:
                with self._range(selected or "unbound_collective"):
                    result = original(*args, **kwargs)
                end.record(stream)
                if interval is not None:
                    interval["collectives"].append((start, end, witness))
                if selected is not None:
                    self.events.append(
                        {
                            "name": selected,
                            "start": start,
                            "end": end,
                            "stream": stream,
                            "collectives": [],
                            "source": witness,
                        }
                    )
                return result
            finally:
                self.inside_collective = False

        setattr(owner, method, observed)
        self.restorations.append((owner, method, original))

    def end(self) -> list[dict]:
        """Finalize one complete observed forward; preserve failed evidence upstream."""
        if self.workload is None or self.active_interval is not None:
            raise RuntimeError("no complete native invocation to finalize")
        workload = self.workload
        self.torch.cuda.synchronize()
        self._finish_profile()
        observed = defaultdict(list)
        for interval in self.events:
            observed[interval["name"]].append(interval)
        expected = self._entries[workload.phase]
        if set(observed) != set(expected):
            missing = sorted(set(expected) - set(observed))
            raise RuntimeError(
                f"incomplete native operation coverage (graph replay cannot use eager events): {missing}"
            )
        rows = []
        for name, intervals in observed.items():
            entry = expected[name]
            attention = entry["component"] == "attention"
            geometry = json.loads(entry["geometry"])
            primitive = entry["component"] == "primitive"
            role = geometry.get("role") if primitive else None
            latency = 0.0
            excluded = []
            for interval in intervals:
                local_ms = interval["start"].elapsed_time(interval["end"])
                for start, end, source in interval["collectives"]:
                    duration = start.elapsed_time(end)
                    local_ms -= duration
                    excluded.append({"source": source, "latency": duration})
                if local_ms <= 0:
                    raise RuntimeError("native collective subtraction left a nonpositive local interval")
                latency += local_ms
            row = {
                **self.provenance,
                **entry,
                "batch_size": workload.batch_size if attention else 1,
                "prefix": workload.prefix if attention and workload.phase == "context" else 0,
                "x": (workload.query if workload.phase == "context" else workload.prefix)
                if attention
                else workload.batch_size
                if primitive and geometry["token_selection"] == "last_per_request"
                else workload.batch_size * workload.query,
                "latency": latency,
                "sample_count": 1,
                "dispatch_kernels": self.dispatches.get(self._dispatch_key(name), []),
                "dispatch_fingerprint": sha256_json(self.dispatches[self._dispatch_key(name)])
                if self.dispatches.get(self._dispatch_key(name))
                else "",
                "measurement_scope": "communication"
                if role == "allreduce"
                else "compute_and_communication"
                if role == "logits"
                else "local_compute",
                "kernel_source": "+".join(
                    sorted(
                        {
                            source
                            for interval in intervals
                            for source in (interval["source"], *interval.get("included_native_calls", []))
                        }
                    )
                ),
                "used_cuda_graph": False,
                "kv_seed_regime": "real_kv"
                if attention and (workload.prefix or workload.phase == "generation")
                else "empty"
                if attention
                else "n/a",
                "state_mode": workload.state_mode if attention else "token_only",
                "phase": workload.phase,
                "sample": workload.sample,
                "invocation": workload.invocation,
                "tp_rank": self.tp_rank,
                "request_ids": workload.request_ids,
                "history_ids": workload.history_ids,
                "excluded_collectives": excluded,
            }
            validate_row(row)
            rows.append(row)
        self.events.clear()
        self.workload = None
        return rows

    def close(self) -> None:
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
        for owner, method, original in reversed(self.restorations):
            setattr(owner, method, original)
        self.restorations.clear()


def require_fused_sglang_norm(result) -> None:
    """The pre interval includes RMSNorm only if native hc_pre actually fused it."""
    if not isinstance(result, tuple) or len(result) != 4 or result[3] is not True:
        raise RuntimeError("native SGLang mHC left RMSNorm outside the observed pre interval")
