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
from collections import defaultdict
from dataclasses import dataclass

from collector.glm53flash_contract import validate_row


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
    quant = getattr(owner, "quant_method", None)
    if quant is not None:
        identity += f"/{type(quant).__module__}.{type(quant).__qualname__}"
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
        self._entries = {
            phase: {entry["name"]: entry for entry in entries} for phase, entries in manifest["phases"].items()
        }

    def begin(self, workload: NativeWorkload) -> None:
        if self.workload is not None or self.events:
            raise RuntimeError("previous native invocation was not finalized")
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("eager collection cannot begin inside CUDA graph capture")
        self.workload = workload

    def wrap(self, owner, method: str, name: str | tuple[str, ...], *, validate_result=None) -> None:
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
                raise RuntimeError("nested compute intervals cannot be summed as disjoint operations")
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

    def wrap_collective(self, owner, method: str) -> None:
        """Witness a blocking same-stream collective without moving or disabling it."""
        original = getattr(owner, method)
        witness = dispatch_identity(owner, method)

        @functools.wraps(original)
        def observed(*args, **kwargs):
            interval = self.active_interval
            if interval is None:
                return original(*args, **kwargs)
            if kwargs.get("async_op", False):
                raise RuntimeError("asynchronous collectives cannot be subtracted from local compute")
            stream = self.torch.cuda.current_stream()
            if stream != interval["stream"]:
                raise RuntimeError("cross-stream collectives require a native fused timing contract")
            if interval.get("inside_collective"):
                return original(*args, **kwargs)
            interval["inside_collective"] = True
            start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
            start.record(stream)
            try:
                result = original(*args, **kwargs)
                end.record(stream)
                interval["collectives"].append((start, end, witness))
                return result
            finally:
                interval["inside_collective"] = False

        setattr(owner, method, observed)
        self.restorations.append((owner, method, original))

    def end(self) -> list[dict]:
        """Finalize one complete observed forward; preserve failed evidence upstream."""
        if self.workload is None or self.active_interval is not None:
            raise RuntimeError("no complete native invocation to finalize")
        workload = self.workload
        self.torch.cuda.synchronize()
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
                else workload.batch_size * workload.query,
                "latency": latency,
                "sample_count": 1,
                "measurement_scope": "local_compute",
                "kernel_source": "+".join(sorted({interval["source"] for interval in intervals})),
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
        for owner, method, original in reversed(self.restorations):
            setattr(owner, method, original)
        self.restorations.clear()


def require_fused_sglang_norm(result) -> None:
    """The pre interval includes RMSNorm only if native hc_pre actually fused it."""
    if not isinstance(result, tuple) or len(result) != 4 or result[3] is not True:
        raise RuntimeError("native SGLang mHC left RMSNorm outside the observed pre interval")
