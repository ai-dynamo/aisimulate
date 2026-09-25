# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe native breakable captures without replacing their graphs or replay loop.

Original integration for vllm-project/vllm@ced6857afa0ea7b2e3f0846a62e1394e90f15607,
``vllm/compilation/breakable_cudagraph.py`` (Apache-2.0). Each native segment
retains its own node registry; one operation may own nodes in several segments
and an eager callable. These are capture identities, never measured latencies.
See THIRD_PARTY_NOTICES.md. No upstream compute implementation is copied.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
import threading
from pathlib import Path

from collector.glm53flash_graph_hooks import NativeGraphOperationObserver

BREAKABLE_SOURCE_PIN = "3cc427612a08e2b9b3fee47548026400c1d0776e2d4747535e59ef5512bdf1e8"
EAGER_RANGE_PREFIX = "aisim.glm53/vllm_piecewise_eager/"


def _require_source(method):
    source = inspect.getsourcefile(method)
    if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != BREAKABLE_SOURCE_PIN:
        raise RuntimeError("piecewise capture methods differ from the reviewed native source")


class PiecewiseCaptureRegistry:
    """Keep physical operation ownership across native graph/eager transitions."""

    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.capture = None
        self.current = None
        self.stack = []
        self.calls = []
        self.graphs = []
        self.eager = []
        self.profiling = False

    def begin_segment(self, capture):
        if self.current is not None or (self.capture is not None and self.capture is not capture):
            raise RuntimeError("piecewise capture changed owner or overlapped graph segments")
        if not capture._capturing or capture._current_graph is None:
            raise RuntimeError("piecewise segment must begin after its actual native capture")
        self.capture = capture
        self.current = {"graph": capture._current_graph, "identity": None, "nodes": {}, "last": None}
        self._flush()

    def _flush(self):
        current = self.current
        if current is None:
            return  # Native eager break: there is no graph to query or synthesize.
        value = self.snapshot()
        identity = (value["capture_id"], value["graph_id"])
        if current["identity"] is None:
            if any(identity[0] == row["identity"][0] or identity[1] == row["identity"][1] for row in self.graphs):
                raise RuntimeError("piecewise segment reused an earlier native capture identity")
            current["identity"] = identity
        if current["identity"] != identity or not current["nodes"].keys() <= value["nodes"].keys():
            raise RuntimeError("piecewise graph identity changed or lost previously observed nodes")
        owner = self.stack[-1] if self.stack else None
        for node in value["nodes"].keys() - current["nodes"].keys():
            current["nodes"][node] = {
                "name": owner["name"] if owner else "native_graph_setup",
                "call_index": owner["index"] if owner else None,
                **value["nodes"][node],
            }
        for node, observed in current["nodes"].items():
            if observed["node_type"] != value["nodes"][node]["node_type"]:
                raise RuntimeError("piecewise node changed its actual source type")
        current["last"] = value

    def enter(self, name, source):
        self._flush()
        token = {"name": name, "source": source, "index": len(self.calls), "completed": False}
        self.calls.append(token)
        self.stack.append(token)
        return token

    def leave(self, token):
        if not self.stack or self.stack[-1] is not token:
            raise RuntimeError("piecewise operation scopes are not properly nested")
        self._flush()
        token["completed"] = True
        self.stack.pop()

    def before_segment_end(self, capture):
        if self.capture is not capture or self.current is None or self.current["graph"] is not capture._current_graph:
            raise RuntimeError("piecewise native segment end lacks its original graph")
        self._flush()
        current, self.current = self.current, None
        return current

    def after_segment_end(self, capture, current):
        if capture._capturing or capture._current_graph is not None or not capture.segments:
            raise RuntimeError("piecewise native segment did not finish normally")
        replay = capture.segments[-1]
        graph = current["graph"]
        if getattr(replay, "__self__", None) is not graph or getattr(replay, "__func__", None) is not getattr(
            graph.replay, "__func__", None
        ):
            raise RuntimeError("piecewise native segment did not retain its original graph replay")
        current["replay"] = replay
        current["position"] = len(capture.segments) - 1
        self.graphs.append(current)

    def wrap_eager(self, fn, record_function):
        if self.current is None or not callable(fn):
            raise RuntimeError("piecewise eager break lacks an enclosing native graph segment")
        index = len(self.eager)
        owner = self.stack[-1] if self.stack else None
        source = inspect.getsourcefile(fn)
        if source is None:
            raise RuntimeError("piecewise eager callable lacks an inspectable native source")
        source = Path(source).resolve()
        if hashlib.sha256(source.read_bytes()).hexdigest() != BREAKABLE_SOURCE_PIN:
            raise RuntimeError("piecewise eager callable differs from the pinned native break source")
        binding = {
            "eager_id": index,
            "name": owner["name"] if owner else "native_graph_setup",
            "call_index": owner["index"] if owner else None,
            "source_file": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "qualname": fn.__qualname__,
            "range": EAGER_RANGE_PREFIX + str(index),
            "original": fn,
        }

        @functools.wraps(fn)
        def wrapped():
            # Capture-time execution remains native initialization. Only an
            # explicitly active measured replay receives a CPU profiler scope.
            if not self.profiling:
                return fn()
            with record_function(binding["range"]):
                return fn()

        binding["wrapped"] = wrapped
        self.eager.append(binding)
        return wrapped

    def bind_eager(self, capture, wrapped):
        if self.capture is not capture or not capture.segments or capture.segments[-1] is not wrapped:
            raise RuntimeError("piecewise native break did not retain its original observed callable")
        binding = next(row for row in self.eager if row["wrapped"] is wrapped)
        if "position" in binding:
            raise RuntimeError("piecewise eager callable was bound more than once")
        binding["position"] = len(capture.segments) - 1

    def finish(self, capture):
        if self.capture is not capture or self.current is not None or capture._capturing or self.stack:
            raise RuntimeError("piecewise capture did not close every native graph and operation")
        if not self.graphs or any(not row["completed"] for row in self.calls):
            raise RuntimeError("piecewise capture lacks complete native operations")
        bindings = [(row["replay"], "graph", row) for row in self.graphs]
        bindings += [(row["wrapped"], "eager", row) for row in self.eager]
        if len(bindings) != len(capture.segments) or len({id(fn) for fn in capture.segments}) != len(bindings):
            raise RuntimeError("piecewise native callable list is incomplete or duplicated")
        if capture.num_graphs != len(self.graphs) or capture.num_eager_breaks != len(self.eager):
            raise RuntimeError("piecewise native segment counters disagree with observed callables")
        result = []
        for position, fn in enumerate(capture.segments):
            found = [(kind, row) for original, kind, row in bindings if original is fn]
            if len(found) != 1:
                raise RuntimeError("piecewise callable has no unique actual capture binding")
            kind, row = found[0]
            if row.get("position") != position:
                raise RuntimeError("piecewise native callable changed its observed capture position")
            if kind == "graph":
                value = {
                    "capture_id": row["identity"][0],
                    "graph_id": row["identity"][1],
                    "nodes": [{"node_id": node, **owner} for node, owner in sorted(row["nodes"].items())],
                    "edges": row["last"]["edges"],
                }
            else:
                value = {key: item for key, item in row.items() if key not in ("wrapped", "original")}
            result.append({"position": position, "kind": kind, **value})
        # Retain actual object identities in memory; they are rechecked before
        # replay rather than reconstructed from serialized integer positions.
        self.native_segments = tuple(capture.segments)
        return {"segments": result, "calls": self.calls, "graph_mutations": False, "formal_admission": False}

    def validate_replay(self, capture):
        if (
            self.capture is not capture
            or not hasattr(self, "native_segments")
            or len(capture.segments) != len(self.native_segments)
            or any(
                current is not original
                for current, original in zip(capture.segments, self.native_segments, strict=True)
            )
        ):
            raise RuntimeError("piecewise replay changed the native initialized callable list")


class NativePiecewiseGraphObserver(NativeGraphOperationObserver):
    """Shared native module hooks for FULL graphs and breakable PIECEWISE graphs."""

    def start_piecewise(self, descriptor):
        if self.registry is not None or not dataclasses.is_dataclass(descriptor):
            raise RuntimeError("piecewise observation requires one original native descriptor")
        shape = dataclasses.asdict(descriptor)
        size = shape.get("num_tokens")
        if (
            type(size) is not int
            or size < 1
            or shape
            != {"num_tokens": size, "num_reqs": None, "uniform": False, "has_lora": False, "num_active_loras": 0}
            or shape["uniform"] is not False
            or shape["has_lora"] is not False
            or type(shape["num_active_loras"]) is not int
        ):
            raise RuntimeError("piecewise descriptor lacks its physical padded token count")
        if set(self.entries["context"]) != set(self.entries["generation"]):
            raise RuntimeError("piecewise phase manifests differ in physical operation identity")
        self.shape = shape
        self.collective_calls = 0
        self.registry = PiecewiseCaptureRegistry(
            lambda: self.api.snapshot(self.torch.cuda.current_stream().cuda_stream)
        )

    def _active(self):
        if isinstance(self.registry, PiecewiseCaptureRegistry):
            return self.registry
        return super()._active()

    def finish_piecewise(self, capture):
        registry = self.registry
        if not isinstance(registry, PiecewiseCaptureRegistry):
            raise RuntimeError("piecewise completion lacks its initialization observer")
        result = registry.finish(capture)
        expected = set(self.entries["context"]) - {"logits"}
        actual = {row["name"] for row in result["calls"]}
        if actual != expected:
            raise RuntimeError("piecewise capture lacks complete native hidden-state operation coverage")
        result.update(
            native_shape_key=self.shape,
            tp_rank=self.tp_rank,
            physical_padded_tokens=self.shape["num_tokens"],
            capture_scope="vllm_piecewise_hidden_states",
            uncaptured_operations=["logits"],
            provenance=self.provenance,
            native_api_libraries=self.api.libraries,
            operations=list(self.entries["context"].values()),
            measurement_method="native_piecewise_capture_ownership",
            accuracy_acceptance="NOT_EVALUATED",
        )
        self.registry = None
        return registry, result


def install_segment_observation(capture_class, current_observer, record_function):
    """Wrap native segment boundaries; the original methods own every transition.

    The observer accessor is thread-local and active only inside the original
    BreakableCUDAGraphWrapper._capture. No replay loop or graph is replaced.
    """
    original_begin = capture_class._begin_segment
    original_end = capture_class._end_segment
    original_eager = capture_class.add_eager
    if getattr(capture_class, "_aisim_piecewise_segments", False):
        raise RuntimeError("piecewise native segment observer is already installed")
    for method in (original_begin, original_end, original_eager):
        _require_source(method)

    def registry():
        observer = current_observer()
        value = observer.registry if observer is not None else None
        return value if isinstance(value, PiecewiseCaptureRegistry) else None

    @functools.wraps(original_begin)
    def begin(capture):
        result = original_begin(capture)
        active = registry()
        if active is not None:
            active.begin_segment(capture)
        return result

    @functools.wraps(original_end)
    def end(capture):
        active = registry()
        if active is None or not capture._capturing:
            return original_end(capture)
        current = active.before_segment_end(capture)
        result = original_end(capture)
        active.after_segment_end(capture, current)
        return result

    @functools.wraps(original_eager)
    def eager(capture, fn):
        active = registry()
        if active is None:
            return original_eager(capture, fn)
        wrapped = active.wrap_eager(fn, record_function)
        result = original_eager(capture, wrapped)
        active.bind_eager(capture, wrapped)
        return result

    capture_class._begin_segment = begin
    capture_class._end_segment = end
    capture_class.add_eager = eager
    capture_class._aisim_piecewise_segments = True
    return (
        (capture_class, "_begin_segment", original_begin),
        (capture_class, "_end_segment", original_end),
        (capture_class, "add_eager", original_eager),
    )


def install_piecewise_capture(wrapper_class, capture_class, observer_for, save_capture, *, torch_module):
    """Observe the original wrapper capture with one thread-local module owner.

    ``observer_for`` selects only the already qualified original model. The
    caller persists each returned registry and resolves its source nodes through
    the actual native clone callbacks after unsubscribing. This adapter does not
    admit replay timing, replace native replay, or relax compiled-model checks.
    """
    original = wrapper_class._capture
    _require_source(original)
    if getattr(wrapper_class, "_aisim_piecewise_capture", False):
        raise RuntimeError("piecewise wrapper capture observer is already installed")
    local = threading.local()
    restorations = install_segment_observation(
        capture_class, lambda: getattr(local, "observer", None), torch_module.profiler.record_function
    )

    @functools.wraps(original)
    def capture(wrapper, entry, *args, **kwargs):
        observer = observer_for(wrapper)
        if observer is None:
            return original(wrapper, entry, *args, **kwargs)
        if (
            getattr(local, "observer", None) is not None
            or wrapper.entries.get(entry.batch_descriptor) is not entry
            or entry.capture is not None
        ):
            raise RuntimeError("piecewise capture must initialize one original native entry")
        local.observer = observer
        try:
            observer.start_piecewise(entry.batch_descriptor)
            result = original(wrapper, entry, *args, **kwargs)
            registry, receipt = observer.finish_piecewise(entry.capture)
        finally:
            local.observer = None
        captures = getattr(wrapper, "_aisim_piecewise_ownership", {})
        if entry.batch_descriptor in captures:
            raise RuntimeError("piecewise capture cannot replace a previously initialized entry")
        captures[entry.batch_descriptor] = {"entry": entry, "capture": entry.capture, "registry": registry}
        wrapper._aisim_piecewise_ownership = captures
        save_capture(wrapper, entry, registry, receipt)
        return result

    wrapper_class._capture = capture
    wrapper_class._aisim_piecewise_capture = True
    return (*restorations, (wrapper_class, "_capture", original))


def captured_piecewise_registry(wrapper, descriptor):
    """Resolve the initialized native entry, never rebuild it from a trace."""
    observation = getattr(wrapper, "_aisim_piecewise_ownership", {}).get(descriptor)
    if (
        observation is None
        or wrapper.entries.get(descriptor) is not observation["entry"]
        or observation["entry"].capture is not observation["capture"]
    ):
        raise RuntimeError("piecewise replay lacks its exact native initialized entry")
    registry = observation["registry"]
    registry.validate_replay(observation["capture"])
    return registry


def piecewise_capture_for_descriptor(manager, native_descriptor):
    """Resolve an actual V2 selection to its original native breakable entry."""
    from collector.glm53flash_vllm_graph_policy import descriptor

    actual = descriptor(native_descriptor)
    tokens = actual.get("num_tokens")
    if (
        type(tokens) is not int
        or tokens < 1
        or actual
        != {
            "cg_mode": "PIECEWISE",
            "num_tokens": tokens,
            "num_reqs": None,
            "uniform_token_count": None,
            "max_query_len": None,
            "num_active_loras": 0,
            "num_ubatches": 1,
        }
    ):
        raise RuntimeError("piecewise observation requires the exact ordinary V2 descriptor")
    wrapper = manager.breakable_cg_runner
    if manager.use_breakable_cg is not True or wrapper is None:
        raise RuntimeError("piecewise observation lacks its initialized native breakable runner")
    expected = {"num_tokens": tokens, "num_reqs": None, "uniform": False, "has_lora": False, "num_active_loras": 0}
    keys = [key for key in wrapper.entries if dataclasses.is_dataclass(key) and dataclasses.asdict(key) == expected]
    if len(keys) != 1:
        raise RuntimeError("piecewise selection lacks one original native initialized entry")
    registry = captured_piecewise_registry(wrapper, keys[0])
    if getattr(registry, "bound_capture", {}).get("native_shape_key") != expected:
        raise RuntimeError("piecewise selection differs from its source-bound captured shape")
    return registry


def bind_piecewise_instantiations(pending, callbacks, api, output, stem):
    """Retain one complete callback stream, then bind each actual executable.

    Every small segment reference identifies an original observed executable in
    the shared receipt. The source node types and complete callback stream stay
    on disk; no timing, graph structure or replay state is reconstructed here.
    """
    from collector.glm53flash_graph_callbacks import record_event_record_types, resolve_registry

    if not pending:
        return
    if callbacks.subscription_closed is not True:
        raise RuntimeError("piecewise instantiation requires completed callback unsubscription")
    output = Path(output)

    def save(path, value):
        with path.open("x") as stream:
            json.dump(value, stream, indent=2)
        return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    observed, receipts = [], {}
    for capture_index, item in enumerate(pending):
        registry = item["registry"]
        registry.validate_replay(item["entry"].capture)
        for graph in registry.graphs:
            key = capture_index, graph["position"]
            receipt = callbacks.receipt(graph["graph"])
            receipts[key] = receipt
            observed.append(
                {
                    "capture_index": key[0],
                    "position": key[1],
                    "actual_graph_exec_id": receipt["actual_graph_exec_id"],
                    "actual_graph_exec_handle": graph["graph"].raw_cuda_graph_exec(),
                }
            )
    shared = save(
        output / f"{stem}-piecewise-callbacks.json",
        {
            "schema": "glm53flash_piecewise_callbacks_v1",
            "callbacks": callbacks.rows,
            "callback_errors": callbacks.errors,
            "callback_subscription_closed": True,
            "observed_executables": observed,
            "graph_mutations": False,
        },
    )
    for capture_index, item in enumerate(pending):
        registry, source = item["registry"], item["source"]
        native_graphs = {row["position"]: row for row in registry.graphs}
        segments = []
        for segment in source["segments"]:
            if segment["kind"] == "eager":
                segments.append(segment)
                continue
            position = segment["position"]
            original = {key: value for key, value in segment.items() if key not in ("kind", "position")}
            original["native_api_libraries"] = source["native_api_libraries"]
            receipt = receipts[capture_index, position]
            graph = native_graphs[position]["graph"]
            path = output / f"{stem}-piecewise-{capture_index}-{position}-event-types.json"
            proof = record_event_record_types(
                api, graph, original, receipt, path, allow_pending_memcpy=True, allow_memset_query=True
            )
            bound = resolve_registry(original, receipt, proof, allow_pending_memcpy=True, allow_memset_query=True)
            bound["shared_callback_receipt"] = shared
            bound["observed_executable"] = next(
                row for row in observed if row["capture_index"] == capture_index and row["position"] == position
            )
            if proof is not None:
                bound["node_type_receipt"] = {
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            segments.append({"position": position, "kind": "graph", **bound})
        bound = {**source, "segments": segments, "source_receipt": item["source_receipt"]}
        artifact = save(output / f"{stem}-piecewise-{capture_index}-bound.json", bound)
        registry.bound_capture = bound
        registry.capture_artifact = artifact
