# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native GLM operation ownership during unchanged FULL graph construction.

Uses the same model/module boundaries as the eager collector. The wrappers
perform read-only native graph inspection only while actual capture is active;
native initialization warmups and graph replay execute without module tracing.
Original integration wrappers; pinned upstream APIs are documented adjacent.
"""

from __future__ import annotations

import dataclasses
import functools

from collector.glm53flash_graph_nodes import CaptureNodeRegistry
from collector.glm53flash_observer import dispatch_identity


class NativeGraphOperationObserver:
    """Duck-typed native hook target, with disjoint graph-node ownership."""

    def __init__(self, manifest, provenance, tp_rank, api, *, torch_module=None):
        if torch_module is None:
            import torch as torch_module

        self.torch = torch_module
        self.manifest = manifest
        self.provenance = dict(provenance)
        self.tp_rank = tp_rank
        self.api = api
        self.registry = None
        self.restorations = []
        self.collective_calls = 0
        self.inside_collective = False
        self.entries = {phase: {entry["name"]: entry for entry in rows} for phase, rows in manifest["phases"].items()}

    def start(self, shape_key, *, torch_compile_enabled, capture_scope="model_with_logits"):
        if torch_compile_enabled is not False:
            raise RuntimeError("Python graph ownership hooks cannot alter native compiled fusion")
        if not self.torch.cuda.is_current_stream_capturing() or self.registry is not None:
            raise RuntimeError("native graph operation ownership needs a fresh actual capture")
        if not dataclasses.is_dataclass(shape_key):
            raise ValueError("native graph key must be the original ShapeKey dataclass")
        self.shape = dataclasses.asdict(shape_key)
        if capture_scope == "model_with_logits":
            size = self.shape.get("size")
        elif capture_scope == "vllm_hidden_states":
            # Pinned V2 ModelCudaGraphManager captures model(**inputs), then
            # stores hidden states. compute_logits runs after the FULL replay.
            mode = self.shape.get("cg_mode")
            self.shape["cg_mode"] = getattr(mode, "name", mode)
            if self.shape["cg_mode"] != "FULL":
                raise ValueError("native vLLM node ownership requires an actual FULL descriptor")
            size = self.shape.get("num_tokens")
        else:
            raise ValueError("unreviewed native graph capture scope")
        if type(size) is not int or size < 1:
            raise ValueError("native graph key lacks physical padded size")
        self.padded_tokens, self.capture_scope = size, capture_scope
        self.collective_calls = 0
        self.registry = CaptureNodeRegistry(lambda: self.api.snapshot(self.torch.cuda.current_stream().cuda_stream))

    def finish(self):
        if self.registry is None or not self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("capture nodes must be retained before the native capture ends")
        result = self.registry.finish()
        expected = dict(self.entries["generation"])
        if self.capture_scope == "vllm_hidden_states":
            if "logits" not in expected:
                raise RuntimeError("complete native vLLM graph manifest lacks its separate logits operation")
            del expected["logits"]
        observed = {row["name"] for row in result["calls"]}
        if observed != expected.keys():
            raise RuntimeError(
                f"native capture has incomplete GLM operation boundaries: {sorted(expected.keys() - observed)}"
            )
        result.update(
            tp_rank=self.tp_rank,
            native_shape_key=self.shape,
            physical_padded_tokens=self.padded_tokens,
            capture_scope=self.capture_scope,
            uncaptured_operations=["logits"] if self.capture_scope == "vllm_hidden_states" else [],
            provenance=self.provenance,
            native_api_libraries=self.api.libraries,
            operations=list(expected.values()),
            formal_admission=False,
            accuracy_acceptance="NOT_EVALUATED",
        )
        self.registry = None
        return result

    def _active(self):
        return self.registry if self.torch.cuda.is_current_stream_capturing() else None

    def wrap(self, owner, method, name, *, validate_result=None, included_by_same_operation=False):
        names = (name,) if isinstance(name, str) else name
        if not names or any(item not in self.entries["generation"] for item in names):
            raise ValueError("native graph hook is absent from complete model manifest")
        original = getattr(owner, method)
        source = dispatch_identity(owner, method)
        previous, count = None, 0

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            nonlocal previous, count
            registry = self._active()
            if registry is None:
                return original(*args, **kwargs)
            if previous is not registry:
                previous, count = registry, 0
            if count >= len(names):
                raise RuntimeError("native captured module exceeded its model occurrences")
            selected = names[count]
            count += 1
            if registry.stack:
                if not included_by_same_operation or registry.stack[-1]["name"] != selected:
                    raise RuntimeError("native nested compute boundary requires a reviewed fused unit")
                result = original(*args, **kwargs)
                if validate_result is not None:
                    validate_result(result)
                return result
            token = registry.enter(selected, source)
            try:
                result = original(*args, **kwargs)
                if validate_result is not None:
                    validate_result(result)
                return result
            finally:
                registry.leave(token)

        setattr(owner, method, wrapped)
        self.restorations.append((owner, method, original))

    def wrap_collective(self, owner, method, names=()):
        original = getattr(owner, method)
        source = dispatch_identity(owner, method)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            registry = self._active()
            if registry is None or self.inside_collective:
                return original(*args, **kwargs)
            if kwargs.get("async_op", False) or self.collective_calls >= len(names):
                raise RuntimeError("native graph collective differs from declared blocking occurrences")
            name = names[self.collective_calls]
            self.collective_calls += 1
            token = registry.enter(name, source)
            self.inside_collective = True
            try:
                return original(*args, **kwargs)
            finally:
                self.inside_collective = False
                registry.leave(token)

        setattr(owner, method, wrapped)
        self.restorations.append((owner, method, original))

    def close(self):
        if self.registry is not None:
            raise RuntimeError("cannot remove native operation hooks during graph capture")
        for owner, method, original in reversed(self.restorations):
            setattr(owner, method, original)
        self.restorations.clear()
