# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic whole-model CUDA events around an actual native FULL replay.

This does not measure constituent Ops or admit graph prediction data. The
existing eager consumer stays unchanged. Native vLLM API/source attribution:
README.glm53flash.md and the canonical THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import dataclasses
import math

BOUNDARY = "native_full_graph_dispatch_to_logits_gpu_v1"


def descriptor_identity(descriptor) -> dict:
    """Serialize actual native declarations, without inventing a graph key."""
    if not dataclasses.is_dataclass(descriptor):
        raise ValueError("native graph descriptor must be the actual dataclass")
    result = dataclasses.asdict(descriptor)
    mode = result.get("cg_mode")
    result["cg_mode"] = getattr(mode, "name", mode)
    if result["cg_mode"] != "FULL":
        raise ValueError("initial whole-graph diagnostic requires actual FULL dispatch")
    for name in ("num_tokens", "num_reqs"):
        if type(result.get(name)) is not int or result[name] <= 0:
            raise ValueError("native graph descriptor lacks positive physical dimensions")
    return result


class NativeFullGraphWindow:
    """Bind one completed native graph object to a logits-ended GPU interval.

    The serving adapter supplies its actual prepared InputBatch and completion
    receipt. It must call this after native preparation and before dispatch,
    then finish only after the original native sampling/completion path. No
    Python module-call count is used as evidence of replayed operations.
    """

    def __init__(self, torch_module):
        self.torch = torch_module
        self.pending = None

    def arm(self, native_record: dict, descriptor, graph) -> None:
        if self.pending is not None:
            raise RuntimeError("previous native graph invocation remains unfinished")
        key = descriptor_identity(descriptor)
        requests = native_record.get("requests", [])
        ids = [item["request_id"] for item in requests]
        if (
            not ids
            or len(ids) != len(set(ids))
            or any(item.get("is_prefilling") is not False or item.get("query") != 1 for item in requests)
            or key["num_tokens"] != native_record.get("actual_padded_tokens")
            or key["num_tokens"] < len(ids)
            or key["num_reqs"] < len(ids)
            or native_record.get("native_mode") != "FULL"
            or not callable(getattr(graph, "replay", None))
        ):
            raise ValueError("native graph key/padding/request geometry is inconsistent")
        self.pending = {
            "record": native_record,
            "descriptor": descriptor,
            "graph": graph,
            "key": key,
            "dispatched": False,
            "returned": False,
            "ended": False,
        }

    def dispatch(self, descriptor, graph, original, *args, **kwargs):
        value = self.pending
        if value is None:
            return original(*args, **kwargs)
        if (
            value["dispatched"]
            or descriptor != value["descriptor"]
            or graph is not value["graph"]
            or self.torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError("native replay changed graph identity or repeated/captured unexpectedly")
        stream = self.torch.cuda.current_stream()
        value["stream"] = stream
        value["start"] = self.torch.cuda.Event(enable_timing=True)
        value["end"] = self.torch.cuda.Event(enable_timing=True)
        value["dispatched"] = True
        value["start"].record(stream)
        result = original(*args, **kwargs)
        if self.torch.cuda.current_stream() != stream:
            raise RuntimeError("native graph dispatch changed its current stream")
        value["returned"] = True
        return result

    def logits(self, original, *args, **kwargs):
        value = self.pending
        result = original(*args, **kwargs)
        if value is not None:
            if not value["returned"] or value["ended"] or self.torch.cuda.current_stream() != value["stream"]:
                raise RuntimeError("native logits lacks a unique completed graph dispatch")
            value["end"].record(value["stream"])
            value["ended"] = True
        return result

    def finish(self) -> dict:
        value = self.pending
        if value is None or not value["ended"]:
            raise RuntimeError("native whole-graph interval lacks its logits endpoint")
        record = value["record"]
        if record.get("model_execute_returned") is not True or record.get("gpu_completed") is not True:
            raise RuntimeError("native request has no completed original model/sample receipt")
        if any(type(item.get("sampled_token_id")) is not int for item in record["requests"]):
            raise RuntimeError("native request lacks its exact sampled token receipt")
        self.torch.cuda.synchronize()
        elapsed = value["start"].elapsed_time(value["end"])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("native graph GPU interval must be finite and positive")
        result = {
            **record,
            "whole_forward_gpu_ms": elapsed,
            "whole_forward_boundary": BOUNDARY,
            "native_graph_descriptor": value["key"],
            "native_graph_process_identity": id(value["graph"]),
            "used_cuda_graph": True,
            "ops_instrumented": False,
            "constituent_operation_coverage": "NOT_EVALUATED",
            "prediction_accuracy_acceptance": "NOT_EVALUATED",
            "scope": "diagnostic_native_full_graph_whole_forward",
        }
        self.pending = None
        return result
