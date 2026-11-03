# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit the backend's existing forward label to NVTX in diagnostic containers."""

import hashlib
import json
import os
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch
from sglang.srt.model_executor.step_span_utils import detailed_annotations_enabled
from sglang.srt.utils.nvtx_utils import profile_range

_SEEN = set()
_GEOMETRY_ORDINAL = 0


def bounded_values(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() > 64:
            return None
        return value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)) and len(value) <= 64 and all(type(v) is int for v in value):
        return list(value)
    raise ValueError("Unknown numerical geometry field")


def geometry(batch, stage, *, graph_entry=False, graph_runner=None):
    global _GEOMETRY_ORDINAL
    if os.environ.get("RSI_FORWARD_GEOMETRY") != "1" or not detailed_annotations_enabled():
        return
    # Deliberate synchronous metadata reads, ONLY inside diagnostic profile windows.
    # No prompt/output IDs are read. These samples cannot qualify unperturbed cadence.
    row = {"schema": "aisimulate.glm-rsi.forward-geometry.v1", "ordinal": _GEOMETRY_ORDINAL,
           "stage": stage, "graph_entry": graph_entry, "pid": os.getpid(),
           "device": torch.cuda.current_device(), "begin_ns": str(time.monotonic_ns()), "status": "recorded",
           "scope": "Synchronized numerical metadata diagnostic; not unperturbed CPU/GPU cadence"}
    try:
        captured_width = getattr(graph_runner, "captured_req_width", None)
        count = batch.input_ids.numel() if batch.input_ids is not None else None
        count_source = "input_ids_shape" if count is not None else "unavailable"
        if (count is None and graph_entry and stage == "draft"
                and type(captured_width) is int and captured_width > 0):
            # Native execute computes raw_num_token = raw_bs * captured_req_width.
            # It intentionally constructs input_ids inside the graph.
            count = batch.batch_size * captured_width
            count_source = "native_graph_captured_req_width"
        row.update(batch_size=batch.batch_size, input_token_count=count, input_token_count_source=count_source,
                   input_ids_present=batch.input_ids is not None, native_captured_req_width=captured_width,
                   seq_lens=bounded_values(batch.seq_lens),
                   seq_lens_cpu=bounded_values(getattr(batch, "seq_lens_cpu", None)),
                   seq_lens_sum=batch.seq_lens_sum, positions_bounded=bounded_values(batch.positions),
                   extend_seq_lens_cpu=bounded_values(getattr(batch, "extend_seq_lens_cpu", None)),
                   extend_prefix_lens_cpu=bounded_values(getattr(batch, "extend_prefix_lens_cpu", None)),
                   num_tokens_per_req=getattr(batch.spec_info, "num_tokens_per_req", None),
                   dsa_topk_seed_present=getattr(batch.spec_info, "dsa_topk_indices", None) is not None)
        if count is None:
            row.update(status="unavailable", reason="Native input shape is not exposed at this boundary")
        row["end_ns"] = str(time.monotonic_ns())
        serialized = json.dumps(row, allow_nan=False)
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        # A failed metadata probe must not abort the measured serving request.
        row = {key: row[key] for key in ("schema", "ordinal", "stage", "graph_entry", "pid", "device", "begin_ns")}
        row.update(status="unavailable", error_type=type(error).__name__, end_ns=str(time.monotonic_ns()),
                   reason="Numerical metadata could not be captured; request execution continues")
        serialized = json.dumps(row)
    root = Path(os.environ["RSI_FORWARD_NVTX_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"geometry-{os.getpid()}.jsonl").open("a") as stream:
        stream.write(serialized + "\n")
    _GEOMETRY_ORDINAL += 1


def graph_execute(function, batch, *args, stage):
    geometry(batch, stage, graph_entry=True, graph_runner=getattr(function, "__self__", None))
    return function(batch, *args)


def model_receipt(runner):
    if id(runner) in _SEEN:
        return
    model = runner.model
    draft = runner.is_draft_worker
    parameters = [{"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
                  for name, value in model.named_parameters()] if draft else []
    quantizers = [{"module": name, "class": type(module.quant_method).__qualname__}
                  for name, module in model.named_modules() if getattr(module, "quant_method", None) is not None]
    dtype_counts = Counter(str(value.dtype) for value in model.parameters())
    root = Path(os.environ["RSI_FORWARD_NVTX_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "aisimulate.glm-rsi.loaded-model.v1", "pid": os.getpid(),
               "cuda_device": torch.cuda.current_device(), "tp_rank": runner.ps.tp_rank,
               "tp_size": runner.ps.tp_size, "assigned_gpu_id": runner.ps.gpu_id,
               "draft": draft, "model_class": type(model).__qualname__,
               "parameter_dtype_counts": dict(dtype_counts), "draft_parameters": parameters,
               "quantizers": quantizers,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (root / f"model-{os.getpid()}-{'draft' if draft else 'target'}.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    _SEEN.add(id(runner))


@contextmanager
def span(name, *, runner, batch=None):
    model_receipt(runner)
    geometry(batch, name)
    with profile_range(name):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
