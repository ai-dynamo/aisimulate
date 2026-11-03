# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit the backend's existing forward label to NVTX in diagnostic containers."""

import hashlib
import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch
from sglang.srt.utils.nvtx_utils import profile_range

_SEEN = set()


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
def span(name, *, runner):
    model_receipt(runner)
    with profile_range(name):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
