# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense low-bit dummy weights for the vLLM and SGLang serving probes.

The probes load a checkpoint-free dummy model (load_format=dummy). The two frameworks' dummy
initializers draw floats from U(-1e-3, 1e-3), which rounds FP8 storage to zero or the smallest
subnormal, and leave integer storage (packed FP4, MXFP4, INT4/INT8) untouched (vllm
model_executor/model_loader/weight_utils.py initialize_single_dummy_weight, sglang
srt/model_loader/weight_utils.py initialize_dummy_weights). Real checkpoints are dense, and the
grouped GEMMs time 8-23% faster on zero-like weights from 2048 tokens (B200, DeepSeek-V3 MoE
shapes) and pick other tactics; the collector times dense weights (collector/helper.py
fill_dense_weight_). A probe whose kernel times serve as the timing reference therefore runs
dense low-bit weights too.

The float weights matter too, through the routing. With every float drawn from U(-1e-3, 1e-3),
a normalized hidden state is scaled by norm weights near 1e-3 and a router's logits come out
near 1e-4, below the U(-1e-3, 1e-3) expert-correction bias of DeepSeek-style routers and the
router bias of GPT-OSS: every prefill token then picks the same experts, and the grouped GEMMs
of the reference ran 10-37% faster than the collector's balanced case (moe timing round,
2026-11-09). Bias-free routers (Gemma 4) route on noise instead.

``densify(named_tensors)`` refills, after the framework's dummy initializer and before its
post-load processing (repacking, scale conversion), every weight matrix: FP8 storage from
N(0, 64) clamped to +-448, 8-bit integers uniform over all bytes, 32-bit integers uniform over
all bits (every bit pattern of these storages is a finite value), 16/32-bit float matrices from
N(0, 0.02) and normalization weights to 1, so hidden states keep unit scale and routers route on
the tokens (logits around 1.7 for a 7168-wide hidden state). Only tensors named as weights are
touched: scales, zero points, biases and the integer index tensors (g_idx, permutations, expert
maps) keep the framework's values.
"""

from __future__ import annotations

_CHUNK = 1 << 27
_MATRIX_SUFFIXES = ("weight", "weight_packed", "qweight")
_NOT_MATRIX = ("scale", "zero", "g_idx", "perm", "bias", "alpha", "map")


def is_weight_matrix(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return leaf.endswith(_MATRIX_SUFFIXES) and not any(s in leaf for s in _NOT_MATRIX)


def is_norm_weight(name: str) -> bool:
    parts = name.rsplit(".", 2)
    return parts[-1] == "weight" and len(parts) > 1 and "norm" in parts[-2]


def _fill(tensor, norm: bool = False) -> bool:
    import torch

    if tensor.is_floating_point():
        if tensor.element_size() >= 2:
            if norm:
                tensor.fill_(1.0)
            else:
                tensor.normal_(0.0, 0.02)
            return True
        flat = tensor.view(-1)
        for start in range(0, flat.numel(), _CHUNK):
            part = flat[start : start + _CHUNK]
            values = torch.randn(part.shape, dtype=torch.float32, device=part.device).mul_(64.0).clamp_(-448.0, 448.0)
            part.copy_(values.to(part.dtype))
            del values
    elif tensor.dtype == torch.uint8:
        tensor.random_(0, 256)
    elif tensor.dtype == torch.int8:
        tensor.random_(-128, 128)
    elif tensor.dtype == torch.int32:
        tensor.random_(-(2**31), 2**31 - 1)
    else:
        return False
    return True


def densify(named_tensors, low_bit: bool = True) -> dict:
    """Refill the weights of ``named_tensors`` ((name, tensor) pairs) in place; ``low_bit=False``
    leaves FP8 and integer storage alone (TRT-LLM's own dummy draw already fills them densely).
    Returns {"tensors": n, "bytes": total} for the probe record."""
    import torch

    n = total = 0
    with torch.no_grad():
        for name, tensor in named_tensors:
            if tensor is None or tensor.device.type == "meta" or not is_weight_matrix(name):
                continue
            data = tensor.data if hasattr(tensor, "data") else tensor
            if not low_bit and not (data.is_floating_point() and data.element_size() >= 2):
                continue
            if _fill(data, norm=is_norm_weight(name)):
                n += 1
                total += data.numel() * data.element_size()
    return {"tensors": n, "bytes": total}


def install_vllm(rec: dict) -> None:
    """Wrap vLLM's DummyModelLoader.load_weights: the framework's dummy draw, then densify.
    BaseModelLoader.load_model runs process_weights_after_loading after load_weights
    (vllm model_executor/model_loader/base_loader.py), so the post-load repacking sees the
    dense values. In-process engine only (the probe keeps EngineCore in-process)."""
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

    original = DummyModelLoader.load_weights

    def load_weights(self, model, model_config):
        original(self, model, model_config)
        named = [*model.named_parameters(), *model.named_buffers()]
        stats = densify(named)
        acc = rec.setdefault("probe_dense_weights", {"tensors": 0, "bytes": 0})
        acc["tensors"] += stats["tensors"]
        acc["bytes"] += stats["bytes"]

    DummyModelLoader.load_weights = load_weights


def install_sglang(rec: dict) -> None:
    """Wrap the initialize_dummy_weights SGLang's DummyModelLoader calls (bound in
    srt/model_loader/loader.py), so post_load_weights and the quant methods'
    process_weights_after_loading run on the dense values."""
    import sglang.srt.model_loader.loader as loader

    original = loader.initialize_dummy_weights

    def initialize_dummy_weights(model, *args, **kwargs):
        original(model, *args, **kwargs)
        stats = densify(model.state_dict().items())
        acc = rec.setdefault("probe_dense_weights", {"tensors": 0, "bytes": 0})
        acc["tensors"] += stats["tensors"]
        acc["bytes"] += stats["bytes"]

    loader.initialize_dummy_weights = initialize_dummy_weights


def install_trtllm(rec: dict) -> None:
    """Wrap the initialize_dummy_weights TRT-LLM's model loader calls for load_format=dummy
    (_torch/pyexecutor/model_loader.py ModelLoader.load): its FP8 / FP4 / INT8 draw is dense
    already, its 16/32-bit floats are U(-1e-3, 1e-3), so only the floats are refilled. Takes
    effect only when the executor builds the model in this process (rec in_process_model)."""
    import tensorrt_llm._torch.pyexecutor.model_loader as model_loader

    original = model_loader.initialize_dummy_weights

    def initialize_dummy_weights(model, *args, **kwargs):
        original(model, *args, **kwargs)
        stats = densify([*model.named_parameters(), *model.named_buffers()], low_bit=False)
        acc = rec.setdefault("probe_dense_weights", {"tensors": 0, "bytes": 0})
        acc["tensors"] += stats["tensors"]
        acc["bytes"] += stats["bytes"]

    model_loader.initialize_dummy_weights = initialize_dummy_weights
