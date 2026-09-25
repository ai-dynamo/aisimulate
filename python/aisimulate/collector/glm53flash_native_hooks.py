# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Version-pinned observation bindings for loaded GLM text decoder objects.

This adapter imports upstream implementations; it does not copy or replace
their compute. Integration paths and immutable Apache-2.0 sources are listed
in README.glm53flash.md and the canonical third-party notices.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from importlib.metadata import version
from pathlib import Path

from collector.glm53flash_contract import BACKENDS, validate_runtime_row
from collector.glm53flash_observer import NativeOperationObserver, require_fused_sglang_norm

SGLANG_PROJECTION_SOURCE_PINS = {
    "srt/layers/communicator.py": "385c6ab93420618dc222be44d65d129a20162cb8ee8b05d86c0e7d4e5a57eaec",
    "srt/layers/communicator_mhc.py": "d1b8b711df8e486fa9fd6a0028ddcb0d2a863bb23300337b5d3cc2407d168fcf",
    "srt/models/deepseek_common/attention_forward_methods/forward_mha.py": (
        "004f83d9e1b70ec3f4cfee4ec1c3474d5ea212c2092de14c9fac72542e7d72bb"
    ),
}


def _check_geometry(layer, index: int, observer: NativeOperationObserver, backend: str) -> None:
    """Bind model graph identity to the module loaded by the native framework."""
    attn = layer.self_attn
    for phase in ("context", "generation"):
        entry = next(row for row in observer.manifest["phases"][phase] if row["name"] == f"attention_{index}")
        geometry = json.loads(entry["geometry"])
        is_kda = type(attn).__name__ == "Glm5NextLinearAttention"
        expected_kind = "kda" if is_kda else "sparse_mla"
        if geometry["layer_kind"] != expected_kind or geometry["backend"] != backend:
            raise RuntimeError(f"loaded native attention kind/backend differs at layer {index}")
        if (
            int(attn.hidden_size) != geometry["hidden_size"]
            or int(attn.num_heads) != geometry["num_heads"] * geometry["tp_size"]
        ):
            raise RuntimeError(f"loaded native attention dimensions differ at layer {index}")
        local_heads = attn.local_num_heads if is_kda else attn.num_local_heads
        if int(local_heads) != geometry["num_heads"]:
            raise RuntimeError(f"loaded native attention TP shape differs at layer {index}")
        if is_kda:
            if (int(attn.head_dim), int(attn.conv_size)) != (geometry["head_dim"], geometry["conv_kernel"]):
                raise RuntimeError(f"loaded native KDA head/conv shape differs at layer {index}")
        elif (int(attn.q_lora_rank), int(attn.kv_lora_rank), int(attn.qk_rope_head_dim)) != (
            geometry["q_lora_rank"],
            geometry["kv_lora_rank"],
            0,
        ):
            raise RuntimeError(f"loaded native NoPE MLA ranks differ at layer {index}")


def install_native_hooks(model, observer: NativeOperationObserver, backend: str) -> list[dict]:
    """Install observation on all 45 actual text layers, never sampled replicas.

    Call after native model loading and before an explicitly eager measured
    request. The serving adapter must uninstall through ``observer.close()``
    before its independent uninstrumented forward/control measurements.
    """
    if backend not in BACKENDS:
        raise RuntimeError("native GLM hooks require the exact qualified framework release")
    actual_version = version(backend)
    for key, value in (
        ("backend", backend),
        ("backend_version", actual_version),
        ("backend_revision", BACKENDS[backend][1]),
    ):
        if observer.provenance.get(key) != value or observer.manifest.get(key) != value:
            raise RuntimeError("native GLM hook package differs from its frozen manifest/provenance")
    # Reuse the exact admitted source contract, including the qualified repair.
    # The serving adapter independently binds the actual worker files/binaries;
    # no version prefix or ambient patched module can admit a different runtime.
    validate_runtime_row(observer.provenance)
    if backend == "sglang":
        package = Path(importlib.import_module("sglang").__file__).resolve().parent
        for relative, wanted in SGLANG_PROJECTION_SOURCE_PINS.items():
            if hashlib.sha256((package / relative).read_bytes()).hexdigest() != wanted:
                raise RuntimeError(f"native SGLang lazy projection source differs: {relative}")
        observer.provenance["native_projection_source_sha256"] = dict(SGLANG_PROJECTION_SOURCE_PINS)
    layers = [module for module in model.modules() if type(module).__name__ == "Glm5NextDecoderLayer"]
    if len(layers) != 45:
        raise RuntimeError(f"expected exactly 45 loaded text decoder layers with MTP off, found {len(layers)}")
    layer_id_attribute = "layer_idx" if backend == "vllm" else "layer_id"
    layers.sort(key=lambda layer: int(getattr(layer, layer_id_attribute)))
    if [int(getattr(layer, layer_id_attribute)) for layer in layers] != list(range(45)):
        raise RuntimeError("native decoder layer identities do not cover 0 through 44")
    # Validate before installing any wrapper, so a failed geometry check cannot
    # leave a partially instrumented model behind.
    for index, layer in enumerate(layers):
        _check_geometry(layer, index, observer, backend)
    inventory = []
    try:
        for index, layer in enumerate(layers):
            observer.wrap(layer.self_attn, "forward", f"attention_{index}")
            if backend == "vllm":
                if getattr(layer, "is_sequence_parallel", False):
                    raise RuntimeError("sequence-parallel mHC needs its own native timing geometry")
                if index == 0:
                    observer.wrap(layer, "hc_pre", "mhc_pre_attn_0")
                fused = (
                    (f"mhc_fused_ffn_{index}",)
                    if index == 0
                    else (
                        f"mhc_fused_attn_{index}",
                        f"mhc_fused_ffn_{index}",
                    )
                )
                observer.wrap(layer, "hc_fused_post_pre", fused)
                if index == 44:
                    observer.wrap(layer, "hc_post", "mhc_post_ffn_44")
            else:
                communicator = layer.layer_communicator
                # These are bound callbacks saved during native construction;
                # replacing layer methods alone would not observe execution.
                observer.wrap(
                    communicator.mhc,
                    "hc_attn_pre",
                    f"mhc_pre_attn_{index}",
                    validate_result=require_fused_sglang_norm,
                )
                observer.wrap(
                    communicator.mhc,
                    "hc_ffn_pre",
                    f"mhc_pre_ffn_{index}",
                    validate_result=require_fused_sglang_norm,
                )
                observer.wrap(communicator.mhc, "hc_post", (f"mhc_post_attn_{index}", f"mhc_post_ffn_{index}"))
                if communicator.qkv_latent_func is not None:
                    # Pinned communicator.py:275 evaluates this saved callback
                    # lazily from attention's fetch_qkv_latent. If a native path
                    # hoists it, it is a disjoint part; when invoked inside the
                    # same attention, the outer interval already includes it.
                    observer.wrap(
                        communicator,
                        "qkv_latent_func",
                        f"attention_{index}",
                        included_by_same_operation=True,
                    )
                if bool(getattr(layer.self_attn.o_proj, "reduce_results", False)):
                    raise RuntimeError("SGLang native attention collective ownership changed")
            # One native FFN includes routing, clamp10, shared and routed experts.
            # Nested generic GEMM/MoE identities cannot certify these semantics.
            observer.wrap(layer.mlp, "forward", f"ffn_{index}")
            inventory.append(
                {
                    "layer": index,
                    "attention_class": f"{type(layer.self_attn).__module__}.{type(layer.self_attn).__name__}",
                }
            )
        text_models = [module for module in model.modules() if type(module).__name__ == "Glm5NextModel"]
        heads = [
            module
            for module in model.modules()
            if getattr(module, "lm_head", None) is not None and getattr(module, "logits_processor", None) is not None
        ]
        if len(text_models) != 1 or len(heads) != 1:
            raise RuntimeError("native text embedding/final norm/logits boundaries are ambiguous")
        text_model, head = text_models[0], heads[0]
        logits = head.logits_processor
        if backend == "vllm":
            if not logits.use_all_gather or logits.head_dtype not in (None, observer.torch.bfloat16):
                raise RuntimeError("vLLM native logits precision/collective differs from admitted geometry")
        elif logits.use_fp32_lm_head or logits.use_attn_tp_group or logits.use_tp_lm_head_all_to_all:
            raise RuntimeError("SGLang native logits precision/collective differs from admitted geometry")
        observer.wrap(text_model.embed_tokens, "forward", "embedding")
        observer.wrap(text_model.norm, "forward", "final_norm")
        observer.wrap(logits, "forward", "logits")
        collective_names = (
            "embedding_allreduce",
            *(name for index in range(45) for name in (f"attention_allreduce_{index}", f"ffn_allreduce_{index}")),
        )
        if backend == "vllm":
            native_model = importlib.import_module("vllm.models.glm5next.nvidia.model")
            parallel = importlib.import_module("vllm.distributed.parallel_state")
            observer.wrap_collective(parallel.get_tp_group(), "all_reduce", collective_names)
        else:
            native_model = importlib.import_module("sglang.srt.layers.communicator_mhc")
            parallel = importlib.import_module("sglang.srt.distributed.parallel_state")
            groups = (parallel.get_tp_group(), parallel.get_attn_tp_group(), parallel.get_moe_tp_group())
            seen = set()
            for group in groups:
                if id(group) not in seen:
                    observer.wrap_collective(group, "all_reduce", collective_names)
                    seen.add(id(group))
        observer.wrap(native_model, "hc_expand", "mhc_expand")
        observer.wrap(native_model, "hc_contract", "mhc_contract")
    except BaseException:
        observer.close()
        raise
    return inventory
