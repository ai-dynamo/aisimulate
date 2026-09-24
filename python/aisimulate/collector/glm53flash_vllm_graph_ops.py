# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only operation ownership inside native vLLM V2 FULL captures.

Original wrappers for vllm-project/vllm@ced6857afa0ea7b2e3f0846a62e1394e90f15607.
No framework model, input, graph or cache implementation is copied or replaced.
The pinned ModelCudaGraphManager captures hidden states, with native compiled
leaves intact; logits execute outside that graph and need separate evidence.
See README.glm53flash.md and THIRD_PARTY_NOTICES.md for source and scope.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

from collector.glm53flash_graph_callbacks import CloneCallbacks, resolve_registry
from collector.glm53flash_graph_hooks import NativeGraphOperationObserver
from collector.glm53flash_graph_nodes import NativeGraphAPI
from collector.glm53flash_native_hooks import install_native_hooks
from collector.glm53flash_vllm_graph_policy import persist_snapshot

SOURCE_PINS = {
    "v1/worker/gpu/cudagraph_utils.py": "6e9c042890603535e300a40df8ee159dbed1058a64a83ae50ff0329e332e05ff",
    "model_executor/offloader/base.py": "5157a59232715e7247761588efb88fc44e14970b580722a1f0b2e8b3a23a10fe",
}


def bind_captured_graphs(manager, pending):
    """Bind only complete native captures to their actual graph objects."""
    existing = getattr(manager, "_aisim_glm53_node_captures", {})
    for descriptor, observations in pending.items():
        graph = manager.graphs.get(descriptor)
        if graph is None or len(observations) != 1:
            raise RuntimeError("native vLLM FULL capture lacks one complete observed graph")
        previous = existing.get(descriptor)
        if previous is not None and previous["graph"] is not graph:
            raise RuntimeError("native vLLM replaced an observed graph without a new manager identity")
        existing[descriptor] = {"graph": graph, "registry": observations[0]}
    manager._aisim_glm53_node_captures = existing


def capture_receipt(manager, descriptor):
    """Retrieve initialization evidence only for the exact selected graph."""
    observation = getattr(manager, "_aisim_glm53_node_captures", {}).get(descriptor)
    if observation is None or observation["graph"] is not manager.graphs.get(descriptor):
        raise RuntimeError("native vLLM replay lacks its exact initialization capture")
    return observation["registry"]


def install(manifest, provenance, output):
    """Install before native ModelCudaGraphManager.capture is first invoked.

    This capture-only adapter never claims logits, replay timings, real request
    completion, or accepted performance rows. The serving observer must join its
    returned registry to actual CUPTI replay IDs and independently retain those
    other boundaries. Native PIECEWISE capture remains unchanged and unobserved.
    """
    import torch
    import vllm
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.offloader.base import NoopOffloader, get_offloader
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager, ModelCudaGraphManager, has_compiled_submodule

    package = Path(vllm.__file__).resolve().parent
    for relative, wanted in SOURCE_PINS.items():
        if hashlib.sha256((package / relative).read_bytes()).hexdigest() != wanted:
            raise RuntimeError("native vLLM graph manager differs from the reviewed capture source")
    if getattr(ModelCudaGraphManager, "_aisim_node_observer", False):
        raise RuntimeError("native vLLM graph observer was already installed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original_model_capture, original_capture = ModelCudaGraphManager.capture, CudaGraphManager.capture
    models = {}
    sequence = 0

    @functools.wraps(original_model_capture)
    def model_capture(manager, model, *args, **kwargs):
        nonlocal sequence
        if has_compiled_submodule(model):
            raise RuntimeError("read-only module capture cannot alter a compiled native model boundary")
        if manager.ubatch_runner is not None:
            raise RuntimeError("native microbatch capture needs a separately reviewed cross-thread node owner")
        if type(get_offloader()) is not NoopOffloader:
            # Native capture joins an offloader stream after forward_fn returns.
            # The reviewed noop implementation inserts no tail work there.
            raise RuntimeError("native offloader tail lies outside the reviewed model capture boundary")
        state = models.get(id(model))
        if state is None:
            rank = get_tensor_model_parallel_rank()
            observer = NativeGraphOperationObserver(manifest, provenance, rank, NativeGraphAPI())
            inventory = install_native_hooks(model, observer, "vllm")
            state = {"observer": observer, "rank": rank, "model": model, "inventory": inventory}
            models[id(model)] = state
            with (output / f"vllm-graph-inventory-rank-{rank}.json").open("x") as stream:
                json.dump({"inventory": inventory, "source_pins": SOURCE_PINS}, stream, indent=2)
        if state["model"] is not model or state["observer"].registry is not None:
            raise RuntimeError("native vLLM capture has a conflicting live model observer")
        manager._aisim_glm53_capture_state = state
        manager._aisim_glm53_capture_pending = {}
        result = original_model_capture(manager, model, *args, **kwargs)
        # This is the initialized native descriptor/candidate inventory, before
        # any real request or holdout is observed. Recording a PIECEWISE entry
        # does not supply its still-missing measured operation ownership.
        persist_snapshot(manager, model, state["rank"], output)
        pending = manager._aisim_glm53_capture_pending
        bind_captured_graphs(manager, pending)
        for descriptor in pending:
            registry = capture_receipt(manager, descriptor)
            with (output / f"vllm-capture-rank-{state['rank']}-{sequence}.json").open("x") as stream:
                json.dump(registry, stream, indent=2, sort_keys=True)
            sequence += 1
        del manager._aisim_glm53_capture_pending
        return result

    @functools.wraps(original_capture)
    def capture(manager, create_forward_fn, *args, **kwargs):
        state = getattr(manager, "_aisim_glm53_capture_state", None)
        if state is None:
            return original_capture(manager, create_forward_fn, *args, **kwargs)

        @functools.wraps(create_forward_fn)
        def factory(descriptor, warmup):
            native_forward = create_forward_fn(descriptor, warmup)
            if warmup or descriptor.cg_mode.name != "FULL":
                return native_forward

            @functools.wraps(native_forward)
            def forward(*forward_args, **forward_kwargs):
                if not torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("native FULL capture factory executed outside its CUDA capture")
                observer = state["observer"]
                observer.start(descriptor, torch_compile_enabled=False, capture_scope="vllm_hidden_states")
                result = native_forward(*forward_args, **forward_kwargs)
                registry = observer.finish()
                manager._aisim_glm53_capture_pending.setdefault(descriptor, []).append(registry)
                return result

            return forward

        serial = state.get("callback_serial", 0)
        state["callback_serial"] = serial + 1
        stem = f"vllm-graph-clones-rank-{state['rank']}-capture-{serial}"
        receipts = {}
        with CloneCallbacks(state["observer"].api, output / f"{stem}.progress.jsonl") as callbacks:
            result = original_capture(manager, factory, *args, **kwargs)
            for descriptor in manager._aisim_glm53_capture_pending:
                graph = manager.graphs.get(descriptor)
                if graph is None:
                    raise RuntimeError("native FULL descriptor has no instantiated graph")
                receipts[descriptor] = callbacks.receipt(graph)
        for index, (descriptor, receipt) in enumerate(receipts.items()):
            path = output / f"{stem}-{index}.json"
            with path.open("x") as stream:
                json.dump(receipt, stream, indent=2)
            observations = manager._aisim_glm53_capture_pending[descriptor]
            if len(observations) != 1:
                raise RuntimeError("native FULL descriptor lacks one complete source capture")
            with (output / f"{stem}-{index}-source.json").open("x") as stream:
                json.dump(observations[0], stream, indent=2)
            registry = resolve_registry(observations[0], receipt)
            registry["instantiation_receipt"] = {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            manager._aisim_glm53_capture_pending[descriptor] = [registry]
        return result

    ModelCudaGraphManager.capture = model_capture
    CudaGraphManager.capture = capture
    ModelCudaGraphManager._aisim_node_observer = True
