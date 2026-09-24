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
import inspect
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
LOGITS_SOURCE_PIN = "6b0603d67b0c756253c2fdc882a3896d2e873a16e9aa2ef877aabca8d36bdb5f"


def install_holdout_capture(output):
    """Retain initialized native graph objects without node/profiler hooks.

    The independent whole-forward observer uses this path. It observes the
    original capture return, before native warmup/requests, and never recovers
    a missing snapshot from a later validation forward.
    """
    import vllm
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.offloader.base import NoopOffloader, get_offloader
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    from collector.glm53flash_vllm_graph_policy import SOURCE_PINS as POLICY_SOURCES

    package = Path(vllm.__file__).resolve().parent
    for relative, wanted in (POLICY_SOURCES | SOURCE_PINS).items():
        if hashlib.sha256((package / relative).read_bytes()).hexdigest() != wanted:
            raise RuntimeError("native graph holdout differs from its initialized dispatch source")
    if getattr(ModelCudaGraphManager, "_aisim_node_observer", False) or getattr(
        ModelCudaGraphManager, "_aisim_holdout_capture", False
    ):
        raise RuntimeError("native graph holdout capture cannot share calibration hooks")
    original = ModelCudaGraphManager.capture

    @functools.wraps(original)
    def capture(manager, model, *args, **kwargs):
        if hasattr(manager, "_aisim_glm53_holdout_capture"):
            raise RuntimeError("native graph holdout manager was captured more than once")
        if type(get_offloader()) is not NoopOffloader:
            raise RuntimeError("native graph holdout does not cover offloaded state or transfers")
        result = original(manager, model, *args, **kwargs)
        value = persist_snapshot(manager, model, get_tensor_model_parallel_rank(), output)
        manager._aisim_glm53_holdout_capture = (model, value, dict(manager.graphs))
        return result

    ModelCudaGraphManager.capture = capture
    ModelCudaGraphManager._aisim_holdout_capture = True


def holdout_policy(manager, model):
    """Require the exact graph objects observed when initialization completed."""
    captured = getattr(manager, "_aisim_glm53_holdout_capture", None)
    if captured is None or captured[0] is not model:
        raise RuntimeError("native graph holdout lacks its pre-request initialization snapshot")
    graphs = captured[2]
    if graphs.keys() != manager.graphs.keys() or any(manager.graphs[key] is not graph for key, graph in graphs.items()):
        raise RuntimeError("native graph holdout changed its initialized graph objects")
    return captured[1]


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


def calibration_policy(manager, model):
    captured = getattr(manager, "_aisim_glm53_ops_capture", None)
    if captured is None or captured[0] is not model:
        raise RuntimeError("native graph calibration lacks its pre-request initialization snapshot")
    graphs = captured[2]
    if graphs.keys() != manager.graphs.keys() or any(manager.graphs[key] is not graph for key, graph in graphs.items()):
        raise RuntimeError("native graph calibration changed its initialized graph objects")
    return captured[1]


class NativeVllmGraphExecution:
    """Profile one completed FULL forward, with source-bound external logits.

    Calibration only. Independent control and holdout never instantiate this
    helper. No event or profiler scope is inserted into a captured model graph.
    """

    def __init__(self, runner, output, rank):
        import torch
        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        self.torch, self.runner, self.output, self.rank = torch, runner, Path(output), rank
        self.active = None
        heads = [
            module
            for module in runner.model.modules()
            if getattr(module, "lm_head", None) is not None and getattr(module, "logits_processor", None) is not None
        ]
        if len(heads) != 1 or type(heads[0].logits_processor) is not LogitsProcessor:
            raise RuntimeError("native graph logits requires one exact source-bound LogitsProcessor")
        logits = heads[0].logits_processor
        source = Path(inspect.getfile(LogitsProcessor)).resolve()
        if hashlib.sha256(source.read_bytes()).hexdigest() != LOGITS_SOURCE_PIN:
            raise RuntimeError("native graph logits source differs from the reviewed V2 boundary")
        if (
            logits.use_all_gather is not True
            or logits.head_dtype not in (None, torch.bfloat16)
            or logits.logits_as_input
            or logits.soft_cap is not None
            or logits.scale != 1.0
            or logits.org_vocab_size != 154880
        ):
            raise RuntimeError("native graph logits precision/collective/config differs from measured geometry")
        original = logits.forward

        @functools.wraps(original)
        def forward(*args, **kwargs):
            if self.active is None:
                return original(*args, **kwargs)
            from collector.glm53flash_graph_nodes import VLLM_LOGITS_RANGE

            active = self.active
            if active["logits_calls"] or not active["range_open"]:
                raise RuntimeError("native graph logits lacks one enclosing execution range")
            if kwargs.get("skip_gather", args[3] if len(args) > 3 else False):
                raise RuntimeError("native graph logits cannot omit its declared all-gather")
            active["logits_calls"] += 1
            with torch.profiler.record_function(VLLM_LOGITS_RANGE):
                return original(*args, **kwargs)

        logits.forward = forward

    def begin(self, record, descriptor):
        if self.active is not None or record["stage"] != "measure" or record["runtime_mode"] != "FULL":
            raise RuntimeError("native graph profiling needs a fresh target FULL forward")
        registry = capture_receipt(self.runner.cudagraph_manager, descriptor)
        if registry.get("capture_scope") != "vllm_hidden_states" or registry.get("uncaptured_operations") != ["logits"]:
            raise RuntimeError("native V2 capture scope differs from hidden states plus external logits")
        artifact = self.runner.cudagraph_manager._aisim_glm53_node_captures[descriptor].get("artifact")
        if (
            artifact is None
            or hashlib.sha256((self.output / artifact["file"]).read_bytes()).hexdigest() != artifact["sha256"]
        ):
            raise RuntimeError("native V2 initialization capture file is absent or changed")
        profiler = self.torch.profiler.profile(
            activities=[self.torch.profiler.ProfilerActivity.CPU, self.torch.profiler.ProfilerActivity.CUDA]
        )
        self.active = {
            "record": record,
            "registry": registry,
            "capture_artifact": artifact,
            "profiler": profiler,
            "range_open": False,
            "logits_calls": 0,
            "finished": False,
        }
        profiler.start()

    def start_range(self):
        from collector.glm53flash_graph_nodes import VLLM_EXECUTION_RANGE

        active = self.active
        if active is None or active["range_open"]:
            raise RuntimeError("native graph execution range was not started exactly once")
        active["range"] = self.torch.profiler.record_function(VLLM_EXECUTION_RANGE)
        active["range"].__enter__()
        active["range_open"] = True

    def end_logits(self):
        active = self.active
        if active is None or not active["range_open"] or active["logits_calls"] != 1 or active["finished"]:
            raise RuntimeError("native graph target omitted or repeated its logits completion")
        active["range"].__exit__(None, None, None)
        active["range_open"] = False
        active["profiler"].stop()
        active["finished"] = True
        self._save_trace(active, failed=False)

    def _save_trace(self, active, *, failed):
        from collector.glm53flash_graph_nodes import trace_forward_identity

        record = active["record"]
        name = f"graph-profile-rank-{self.rank}-forward-{record['invocation']}.json"
        if failed:
            name = "failed-" + name
        path = self.output / name
        if path.exists():
            raise RuntimeError("native graph trace cannot overwrite a previous forward")
        active["profiler"].export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
        trace["aisim_native_forward"] = trace_forward_identity(record)
        trace["aisim_native_execution"] = {
            "backend": "vllm",
            "source_boundary": "GPUModelRunner.prepare_inputs_return_to_compute_logits",
            "logits_source_sha256": LOGITS_SOURCE_PIN,
            "failed": failed,
        }
        path.write_text(json.dumps(trace))
        active["trace"], active["path"] = trace, path

    def finish(self, record):
        from collector.glm53flash_graph_nodes import bind_replay_kernels, bind_vllm_execution_activity

        active = self.active
        if (
            active is None
            or active["record"] is not record
            or not active["finished"]
            or record.get("gpu_completed") is not True
            or record.get("native_graph_replay_completed") is not True
            or any(type(request.get("sampled_token_id")) is not int for request in record["requests"])
        ):
            raise RuntimeError("native graph profile lacks its exact completed same-request forward")
        events = active["trace"]["traceEvents"]
        launches = [row for row in events if row.get("cat") == "cuda_runtime" and row.get("name") == "cudaGraphLaunch"]
        if len(launches) != 1:
            raise RuntimeError("native V2 profile lacks one actual FULL graph launch")
        nodes = bind_replay_kernels(active["registry"], events, correlation=launches[0]["args"]["correlation"])
        binding = bind_vllm_execution_activity(nodes, events)
        binding.update(
            trace_file=active["path"].name, trace_sha256=hashlib.sha256(active["path"].read_bytes()).hexdigest()
        )
        result = {
            **record,
            "replay_nodes": binding,
            "capture_registry_file": active["capture_artifact"]["file"],
            "capture_registry_sha256": active["capture_artifact"]["sha256"],
            "logits_source_sha256": LOGITS_SOURCE_PIN,
            "profiled": True,
            "ops_instrumented": True,
            "measurement_method": "native_cupti_graph_nodes_and_external_logits",
            "formal_admission": False,
            "accuracy_acceptance": "NOT_EVALUATED",
        }
        self.active = None
        return result

    def abort(self, error):
        active = self.active
        if active is None:
            return
        if active["range_open"]:
            active["range"].__exit__(type(error), error, error.__traceback__)
            active["range_open"] = False
        if not active["finished"]:
            active["profiler"].stop()
            self._save_trace(active, failed=True)
        self.active = None


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
    if getattr(ModelCudaGraphManager, "_aisim_node_observer", False) or getattr(
        ModelCudaGraphManager, "_aisim_holdout_capture", False
    ):
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
        policy = persist_snapshot(manager, model, state["rank"], output)
        manager._aisim_glm53_ops_capture = (model, policy, dict(manager.graphs))
        pending = manager._aisim_glm53_capture_pending
        bind_captured_graphs(manager, pending)
        for descriptor in pending:
            registry = capture_receipt(manager, descriptor)
            path = output / f"vllm-capture-rank-{state['rank']}-{sequence}.json"
            with path.open("x") as stream:
                json.dump(registry, stream, indent=2, sort_keys=True)
            manager._aisim_glm53_node_captures[descriptor]["artifact"] = {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
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
