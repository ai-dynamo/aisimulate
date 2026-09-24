# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang FULL graph capture ownership and real-request replay observations.

Original API wrappers for sgl-project/sglang@94602c9c2b7cbdb8efd5c52802dac6a1c180089e.
See README.glm53flash.md for exact native source and acceptance boundaries.
No CUDA graph node or native request/cache/state is changed.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from pathlib import Path

from collector.glm53flash_graph_hooks import NativeGraphOperationObserver
from collector.glm53flash_graph_nodes import (
    EXECUTION_RANGE,
    NativeGraphAPI,
    bind_execution_activity,
    bind_replay_kernels,
)
from collector.glm53flash_native_hooks import install_native_hooks

SOURCE_PINS = {
    "srt/model_executor/runner/decode_cuda_graph_runner.py": (
        "55892739b9c577ae43a60d5d31eac53f81e2b4aeca57ef5368b9c881117889d8"
    ),
    "srt/model_executor/runner_backend/full_cuda_graph_backend.py": (
        "0dc52a9a581636a20f5070cbb81d921bc56e4fb3394a9a1cf601747271c6905b"
    ),
    "srt/model_executor/runner/shape_key.py": "26e3f15209b654345a35966bd817ff8d0eb6c4c118527e78ca1e89942d5ea2c5",
}


def install(manifest, provenance, output, *, holdout=False):
    import sglang
    import torch
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend

    package = Path(sglang.__file__).resolve().parent
    for relative, wanted in SOURCE_PINS.items():
        if hashlib.sha256((package / relative).read_bytes()).hexdigest() != wanted:
            raise RuntimeError("native FULL graph source differs from the audited capture boundary")
    if getattr(FullCudaGraphBackend, "_aisim_node_observer", False):
        raise RuntimeError("native graph observer was already installed")
    if holdout == (manifest is not None):
        raise ValueError("graph calibration requires a manifest; its independent holdout forbids module hooks")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original_capture = FullCudaGraphBackend.capture_one
    original_replay = FullCudaGraphBackend.replay
    original_execute = DecodeCudaGraphRunner.execute
    states = {}

    def state_for(backend):
        state = states.get(id(backend))
        if state is not None:
            return state
        runner = backend._cuda_graph_runner
        if type(runner).__name__ != "DecodeCudaGraphRunner" or runner.enable_torch_compile is not False:
            raise RuntimeError("initial graph Ops requires actual uncompiled native FULL decode")
        if backend._reuse_output_buffer:
            raise RuntimeError("graph tail output copying needs a separately reviewed operation boundary")
        model_runner = runner.model_runner
        rank = int(model_runner.ps.tp_rank)
        state = {"rank": rank, "runner": model_runner, "captures": {}, "observer": None}
        if not holdout:
            observer = NativeGraphOperationObserver(manifest, provenance, rank, NativeGraphAPI())
            inventory = install_native_hooks(model_runner.model, observer, "sglang")
            state["observer"] = observer
            with (output / f"graph-inventory-rank-{rank}.json").open("x") as stream:
                json.dump({"inventory": inventory, "source_pins": SOURCE_PINS}, stream, indent=2)
        states[id(backend)] = state
        model_runner._aisim_glm53_graph_pending = {}
        return state

    @functools.wraps(original_capture)
    def capture(backend, shape_key, forward_fn, *args, **kwargs):
        state = state_for(backend)
        observer = state["observer"]
        captured = []

        @functools.wraps(forward_fn)
        def forward():
            if observer is None or not torch.cuda.is_current_stream_capturing():
                return forward_fn()
            observer.start(shape_key, torch_compile_enabled=False)
            result = forward_fn()
            captured.append(observer.finish())
            return result

        result = original_capture(backend, shape_key, forward, *args, **kwargs)
        graph = backend._graphs[shape_key]
        if observer is not None:
            if len(captured) != 1:
                raise RuntimeError("actual native graph did not capture exactly one complete model")
            state["captures"][shape_key] = (graph, captured[0])
            with (output / f"capture-nodes-rank-{state['rank']}.jsonl").open("a") as stream:
                stream.write(json.dumps(captured[0], sort_keys=True) + "\n")
        else:
            state["captures"][shape_key] = (graph, None)
        return result

    @functools.wraps(original_replay)
    def replay(backend, shape_key, *args, **kwargs):
        state = state_for(backend)
        record = getattr(state["runner"], "_aisim_glm53_graph_forward", None)
        if record is None or record.get("stage") != "measure":
            return original_replay(backend, shape_key, *args, **kwargs)
        active = state.get("active")
        if active is None or active["record"] is not record or active.get("replayed"):
            raise RuntimeError("native replay lacks one enclosing metadata-to-logits execution")
        graph, registry = state["captures"].get(shape_key, (None, None))
        if graph is None or graph is not backend._graphs[shape_key]:
            raise RuntimeError("actual native replay lacks its initialization capture identity")
        active.update(replayed=True, registry=registry, shape_key=shape_key)
        return original_replay(backend, shape_key, *args, **kwargs)

    @functools.wraps(original_execute)
    def execute(native_runner, forward_batch, *args, **kwargs):
        runner = native_runner.model_runner
        record = getattr(runner, "_aisim_glm53_graph_forward", None)
        if record is None or record.get("stage") != "measure":
            return original_execute(native_runner, forward_batch, *args, **kwargs)
        if not isinstance(native_runner.backend, FullCudaGraphBackend):
            raise RuntimeError("target native decode did not choose the qualified FULL backend")
        state = state_for(native_runner.backend)
        if record["phase"] != "generation" or record["query_lengths"] != [1] * record["batch_size"]:
            raise RuntimeError("graph Ops target is outside the native one-token decode scope")
        if state.get("active") is not None or record["invocation"] in runner._aisim_glm53_graph_pending:
            raise RuntimeError("target executed multiple native decode runners")
        glue = native_runner._metadata_glue
        if glue is not None and not glue.disabled:
            raise RuntimeError("native metadata glue graph requires a separate capture-node ownership registry")
        active = {"record": record, "replayed": False}
        state["active"] = active
        profiler = None
        if not holdout:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            )
            profiler.start()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream()
        try:
            # Pinned execute(): load_batch (copies + attention metadata),
            # backend.replay (the model including logits), shared-read publish.
            # Sampling and observer token/state readback happen after it returns.
            start.record(stream)
            if profiler is None:
                result = original_execute(native_runner, forward_batch, *args, **kwargs)
            else:
                with torch.profiler.record_function(EXECUTION_RANGE):
                    result = original_execute(native_runner, forward_batch, *args, **kwargs)
            end.record(stream)
            if torch.cuda.current_stream() != stream or not active["replayed"]:
                raise RuntimeError("native FULL execution changed stream or omitted its registered replay")
        finally:
            state["active"] = None
            if profiler is not None:
                profiler.stop()
        registry, shape_key = active["registry"], active["shape_key"]
        if profiler is not None:
            path = output / f"graph-profile-rank-{state['rank']}-forward-{record['invocation']}.json"
            profiler.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
            launches = [
                row
                for row in trace["traceEvents"]
                if row.get("cat") == "cuda_runtime" and row.get("name") == "cudaGraphLaunch"
            ]
            if len(launches) != 1:
                raise RuntimeError("actual FULL execution has ambiguous CUPTI graph launches")
            nodes = bind_replay_kernels(registry, trace["traceEvents"], correlation=launches[0]["args"]["correlation"])
            binding = bind_execution_activity(nodes, trace["traceEvents"])
            binding.update(trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), trace_file=path.name)
        else:
            binding = None
        runner._aisim_glm53_graph_pending[record["invocation"]] = {
            "start": start,
            "end": end,
            "native_shape_key": dataclasses.asdict(shape_key),
            "capture_registry_sha256": (
                hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest()
                if registry is not None
                else None
            ),
            "capture_registry_file": f"capture-nodes-rank-{state['rank']}.jsonl" if registry is not None else None,
            "replay_nodes": binding,
            "ops_instrumented": not holdout,
            "profiled": not holdout,
            "native_execute_source_sha256": SOURCE_PINS["srt/model_executor/runner/decode_cuda_graph_runner.py"],
        }
        return result

    FullCudaGraphBackend.capture_one = capture
    FullCudaGraphBackend.replay = replay
    DecodeCudaGraphRunner.execute = execute
    FullCudaGraphBackend._aisim_node_observer = True


def finish_native_forward(runner, record):
    """Attach graph readings only after native sampling and device completion."""
    if record.get("stage") != "measure":
        return None
    pending = getattr(runner, "_aisim_glm53_graph_pending", {}).pop(record["invocation"], None)
    if pending is None or record.get("gpu_completed") is not True or record.get("runtime_mode") != "FULL":
        raise RuntimeError("native graph target lacks its completed actual replay receipt")
    if any(type(row.get("sampled_token_id")) is not int for row in record["requests"]):
        raise RuntimeError("native graph target lacks completed same-request samples")
    elapsed = pending.pop("start").elapsed_time(pending.pop("end"))
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("native whole-graph GPU interval is not positive")
    return {
        **record,
        **pending,
        "whole_forward_gpu_ms": elapsed,
        "whole_forward_boundary": "native_full_graph_metadata_to_logits_gpu_v1",
        "measurement_method": "native_cupti_graph_nodes"
        if pending["profiled"]
        else "uninstrumented_native_full_graph_gpu_events",
        "formal_admission": False,
        "accuracy_acceptance": "NOT_EVALUATED",
    }
