# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe each native worker's selected device once, outside forward timing.

API integration: vllm-project/vllm@ced6857afa0ea7b2e3f0846a62e1394e90f15607,
vllm/v1/worker/gpu_worker.py:Worker.init_device (Apache-2.0). This original
wrapper calls the native initializer unchanged, then reads device properties.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from collector.glm53flash_protocol import native_gpu_identity, validate_gb300_identity
from collector.glm53flash_runtime_identity import observe_vllm_runtime_closure


def install(module) -> None:
    worker_class = module.Worker
    if getattr(worker_class, "_aisim_glm53_hardware_installed", False):
        return
    original = worker_class.init_device

    @functools.wraps(original)
    def init_device(worker, *args, **kwargs):
        result = original(worker, *args, **kwargs)
        parallel = worker.parallel_config
        output = Path(os.environ["DYN_FPM_BENCHMARK_OUTPUT_PATH"])
        receipt = {
            "schema_version": 1,
            "backend": "vllm",
            "backend_version": importlib.metadata.version("vllm"),
            "collector_provenance_sha256": hashlib.sha256(
                output.with_name("collector-provenance.json").read_bytes()
            ).hexdigest(),
            "tp_rank": worker.rank,
            "tp_size": parallel.tensor_parallel_size,
            "hardware": native_gpu_identity(module.torch),
            "worker_source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        }
        try:
            closure = observe_vllm_runtime_closure(
                receipt["backend_version"], Path(__file__).with_name("runtime-source-sha256.json")
            )
            if closure is not None:
                receipt["runtime_closure"] = closure
            if (
                type(worker.rank) is not int
                or not 0 <= worker.rank < parallel.tensor_parallel_size
                or parallel.tensor_parallel_size not in (2, 4)
                or parallel.pipeline_parallel_size != 1
                or parallel.data_parallel_size != 1
            ):
                raise ValueError("GLM hardware evidence requires native pure TP2/TP4")
            validate_gb300_identity(receipt["hardware"])
            if str(worker.device) != f"cuda:{receipt['hardware']['cuda_device_index']}":
                raise ValueError("vLLM worker device differs from its selected CUDA device")
            receipt["status"] = "passed"
        except Exception as error:
            receipt.update(status="failed", error=str(error))
            raise
        finally:
            path = output.with_name(f"native-device-rank-{worker.rank}.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            # A reused result directory must not erase a previous attempt.
            with path.open("x") as stream:
                stream.write(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        return result

    worker_class.init_device = init_device
    worker_class._aisim_glm53_hardware_installed = True
