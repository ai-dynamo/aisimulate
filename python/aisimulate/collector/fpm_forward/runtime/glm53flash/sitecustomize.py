# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazily activate source-checked scheduler and native-worker observation."""

import os

if os.environ.get("DYN_FPM_GLM53FLASH_REAL_KV") == "1":
    import hashlib
    import importlib.abc
    import importlib.machinery
    import importlib.util
    import json
    import sys
    from pathlib import Path

    _TARGET = "dynamo.vllm.instrumented_scheduler"
    _WORKER_TARGET = "vllm.v1.worker.gpu_worker"

    def _verify_sources():
        expected = json.loads(Path(__file__).with_name("runtime-source-sha256.json").read_text())
        for path, digest in expected.items():
            module_path = path.removesuffix("/__init__.py") if path.endswith("/__init__.py") else path[:-3]
            module = module_path.replace("/", ".")
            spec = importlib.util.find_spec(module)
            if spec is None or spec.origin is None:
                raise RuntimeError(f"required pinned source is unavailable: {path}")
            actual = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
            if actual != digest:
                raise RuntimeError(f"pinned source mismatch: {path}; expected={digest}; actual={actual}")

    class _SchedulerLoader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            try:
                if module.__name__ == _WORKER_TARGET:
                    pins = json.loads(Path(__file__).with_name("runtime-source-sha256.json").read_text())
                    actual = hashlib.sha256(Path(module.__spec__.origin).read_bytes()).hexdigest()
                    if actual != pins["vllm/v1/worker/gpu_worker.py"]:
                        raise RuntimeError("pinned native GPU worker source mismatch")
                    self.original.exec_module(module)
                    from glm53flash_worker_hardware import install

                    if os.environ.get("AISIM_GLM53_PURPOSE", "fpm") == "fpm":
                        install(module)
                    return
                _verify_sources()
                self.original.exec_module(module)
                adapter = sys.modules.get("glm53flash_scheduler")
                if adapter is not None and not hasattr(adapter, "Glm53FlashRealKVScheduler"):
                    # A spawned worker may unpickle the adapter class first.
                    # Its import needs this native base before it can finish;
                    # the adapter publishes its completed class at module end.
                    return
                from glm53flash_scheduler import Glm53FlashRealKVScheduler

                module.InstrumentedScheduler = Glm53FlashRealKVScheduler
            except Exception as error:
                # This loader runs on a later explicit scheduler import, not
                # during sitecustomize initialization. Raising fails that import
                # (including adapter-first spawn imports) and permits traceback,
                # preflight audit and process cleanup instead of skipping them.
                raise RuntimeError(f"GLM-5.3-Flash real-KV producer activation failed: {error}") from error

        def __getattr__(self, name):
            return getattr(self.original, name)

    class _SchedulerFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in {_TARGET, _WORKER_TARGET}:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _SchedulerLoader(spec.loader)
            return spec

    # NVRTC/compiler helpers inherit PYTHONPATH and this activation variable.
    # They must not import vLLM/Dynamo merely by starting a Python interpreter.
    sys.meta_path.insert(0, _SchedulerFinder())
