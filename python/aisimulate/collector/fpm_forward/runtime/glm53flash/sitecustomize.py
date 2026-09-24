# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazily activate the source-checked overlay only in scheduler processes."""

import os

if os.environ.get("DYN_FPM_GLM53FLASH_REAL_KV") == "1":
    import hashlib
    import importlib.abc
    import importlib.machinery
    import importlib.metadata
    import importlib.util
    import sys
    from pathlib import Path

    _TARGET = "dynamo.vllm.instrumented_scheduler"

    def _source_pins():
        from collector.glm53flash_runtime_identity import vllm_source_pins

        version = importlib.metadata.version("vllm")
        if version != __import__("vllm").__version__:
            raise RuntimeError("vLLM package metadata and imported runtime versions differ")
        return vllm_source_pins(version, Path(__file__).with_name("runtime-source-sha256.json"))

    def _verify_sources():
        expected = _source_pins()
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

    class _WorkerLoader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            expected = _source_pins()
            source = module.__name__.replace(".", "/") + ".py"
            actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            if actual != expected[source]:
                raise RuntimeError("native vLLM worker differs from the pinned Ops source")
            from collector.glm53flash_vllm_runtime import install, install_v2, install_worker_lifecycle

            installers = {
                "vllm.v1.worker.gpu_model_runner": install,
                "vllm.v1.worker.gpu.model_runner": install_v2,
                "vllm.v1.worker.gpu_worker": install_worker_lifecycle,
            }
            installers[module.__name__]()

        def __getattr__(self, name):
            return getattr(self.original, name)

    class _SchedulerFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            worker = fullname in (
                "vllm.v1.worker.gpu_model_runner",
                "vllm.v1.worker.gpu.model_runner",
                "vllm.v1.worker.gpu_worker",
            ) and os.environ.get("AISIM_GLM53_PURPOSE") in (
                "ops",
                "ops_holdout",
                "ops_graph_holdout",
            )
            if fullname != _TARGET and not worker:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _WorkerLoader(spec.loader) if worker else _SchedulerLoader(spec.loader)
            return spec

    # NVRTC/compiler helpers inherit PYTHONPATH and this activation variable.
    # They must not import vLLM/Dynamo merely by starting a Python interpreter.
    sys.meta_path.insert(0, _SchedulerFinder())
