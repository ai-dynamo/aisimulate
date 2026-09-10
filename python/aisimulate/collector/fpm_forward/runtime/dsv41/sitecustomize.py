# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazily activate the source-checked overlay only in scheduler processes."""

import os

if os.environ.get("DYN_FPM_DSV41_REAL_KV") == "1":
    import hashlib
    import importlib.abc
    import importlib.machinery
    import importlib.util
    import json
    import sys
    from pathlib import Path

    _TARGET = "dynamo.vllm.instrumented_scheduler"

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
                _verify_sources()
                self.original.exec_module(module)
                from dsv41_scheduler import DeepseekV41RealKVScheduler

                module.InstrumentedScheduler = DeepseekV41RealKVScheduler
            except BaseException as error:
                sys.stderr.write(f"V4.1 real-KV producer preflight failed: {error}\n")
                sys.stderr.flush()
                os._exit(78)

        def __getattr__(self, name):
            return getattr(self.original, name)

    class _SchedulerFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _SchedulerLoader(spec.loader)
            return spec

    # NVRTC/compiler helpers inherit PYTHONPATH and this activation variable.
    # They must not import vLLM/Dynamo merely by starting a Python interpreter.
    sys.meta_path.insert(0, _SchedulerFinder())
