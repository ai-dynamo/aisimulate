# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicitly activated, fail-closed source-checked native scheduler overlay."""

import os

if os.environ.get("DYN_FPM_DSV41_REAL_KV") == "1":
    try:
        import hashlib
        import importlib.util
        import json
        import sys
        from pathlib import Path

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
        import dynamo.vllm.instrumented_scheduler as native
        from dsv41_scheduler import DeepseekV41RealKVScheduler

        native.InstrumentedScheduler = DeepseekV41RealKVScheduler
    except BaseException as error:
        # Python otherwise logs sitecustomize exceptions and starts unpatched.
        import sys

        sys.stderr.write(f"V4.1 real-KV producer preflight failed: {error}\n")
        sys.stderr.flush()
        os._exit(78)
