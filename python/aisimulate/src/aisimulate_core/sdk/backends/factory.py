# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aisimulate_core.sdk import common
from aisimulate_core.sdk.backends.base_backend import BaseBackend
from aisimulate_core.sdk.backends.sglang_backend import SGLANGBackend
from aisimulate_core.sdk.backends.trtllm_backend import TRTLLMBackend
from aisimulate_core.sdk.backends.vllm_backend import VLLMBackend
from aisimulate_core.sdk.backends.vllm_backend_xpu import VLLMXPUBackend
from aisimulate_core.sdk.perf_database import is_xpu_system, is_xpu_system_spec


def get_backend(backend_name: str, system_name: str | None = None, system_spec=None) -> BaseBackend:
    """
    Get the backend class by the backend name.

    On an XPU ``system_name``, vLLM is served by the XPU-calibrated subclass;
    otherwise the default backends are returned.

    Raises:
        ValueError: If the backend name is not found.
    """
    name = common.BackendName[backend_name]
    xpu = is_xpu_system_spec(system_spec) if system_spec is not None else is_xpu_system(system_name)
    if name == common.BackendName.vllm and xpu:
        return VLLMXPUBackend()

    backend_map = {
        common.BackendName.trtllm: TRTLLMBackend,
        common.BackendName.sglang: SGLANGBackend,
        common.BackendName.vllm: VLLMBackend,
    }

    backend_class = backend_map.get(name)
    if backend_class is None:
        raise ValueError(f"Unknown backend: {backend_name}")

    return backend_class()
