# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Python API for the AISim estimator core.

The unified :mod:`aisim` distribution owns these implementations.
Application orchestration lives in ``aisim.sdk``.

The names in :data:`__all__` are the deliberately small, supported facade for
performance-model consumers. They are resolved lazily so importing
``aisim_core.sdk`` does not eagerly load the model registry, perf
database, or native engine. Existing explicit module imports remain supported.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "AttentionBackend",
    "EngineHandle",
    "ForwardPassPerfModelConfig",
    "ForwardPassPerfOptions",
    "MoEBackend",
    "ModelConfig",
    "RuntimeConfig",
    "RustForwardPassPerfModel",
    "compile_engine",
    "estimate_kv_cache",
    "estimate_num_gpu_blocks",
    "estimate_state_cache",
]

_PUBLIC_EXPORTS = {
    "AttentionBackend": ("aisim_core.sdk.common", "AttentionBackend"),
    "EngineHandle": ("aisim_core.sdk.engine", "EngineHandle"),
    "ForwardPassPerfModelConfig": (
        "aisim_core.sdk.rust_engine_step",
        "ForwardPassPerfModelConfig",
    ),
    "ForwardPassPerfOptions": (
        "aisim_core.sdk.rust_engine_step",
        "ForwardPassPerfOptions",
    ),
    "ModelConfig": ("aisim_core.sdk.config", "ModelConfig"),
    "MoEBackend": ("aisim_core.sdk.common", "MoEBackend"),
    "RuntimeConfig": ("aisim_core.sdk.config", "RuntimeConfig"),
    "RustForwardPassPerfModel": (
        "aisim_core.sdk.rust_engine_step",
        "RustForwardPassPerfModel",
    ),
    "compile_engine": ("aisim_core.sdk.engine", "compile_engine"),
    "estimate_kv_cache": ("aisim_core.sdk.memory", "estimate_kv_cache"),
    "estimate_state_cache": ("aisim_core.sdk.memory", "estimate_state_cache"),
    "estimate_num_gpu_blocks": (
        "aisim_core.sdk.memory",
        "estimate_num_gpu_blocks",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve one stable facade export without eagerly importing the SDK."""
    try:
        module_name, attribute = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


if TYPE_CHECKING:
    from aisim_core.sdk.common import AttentionBackend, MoEBackend
    from aisim_core.sdk.config import ModelConfig, RuntimeConfig
    from aisim_core.sdk.engine import EngineHandle, compile_engine
    from aisim_core.sdk.memory import estimate_kv_cache, estimate_num_gpu_blocks, estimate_state_cache
    from aisim_core.sdk.rust_engine_step import (
        ForwardPassPerfModelConfig,
        ForwardPassPerfOptions,
        RustForwardPassPerfModel,
    )
