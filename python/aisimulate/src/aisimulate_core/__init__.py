# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISimulate estimator and native performance-model API."""

from __future__ import annotations

from importlib import import_module

from aisimulate import __version__

__all__ = [
    "AicEngine",
    "RustForwardPassPerfModel",
    "__version__",
    "_build_smoke",
    "engine_spec_bincode_from_json",
    "engine_spec_schema_version",
    "gemm_quant_util_levels",
    "moe_quant_util_levels",
    "op_from_spec_json",
    "ops_json_from_ops",
    "resolve_op_sources_report_json",
    "table_view_attributes",
    "weights_ops_json",
]


def __getattr__(name: str) -> object:
    """Load native APIs on use while keeping metadata imports lightweight."""
    if name not in __all__ or name == "__version__":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("aisimulate_core._native"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
