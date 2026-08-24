# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public AISimulate core namespace.

The implementation keeps the ``aiconfigurator_core`` package importable for
the AIC 0.12.0 compatibility window. New consumers should import this module or
``aisimulate_core.sdk``.
"""

from importlib.metadata import version

from aiconfigurator_core import (
    AicEngine,
    RustForwardPassPerfModel,
    _build_smoke,
    engine_spec_bincode_from_json,
    engine_spec_schema_version,
    gemm_quant_util_levels,
    moe_quant_util_levels,
    op_from_spec_json,
    ops_json_from_ops,
    resolve_op_sources_report_json,
    table_view_attributes,
    weights_ops_json,
)

__version__ = version("aisimulate")

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
