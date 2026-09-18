# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISimulate estimator and native performance-model API."""

from aisimulate import __version__

from ._native import (
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
