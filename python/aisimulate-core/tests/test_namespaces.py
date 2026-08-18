# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.metadata

from packaging.requirements import Requirement


def test_new_and_compatibility_namespaces_export_the_same_native_types() -> None:
    aisimulate_core = importlib.import_module("aisimulate_core")
    aiconfigurator_core = importlib.import_module("aiconfigurator_core")

    assert aisimulate_core.AicEngine is aiconfigurator_core.AicEngine
    assert aisimulate_core.RustForwardPassPerfModel is aiconfigurator_core.RustForwardPassPerfModel
    for name in (
        "engine_spec_schema_version",
        "gemm_quant_util_levels",
        "moe_quant_util_levels",
        "op_from_spec_json",
        "ops_json_from_ops",
        "resolve_op_sources_report_json",
        "table_view_attributes",
        "weights_ops_json",
    ):
        assert getattr(aisimulate_core, name) is getattr(aiconfigurator_core, name)
    assert aisimulate_core.__version__ == importlib.metadata.version("aisimulate-core")


def test_public_sdk_facade_and_explicit_modules_are_available() -> None:
    sdk = importlib.import_module("aisimulate_core.sdk")
    errors = importlib.import_module("aisimulate_core.sdk.errors")
    table_view = importlib.import_module("aisimulate_core.sdk.engine_table_view")
    compatibility_errors = importlib.import_module("aiconfigurator_core.sdk.errors")
    compatibility_table_view = importlib.import_module("aiconfigurator_core.sdk.engine_table_view")

    assert "compile_engine" in sdk.__all__
    assert errors.PerfDataNotAvailableError is compatibility_errors.PerfDataNotAvailableError
    assert table_view is compatibility_table_view


def test_core_has_no_dynamo_install_dependency() -> None:
    requirements = importlib.metadata.requires("aisimulate-core") or []
    dependency_names = {Requirement(requirement).name for requirement in requirements}

    assert "dynamo" not in dependency_names
    assert "ai-dynamo" not in dependency_names
