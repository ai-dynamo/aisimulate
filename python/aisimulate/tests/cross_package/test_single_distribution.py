# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path

from packaging.requirements import Requirement

APPLICATION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


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
    assert aisimulate_core.__version__ == importlib.metadata.version("aisimulate")


def test_public_sdk_facade_and_explicit_modules_are_available() -> None:
    sdk = importlib.import_module("aisimulate_core.sdk")
    errors = importlib.import_module("aisimulate_core.sdk.errors")
    table_view = importlib.import_module("aisimulate_core.sdk.engine_table_view")
    compatibility_errors = importlib.import_module("aiconfigurator_core.sdk.errors")
    compatibility_table_view = importlib.import_module("aiconfigurator_core.sdk.engine_table_view")

    assert "compile_engine" in sdk.__all__
    assert errors.PerfDataNotAvailableError is compatibility_errors.PerfDataNotAvailableError
    assert table_view is compatibility_table_view


def test_aisimulate_owns_compatibility_namespaces_without_split_dependencies() -> None:
    aisimulate_core = importlib.import_module("aisimulate_core")
    requirements = importlib.metadata.requires("aisimulate") or []
    dependency_names = {Requirement(requirement).name for requirement in requirements}

    assert importlib.metadata.version("aisimulate") == aisimulate_core.__version__
    assert "aisimulate-core" not in dependency_names
    assert "aiconfigurator-core" not in dependency_names
    assert "dynamo" not in dependency_names
    assert "ai-dynamo" not in dependency_names


def test_native_compatibility_module_forwards_to_the_unified_runtime() -> None:
    runtime = importlib.import_module("aisimulate._runtime")
    compatibility_runtime = importlib.import_module("aiconfigurator_core._aiconfigurator_core")

    assert compatibility_runtime.AicEngine is runtime.AicEngine
    assert compatibility_runtime.RustForwardPassPerfModel is runtime.RustForwardPassPerfModel


def test_native_compatibility_wildcard_import_preserves_runtime_identity() -> None:
    runtime = importlib.import_module("aisimulate._runtime")
    compatibility_runtime = importlib.import_module("aiconfigurator_core._aiconfigurator_core")
    namespace: dict[str, object] = {}

    exec("from aiconfigurator_core._aiconfigurator_core import *", {}, namespace)

    assert set(namespace) == set(compatibility_runtime.__all__)
    for name in compatibility_runtime.__all__:
        assert namespace[name] is getattr(runtime, name)


def test_legacy_source_paths_are_non_recursive_views_of_canonical_sources() -> None:
    compatibility_root = APPLICATION_ROOT / "aic-core"

    assert not compatibility_root.is_symlink()
    assert (compatibility_root / "src").resolve(strict=True) == (APPLICATION_ROOT / "src").resolve(strict=True)
    assert (compatibility_root / "rust/aiconfigurator-core/src").resolve(strict=True) == (
        REPOSITORY_ROOT / "crates/core/src/perfmodel"
    ).resolve(strict=True)
    assert (compatibility_root / "rust/aiconfigurator-core/tests").resolve(strict=True) == (
        REPOSITORY_ROOT / "crates/core/tests/perfmodel"
    ).resolve(strict=True)
    assert (compatibility_root / "rust/aiconfigurator-core/parity_tests").resolve(strict=True) == (
        REPOSITORY_ROOT / "crates/core/parity_tests/perfmodel"
    ).resolve(strict=True)
