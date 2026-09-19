# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement

APPLICATION_ROOT = Path(__file__).resolve().parents[2]


def test_core_metadata_is_cold_and_lazy_native_exports_retain_identity() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import aisimulate_core as core
from aisimulate_core import fpm_profile, quantization
blocked = ('aisimulate._runtime', 'aisimulate_core._native', 'aisimulate_core.sdk')
assert not any(name in sys.modules for name in blocked)
assert set(core.__all__) <= set(dir(core))
try:
    core.unknown_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown native export must fail')
assert 'aisimulate._runtime' not in sys.modules
core.AicEngine
from aisimulate import _runtime
namespace = {}
exec('from aisimulate_core import *', namespace)
for name in core.__all__:
    assert namespace[name] is getattr(core, name)
    if name != '__version__':
        assert getattr(core, name) is getattr(_runtime, name)
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_canonical_core_exports_the_unified_native_types() -> None:
    core = importlib.import_module("aisimulate_core")
    runtime = importlib.import_module("aisimulate._runtime")

    for name in (
        "AicEngine",
        "RustForwardPassPerfModel",
        "engine_spec_schema_version",
        "gemm_quant_util_levels",
        "moe_quant_util_levels",
        "op_from_spec_json",
        "ops_json_from_ops",
        "resolve_op_sources_report_json",
        "table_view_attributes",
        "weights_ops_json",
    ):
        assert getattr(core, name) is getattr(runtime, name)
    assert core.__version__ == importlib.metadata.version("aisimulate")


def test_public_sdk_facade_and_explicit_modules_share_identity() -> None:
    sdk = importlib.import_module("aisimulate_core.sdk")
    engine = importlib.import_module("aisimulate_core.sdk.engine")
    errors = importlib.import_module("aisimulate_core.sdk.errors")
    application_errors = importlib.import_module("aisimulate.sdk.errors")

    assert sdk.compile_engine is engine.compile_engine
    assert errors.PerfDataNotAvailableError is application_errors.PerfDataNotAvailableError


def test_aisimulate_owns_core_without_split_dependencies() -> None:
    aisimulate_core = importlib.import_module("aisimulate_core")
    requirements = importlib.metadata.requires("aisimulate") or []
    dependency_names = {Requirement(requirement).name for requirement in requirements}

    assert importlib.metadata.version("aisimulate") == aisimulate_core.__version__
    assert "aisimulate-core" not in dependency_names
    assert "aiconfigurator-core" not in dependency_names
    assert "dynamo" not in dependency_names
    assert "ai-dynamo" not in dependency_names


def test_native_binding_module_forwards_to_the_unified_runtime() -> None:
    runtime = importlib.import_module("aisimulate._runtime")
    compatibility_runtime = importlib.import_module("aisimulate_core._native")

    assert compatibility_runtime.AicEngine is runtime.AicEngine
    assert compatibility_runtime.RustForwardPassPerfModel is runtime.RustForwardPassPerfModel


def test_native_binding_wildcard_import_preserves_runtime_identity() -> None:
    runtime = importlib.import_module("aisimulate._runtime")
    compatibility_runtime = importlib.import_module("aisimulate_core._native")
    namespace: dict[str, object] = {}

    exec("from aisimulate_core._native import *", {}, namespace)

    assert set(namespace) == set(compatibility_runtime.__all__)
    for name in compatibility_runtime.__all__:
        assert namespace[name] is getattr(runtime, name)


def test_only_canonical_source_packages_remain() -> None:
    source = APPLICATION_ROOT / "src"
    assert {path.name for path in source.iterdir() if path.is_dir()} == {"aisimulate", "aisimulate_core"}
    assert not (APPLICATION_ROOT / "aic-core").exists()


def test_legacy_import_namespaces_are_not_installed() -> None:
    # Run this in the clean wheel environment too: source-only absence is insufficient.
    assert importlib.util.find_spec("aiconfigurator") is None
    assert importlib.util.find_spec("aiconfigurator_core") is None


def test_resource_paths_belong_to_the_canonical_core() -> None:
    from importlib.resources import files

    core = files("aisimulate_core")
    assert (core / "systems" / "h200_sxm.yaml").is_file()
    assert (core / "model_configs").is_dir()
    assert (files("aisimulate") / "legacy_cli" / "example.yaml").is_file()
    assert (files("aisimulate") / "generator" / "config" / "deployment_config.yaml").is_file()
