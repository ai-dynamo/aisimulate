# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Includes changes adapted from:
# https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/cross_package/test_import_contract.py

"""Import compatibility contract between the AIC and AIC Core wheels."""

from __future__ import annotations

import importlib
import importlib.resources
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

CORE_SDK_LEAF_MODULES = [
    "afd_partition",
    "attention_lanes",
    "backends.base_backend",
    "backends.factory",
    "backends.sglang_backend",
    "backends.trtllm_backend",
    "backends.vllm_backend",
    "common",
    "config",
    "config_builders",
    "engine",
    "engine_table_view",
    "errors",
    "fpm_profile",
    "inference_summary",
    "memory",
    "models.base",
    "models.blocks.moe",
    "models.blocks.vit",
    "models.deepseek",
    "models.deepseek_v32",
    "models.deepseek_v4",
    "models.gemma4",
    "models.gpt",
    "models.helpers",
    "models.hybrid_moe",
    "models.kimi_k3",
    "models.llama",
    "models.minimax_m3",
    "models.moe",
    "models.muse_glimmer",
    "models.nemotron_h",
    "models.nemotron_nas",
    "models.qwen35",
    "models.qwen3vl",
    "models.step3p7",
    "models.vit_ops",
    "operations.afd_transfer",
    "operations.attention",
    "operations.base",
    "operations.communication",
    "operations.dsa",
    "operations.dsv4",
    "operations.elementwise",
    "operations.embedding",
    "operations.fpm_forward",
    "operations.gemm",
    "operations.mamba",
    "operations.mla",
    "operations.moe",
    "operations.moe_comm",
    "operations.msa",
    "operations.overlap",
    "operations.util_empirical",
    "perf_database",
    # perf_interp.* retired with the Python per-call query stack (#1357 PR-5):
    # per-op interpolation lives in the compiled engine.
    "performance_result",
    "rust_engine_step",
    "speculation.base",
    "speculation.dense_draft",
    "speculation.dflash",
    "speculation.draft_model",
    "speculation.dspark",
    "speculation.eagle",
    "speculation.materialize",
    "speculation.mtp",
    "speculation.ngram",
    "step_estimate",
    "system_spec",
    "utils",
    "work_delta.field",
    "work_delta.planner",
    "work_delta.solver",
]


def _discover_python_leaves(root: object, prefix: str = "") -> set[str]:
    """Return import suffixes for every non-package Python module below root."""
    modules: set[str] = set()
    for child in root.iterdir():
        if child.name == "__pycache__":
            continue
        if child.is_dir():
            modules.update(_discover_python_leaves(child, f"{prefix}{child.name}."))
        elif child.name.endswith(".py") and child.name != "__init__.py":
            modules.add(f"{prefix}{child.name.removesuffix('.py')}")
    return modules


def test_import_contract_covers_every_core_sdk_leaf() -> None:
    """A new core SDK module must add a legacy wrapper and contract case."""
    core_sdk_root = importlib.resources.files("aiconfigurator_core.sdk")

    assert set(CORE_SDK_LEAF_MODULES) == _discover_python_leaves(core_sdk_root)


@pytest.mark.parametrize("module_suffix", CORE_SDK_LEAF_MODULES)
def test_legacy_leaf_module_is_canonical_module(module_suffix: str) -> None:
    """Every compatibility leaf must share caches and private module state."""
    legacy_name = f"aiconfigurator.sdk.{module_suffix}"
    canonical_name = f"aiconfigurator_core.sdk.{module_suffix}"

    legacy_module = importlib.import_module(legacy_name)
    canonical_module = importlib.import_module(canonical_name)

    assert legacy_module is canonical_module
    assert sys.modules[legacy_name] is sys.modules[canonical_name]


@pytest.mark.parametrize("namespace", ["aiconfigurator.sdk", "aisimulate_core.sdk"])
def test_fpm_profile_alias_preserves_module_and_type_identity(namespace: str) -> None:
    """All public profile imports must share the canonical classes and state."""
    alias_name = f"{namespace}.fpm_profile"
    alias = importlib.import_module(alias_name)
    canonical = importlib.import_module("aiconfigurator_core.sdk.fpm_profile")

    assert alias is canonical
    assert sys.modules[alias_name] is canonical
    for name in ("FpmModelProfile", "FpmDeploymentProfile", "FpmResourceProfile"):
        assert getattr(alias, name) is getattr(canonical, name)


@pytest.mark.parametrize("namespace", ["aiconfigurator.sdk", "aisimulate_core.sdk"])
def test_fpm_profile_alias_instances_load_and_compile(namespace: str, tmp_path: Path) -> None:
    """Profiles created through either facade must reach native compilation."""
    from aiconfigurator_core.sdk import engine
    from aiconfigurator_core.sdk.fpm_profile import FpmModelProfile, load_fpm_profile

    alias = importlib.import_module(f"{namespace}.fpm_profile")
    profile = alias.FpmModelProfile.model_validate(
        {
            "schema_version": 1,
            "model": "test/unknown-decoder",
            "model_revision": "import-contract-fixture-v1",
            "architecture": "ImportContractDecoderForCausalLM",
            "context_length": 4096,
            "num_experts": 0,
            "provenance": "Synthetic metadata for import compatibility; not a GPU measurement.",
            "deployments": [
                {
                    "system": "test_gpu",
                    "backend": "vllm",
                    "backend_version": "0.25.1",
                    "tp": 1,
                    "dp": 1,
                    "moe_tp": 1,
                    "moe_ep": 1,
                    "gemm_quant_mode": "fp8",
                    "moe_quant_mode": "fp8",
                    "fmha_quant_mode": "fp8",
                    "comm_quant_mode": "half",
                    "kv_cache_dtype": "fp8",
                    "resources": {
                        "weights_bytes": 100,
                        "activations_bytes": 20,
                        "runtime_overhead_bytes": 30,
                        "comm_overhead_bytes": 50,
                        "kv_bytes_per_token": 10,
                        "cache_layout": "linear",
                        "max_num_tokens": 8192,
                        "max_batch_size": 256,
                        "provenance": "Synthetic rank-local resource bounds; excludes CUDA graphs.",
                    },
                }
            ],
        }
    )

    loaded = load_fpm_profile(profile)
    assert type(loaded) is FpmModelProfile
    assert loaded.model_dump() == profile.model_dump()

    canonical = json.loads(
        engine.aiconfigurator_core.RustForwardPassPerfModel.normalize_config(
            json.dumps(
                {
                    "model": profile.model,
                    "system": "test_gpu",
                    "backend": "vllm",
                    "backend_version": "0.25.1",
                    "worker_type": "aggregated",
                    "estimation_mode": "fpm_interpolation",
                    "fpm_profile": profile.model_dump(mode="json"),
                }
            )
        )
    )
    assert canonical["fpm_profile"] == loaded.model_dump(mode="json")
    assert canonical["estimator_config"]["fpm_interpolation"]["method"] == "direct"
    compiled = engine.compile_engine(
        profile.model,
        "test_gpu",
        "vllm",
        "0.25.1",
        systems_path=str(tmp_path),
        forward_model="fpm",
        fpm_profile=profile,
        fpm_interpolation=canonical["estimator_config"]["fpm_interpolation"]["method"],
    )
    assert isinstance(compiled, bytes)
    assert compiled


@pytest.mark.parametrize("package_suffix", ["models", "operations", "speculation"])
def test_legacy_package_reexports_canonical_public_surface(package_suffix: str) -> None:
    """Package facades preserve child wrappers and export canonical objects."""
    legacy_package = importlib.import_module(f"aiconfigurator.sdk.{package_suffix}")
    canonical_package = importlib.import_module(f"aiconfigurator_core.sdk.{package_suffix}")

    assert legacy_package.__all__ == canonical_package.__all__
    for public_name in canonical_package.__all__:
        assert getattr(legacy_package, public_name) is getattr(canonical_package, public_name)


def test_models_package_delegates_private_registry() -> None:
    """Private registry access sees the canonical registry, not a copied one."""
    legacy_models = importlib.import_module("aiconfigurator.sdk.models")
    canonical_models = importlib.import_module("aiconfigurator_core.sdk.models")

    assert legacy_models._MODEL_REGISTRY is canonical_models._MODEL_REGISTRY


@pytest.mark.parametrize(
    ("package_suffix", "attribute"),
    [
        ("models", "_get_model_info"),
        ("operations", "clear_all_op_caches"),
    ],
)
def test_legacy_package_patch_updates_canonical_package(package_suffix: str, attribute: str) -> None:
    """Patching a legacy package attribute must affect canonical code."""
    canonical_package = importlib.import_module(f"aiconfigurator_core.sdk.{package_suffix}")

    with patch(f"aiconfigurator.sdk.{package_suffix}.{attribute}") as mocked:
        assert getattr(canonical_package, attribute) is mocked

    assert getattr(canonical_package, attribute) is not mocked


def test_operations_baseline_exports_survive() -> None:
    """Frozen baseline of the public ``operations`` surface.

    The facade tests above compare the two LIVE facades to each other, so
    they stay green even when a previously exported name disappears from
    both at once. This literal list pins the surface as of the Python
    engine-step retirement (#1521): removing a name from it is a public-SDK
    break and must be a deliberate, reviewed edit here — after a deprecation
    window — never a side effect.
    """
    baseline = {
        # (Mamba2 removed deliberately: the deprecated composite's window
        #  closed with the deprecation-cleanup PR.)
        "FPMForwardOp",
        "Mamba2Kernel",
        "GDNKernel",
        "KDAKernel",
        "GEMM",
        "MoE",
        "ContextAttention",
        "GenerationAttention",
        "ContextMLA",
        "GenerationMLA",
        "CustomAllReduce",
        "MoEAllToAll",
        "AFDTransfer",
        "AFDCombine",
        "Embedding",
        "ElementWise",
        "P2P",
    }
    operations = importlib.import_module("aiconfigurator.sdk.operations")
    exported = set(operations.__all__)
    missing = baseline - exported
    assert not missing, f"public operations exports removed without a deprecation window: {sorted(missing)}"
    for name in sorted(baseline):
        assert getattr(operations, name) is not None


def test_fpm_forward_op_keeps_legacy_constructor_layout() -> None:
    """Baseline signature pin for the exported ``FPMForwardOp``.

    The legacy layout is ``(phase, model_config, model_path, sol_fn=None,
    weight_bytes=0.0, sol_ops=None)``. The ``sol_fn`` slot is retired but
    keeps its position so positional ``weight_bytes``/``sol_ops`` callers
    keep their meaning; passing a callback raises a targeted migration
    error instead of silently rebinding parameters.
    """
    import inspect

    from aiconfigurator.sdk.operations import FPMForwardOp

    params = list(inspect.signature(FPMForwardOp.__init__).parameters)
    assert params == ["self", "phase", "model_config", "model_path", "sol_fn", "weight_bytes", "sol_ops"]


def test_representative_from_imports_return_canonical_objects() -> None:
    """The user-facing from-import form remains backward compatible."""
    from aiconfigurator.sdk.config import ModelConfig as LegacyModelConfig
    from aiconfigurator.sdk.models import GPTModel as LegacyGPTModel
    from aiconfigurator.sdk.operations import GEMM as LEGACY_GEMM
    from aiconfigurator_core.sdk.config import ModelConfig
    from aiconfigurator_core.sdk.models import GPTModel
    from aiconfigurator_core.sdk.operations import GEMM

    assert LegacyModelConfig is ModelConfig
    assert LegacyGPTModel is GPTModel
    assert LEGACY_GEMM is GEMM


@pytest.mark.parametrize("module_suffix", [name for name in CORE_SDK_LEAF_MODULES if name.startswith("speculation.")])
def test_aisimulate_speculation_leaf_preserves_identity(module_suffix: str) -> None:
    preferred = importlib.import_module(f"aisimulate_core.sdk.{module_suffix}")
    canonical = importlib.import_module(f"aiconfigurator_core.sdk.{module_suffix}")
    assert preferred is canonical


def test_aisimulate_speculation_package_preserves_registry() -> None:
    preferred = importlib.import_module("aisimulate_core.sdk.speculation")
    canonical = importlib.import_module("aiconfigurator_core.sdk.speculation")
    for name in canonical.__all__:
        assert getattr(preferred, name) is getattr(canonical, name)
    assert (
        preferred.get_spec_scheme_cls("mtp")
        is importlib.import_module("aiconfigurator_core.sdk.speculation.mtp").MTPScheme
    )
