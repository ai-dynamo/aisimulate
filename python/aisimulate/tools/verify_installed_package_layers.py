# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test installed upper/core package layers outside the source checkout."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.resources
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Running this file directly prepends ``tools/`` to sys.path. Remove that path
# so installed-package checks cannot accidentally resolve repository helpers.
_TOOLS_DIR = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _TOOLS_DIR]


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_distribution_files(name: str, required: tuple[str, ...]) -> None:
    distribution = importlib.metadata.distribution(name)
    files = {str(path): path for path in distribution.files or ()}
    missing = [path for path in required if path not in files or not distribution.locate_file(files[path]).is_file()]
    if missing:
        raise RuntimeError(f"distribution {name!r} is missing installed files: {missing}")


def _forbid_distribution_files(name: str, fragments: tuple[str, ...]) -> None:
    distribution = importlib.metadata.distribution(name)
    files = tuple(str(path) for path in distribution.files or ())
    forbidden = sorted(path for path in files if any(fragment in path for fragment in fragments))
    if forbidden:
        raise RuntimeError(f"distribution {name!r} contains forbidden files: {forbidden}")


def _forbid_module(name: str) -> None:
    if importlib.util.find_spec(name) is not None:
        raise RuntimeError(f"module {name!r} belongs to an uninstalled layer")


def _verify_core(*, exercise_engine: bool) -> str:
    core_version = _distribution_version("aiconfigurator-core")
    if core_version is None:
        raise RuntimeError("aiconfigurator-core distribution is not installed")
    _require_distribution_files(
        "aiconfigurator-core",
        (
            "aiconfigurator_core/__init__.py",
            "aiconfigurator_core/_aiconfigurator_core.pyi",
            "aiconfigurator_core/model_configs/meta-llama--Meta-Llama-3.1-8B_config.json",
            "aiconfigurator_core/py.typed",
            "aiconfigurator_core/sdk/__init__.py",
            "aiconfigurator_core/sdk/engine.py",
            "aiconfigurator_core/sdk/memory.py",
            "aiconfigurator_core/systems/h100_sxm.yaml",
        ),
    )
    _forbid_distribution_files(
        "aiconfigurator-core",
        (
            "config_adapter",
            "gap_analysis",
            "auto-gap-analysis",
            "datasets/",
            "reports/",
            "/skills/",
            "/tools/",
            "web/",
            "webapp/",
        ),
    )

    core = importlib.import_module("aiconfigurator_core")
    if core._build_smoke() != 1:
        raise RuntimeError("native core extension returned an unexpected schema version")

    sdk = importlib.import_module("aiconfigurator_core.sdk")
    protected_sdk_modules = {
        "aiconfigurator_core.sdk.engine",
        "aiconfigurator_core.sdk.memory",
        "aiconfigurator_core.sdk.rust_engine_step",
    }
    eagerly_loaded_modules = protected_sdk_modules.intersection(sys.modules)
    if eagerly_loaded_modules:
        raise RuntimeError(f"aiconfigurator_core.sdk eagerly loaded modules: {sorted(eagerly_loaded_modules)}")

    expected_facade = {
        "EngineHandle",
        "ModelConfig",
        "RuntimeConfig",
        "RustForwardPassPerfModel",
        "compile_engine",
        "estimate_kv_cache",
        "estimate_num_gpu_blocks",
    }
    if set(sdk.__all__) != expected_facade:
        raise RuntimeError(f"unexpected aiconfigurator_core.sdk facade: {sdk.__all__!r}")
    for public_name in expected_facade:
        if getattr(sdk, public_name, None) is None:
            raise RuntimeError(f"aiconfigurator_core.sdk is missing {public_name}")

    for module in (
        "aiconfigurator_core.sdk.engine",
        "aiconfigurator_core.sdk.memory",
        "aiconfigurator_core.sdk.perf_database",
    ):
        importlib.import_module(module)

    resources = importlib.resources.files("aiconfigurator_core")
    required_resources = (
        resources / "model_configs" / "meta-llama--Meta-Llama-3.1-8B_config.json",
        resources / "systems" / "h100_sxm.yaml",
        resources / "systems" / "data" / "b200_sxm" / "gemm" / "vllm" / "0.19.0" / "gemm_perf.parquet",
        resources / "systems" / "data" / "l40s" / "gemm" / "vllm" / "0.22.0" / "reuse.yaml",
        resources / "systems" / "data" / "b200_sxm" / "gemm" / "vllm" / "0.19.0" / "collection_meta.yaml",
    )
    missing = [str(path) for path in required_resources if not path.is_file()]
    if missing:
        raise RuntimeError(f"standalone core is missing bundled resources: {missing}")

    if exercise_engine:
        from aiconfigurator_core.sdk.engine import EngineHandle

        engine = EngineHandle.compile(
            "MiniMaxAI/MiniMax-M2.5",
            "b200_sxm",
            "vllm",
            backend_version="0.19.0",
            tp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
        )
        prefill_ms = engine.predict_prefill_latency(1, 1024, 0)
        decode_ms = engine.predict_decode_latency(1, 1024, 2)
        if not (prefill_ms > 0 and decode_ms > 0):
            raise RuntimeError(f"standalone core produced invalid latencies: {prefill_ms=}, {decode_ms=}")

    return core_version


def _verify_upper(*, import_runtime: bool) -> str:
    aic_version = _distribution_version("aiconfigurator")
    if aic_version is None:
        raise RuntimeError("aiconfigurator distribution is not installed")
    _require_distribution_files(
        "aiconfigurator",
        (
            "aiconfigurator/cli/main.py",
            "aiconfigurator/generator/api.py",
            "aiconfigurator/sdk/_compat.py",
            "aiconfigurator/sdk/config_adapter/__init__.py",
            "aiconfigurator/sdk/config_adapter/schemas/estimate-request-v1.schema.json",
            "aiconfigurator/sdk/engine.py",
            "aiconfigurator/sdk/memory.py",
            "aiconfigurator/sdk/task_v2.py",
        ),
    )
    _forbid_distribution_files(
        "aiconfigurator",
        (
            "gap_analysis",
            "auto-gap-analysis",
            "datasets/",
            "reports/",
            "/skills/",
            "/tools/",
            "web/",
            "webapp/",
            ".agents/",
        ),
    )
    config_adapter = importlib.import_module("aiconfigurator.sdk.config_adapter")
    if config_adapter.EstimateRequestV1.schema_path().is_file() is False:
        raise RuntimeError("upper package is missing the config-adapter JSON Schema")
    if import_runtime:
        for module in ("aiconfigurator.cli.main", "aiconfigurator.generator.api"):
            importlib.import_module(module)
    for module in ("aiconfigurator.webapp", "spica"):
        _forbid_module(module)
    return aic_version


def _verify_legacy_sdk_compatibility() -> None:
    """Verify representative legacy aliases and the upper-owned Task API."""
    for module_name, public_name in (
        ("engine", "EngineHandle"),
        ("memory", "estimate_kv_cache"),
    ):
        canonical = importlib.import_module(f"aiconfigurator_core.sdk.{module_name}")
        legacy = importlib.import_module(f"aiconfigurator.sdk.{module_name}")
        if legacy is not canonical:
            raise RuntimeError(
                f"aiconfigurator.sdk.{module_name} is not the canonical aiconfigurator_core.sdk.{module_name} module"
            )
        if getattr(legacy, public_name) is not getattr(canonical, public_name):
            raise RuntimeError(f"legacy {public_name} is not the canonical core object")

    task_module = importlib.import_module("aiconfigurator.sdk.task_v2")
    if task_module.Task.__module__ != "aiconfigurator.sdk.task_v2":
        raise RuntimeError("Task must remain implemented by the upper aiconfigurator package")
    if importlib.util.find_spec("aiconfigurator_core.sdk.task_v2") is not None:
        raise RuntimeError("Task must not be shipped by the standalone core package")


def _verify_fpm_workflow() -> str:
    """Verify the installed application owns a runnable FPM workflow."""

    app_version = _distribution_version("aisimulate")
    if app_version is None:
        raise RuntimeError("aisimulate distribution is not installed")
    _require_distribution_files(
        "aisimulate",
        (
            "collector/__init__.py",
            "collector/model_cases.py",
            "collector/cases/base_ops/mla_module.yaml",
            "collector/cases/models/GlmMoeDsaForCausalLM_cases.yaml",
            "collector/cases/models/MiniMaxM3ForCausalLM_cases.yaml",
            "collector/fpm_forward/__main__.py",
            "collector/fpm_forward/cli.py",
            "collector/fpm_forward/runtime/fpm_exec.sh",
            "collector/fpm_forward/runtime/preflight.py",
        ),
    )

    runtime = importlib.resources.files("collector.fpm_forward.runtime")
    planner = importlib.import_module("collector.fpm_forward.planner")
    runner = importlib.import_module("collector.fpm_forward.runner")
    distribution = importlib.metadata.distribution("aisimulate")
    distribution_files = {str(path): path for path in distribution.files or ()}
    distribution_root = Path(os.fspath(distribution.locate_file(""))).resolve()

    def exact_distribution_path(relative_path: str) -> Path:
        record_path = distribution_files.get(relative_path)
        if record_path is None:
            raise RuntimeError(f"AISimulate RECORD does not own required FPM path: {relative_path}")
        located = Path(os.fspath(distribution.locate_file(record_path))).resolve()
        expected = (distribution_root / Path(*relative_path.split("/"))).resolve()
        if located != expected or not located.is_relative_to(distribution_root):
            raise RuntimeError(f"AISimulate RECORD resolves FPM path outside the distribution: {relative_path}")
        return located

    for module, relative_path in (
        (planner, "collector/fpm_forward/planner.py"),
        (runner, "collector/fpm_forward/runner.py"),
    ):
        if Path(module.__file__).resolve() != exact_distribution_path(relative_path):
            raise RuntimeError(f"installed FPM module did not resolve from its exact RECORD path: {module.__file__}")
    for name, relative_path in (
        ("fpm_exec.sh", "collector/fpm_forward/runtime/fpm_exec.sh"),
        ("preflight.py", "collector/fpm_forward/runtime/preflight.py"),
    ):
        asset = Path(os.fspath(runtime / name)).resolve()
        if not asset.is_file():
            raise RuntimeError(f"installed FPM runtime asset is missing: {asset}")
        if asset != exact_distribution_path(relative_path):
            raise RuntimeError(f"installed FPM runtime asset did not resolve from its exact RECORD path: {asset}")

    env = {
        key: value for key, value in os.environ.items() if key not in {"FPM_COLLECTOR_SOURCE_REVISION", "PYTHONPATH"}
    }
    plans = []
    with tempfile.TemporaryDirectory(prefix="aisimulate-installed-fpm-") as root:
        for run_number in (1, 2):
            workdir = Path(root) / f"run-{run_number}"
            workdir.mkdir()
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "collector.fpm_forward",
                    "--model-path",
                    "nvidia/GLM-5.2-NVFP4",
                    "--gpu",
                    "b200_sxm",
                    "--fpm-max-gpus",
                    "4",
                    "--plan-only",
                ],
                cwd=workdir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            try:
                plan = json.loads(completed.stdout)
            except json.JSONDecodeError:
                plan = {}
            revision = plan.get("aic_revision")
            if (
                completed.returncode != 0
                or plan.get("schema_name") != "aic_fpm_collection_plan"
                or not isinstance(revision, str)
                or not revision.startswith(f"installed:aisimulate=={app_version}:record-sha256:")
                or not plan.get("cells")
                or str(workdir) in completed.stdout
            ):
                raise RuntimeError(
                    f"installed FPM module entry point failed:\nstdout={completed.stdout}\nstderr={completed.stderr}"
                )
            plans.append(plan)
    if plans[0] != plans[1]:
        raise RuntimeError("installed FPM plan identity is not stable across outside-checkout working directories")
    print(
        f"Verified installed AISimulate {app_version} FPM workflow and runtime assets "
        f"with revision {plans[0]['aic_revision']}"
    )
    return app_version


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", choices=("core", "fpm", "full", "upper"), required=True)
    parser.add_argument("--exercise-engine", action="store_true")
    args = parser.parse_args()

    if args.expect == "fpm":
        _verify_fpm_workflow()
        return 0

    if args.expect == "core":
        core_version = _verify_core(exercise_engine=args.exercise_engine)
        if _distribution_version("aiconfigurator") is not None:
            raise RuntimeError("core-only install unexpectedly contains the aiconfigurator distribution")
        for module in ("aiconfigurator", "spica"):
            _forbid_module(module)
        print(
            f"Verified standalone aiconfigurator-core {core_version}, including canonical "
            "aiconfigurator_core.sdk imports, resources, and native extension"
        )
        return 0

    if args.expect == "full":
        core_version = _verify_core(exercise_engine=args.exercise_engine)
        aic_version = _verify_upper(import_runtime=True)
        if core_version != aic_version:
            raise RuntimeError(f"upper/core version mismatch: {aic_version=} {core_version=}")
        _verify_legacy_sdk_compatibility()
        print(
            f"Verified full aiconfigurator {aic_version} with standalone core, legacy SDK aliases, "
            "and upper-owned aiconfigurator.sdk.task_v2.Task"
        )
        return 0

    aic_version = _verify_upper(import_runtime=False)
    if _distribution_version("aiconfigurator-core") is not None:
        raise RuntimeError("upper-only install unexpectedly contains aiconfigurator-core metadata")
    _forbid_module("aiconfigurator_core")
    if importlib.util.find_spec("aiconfigurator.sdk.engine") is None:
        raise RuntimeError("upper-only install is missing the legacy aiconfigurator.sdk.engine wrapper")
    print(
        f"Verified upper-only aiconfigurator payload {aic_version}; legacy SDK wrappers remain installed "
        "and await the intentionally absent core dependency"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
