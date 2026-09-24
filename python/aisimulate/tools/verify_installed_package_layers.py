# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the unified wheel outside the source checkout."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.resources
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.requirements import Requirement

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


def _verify_metadata() -> str:
    wheel_version = _distribution_version("aisimulate")
    if wheel_version is None:
        raise RuntimeError("aisimulate distribution is not installed")
    stale_distributions = [
        name for name in ("aisimulate-core", "aiconfigurator-core") if _distribution_version(name) is not None
    ]
    if stale_distributions:
        raise RuntimeError(f"split Python distributions are unexpectedly installed: {stale_distributions}")

    requirements = importlib.metadata.requires("aisimulate") or []
    requirement_names = {Requirement(requirement).name for requirement in requirements}
    stale_requirements = requirement_names.intersection({"aisimulate-core", "aiconfigurator-core"})
    if stale_requirements:
        raise RuntimeError(f"aisimulate still declares split Python dependencies: {sorted(stale_requirements)}")
    return wheel_version


def _verify_payload() -> None:
    _require_distribution_files(
        "aisimulate",
        (
            "aisimulate/__init__.py",
            "aisimulate_core/__init__.py",
            "aisimulate/legacy_cli/main.py",
            "aisimulate/generator/api.py",
            "aisimulate/sdk/config_adapter/schemas/estimate-request-v1.schema.json",
            "aisimulate_core/_native.py",
            "aisimulate_core/_native.pyi",
            "aisimulate_core/model_configs/meta-llama--Meta-Llama-3.1-8B_config.json",
            "aisimulate_core/sdk/engine.py",
            "aisimulate_core/sdk/memory.py",
            "aisimulate_core/systems/h100_sxm.yaml",
            "collector/__init__.py",
            "collector/model_cases.py",
            "collector/cases/base_ops/mla_module.yaml",
            "collector/fpm_forward/cli.py",
            "collector/fpm_forward/__init__.py",
            "collector/fpm_forward/runtime/fpm_exec.sh",
        ),
    )
    _forbid_distribution_files(
        "aisimulate",
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

    resources = importlib.resources.files("aisimulate_core")
    required_resources = (
        resources / "model_configs" / "meta-llama--Meta-Llama-3.1-8B_config.json",
        resources / "systems" / "h100_sxm.yaml",
        resources / "systems" / "data" / "b200_sxm" / "gemm" / "vllm" / "0.24.0" / "gemm_perf.parquet",
        # Live reuse declaration and provenance sidecar from the post-prune tree.
        resources / "systems" / "data" / "b200_sxm" / "gemm" / "trtllm" / "1.3.0rc23" / "reuse.yaml",
        resources / "systems" / "data" / "b200_sxm" / "gemm" / "vllm" / "0.24.0" / "collection_meta.yaml",
    )
    missing = [str(path) for path in required_resources if not path.is_file()]
    if missing:
        raise RuntimeError(f"aisimulate is missing bundled core resources: {missing}")


def _verify_imports() -> None:
    runtime = importlib.import_module("aisimulate._runtime")
    compatibility_runtime = importlib.import_module("aisimulate_core._native")
    core = importlib.import_module("aisimulate_core")
    importlib.import_module("collector")
    importlib.import_module("collector.fpm_forward")
    if core.AicEngine is not runtime.AicEngine or compatibility_runtime.AicEngine is not runtime.AicEngine:
        raise RuntimeError("AicEngine identity differs across canonical native bindings")
    for name in ("aiconfigurator", "aiconfigurator_core"):
        if importlib.util.find_spec(name) is not None:
            raise RuntimeError(f"removed legacy import namespace is still installed: {name}")

    sdk = importlib.import_module("aisimulate_core.sdk")
    expected_facade = {
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
    }
    if set(sdk.__all__) != expected_facade:
        raise RuntimeError(f"unexpected aisimulate_core.sdk facade: {sdk.__all__!r}")
    for module_name, public_name in (
        ("engine", "EngineHandle"),
        ("memory", "estimate_kv_cache"),
        ("rust_engine_step", "ForwardPassPerfModelConfig"),
        ("rust_engine_step", "ForwardPassPerfOptions"),
    ):
        canonical = importlib.import_module(f"aisimulate_core.sdk.{module_name}")
        legacy = importlib.import_module(f"aisimulate.sdk.{module_name}")
        if getattr(sdk, public_name) is not getattr(canonical, public_name):
            raise RuntimeError(f"SDK facade export {public_name} lost object identity")
        if legacy is not canonical or getattr(legacy, public_name) is not getattr(canonical, public_name):
            raise RuntimeError(f"legacy SDK alias for {module_name}.{public_name} lost object identity")


def _exercise_engine() -> None:
    from aisimulate_core.sdk.engine import EngineHandle

    engine = EngineHandle.compile(
        "MiniMaxAI/MiniMax-M2.5",
        "b200_sxm",
        "vllm",
        backend_version="0.24.0",
        tp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
    )
    prefill_ms = engine.predict_prefill_latency(1, 1024, 0)
    decode_ms = engine.predict_decode_latency(1, 1024, 2)
    if not (prefill_ms > 0 and decode_ms > 0):
        raise RuntimeError(f"unified engine produced invalid latencies: {prefill_ms=}, {decode_ms=}")


def _verify_fpe_probe_results(payload: dict) -> None:
    """Require complete passing probes for each selected role and topology."""
    role_phases = {
        "agg": {"prefill", "decode_start", "decode_end", "mixed"},
        "prefill": {"prefill"},
        "decode": {"decode_start", "decode_end"},
    }
    groups: dict[tuple[str, tuple[str, ...]], set[str]] = {}
    for row in payload["results"]:
        if row["status"] != "PASS":
            raise RuntimeError("installed FPE sentinels did not execute successfully")
        latency = row["latency_ms"]
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise RuntimeError("installed FPE command produced invalid latencies")
        roles = tuple(sorted(row["roles"].split("|")))
        if not set(roles) <= role_phases.keys():
            raise RuntimeError("installed FPE command produced invalid roles")
        compile_args = json.loads(row["reproducer"])["compile"]
        key = (json.dumps(compile_args, sort_keys=True), roles)
        phases = groups.setdefault(key, set())
        if row["phase"] in phases:
            raise RuntimeError("installed FPE command produced duplicate phases")
        phases.add(row["phase"])
    if not groups or payload["metadata"]["plan_count"] != len(groups):
        raise RuntimeError("installed FPE plan count does not match probe results")
    for (_, roles), phases in groups.items():
        if phases != set().union(*(role_phases[role] for role in roles)):
            raise RuntimeError("installed FPE command produced incomplete topology phases")
    if set().union(*groups.values()) != role_phases["agg"]:
        raise RuntimeError("installed FPE command did not exercise all required phases")


def _exercise_fpe_matrix() -> None:
    """Launch the actual matrix command without exposing a source package."""
    generator = Path(__file__).resolve().parent / "support_matrix/generate_fpe_support_matrix.py"
    with tempfile.TemporaryDirectory(prefix="aisim-installed-fpe-") as directory:
        output = Path(directory) / "matrix"
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(generator),
                "--system",
                "b200_sxm",
                "--backend",
                "vllm",
                "--backend-version",
                "0.24.0",
                "--model",
                "Qwen/Qwen3-32B",
                "--max-topologies-per-role",
                "1",
                "--max-workers",
                "1",
                "--output-dir",
                str(output),
            ],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode:
            raise RuntimeError(f"installed FPE command failed: {result.stdout}\n{result.stderr}")
        payload = json.loads((output / "fpe_support_matrix.json").read_text())
        _verify_fpe_probe_results(payload)


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
            "collector/glm53flash_protocol.py",
            "collector/glm53flash_sglang_runtime.py",
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
        (importlib.import_module("collector.glm53flash_protocol"), "collector/glm53flash_protocol.py"),
        (importlib.import_module("collector.glm53flash_sglang_runtime"), "collector/glm53flash_sglang_runtime.py"),
        (importlib.import_module("collector.fpm_forward.sglang_driver"), "collector/fpm_forward/sglang_driver.py"),
    ):
        if Path(module.__file__).resolve() != exact_distribution_path(relative_path):
            raise RuntimeError(f"installed FPM module did not resolve from its exact RECORD path: {module.__file__}")
    for name, relative_path in (
        ("fpm_exec.sh", "collector/fpm_forward/runtime/fpm_exec.sh"),
        ("preflight.py", "collector/fpm_forward/runtime/preflight.py"),
        ("glm53flash/sitecustomize.py", "collector/fpm_forward/runtime/glm53flash/sitecustomize.py"),
        ("glm53flash/glm53flash_scheduler.py", "collector/fpm_forward/runtime/glm53flash/glm53flash_scheduler.py"),
        ("glm53flash/runtime-paths.json", "collector/fpm_forward/runtime/glm53flash/runtime-paths.json"),
        (
            "glm53flash/runtime-source-sha256.json",
            "collector/fpm_forward/runtime/glm53flash/runtime-source-sha256.json",
        ),
        ("glm53flash/README.md", "collector/fpm_forward/runtime/glm53flash/README.md"),
        ("glm53flash/LICENSE", "collector/fpm_forward/runtime/glm53flash/LICENSE"),
        (
            "glm53flash_sglang/runtime-source-sha256.json",
            "collector/fpm_forward/runtime/glm53flash_sglang/runtime-source-sha256.json",
        ),
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
    parser.add_argument("--expect", choices=("fpm", "full", "unified"), default="unified")
    parser.add_argument("--exercise-engine", action="store_true")
    parser.add_argument(
        "--exercise-fpe", action="store_true", help="Run repository FPE tooling against the installed wheel"
    )
    args = parser.parse_args()

    if args.expect == "fpm":
        _verify_fpm_workflow()
        return 0

    wheel_version = _verify_metadata()
    _verify_payload()
    _verify_imports()
    if args.exercise_engine:
        _exercise_engine()
    if args.exercise_fpe:
        _exercise_fpe_matrix()
    print(f"Verified unified aisimulate {wheel_version}: application, canonical SDKs, resources, and native runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
