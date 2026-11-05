# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-wheel regression for the installed FPM workflow."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

# Release wheel builds on arm64 routinely approach the suite's 180-second
# per-test limit, so keep a larger guard for this build-and-install workflow.
pytestmark = [pytest.mark.e2e, pytest.mark.build, pytest.mark.timeout(600)]

REPO_ROOT = Path(__file__).resolve().parents[5]
APP_ROOT = REPO_ROOT / "python" / "aisimulate"
VERIFY_INSTALLED_LAYERS = APP_ROOT / "tools" / "verify_installed_package_layers.py"


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None, timeout: int = 300):
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_built_application_wheel_runs_installed_fpm_plan_and_resolves_runtime_assets(tmp_path):
    supplied_wheel = os.environ.get("AISIMULATE_TEST_WHEEL")
    if supplied_wheel:
        wheels = (Path(supplied_wheel).resolve(),)
    else:
        wheel_dir = tmp_path / "wheel"
        wheel_dir.mkdir()
        _run(
            [sys.executable, "-m", "maturin", "build", "--release", "--out", str(wheel_dir)],
            cwd=APP_ROOT,
            timeout=600,
        )
        wheels = tuple(wheel_dir.glob("aisim-*.whl"))
    assert len(wheels) == 1
    assert wheels[0].is_file()

    unrelated_repository = tmp_path / "unrelated-git"
    unrelated_repository.mkdir()
    _run(["git", "init"], cwd=unrelated_repository)
    (unrelated_repository / "host-source.txt").write_text("unrelated source\n", encoding="utf-8")
    _run(["git", "add", "host-source.txt"], cwd=unrelated_repository)
    _run(
        [
            "git",
            "-c",
            "user.name=AISimulate Build Test",
            "-c",
            "user.email=aisimulate-build-test@example.invalid",
            "commit",
            "-m",
            "unrelated host commit",
        ],
        cwd=unrelated_repository,
    )
    unrelated_head = _run(["git", "rev-parse", "HEAD"], cwd=unrelated_repository).stdout.strip()

    # Put the complete virtual environment under an unrelated Git checkout.
    # The installed planner must use wheel provenance before it can observe
    # this ambient repository and its deliberately different HEAD.
    install_root = unrelated_repository / "installed"
    _run([sys.executable, "-m", "venv", str(install_root)], cwd=unrelated_repository)
    installed_python = install_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run(
        [str(installed_python), "-m", "pip", "install", "--no-deps", str(wheels[0])],
        cwd=unrelated_repository,
    )

    # Reuse installed third-party dependencies without exposing either app or
    # core source trees. The unified wheel must own both packages and native code.
    dependency_paths = [Path(sysconfig.get_path("purelib")).resolve()]
    installed_purelib = Path(
        _run(
            [str(installed_python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            cwd=unrelated_repository,
        ).stdout.strip()
    )
    (installed_purelib / "aisimulate-build-test-dependencies.pth").write_text(
        "".join(f"{path}\n" for path in dependency_paths),
        encoding="utf-8",
    )

    env = {
        key: value for key, value in os.environ.items() if key not in {"FPM_COLLECTOR_SOURCE_REVISION", "PYTHONPATH"}
    }
    env["PYTHONNOUSERSITE"] = "1"
    completed = _run(
        [str(installed_python), str(VERIFY_INSTALLED_LAYERS), "--expect", "fpm"],
        cwd=unrelated_repository,
        env=env,
    )
    installed_version = _run(
        [
            str(installed_python),
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('aisim'))",
        ],
        cwd=unrelated_repository,
        env=env,
    ).stdout.strip()

    expected_version = tomllib.loads((APP_ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert installed_version == expected_version
    assert f"Verified installed AISimulate {installed_version} FPM workflow" in completed.stdout
    assert f"installed:aisim=={installed_version}:record-sha256:" in completed.stdout
    assert unrelated_head not in completed.stdout
    assert str(unrelated_repository) not in completed.stdout

    learned = _run(
        [
            str(installed_python),
            "-c",
            textwrap.dedent(
                """
                import hashlib
                import json
                import shutil
                import sys
                from pathlib import Path

                import aisimulate
                import aisimulate_core
                from aisimulate import _runtime
                from aisimulate_core.sdk import fpm_bundle, fpm_learned, gbdt_example, rust_engine_step
                from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

                installed = Path(sys.argv[1]).resolve()
                modules = (
                    aisimulate, aisimulate_core, _runtime, fpm_bundle, fpm_learned, gbdt_example, rust_engine_step
                )
                for module in modules:
                    assert Path(module.__file__).resolve().is_relative_to(installed), module.__file__

                identity = {
                    "model": "wheel-test-model", "system": "wheel-test-system", "backend": "vllm",
                    "backend_version": "1", "worker_type": "decode",
                }
                rows = [[{
                    "version": 1, "worker_id": "wheel-test", "dp_rank": 0, "counter_id": index,
                    "wall_time": (1 + index % 8) / 1000,
                    "scheduled_requests": {
                        "num_decode_requests": 1 + index % 8, "sum_decode_kv_tokens": 100 * (1 + index % 8)
                    },
                }] for index in range(32)]
                source = Path("training.json")
                source.write_text(json.dumps(rows))
                config = fpm_learned.train_bundle(
                    rows, "decode", output_dir="learned-bundle", identity=identity,
                    provenance={
                        "anchor_kind": "b",
                        "sources": [{
                            "reference": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()
                        }],
                        "evaluation_sources": [],
                    },
                    max_iter=3, max_leaf_nodes=3, min_samples_leaf=2, min_store_rows=5,
                )
                before = RustForwardPassPerfModel.best_available(config)
                expected = before.estimate_forward_pass_time_ms(rows[0])
                assert expected is not None
                before.close()
                installed_config = fpm_learned.install_bundle("learned-bundle/bundle.json", "systems")
                shutil.rmtree("learned-bundle")
                source.unlink()
                frozen_config = fpm_learned.inspect_bundle(
                    installed_config["estimator_config"]["fpm_learned"]["bundle_path"], adaptation="frozen"
                )
                frozen = RustForwardPassPerfModel.best_available(frozen_config)
                assert frozen.estimate_forward_pass_time_ms(rows[0]) == expected
                frozen.close()
                model = RustForwardPassPerfModel.best_available({
                    **identity, "estimation_mode": "fpm_learned", "systems_paths": [str(Path("systems").resolve())],
                })
                assert model.estimate_forward_pass_time_ms(rows[0]) == expected
                assert model.dependency_info()["files"]
                assert all(Path(item["path"]).is_file() for item in model.dependency_info()["files"])
                model.tune_with_fpms(rows[:2])
                expected = model.estimate_forward_pass_time_ms(rows[0])
                model.save_state_snapshot("seeded-snapshot")
                shutil.move("seeded-snapshot", "moved-seeded-snapshot")
                shutil.rmtree("systems")
                restored = RustForwardPassPerfModel.best_available(state_snapshot="moved-seeded-snapshot")
                assert restored.estimate_forward_pass_time_ms(rows[0]) == expected
                assert restored.readiness_report()["execution"]["ready"]
                try:
                    restored.tune_with_fpms(rows[2:3])
                except ValueError as error:
                    assert "on_tune" in str(error)
                else:
                    raise AssertionError("Fixed seeded snapshot accepted feedback")
                continuation = restored.diagnostics()["provenance"]["config"]
                continuation["estimator_config"]["state_snapshot"]["on_tune"] = "continue"
                restored.close()
                resumed = RustForwardPassPerfModel.best_available(continuation)
                for learner in (model, resumed):
                    learner.tune_with_fpms(rows[2:6])
                assert resumed.export_learned_artifact() == model.export_learned_artifact()
                resumed.close()
                model.close()
                print("Verified installed learned train, bundle, install and discovery; portable continuation")
                """
            ),
            str(installed_purelib),
        ],
        cwd=unrelated_repository,
        env=env,
    )
    assert "Verified installed learned train, bundle, install and discovery" in learned.stdout
    metadata = _run(
        [
            str(installed_python),
            "-c",
            textwrap.dedent(
                """
                import importlib.metadata
                from packaging.requirements import Requirement

                entries = [Requirement(value) for value in importlib.metadata.requires("aisimulate")]
                learned = [entry for entry in entries if entry.name.lower().replace("_", "-") == "scikit-learn"]
                assert len(learned) == 1
                assert str(learned[0].specifier) == "==1.9.1"
                assert learned[0].marker.evaluate({"extra": "learned"})
                assert not learned[0].marker.evaluate({"extra": ""})
                print("Verified supported optional trainer requirement")
                """
            ),
        ],
        cwd=unrelated_repository,
        env=env,
    )
    assert "Verified supported optional trainer requirement" in metadata.stdout
    example_env = {**env, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    external_inputs = unrelated_repository / "external-example-inputs"
    _run(
        [
            str(installed_python),
            "-m",
            "aisimulate_core.sdk.gbdt_example",
            "generate",
            "--output-dir",
            str(external_inputs),
        ],
        cwd=unrelated_repository,
        env=example_env,
    )
    for action in ("verify-bundle", "run"):
        example = _run(
            [
                str(installed_python),
                "-m",
                "aisimulate_core.sdk.gbdt_example",
                action,
                "--output-dir",
                str(unrelated_repository / f"example-{action}"),
                *(["--input-dir", str(external_inputs)] if action == "run" else []),
            ],
            cwd=unrelated_repository,
            env=example_env,
        )
        result = json.loads(example.stdout)
        assert result["hardware_measured"] is False
        assert result["qualification"] == "functional_only_synthetic"
        assert Path(result["sdk_origin"]).resolve().is_relative_to(installed_purelib)
        assert Path(result["native"]["path"]).resolve().is_relative_to(installed_purelib)
        expected_check = (
            "explicit_learned_catalogue_discovery"
            if action == "verify-bundle"
            else "continued_feedback_matches_uninterrupted_bits"
        )
        assert expected_check in result["checks"]
        if action == "run":
            assert result["readiness"]["execution"]["ready"]
            assert result["feedback_observations"] == 40
            assert result["feedback_cursor"]["next_iteration"] == 20
            assert "seeded_b_receives_external_c_feedback" in result["checks"]
            assert "application_cursor_bound_to_feedback_and_snapshot" in result["checks"]
        else:
            assert all(report["execution"]["ready"] for report in result["readiness"])

    # The shipped seed is usable with the base runtime; only run/train needs
    # the optional offline trainer. Reject imports rather than reuse its presence.
    _run(
        [
            str(installed_python),
            "-c",
            textwrap.dedent(
                """
                import importlib.abc
                import runpy
                import sys

                class RejectTrainer(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path, target=None):
                        if fullname == "sklearn" or fullname.startswith("sklearn."):
                            raise AssertionError("Packaged verification imported the offline trainer")

                sys.meta_path.insert(0, RejectTrainer())
                sys.argv = ["gbdt_example", "verify-bundle", "--output-dir", "example-without-trainer"]
                runpy.run_module("aisimulate_core.sdk.gbdt_example", run_name="__main__")
                """
            ),
        ],
        cwd=unrelated_repository,
        env=example_env,
    )

    # A positive optimized-Python run could hide removed assertions. Deliberately
    # break only evaluation lineage and require the shipped verifier to reject it.
    damaged = unrelated_repository / "damaged-example"
    shutil.copytree(installed_purelib / "aisimulate_core/systems/fpm_learned/examples/synthetic-functional-v1", damaged)
    manifest = damaged / "bundle.json"
    declaration = json.loads(manifest.read_text())
    declaration["provenance"]["evaluation_sources"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(declaration))
    with pytest.raises(subprocess.CalledProcessError) as caught:
        _run(
            [
                str(installed_python),
                "-O",
                "-m",
                "aisimulate_core.sdk.gbdt_example",
                "verify-bundle",
                "--bundle-path",
                str(manifest),
                "--output-dir",
                str(unrelated_repository / "rejected-example"),
            ],
            cwd=unrelated_repository,
            env=example_env,
        )
    assert "evaluation lineage differs" in caught.value.stderr
