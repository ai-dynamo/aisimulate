# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual rendered entrypoints from a Slurm-like working directory."""

from __future__ import annotations

import json
import os
import py_compile
import shutil
import subprocess
import sys
import zipfile

import pytest
from collector.fpm_forward import runner, runtime_probe
from collector.fpm_forward.runtime_instrumentation import freeze_instrumentation, load_instrumentation

from .test_fpm_runtime_probe import _inputs

pytestmark = pytest.mark.unit


def _staged(tmp_path):
    launch, manifest, _ = _inputs(tmp_path)
    package = tmp_path / "adapter"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "classes.py").write_text(
        "import fpm_memory_observer as helper\n"
        "from .constants import ENABLED\n"
        "class ObservedWorker:\n"
        "    def marker(self): return helper.MARKER if ENABLED else 'disabled'\n"
        "class ObservedInstrumentedScheduler(ObservedWorker): pass\n"
    )
    (package / "constants.py").write_text("ENABLED = True\n")
    (package / "other_runtime.py").write_text("raise RuntimeError('unused adapter must not be imported')\n")
    (tmp_path / "fpm_memory_observer.py").write_text("MARKER = 'custom frozen helper'\n")
    value = json.loads(manifest.read_text())
    value.update(
        files=[
            "adapter/__init__.py",
            "adapter/classes.py",
            "adapter/constants.py",
            "adapter/other_runtime.py",
            "fpm_memory_observer.py",
            "source-notes.txt",
        ],
        worker_class="adapter.classes.ObservedWorker",
        scheduler_class="adapter.classes.ObservedInstrumentedScheduler",
    )
    manifest.write_text(json.dumps(value))
    frozen = freeze_instrumentation(load_instrumentation(manifest), tmp_path / "frozen")
    plan = runtime_probe.build_runtime_probe_plan("tp4", launch, frozen)
    cell = plan.cells[0]
    workdir = tmp_path / "slurm-workdir"
    workdir.mkdir()
    context = runtime_probe.launch_context(plan, cell, configuration="tp4", attempt_id="test-attempt")
    for path in runtime_probe.stage_runtime_instrumentation(frozen, workdir, context):
        if path.parent != workdir:
            shutil.copy2(path, workdir / path.name)
    with zipfile.ZipFile(workdir / "runtime-instrumentation.zip") as archive:
        archive.extractall(workdir / "runtime-instrumentation")
    (workdir / "fpm_memory_observer.py").write_text("MARKER = 'wrong Slurm cwd helper'\n")
    args = runner._cell_generator_overrides(plan, cell, {})["params"]["agg"]["extra_cli_args"]
    classes = [args[args.index(flag) + 1] for flag in ("--worker-cls", "--scheduler-cls")]
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join([str(workdir / "runtime-instrumentation"), str(workdir)]),
        "AISIMULATE_RUNTIME_CONTEXT": str(workdir / "runtime-probe-context.json"),
        "AISIMULATE_RUNTIME_INSTRUMENTATION": str(workdir / "runtime-instrumentation" / "manifest.json"),
        "AISIMULATE_RUNTIME_OBSERVATION_DIR": str(workdir / "results"),
    }
    return workdir, env, classes, frozen


def _run(workdir, env, classes, prefix=""):
    script = prefix + "\nimport importlib, json\nclasses = " + repr(classes) + "\n"
    script += (
        "values = []\n"
        "for name in classes:\n"
        "    module, _, attr = name.rpartition('.')\n"
        "    cls = getattr(importlib.import_module(module), attr)\n"
        "    values.append(cls().marker())\n"
        "print(json.dumps(values))\n"
    )
    return subprocess.run([sys.executable, "-c", script], cwd=workdir, env=env, capture_output=True, text=True)


def test_rendered_worker_and_scheduler_use_frozen_helper_in_slurm_cwd(tmp_path):
    workdir, env, classes, frozen = _staged(tmp_path)
    result = _run(workdir, env, classes)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["custom frozen helper", "custom frozen helper"]
    receipts = list((workdir / "results").glob("runtime-instrumentation-imports-*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["status"] == "verified"
    assert receipt["bundle_sha256"] == frozen.sha256
    assert receipt["files"] == frozen.files
    assert "adapter.constants" in receipt["imports"]
    assert "adapter.other_runtime" not in receipt["imports"]
    helper = receipt["imports"]["fpm_memory_observer"]
    assert helper["path"] == str(workdir / "runtime-instrumentation" / "fpm_memory_observer.py")
    assert helper["sha256"] == frozen.files["fpm_memory_observer.py"]


@pytest.mark.parametrize("changed", ["fpm_memory_observer.py", "source-notes.txt"])
def test_runtime_bundle_tampering_fails_before_adapter_import(tmp_path, changed):
    workdir, env, classes, _ = _staged(tmp_path)
    with (workdir / "runtime-instrumentation" / changed).open("a") as handle:
        handle.write("\n# changed after staging\n")
    result = _run(workdir, env, classes)
    assert result.returncode != 0
    assert "bundle hash mismatch" in result.stderr
    receipt = json.loads(next((workdir / "results").glob("runtime-instrumentation-imports-*.json")).read_text())
    assert receipt["status"] == "failed"
    assert not receipt["imports"]


def test_wrong_helper_imported_before_runtime_entrypoint_is_rejected(tmp_path):
    workdir, env, classes, _ = _staged(tmp_path)
    result = _run(workdir, env, classes, "import fpm_memory_observer\n")
    assert result.returncode != 0
    assert "already imported outside verified bundle loading" in result.stderr


def test_verified_source_is_executed_instead_of_unchecked_cached_bytecode(tmp_path):
    workdir, env, classes, _ = _staged(tmp_path)
    helper = workdir / "runtime-instrumentation" / "fpm_memory_observer.py"
    original = helper.read_bytes()
    helper.write_text("MARKER = 'wrong cached helper'\n")
    py_compile.compile(str(helper), invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH, doraise=True)
    helper.write_bytes(original)
    result = _run(workdir, env, classes)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["custom frozen helper", "custom frozen helper"]


def test_spawned_worker_and_scheduler_unpickling_use_verified_entrypoint(tmp_path):
    workdir, env, classes, _ = _staged(tmp_path)
    script = workdir / "spawn_check.py"
    script.write_text(
        "import importlib, multiprocessing\n"
        "def child(cls, queue):\n"
        "    queue.put(cls().marker())\n"
        "if __name__ == '__main__':\n"
        "    context = multiprocessing.get_context('spawn')\n"
        f"    for name in {classes!r}:\n"
        "        module, _, attr = name.rpartition('.')\n"
        "        cls = getattr(importlib.import_module(module), attr)\n"
        "        original = importlib.import_module('adapter.classes')\n"
        "        assert isinstance(cls(), original.ObservedWorker)\n"
        "        queue = context.Queue()\n"
        "        process = context.Process(target=child, args=(cls, queue))\n"
        "        process.start()\n"
        "        assert queue.get(timeout=10) == 'custom frozen helper'\n"
        "        process.join(10)\n"
        "        assert process.exitcode == 0\n"
    )
    result = subprocess.run([sys.executable, str(script)], cwd=workdir, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    receipts = [json.loads(p.read_text()) for p in (workdir / "results").glob("runtime-instrumentation-imports-*.json")]
    assert len(receipts) == 3
    assert all(item["status"] == "verified" for item in receipts)
    assert len({item["pid"] for item in receipts}) == 3


def test_frozen_package_resources_use_standard_importlib_reader(tmp_path):
    workdir, env, classes, _ = _staged(tmp_path)
    root = workdir / "runtime-instrumentation"
    (root / "adapter/data.txt").write_text("frozen package data")
    (root / "adapter/classes.py").write_text(
        "from importlib.resources import files\n"
        "class ObservedWorker:\n"
        "    def marker(self): return files(__package__).joinpath('data.txt').read_text()\n"
        "class ObservedInstrumentedScheduler(ObservedWorker): pass\n"
    )
    manifest = root / "manifest.json"
    value = json.loads(manifest.read_text())
    value["files"].append("adapter/data.txt")
    manifest.write_text(json.dumps(value))
    frozen = load_instrumentation(manifest)
    context_path = workdir / "runtime-probe-context.json"
    context = json.loads(context_path.read_text())
    context["bundle_sha256"] = frozen.sha256
    context_path.write_text(json.dumps(context))
    result = _run(workdir, env, classes)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["frozen package data", "frozen package data"]
    receipt = json.loads(next((workdir / "results").glob("runtime-instrumentation-imports-*.json")).read_text())
    assert receipt["files"]["adapter/data.txt"] == frozen.files["adapter/data.txt"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires a fork-capable platform")
def test_forked_worker_preserves_imports_with_its_own_receipt(tmp_path):
    workdir, env, _classes, _ = _staged(tmp_path)
    script = (
        "import multiprocessing, os, json\n"
        "import fpm_runtime_instrumentation as binding\n"
        "cls = binding.ObservedWorker\n"
        "parent = binding.observed_binding()\n"
        "def child(queue):\n"
        "    queue.put({'actual_pid': os.getpid(), 'binding': binding.observed_binding(), 'value': cls().marker()})\n"
        "context = multiprocessing.get_context('fork')\n"
        "queue = context.Queue()\n"
        "process = context.Process(target=child, args=(queue,))\n"
        "process.start()\n"
        "child_result = queue.get(timeout=10)\n"
        "process.join(10)\n"
        "assert process.exitcode == 0\n"
        "print(json.dumps({'parent': parent, 'child': child_result}))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=workdir, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    parent, child = value["parent"], value["child"]
    assert child["binding"]["pid"] == child["actual_pid"] != parent["pid"]
    assert child["binding"]["imports"] == parent["imports"]
    assert child["value"] == "custom frozen helper"
    receipts = [
        json.loads(path.read_text()) for path in (workdir / "results").glob("runtime-instrumentation-imports-*.json")
    ]
    assert len(receipts) == 2
    assert next(item for item in receipts if item["pid"] == parent["pid"]) == parent
    assert next(item for item in receipts if item["pid"] == child["actual_pid"]) == child["binding"]


@pytest.mark.parametrize(
    "mutation", ["digest", "origin", "module-type", "classes-type", "missing-context", "pid", "hostname"]
)
def test_importer_requires_observed_binding_for_new_context_and_keeps_legacy_unverified(tmp_path, mutation):
    from collector.fpm_forward.runtime_instrumentation import runtime_binding
    from collector.fpm_forward.runtime_observations import validate_observations

    from .test_runtime_observations import _mutate, _write, observation_fixture

    path, launches = observation_fixture(tmp_path)
    old = validate_observations(path, launches)["tp2"]
    assert old["status"] == "complete"
    assert old["provenance"]["instrumentation"]["import_binding"] == "unverified_legacy"
    index = json.loads(path.read_text())
    attempt = index["configurations"]["tp2"]["attempts"][0]
    for phase in attempt["phases"].values():
        reference = phase["launch_manifest"]
        launch_path = path.parent / reference["path"]
        context = json.loads(launch_path.read_text())
        context["instrumentation_binding"] = runtime_binding()
        reference["sha256"] = _write(launch_path, context)["sha256"]
    path.write_text(json.dumps(index))
    missing = validate_observations(path, launches)["tp2"]
    assert missing["status"] == "incomplete"
    assert "verified instrumentation imports" in str(missing["diagnostics"])

    bundle = load_instrumentation(path.parent / attempt["bundle"]["manifest"])

    def add_receipt(record):
        # Synthetic observations exercise CPU validation only; runtime execution
        # is independently covered by the rendered-entrypoint subprocess tests.
        record["instrumentation_binding"] = {
            **runtime_binding(),
            **{key: record[key] for key in ("attempt_id", "configuration", "phase", "bundle_sha256")},
            "status": "verified",
            "expected_bundle_sha256": bundle.sha256,
            "hostname": "synthetic-host",
            "pid": 123,
            "root": "/frozen",
            "files": bundle.files,
            "imports": {
                "observer": {
                    "file": "observer.py",
                    "path": "/frozen/observer.py",
                    "sha256": bundle.files["observer.py"],
                }
            },
            "classes": {key: bundle.manifest[key] for key in ("worker_class", "scheduler_class")},
        }
        record["cpu_affinity_observation"] = {"status": "complete", "hostname": "synthetic-host", "pid": 123}

    _mutate(path, add_receipt)
    new = validate_observations(path, launches)["tp2"]
    assert new["status"] == "complete", new
    assert new["provenance"]["instrumentation"]["import_binding"] == "verified"
    if mutation == "missing-context":
        index = json.loads(path.read_text())
        for phase in index["configurations"]["tp2"]["attempts"][0]["phases"].values():
            reference = phase["launch_manifest"]
            launch_path = path.parent / reference["path"]
            context = json.loads(launch_path.read_text())
            context.pop("instrumentation_binding")
            reference["sha256"] = _write(launch_path, context)["sha256"]
        path.write_text(json.dumps(index))
    else:

        def change(record):
            evidence = record["instrumentation_binding"]
            if mutation == "digest":
                evidence["imports"]["observer"]["sha256"] = "c" * 64
            elif mutation == "origin":
                evidence["imports"]["observer"]["path"] = "/cwd/observer.py"
            elif mutation == "module-type":
                evidence["imports"]["observer"] = []
            elif mutation == "pid":
                evidence["pid"] = 124
            elif mutation == "hostname":
                evidence["hostname"] = "different-host"
            else:
                evidence["classes"] = []

        _mutate(path, change)
    changed = validate_observations(path, launches)["tp2"]
    assert changed["status"] == "incomplete"
    assert "instrumentation" in str(changed["diagnostics"])
