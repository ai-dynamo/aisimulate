# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen CPU policy and legacy resume tests, without a scheduler or GPU."""

import json
from dataclasses import replace

import pytest
from collector.fpm_forward import cli
from collector.fpm_forward.config import FPMCollectionOptions, resolve_slurm_cpu_policy
from collector.fpm_forward.repeatability import load_repeatability_source

from .test_fpm_profile_collection import _argv, _plan, _profile

pytestmark = pytest.mark.unit


def _options(*extra):
    return FPMCollectionOptions.from_args(
        cli._parser().parse_args(
            [
                *_argv(_profile()),
                "--fpm-executor",
                "slurm",
                "--fpm-slurm-container-image",
                "runtime.sqsh",
                *extra,
            ]
        )
    )


def _archive(tmp_path, options):
    plan = _plan(_profile(), options=options)
    root = tmp_path / "artifacts" / plan.sha256[:16]
    root.mkdir(parents=True)
    (root / "collection-plan.json").write_text(json.dumps(plan.to_dict()))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "fpm_forward.json").write_text(json.dumps({"plan_sha256": plan.sha256, "cells": {}}))
    return plan, root, checkpoint


def test_new_slurm_plan_defaults_and_explicit_cpu_policy_are_frozen(tmp_path):
    defaults = _options()
    assert (defaults.slurm_cpus_per_task, defaults.slurm_cpu_bind) == (16, "cores")
    explicit = _options("--fpm-slurm-cpus-per-task", "32", "--fpm-slurm-cpu-bind", "none")
    plan, root, checkpoint = _archive(tmp_path, explicit)
    assert load_repeatability_source(root).to_dict() == plan.to_dict()
    assert resolve_slurm_cpu_policy(None, None, resume=True, checkpoint_dir=checkpoint, artifact_root=root.parent) == (
        32,
        "none",
    )
    assert resolve_slurm_cpu_policy(8, "cores", resume=True, checkpoint_dir=checkpoint, artifact_root=root.parent) == (
        8,
        "cores",
    )


def test_legacy_slurm_plan_keeps_missing_policy_and_exact_bytes(tmp_path):
    options = replace(_options(), slurm_cpus_per_task=None, slurm_cpu_bind=None)
    plan, root, checkpoint = _archive(tmp_path, options)
    before = (root / "collection-plan.json").read_bytes()
    loaded = load_repeatability_source(root)
    assert "slurm_cpus_per_task" not in loaded.to_dict()["options"]
    assert loaded.sha256 == plan.sha256
    assert resolve_slurm_cpu_policy(None, None, resume=True, checkpoint_dir=checkpoint, artifact_root=root.parent) == (
        None,
        None,
    )
    assert (root / "collection-plan.json").read_bytes() == before


@pytest.mark.parametrize("provided", [(32, None), (None, "none"), (8, None), (None, "cores")])
def test_partial_collection_resume_inherits_only_omitted_cpu_fields(tmp_path, provided):
    from collector.fpm_forward import runner

    options = _options("--fpm-slurm-cpus-per-task", "32", "--fpm-slurm-cpu-bind", "none")
    plan, root, checkpoint = _archive(tmp_path, options)
    checkpoint_path = checkpoint / "fpm_forward.json"
    checkpoint_path.write_text(
        json.dumps({"schema": runner.CHECKPOINT_SCHEMA, "plan_sha256": plan.sha256, "cells": {}})
    )
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    cpus, binding = resolve_slurm_cpu_policy(
        *provided, resume=True, checkpoint_dir=checkpoint, artifact_root=root.parent
    )
    assert (cpus, binding) == (provided[0] if provided[0] is not None else 32, provided[1] or "none")
    resumed = _plan(_profile(), options=replace(options, slurm_cpus_per_task=cpus, slurm_cpu_bind=binding))
    if (cpus, binding) == (32, "none"):
        assert runner._load_checkpoint(checkpoint_path, resumed, True)["plan_sha256"] == plan.sha256
    else:
        with pytest.raises(ValueError, match="does not match the current frozen plan"):
            runner._load_checkpoint(checkpoint_path, resumed, True)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("provided", [(32, None), (None, "none")])
def test_partial_resume_never_invents_missing_legacy_policy(tmp_path, provided):
    options = replace(_options(), slurm_cpus_per_task=None, slurm_cpu_bind=None)
    _, root, checkpoint = _archive(tmp_path, options)
    cpus, binding = resolve_slurm_cpu_policy(
        *provided, resume=True, checkpoint_dir=checkpoint, artifact_root=root.parent
    )
    assert (cpus, binding) == provided
    with pytest.raises(ValueError, match="CPU policy"):
        replace(options, slurm_cpus_per_task=cpus, slurm_cpu_bind=binding)


def test_partial_resume_requires_untampered_saved_plan(tmp_path):
    _, root, checkpoint = _archive(tmp_path, _options())
    path = root / "collection-plan.json"
    payload = json.loads(path.read_text())
    payload["options"]["slurm_cpu_bind"] = "none"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA-256"):
        resolve_slurm_cpu_policy(16, None, resume=True, checkpoint_dir=checkpoint, artifact_root=root.parent)


@pytest.mark.parametrize(
    "updates",
    [
        {"slurm_cpus_per_task": True},
        {"slurm_cpus_per_task": 0},
        {"slurm_cpus_per_task": 1.5},
        {"slurm_cpu_bind": "mask_cpu:1"},
        {"slurm_cpu_bind": None},
        {"executor": "kubernetes"},
    ],
)
def test_invalid_or_partial_cpu_policy_is_rejected(updates):
    with pytest.raises(ValueError, match="CPU|cpu"):
        replace(_options(), **updates)


@pytest.mark.parametrize("flag,value", [("--fpm-slurm-cpus-per-task", "16"), ("--fpm-slurm-cpu-bind", "cores")])
def test_cpu_options_are_rejected_on_kubernetes(flag, value):
    with pytest.raises(ValueError, match="require --fpm-executor slurm"):
        FPMCollectionOptions.from_args(cli._parser().parse_args([*_argv(_profile()), flag, value]))


def test_legacy_worker_retry_rejected_before_retained_artifacts_are_removed(tmp_path, monkeypatch):
    from collector.fpm_forward import runner

    plan, root, checkpoint = _archive(
        tmp_path,
        replace(_options(), slurm_cpus_per_task=None, slurm_cpu_bind=None),
    )
    raw = root / "cells" / plan.cells[0].cell_id / "raw"
    raw.mkdir(parents=True)
    preserved = raw / "incomplete-native-evidence.json"
    preserved.write_text('{"diagnostic": "original failed attempt"}\n')
    payload = {
        "schema": runner.CHECKPOINT_SCHEMA,
        "plan_sha256": plan.sha256,
        "cells": {cell.cell_id: {"status": "failed", "attempt_id": "old-attempt"} for cell in plan.cells},
    }
    (checkpoint / "fpm_forward.json").write_text(json.dumps(payload))
    monkeypatch.setattr(runner, "_cell_runner", lambda *a: pytest.fail("legacy retry launched or cleaned GPU steps"))
    with pytest.raises(ValueError, match="no frozen CPU policy"):
        runner.run_collection(
            plan,
            generator_overrides={},
            checkpoint_dir=str(checkpoint),
            artifact_root=str(root.parent),
            resume=True,
            retry_failed=True,
            publish_database=False,
        )
    assert preserved.read_text() == '{"diagnostic": "original failed attempt"}\n'
    assert json.loads((checkpoint / "fpm_forward.json").read_text()) == payload
