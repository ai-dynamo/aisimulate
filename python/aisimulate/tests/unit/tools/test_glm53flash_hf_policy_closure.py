# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise copied dataset scripts independently of repository module caches."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from tools.glm53flash_hf import glm53flash as policy
from tools.glm53flash_hf import import_glm53flash as integration
from tools.glm53flash_hf import profile

pytestmark = pytest.mark.unit


@pytest.fixture
def copied_dataset(tmp_path):
    destination = tmp_path / "TEST_ONLY_DATASET"
    (destination / "scripts").mkdir(parents=True)
    integration.copy_policy(destination)
    return destination


def test_copied_policy_imports_in_isolated_process(copied_dataset):
    scripts = copied_dataset / "scripts"
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import importlib, json, pathlib, sys; "
            "sys.path.insert(0, sys.argv[1]); "
            "modules = [importlib.import_module(p.stem) for p in pathlib.Path(sys.argv[1]).glob('*.py')]; "
            "print(json.dumps({m.__name__: str(pathlib.Path(m.__file__).resolve()) for m in modules}))",
            str(scripts),
        ],
        cwd=copied_dataset,
        text=True,
        capture_output=True,
        check=True,
    )
    loaded = json.loads(process.stdout)
    assert set(loaded) == {
        "glm53flash",
        "raw_campaign",
        "native_roots",
        "raw_archive",
        "external_control",
        "external_control_vllm",
        "external_control_current",
        "external_control_sglang_mixed",
        "external_control_sglang_factory",
        "closed_history",
        "cleanup_reconciliation",
        "cleanup_executor",
        "accounting_termination",
        "portable_history",
        "import_glm53flash",
        "profile",
    }
    assert all(Path(path).parent == scripts for path in loaded.values())


def test_embedded_policy_keeps_dependency_closure_after_import_path_restored(copied_dataset):
    scripts = copied_dataset / "scripts"
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import json, pathlib, sys; "
            "sys.path.insert(0, sys.argv[1]); import glm53flash as p; sys.path.pop(0); "
            "assert sys.argv[1] not in sys.path; "
            "p.portable_history.h.dependencies(p.raw_campaign.archive); "
            "assert p.publication_revisions({'producer_revision': 'a'*40}, {}, [], pathlib.Path('.'), 'b'*40, "
            "dict(backend='sglang', weight_quantization='fp8', tp=2)) is None; "
            "print(json.dumps({n: str(pathlib.Path(sys.modules[n[:-3]].__file__).resolve()) "
            "for n in p.POLICY_MODULES if n not in {'import_glm53flash.py', 'profile.py', 'cleanup_executor.py'}}))",
            str(scripts),
        ],
        cwd=copied_dataset,
        text=True,
        capture_output=True,
        check=True,
    )
    assert all(Path(path).parent == scripts for path in json.loads(process.stdout).values())


def test_profile_binds_all_copied_policy_bytes(copied_dataset):
    controls = profile._policy_controls(copied_dataset)
    assert set(controls) == {
        "scripts/glm53flash.py",
        "scripts/raw_campaign.py",
        "scripts/native_roots.py",
        "scripts/raw_archive.py",
        "scripts/external_control.py",
        "scripts/external_control_vllm.py",
        "scripts/external_control_current.py",
        "scripts/external_control_sglang_mixed.py",
        "scripts/external_control_sglang_factory.py",
        "scripts/closed_history.py",
        "scripts/cleanup_reconciliation.py",
        "scripts/cleanup_executor.py",
        "scripts/accounting_termination.py",
        "scripts/portable_history.py",
        "scripts/import_glm53flash.py",
        "scripts/profile.py",
    }
    assert all(policy.sha(copied_dataset / path) == digest for path, digest in controls.items())


@pytest.mark.parametrize("module", list(policy.POLICY_MODULES))
@pytest.mark.parametrize("mutation", ["missing", "modified", "symlink"])
def test_unreviewed_external_control_rejected_before_loading_stage(copied_dataset, module, mutation):
    path = copied_dataset / "scripts" / module
    if mutation == "modified":
        path.write_bytes(path.read_bytes() + b"\n# Unreviewed replacement\n")
    else:
        path.unlink()
        if mutation == "symlink":
            path.symlink_to(Path(policy.__file__).with_name(module))
    with pytest.raises(ValueError, match="reviewed GLM policy: " + module):
        profile._load_import(copied_dataset / "MISSING_STAGE", copied_dataset, copied_dataset / "MISSING_IMPORT")
