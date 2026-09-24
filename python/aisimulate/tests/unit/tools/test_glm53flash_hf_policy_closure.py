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
        "raw_archive",
        "external_control",
        "external_control_vllm",
    }
    assert all(Path(path).parent == scripts for path in loaded.values())


def test_profile_binds_all_copied_policy_bytes(copied_dataset):
    controls = profile._policy_controls(copied_dataset)
    assert set(controls) == {
        "scripts/glm53flash.py",
        "scripts/raw_campaign.py",
        "scripts/raw_archive.py",
        "scripts/external_control.py",
        "scripts/external_control_vllm.py",
    }
    assert all(policy.sha(copied_dataset / path) == digest for path, digest in controls.items())


@pytest.mark.parametrize("module", ["external_control.py", "external_control_vllm.py"])
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
