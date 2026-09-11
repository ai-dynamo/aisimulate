# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

VERIFY_RELEASE_WHEELS = Path(__file__).resolve().parents[3] / "tools" / "verify_release_wheels.py"


@pytest.fixture
def verifier():
    spec = importlib.util.spec_from_file_location("verify_release_wheels", VERIFY_RELEASE_WHEELS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spica_scan_covers_all_archive_member_types(verifier):
    names = {
        "aiconfigurator/__init__.py",
        "spica/",
        "spica/native.so",
        "spica/data/model.bin",
    }

    assert verifier._spica_entries(names) == [
        "spica/",
        "spica/data/model.bin",
        "spica/native.so",
    ]


def test_release_verifier_rejects_stale_spica_archive_member(verifier, monkeypatch, tmp_path):
    wheel = tmp_path / "aisimulate-1.2.0-py3-none-any.whl"
    required = {
        "aiconfigurator/__init__.py",
        "aiconfigurator/cli/main.py",
        "aiconfigurator/generator/api.py",
        "aiconfigurator/logging_utils.py",
        "aiconfigurator/sdk/_compat.py",
        "aiconfigurator/sdk/config_adapter/__init__.py",
        "aiconfigurator/sdk/config_adapter/schemas/estimate-request-v1.schema.json",
        "aiconfigurator/sdk/engine.py",
        "aiconfigurator/sdk/task_v2.py",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in required:
            archive.writestr(name, "")
        archive.writestr(
            "aisimulate-1.2.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: aisimulate\nVersion: 1.2.0\n",
        )
        archive.writestr("spica/data/model.bin", b"stale")

    monkeypatch.setattr(verifier, "_source_payloads", set)
    monkeypatch.setattr(sys, "argv", ["verify_release_wheels.py", str(tmp_path)])

    with pytest.raises(RuntimeError, match=r"removed Spica payload.*spica/data/model\.bin"):
        verifier.main()


def test_infra_scan_rejects_gap_skill_tool_dataset_report_and_web_payloads(verifier):
    names = {
        ".agents/skills/adapt-server-config/SKILL.md",
        "aiconfigurator/datasets/private.json",
        "aiconfigurator/gap_analysis/pipeline.py",
        "aiconfigurator/reports/gap.html",
        "aiconfigurator/skills/helper.py",
        "aiconfigurator/tools/import.py",
        "datasets/source.csv",
        "reports/generated.json",
        "tools/private_helper.py",
        "vendor/datasets/source.csv",
        "vendor/reports/generated.json",
        "vendor/web/dashboard.js",
        "vendor/webapp/dashboard.js",
        "webapp/dashboard.js",
    }

    assert verifier._infra_entries(names) == sorted(names)


def test_config_adapter_readme_remains_repository_only(verifier):
    payload = verifier._source_payloads()

    assert "aiconfigurator/sdk/config_adapter/README.md" not in payload
    assert "aiconfigurator/sdk/config_adapter/schemas/estimate-request-v1.schema.json" in payload
    assert "collector/cases/base_ops/mla_module.yaml" in payload
    assert "collector/fpm_forward/runtime/fpm_exec.sh" in payload


def test_release_verifier_checks_packaged_legal_files(verifier, monkeypatch, tmp_path):
    wheel = tmp_path / "aisimulate-1.2.0-py3-none-any.whl"
    for name in verifier.LEGAL_FILES:
        (tmp_path / name).write_text(f"canonical {name}\n")
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in verifier.LEGAL_FILES:
            archive.writestr(
                f"aisimulate-1.2.0.dist-info/licenses/{name}",
                (tmp_path / name).read_bytes(),
            )

    monkeypatch.setattr(verifier, "PACKAGE_ROOT", tmp_path)
    verifier._verify_legal_files(wheel)


def test_release_verifier_rejects_missing_legal_file(verifier, monkeypatch, tmp_path):
    wheel = tmp_path / "aisimulate-1.2.0-py3-none-any.whl"
    (tmp_path / "LICENSE").write_text("canonical license\n")
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "aisimulate-1.2.0.dist-info/licenses/LICENSE",
            (tmp_path / "LICENSE").read_bytes(),
        )

    monkeypatch.setattr(verifier, "PACKAGE_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="expected one packaged THIRD_PARTY_NOTICES.md"):
        verifier._verify_legal_files(wheel)


def test_rust_crate_package_rejects_upper_payload(verifier, monkeypatch):
    result = subprocess.CompletedProcess(
        args=["cargo"],
        returncode=0,
        stdout="Cargo.toml\nsrc/lib.rs\naiconfigurator/sdk/config_adapter/api.py\n",
        stderr="",
    )
    monkeypatch.setattr(verifier.subprocess, "run", lambda *args, **kwargs: result)

    with pytest.raises(RuntimeError, match="config_adapter"):
        verifier._verify_rust_crate_package()


@pytest.mark.parametrize("root", ["datasets", "reports"])
def test_rust_crate_package_rejects_infra_roots(verifier, monkeypatch, root):
    result = subprocess.CompletedProcess(
        args=["cargo"],
        returncode=0,
        stdout=f"Cargo.toml\nsrc/lib.rs\n{root}/private.json\n",
        stderr="",
    )
    monkeypatch.setattr(verifier.subprocess, "run", lambda *args, **kwargs: result)

    with pytest.raises(RuntimeError, match=root):
        verifier._verify_rust_crate_package()
