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
        "aisimulate/legacy_cli/main.py",
        "aisimulate/generator/api.py",
        "aisimulate/logging_utils.py",
        "aisimulate/sdk/_compat.py",
        "aisimulate/sdk/config_adapter/__init__.py",
        "aisimulate/sdk/config_adapter/schemas/estimate-request-v1.schema.json",
        "aisimulate/sdk/engine.py",
        "aisimulate/sdk/task_v2.py",
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

    assert "aisimulate/sdk/config_adapter/README.md" not in payload
    assert "aisimulate/sdk/config_adapter/schemas/estimate-request-v1.schema.json" in payload
    assert "collector/cases/base_ops/mla_module.yaml" in payload
    assert "collector/fpm_forward/runtime/fpm_exec.sh" in payload
    assert "collector/glm53flash_shard_contract.py" in payload
    assert "collector/glm53flash_jsonl.py" in payload
    assert "collector/glm53flash_sglang_retained.py" in payload
    assert "collector/collect_glm53flash.py" in payload

    assert "collector/glm53flash_tail_qualification.py" in payload
    tail = "collector/fpm_forward/runtime/glm53flash_vllm_tail_repair"
    assert f"{tail}/qualification/expected-runtime.json" in payload
    assert f"{tail}/packaged-verifier-executed-inputs.json" in payload
    for role in ("candidate", "reference"):
        for asset in (
            "actual-build-receipt.json",
            "expected-source-sha256.json",
            "expected-native-binaries.json",
            "combined-repair.patch.b64",
            "combined-repair.review.diff",
            "README.md",
            "LICENSE",
        ):
            assert f"{tail}/{role}/{asset}" in payload


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


@pytest.mark.parametrize("package", ["aiconfigurator", "aiconfigurator_core"])
def test_release_verifier_rejects_removed_import_packages(verifier, tmp_path, package):
    wheel = tmp_path / "aisimulate-1.2.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{package}/__init__.py", "")
        archive.writestr(
            "aisimulate-1.2.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: aisimulate\nVersion: 1.2.0\n",
        )
    with pytest.raises(RuntimeError, match="removed legacy import packages"):
        verifier._verify_wheel(wheel, set())
