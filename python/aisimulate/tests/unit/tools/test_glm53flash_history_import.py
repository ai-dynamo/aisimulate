# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY canonical routing tests; no actual campaign acceptance or Hub calls."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from tools.glm53flash_hf import glm53flash as policy
from tools.glm53flash_hf import import_glm53flash as integration

pytestmark = pytest.mark.unit
CANDIDATE = Path(policy.__file__).resolve().parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": path.name, "sha256": policy.sha(path)}


def current_records(tmp_path):
    return [{"external_control": write(tmp_path / "control.json", {"schema": "glm53flash_external_control_v3"})}]


def test_current_controls_cannot_lose_both_policy_and_proof(tmp_path):
    records = current_records(tmp_path)
    with pytest.raises(ValueError, match="requires immutable attempt history"):
        policy.validate_attempt_history({"policy": policy.POLICY}, {}, records, tmp_path, "a" * 64)


def test_genuine_legacy_without_current_controls_keeps_original_contract(tmp_path, monkeypatch):
    records = [{"external_control": write(tmp_path / "control.json", {"schema": "glm53flash_external_control_v1"})}]
    monkeypatch.setattr(
        policy.portable_history, "verify_portable_history", lambda *a, **k: pytest.fail("legacy invoked new gate")
    )
    assert policy.validate_attempt_history({"policy": policy.POLICY}, {}, records, tmp_path, "a" * 64) == {}
    with pytest.raises(ValueError, match="silently ignore"):
        policy.validate_attempt_history(
            {"policy": policy.POLICY, "external_raw_history": {}}, {}, records, tmp_path, "a" * 64
        )


def test_current_offline_dispatch_requires_same_explicit_provenance(tmp_path, monkeypatch):
    entry = {"contract": "TEST_ONLY_contract", "proof": {"path": "TEST_ONLY_proof"}}
    calls = []

    def verifier(root, supplied, records, *, expected_stage_sha256, archive):
        calls.append((root, supplied, expected_stage_sha256, archive))
        return {"files": {"history/TEST_ONLY_proof": "b" * 64}}

    monkeypatch.setattr(policy.portable_history, "verify_portable_history", verifier)
    receipt = {"policy": policy.HISTORY_POLICY, "external_raw_history": entry}
    records = current_records(tmp_path)
    with pytest.raises(ValueError, match="provenance differs"):
        policy.validate_attempt_history(receipt, {}, records, tmp_path, "a" * 64)
    assert not calls
    assert policy.validate_attempt_history(receipt, {"external_raw_history": entry}, records, tmp_path, "a" * 64) == {
        "history/TEST_ONLY_proof": "b" * 64
    }
    assert len(calls) == 1 and calls[0][2] == "a" * 64


def test_current_import_requires_full_archive_gate_before_destination(tmp_path, monkeypatch):
    # Outer accepted stage validation is stubbed only to isolate the new import
    # ordering. This fixture is not an acceptance-stage positive.
    stage = tmp_path / "TEST_ONLY_stage"
    stage.mkdir()
    write(stage / "stage.json", {"TEST_ONLY": True})
    acceptance = write(stage / "acceptance.json", {"TEST_ONLY": True})
    records = current_records(tmp_path)
    external = tmp_path / "external.json"
    write(external, records)
    monkeypatch.setattr(policy, "validate_stage", lambda root: {"acceptance": acceptance})
    monkeypatch.setattr(policy, "validate_external_receipts", lambda *a, **k: {})
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="requires raw archive history"):
        integration.prepare(tmp_path / "absent-base", stage, output, external, "1" * 40, "2026-09-24")
    assert not output.exists()

    def reject_full_archive(*args, **kwargs):
        raise ValueError("TEST_ONLY original tar changed")

    monkeypatch.setattr(policy.portable_history, "prepare_portable_history", reject_full_archive)
    with pytest.raises(ValueError, match="original tar changed"):
        integration.prepare(
            tmp_path / "absent-base",
            stage,
            output,
            external,
            "1" * 40,
            "2026-09-24",
            history={"contract": policy.portable_history.h.CONTRACT, "proof": {}},
            bundles={},
        )
    assert not output.exists()


def test_complete_distributed_source_closure_is_copied(tmp_path):
    (tmp_path / "scripts").mkdir()
    integration.copy_policy(tmp_path)
    assert {
        "closed_history.py",
        "native_roots.py",
        "cleanup_reconciliation.py",
        "cleanup_executor.py",
        "accounting_termination.py",
        "portable_history.py",
        "external_control_sglang_mixed.py",
        "external_control_sglang_factory.py",
        "profile.py",
        "import_glm53flash.py",
    } <= set(policy.POLICY_MODULES)
    for name in policy.POLICY_MODULES:
        assert (tmp_path / "scripts" / name).read_bytes() == (CANDIDATE / name).read_bytes()
    cold = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys;sys.path.insert(0,sys.argv[1]);import glm53flash as p;"
            "p.portable_history.h.dependencies(p.raw_campaign.archive);"
            "assert len(p.POLICY_MODULES)==16;print('COLD_COMPLETE_POLICY_PASS')",
            str(tmp_path / "scripts"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert cold.stdout.strip() == "COLD_COMPLETE_POLICY_PASS"
