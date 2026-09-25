# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY real filesystem aliases; no remote, GPU, or acceptance claims."""

import copy
import hashlib
import json
import shutil
import sys

import pytest

from tools.glm53flash_hf import external_control_current as current
from tools.glm53flash_hf import raw_archive as archive
from tools.glm53flash_hf import raw_campaign as campaign

pytestmark = pytest.mark.unit
LABEL = [{"backend": "vllm", "weight_quantization": "fp8", "tp": 2, "phase": "prefill", "role": "calibration"}]
URI = "ssh://ocijhb/scratch/TEST_ONLY/campaign.tar.gz"


@pytest.fixture
def roots(tmp_path):
    scratch = tmp_path / "scratch"
    canonical = scratch / "projects/user/task"
    canonical.mkdir(parents=True)
    (scratch / "users").mkdir()
    (scratch / "users/user").symlink_to("../projects/user", target_is_directory=True)
    alias = tmp_path / "lustre"
    alias.symlink_to(scratch, target_is_directory=True)
    original = alias / "users/user/task"
    source = canonical / "campaign"
    source.mkdir()
    (source / "failed.log").write_text("TEST_ONLY original failed attempt\n")
    (source / "accepted.json").write_text('{"TEST_ONLY":true}\n')
    return original, canonical, archive.create_storage_binding(original, canonical)


def test_two_level_alias_archive_and_offline_closed_replay(roots, tmp_path):
    original, canonical, proof = roots
    assert len(proof["proof"]["aliases"]) == 2
    output = tmp_path / "bundle"
    result = archive.create_archive(original / "campaign", output, URI, LABEL, storage_root_binding=proof)
    document = json.loads((output / archive.INPUT_MANIFEST).read_bytes())
    assert document["source_path"] == str(canonical / "campaign")
    assert document["original_source_path"] == str(original / "campaign")
    assert document["storage_root_binding"] == result["storage_root_binding"] == proof
    assert json.loads((output / "receipt.json").read_bytes())["source_recheck"] == "STAT_AND_SHA256_PASS"
    # Offline verification uses embedded inventory/proof, never host live inode claims.
    shutil.rmtree(canonical)
    assert archive.verify_bundle(output) == result["verification"]
    assert archive.storage_path(original / "campaign/failed.log", proof) == canonical / "campaign/failed.log"


@pytest.mark.parametrize("defect", ["wrong_root", "wrong_inode", "digest", "escape", "outside", "missing_proof"])
def test_root_or_suffix_forgery_rejected(roots, tmp_path, defect):
    original, canonical, proof = roots
    if defect in {"wrong_root", "wrong_inode", "digest"}:
        changed = copy.deepcopy(proof)
        if defect == "wrong_root":
            other = tmp_path / "other"
            other.mkdir()
            changed["proof"]["canonical_root"] = str(other)
        elif defect == "wrong_inode":
            changed["proof"]["inode"] += 1
        else:
            changed["sha256"] = "0" * 64
        if defect != "digest":
            changed["sha256"] = hashlib.sha256(archive.canonical(changed["proof"]).encode()).hexdigest()
        with pytest.raises(ValueError):
            archive.validate_storage_binding(changed, live=True)
    else:
        path = (
            str(original) + "/../campaign"
            if defect == "escape"
            else tmp_path / "outside"
            if defect == "outside"
            else original / "campaign"
        )
        with pytest.raises(ValueError):
            archive.storage_path(path, None if defect == "missing_proof" else proof, live=True)


def test_member_symlink_does_not_inherit_root_permission(roots, tmp_path):
    original, canonical, proof = roots
    (canonical / "campaign/link").symlink_to(canonical / "campaign/accepted.json")
    with pytest.raises(ValueError):
        archive.create_archive(original / "campaign", tmp_path / "bundle", URI, LABEL, storage_root_binding=proof)
    with pytest.raises(ValueError):
        archive.storage_path(original / "campaign/link", proof, live=True)


def test_alias_retarget_during_archive_preserves_failure(roots, tmp_path, monkeypatch):
    original, canonical, proof = roots
    old = archive.write_archive

    def retarget(fd, output):
        old(fd, output)
        alias = tmp_path / "lustre"
        alias.unlink()
        alias.symlink_to(tmp_path / "missing", target_is_directory=True)

    monkeypatch.setattr(archive, "write_archive", retarget)
    output = tmp_path / "bundle"
    with pytest.raises((ValueError, OSError)):
        archive.create_archive(original / "campaign", output, URI, LABEL, storage_root_binding=proof)
    assert (output / "failure.json").is_file() and not (output / "receipt.json").exists()
    assert (canonical / "campaign/failed.log").is_file()


def test_proof_removal_from_receipt_rejected_offline(roots, tmp_path):
    original, _, proof = roots
    output = tmp_path / "bundle"
    archive.create_archive(original / "campaign", output, URI, LABEL, storage_root_binding=proof)
    p = output / "receipt.json"
    value = json.loads(p.read_bytes())
    value.pop("storage_root_binding")
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="storage proof"):
        archive.verify_bundle(output)


def test_current_control_keeps_original_run_path_and_uses_storage_for_inventory(roots, monkeypatch):
    original, canonical, proof = roots
    cid = "TEST_ONLY_child"
    native = original / "prepared/native/cell"
    child = dict(native_directory=str(native), child_plan_sha256="a" * 64, role="calibration", phase="prefill")
    plan_path = "prepared/plans/TEST_ONLY_child.json"
    plan = dict(
        backend="sglang",
        sha256="a" * 64,
        options={"dataset_role": "calibration"},
        cells=[dict(cell_id=cid, workload_kind="prefill")],
    )
    run = dict(
        cell_id=cid,
        raw_root=str(original / "campaign/raw"),
        started="campaign/started.json",
        collector_provenance="campaign/raw/collector-provenance.json",
        host_wheel_verification="campaign/host.json",
        producer_wheel_verification="campaign/producer.json",
    )
    document = dict(
        schema=current.SCHEMA,
        adapter=current.SGLANG,
        original_task_root=str(original),
        runs=[run],
        files=[
            {"original_path": p, "sha256": "b" * 64}
            for p in (
                run["started"],
                run["collector_provenance"],
                run["host_wheel_verification"],
                run["producer_wheel_verification"],
            )
        ],
    )
    context = dict(admission={}, backend="sglang", children={cid: child}, files={plan_path: "c" * 64})
    monkeypatch.setattr(current, "frozen_contract", lambda *_: context)

    def get(name):
        return json.dumps(plan if name == plan_path else {"attempt_id": "TEST_ONLY_attempt"}).encode()

    spec = dict(cell_id=cid, raw_root=run["raw_root"], attempt_id="TEST_ONLY_attempt", plan={"path": plan_path})
    evidence = {"receipts": [{"path": "collector-provenance.json", "sha256": "b" * 64}]}
    inventory = [
        {"kind": "file", "path": item["original_path"], "sha256": item["sha256"]} for item in document["files"]
    ]
    args = (document, get, {}, [(spec, evidence)], {plan_path: plan}, str(original), inventory, canonical)
    current.bind_role(*args, storage_root_binding=proof)
    with pytest.raises(ValueError, match="archive omits"):
        current.bind_role(*args)
    changed = dict(spec, raw_root=str(canonical / "campaign/raw"))
    with pytest.raises(ValueError, match="plan/raw identity"):
        current.bind_role(
            document,
            get,
            {},
            [(changed, evidence)],
            {plan_path: plan},
            str(original),
            inventory,
            canonical,
            storage_root_binding=proof,
        )


def test_complete32_campaign_binding_preserves_original_alias_and_failed_files(roots, tmp_path, request, monkeypatch):
    from tests.unit.tools.test_glm53flash_hf_publication import staged
    from tests.unit.tools.test_glm53flash_raw_campaign import hydrate_native_fixture

    original, canonical, proof = roots
    stage, _ = staged.__wrapped__(tmp_path, request)
    work = canonical / "analysis"
    hydrate_native_fixture(stage, work)
    monkeypatch.setattr(campaign, "production", lambda _: None)  # TEST_ONLY receipts only.
    monkeypatch.setitem(sys.modules, "raw_campaign", campaign)
    monkeypatch.setitem(sys.modules, "raw_archive", archive)
    # Relative original metadata is interpreted through an explicit original base.
    plan = campaign.make_plan(stage, original / "analysis", storage_root_binding=proof)
    plan["manifest_base"] = str(original / "analysis")
    for job in plan["jobs"]:
        job["source_root"] = str(original / "analysis/campaigns" / campaign.name(job))
        job["uri"] = URI.replace("campaign.tar.gz", campaign.name(job) + ".tar.gz")
    bundles = {}
    for job in plan["jobs"]:
        label = campaign.name(job)
        output = tmp_path / ("bundle-" + label)
        campaign.archive_one(stage, plan, label, output)
        bundles[label] = str(output)
    bound = tmp_path / "bound"
    records = campaign.bind(stage, plan, bundles, bound)
    assert len(records) == 32 and all(r["manifest_base"] == str(original / "analysis") for r in records)
    shutil.rmtree(canonical)
    assert campaign.validate(records, stage, bound)
