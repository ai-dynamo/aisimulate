# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY tiny archives exercise binding; none are formal or publishable data."""

import copy
import sys
from pathlib import Path

import pytest

from tools.glm53flash_hf import glm53flash as policy
from tools.glm53flash_hf import import_glm53flash as integration
from tools.glm53flash_hf import raw_archive, raw_campaign

pytestmark = pytest.mark.unit


def hydrate_native_fixture(stage_root, work):
    """Test-local synthetic report/plan/raw setup, including one sharded role."""
    work.mkdir()
    stage = policy.read(stage_root / "stage.json")
    report = policy.read(stage_root / stage["acceptance"]["path"])
    original = policy.read(stage_root / stage["input_manifest"]["path"])
    entries = []
    for source in stage["sources"]:
        source.setdefault("original_consumer_paths", ["original/" + Path(source["path"]).name])
    for cell_index, cell in enumerate(report["cells"]):
        part = next(
            p
            for p in stage["configurations"]
            if (p["backend"], p["weight_quantization"], p["tp"])
            == (cell["backend"], cell["weight_quantization"], cell["tp"])
        )
        if "prediction_provenance" not in cell:
            cell["prediction_provenance"] = {
                "data_receipts": [
                    {"path": s["original_consumer_paths"][0], "sha256": s["sha256"]}
                    for s in stage["sources"]
                    if s["sha256"]
                    in (part["source_partition"]["parquet_sha256"], part["source_partition"]["metadata_sha256"])
                ]
            }
        prediction = cell["prediction_provenance"]
        entry = {"consumer_data": prediction["data_receipts"], "consumer_config": prediction.get("config", {})}
        for role in ("calibration", "holdout"):
            label = {**{k: cell[k] for k in raw_campaign.FIELDS[:-1]}, "role": role}
            label_name = raw_campaign.name(label)
            closed = work / "campaigns" / label_name
            closed.mkdir(parents=True)
            (closed / "failed-attempt-1.log").write_text("TEST_ONLY failure preserved\n")
            plan = {
                "backend": cell["backend"],
                "options": {"dataset_role": role},
                "sha256": "a" * 64,
                "cells": [
                    {
                        "cell_id": label_name,
                        "topology": {"tp": cell["tp"]},
                        "weight_quantization": cell["weight_quantization"],
                        "workload_kind": cell["phase"],
                    }
                ],
            }
            plan_path = work / "plans" / (label_name + ".json")
            integration.write(plan_path, plan)
            spec = {
                "cell_id": label_name,
                "plan": {"path": plan_path.relative_to(work).as_posix(), "sha256": policy.sha(plan_path)},
            }
            evidence = {"backend_version": part["backend_version"], "test_fixture": True}
            children, native_children = [], []
            sharded = cell_index == 0 and role == "calibration"
            for shard in range(2 if sharded else 1):
                raw = closed / ("accepted-" + str(shard))
                raw.mkdir()
                native_file = raw / "benchmark.json"
                native_file.write_text("TEST_ONLY " + label_name + str(shard))
                child = copy.deepcopy(spec)
                if sharded:
                    child["cell_id"] += "-shard" + str(shard)
                    child_plan = copy.deepcopy(plan)
                    child_plan["cells"][0]["cell_id"] = child["cell_id"]
                    child_path = work / "plans" / (child["cell_id"] + ".json")
                    integration.write(child_path, child_plan)
                    child["plan"] = {"path": child_path.relative_to(work).as_posix(), "sha256": policy.sha(child_path)}
                child.update(raw_root=raw.relative_to(work).as_posix(), attempt_id="TEST_ONLY_attempt")
                native = {
                    "backend_version": part["backend_version"],
                    "receipts": [{"path": native_file.name, "sha256": policy.sha(native_file)}],
                    "runtime_run_id": "TEST_ONLY_run",
                    "runtime_grid_digest": "a" * 64,
                }
                children.append(child)
                native_children.append(dict(native, child_cell_id=child["cell_id"], source_plan_sha256="a" * 64))
            if sharded:
                shard_path = work / "plans" / (label_name + "-shards.json")
                integration.write(shard_path, {"test_only": True, "children": [s["cell_id"] for s in children]})
                spec.update(
                    shards=children,
                    shard_manifest={"path": shard_path.relative_to(work).as_posix(), "sha256": policy.sha(shard_path)},
                )
                evidence["shards"] = native_children
            else:
                spec = children[0]
                evidence.update(
                    {k: v for k, v in native_children[0].items() if k not in ("child_cell_id", "source_plan_sha256")}
                )
            entry[role] = spec
            cell[role + "_evidence"] = evidence
        entries.append(entry)
    original["entries"] = entries
    original["test_only"] = True
    integration.write(stage_root / stage["input_manifest"]["path"], original)
    report["input_manifest_sha256"] = raw_campaign.digest(original)
    integration.write(stage_root / stage["acceptance"]["path"], report)
    stage["input_manifest"]["sha256"] = policy.sha(stage_root / stage["input_manifest"]["path"])
    stage["input_manifest_sha256"] = stage["input_manifest"]["sha256"]
    stage["acceptance"]["sha256"] = policy.sha(stage_root / stage["acceptance"]["path"])
    integration.write(stage_root / "stage.json", stage)


def build_bound_fixture(stage_root, base, monkeypatch):
    hydrate_native_fixture(stage_root, base / "TEST_ONLY_NATIVE")
    # The real command has no bypass. Test-only monkeypatch is process-local.
    monkeypatch.setattr(raw_campaign, "production", lambda _: None)
    monkeypatch.setitem(sys.modules, "raw_campaign", raw_campaign)
    monkeypatch.setitem(sys.modules, "raw_archive", raw_archive)
    plan = raw_campaign.make_plan(stage_root, base / "TEST_ONLY_NATIVE")
    for job in plan["jobs"]:
        job["source_root"] = str(base / "TEST_ONLY_NATIVE/campaigns" / raw_campaign.name(job))
        job["uri"] = "ssh://ocijhb/lustre/TEST_ONLY/" + raw_campaign.name(job) + "/campaign.tar.gz"
    archive_root = base / "TEST_ONLY_ARCHIVES"
    archive_root.mkdir()
    bundles = {}
    for job in plan["jobs"]:
        name = raw_campaign.name(job)
        output = archive_root / name
        raw_campaign.archive_one(stage_root, plan, name, output)
        bundles[name] = str(output)
    destination = base / "TEST_ONLY_BOUND_EVIDENCE"
    records = raw_campaign.bind(stage_root, plan, bundles, destination)
    return plan, bundles, destination, records


@pytest.fixture
def bound(tmp_path, request, monkeypatch):
    from tests.unit.tools.test_glm53flash_hf_publication import staged

    stage_root, _ = staged.__wrapped__(tmp_path, request)
    return stage_root, *build_bound_fixture(stage_root, tmp_path, monkeypatch)


def test_full32_archives_preserve_failed_files_and_exact_stage_consumer_lineage(bound):
    stage, plan, bundles, root, records = bound
    files = policy.validate_external_receipts(records, stage_root=stage, evidence_root=root)
    assert len(records) == 32 and len(files) >= 128
    assert sum(len(r["native_roots"]) for r in records) == 33
    for record in records:
        inventory = list(raw_archive.inventory_records(root / record["source_inventory"]["path"]))
        assert any(item["path"] == "failed-attempt-1.log" for item in inventory)
        assert record["consumer_sources"]
        assert (Path(bundles[raw_campaign.name(record)]) / "campaign.tar.gz").exists()
    assert not list(root.rglob("*.tar.gz"))  # Big archives never enter small import bundle.
    assert all(job["source_root"] for job in plan["jobs"])


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "wrong_stage",
        "role",
        "consumer_origin",
        "consumer_sha",
        "raw_prefix",
        "missing_shard",
        "archive_sha",
        "uri",
        "no_attestation",
    ],
)
def test_misbound_external_evidence_is_rejected(bound, mutation):
    stage, _, _, root, original = bound
    records = copy.deepcopy(original)
    record = records[0]
    if mutation == "duplicate":
        records.append(copy.deepcopy(record))
    elif mutation == "missing":
        records.pop()
    elif mutation == "wrong_stage":
        record["stage_sha256"] = "0" * 64
    elif mutation == "role":
        record["role"] = "holdout"
    elif mutation.startswith("consumer"):
        record["consumer_sources"][0]["original_path" if mutation == "consumer_origin" else "sha256"] = "bad"
    elif mutation == "raw_prefix":
        record["native_roots"][0]["archive_prefix"] = "failed-attempts"
    elif mutation == "missing_shard":
        next(r for r in records if len(r["native_roots"]) == 2)["native_roots"].pop()
    elif mutation == "archive_sha":
        record["sha256"] = "0" * 64
    elif mutation == "uri":
        record["uri"] += "?token=TEST_ONLY"
    else:
        record.pop("archive_receipt")
    with pytest.raises((ValueError, KeyError)):
        policy.validate_external_receipts(records, stage_root=stage, evidence_root=root)


def test_labels_only_low_level_receipt_is_not_formal_evidence(bound):
    stage, _, bundles, root, _ = bound
    records = []
    for bundle in bundles.values():
        records.extend(policy.read(Path(bundle) / "external-raw-evidence.json"))
    with pytest.raises(ValueError, match="unbound raw evidence stage"):
        policy.validate_external_receipts(records, stage_root=stage, evidence_root=root)


def test_inventory_tamper_rejected_even_if_all_local_receipts_are_rehashed(bound):
    stage, _, _, root, records = bound
    record = records[0]
    path = root / record["source_inventory"]["path"]
    inventory = list(raw_archive.inventory_records(path))
    next(i for i in inventory if i["kind"] == "file" and i["path"].endswith("benchmark.json"))["sha256"] = "0" * 64
    path.write_text("".join(raw_archive.canonical(item) + "\n" for item in inventory))
    record["source_inventory"].update(sha256=policy.sha(path), bytes=path.stat().st_size)
    attestation_path = root / record["archive_receipt"]["path"]
    attestation = policy.read(attestation_path)
    attestation["source_inventory"].update({k: record["source_inventory"][k] for k in ("sha256", "bytes")})
    input_path = root / record["archive_input_manifest"]["path"]
    original = policy.read(input_path)
    original["source_inventory"] = attestation["source_inventory"]
    integration.write(input_path, original)
    record["archive_input_manifest"].update(sha256=policy.sha(input_path), bytes=input_path.stat().st_size)
    attestation["input_manifest"].update({k: record["archive_input_manifest"][k] for k in ("sha256", "bytes")})
    integration.write(attestation_path, attestation)
    record["archive_receipt"].update(sha256=policy.sha(attestation_path), bytes=attestation_path.stat().st_size)
    with pytest.raises(ValueError, match="native file set/SHA"):
        policy.validate_external_receipts(records, stage_root=stage, evidence_root=root)


def test_bind_reextracts_archive_and_preserves_failed_outputs(bound, tmp_path):
    stage, plan, bundles, _, _ = bound
    bundle = Path(next(iter(bundles.values())))
    with (bundle / "campaign.tar.gz").open("ab") as stream:
        stream.write(b"broken")
    output = tmp_path / "TEST_ONLY_FAILED_BIND"
    with pytest.raises(ValueError, match="SHA256"):
        raw_campaign.bind(stage, plan, bundles, output)
    assert policy.read(output / "failure.json")["status"] == "FAILED"
    assert (bundle / "campaign.tar.gz").exists()


def test_production_entry_rejects_fixture_before_archiving(tmp_path, request):
    from tests.unit.tools.test_glm53flash_hf_publication import staged

    stage, _ = staged.__wrapped__(tmp_path, request)
    with pytest.raises(ValueError, match="synthetic/test/diagnostic"):
        raw_campaign.make_plan(stage, tmp_path)


def test_all_source_roots_are_explicit_and_must_enclose_accepted_data(bound, tmp_path):
    stage, plan, _, _, _ = bound
    invalid = copy.deepcopy(plan)
    invalid["jobs"][0]["source_root"] = str(tmp_path / "absent")
    with pytest.raises((ValueError, FileNotFoundError)):
        raw_campaign.archive_one(stage, invalid, raw_campaign.name(plan["jobs"][0]), tmp_path / "uncreated")
    assert not (tmp_path / "uncreated").exists()


def test_archive_cannot_write_inside_another_roles_source(bound):
    stage, plan, _, _, _ = bound
    first, other = plan["jobs"][:2]
    output = Path(other["source_root"]) / "uncreated"
    with pytest.raises(ValueError, match="inside campaign/stage input"):
        raw_campaign.archive_one(stage, plan, raw_campaign.name(first), output)
    assert not output.exists()
