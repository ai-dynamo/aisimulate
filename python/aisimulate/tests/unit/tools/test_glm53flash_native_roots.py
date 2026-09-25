# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY collection/pod bridge; no original GPU data or acceptance claims."""

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.fpm_forward.native_artifact import _validate_collector_provenance
from tools.glm53flash_hf import native_roots as roots
from tools.glm53flash_hf import raw_archive as archive
from tools.glm53flash_hf import raw_campaign as campaign

from . import test_glm53flash_hf_publication as publication_fixture
from . import test_glm53flash_portable_history as portable_fixture
from . import test_glm53flash_raw_campaign as campaign_fixture

pytestmark = pytest.mark.unit


def test_public_collector_boundary_preserves_plan_attempt_and_pod(tmp_path):
    cid = "TEST_ONLY_child"
    collection = tmp_path / "cells" / cid / "raw"
    pod = collection / "node0000"
    pod.mkdir(parents=True)
    original = {
        "schema_name": "aic_fpm_collector_provenance",
        "schema_version": 1,
        "cell_id": cid,
        "plan_sha256": "TEST_ONLY_plan",
        "attempt_id": "TEST_ONLY_attempt",
        "runtime": {"backend": "vllm", "backend_version": "0.30.0"},
    }
    path = pod / "collector-provenance.json"
    path.write_text(json.dumps(original))
    before = path.read_bytes()
    spec = {"cell_id": cid, "raw_root": str(collection), roots.FIELD: roots.SCOPE}
    actual_collection, actual_pod = roots.collection(spec, tmp_path)
    cell = SimpleNamespace(cell_id=cid, backend="vllm", state_protocol="glm53flash_same_request_real_hybrid_v1")
    ranks = [(pod / "benchmark.json", {"producer": {"vllm_package_version": "0.30.0"}})]

    def validate(root, plan="TEST_ONLY_plan", attempt="TEST_ONLY_attempt"):
        return _validate_collector_provenance(
            cell, root, ranks, expected_plan_sha256=plan, expected_attempt_id=attempt, expected_backend_version="0.30.0"
        )

    with pytest.raises(ValueError, match="not scoped"):
        validate(actual_pod)
    assert validate(actual_collection) == ("0.30.0", "TEST_ONLY_attempt")
    for kwargs in ({"plan": "wrong"}, {"attempt": "wrong"}):
        with pytest.raises(ValueError, match="mismatch"):
            validate(actual_collection, **kwargs)
    assert actual_pod == pod and path.read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        {"native_root_scope": None},
        {"native_root_scope": "unknown"},
        {"raw_root": "cells/c/raw"},
        {"raw_root": "/cells/foreign/raw"},
        {"raw_root": "/cells/c/raw/node0000"},
        {"raw_root": "/cells/c/../raw"},
        {"original_pod_root": "/cells/c/raw/node0001"},
    ],
)
def test_new_scope_cannot_be_inferred_or_redirected(change):
    spec = {"cell_id": "c", "raw_root": "/cells/c/raw", roots.FIELD: roots.SCOPE, **change}
    with pytest.raises(ValueError):
        roots.collection(spec, "/")


def test_mixed_or_unmarked_pod_mapping_rejects():
    with pytest.raises(ValueError, match="mixed"):
        roots.uniform([{}, {roots.FIELD: roots.SCOPE}])
    with pytest.raises(ValueError, match="explicit"):
        roots.scope({"original_pod_root": "/cells/c/raw/node0000"})


@pytest.fixture
def collection_history():
    fixture = portable_fixture.PortableTests()
    fixture.setUp()
    try:
        fixture.enable_collection_scope()
        yield fixture
    finally:
        fixture.doCleanups()


def test_collection_history_survives_offline_without_original_or_tar(collection_history):
    fixture = collection_history
    fixture.prepare()
    original_attempts = {b: copy.deepcopy(v["attempts"]) for b, v in fixture.ledgers.items()}
    shutil.rmtree(fixture.source)
    shutil.rmtree(fixture.root / "bound")
    for bundle in set(fixture.bundles.values()):
        shutil.rmtree(bundle)
    result = fixture.verify()
    assert result["state"] == "PORTABLE_METADATA_HISTORY_PASS_NO_FRESH_TAR_VERIFICATION"
    assert result["labels"] == 32
    assert {b: v["attempts"] for b, v in fixture.ledgers.items()} == original_attempts
    assert all(any(a["terminal_state"] == "COLLECTION_FAILED_PRESERVED" for a in v) for v in original_attempts.values())


@pytest.mark.parametrize("mutation", ["extra", "missing", "symlink"])
def test_closed_archive_rejects_extra_missing_or_symlink_pod(collection_history, mutation):
    fixture = collection_history
    collection = Path(fixture.plan["jobs"][0]["accepted_raw_roots"][0])
    if mutation == "extra":
        (collection / "node0001").mkdir()
    elif mutation == "missing":
        shutil.rmtree(collection / "node0000")
    else:
        (collection / "node0001").symlink_to(collection / "node0000", target_is_directory=True)
    with pytest.raises(ValueError):
        fixture.make_archive()


@pytest.mark.parametrize("mutation", ["scope", "pod", "attempt", "raw", "unmarked"])
def test_portable_mapping_cannot_change_after_actual_tar_gate(collection_history, mutation):
    fixture = collection_history
    fixture.prepare()
    record = fixture.records[0]["native_roots"][0]
    if mutation == "scope":
        record[roots.FIELD] = "unknown"
    elif mutation == "pod":
        record["original_pod_root"] = str(Path(record["raw_root"]) / "node0001")
    elif mutation == "attempt":
        record["attempt_id"] = "foreign"
    elif mutation == "raw":
        record["raw_root"] = record["original_pod_root"]
    else:
        record.pop(roots.FIELD)
    with pytest.raises(ValueError):
        fixture.verify()


def test_real_archive_binding_propagates_collection_scope_and_rejects_empty_foreign_pod(tmp_path, monkeypatch):
    stage, _ = publication_fixture.staged.__wrapped__(tmp_path, SimpleNamespace(param="fp8"))
    # The complete72 current external-control attachment is tested separately;
    # this TEST_ONLY fixture isolates real archive and accepted-file binding.
    monkeypatch.setattr(campaign, "external_controls", lambda *a, **k: {})
    plan, _bundles, output, records = campaign_fixture.build_bound_fixture(
        stage, tmp_path, monkeypatch, collection_scope=True
    )
    assert len(records) == 32
    assert all(roots.scope(job) == roots.SCOPE for job in plan["jobs"])
    assert sum(len(job["accepted_native_roots"]) for job in plan["jobs"]) == 33
    campaign.validate(records, stage, output)
    for record in records:
        assert all(roots.scope(item) == roots.SCOPE for item in record["native_roots"])
    record = records[0]
    inventory = output / record["source_inventory"]["path"]
    rows = list(archive.inventory_records(inventory))
    prefix = record["native_roots"][0]["archive_prefix"]
    extra = next(copy.deepcopy(r) for r in rows if r["path"] == prefix)
    extra["path"] = prefix + "/node0001"
    last_child = max(i for i, row in enumerate(rows) if row["path"] == prefix or row["path"].startswith(prefix + "/"))
    rows.insert(last_child + 1, extra)
    inventory.write_text("".join(json.dumps(r) + "\n" for r in rows))
    record["source_inventory"]["sha256"] = archive.sha_file(inventory)
    record["source_inventory"]["bytes"] = inventory.stat().st_size
    # Invoke the real metadata binding to isolate the empty-directory predicate;
    # no attestation is forged or treated as an accepted archive.
    with pytest.raises(ValueError, match="extra pod"):
        campaign._bindings(stage, campaign.context(stage), record, output)


def test_collection_scope_requires_original_external_controls():
    spec = {"cell_id": "c", "raw_root": "/cells/c/raw", roots.FIELD: roots.SCOPE}
    with pytest.raises(ValueError, match="requires original current"):
        campaign.external_controls({}, {}, Path("/"), [(spec, {"receipts": []})], {}, Path("/"))


def test_scope_on_shard_parent_is_not_silently_ignored():
    spec = {"shards": [{"cell_id": "c"}], roots.FIELD: roots.SCOPE}
    with pytest.raises(ValueError, match="each child"):
        campaign.native_pairs(spec, {"shards": []})
