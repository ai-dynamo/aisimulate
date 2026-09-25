# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY publication naming integrity, separate from native acceptance.

The complete v2 controller validator is exercised without mocks in
test_glm53flash_external_control_current. These tests isolate merging and
canonical revalidation after that boundary; the mocked trust seam is local to
pytest and cannot admit a production attachment.
"""

import copy
import shutil

import pyarrow.parquet as pq
import pytest

from tests.unit.tools import test_glm53flash_hf_publication as fixtures
from tools.glm53flash_hf import external_control, external_control_current
from tools.glm53flash_hf import glm53flash as policy
from tools.glm53flash_hf import import_glm53flash as integration

pytestmark = pytest.mark.unit


@pytest.fixture
def staged(tmp_path, request):
    return fixtures.staged.__wrapped__(tmp_path, request)


def controls(tmp_path, monkeypatch, part):
    """Stub only the independently tested native/external validator boundary."""
    identities = {
        "planner_source_commit": "1" * 40,
        "planner_wheel_sha256": "2" * 64,
        "native_producer_revision": "3" * 40,
        "native_producer_wheel_sha256": "4" * 64,
    }
    records = []
    for phase in ("prefill", "decode"):
        for role in ("calibration", "holdout"):
            path = tmp_path / f"{phase}-{role}.json"
            integration.write(path, {"schema": external_control_current.SCHEMA, "TEST_ONLY_identity": identities})
            records.append(
                dict(
                    **{k: part[k] for k in ("backend", "weight_quantization", "tp")},
                    phase=phase,
                    role=role,
                    external_control={"path": path.name, "sha256": policy.sha(path)},
                )
            )
    monkeypatch.setattr(external_control, "validate", lambda root, doc: ({}, None, doc))
    monkeypatch.setattr(external_control_current, "configuration_revisions", lambda doc, *_: doc["TEST_ONLY_identity"])
    return records, identities


def metadata(value="installed:aisimulate==0.13.0:record-sha256:" + "5" * 64):
    return dict(
        revision_identity_schema=policy.REVISION_SCHEMA,
        aic_revision=value,
        planner_revision=value,
        producer_revision=value,
        producer_revision_semantics=policy.PLANNER_ALIAS,
    )


def consumer():
    return dict(
        distribution="aisimulate",
        version="0.13.0",
        payload_sha256="6" * 64,
        api="RustForwardPassPerfModel.best_available",
    )


@pytest.mark.parametrize(
    "field",
    ["planner_source_commit", "planner_wheel_sha256", "native_producer_revision", "native_producer_wheel_sha256"],
)
def test_rehashed_mixed_phase_identity_is_rejected(tmp_path, monkeypatch, field):
    part = dict(backend="sglang", weight_quantization="fp8", tp=2)
    records, _ = controls(tmp_path, monkeypatch, part)
    ref = records[-1]["external_control"]
    path = tmp_path / ref["path"]
    doc = policy.read(path)
    doc["TEST_ONLY_identity"][field] = "9" * len(doc["TEST_ONLY_identity"][field])
    integration.write(path, doc)
    ref["sha256"] = policy.sha(path)
    with pytest.raises(ValueError, match="phase/role native producer or planner"):
        policy.publication_revisions(metadata(), {"consumer": consumer()}, records, tmp_path, "7" * 40, part)


@pytest.mark.parametrize(
    "field",
    [
        "aic_revision",
        "planner_revision",
        "producer_revision",
        "producer_revision_semantics",
        "revision_identity_schema",
    ],
)
def test_new_partition_aliases_bind_original_planner(field):
    original = {"aic_revision": "installed:original:record"}
    meta = metadata(original["aic_revision"])
    meta[field] = "changed"
    with pytest.raises(ValueError, match="planner revision"):
        policy.validate_planner_revision(meta, original)


def test_legacy_metadata_is_readable_without_invented_native_or_analysis_revision(tmp_path):
    meta = {"producer_revision": "1" * 40}
    assert policy.validate_planner_revision(meta, {"aic_revision": "1" * 40}) is False
    assert (
        policy.publication_revisions(
            meta, {}, [], tmp_path, "2" * 40, dict(backend="sglang", weight_quantization="fp8", tp=2)
        )
        is None
    )
    assert meta == {"producer_revision": "1" * 40}


def snapshot(staged, tmp_path, monkeypatch):
    root = staged[0]
    stage = policy.read(root / "stage.json")
    part = next(p for p in stage["configurations"] if p["backend"] == "sglang")
    meta_path = root / part["metadata"]["path"]
    meta = policy.read(meta_path)
    meta.update(metadata("1" * 40))
    integration.write(meta_path, meta)
    part["metadata"]["sha256"] = policy.sha(meta_path)
    report_path = root / stage["acceptance"]["path"]
    report = policy.read(report_path)
    report["consumer"] = consumer()
    integration.write(report_path, report)
    stage["acceptance"]["sha256"] = policy.sha(report_path)
    integration.write(root / "stage.json", stage)
    policy.validate_stage(root)  # Real full eight-partition/source revalidation.
    records, _ = controls(tmp_path, monkeypatch, part)
    integration.write(tmp_path / "external.json", records)
    monkeypatch.setattr(policy, "validate_external_receipts", lambda *_a, **_k: {})
    revisions = policy.publication_revisions(meta, report, records, tmp_path, "7" * 40, part)
    receipt = dict(
        policy=policy.POLICY,
        source_revision="7" * 40,
        stage={"path": str((root / "stage.json").relative_to(tmp_path)), "sha256": policy.sha(root / "stage.json")},
        external_raw_evidence={"path": "external.json", "sha256": policy.sha(tmp_path / "external.json")},
        revision_identity=revisions,
    )
    integration.write(tmp_path / "import.json", receipt)
    target = tmp_path / "canonical.parquet"
    shutil.copyfile(root / part["parquet"]["path"], target)
    integration.write(
        tmp_path / "canonical.metadata.json",
        dict(
            meta,
            import_policy=policy.POLICY,
            supporting_files=[{"path": "import.json", "sha256": policy.sha(tmp_path / "import.json")}],
        ),
    )
    row = pq.read_table(target).to_pylist()[0]
    manifest = dict(
        **{
            k: row[k]
            for k in (
                "system",
                "tp",
                "pp",
                "dp",
                "moe_tp",
                "moe_ep",
                "cp",
                "parallel_strategy",
                "weight_quantization",
                "kv_cache_dtype",
            )
        },
        model_id=part["model_id"],
        model_revision=part["model_revision"],
        framework=part["backend"],
        framework_version=part["backend_version"],
        parallelism=f"pure-tp{part['tp']}",
        aisim_commit="3" * 40,
        aisim_commit_status="recorded",
        aisim_commit_semantics="native_producer_revision",
        provenance=dict(
            source_campaign_id=policy.CAMPAIGN,
            source_revision="7" * 40,
            import_receipt="import.json",
            import_receipt_sha256=policy.sha(tmp_path / "import.json"),
            revision_identity=revisions,
            producer_revisions=["1" * 40],
            producer_revisions_semantics=policy.PLANNER_ALIAS,
            producer_identity_missing=False,
        ),
        fpm=[
            dict(
                path=target.name,
                metadata_path="canonical.metadata.json",
                sha256=part["parquet"]["sha256"],
                row_count=part["rows"],
            )
        ],
    )
    return manifest


def test_canonical_split_host_identity_rederives_and_keeps_original_metadata(staged, tmp_path, monkeypatch):
    manifest = snapshot(staged, tmp_path, monkeypatch)
    before = (tmp_path / "canonical.metadata.json").read_bytes()
    assert policy.validate_snapshot(tmp_path, manifest) == {"import.json"}
    assert (tmp_path / "canonical.metadata.json").read_bytes() == before
    identity = manifest["provenance"]["revision_identity"]
    assert identity["planner_source_commit"] == "1" * 40
    assert identity["native_producer_revision"] == manifest["aisim_commit"] == "3" * 40
    assert identity["analysis_revision"]["publication_tool_revision"] == "7" * 40


@pytest.mark.parametrize(
    "field",
    [
        "planner_source_commit",
        "native_producer_revision",
        "native_producer_wheel_sha256",
        "analysis_revision",
        "external_control_sha256",
    ],
)
def test_consistently_rehashed_canonical_revision_spoof_rejects(staged, tmp_path, monkeypatch, field):
    manifest = snapshot(staged, tmp_path, monkeypatch)
    changed = copy.deepcopy(manifest["provenance"]["revision_identity"])
    changed[field] = "TEST_ONLY_forgery"
    receipt = policy.read(tmp_path / "import.json")
    receipt["revision_identity"] = changed
    integration.write(tmp_path / "import.json", receipt)
    manifest["provenance"].update(revision_identity=changed, import_receipt_sha256=policy.sha(tmp_path / "import.json"))
    meta = policy.read(tmp_path / "canonical.metadata.json")
    meta["supporting_files"][0]["sha256"] = policy.sha(tmp_path / "import.json")
    integration.write(tmp_path / "canonical.metadata.json", meta)
    with pytest.raises(ValueError, match="canonical revision identity"):
        policy.validate_snapshot(tmp_path, manifest)
