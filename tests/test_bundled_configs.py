# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for copied configurations omitted from dependency metadata."""

from __future__ import annotations

import csv
import io
import json
import zipfile

import pytest

from scripts.ci import check_bundled_configs as checker

pytestmark = pytest.mark.unit
REVISION = "1" * 40
CONFIG = "aisimulate_core/model_configs/example_config.json"


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "source"
    path = root / "python/aisimulate/src" / CONFIG
    path.parent.mkdir(parents=True)
    path.write_text('{"model_type": "example"}\n')
    license_text = "Example custom license. Copyright Example Owner.\n"
    (root / "LICENSE").write_text("Original project license text.\n")
    (root / "THIRD_PARTY_NOTICES.md").write_text(
        "## Example configuration\n\n"
        f"example_config.json from https://example.com/model/blob/{REVISION}/config.json\n"
        f"```text\n{license_text}```\n"
    )
    entry = dict(
        path=CONFIG,
        sha256=checker.digest(path.read_bytes()),
        origin="upstream",
        license="LicenseRef-Example",
        copyright_owner="Example Owner",
        modified=False,
        source=dict(
            url=f"https://example.com/model/blob/{REVISION}/config.json", revision=REVISION, path="config.json"
        ),
        notice=dict(
            file="THIRD_PARTY_NOTICES.md",
            section="Example configuration",
            license_text_sha256=checker.text_digest(license_text),
        ),
    )
    inventory = root / "scripts/ci/bundled_configs.json"
    inventory.parent.mkdir(parents=True)

    def save(entries=None):
        inventory.write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    licenses={"LicenseRef-Example": {"license_text_sha256": [checker.text_digest(license_text)]}},
                    files=entries if entries is not None else [entry],
                )
            )
        )

    save()
    return root, path, entry, inventory, save


def wheel_file(root, path, name="test.whl", extra=None):
    wheel = root.parent / name
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr(CONFIG, path.read_bytes())
        for name in ["LICENSE", "THIRD_PARTY_NOTICES.md"]:
            z.writestr(f"aisimulate-1.dist-info/licenses/{name}", (root / name).read_bytes())
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return wheel


def test_custom_licensed_config_is_in_csv_and_provenance(fixture, tmp_path):
    root, path, entry, inventory, _ = fixture
    wheel = wheel_file(root, path)
    entries, wheel_sha = checker.validate(root, inventory, wheel_path=wheel)
    checker.write_reports(tmp_path / "report", entries, wheel_sha)
    rows = list(csv.DictReader(io.StringIO((tmp_path / "report/bundled-deps.csv").read_text())))
    assert rows == [
        dict(
            dependency_type="bundled_model_config",
            name=CONFIG,
            version="sha256:" + entry["sha256"],
            spdx_license="LicenseRef-Example",
        )
    ]
    report = json.loads((tmp_path / "report/bundled-provenance.json").read_text())
    assert report["wheel_sha256"] == checker.digest(wheel.read_bytes())
    assert report["license_approval_asserted"] is False
    assert report["files"][0]["source"]["revision"] == REVISION


@pytest.mark.parametrize("location", ["aisimulate_core/model_configs/new.json", "aisimulate/unexpected/copy.json"])
def test_new_config_requires_an_inventory_entry(fixture, location):
    root, _, _, inventory, _ = fixture
    path = root / "python/aisimulate/src" / location
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"architectures": ["ExampleModel"]}')
    with pytest.raises(checker.InventoryError, match="no provenance entry"):
        checker.validate(root, inventory)


def test_existing_config_cannot_change_without_provenance_update(fixture):
    root, path, _, inventory, _ = fixture
    path.write_text('{"model_type": "changed"}')
    with pytest.raises(checker.InventoryError, match="bytes changed"):
        checker.validate(root, inventory)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", None, "upstream source"),
        ("license", "UNKNOWN", "identified license"),
        ("copyright_owner", "", "copyright_owner"),
        ("modified", None, "modification status"),
        ("origin", "approved", "origin classification"),
        ("notice", None, "notice reference"),
    ],
)
def test_incomplete_origin_or_license_metadata_fails(fixture, field, value, message):
    root, _, entry, inventory, save = fixture
    entry[field] = value
    save()
    with pytest.raises(checker.InventoryError, match=message):
        checker.validate(root, inventory)


@pytest.mark.parametrize("revision", ["main", "v1.0", "", "1" * 39])
def test_upstream_reference_must_be_immutable(fixture, revision):
    root, _, entry, inventory, save = fixture
    entry["source"]["revision"] = revision
    save()
    with pytest.raises(checker.InventoryError, match="immutable full commit"):
        checker.validate(root, inventory)


def test_modified_upstream_config_requires_description(fixture):
    root, _, entry, inventory, save = fixture
    entry["modified"] = True
    save()
    with pytest.raises(checker.InventoryError, match="modifications"):
        checker.validate(root, inventory)


def test_notice_must_identify_the_file_and_revision(fixture):
    root, _, _, inventory, _ = fixture
    notice = root / "THIRD_PARTY_NOTICES.md"
    notice.write_text(notice.read_text().replace(REVISION, "2" * 40))
    with pytest.raises(checker.InventoryError, match="upstream revision"):
        checker.validate(root, inventory)


def test_truncated_license_text_fails(fixture):
    root, _, _, inventory, _ = fixture
    notice = root / "THIRD_PARTY_NOTICES.md"
    notice.write_text(notice.read_text().replace("Copyright Example Owner.", ""))
    with pytest.raises(checker.InventoryError, match="complete license text"):
        checker.validate(root, inventory)


def test_custom_config_cannot_be_mislabeled_as_a_permissive_license(fixture):
    root, _, entry, inventory, save = fixture
    entry["license"] = "MIT"
    save()
    with pytest.raises(checker.InventoryError, match="unregistered license identifier"):
        checker.validate(root, inventory)
    contents = json.loads(inventory.read_text())
    contents["licenses"]["MIT"] = {"license_text_sha256": ["f" * 64]}
    inventory.write_text(json.dumps(contents))
    with pytest.raises(checker.InventoryError, match="does not match the registered legal text"):
        checker.validate(root, inventory)


def test_license_copy_does_not_substitute_for_required_attribution(fixture):
    root, _, entry, inventory, save = fixture
    entry["notice"]["required_attribution"] = ["Copyright Example Owner."]
    save()
    # This phrase already exists inside the agreement; a separate required
    # attribution notice must still be present in the surrounding prose.
    with pytest.raises(checker.InventoryError, match="required attribution outside"):
        checker.validate(root, inventory)
    notice = root / "THIRD_PARTY_NOTICES.md"
    notice.write_text(notice.read_text() + "\nCopyright Example Owner.\n")
    checker.validate(root, inventory)


def test_legacy_unknown_is_reported_without_license_approval(fixture, monkeypatch, tmp_path):
    root, path, entry, inventory, save = fixture
    entry.update(origin="legacy_unresolved", license="UNKNOWN", unresolved_reason="Existing provenance debt")
    save()
    monkeypatch.setattr(checker, "legacy_baseline", lambda: {CONFIG: checker.digest(path.read_bytes())})
    entries, wheel_sha = checker.validate(root, inventory)
    checker.write_reports(tmp_path / "report", entries, wheel_sha)
    rows = list(csv.DictReader(io.StringIO((tmp_path / "report/bundled-deps.csv").read_text())))
    assert rows[0]["spdx_license"] == "UNKNOWN"


def test_new_unknown_cannot_expand_the_baseline(fixture):
    root, _, entry, inventory, save = fixture
    entry.update(origin="legacy_unresolved", license="UNKNOWN", unresolved_reason="Pretend existing")
    save()
    with pytest.raises(checker.InventoryError, match="cannot use the unresolved baseline"):
        checker.validate(root, inventory)


def test_updated_hash_cannot_hide_changed_legacy_bytes(fixture, monkeypatch):
    root, path, entry, inventory, save = fixture
    before = checker.digest(path.read_bytes())
    monkeypatch.setattr(checker, "legacy_baseline", lambda: {CONFIG: before})
    path.write_text('{"model_type": "changed"}')
    entry.update(
        sha256=checker.digest(path.read_bytes()),
        origin="legacy_unresolved",
        license="UNKNOWN",
        unresolved_reason="Old file, new bytes",
    )
    save()
    with pytest.raises(checker.InventoryError, match="cannot use the unresolved baseline"):
        checker.validate(root, inventory)


def test_frozen_baseline_integrity_cannot_be_changed(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"files": {}}')
    monkeypatch.setattr(checker, "BASELINE", baseline)
    with pytest.raises(checker.InventoryError, match="frozen unresolved baseline changed"):
        checker.legacy_baseline()


def test_wheel_only_copied_config_is_caught(fixture):
    root, path, _, inventory, _ = fixture
    wheel = wheel_file(root, path, extra={"another/package/config.json": '{"model_type":"copied"}'})
    with pytest.raises(checker.InventoryError, match="unexpected=.*another/package/config.json"):
        checker.validate(root, inventory, wheel_path=wheel)


def test_wheel_bytes_must_match_the_declared_source(fixture):
    root, path, _, inventory, _ = fixture
    wheel = root.parent / "different.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr(CONFIG, '{"model_type":"replaced during packaging"}')
    with pytest.raises(checker.InventoryError, match="bytes changed"):
        checker.validate(root, inventory, wheel_path=wheel)


def test_wheel_cannot_drop_required_license_text(fixture):
    root, path, _, inventory, _ = fixture
    wheel = root.parent / "no-legal.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr(CONFIG, path.read_bytes())
    with pytest.raises(checker.InventoryError, match="packaged THIRD_PARTY_NOTICES.md"):
        checker.validate(root, inventory, wheel_path=wheel)


def test_rename_without_a_new_inventory_entry_fails(fixture):
    root, path, _, inventory, _ = fixture
    path.rename(path.with_name("renamed.json"))
    with pytest.raises(checker.InventoryError, match="no provenance entry"):
        checker.validate(root, inventory)


def test_historical_subset_requires_matching_hash_and_notice(fixture):
    root, path, entry, inventory, save = fixture
    stale = {**entry, "path": "aisimulate_core/model_configs/newer_config.json"}
    save([entry, stale])
    with pytest.raises(checker.InventoryError, match="missing source"):
        checker.validate(root, inventory)
    entries, _ = checker.validate(root, inventory, wheel_path=wheel_file(root, path), allow_subset=True)
    assert len(entries) == 1
    path.write_text('{"model_type":"older but unrecorded"}')
    with pytest.raises(checker.InventoryError, match="bytes changed"):
        checker.validate(root, inventory, allow_subset=True)


def test_duplicate_inventory_paths_and_json_keys_fail(fixture):
    root, _, entry, inventory, save = fixture
    save([entry, entry])
    with pytest.raises(checker.InventoryError, match="duplicate inventory path"):
        checker.validate(root, inventory)
    inventory.write_text('{"schema_version":1,"files":[],"files":[]}')
    with pytest.raises(checker.InventoryError, match="duplicate JSON key"):
        checker.validate(root, inventory)


@pytest.mark.parametrize("path", ["/absolute.json", "../escape.json", "a/../b.json", "a\\b.json", "a//b.json"])
def test_paths_are_unambiguous_and_relative(path):
    with pytest.raises(checker.InventoryError, match="invalid relative path"):
        checker.relative_path(path)


def test_current_inventory_is_complete_and_does_not_relabel_unknowns():
    entries, _ = checker.validate(checker.ROOT, checker.ROOT / checker.INVENTORY)
    for entry in entries:
        if entry["origin"] == "legacy_unresolved":
            assert entry["license"] == "UNKNOWN"
    qwen = [e for e in entries if "Qwen3.8" in e["path"]]
    llama = [e for e in entries if "Llama-4-" in e["path"]]
    assert len(qwen) == len(llama) == 2
    assert {e["license"] for e in qwen} == {"LicenseRef-Qwen3.8-Max"}
    assert {e["license"] for e in llama} == {"LicenseRef-Llama-4-Community"}


def test_content_changes_change_osrb_identity(fixture, tmp_path):
    _, _, entry, _, _ = fixture
    checker.write_reports(tmp_path / "before", [entry], None)
    checker.write_reports(tmp_path / "after", [{**entry, "sha256": "f" * 64}], None)
    before = list(csv.DictReader(io.StringIO((tmp_path / "before/bundled-deps.csv").read_text())))[0]
    after = list(csv.DictReader(io.StringIO((tmp_path / "after/bundled-deps.csv").read_text())))[0]
    assert before["name"] == after["name"]
    assert before["version"] != after["version"]
