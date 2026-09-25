# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY new import branch: real tar/history, mocked outer native/dataset transport.

The original legacy canonical-manager integration is tested separately. This case
never claims compatible production native/stage acceptance or publication.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.glm53flash_hf import glm53flash as policy
from tools.glm53flash_hf import import_glm53flash as integration

from . import test_glm53flash_hf_publication as publication_fixture
from . import test_glm53flash_portable_history as portable_fixture

pytestmark = pytest.mark.unit

archive, h = portable_fixture.archive, portable_fixture.h


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.mark.parametrize("collection_scope", [False, True])
def test_import_copies_portable_provenance_and_offline_snapshot_never_reads_tar(
    tmp_path, monkeypatch, collection_scope
):
    stage, _ = publication_fixture.staged.__wrapped__(tmp_path, SimpleNamespace(param="fp8"))
    fixture = portable_fixture.PortableTests()
    fixture.setUp()
    try:
        if collection_scope:
            fixture.enable_collection_scope()
        fixture.stage = stage
        fixture.plan["stage_sha256"] = policy.sha(stage / "stage.json")
        bundles = fixture.both_bundles()
        bound, _ = fixture.bind(bundles)
        history_ref = h.reference(bound / h.BOUND_PROOF)
        history_ref["path"] = h.BOUND_PROOF
        history = {"contract": h.CONTRACT, "proof": history_ref}
        base = tmp_path / "TEST_ONLY_base"
        (base / "scripts").mkdir(parents=True)
        (base / "scripts/manage_dataset.py").write_text("# TEST_ONLY outer transport\n")
        index = {"configuration_manifests": [], "history_manifests": [], "source": {"commit_time": "TEST_ONLY"}}
        write(base / "catalog/index.json", index)
        write(base / "catalog/fpm.json", {"records": []})
        write(base / "catalog/measurements.json", {"records": []})
        monkeypatch.setattr(integration, "MANAGER_SHA256", policy.sha(base / "scripts/manage_dataset.py"))
        monkeypatch.setattr(integration, "route_policy", lambda path: None)
        # The stage's synthetic points do not purport to be this fixture's72
        # native children. Keep that unrelated acceptance validator explicitly
        # mocked; the actual full-tar and portable history predicates stay real.
        monkeypatch.setattr(policy, "validate_external_receipts", lambda *a, **k: {})

        class TestOnlyManager:
            @staticmethod
            def schema_fingerprint(path):
                return "TEST_ONLY_outer_schema_fingerprint"

            @staticmethod
            def configuration_path(identity):
                return "TEST_ONLY_configs/" + "-".join(
                    str(identity[k]) for k in ("framework", "weight_quantization", "tp")
                )

            @staticmethod
            def write_catalogs(root, fpm, measurements, manifests, historical, commit_time):
                write(root / "catalog/index.json", dict(index, configuration_manifests=manifests))
                write(root / "catalog/fpm.json", {"records": fpm})
                write(root / "catalog/measurements.json", {"records": measurements})

            @staticmethod
            def validate_dataset(root, write_report=False):
                manifests = policy.read(root / "catalog/index.json")["configuration_manifests"]
                for name in manifests:
                    policy.validate_snapshot(root, policy.read(root / name))
                return {"TEST_ONLY_manifests": len(manifests), "native_acceptance": "NOT_EVALUATED"}

        monkeypatch.setattr(integration, "load_manager", lambda root: TestOnlyManager())
        output = tmp_path / "TEST_ONLY_canonical"
        actual_tar_checks = []
        original_verify = archive.verify_bundle

        def verify_original(bundle):
            actual_tar_checks.append(Path(bundle))
            return original_verify(bundle)

        monkeypatch.setattr(archive, "verify_bundle", verify_original)
        result = integration.prepare(
            base,
            stage,
            output,
            bound / "external-raw-evidence.json",
            "1" * 40,
            "2026-09-24",
            history=history,
            bundles=bundles,
        )
        assert result["policy"] == policy.HISTORY_POLICY
        assert len(result["added_manifests"]) == 8
        assert len(actual_tar_checks) == 2
        for bundle in set(bundles.values()):
            shutil.rmtree(bundle)
        shutil.rmtree(fixture.source)
        shutil.rmtree(bound)

        def no_tar(*args, **kwargs):
            raise AssertionError("Offline canonical validation must not touch tar")

        monkeypatch.setattr(archive, "verify_bundle", no_tar)
        monkeypatch.setattr(archive, "verify_archive", no_tar)
        for name in result["added_manifests"]:
            manifest = policy.read(output / name)
            receipt = policy.read(output / manifest["provenance"]["import_receipt"])
            entry = receipt["external_raw_history"]
            assert entry == manifest["provenance"]["external_raw_history"]
            assert entry["proof"]["path"].startswith("history/")
            paths = policy.validate_snapshot(output, manifest)
            assert any("/history/portable-history.json" in p for p in paths)
            assert all((output / p).is_file() for p in paths)
            metadata = policy.read(output / manifest["fpm"][0]["metadata_path"])
            assert paths <= {r["path"] for r in metadata["supporting_files"]}
        # A valid-looking policy downgrade cannot hide the copied proof.
        manifest = policy.read(output / result["added_manifests"][0])
        receipt_path = output / manifest["provenance"]["import_receipt"]
        receipt = policy.read(receipt_path)
        receipt["policy"] = policy.POLICY
        write(receipt_path, receipt)
        manifest["provenance"]["import_receipt_sha256"] = policy.sha(receipt_path)
        try:
            policy.validate_snapshot(output, manifest)
        except ValueError as error:
            assert "silently ignore" in str(error)
        else:
            raise AssertionError("Legacy downgrade must not ignore an explicit history proof")
    finally:
        fixture.doCleanups()
