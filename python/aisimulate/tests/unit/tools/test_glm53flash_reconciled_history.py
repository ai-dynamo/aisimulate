# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY: real tiny tar + offline proof, mocked original stage acceptance."""

import copy
import json
import shutil
import unittest
from pathlib import Path

import pytest

from . import test_glm53flash_portable_history as base
from .test_glm53flash_cleanup_reconciliation import Fixture, c, ref

pytestmark = pytest.mark.unit


class ReconciledHistoryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = base.PortableTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.b = self.fixture

    def reconcile_one(self, backend="vllm", *, accounting=False):
        ledger = self.b.ledgers[backend]
        old = ledger["attempts"][0]
        factory = Fixture
        if accounting:
            from .test_glm53flash_accounting_termination import AccountingFixture

            factory = AccountingFixture
        f = factory(
            self.b.source,
            backend=backend,
            job=old["job"],
            cid=old["cell_id"],
            directory=old["original_attempt_directory"],
        )
        output = self.b.source / ("reconciled-" + old["job"])
        proof = f.run(output)
        original = proof["original"]
        old.update(
            terminal_state="COLLECTION_FAILED_PRESERVED",
            **{k: original["references"][k] for k in ("started", "final", "checkpoint")},
        )
        ledger["schema"] = c.SELECTION_SCHEMA
        ledger["selections"][f.cid]["reconciliation"] = ref(self.b.source, output / "receipt.json")
        path = Path(self.b.inputs[backend]["ledger"]["path"])
        path.write_text(json.dumps(ledger))
        self.b.inputs[backend]["ledger"] = base.h.reference(path)
        return f, output, proof

    def test_historical_accounting_real_archive_then_offline_requires_both_captures(self):
        f, _, _ = self.reconcile_one(accounting=True)
        self.reconcile_one("sglang", accounting=True)
        self.b.prepare()
        for path in set(self.b.bundles.values()):
            shutil.rmtree(path)
        shutil.rmtree(self.b.source)
        shutil.rmtree(self.b.root / "bound")
        result = base.p.verify_portable_history(
            self.b.portable, self.b.entry, self.b.records, expected_stage_sha256="a" * 64, archive=base.archive
        )
        self.assertEqual(result["state"], "PORTABLE_METADATA_HISTORY_PASS_NO_FRESH_TAR_VERIFICATION")
        self.assertEqual(self.b.ledgers["vllm"]["attempts"][0]["terminal_state"], "COLLECTION_FAILED_PRESERVED")

    def test_offline_rehashed_historical_proof_cannot_omit_after_capture(self):
        f, _, _ = self.reconcile_one(accounting=True)
        _, history = self.b.make_archive()
        snapshot = copy.deepcopy(history["snapshot"])
        proof = base.h._decoded(snapshot["reconciliations"][f.cid])
        proof["cleanup"].pop("after")
        record = base.h._record(json.dumps(proof).encode())
        snapshot["reconciliations"][f.cid] = record
        snapshot["ledger"]["selections"][f.cid]["reconciliation"].update({k: record[k] for k in ("sha256", "bytes")})
        ledger_record = base.h._record(json.dumps(snapshot["ledger"]).encode())
        snapshot["input_bytes"]["ledger"] = ledger_record
        snapshot["inputs"]["ledger"].update({k: ledger_record[k] for k in ("sha256", "bytes")})
        with self.assertRaisesRegex(ValueError, "termination proof fields"):
            base.h._validate_snapshot(snapshot)

    def test_reconciled_failed_history_real_tar_then_offline_without_originals(self):
        f, _, _ = self.reconcile_one()
        self.reconcile_one("sglang")
        self.b.prepare()
        raw = (self.b.portable / "portable-history.json").read_bytes()
        self.assertIn(b"ACTUAL_TAR_AND_CLOSED_HISTORY_VERIFIED_AT_IMPORT", raw)
        for path in set(self.b.bundles.values()):
            shutil.rmtree(path)
        shutil.rmtree(self.b.source)
        shutil.rmtree(self.b.root / "bound")
        result = base.p.verify_portable_history(
            self.b.portable, self.b.entry, self.b.records, expected_stage_sha256="a" * 64, archive=base.archive
        )
        self.assertEqual(result["state"], "PORTABLE_METADATA_HISTORY_PASS_NO_FRESH_TAR_VERIFICATION")
        self.assertEqual(self.b.ledgers["vllm"]["attempts"][0]["terminal_state"], "COLLECTION_FAILED_PRESERVED")

    def test_old_ledger_does_not_admit_reconciliation(self):
        self.reconcile_one()
        ledger = self.b.ledgers["vllm"]
        ledger["schema"] = c.LEGACY_SELECTION_SCHEMA
        path = Path(self.b.inputs["vllm"]["ledger"]["path"])
        path.write_text(json.dumps(ledger))
        self.b.inputs["vllm"]["ledger"] = base.h.reference(path)
        with self.assertRaisesRegex(ValueError, "legacy"):
            self.b.make_archive()

    def test_added_original_file_after_reconciliation_rejects_actual_archive(self):
        f, _, _ = self.reconcile_one()
        (f.cell / "new-after-proof.txt").write_text("TEST_ONLY unexpected file")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.b.make_archive()

    def test_changed_reviewed_log_rejects_actual_archive(self):
        f, _, _ = self.reconcile_one()
        f.log.write_text("TEST_ONLY unexplained new native error")
        with self.assertRaisesRegex(ValueError, "history member"):
            self.b.make_archive()

    def test_missing_failure_marker_cannot_be_hidden_by_valid_receipt(self):
        _, output, _ = self.reconcile_one()
        (output / "failure.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "failure is preserved"):
            self.b.make_archive()

    def test_offline_missing_embedded_reconciliation_rejects(self):
        self.reconcile_one()
        bundle, proof = self.b.make_archive()
        proof = copy.deepcopy(proof)
        proof["snapshot"]["reconciliations"] = {}
        with self.assertRaisesRegex(ValueError, "missing or orphan"):
            base.h.verify_bundle_history(bundle, proof, base.archive)

    def test_offline_rehashed_wrong_cluster_proof_rejects(self):
        f, _, _ = self.reconcile_one()
        _, history = self.b.make_archive()
        snapshot = copy.deepcopy(history["snapshot"])
        proof = base.h._decoded(snapshot["reconciliations"][f.cid])
        proof["cleanup"]["cluster_command"]["stdout"] = "ClusterName = other\n"
        record = base.h._record(json.dumps(proof).encode())
        snapshot["reconciliations"][f.cid] = record
        snapshot["ledger"]["selections"][f.cid]["reconciliation"].update({k: record[k] for k in ("sha256", "bytes")})
        ledger_record = base.h._record(json.dumps(snapshot["ledger"]).encode())
        snapshot["input_bytes"]["ledger"] = ledger_record
        snapshot["inputs"]["ledger"].update({k: ledger_record[k] for k in ("sha256", "bytes")})
        with self.assertRaisesRegex(ValueError, "scheduler cluster"):
            base.h._validate_snapshot(snapshot)

    def test_changed_original_allocation_rejects_archive_inventory(self):
        f, _, _ = self.reconcile_one()
        f.paths["allocation"].write_text("TEST_ONLY changed original allocation")
        with self.assertRaisesRegex(ValueError, "history member"):
            self.b.make_archive()
