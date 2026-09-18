# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject corrupt coverage evidence and expose bad timing rows."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import audit_pr244_data as audit


class CollectionEvidenceTest(unittest.TestCase):
    def setUp(self):
        ids = ["case-a", "case-b", "case-c"]
        digest = audit.plan_hash(ids)
        attempt = {
            "id": "gemm",
            "op": "gemm",
            "plan_sha256": digest,
            "plan_source": "archived_case_plan",
            "outcomes": "DFU",
            "reported_counts": {"expected": 3, "done": 1, "failed": 1, "unattempted": 1},
        }
        self.ledger = {
            "plans": {digest: ids},
            "systems": {system: {"attempts": [copy.deepcopy(attempt)]} for system in audit.SYSTEMS},
        }

    def test_unattempted_cases_are_counted(self):
        summary = audit.summarize_cases(self.ledger)
        self.assertEqual(summary["b300_sxm"]["cases"], {"expected": 3, "done": 1, "failed": 1, "unattempted": 1})

    def test_rejects_tampered_plan_and_outcomes(self):
        mutations = [
            lambda ledger: next(iter(ledger["plans"].values())).append("case-d"),
            lambda ledger: ledger["systems"]["b300_sxm"]["attempts"][0].update(outcomes="DF"),
            lambda ledger: ledger["systems"]["b300_sxm"]["attempts"][0].update(outcomes="DDD"),
            lambda ledger: ledger["systems"]["b300_sxm"]["attempts"][0].update(outcomes="DFX"),
            lambda ledger: ledger["systems"]["b300_sxm"]["attempts"].append(
                ledger["systems"]["b300_sxm"]["attempts"][0]
            ),
        ]
        for mutate in mutations:
            ledger = copy.deepcopy(self.ledger)
            mutate(ledger)
            with self.assertRaises(ValueError):
                audit.summarize_cases(ledger)

    def test_reports_duplicate_null_and_invalid_latency_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "test.parquet"
            pq.write_table(
                pa.table(
                    {
                        "m": [1, 1, 2, 3],
                        "gemm_dtype": ["fp8", "fp8", None, "fp8"],
                        "latency": [1.0, -1.0, float("inf"), 0.0],
                    }
                ),
                path,
            )
            path.with_name("collection_meta.yaml").write_text("schema_version: 2\n")
            with patch.object(audit, "ROOT", root):
                result = audit.summarize_table(path)
            self.assertEqual(
                result["anomalies"],
                {"duplicate_physical_keys": 1, "null_cells": 1, "nonfinite_latency": 1, "nonpositive_latency": 2},
            )
            self.assertEqual(result["latency_unit"], "ms")
            self.assertEqual(result["columns"]["m"]["max"], 3)


if __name__ == "__main__":
    unittest.main()
