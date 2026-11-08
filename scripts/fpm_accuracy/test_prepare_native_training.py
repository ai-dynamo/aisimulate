# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral checks for reviewed selection and lossless native-line staging."""

import contextlib
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.fpm_accuracy.prepare_native_training import main, prepare
from scripts.fpm_accuracy.test_validate_native_capture import expectation, record
from scripts.fpm_accuracy.validate_native_capture import expected_workers, file_hash, validate_capture


class NativeTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        (self.run / "traces").mkdir()
        (self.run / "receipts").mkdir()
        self.selection_path = self.run / "aggregated.selection.json"
        self.report_path = self.run / "receipts/aggregated.capture.json"

    def write_capture(self, kinds=("prefill", "decode", "prefill", "decode"), compressed=False):
        rows = [record(i, kind=kind) for i, kind in enumerate(kinds)]
        lines = [(json.dumps(row, separators=(", ", " : ")) + "\n").encode() for row in rows]
        self.source = self.run / ("traces/native.jsonl.gz" if compressed else "traces/native.jsonl")
        data = b"".join(lines)
        self.source.write_bytes(gzip.compress(data, mtime=0) if compressed else data)
        report = validate_capture([self.run / "traces"], expected_workers([expectation()]))
        self.report_path.write_text(json.dumps(report))
        self.selection = [
            {
                "path": str(self.source.relative_to(self.run)),
                "sha256": file_hash(self.source),
                "first_line": 1,
                "last_line": len(rows),
            }
        ]
        self.save_selection()
        return lines

    def save_selection(self):
        self.selection_path.write_text(json.dumps(self.selection))

    def test_selected_lines_and_native_values_are_preserved_exactly(self):
        lines = self.write_capture(
            ("decode", "idle", "prefill", "decode", "mixed", "prefill", "decode", "mixed", "decode"), compressed=True
        )
        original = self.source.read_bytes()
        self.selection[0].update(first_line=2, last_line=8)
        self.save_selection()
        receipt = prepare(self.run, "aggregated")
        train = self.run / "selected/aggregated.train.jsonl"
        test = self.run / "selected/aggregated.test.jsonl"
        self.assertEqual(train.read_bytes(), lines[2] + lines[3] + lines[4])
        self.assertEqual(test.read_bytes(), lines[5] + lines[6] + lines[7])
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(receipt["excluded"], {"outside_selected_ranges": 2, "idle_in_selected_ranges": 1})
        self.assertEqual(
            receipt["counts"],
            {
                "train:prefill": 1,
                "train:decode": 1,
                "train:mixed": 1,
                "test:prefill": 1,
                "test:decode": 1,
                "test:mixed": 1,
            },
        )
        self.assertEqual(receipt["selection_sha256"], file_hash(self.selection_path))
        self.assertEqual(receipt["capture_report_sha256"], file_hash(self.report_path))
        self.assertEqual(receipt["outputs"]["test"]["sha256"], file_hash(test))
        self.assertEqual(json.loads((self.run / "receipts/aggregated.selection.json").read_text()), receipt)
        self.assertFalse(list(self.run.rglob("*.partial")))

    def test_invalid_report_and_role_binding_are_rejected(self):
        self.write_capture()
        report = json.loads(self.report_path.read_text())
        for bad in ({**report, "status": "review_required"}, {**report, "expected_workers": []}):
            with self.subTest(bad=bad["status"]):
                self.report_path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    prepare(self.run, "aggregated")
        report["expected_workers"][0]["declared_role"] = "decode"
        self.report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "role"):
            prepare(self.run, "aggregated")

    def test_changed_source_cannot_reuse_previous_validation(self):
        self.write_capture()
        with self.source.open("ab") as output:
            output.write(b"\n")
        with self.assertRaisesRegex(ValueError, "source hash"):
            prepare(self.run, "aggregated")
        self.assertFalse((self.run / "selected").exists())

    def test_selection_cannot_add_unvalidated_source(self):
        self.write_capture()
        extra = self.run / "traces/unvalidated.jsonl"
        extra.write_text(json.dumps(record(100)) + "\n")
        self.selection[0].update(path="traces/unvalidated.jsonl", sha256=file_hash(extra))
        self.save_selection()
        with self.assertRaisesRegex(ValueError, "source hash"):
            prepare(self.run, "aggregated")

    def test_ranges_must_be_explicit_nonempty_integers(self):
        self.write_capture()
        for first, last in ((None, None), (True, 3), (0, 2), (3, 2), (1.0, 2)):
            with self.subTest(first=first, last=last):
                self.selection[0].update(first_line=first, last_line=last)
                self.save_selection()
                with self.assertRaisesRegex(ValueError, "line range"):
                    prepare(self.run, "aggregated")

    def test_duplicate_source_and_path_escape_are_rejected(self):
        self.write_capture()
        self.selection.append(dict(self.selection[0]))
        self.save_selection()
        with self.assertRaisesRegex(ValueError, "repeated"):
            prepare(self.run, "aggregated")
        self.selection = [{**self.selection[0], "path": "../outside.jsonl"}]
        self.save_selection()
        with self.assertRaisesRegex(ValueError, "inside this run"):
            prepare(self.run, "aggregated")

    def test_out_of_bounds_range_leaves_no_completed_outputs(self):
        self.write_capture()
        self.selection[0]["last_line"] = 100
        self.save_selection()
        with self.assertRaisesRegex(ValueError, "beyond"):
            prepare(self.run, "aggregated")
        self.assertTrue((self.run / "selected/aggregated.train.jsonl.partial").exists())
        self.assertFalse((self.run / "selected/aggregated.train.jsonl").exists())
        self.assertFalse((self.run / "receipts/aggregated.selection.json").exists())

    def test_one_busy_row_cannot_produce_an_empty_test_set(self):
        self.write_capture(("decode",))
        with self.assertRaisesRegex(ValueError, "empty training or test"):
            prepare(self.run, "aggregated")
        self.assertFalse((self.run / "receipts/aggregated.selection.json").exists())

    def test_existing_outputs_are_not_replaced(self):
        self.write_capture()
        (self.run / "selected").mkdir()
        target = self.run / "selected/aggregated.test.jsonl"
        target.write_bytes(b"existing data")
        with self.assertRaisesRegex(ValueError, "fresh"):
            prepare(self.run, "aggregated")
        self.assertEqual(target.read_bytes(), b"existing data")

    def test_cli_stages_without_environment_variables(self):
        self.write_capture()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--run-dir", str(self.run), "--role", "aggregated"]), 0)
        self.assertTrue((self.run / "receipts/aggregated.selection.json").is_file())


if __name__ == "__main__":
    unittest.main()
