# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for chronological online feedback preparation."""

import contextlib
import gzip
import io
import json
import unittest
from pathlib import Path

from aisimulate_core.sdk.fpm_learned import iter_fpm_records
from scripts.fpm_accuracy import test_prepare_native_training as training_tests
from scripts.fpm_accuracy.prepare_native_training import main, prepare
from scripts.fpm_accuracy.test_validate_native_capture import expectation, record
from scripts.fpm_accuracy.validate_native_capture import expected_workers, file_hash, validate_capture


class NativeOnlineTests(unittest.TestCase):
    setUp = training_tests.NativeTrainingTests.setUp
    write_capture = training_tests.NativeTrainingTests.write_capture
    save_selection = training_tests.NativeTrainingTests.save_selection

    def prepare_offline(self, role="aggregated"):
        self.role = role
        self.selection_path = self.run / f"{role}.selection.json"
        self.report_path = self.run / f"receipts/{role}.capture.json"
        self.write_capture()
        report = validate_capture([self.run / "traces"], expected_workers([expectation(role)]))
        self.report_path.write_text(json.dumps(report))
        self.offline_receipt = prepare(self.run, role)

    def prepare_feedback(self, kinds=("decode", "prefill", "decode"), *, data=None):
        self.feedback_dir = self.run / "later-traces"
        self.feedback_dir.mkdir()
        rows = [record(100 + i, kind=kind) for i, kind in enumerate(kinds)]
        lines = [(json.dumps(row) + "\n").encode() for row in rows]
        self.feedback = self.feedback_dir / "000.jsonl"
        self.feedback.write_bytes(b"".join(lines) if data is None else data)
        self.online_selection_path = self.run / f"{self.role}.online.selection.json"
        self.online_report_path = self.run / f"receipts/{self.role}.online.capture.json"
        self.online_receipt_path = self.run / f"receipts/{self.role}.online.selection.json"
        self.refresh_online_inputs()
        return lines

    def refresh_online_inputs(self):
        report = validate_capture([self.feedback_dir], expected_workers([expectation(self.role)]))
        self.assertEqual(report["status"], "valid")
        self.online_report_path.write_text(json.dumps(report))
        entries = []
        for path in sorted(self.feedback_dir.iterdir()):
            opener = gzip.open if path.suffix == ".gz" else Path.open
            with opener(path, "rb") as source:
                lines = sum(1 for _ in source)
            entries.append(
                {
                    "path": str(path.relative_to(self.run)),
                    "sha256": file_hash(path),
                    "first_line": 1,
                    "last_line": lines,
                }
            )
        self.online_selection_path.write_text(json.dumps(entries))

    def test_chronological_feedback_preserves_all_busy_bytes_and_offline_outputs(self):
        self.prepare_offline()
        offline_paths = [
            self.run / "selected/aggregated.train.jsonl",
            self.run / "selected/aggregated.test.jsonl",
            self.run / "receipts/aggregated.selection.json",
        ]
        before = {path: path.read_bytes() for path in offline_paths}
        lines = self.prepare_feedback(("decode", "idle", "prefill", "mixed", "decode"))
        self.feedback.write_bytes(lines[0] + lines[1])
        later = self.feedback_dir / "001.jsonl.gz"
        later.write_bytes(gzip.compress(b"".join(lines[2:]), mtime=0))
        self.refresh_online_inputs()
        receipt = prepare(self.run, "aggregated", "online")
        output = self.run / "selected/aggregated.online.jsonl"
        self.assertEqual(output.read_bytes(), lines[0] + b"".join(lines[2:]))
        events = [json.loads(line)["event"] for line in (lines[0], *lines[2:])]
        expected = [{**event["fpm"], "observed_at_unix_ms": event["observed_at_unix_ms"]} for event in events]
        self.assertEqual(list(iter_fpm_records([output])), expected)
        self.assertEqual(receipt["counts"], {"online:decode": 2, "online:prefill": 1, "online:mixed": 1})
        self.assertEqual(receipt["excluded"], {"idle_in_selected_ranges": 1})
        self.assertEqual(receipt["split"], "chronological_feedback")
        self.assertEqual(receipt["outputs"]["online"]["sha256"], file_hash(output))
        self.assertEqual(receipt["selection_sha256"], file_hash(self.online_selection_path))
        self.assertEqual(receipt["capture_report_sha256"], file_hash(self.online_report_path))
        self.assertEqual(receipt["offline_selection_receipt_sha256"], file_hash(offline_paths[-1]))
        self.assertEqual(receipt["offline_latest_observed_at_unix_ms"], 1003)
        self.assertEqual(json.loads(self.online_receipt_path.read_text()), receipt)
        self.assertEqual({path: path.read_bytes() for path in offline_paths}, before)

    def test_online_filters_work_incompatible_with_dedicated_role(self):
        for role, selected in (("prefill", [0, 4]), ("decode", [1])):
            with self.subTest(role=role):
                self.setUp()
                self.prepare_offline(role)
                lines = self.prepare_feedback(("prefill", "decode", "mixed", "idle", "prefill"))
                receipt = prepare(self.run, role, "online")
                self.assertEqual(
                    (self.run / f"selected/{role}.online.jsonl").read_bytes(), b"".join(lines[i] for i in selected)
                )
                self.assertEqual(receipt["excluded"]["role_incompatible_in_selected_ranges"], 4 - len(selected))

    def test_offline_capture_and_train_test_files_cannot_be_reused(self):
        for origin in ("traces/native.jsonl", "selected/aggregated.train.jsonl", "selected/aggregated.test.jsonl"):
            with self.subTest(origin=origin):
                self.setUp()
                self.prepare_offline()
                self.prepare_feedback(data=(self.run / origin).read_bytes())
                with self.assertRaisesRegex(ValueError, "cannot reuse an offline"):
                    prepare(self.run, "aggregated", "online")
                self.assertFalse(self.online_receipt_path.exists())

    def test_reformatted_heldout_rows_fail_the_later_observation_boundary(self):
        self.prepare_offline()
        test = self.run / "selected/aggregated.test.jsonl"
        repacked = b"".join((json.dumps(json.loads(line)) + "\n").encode() for line in test.read_bytes().splitlines())
        self.assertNotEqual(repacked, test.read_bytes())
        self.prepare_feedback(data=repacked)
        with self.assertRaisesRegex(ValueError, "strictly later"):
            prepare(self.run, "aggregated", "online")
        self.assertFalse(self.online_receipt_path.exists())

    def test_repacked_unselected_offline_capture_rows_cannot_become_feedback(self):
        self.role = "aggregated"
        lines = self.write_capture(("decode",) * 6)
        self.selection[0]["last_line"] = 4
        self.save_selection()
        prepare(self.run, self.role)
        repacked = b"".join((json.dumps(json.loads(line)) + "\n").encode() for line in lines[4:])
        self.prepare_feedback(data=repacked)
        with self.assertRaisesRegex(ValueError, "counters must increase"):
            prepare(self.run, self.role, "online")
        self.assertFalse(self.online_receipt_path.exists())

    def test_timestamp_boundary_ties_and_backwards_order_are_rejected(self):
        for times, message in (([1003], "strictly later"), ([1101, 1100], "timestamps go backwards")):
            with self.subTest(times=times):
                self.setUp()
                self.prepare_offline()
                rows = [record(100 + i) for i in range(len(times))]
                for row, observed in zip(rows, times, strict=True):
                    row["event"]["observed_at_unix_ms"] = observed
                self.prepare_feedback(data=b"".join((json.dumps(row) + "\n").encode() for row in rows))
                with self.assertRaisesRegex(ValueError, message):
                    prepare(self.run, "aggregated", "online")

    def test_reversed_selected_segments_fail_without_sorting(self):
        self.prepare_offline()
        lines = self.prepare_feedback()
        self.feedback.write_bytes(lines[0])
        (self.feedback_dir / "001.jsonl").write_bytes(b"".join(lines[1:]))
        self.refresh_online_inputs()
        selection = json.loads(self.online_selection_path.read_text())
        self.online_selection_path.write_text(json.dumps(selection[::-1]))
        with self.assertRaisesRegex(ValueError, "timestamps go backwards"):
            prepare(self.run, "aggregated", "online")

    def test_restarted_same_worker_counter_fails_even_with_later_timestamp(self):
        self.prepare_offline()
        row = record(0)
        row["event"]["observed_at_unix_ms"] = 2000
        self.prepare_feedback(data=(json.dumps(row) + "\n").encode())
        with self.assertRaisesRegex(ValueError, "counters must increase"):
            prepare(self.run, "aggregated", "online")

    def test_online_requires_one_dp_zero_stream(self):
        for multi in (True, False):
            with self.subTest(multiple_workers=multi):
                self.setUp()
                self.prepare_offline()
                self.prepare_feedback()
                report = json.loads(self.online_report_path.read_text())
                if multi:
                    report["expected_workers"].append(report["expected_workers"][0])
                else:
                    report["expected_workers"][0]["identity"]["dp_rank"] = 1
                self.online_report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "exactly one expected DP=0"):
                    prepare(self.run, "aggregated", "online")

    def test_offline_receipt_and_output_integrity_are_required(self):
        for target, message in (
            ("aggregated.selection.json", "offline selection or capture report"),
            ("selected/aggregated.test.jsonl", "offline output hash"),
        ):
            with self.subTest(target=target):
                self.setUp()
                self.prepare_offline()
                self.prepare_feedback()
                path = self.run / target
                path.write_bytes(path.read_bytes().replace(b" ", b"  "))
                with self.assertRaisesRegex(ValueError, message):
                    prepare(self.run, "aggregated", "online")

    def test_online_source_hash_must_match_its_capture_report(self):
        self.prepare_offline()
        self.prepare_feedback()
        with self.feedback.open("ab") as source:
            source.write(b"\n")
        with self.assertRaisesRegex(ValueError, "source hash"):
            prepare(self.run, "aggregated", "online")

    def test_online_cli_publishes_once_and_requires_role_compatible_rows(self):
        self.prepare_offline("decode")
        self.prepare_feedback(("decode",))
        args = ["--run-dir", str(self.run), "--role", "decode", "--mode", "online"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertTrue(self.online_receipt_path.exists())
        with self.assertRaisesRegex(ValueError, "fresh outputs"):
            prepare(self.run, "decode", "online")
        self.setUp()
        self.prepare_offline("decode")
        self.prepare_feedback(("prefill", "mixed", "idle"))
        with self.assertRaisesRegex(ValueError, "empty role-compatible"):
            prepare(self.run, "decode", "online")
        self.assertFalse(self.online_receipt_path.exists())


if __name__ == "__main__":
    unittest.main()
