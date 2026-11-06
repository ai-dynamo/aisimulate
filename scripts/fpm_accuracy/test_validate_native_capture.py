# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local integrity checks using original synthetic native-envelope fixtures."""

import contextlib
import copy
import gzip
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.fpm_accuracy.validate_native_capture import expected_workers, main, validate_capture


def record(counter=0, worker="worker-1", kind="decode", namespace="owned-run"):
    p, d = int(kind in ("prefill", "mixed")), int(kind in ("decode", "mixed"))
    return {
        "timestamp": counter,
        "event": {
            "schema": "dynamo.fpm.trace.v1",
            "capture_mode": "full",
            "observed_at_unix_ms": 1000 + counter,
            "source": {"namespace": namespace, "component": "backend", "producer_id": worker},
            "fpm": {
                "version": 1,
                "worker_id": worker,
                "dp_rank": 0,
                "counter_id": counter,
                "scheduled_requests": {
                    "num_prefill_requests": p,
                    "sum_prefill_tokens": 8 * p,
                    "sum_prefill_kv_tokens": 16 * p,
                    "num_decode_requests": d,
                    "sum_decode_kv_tokens": 32 * d,
                },
                "queued_requests": {},
                "wall_time": 0.01 if p or d else 0,
            },
        },
    }


def expectation(role="aggregated", worker="worker-1", namespace="owned-run"):
    return [role, namespace, "backend", worker, worker, "0"]


class NativeCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.traces = self.root / "traces"
        self.traces.mkdir()

    def write(self, rows, name="capture.jsonl", directory=None):
        path = (directory or self.traces) / name
        data = "".join(json.dumps(row) + "\n" for row in rows).encode()
        path.write_bytes(gzip.compress(data, mtime=0) if name.endswith(".gz") else data)
        return path

    def check(self, specs=None, directories=None):
        return validate_capture(directories or [self.traces], expected_workers(specs or [expectation()]))

    def test_aggregate_preserves_sources_and_reports_every_work_kind(self):
        path = self.write(
            [record(i, kind=kind) for i, kind in enumerate(("idle", "prefill", "decode", "mixed"))], "capture.jsonl.gz"
        )
        before = path.read_bytes()
        report = self.check()
        self.assertEqual(report["status"], "valid")
        self.assertEqual(
            report["counts"],
            {"raw": 4, "valid_records": 4, "busy": 3, "idle": 1, "prefill": 1, "decode": 1, "mixed": 1},
        )
        self.assertEqual(report["workers"][0]["declared_role"], "aggregated")
        self.assertEqual(report["files"][0]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertTrue(report["files"][0]["complete_read"])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(report["publisher_losslessness"], "not_proven")
        self.assertEqual(report["model_readiness"], "not_assessed")

    def test_pd_roles_do_not_change_for_decode_admission(self):
        prefill, decode = self.root / "prefill", self.root / "decode"
        prefill.mkdir()
        decode.mkdir()
        self.write([record(worker="p", kind="prefill")], directory=prefill)
        self.write([record(worker="d", kind="mixed"), record(1, worker="d")], directory=decode)
        report = self.check([expectation("prefill", "p"), expectation("decode", "d")], [prefill, decode])
        self.assertEqual(report["status"], "valid")
        workers = {row["declared_role"]: row for row in report["workers"]}
        self.assertEqual(workers["decode"]["counts"]["mixed"], 1)
        self.assertEqual(workers["decode"]["counts"]["decode"], 1)

    def test_native_role_conflict_fails(self):
        row = record()
        row["event"]["fpm"]["worker_type"] = "prefill"
        self.write([row])
        report = self.check()
        self.assertEqual(report["status"], "invalid")
        self.assertEqual(report["issues"]["role_conflict"], 1)

    def test_missing_and_unexpected_workers_fail(self):
        self.write([record(worker="other")])
        report = self.check()
        self.assertEqual(report["status"], "invalid")
        self.assertEqual(report["issues"]["missing_worker"], 1)
        self.assertEqual(report["issues"]["unexpected_worker"], 1)

    def test_identity_includes_namespace_and_rank(self):
        first, second = record(), record(namespace="another-run")
        self.write([first, second])
        report = self.check([expectation(), expectation(namespace="another-run")])
        self.assertEqual(report["status"], "valid")
        second["event"]["fpm"]["dp_rank"] = 1
        self.write([first, second])
        self.assertEqual(self.check([expectation(), expectation(namespace="another-run")])["status"], "invalid")

    def test_corrupt_and_truncated_gzip_never_pass(self):
        valid = gzip.compress((json.dumps(record()) + "\n").encode(), mtime=0)
        for data in (b"not gzip", valid[:-6], valid + valid[:-6], b"\x1f\x8b\x08\x00" + b"\xff" * 30):
            with self.subTest(data=data[:10]):
                path = self.traces / "capture.jsonl.gz"
                path.write_bytes(data)
                report = self.check()
                self.assertEqual(report["status"], "invalid")
                self.assertFalse(report["files"][0]["complete_read"])
                self.assertIn("file_read_error", report["issues"])

    def test_malformed_json_is_not_silently_skipped(self):
        path = self.write([record()])
        with path.open("ab") as stream:
            stream.write(b'{"event":\n')
        report = self.check()
        self.assertEqual(report["status"], "invalid")
        self.assertEqual(report["counts"]["raw"], 2)
        self.assertEqual(report["issues"]["invalid_record"], 1)
        self.assertTrue(report["files"][0]["complete_read"])

    def test_empty_gzip_segment_does_not_pass_beside_valid_records(self):
        self.write([record()], "capture.000000.jsonl.gz")
        (self.traces / "capture.000001.jsonl.gz").write_bytes(b"")
        report = self.check()
        self.assertEqual(report["status"], "invalid")
        self.assertEqual(report["issues"]["empty_gzip"], 1)
        self.assertFalse(report["files"][1]["complete_read"])

    def test_invalid_counters_and_durations_fail(self):
        bad_values = (float("nan"), float("inf"), -1, True, "1", None)
        for field in ("num_decode_requests", "sum_decode_kv_tokens", "wall_time", "counter_id"):
            values = bad_values + ((0,) if field == "wall_time" else (1.25,))
            for value in values:
                with self.subTest(field=field, value=value):
                    row = record()
                    target = row["event"]["fpm"]
                    if field in target["scheduled_requests"]:
                        target = target["scheduled_requests"]
                    target[field] = value
                    self.write([row])
                    self.assertEqual(self.check()["status"], "invalid")

    def test_inconsistent_counts_and_duplicate_json_keys_fail(self):
        row = record(kind="prefill")
        row["event"]["fpm"]["scheduled_requests"]["sum_prefill_tokens"] = 0
        self.write([row])
        self.assertEqual(self.check()["status"], "invalid")
        path = self.write([record()])
        path.write_text(path.read_text().replace('"wall_time": 0.01', '"wall_time": 0.01, "wall_time": 0.02'))
        self.assertEqual(self.check()["status"], "invalid")

    def test_duplicates_conflicts_and_backwards_counters_require_review(self):
        first = record(4)
        different = copy.deepcopy(first)
        different["event"]["fpm"]["wall_time"] = 0.02
        for rows, code in (
            ([first, first], "duplicate_counter"),
            ([first, different], "reused_counter_different_payload"),
            ([first, record(0)], "counter_backwards"),
        ):
            with self.subTest(code=code):
                self.write(rows)
                report = self.check()
                self.assertEqual(report["status"], "review_required")
                self.assertEqual(report["issues"][code], 1)

    def test_gaps_are_warnings_not_loss_or_failure(self):
        self.write([record(3), record(8)])
        report = self.check()
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["counts"]["counter_jump_slots"], 4)
        self.assertEqual(report["severity_counts"], {"warning": 1})

    def test_out_of_order_gap_is_only_a_scan_order_jump(self):
        self.write([record(1), record(3), record(2), record(3)])
        report = self.check()
        self.assertEqual(report["status"], "review_required")
        self.assertEqual(report["counts"]["counter_jump_slots"], 1)
        self.assertNotIn("counter_ids_not_observed", report["counts"])

    def test_idle_only_worker_requires_review(self):
        self.write([record(kind="idle")])
        report = self.check()
        self.assertEqual(report["status"], "review_required")
        self.assertEqual(report["issues"]["worker_has_no_busy_records"], 1)

    def test_cli_writes_fresh_report_and_refuses_overwrite_or_input_overlap(self):
        self.write([record()])
        report = self.root / "quality.json"
        args = ["--trace-dir", str(self.traces), "--expect", *expectation(), "--report", str(report)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        original = report.read_bytes()
        for destination in (report, self.traces / "quality.json"):
            with (
                self.subTest(destination=destination),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main([*args[:-1], str(destination)])
        self.assertEqual(report.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
