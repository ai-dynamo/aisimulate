# SPDX-License-Identifier: Apache-2.0
"""CPU-only boundary checks for the published replay input contract."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import replay


class ReplayInputTests(unittest.TestCase):
    def config(self):
        return {
            "model_name": "model",
            "backend": "sglang",
            "system_name": "gb300",
            "backend_version": "version",
            "decoder_replay": False,
            "systems_path": "systems",
        }

    def row(self):
        spec = {
            "engine": {
                "rank": {"timing_model": {"config": {"database_mode": "SILICON"}}}
            },
            "requests": [
                {
                    "id": "r",
                    "input_tokens": 2,
                    "input_token_ids": [17, 19],
                    "output_tokens": 2,
                    "arrival_time_ms": 0.3,
                }
            ],
        }
        return {
            "recovery_status": "exact_original_spec_sha256",
            "replay_spec": spec,
            "replay_spec_sha256": hashlib.sha256(
                replay.canonical(spec).encode()
            ).hexdigest(),
        }

    def test_only_timing_provider_changes(self):
        row = self.row()
        saved = deepcopy(row)
        result = replay.current_spec(row, self.config())
        self.assertEqual(row, saved)
        self.assertEqual(result["requests"], row["replay_spec"]["requests"])
        self.assertEqual(
            result["engine"]["rank"]["timing_model"]["config"]["database_mode"], "SOL"
        )

    def test_token_or_arrival_mutation_rejected(self):
        for key, value in [("input_token_ids", [17, 20]), ("arrival_time_ms", 0.4)]:
            with self.subTest(key=key):
                row = self.row()
                row["replay_spec"]["requests"][0][key] = value
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    replay.current_spec(row, self.config())

    def test_missing_recovery_is_not_guessed(self):
        row = self.row()
        row["recovery_status"] = "unavailable"
        with self.assertRaisesRegex(ValueError, "unavailable"):
            replay.current_spec(row, self.config())

    def test_corrupted_published_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "input.json"
            data.write_text("{}")
            (root / "artifact-hashes.json").write_text(
                json.dumps({"input.json": {"sha256": replay.digest(data)}})
            )
            with patch.object(replay, "ROOT", root):
                self.assertEqual(replay.checked_bytes(data), b"{}")
                data.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    replay.checked_bytes(data)

    def test_python_facade_source_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / "model.py"
            module.write_text("# changed\n")
            identity = {"modules": {"model": {"sha256": "0" * 64}}}
            with patch.object(
                replay, "checked_bytes", return_value=json.dumps(identity).encode()
            ):
                with patch.object(
                    replay.importlib,
                    "import_module",
                    return_value=SimpleNamespace(__file__=str(module)),
                ):
                    with self.assertRaisesRegex(
                        ValueError, "Python predictor source mismatch"
                    ):
                        replay.qualify_python_sources()

    def report(self):
        return {
            "per_request": [
                {
                    "request_id": "r",
                    "terminal_status": "completed",
                    "arrival_time_ms": 2.0,
                    "ttft_ms": 3.0,
                    "first_token_ms": 5.0,
                    "last_token_ms": 9.0,
                    "terminal_time_ms": 9.0,
                    "output_length": 3,
                    "itl_ms": 2.0,
                }
            ]
        }

    def test_http_metrics_preserve_arrival_and_token_span(self):
        result = replay.http_metrics(self.report(), ["r"], 1.0)
        self.assertEqual(
            result,
            {
                "ttft_ms": 3.0,
                "request_latency_ms": 7.0,
                "last_token_latency_ms": 7.0,
                "output_tokens_per_second": 375.0,
                "average_tpot_ms": 2.0,
                "exact_itl_ms": 2.0,
            },
        )

    def test_duplicate_or_missing_request_rejected(self):
        report = self.report()
        report["per_request"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            replay.http_metrics(report, ["r"], 0.0)
        with self.assertRaisesRegex(ValueError, "missing"):
            replay.http_metrics(self.report(), ["absent"], 0.0)

    def test_incomplete_request_rejected(self):
        report = self.report()
        report["per_request"][0]["terminal_status"] = "failed"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            replay.http_metrics(report, ["r"], 0.0)

    def test_nonpositive_or_nonfinite_metric_rejected(self):
        for value in [0.0, float("nan"), float("inf")]:
            with self.subTest(value=value):
                report = self.report()
                report["per_request"][0]["ttft_ms"] = value
                with self.assertRaisesRegex(ValueError, "invalid metric"):
                    replay.http_metrics(report, ["r"], 0.0)

    def test_invalid_cohort_duration_rejected(self):
        with self.assertRaisesRegex(ValueError, "completion time"):
            replay.http_metrics(self.report(), ["r"], 10.0)

    def test_single_output_has_no_tpot(self):
        report = self.report()
        report["per_request"][0]["output_length"] = 1
        self.assertNotIn("average_tpot_ms", replay.http_metrics(report, ["r"], 0.0))


if __name__ == "__main__":
    unittest.main()
