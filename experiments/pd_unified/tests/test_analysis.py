"""Contract tests use synthetic report fixtures; no simulator is imported."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "analyze_results.py"
_SPEC = importlib.util.spec_from_file_location("baseline_analysis", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
analyze_run = _MODULE.analyze_run


def request(uuid: str, *, output: int = 3, ttft: float = 10.0, itl: float = 20.0) -> dict:
    return {
        "uuid": uuid,
        "terminal_status": "completed",
        "arrival_time_ms": 0.0,
        "first_admit_ms": 2.0,
        "first_token_ms": ttft,
        "last_token_ms": ttft + max(0, output - 1) * itl,
        "terminal_time_ms": ttft + max(0, output - 1) * itl,
        "ttft_ms": ttft,
        "itl_ms": itl if output > 1 else None,
        "input_length": 32,
        "requested_output_length": output,
        "output_length": output,
    }


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(
        self,
        records: list[dict],
        *,
        expected: int | None = None,
        reported: int | None = None,
        mode: str = "aggregated",
        telemetry: list[dict] | None = None,
    ) -> dict:
        if expected is None:
            expected = len(records)
        if reported is None:
            reported = expected
        native = {
            "summary": {
                "num_requests": reported,
                "completed_requests": sum(row["terminal_status"] == "completed" for row in records),
                "duration_ms": 1000.0,
                "native_only_evidence": "preserve me",
            },
            "per_request": records,
            "telemetry": {"sample_interval_ms": 1000.0, "samples": telemetry or []},
        }
        (self.root / "prediction.json").write_text(json.dumps(native), encoding="utf-8")
        (self.root / "run-metadata.json").write_text(
            json.dumps(
                {
                    "case": "synthetic-unit-fixture",
                    "engine_mode": mode,
                    "num_gpus": 4,
                    "speedup": 1.0,
                    "input_requests": expected,
                    "sla": {"ttft_ms": 5000.0, "itl_ms": 50.0},
                }
            ),
            encoding="utf-8",
        )
        return native

    def read_csv(self, name: str) -> list[dict]:
        with (self.root / name).open(encoding="utf-8", newline="") as source:
            return list(csv.DictReader(source))

    def test_single_output_skips_itl_sla_and_statistics(self) -> None:
        self.write([request("a", output=1), request("b", itl=60.0)])
        result = analyze_run(self.root)
        self.assertTrue(result["valid"])
        self.assertEqual(result["num_sla_good_full_output"], 1)
        self.assertEqual(result["goodput_request_rps"], 1.0)
        self.assertEqual(result["distributions"]["request_mean_itl"]["count"], 1)
        self.assertEqual(result["p99_request_mean_itl_ms"], 60.0)
        self.assertEqual(self.read_csv("requests.csv")[0]["request_mean_itl_ms"], "")

    def test_zero_completions_are_invalid_and_keep_evidence(self) -> None:
        record = request("rejected")
        record.update(
            terminal_status="rejected",
            output_length=0,
            first_token_ms=None,
            last_token_ms=None,
            ttft_ms=None,
            itl_ms=None,
        )
        native = self.write([record])
        result = analyze_run(self.root)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["p99_ttft_ms"])
        self.assertIsNone(result["goodput_request_rps"])
        self.assertEqual(result["observed_goodput_request_rps"], 0.0)
        self.assertEqual(result["terminal_status_counts"]["rejected"], 1)
        self.assertEqual(result["native_summary"], native["summary"])

    def test_completed_but_truncated_output_is_invalid(self) -> None:
        row = request("short")
        row["output_length"] = 2
        self.write([row])
        result = analyze_run(self.root)
        self.assertFalse(result["valid"])
        self.assertEqual(result["num_completed_output_mismatch"], 1)
        self.assertEqual(result["num_completed_full_output"], 0)
        self.assertIsNone(result["goodput_request_rps"])

    def test_missing_terminal_and_not_yet_reported_requests_are_distinct(self) -> None:
        self.write([request("done")], expected=3, reported=2)
        result = analyze_run(self.root)
        self.assertFalse(result["valid"])
        self.assertEqual(result["num_reported_without_terminal_record"], 1)
        self.assertEqual(result["num_input_minus_reported"], 1)
        self.assertEqual(result["num_completed_full_output"], 1)

    def test_aggregated_telemetry_labels_cache_and_separate_router_queue(self) -> None:
        sample = {
            "sample_ordinal": 1,
            "kind": "periodic",
            "interval_start_ms": 0,
            "sampled_at_ms": 1000,
            "traffic": {"arriving_requests": 1, "completed_requests": 1},
            "prefill_scheduler_metrics": [],
            "decode_scheduler_metrics": [
                {
                    "worker_id": 0,
                    "dp_rank": 0,
                    "active_cache_usage": 0.3,
                    "physical_cache_usage": 0.5,
                    "running_requests": 2,
                    "waiting_requests": 4,
                }
            ],
            "decode_interval_metrics": {
                "preemptions": 1,
                "cache_hit_tokens": 0,
                "cache_total_tokens": 0,
            },
            "router_pending_prefill_requests": 0,
            "router_pending_decode_requests": 7,
            "active_decode_ids": [0],
        }
        self.write([request("ok")], telemetry=[sample])
        result = analyze_run(self.root)
        self.assertTrue(result["valid"])
        row = self.read_csv("scheduler_samples.csv")[0]
        self.assertEqual(row["role"], "aggregated")
        self.assertEqual(row["waiting_requests"], "4")
        self.assertEqual(row["router_pending_decode_requests"], "7")
        self.assertEqual(row["physical_cache_usage"], "0.5")
        traffic = self.read_csv("traffic_samples.csv")[0]
        self.assertEqual(traffic["aggregated_scheduler_cache_hit_ratio"], "")
        self.assertEqual(traffic["aggregated_preemptions"], "1")
        self.assertFalse(any("gpu_util" in key for key in row))

    def test_percentiles_are_equal_per_request_mean_itl(self) -> None:
        # Ten-token and hundred-token requests receive equal percentile weight.
        self.write(
            [
                request("short", output=2, ttft=10.0, itl=10.0),
                request("long", output=101, ttft=100.0, itl=100.0),
            ]
        )
        result = analyze_run(self.root)
        self.assertTrue(result["valid"])
        self.assertEqual(result["p50_request_mean_itl_ms"], 55.0)
        self.assertAlmostEqual(result["p95_request_mean_itl_ms"], 95.5)
        self.assertAlmostEqual(result["p99_request_mean_itl_ms"], 99.1)
        self.assertEqual(result["p50_ttft_ms"], 55.0)
        self.assertEqual(result["p99_first_admit_wait_ms"], 2.0)

    def test_pd_wait_is_combined_post_prefill_wait(self) -> None:
        row = request("pd")
        row.update(source_held_ms=3.0, decode_admit_ms=8.0)
        self.write([row], mode="disaggregated")
        result = analyze_run(self.root)
        self.assertEqual(result["p99_post_prefill_wait_ms"], 5.0)
        self.assertIn("not isolated transfer", result["metric_semantics"]["post_prefill_wait"])
        self.assertEqual(self.read_csv("requests.csv")[0]["post_prefill_wait_ms"], "5.0")

    def test_telemetry_jsonl_takes_precedence_and_outputs_exist(self) -> None:
        self.write([request("ok")])
        (self.root / "telemetry.jsonl").write_text(
            json.dumps({"sample_ordinal": 0, "kind": "baseline", "sampled_at_ms": 0}) + "\n",
            encoding="utf-8",
        )
        result = analyze_run(self.root)
        self.assertEqual(result["telemetry_source"], "telemetry.jsonl")
        self.assertEqual(result["num_telemetry_samples"], 1)
        self.assertEqual(
            json.loads((self.root / "analysis.json").read_text(encoding="utf-8")),
            result,
        )
        for name in ("requests.csv", "scheduler_samples.csv", "traffic_samples.csv"):
            self.assertTrue((self.root / name).is_file())

    def test_missing_latency_does_not_publish_valid_goodput(self) -> None:
        row = request("missing")
        row.update(ttft_ms=None, first_token_ms=None)
        self.write([row])
        result = analyze_run(self.root)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["goodput_request_rps"])


if __name__ == "__main__":
    unittest.main()
