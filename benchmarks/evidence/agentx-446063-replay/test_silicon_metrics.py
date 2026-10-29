# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-authored interval fixtures; no upstream implementation is copied."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from silicon_metrics import SCHEMA, load_silicon_metrics

ORIGIN = 1_792_599_226_802_738_376


def series(kind, values, rank=0, mode=None, dp=0):
    labels = {"tp_rank": str(rank), "dp_rank": str(dp), "model_name": "test-model"}
    if mode is not None:
        labels["mode"] = mode
    return {
        "endpoint_url": "http://fixture.invalid/metrics",
        "labels": labels,
        "timeslices": [
            {
                "start_ns": ORIGIN + int(start * 1e9),
                "end_ns": ORIGIN + int(end * 1e9),
                "avg" if kind == "gauge" else "total": value,
            }
            for start, end, value in values
        ],
    }


def fixture():
    metrics = {
        "sglang:max_total_num_tokens": {
            "type": "gauge",
            "series": [series("gauge", [(0, 1, 100), (1, 2, 100)], rank) for rank in range(2)],
        },
        "sglang:kv_used_tokens": {"type": "gauge", "series": [series("gauge", [(0, 1, 20), (1, 2, 30)])]},
        "sglang:kv_evictable_tokens": {"type": "gauge", "series": [series("gauge", [(0, 1, 10)])]},
        "sglang:realtime_tokens": {
            "type": "counter",
            "series": [
                series("counter", [(0, 1, count), (1, 1.5, count / 2)], mode=mode)
                for mode, count in (("prefill_compute", 20), ("prefill_cache", 80), ("decode", 5))
            ],
        },
        "sglang:prefill_effective_tokens": {
            "type": "counter",
            "series": [
                series("counter", [(0, 1, count)], rank, mode)
                for mode, count in (("device_hit", 80), ("host_hit", 0), ("storage_hit", 0), ("input", 20))
                for rank in [0]
            ],
        },
    }
    return {
        "schema_version": "1.1",
        "metrics_phase": "profiling",
        "benchmark_id": "self-authored-fixture",
        "summary": {"phase_time_ranges": {"profiling": {"start_ns": ORIGIN, "end_ns": ORIGIN + 1_500_000_000}}},
        "metrics": metrics,
    }


class SiliconMetricsTests(unittest.TestCase):
    def load(self, data, point_id=446063):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps(data))
            return load_silicon_metrics(path, point_id)

    def test_tp_capacity_gaps_and_partial_counter_counts(self):
        result = self.load(fixture())
        profile = result["phases"]["profile"]
        self.assertEqual(profile["start_ns"], str(ORIGIN))
        self.assertEqual(profile["duration_s"], 1.5)
        rows = profile["series"]
        self.assertEqual([r["value"] for r in rows["cache_capacity"]], [100, 100])
        self.assertEqual(len(rows["cache_evictable"]), 1)
        self.assertEqual(rows["running"], [])
        self.assertEqual(rows["host"][0]["value"], 0)
        self.assertEqual(rows["scheduler_queries"][1], {"start": 1.0, "end": 1.5, "value": 50.0, "coverage": 1.0})
        self.assertEqual(rows["admission_input"][0]["value"], 100)
        self.assertEqual(len(result["provenance"]["source_sha256"]), 64)

    def test_counter_resets_negative_and_nonfinite_intervals_are_missing(self):
        data = fixture()
        stream = data["metrics"]["sglang:realtime_tokens"]["series"][0]
        stream["timeslices"][0]["counter_reset"] = True
        stream["timeslices"][1]["total"] = -1
        data["metrics"]["sglang:kv_used_tokens"]["series"][0]["timeslices"][0]["avg"] = float("nan")
        result = self.load(data)
        rows = result["phases"]["profile"]["series"]
        self.assertEqual(rows["prefill_tokens"], [])
        self.assertEqual(rows["scheduler_queries"], [])
        self.assertEqual(len(rows["cache_used"]), 1)
        self.assertTrue(any(d["reason"] == "counter_reset" for d in result["diagnostics"]))

    def test_tp_placeholders_and_observed_missing_host(self):
        data = fixture()
        counters = data["metrics"]["sglang:prefill_effective_tokens"]["series"]
        counters.append(series("counter", [(0, 1, 0)], 1, "device_hit"))
        counters[:] = [s for s in counters if s["labels"]["mode"] != "storage_hit"]
        rows = self.load(data)["phases"]["profile"]["series"]
        self.assertEqual(rows["hbm"][0]["value"], 80)
        self.assertEqual(rows["host"], [])
        self.assertEqual(rows["admission_input"], [])

    def test_distinct_dp_pools_sum_but_require_joint_coverage(self):
        data = fixture()
        capacity = data["metrics"]["sglang:max_total_num_tokens"]["series"]
        capacity.append(series("gauge", [(0, 1, 200)], dp=1))
        rows = self.load(data)["phases"]["profile"]["series"]["cache_capacity"]
        self.assertEqual(rows, [{"start": 0.0, "end": 1.0, "value": 300, "coverage": 1.0}])

    def test_misaligned_counter_buckets_preserve_common_coverage_counts(self):
        data = fixture()
        streams = data["metrics"]["sglang:realtime_tokens"]["series"]
        streams[1] = series("counter", [(0.5, 1.5, 100)], mode="prefill_cache")
        rows = self.load(data)["phases"]["profile"]["series"]
        self.assertEqual(rows["scheduler_hits"][0]["value"], 100)
        self.assertEqual(
            rows["scheduler_queries"],
            [
                {"start": 0.5, "end": 1.0, "value": 60.0, "coverage": 1.0},
                {"start": 1.0, "end": 1.5, "value": 60.0, "coverage": 1.0},
            ],
        )

    def test_conflicting_tp_replicas_fail(self):
        data = fixture()
        data["metrics"]["sglang:max_total_num_tokens"]["series"][1]["timeslices"][0]["avg"] = 200
        with self.assertRaisesRegex(ValueError, "Conflicting TP replicas"):
            self.load(data)

    def test_warmup_uses_its_exact_origin(self):
        data = fixture()
        data["summary"]["phase_time_ranges"]["warmup"] = {"start_ns": ORIGIN - 2_000_000_000, "end_ns": ORIGIN}
        data["warmup_metrics"] = {
            "sglang:num_running_reqs": {"type": "gauge", "series": [series("gauge", [(-2, -1, 0)])]}
        }
        rows = self.load(data)["phases"]["preparation"]["series"]["running"]
        self.assertEqual(rows[0], {"start": 0.0, "end": 1.0, "value": 0, "coverage": 1.0})

    def test_compact_roundtrip_wrong_point_version_and_interval(self):
        compact = self.load(fixture())
        self.assertEqual(compact["schema"], SCHEMA)
        self.assertEqual(self.load(compact), compact)
        with self.assertRaisesRegex(ValueError, "point_id"):
            self.load(compact, 438956)
        bad = copy.deepcopy(compact)
        bad["schema"] = "aisimulate.agentx.silicon.v2"
        with self.assertRaisesRegex(ValueError, "Unsupported silicon schema"):
            self.load(bad)
        bad = copy.deepcopy(compact)
        bad["phases"]["profile"]["series"]["cache_used"][0]["coverage"] = 2
        with self.assertRaisesRegex(ValueError, "Invalid compact interval"):
            self.load(bad)
        self.assertIsNone(load_silicon_metrics(None, 446063))


if __name__ == "__main__":
    unittest.main()
