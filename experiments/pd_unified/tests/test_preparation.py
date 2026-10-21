"""Data-selection tests; no AIS/Dynamo native libraries or simulation required."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("baseline_preparation", BASE / "run_baseline.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def row(timestamp, isl=16, osl=2):
    return {"timestamp": timestamp, "input_length": isl, "output_length": osl, "hash_ids": [1]}


class PreparationTests(unittest.TestCase):
    def test_request_limit_keeps_all_simultaneous_arrivals(self):
        selected, report = runner.select_rows([row(0), row(10), row(10), row(20)], 2, 128, "filter")
        self.assertEqual([r["timestamp"] for r in selected], [0, 10, 10])
        self.assertEqual(report["selected_request_count"], 3)

    def test_context_filter_preserves_original_gaps_and_rejects_prompt_at_limit(self):
        rows = [row(0), row(10, 128, 0), row(15, 120, 10), row(20, 120, 8)]
        selected, report = runner.select_rows(rows, 0, 128, "filter")
        self.assertEqual([r["timestamp"] for r in selected], [0, 20])
        self.assertEqual(report["excluded_for_context"], 2)
        self.assertEqual(selected[-1]["output_length"], 8)

    def test_strict_context_and_out_of_order_input_fail_explicitly(self):
        with self.assertRaises(ValueError):
            runner.select_rows([row(0, 120, 10)], 0, 128, "error")
        with self.assertRaises(ValueError):
            runner.select_rows([row(2), row(1)], 0, 128, "filter")

    def test_derivative_is_reused_and_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "Mooncake_FAST25"
            source = dataset / "upstream/FAST25-release/traces/conversation_trace.jsonl"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(row(0)) + "\n", encoding="utf-8")
            original = source.read_bytes()
            manifest = {"revision": "test", "files": [{
                "local_path": source.relative_to(dataset).as_posix(),
                "sha256": hashlib.sha256(original).hexdigest(), "size": len(original),
            }]}
            (dataset / "upstream_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            first, meta = runner.prepare_trace(root, "conversation", 0, 128, "filter")
            second, _ = runner.prepare_trace(root, "conversation", 0, 128, "filter")
            self.assertEqual(first, second)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(meta["selection"]["selected_request_count"], 1)
            first.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "derived trace was modified"):
                runner.prepare_trace(root, "conversation", 0, 128, "filter")

    def test_equal_gpu_budget_and_no_fixed_timing(self):
        for case in runner.CASES:
            config = runner.yaml.safe_load((BASE / "configs" / f"{case}.yaml").read_text(encoding="utf-8"))
            runner.validate_experiment_contract(config, case)
        config["engine"]["workers"]["prefill"]["timing"] = {"type": "fixed", "prefill_ms": 1}
        with self.assertRaisesRegex(ValueError, "fixed/polynomial"):
            runner.validate_experiment_contract(config, case)

    @patch.dict("os.environ", {"AIC_ALLOW_UNLISTED_VERSIONS": "0"})
    def test_baselines_pin_the_maintained_current_query_version(self):
        # Read the packaged contract without importing the native performance
        # model. A raw data directory or floating alias is not our baseline pin.
        slots_path = (
            BASE.parents[1] / "python/aisimulate/src/aisimulate_core/systems/query_versions.yaml"
        )
        slots = runner.yaml.safe_load(slots_path.read_text(encoding="utf-8"))
        for case in runner.CASES:
            with self.subTest(case=case):
                config = runner.yaml.safe_load((BASE / "configs" / f"{case}.yaml").read_text(encoding="utf-8"))
                engine = config["engine"]
                backend = engine["backend"]
                override = slots.get("overrides", {}).get(engine["hardware"], {}).get(backend)
                entry = override if override is not None else slots["defaults"][backend]
                current = str(entry["current"])
                self.assertIsInstance(engine["backend_version"], str)
                self.assertEqual(
                    engine["backend_version"], current,
                    f"{case}: pin the maintained current version {current}; "
                    "do not query a raw data version or a floating alias",
                )

    def test_default_fp8_kv_geometry_matches_pd_transfer_bytes(self):
        # The pinned Qwen/vLLM defaults use 1-byte FP8 KV. Independently derive
        # transfer size from checkpoint geometry, rather than torch_dtype (BF16).
        model_file = (
            BASE.parents[1] / "python/aisimulate/src/aisimulate_core/model_configs"
            / "Qwen--Qwen3-32B-FP8_config.json"
        )
        model = json.loads(model_file.read_text(encoding="utf-8"))
        fp8_bytes = 2 * model["num_hidden_layers"] * model["num_key_value_heads"] * model["head_dim"]
        for case in runner.CASES:
            with self.subTest(case=case):
                config = runner.yaml.safe_load((BASE / "configs" / f"{case}.yaml").read_text(encoding="utf-8"))
                engine = config["engine"]
                self.assertEqual(engine["model"], "Qwen/Qwen3-32B-FP8")
                for field in ("kvcache_quant_mode", "fmha_quant_mode"):
                    self.assertNotIn(field, engine)
                    for worker in engine["workers"].values():
                        self.assertNotIn(field, worker["timing"])
                if engine["mode"] == "disaggregated":
                    for worker in engine["workers"].values():
                        self.assertEqual(worker["parallelism"]["tensor"], 1)
                        self.assertEqual(worker["parallelism"]["pipeline"], 1)
                    self.assertEqual(engine["kv_transfer"]["bytes_per_token"], fp8_bytes)


if __name__ == "__main__":
    unittest.main()
