"""Compile baseline configs through the installed AIS/Dynamo SDK without replay.

Run in the WSL dynosim-src environment with:
    python -m unittest discover -s tests -p test_sdk_contract.py -v

These tests use the real schemas, adapter discovery, compiler and runner
capabilities, then materialize the same launch arguments as Dynamo. This loads
performance data and resolves KV capacity without querying forward latency or
running Replay. Both native extensions and the checked-in performance data are
required.
Missing SDK imports skip this suite on non-Linux development hosts; on Linux,
they fail so an incomplete deployment cannot silently pass the regression.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml


BASE = Path(__file__).resolve().parents[1]
CASES = ("agg1", "agg4", "p1d3", "p2d2", "p3d1")
UNSUPPORTED_PRECISION_ERROR = (
    "runner does not support engine model controls "
    "['kvcache_quant_mode', 'fmha_quant_mode']; use --stack engine"
)


class SDKContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            # Verify both native packages exist; do not replace them with stubs.
            from aisimulate import _runtime  # noqa: F401
            from dynamo import _core  # noqa: F401

            from aisimulate.compiler import prediction_to_replay_spec
            from aisimulate.config.cli import CorePredictionConfig
            from aisimulate.config.common import split_config_sections
            from aisimulate.config_adapter import (
                PredictionAdapterContext,
                resolve_config_adapters,
            )
            from aisimulate.runner import materialize_engine_launch_config
            from aisimulate.stack import resolve_runner_factory
            from dynamo.mocker.config import normalize_mocker_config
        except ImportError as exc:
            if not sys.platform.startswith("linux"):
                raise unittest.SkipTest(
                    f"Full AIS/Dynamo SDK is unavailable on this host: {exc}. "
                    "Run this contract suite in the WSL dynosim-src environment."
                ) from exc
            raise

        cls.sdk = {
            "compile": prediction_to_replay_spec,
            "schema": CorePredictionConfig,
            "split": split_config_sections,
            "adapter_context": PredictionAdapterContext,
            "adapters": resolve_config_adapters,
            "materialize_launch": materialize_engine_launch_config,
            "normalize_mocker": normalize_mocker_config,
        }
        cls.factory = resolve_runner_factory("dynamo")
        temporary = tempfile.TemporaryDirectory(prefix="pd-unified-sdk-contract-")
        cls.addClassCleanup(temporary.cleanup)
        cls.trace = Path(temporary.name) / "requests.jsonl"
        cls.trace.write_text(
            json.dumps({
                "timestamp": 0,
                "input_length": 32,
                "output_length": 4,
                "hash_ids": [1],
            }) + "\n",
            encoding="utf-8",
        )

    def compile_case(
        self,
        case: str,
        *,
        explicit_bf16: bool = False,
        backend_version: str | None = None,
    ):
        raw = yaml.safe_load(
            (BASE / "configs" / f"{case}.yaml").read_text(encoding="utf-8")
        )
        raw["traffic"]["source"]["paths"] = [str(self.trace)]
        if backend_version is not None:
            raw["engine"]["backend_version"] = backend_version
        if explicit_bf16:
            raw["engine"]["fmha_quant_mode"] = "bfloat16"
            raw["engine"]["kvcache_quant_mode"] = "bfloat16"
        core_raw, adapter_raw = self.sdk["split"](raw, command="predict")
        config = self.sdk["schema"].model_validate(core_raw)
        context = self.sdk["adapter_context"](
            engine=config.engine.model_dump(mode="json", exclude_none=True),
            traffic=config.traffic.model_dump(mode="json", exclude_none=True),
            evaluation=config.evaluation.model_dump(mode="json", exclude_none=True),
        )
        adapters = self.sdk["adapters"](
            f"dynamo.{section}" for section in adapter_raw
        )
        adapter_specs = {
            f"dynamo.{section}": adapters[f"dynamo.{section}"].compile_prediction(
                value, context
            )
            for section, value in adapter_raw.items()
        }
        self.assertIn("dynamo.router", adapter_specs)
        return self.sdk["compile"](
            config, adapter_specs=adapter_specs, execution_mode="offline"
        )

    def materialize_role(self, deployment, role: str):
        field = "agg_engine_args" if role == "aggregated" else f"{role}_engine_args"
        return self.sdk["normalize_mocker"](
            self.sdk["materialize_launch"](
                deployment.backend,
                deployment.backend_version,
                deployment.parallel_config,
                getattr(deployment, field),
                role,
            )
        )

    def test_all_baselines_compile_and_match_dynamo_capabilities(self):
        for case in CASES:
            with self.subTest(case=case):
                spec = self.compile_case(case)
                self.factory.capabilities().require_compatible(spec)
                deployment = spec.backend_deployment
                if case.startswith("p"):
                    for role in ("prefill", "decode"):
                        args = getattr(deployment, f"{role}_engine_args")
                        self.assertEqual(args["kv_transfer_bytes_per_token"], 131072)

    def test_all_baselines_materialize_performance_data_and_kv_capacity(self):
        for case in CASES:
            spec = self.compile_case(case)
            self.factory.capabilities().require_compatible(spec)
            deployment = spec.backend_deployment
            self.assertEqual(deployment.backend_version, "0.24.0")
            roles = (
                ("aggregated",)
                if deployment.deployment_mode == "agg"
                else ("prefill", "decode")
            )
            for role in roles:
                with self.subTest(case=case, role=role):
                    launch = self.materialize_role(deployment, role)
                    engine = launch["engine"]
                    self.assertGreater(engine["num_gpu_blocks"], 0)
                    self.assertEqual(engine["block_size"], 64)
                    self.assertEqual(engine["worker_type"], role)
                    self.assertEqual(
                        engine["timing_model"]["config"]["backend_version"],
                        deployment.backend_version,
                    )
                    if role != "aggregated":
                        self.assertEqual(engine["kv_transfer_bytes_per_token"], 131072)

    def test_unlisted_version_passes_capabilities_but_fails_materialization(self):
        # Keep the release database guard enabled even if the developer shell
        # has opted into raw, unmaintained data coordinates.
        with patch.dict(os.environ, {"AIC_ALLOW_UNLISTED_VERSIONS": "0"}):
            spec = self.compile_case("agg1", backend_version="0.25.0")
            self.factory.capabilities().require_compatible(spec)
            with self.assertRaisesRegex(ValueError, "old-style raw version query"):
                self.materialize_role(spec.backend_deployment, "aggregated")

    def test_explicit_bf16_overrides_reproduce_runner_rejection(self):
        for case in CASES:
            with self.subTest(case=case):
                spec = self.compile_case(case, explicit_bf16=True)
                with self.assertRaisesRegex(
                    ValueError, re.escape(UNSUPPORTED_PRECISION_ERROR)
                ):
                    self.factory.capabilities().require_compatible(spec)


if __name__ == "__main__":
    unittest.main()
