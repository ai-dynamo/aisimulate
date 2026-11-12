# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FPM parsing must finish before resource supervision loads the runtime."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _config() -> dict:
    return {
        "engine": {
            "mode": "aggregated",
            "model": "test/unregistered-decoder",
            "hardware": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.25.1",
            "workers": {"aggregated": {"scheduler": {"max_batched_tokens": 1024, "max_sequences": 4}}},
            "fpm_profile": {
                "schema_version": 1,
                "model": "test/unregistered-decoder",
                "model_revision": "import-boundary-fixture-v1",
                "architecture": "UnregisteredDecoderForCausalLM",
                "context_length": 4096,
                "num_experts": 0,
                "provenance": "Synthetic metadata for pre-budget validation; no timing measurements.",
                "deployments": [
                    {
                        "system": "h200_sxm",
                        "backend": "vllm",
                        "backend_version": "0.25.1",
                        "tp": 1,
                        "dp": 1,
                        "moe_tp": 1,
                        "moe_ep": 1,
                        "gemm_quant_mode": "fp8",
                        "moe_quant_mode": "fp8",
                        "fmha_quant_mode": "fp8",
                        "comm_quant_mode": "half",
                        "kv_cache_dtype": "fp8",
                        "resources": {
                            "weights_bytes": 100,
                            "activations_bytes": 20,
                            "runtime_overhead_bytes": 30,
                            "comm_overhead_bytes": 50,
                            "kv_bytes_per_token": 10,
                            "cache_layout": "linear",
                            "max_num_tokens": 8192,
                            "max_batch_size": 256,
                            "provenance": "Declared bounds for import validation only.",
                        },
                    }
                ],
            },
        }
    }


_IMPORT_GUARD = """
import sys
from pathlib import Path

blocked = ('aisimulate._runtime', 'aisimulate.sweeper', 'numpy', 'pandas', 'aiconfigurator',
           'aiconfigurator_core', 'aisimulate_core._native', 'aisimulate_core.sdk',
           'urllib.request', 'huggingface_hub', 'transformers')

def assert_lightweight():
    loaded = [name for name in sys.modules if any(name == item or name.startswith(item + '.') for item in blocked)]
    assert not loaded, loaded
    core = {name for name in sys.modules if name == 'aisimulate_core' or name.startswith('aisimulate_core.')}
    allowed = {'aisimulate_core', 'aisimulate_core.ais_perf_model', 'aisimulate_core.fpm_profile',
               'aisimulate_core.quantization'}
    assert core <= allowed, core

class RejectRuntimeImports:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == item or fullname.startswith(item + '.') for item in blocked):
            raise AssertionError('pre-budget runtime import: ' + fullname)

assert_lightweight()
sys.meta_path.insert(0, RejectRuntimeImports())
"""

_SUPERVISION_PROBE = (
    _IMPORT_GUARD
    + """
from aisimulate import supervision

calls = []
def supervised_child(*args, **kwargs):
    assert_lightweight()
    calls.append(args)
    return {'status': 'completed', 'exit_code': 0}

supervision.run_process = supervised_child
config, output, command, error = sys.argv[1:]
try:
    code = supervision.main([command, '--config', config, '--output-dir', output, '--overwrite'])
except SystemExit as exc:
    code = exc.code
assert_lightweight()
if error:
    assert code == 2, code
    assert not calls, calls
    assert (Path(output) / 'existing.txt').read_text() == 'keep existing results'
else:
    assert code == 0, code
    assert len(calls) == 1, calls
    from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
    config_type = CorePredictionConfig if command == 'predict' else CoreRecommendationConfig
    parsed = config_type.from_yaml(config)
    again = config_type.model_validate_json(parsed.model_dump_json())
    if parsed.engine.ais_perf_model is None:
        assert again == parsed
    else:
        # A dump writes what the key means and leaves the key out, so the two configs are compared by their dumps.
        assert again.model_dump_json() == parsed.model_dump_json()
    assert_lightweight()
"""
)


def _add_old_transfer_policy(engine: dict, case: str) -> str:
    """Write the key and an old ``transfer_policy`` into an engine; return the error to expect, or no text."""
    del engine["fpm_profile"]
    repeated = case.startswith("repeated")
    old = {"transfer_policy": ["xshape"] if repeated else ["xquant"]}
    if "worker" in case:
        # Only the decode worker gets a policy from the key, so the key is compared with the timing of that worker.
        engine["mode"] = "disaggregated"
        worker = engine["workers"].pop("aggregated")
        engine["workers"] = {"prefill": worker, "decode": {**worker, "timing": old}}
        engine["ais_perf_model"] = [{"worker_type": "prefill"}, {"worker_type": "decode", "transfer_policy": "xshape"}]
        path = "engine.workers.decode.timing.transfer_policy"
    else:
        engine.update(old)
        engine["ais_perf_model"] = {"transfer_policy": "xshape"}
        path = "engine.transfer_policy"
    return "" if repeated else f"old estimator fields that say something else: {path}. Remove the old fields."


@pytest.mark.parametrize("command", ["predict", "recommend"])
@pytest.mark.parametrize(
    "case",
    [
        "no-profile",
        "profile",
        "mixed-profile",
        "unknown-quant",
        "boolean-topology",
        "wrong-model",
        "misspelled-perf-model-key",
        "repeated-transfer-policy",
        "conflicting-transfer-policy",
        "repeated-worker-transfer-policy",
        "conflicting-worker-transfer-policy",
    ],
)
def test_supervised_profile_validation_is_runtime_free(command: str, case: str, tmp_path: Path) -> None:
    """Exercise the real parser and envelope validation before the child starts."""
    raw = _config()
    if command == "recommend":
        raw["optimization"] = {"constraints": {"max_candidate_gpus": 1}}
    error = ""
    if case == "no-profile":
        del raw["engine"]["fpm_profile"]
    elif case == "mixed-profile":
        grouped = deepcopy(raw["engine"]["fpm_profile"]["deployments"][0])
        grouped["tp"] = 2
        grouped["resources"]["cache_layout"] = "grouped"
        grouped["resources"].pop("kv_bytes_per_token")
        grouped["resources"]["cache_groups"] = [
            {"name": "full", "kind": "attention", "num_layers": 1, "block_size_tokens": 64, "page_size_bytes": 64}
        ]
        raw["engine"]["fpm_profile"]["deployments"].append(grouped)
    elif case == "unknown-quant":
        raw["engine"]["fpm_profile"]["deployments"][0]["fmha_quant_mode"] = "unsupported"
        error = "unknown FPM fmha_quant_mode"
    elif case == "boolean-topology":
        raw["engine"]["fpm_profile"]["deployments"][0]["tp"] = True
        error = "Input should be a valid integer"
    elif case == "wrong-model":
        raw["engine"]["model"] = "test/different-decoder"
        error = "engine.model must match engine.fpm_profile.model"
    elif case == "misspelled-perf-model-key":
        # Telling a perf model field from a misspelled key must not load the perf model itself.
        raw["engine"]["ais_perf_model"] = {"fpm_profile": raw["engine"].pop("fpm_profile"), "estimaton_mode": "auto"}
        error = "estimaton_mode: not a perf model field"
        if command == "recommend":
            error = "estimaton_mode cannot be checked in a recommend config"
    elif case.endswith("transfer-policy"):
        # Comparing an old transfer_policy with the key must not load the perf model to read the policy.
        error = _add_old_transfer_policy(raw["engine"], case)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw))
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("keep existing results")

    result = subprocess.run(
        [sys.executable, "-c", _SUPERVISION_PROBE, str(config), str(output), command, error],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if error:
        assert error in result.stderr


def test_onboarding_parser_is_runtime_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _IMPORT_GUARD
            + """
from aisimulate.cli_args import build_parser
parser = build_parser()
try:
    parser.parse_args(['onboard', 'init', '--help'])
except SystemExit as exc:
    assert exc.code == 0
assert_lightweight()
""",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--model" in result.stdout
