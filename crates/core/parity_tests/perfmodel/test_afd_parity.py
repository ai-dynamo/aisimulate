# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-class AFD parity suite (continuous-batching rebuild).

Pins the continuous AFD replay path (SglangCore + injected ``afd``
timing provider) against the rebuilt goldens:

- `TestAfdE2eGolden` compares supported Qwen3-30B-A3B DP+EP replay cases against
  regenerated native goldens at 1e-6 relative tolerance. Coverage includes
  unsaturated and saturated admission, both AFD+PD roles, chunked prefill and
  the two-stage resource layout. Wall-clock fields are excluded.

Run from the repository root:

    python/aisimulate/.venv/bin/pytest -q crates/core/parity_tests/perfmodel/test_afd_parity.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "python/aisimulate/src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
AISIMULATE_CLI = _REPO_ROOT / "python/aisimulate/.venv/bin/aisimulate"

import afd_parity

E2E_CASES = (
    "qwen3-30b-a3b-h200-sglang-pure-both",
    "qwen3-30b-a3b-h200-sglang-pure-both-saturated",
    "qwen3-30b-a3b-h200-sglang-pd-decode",
    "qwen3-30b-a3b-h200-sglang-pd-prefill",
    "qwen3-30b-a3b-h200-sglang-chunked-prefill",
    "qwen3-30b-a3b-h200-sglang-two-stage",
)


def _golden(filename: str) -> dict:
    return json.loads((GOLDEN_DIR / filename).read_text())


def _run_native(config: Path, *overrides: str) -> dict:
    env = dict(os.environ)
    with tempfile.TemporaryDirectory(prefix="afd-parity-") as run_dir:
        subprocess.run(
            [
                str(AISIMULATE_CLI),
                "predict",
                "--stack",
                "engine",
                "--config",
                str(config),
                "--output-dir",
                run_dir,
                "--format",
                "json",
            ]
            + [argument for override in overrides for argument in ("--set", override)],
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        return json.loads((Path(run_dir) / "prediction.json").read_text())


@pytest.mark.parametrize(
    "config_name", ["afd-predict-pure", "afd-predict-plus-pd-decode", "afd-predict-plus-pd-prefill"]
)
def test_cached_prefix_native_replay_prices_only_the_uncached_suffix(config_name):
    config = GOLDEN_DIR.parent / "configs" / f"{config_name}.yaml"
    # Start cold and run serially: the first request populates the radix cache,
    # then the remaining three reuse the shared 512-token prefix.
    overrides = ("traffic.stop.requests=4", "traffic.load.concurrency=1")
    uncached = _run_native(config, *overrides)
    uncached = uncached.get("summary", uncached)
    cached = _run_native(config, *overrides, "traffic.source.cached_prefix_tokens=512")
    cached = cached.get("summary", cached)
    for summary in (uncached, cached):
        assert summary["completed_requests"] == summary["num_requests"] == 4
        assert summary["total_input_tokens"] == 4096
        assert summary["total_output_tokens"] == 512
    assert uncached["committed_prefill_tokens"] == 4096
    assert cached["committed_prefill_tokens"] == 1024 + 3 * 512
    assert cached["first_admission_prefix_cache_reused_ratio"] == pytest.approx(3 * 512 / 4096)
    assert cached["mean_ttft_ms"] < uncached["mean_ttft_ms"]


@pytest.mark.parametrize(
    "config_name", ["afd-predict-pure", "afd-predict-plus-pd-decode", "afd-predict-plus-pd-prefill"]
)
def test_kernel_selectors_native_cli_complete_in_all_afd_roles(config_name):
    config = GOLDEN_DIR.parent / "configs" / f"{config_name}.yaml"
    overrides = (
        "traffic.stop.requests=4",
        "traffic.load.concurrency=1",
        "engine.moe_kernel_source=sglang_fused_moe_triton",
    )
    fa3 = _run_native(config, *overrides, "engine.attention_backend=fa3")
    triton = _run_native(config, *overrides, "engine.attention_backend=triton")
    fa3, triton = fa3.get("summary", fa3), triton.get("summary", triton)
    for summary in (fa3, triton):
        assert summary["completed_requests"] == summary["num_requests"] == 4
        assert summary["total_input_tokens"] == 4096
        assert summary["total_output_tokens"] == 512


@pytest.mark.parametrize("lane", ["fa3", "triton"])
def test_native_afd_compilation_preserves_selectors_on_the_executed_ops(lane, monkeypatch):
    import yaml

    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate_core.sdk import engine

    compiled = []
    build = engine.build_engine_spec_json

    def capture(*args, **kwargs):
        raw = build(*args, **kwargs)
        compiled.append(json.loads(raw))
        return raw

    monkeypatch.setattr(engine, "build_engine_spec_json", capture)
    raw = yaml.safe_load((GOLDEN_DIR.parent / "configs" / "afd-predict-pure.yaml").read_text())
    source = "sglang_fused_moe_triton"
    raw["engine"].update(attention_backend=lane, moe_kernel_source=source)
    raw["traffic"]["stop"]["requests"] = 1
    raw["traffic"]["load"]["concurrency"] = 1
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    report = EngineReplayRunnerFactory().create(0).run(spec)
    assert report.metrics["completed_requests"] == 1
    assert len(compiled) == 2
    # A and F compile full model graphs, then AFD keeps each pool's partition.
    a, f = sorted(compiled, key=lambda item: item["engine"]["moe_ep_size"])
    for phase in ("context_ops", "generation_ops"):
        attention = [op[kind] for op in a[phase] for kind in ("ContextAttention", "GenerationAttention") if kind in op]
        moe = [op["Moe"] for op in f[phase] if "Moe" in op]
        assert attention and moe
        assert all(op["lane_order"][0] == lane for op in attention)
        assert all(op["moe_kernel_source"] == source for op in moe)


@pytest.mark.parametrize(
    "override,message",
    [
        ("engine.attention_backend=trtllm_mha", "attention_backend"),
        ("engine.moe_kernel_source=source_that_does_not_exist", "source_that_does_not_exist"),
    ],
)
def test_invalid_kernel_selectors_fail_in_native_model_construction(override, message):
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run_native(GOLDEN_DIR.parent / "configs" / "afd-predict-pure.yaml", override)
    assert message in error.value.stdout + error.value.stderr


@pytest.mark.parametrize("case_id", E2E_CASES)
class TestAfdE2eGolden:
    def test_native_run_matches_golden(self, case_id: str, tmp_path: Path):
        config = _REPO_ROOT / _golden("afd_e2e.json")["cases"][case_id]["config"]
        # The demo configs are committed parity assets: a missing file means
        # the repository is broken, not that the case is optional.
        assert config.is_file(), f"golden demo config missing: {config}"
        prediction = _run_native(config)
        actual_path = tmp_path / "prediction.json"
        actual_path.write_text(json.dumps(prediction))
        failures = afd_parity._check_e2e(case_id, actual_path)
        assert not failures, "AFD native e2e diverged from the golden:\n" + "\n".join(failures)
