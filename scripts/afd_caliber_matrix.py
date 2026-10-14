# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caliber matrix: frozen decode identity vs full instantaneous pricing.

Runs one AFD config in both pricing calibers across open-loop (poisson /
constant_rate) load rates and the closed-loop steady state, then reports the
per-metric deviation. Each YAML pins
`engine.estimator_config.afd.frozen_decode_identity`, which is saved in replay identity.

Usage:
    python3 scripts/afd_caliber_matrix.py \
        --config crates/core/parity_tests/perfmodel/configs/afd-predict-pure.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
AISIMULATE_CLI = _REPO_ROOT / "python/aisimulate/.venv/bin/aisimulate"

METRICS = (
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_e2e_latency_ms",
    "output_throughput_tok_s",
    "duration_ms",
    "completed_requests",
)

# (label, load override). Rates bracket the pure golden's closed-loop service
# rate (~4.3 rps at concurrency 8): below, near, and above saturation.
OPEN_LOOP_LOADS = (
    ("poisson-1rps", {"type": "poisson", "requests_per_second": 1.0, "seed": 17}),
    ("poisson-3rps", {"type": "poisson", "requests_per_second": 3.0, "seed": 17}),
    ("poisson-6rps", {"type": "poisson", "requests_per_second": 6.0, "seed": 17}),
    ("constant-3rps", {"type": "constant_rate", "requests_per_second": 3.0}),
)


def _run(config_path: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="afd-caliber-") as run_dir:
        subprocess.run(
            [str(AISIMULATE_CLI), "predict", "--config", str(config_path), "--output-dir", run_dir, "--format", "json"],
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        prediction = json.loads((Path(run_dir) / "prediction.json").read_text())
    return prediction.get("summary", prediction)


def _variant(base: dict, label: str, load: dict | None, *, frozen: bool) -> Path:
    config = json.loads(json.dumps(base))
    if load is not None:
        config["traffic"]["load"] = load
    estimator = config.setdefault("engine", {}).setdefault("estimator_config", {})
    if estimator.get("afd") is None:
        estimator["afd"] = {}
    estimator["afd"]["frozen_decode_identity"] = frozen
    fd, name = tempfile.mkstemp(prefix=f"afd-caliber-{label}-", suffix=".yaml")
    path = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            yaml.safe_dump(config, handle)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _deviation(frozen: float, instantaneous: float) -> float:
    if frozen == instantaneous:
        return 0.0
    if frozen == 0:
        return float("inf")
    return abs(instantaneous - frozen) / abs(frozen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="base AFD predict config (YAML)")
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text())

    cases = [("closed-loop", None), *OPEN_LOOP_LOADS]
    rows: list[tuple[str, str, float, float, float]] = []
    for label, load in cases:
        config_paths = []
        try:
            for frozen in (True, False):
                config_path = _variant(base, label, load, frozen=frozen)
                config_paths.append(config_path)
                summary = _run(config_path)
                if frozen:
                    frozen_summary = summary
                else:
                    instantaneous_summary = summary
        finally:
            for config_path in config_paths:
                config_path.unlink(missing_ok=True)
        for metric in METRICS:
            frozen_value = float(frozen_summary.get(metric) or 0.0)
            instantaneous_value = float(instantaneous_summary.get(metric) or 0.0)
            deviation = _deviation(frozen_value, instantaneous_value)
            rows.append((label, metric, frozen_value, instantaneous_value, deviation))

    print(f"{'case':<16} {'metric':<28} {'frozen':>14} {'instantaneous':>14} {'dev':>9}")
    worst = 0.0
    for label, metric, frozen_value, instantaneous_value, deviation in rows:
        print(f"{label:<16} {metric:<28} {frozen_value:>14.4f} {instantaneous_value:>14.4f} {deviation:>8.2%}")
        if metric != "completed_requests":
            worst = max(worst, deviation)
    print(f"\nmax deviation (excluding completed_requests): {worst:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
