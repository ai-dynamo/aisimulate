# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an AFD replay across closed-loop and open-loop load levels.

All cases use current per-pass batch/context pricing.
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
AISIMULATE_CLI = [sys.executable, "-m", "aisimulate"]

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
            [*AISIMULATE_CLI, "predict", "--config", str(config_path), "--output-dir", run_dir, "--format", "json"],
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        prediction = json.loads((Path(run_dir) / "prediction.json").read_text())
    return prediction.get("summary", prediction)


def _variant(base: dict, label: str, load: dict | None) -> Path:
    config = json.loads(json.dumps(base))
    if load is not None:
        config["traffic"]["load"] = load
    fd, name = tempfile.mkstemp(prefix=f"afd-caliber-{label}-", suffix=".yaml")
    path = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            yaml.safe_dump(config, handle)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="base AFD predict config (YAML)")
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text())
    results = {}
    for label, load in [("closed-loop", None), *OPEN_LOOP_LOADS]:
        path = _variant(base, label, load)
        try:
            summary = _run(path)
            results[label] = {metric: summary.get(metric) for metric in METRICS}
        finally:
            path.unlink(missing_ok=True)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
