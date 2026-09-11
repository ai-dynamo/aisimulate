#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise public recommend -> predict using the exact installed wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml


def verify(wheel: Path) -> dict:
    distribution = importlib.metadata.distribution("aisimulate")
    installed_root = Path(distribution.locate_file("")).resolve()
    with zipfile.ZipFile(wheel) as archive:
        for name in ("aisimulate", "aisimulate._runtime", "aisimulate_core", "aiconfigurator", "aiconfigurator_core"):
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            relative = path.relative_to(installed_root).as_posix()
            if path.read_bytes() != archive.read(relative):
                raise RuntimeError(f"loaded {name} does not match the wheel")

    fixture = Path(__file__).resolve().parents[1] / "tests/e2e/configs/unified_cli"
    with tempfile.TemporaryDirectory(prefix="aisim-installed-cli-") as directory:
        root = Path(directory)
        shutil.copytree(fixture / "fixtures/tiny-model", root / "model")
        config = yaml.safe_load((fixture / "recommend/engine/03-preset-off-ttft.yaml").read_text())
        config["engine"]["model"] = str(root / "model")
        config["optimizer"].update(max_trials=2, parallelism=1)
        config_path = root / "recommend.yaml"
        config_path.write_text(yaml.safe_dump(config))

        def run(*arguments: str) -> object:
            completed = subprocess.run(
                [sys.executable, "-I", "-m", "aisimulate", *arguments, "--format", "json"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"installed CLI failed ({completed.returncode}): {completed.stdout}\n{completed.stderr}"
                )
            return json.loads(completed.stdout)

        rows = run(
            "recommend",
            "--stack",
            "engine",
            "--config",
            str(config_path),
            "--output-dir",
            str(root / "recommendations"),
        )
        outputs = sorted((root / "recommendations/recommendations").glob("*.yaml"))
        if not rows or len(outputs) != len(rows):
            raise RuntimeError("recommendation rows and runnable configurations disagree")
        counts = []
        for index, output in enumerate(outputs):
            summary = run(
                "predict",
                "--stack",
                "engine",
                "--config",
                str(output),
                "--output-dir",
                str(root / f"prediction-{index}"),
            )
            saved = json.loads((root / f"prediction-{index}/prediction.json").read_text())
            saved = saved.get("summary", saved)
            if summary["completed_requests"] != 6 or saved["completed_requests"] != 6:
                raise RuntimeError("installed replay did not complete the six requested requests")
            counts.append(summary["completed_requests"])
    return {
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "version": distribution.version,
        "recommendations": len(outputs),
        "completed_requests": counts,
        "qualification": "installed_public_cli_round_trip_with_fixed_timing",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.wheel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
