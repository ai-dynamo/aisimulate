# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``python -m aisimulate.vl.collect``: measure one image workload's frontend stages into a host cost table.

The measurement itself runs in the serving host's SGLang interpreter
(``--sglang-python``, usually another virtual environment); this command
lowers its recording into one table row and saves the table. Nothing here
needs a GPU. The workload can be read from a prediction YAML (``--config``) so
the table row matches what ``aisimulate predict`` will look up, and a saved
recording (``--recording``, for example a ``.failed-*.json``) can be lowered
again without measuring.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ...config.common import load_yaml
from ...config.engine import FrontendMeasurementConfig, feature_transport
from ..table import SGLANG_VERSION, load_table, update_table
from .lower import STAGE_LABELS, environment, frontend_row

WORKER = Path(__file__).resolve().parent / "_sglang" / "frontend_worker.py"


def _images(value: str) -> tuple[int, int, int]:
    parts = value.lower().split("x")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("images must be HEIGHTxWIDTH or HEIGHTxWIDTHxCOUNT")
    height, width = int(parts[0]), int(parts[1])
    count = int(parts[2]) if len(parts) == 3 else 1
    return height, width, count


def _show(path: Path) -> int:
    table = load_table(path)
    env = table.environment
    print(f"{path}: {env.cpu}, sglang {env.sglang_version}, python {env.python}")
    for row in table.rows:
        labels = STAGE_LABELS[row.measured_for.frontend]
        stages = ", ".join(
            f"{label}({stage.workers}) {stage.service_ms:.2f} ms"
            for label, stage in zip(labels, row.stages, strict=True)
        )
        print(f"  {row.measured_for.describe():70s} | {stages} | {table.digest(row)}")
    return 0


def _from_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Fill unset arguments from a prediction YAML's host-aware worker (aggregated or prefill) and image workload."""
    raw = load_yaml(args.config)
    try:
        workers = raw["engine"]["workers"]
        worker = workers.get("aggregated") or workers["prefill"]
        images = raw["traffic"]["source"]["images"]
    except (KeyError, TypeError, AttributeError):
        parser.error(f"{args.config} has no aggregated or prefill worker with an image workload")
    profile = worker.get("host_profile") or {}
    if not profile and (args.table is None or args.frontend is None):
        parser.error(f"{args.config} has no host_profile; pass --table and --frontend explicitly")
    tensor = (worker.get("parallelism") or {}).get("tensor", 1)
    defaults = {
        "table": profile.get("path"),
        "model": raw.get("engine", {}).get("model"),
        "frontend": profile.get("frontend"),
        "images": (int(images["height"]), int(images["width"]), int(images.get("count", 1))),
        "encoding": images.get("encoding", "png"),
        "min_pixels": images.get("min_pixels"),
        "max_pixels": images.get("max_pixels"),
        "tp": tensor if isinstance(tensor, int) else None,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.tp is None:
        parser.error(f"{args.config} searches parallelism.tensor; pass --tp for the width to measure")


def _check_sglang(python: str) -> None:
    probe = subprocess.run(
        [python, "-c", "import sglang.version as v; print(v.__version__)"], capture_output=True, text=True, check=False
    )
    if probe.returncode:
        raise SystemExit(
            f"{python} has no importable sglang; pass --sglang-python <serving venv python "
            f"with sglang {SGLANG_VERSION}>"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="aisimulate.vl.collect", description=__doc__)
    parser.add_argument("--config", help="prediction YAML whose aggregated worker and images define the workload")
    parser.add_argument("--recording", help="lower this saved worker recording instead of measuring")
    parser.add_argument("--table", help="host cost table to create or extend (default: host_profile.path of --config)")
    parser.add_argument("--show", action="store_true", help="print the table's rows and exit")
    parser.add_argument("--model", help="model identifier as written in engine.model")
    parser.add_argument("--frontend", choices=("python", "rust"))
    parser.add_argument("--tp", type=int, help="tensor-parallel width the frontend serves (Rust: inline below 2)")
    parser.add_argument("--images", type=_images, metavar="HxW[xN]")
    parser.add_argument("--encoding", choices=("png", "jpeg"))
    parser.add_argument("--min-pixels", type=int, help="processor rescale floor; default: the checkpoint's")
    parser.add_argument("--max-pixels", type=int, help="processor rescale ceiling; default: the checkpoint's")
    parser.add_argument("--text-tokens", type=int, default=128, help="text tokens beside the images; recorded only")
    parser.add_argument("--sglang-python", default=sys.executable, help="interpreter with sglang installed")
    args = parser.parse_args(argv)
    if args.config is not None:
        _from_config(args, parser)
    if args.table is None:
        parser.error("--table is required")
    path = Path(args.table)
    if args.show:
        return _show(path)
    for name in ("model", "frontend", "images"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required")
    encoding = args.encoding or "png"
    tp = args.tp or 1
    height, width, count = args.images
    measurement = FrontendMeasurementConfig(
        model=args.model,
        frontend=args.frontend,
        feature_transport=feature_transport(args.frontend, tp),
        height=height,
        width=width,
        count=count,
        encoding=encoding,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    if args.recording is not None:
        recording = json.loads(Path(args.recording).read_text())
    else:
        recording = _measure(args, measurement)
    try:
        row = frontend_row(recording, measurement)
    except ValueError as exc:
        failed = path.with_name(f"{path.name}.failed-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.write_text(json.dumps(recording, indent=1))
        raise SystemExit(f"{exc}\nThe raw recording was kept at {failed}; lower it again with --recording") from exc
    table = update_table(path, environment(recording), row)
    labels = STAGE_LABELS[args.frontend]
    stages = "; ".join(
        f"{label}({stage.workers}) {stage.service_ms:.2f} ms" for label, stage in zip(labels, row.stages, strict=True)
    )
    print(f"{path}: {measurement.describe()} -> {stages} [{table.digest(row)}]")
    return 0


def _measure(args: argparse.Namespace, measurement: FrontendMeasurementConfig) -> dict:
    """Run the worker in the serving interpreter and return its recording."""
    _check_sglang(args.sglang_python)
    command = [
        args.sglang_python,
        str(WORKER),
        "--model",
        args.model,
        "--frontend",
        args.frontend,
        "--height",
        str(measurement.height),
        "--width",
        str(measurement.width),
        "--count",
        str(measurement.count),
        "--encoding",
        measurement.encoding,
        "--text-tokens",
        str(args.text_tokens),
        "--tp",
        str(args.tp or 1),
    ]
    for name, value in (("--min-pixels", measurement.min_pixels), ("--max-pixels", measurement.max_pixels)):
        if value is not None:
            command += [name, str(value)]
    completed = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    lines = completed.stdout.splitlines()
    if not lines:
        raise SystemExit("the worker exited without printing a recording")
    return json.loads(lines[-1])


if __name__ == "__main__":
    raise SystemExit(main())
