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

from ...config.common import _absolute_path, load_yaml
from ...config.engine import FrontendMeasurementConfig, FrontendStageConfig, feature_transport
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


# Provenance `--show` prints per row, so a re-collection that replaced a row is visible.
PROVENANCE_SHOWN = ("host", "threads", "image_processor_backend", "processor_workers", "sampled_at")
LIBRARY_VERSIONS_SHOWN = ("transformers", "torch")


def _stages(frontend: str, stages: list[FrontendStageConfig]) -> str:
    """Stages labeled by `STAGE_LABELS`, or by position when a row holds another number of them."""
    labels = STAGE_LABELS[frontend]
    if len(stages) != len(labels):
        labels = [f"stage{i}" for i in range(1, len(stages) + 1)]
    return ", ".join(
        f"{label}({stage.workers}) {stage.service_ms:.2f} ms" for label, stage in zip(labels, stages, strict=True)
    )


def _show(path: Path) -> int:
    try:
        table = load_table(path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    env = table.environment
    print(f"{path}: {env.cpu}, sglang {env.sglang_version}, python {env.python}")
    for row in table.rows:
        stages = _stages(row.measured_for.frontend, row.stages)
        print(f"  {row.measured_for.describe():70s} | {stages} | {table.digest(row)}")
        for keys in (PROVENANCE_SHOWN, LIBRARY_VERSIONS_SHOWN):
            shown = ", ".join(f"{key} {row.provenance[key]}" for key in keys if key in row.provenance)
            if shown:
                print(f"    {shown}")
    return 0


def _from_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Fill unset arguments from a prediction YAML's host profile and image workload.

    The profile sits on the host-aware language worker (aggregated or prefill) or,
    for a native encoder pool, on `workers.encoder`.
    """
    raw = load_yaml(args.config)
    try:
        workers = raw["engine"]["workers"]
        worker = workers.get("aggregated") or workers.get("prefill") or {}
        images = raw["traffic"]["source"]["images"]
    except (KeyError, TypeError, AttributeError):
        parser.error(f"{args.config} has no language worker with an image workload")
    encoder = workers.get("encoder") or {}
    if not worker.get("host_profile") and encoder.get("host_profile"):
        worker = {"host_profile": encoder["host_profile"]}
    profile = worker.get("host_profile") or {}
    if not profile and (args.table is None or args.frontend is None):
        parser.error(f"{args.config} has no host_profile; pass --table and --frontend explicitly")
    defaults = {
        "table": profile.get("path"),
        "model": raw.get("engine", {}).get("model"),
        "frontend": profile.get("frontend"),
        "images": (int(images["height"]), int(images["width"]), int(images.get("count", 1))),
        "encoding": images.get("encoding", "png"),
        "min_pixels": images.get("min_pixels"),
        "max_pixels": images.get("max_pixels"),
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


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
    parser.add_argument(
        "--frontend",
        choices=("python", "rust"),
        help="python: SGLang's HTTP server with its TokenizerManager; rust: the embedded Rust server",
    )
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
    path = Path(_absolute_path(args.table))
    if args.show:
        return _show(path)
    for name in ("model", "frontend", "images"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required")
    if args.recording is not None:
        recording = json.loads(Path(args.recording).read_text())
        # A kept recording is lowered under the measurement it was taken for, never relabeled.
        measured_for = recording.get("measured_for")
        if measured_for is None:
            raise SystemExit(f"{args.recording} carries no measurement identity; measure again instead of lowering it")
        taken_for = FrontendMeasurementConfig.model_validate(measured_for)
        named = _measurement(args, taken_for.feature_transport)
        if taken_for != named:
            raise SystemExit(
                f"{args.recording} was measured for {taken_for.describe()}; this run names {named.describe()}"
            )
        recordings = [(taken_for, recording)]
    else:
        # One collection measures every transport the frontend serves: the Python
        # tokenizer manager always publishes through shared memory, the Rust workers
        # keep features inline on one rank and use shared memory above it.
        recordings = []
        for tensor in (1, 2) if args.frontend == "rust" else (1,):
            measurement = _measurement(args, feature_transport(args.frontend, tensor))
            recording = _measure(args, measurement, tensor)
            recording["measured_for"] = measurement.model_dump(mode="json")
            recordings.append((measurement, recording))
    for measurement, recording in recordings:
        try:
            row = frontend_row(recording, measurement)
            table = update_table(path, environment(recording), row)
        except ValueError as exc:
            # A lowering failure or a table from another serving environment keeps the measurement.
            failed = path.with_name(f"{path.name}.failed-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
            failed.parent.mkdir(parents=True, exist_ok=True)
            failed.write_text(json.dumps(recording, indent=1))
            raise SystemExit(f"{exc}\nThe raw recording was kept at {failed}; lower it again with --recording") from exc
        print(f"{path}: {measurement.describe()} -> {_stages(measurement.frontend, row.stages)} [{table.digest(row)}]")
    return 0


def _measurement(args: argparse.Namespace, transport: str) -> FrontendMeasurementConfig:
    """The row identity this run names for one feature transport."""
    height, width, count = args.images
    return FrontendMeasurementConfig(
        model=args.model,
        frontend=args.frontend,
        feature_transport=transport,
        height=height,
        width=width,
        count=count,
        encoding=args.encoding or "png",
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )


def _measure(args: argparse.Namespace, measurement: FrontendMeasurementConfig, tensor: int) -> dict:
    """Run the worker in the serving interpreter under `tensor` ranks and return its recording."""
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
        str(tensor),
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
