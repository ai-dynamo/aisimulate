# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``python -m aisimulate.vl.collect``: measure one image workload's frontend stages into a host cost table.

The measurement itself runs in the serving host's SGLang interpreter
(``--sglang-python``, which may be another virtual environment); this command
lowers its recording into one table row and saves the table. Nothing here
needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ..table import SGLANG_REVISION, HostCostTable, RowIdentity, Shape, load_table, save_table, upsert_row
from .lower import frontend_row

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
    for row in table.rows:
        stages = ", ".join(f"{stage.resource}({stage.workers}) {stage.cost.const_ms:.2f} ms" for stage in row.stages)
        print(f"{row.identity.frontend:6s} {row.shape.describe():40s} {row.identity.cpu} | {stages} | {row.digest()}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="aisimulate.vl.collect", description=__doc__)
    parser.add_argument("--table", required=True, help="host cost table to create or extend")
    parser.add_argument("--show", action="store_true", help="print the table's rows and exit")
    parser.add_argument("--model", help="model identifier as written in engine.model")
    parser.add_argument("--frontend", choices=("python", "rust"))
    parser.add_argument("--images", type=_images, metavar="HxW[xN]")
    parser.add_argument("--encoding", choices=("png", "jpeg"), default="png")
    parser.add_argument("--text-tokens", type=int, default=128)
    parser.add_argument("--min-pixels", type=int, help="processor rescale floor; default: the checkpoint's")
    parser.add_argument("--max-pixels", type=int, help="processor rescale ceiling; default: the checkpoint's")
    parser.add_argument(
        "--levels", type=int, default=0, help="highest concurrency to sample (python: 16, rust: worker count)"
    )
    parser.add_argument("--sglang-python", default=sys.executable, help="interpreter with sglang installed")
    args = parser.parse_args(argv)
    path = Path(args.table)
    if args.show:
        return _show(path)
    for name in ("model", "frontend", "images"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required")
    height, width, count = args.images
    command = [
        args.sglang_python,
        str(WORKER),
        "--model",
        args.model,
        "--frontend",
        args.frontend,
        "--height",
        str(height),
        "--width",
        str(width),
        "--count",
        str(count),
        "--encoding",
        args.encoding,
        "--text-tokens",
        str(args.text_tokens),
        "--levels",
        str(args.levels),
    ]
    for name, value in (("--min-pixels", args.min_pixels), ("--max-pixels", args.max_pixels)):
        if value is not None:
            command += [name, str(value)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        sys.stderr.write(completed.stderr[-8000:])
        return completed.returncode
    recording = json.loads(completed.stdout.splitlines()[-1])
    identity = RowIdentity(
        cpu=recording["cpu"], sglang_revision=SGLANG_REVISION, model=args.model, frontend=args.frontend
    )
    shape = Shape(
        height=height,
        width=width,
        count=count,
        encoding=args.encoding,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        text_tokens=args.text_tokens,
    )
    row = frontend_row(recording, identity=identity, shape=shape)
    table = load_table(path) if path.exists() else HostCostTable()
    save_table(path, upsert_row(table, row))
    stages = "; ".join(f"{stage.resource}({stage.workers}) {stage.cost.const_ms:.2f} ms" for stage in row.stages)
    print(f"{path}: {args.frontend} {shape.describe()} -> {stages} [{row.digest()}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
