# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``python -m aisimulate.vl.calibrate``: sample one image workload into a host profile."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from ..profile import SGLANG_REVISION, HostProfile, ProfileIdentity
from .rust_frontend import frontend_from_timing
from .scheduler_host import batch_costs, host_from_costs, measure_receive
from .workload import generate_images


def _images(value: str) -> tuple[int, int, int]:
    parts = value.lower().split("x")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("images must be HEIGHTxWIDTH or HEIGHTxWIDTHxCOUNT")
    height, width = int(parts[0]), int(parts[1])
    count = int(parts[2]) if len(parts) == 3 else 1
    return height, width, count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="aisimulate.vl.calibrate", description=__doc__)
    parser.add_argument("--frontend", choices=("python", "rust"), required=True)
    parser.add_argument("--model", required=True, help="model path or Hugging Face identifier")
    parser.add_argument("--images", type=_images, required=True, metavar="HxW[xN]")
    parser.add_argument("--encoding", choices=("png", "jpeg"), default="png")
    parser.add_argument("--text-tokens", type=int, default=128)
    parser.add_argument("--rust-timing", help="JSONL written by the patched Rust server (frontend=rust)")
    parser.add_argument("--mm-workers", type=int, default=8, help="Rust multimodal worker count")
    parser.add_argument("--batch-costs", help="JSON with measured select/launch_*/result costs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    height, width, count = args.images
    images = generate_images(height, width, count, args.encoding)
    provenance = {
        "sampled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "images": {"height": height, "width": width, "count": count, "encoding": args.encoding},
        "text_tokens": args.text_tokens,
    }
    if args.frontend == "python":
        from .python_frontend import measure_python_frontend

        measured = measure_python_frontend(args.model, images, args.encoding, text_tokens=args.text_tokens)
        frontend = measured["frontend"]
        provenance["frontend"] = measured["provenance"]
    else:
        if args.rust_timing is None:
            parser.error("--rust-timing is required for frontend=rust")
        frontend = frontend_from_timing(args.rust_timing, mm_workers=args.mm_workers)
        provenance["frontend"] = {"timing": str(Path(args.rust_timing).resolve())}
    receive = measure_receive(args.model, images, args.encoding, text_tokens=args.text_tokens)
    batch, missing = batch_costs(args.batch_costs)
    profile = HostProfile(
        identity=ProfileIdentity(
            sglang_revision=SGLANG_REVISION, model=args.model, frontend=args.frontend, image_encoding=args.encoding
        ),
        host=host_from_costs(receive, batch),
        frontend=frontend,
        missing=missing,
        provenance=provenance,
    )
    Path(args.output).write_text(json.dumps(profile.model_dump(mode="json"), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
