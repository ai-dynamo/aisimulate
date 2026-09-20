# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``python -m aisimulate.vl.calibrate``: sample one image workload into a host profile."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from ..profile import SGLANG_REVISION, SGLANG_VERSION, HostProfile, ProfileIdentity, ProfileImages
from .rust_frontend import PROCESSOR_LABEL, frontend_from_timing, load_worker_spans
from .scheduler_host import batch_costs, host_from_costs, measure_receive
from .workload import generate_images


def _images(value: str) -> tuple[int, int, int]:
    parts = value.lower().split("x")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("images must be HEIGHTxWIDTH or HEIGHTxWIDTHxCOUNT")
    height, width = int(parts[0]), int(parts[1])
    count = int(parts[2]) if len(parts) == 3 else 1
    return height, width, count


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _threads() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _check_sglang_version() -> str:
    from sglang.version import __version__

    if not str(__version__).startswith(SGLANG_VERSION):
        raise SystemExit(
            f"installed sglang {__version__} is not {SGLANG_VERSION}; the cost boundaries are pinned to that release"
        )
    return str(__version__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="aisimulate.vl.calibrate", description=__doc__)
    parser.add_argument("--frontend", choices=("python", "rust"), required=True)
    parser.add_argument("--model", required=True, help="model path or Hugging Face identifier")
    parser.add_argument("--images", type=_images, required=True, metavar="HxW[xN]")
    parser.add_argument("--encoding", choices=("png", "jpeg"), default="png")
    parser.add_argument("--text-tokens", type=int, default=128)
    parser.add_argument("--min-pixels", type=int, help="processor rescale floor; default: the checkpoint's")
    parser.add_argument("--max-pixels", type=int, help="processor rescale ceiling; default: the checkpoint's")
    parser.add_argument("--rust-timing", help="JSONL written by the patched Rust server (frontend=rust)")
    parser.add_argument("--mm-workers", type=int, default=8, help="Rust multimodal worker count")
    parser.add_argument("--batch-costs", help="JSON with measured select/launch_*/result costs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    installed_version = _check_sglang_version()
    height, width, count = args.images
    images = generate_images(height, width, count, args.encoding)
    provenance = {
        "sampled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "sglang_version": installed_version,
        "nice": os.nice(0) if hasattr(os, "nice") else None,
    }
    from .python_frontend import image_process_config

    mm_process_config = image_process_config(args.min_pixels, args.max_pixels)
    if args.frontend == "python":
        from .python_frontend import measure_python_frontend

        measured = measure_python_frontend(
            args.model, images, args.encoding, text_tokens=args.text_tokens, mm_process_config=mm_process_config
        )
        frontend = measured["frontend"]
        provenance["frontend"] = measured["provenance"]
        processor = measured["provenance"]["processor_class"]
    else:
        if args.rust_timing is None:
            parser.error("--rust-timing is required for frontend=rust")
        frontend = frontend_from_timing(args.rust_timing, mm_workers=args.mm_workers)
        provenance["frontend"] = {
            "timing": str(Path(args.rust_timing).resolve()),
            "worker_spans": len(load_worker_spans(args.rust_timing)),
            "includes": {
                "mm_worker": ["rust_worker boundary span (payload, fetch, hash, decode, patchify, layout, pack)"]
            },
        }
        processor = PROCESSOR_LABEL
    receive = measure_receive(
        args.model,
        images,
        args.encoding,
        frontend=args.frontend,
        text_tokens=args.text_tokens,
        mm_process_config=mm_process_config,
    )
    provenance["receive"] = {"includes": receive.includes, **receive.provenance}
    batch, missing = batch_costs(args.batch_costs)
    profile = HostProfile(
        identity=ProfileIdentity(
            sglang_revision=SGLANG_REVISION,
            model=args.model,
            frontend=args.frontend,
            images=ProfileImages(
                height=height,
                width=width,
                count=count,
                encoding=args.encoding,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            ),
            text_tokens=args.text_tokens,
            processor=processor,
            cpu=_cpu_model(),
            threads=_threads(),
        ),
        host=host_from_costs(receive.cost, batch),
        frontend=frontend,
        missing=missing,
        provenance=provenance,
    )
    Path(args.output).write_text(json.dumps(profile.model_dump(mode="json"), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
