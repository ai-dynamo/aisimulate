#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a supervised AISim replay and record the complete CLI wall time."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    config = args.config.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    execution_path = output.with_name(output.name + "-execution.json")
    log_path = output.with_name(output.name + ".log")
    if output.exists() or execution_path.exists() or log_path.exists():
        parser.error("Use a new output directory; prior evidence will not be overwritten")
    command = [
        str(repo / "python/aisimulate/.venv/bin/aisimulate"),
        "predict",
        "--config",
        str(config),
        "--capture-telemetry",
        "--capture-per-request",
        "--telemetry-sample-interval-ms",
        "1000",
        "--output-dir",
        str(output),
    ]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).splitlines()
    metadata = {
        "schema_version": 1,
        "variant": args.variant,
        "status": "running",
        "command": command,
        "aisim_commit": commit,
        "checkout_changes": dirty,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
        },
        "wall_time_scope": (
            "Complete CLI process: startup, resource preflight, full corpus import, "
            "preparation, profiling and grace, telemetry serialization. "
            "Excludes source download/transfer, environment build and plotting."
        ),
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    execution_path.write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.perf_counter()
    try:
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=False)
        metadata["returncode"] = result.returncode
        metadata["status"] = "completed" if result.returncode == 0 else "failed"
    except BaseException as error:
        metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metadata["error"] = repr(error)
        raise
    finally:
        metadata["wall_time_seconds"] = time.perf_counter() - started
        metadata["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        execution_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2), flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
