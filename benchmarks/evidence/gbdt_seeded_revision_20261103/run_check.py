#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one explicit local check and retain its exact command, status and log."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--cwd", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command:
    parser.error("an explicit command is required")
args.output.parent.mkdir(parents=True, exist_ok=True)
record = {"command": command, "cwd": str(args.cwd.resolve()),
          "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "environment": {key: os.environ.get(key) for key in
                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH",
                           "CARGO_TARGET_DIR", "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0")}}
start = time.monotonic()
with args.output.with_suffix(".log").open("wb") as log:
    result = subprocess.run(command, cwd=args.cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
record.update(exit_code=result.returncode, seconds=time.monotonic()-start)
record["log_sha256"] = hashlib.sha256(args.output.with_suffix(".log").read_bytes()).hexdigest()
args.output.with_suffix(".json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record, indent=2), flush=True)
raise SystemExit(result.returncode)
