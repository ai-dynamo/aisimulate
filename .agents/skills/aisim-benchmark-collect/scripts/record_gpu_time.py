# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a finished Slurm or Kubernetes job record into a ``gpu_time`` JSON document.

Parsing is done by ``aisim_core.data.gpu_time``; this script only reads files
you saved from the scheduler and prints the result with its check findings.
It never contacts a scheduler. ``--gpu-type`` must come from an observed fact (node
inventory, ``nvidia-smi``), not a guess; without it the type stays null.

    sacct -j JOBIDS --parsable2 -o JobID,JobName,State,Submit,Start,End,ElapsedRaw,Elapsed,NNodes,AllocTRES > sacct.txt
    kubectl get job NAME -o json > job.json; kubectl get pods -l job-name=NAME -o json > pods.json
"""

import argparse
import json
import sys
from pathlib import Path

from aisim_core.data.gpu_time import (
    KubeParseError,
    SlurmParseError,
    check_gpu_time,
    gpu_time_from_kubernetes,
    gpu_time_from_sacct,
)


def _items(doc):
    return doc["items"] if isinstance(doc, dict) and "items" in doc else doc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="source", required=True)
    s = sub.add_parser("slurm")
    s.add_argument("--sacct", required=True, type=Path)
    k = sub.add_parser("k8s")
    k.add_argument("--job", required=True, type=Path)
    k.add_argument("--pods", required=True, type=Path)
    for sp in (s, k):
        sp.add_argument("--gpu-type")
    args = p.parse_args(argv)
    try:
        if args.source == "slurm":
            gpu_time = gpu_time_from_sacct(args.sacct.read_text(encoding="utf-8"), gpu_type=args.gpu_type)
        else:
            job = json.loads(args.job.read_text(encoding="utf-8"))
            pods = _items(json.loads(args.pods.read_text(encoding="utf-8")))
            gpu_time = gpu_time_from_kubernetes(job, pods, gpu_type=args.gpu_type)
    except (SlurmParseError, KubeParseError) as exc:
        # Unfinished or inconsistent job records are not turned into numbers.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = {
        "gpu_time": gpu_time.model_dump(by_alias=True, mode="json"),
        "findings": {
            level: [f.__dict__ for f in check_gpu_time(gpu_time, level)] for level in ("archival", "qualification")
        },
    }
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
