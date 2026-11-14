#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and audit the local ``data/op-latency/`` publish tree from the packaged op data (AISIM-2340).

Local only: never uploads. Build then audit::

    python tools/perf_database/publish_op_latency_tree.py build --out /tmp/op-latency-publish
    python tools/perf_database/publish_op_latency_tree.py audit --out /tmp/op-latency-publish --json audit.json

The converter is :mod:`op_latency_publish` (next to this script); see its docstring for the layout and hash
policy. The permanent schema, path and hash contract is :mod:`aisim_core.data.op_schema`.
"""

import argparse
import json
import sys
from pathlib import Path

from op_latency_publish import audit_tree, build_publish_tree

import aisim_core
from aisim_core.data.op_schema import SOURCE_MAP_PATH

SYSTEMS = Path(aisim_core.__file__).parent / "systems"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "audit"))
    parser.add_argument("--out", type=Path, required=True, help="publish tree (build: must not exist)")
    parser.add_argument("--systems-dir", type=Path, default=SYSTEMS)
    parser.add_argument("--dataset-id", default="local/op-latency-publish")
    parser.add_argument("--allow-dirty", action="store_true", help="build from an uncommitted source tree")
    parser.add_argument("--json", type=Path, help="write the result summary here")
    args = parser.parse_args(argv)
    data = args.systems_dir / "data"
    reuse, versions = args.systems_dir / "perf_data_reuse_manifest.yaml", args.systems_dir / "query_versions.yaml"
    if args.command == "build":
        result = build_publish_tree(
            data,
            reuse,
            versions,
            args.out,
            dataset_id=args.dataset_id,
            systems_dir=args.systems_dir,
            allow_dirty=args.allow_dirty,
        )
        summary = {k: v for k, v in result.__dict__.items() if k not in ("map_rows", "out")}
        summary["out"] = str(result.out)
        summary["source_map"] = SOURCE_MAP_PATH
        ok = True
    else:
        report = audit_tree(args.out, data, reuse_manifest=reuse, query_versions=versions)
        summary, ok = report.to_dict(), report.ok
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
