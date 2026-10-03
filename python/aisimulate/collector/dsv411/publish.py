# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish admitted dsv411 runs into a perf-database systems tree.

    python -m collector.dsv411.publish --system h20_3e --backend sglang --systems-root .../systems \
        --runs raw/sglang-tp2-shard0 raw/sglang-tp2-shard1 raw/sglang-tp4 \
        --image lmsysorg/sglang:v0.5.21 --torch 2.13.0 --nccl 2.30.7

Every run directory goes through the producer's own admission (``contract.aggregate_run``); the
admitted rows of all runs (shards, TP sizes) are concatenated — a physical key measured twice is
an error, never averaged — and written as
``data/<system>/dsv411/<backend>/<version>/dsv411_module_perf.parquet`` with its
``collection_meta.yaml`` (one collection event per run).
"""

from __future__ import annotations

import argparse
import importlib
from datetime import date
from pathlib import Path

from . import contract
from .plan import PRODUCERS


def main(argv=None):
    from collector import provenance

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", required=True)
    parser.add_argument("--backend", required=True, choices=sorted(PRODUCERS))
    parser.add_argument("--systems-root", required=True, type=Path)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--torch", required=True)
    parser.add_argument("--nccl", required=True)
    parser.add_argument("--collected-at", default=date.today().isoformat())
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--allow-smoke", action="store_true", help="publish smoke runs (dry trees only; never into the packaged data)"
    )
    args = parser.parse_args(argv)

    module_name = PRODUCERS[args.backend]
    producer = importlib.import_module(module_name)
    per_run, events_meta = [], []
    for raw in args.runs:
        rows, meta = contract.aggregate_run(
            raw,
            framework_commit=producer.FRAMEWORK_COMMIT,
            framework_version=producer.FRAMEWORK_VERSION,
            expected_sm=producer.EXPECTED_SM,
            required_sources=producer.REQUIRED_SOURCES,
        )
        if meta["plan"]["purpose"] != "calibration" and not args.allow_smoke:
            raise SystemExit(
                f"{raw}: purpose {meta['plan']['purpose']!r} is not publishable (use --allow-smoke for dry trees)"
            )
        per_run.append(rows)
        events_meta.append((raw, meta, len(rows)))
    rows = contract.pool_runs(per_run)
    digests = {m["runtime_digest"] for _, m, _ in events_meta}
    if len(digests) != 1:
        raise SystemExit("runs come from different images")
    runtime_digest = digests.pop()
    closures = provenance.load_closures(args.repo_root / "collector/hash_closures.yaml")
    runtime = dict(
        framework=args.backend,
        version=producer.FRAMEWORK_VERSION,
        image=args.image,
        image_digest=runtime_digest,
        source_commit=producer.FRAMEWORK_COMMIT,
        abi=dict(torch=args.torch, nccl=args.nccl),
    )
    events = [
        dict(
            collector_ref=meta["plan"]["collector_revision"],
            collector_hash=provenance.collector_hash(module_name, args.repo_root, closures),
            case_plan_hash="sha256:" + meta["plan_sha256"],
            collected_at=args.collected_at,
            rows=count,
            status="complete",
            runtime=runtime,
        )
        for _, meta, count in events_meta
    ]
    dest = args.systems_root / "data" / args.system / "dsv411" / args.backend / producer.FRAMEWORK_VERSION
    dest.mkdir(parents=True, exist_ok=True)
    contract.write_parquet(rows, dest / f"{contract.TABLE}.parquet")
    provenance.write_collection_meta(
        dest,
        runtime,
        {contract.TABLE: dict(rows=len(rows), status="complete", collections=events)},
        provenance_tier="collected",
    )
    print(f"wrote {dest / contract.TABLE}.parquet rows={len(rows)} runs={len(events)}")


if __name__ == "__main__":
    main()
