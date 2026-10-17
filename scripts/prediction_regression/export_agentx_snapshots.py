#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export exact initial selections from an installed, pinned AIPerf reference.

This tool invokes upstream code; it does not approximate NumPy's RNG in Rust.
Run in the reference AIPerf environment, not the AISimulate runtime environment.
Source: SemiAnalysisAI/aiperf@754356e9a39acc6cc6afb242d123bb57c3fb6f75,
src/aiperf/timing/trajectory_source.py (Apache-2.0, NVIDIA copyright).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

REFERENCE_REVISION = "754356e9a39acc6cc6afb242d123bb57c3fb6f75"


class SequentialCursor:
    """Record every sequential sampler draw, including unspawnable samples."""

    def __init__(self, ids):
        self.ids = tuple(ids)
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("reference roots must be nonempty and uniquely identified")
        self.cursor = 0

    def next_conversation_id(self):
        selected = self.ids[self.cursor % len(self.ids)]
        self.cursor += 1
        return selected


def export_snapshot(metadata, source_play_map, *, lanes, seed, trajectory_type):
    """Invoke the reference selector and retain its actual shared cursor."""
    if type(lanes) is not int or lanes <= 0 or type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("lanes must be positive and seed must be an unsigned 64-bit integer")
    strategy = getattr(metadata.sampling_strategy, "value", metadata.sampling_strategy)
    if str(strategy).lower() != "sequential":
        raise ValueError("only the pinned reference's sequential sampler is supported")
    root_ids = [item.conversation_id for item in metadata.conversations if getattr(item, "is_root", True) is not False]
    if set(source_play_map) != set(root_ids) or any(not isinstance(v, str) or not v for v in source_play_map.values()):
        raise ValueError("source-play-map must map every reference root to its normalized AISimulate source_play_id")
    if len(set(source_play_map.values())) != len(root_ids):
        raise ValueError("distinct reference roots must map to distinct source plays")
    sampler = SequentialCursor(root_ids)
    source = trajectory_type(
        dataset_metadata=metadata,
        dataset_sampler=sampler,
        concurrency=lanes,
        random_seed=seed,
        start_min_ratio=0.25,
        start_max_ratio=0.75,
        allow_dataset_wrap=True,
    )
    if len(source.trajectories) != lanes:
        raise ValueError("reference could not construct every requested lane")
    if any(item.snapshot is None for item in source.trajectories):
        raise ValueError("timestamp-less reference snapshots cannot be imported as time cuts")
    return {
        "seed": seed,
        "initial_selections": [
            {"source_play_id": source_play_map[item.conversation_id], "t_star_ms": item.snapshot.t_star_ms}
            for item in source.trajectories
        ],
        "corpus_cursor": sampler.cursor,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiperf-source", type=Path, required=True)
    parser.add_argument(
        "--metadata", type=Path, required=True, help="DatasetMetadata.model_dump_json() from the reference loader"
    )
    parser.add_argument(
        "--source-play-map",
        type=Path,
        required=True,
        help="JSON object: reference root ID -> normalized AIS source_play_id",
    )
    parser.add_argument("--lanes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    checkout = args.aiperf_source.resolve()
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if head != REFERENCE_REVISION:
        parser.error(f"reference checkout must be exactly {REFERENCE_REVISION}, got {head}")
    if subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all", "--", "src/aiperf"], text=True
    ).strip():
        parser.error("reference AIPerf source has local modifications")
    sys.path.insert(0, str(checkout / "src"))
    module = importlib.import_module("aiperf.timing.trajectory_source")
    if Path(module.__file__).resolve() != checkout / "src/aiperf/timing/trajectory_source.py":
        parser.error("loaded trajectory source does not belong to the verified checkout")
    metadata_type = importlib.import_module("aiperf.common.models").DatasetMetadata
    metadata_bytes = args.metadata.read_bytes()
    mapping_bytes = args.source_play_map.read_bytes()
    metadata = metadata_type.model_validate_json(metadata_bytes)
    result = export_snapshot(
        metadata, json.loads(mapping_bytes), lanes=args.lanes, seed=args.seed, trajectory_type=module.TrajectorySource
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "reference_revision": head,
                "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "source_play_map_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
                "source_order": [
                    item.conversation_id
                    for item in metadata.conversations
                    if getattr(item, "is_root", True) is not False
                ],
                "sampling_strategy": "sequential",
                "start_ratio": [0.25, 0.75],
            }
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
