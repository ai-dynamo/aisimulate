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
import math
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


def _source_time_bounds(metadata, root_id):
    """Require complete root/child request clocks; branch markers are not requests."""
    by_id = {item.conversation_id: item for item in metadata.conversations}
    if len(by_id) != len(metadata.conversations):
        raise ValueError("duplicate reference conversation IDs")
    pending = [root_id]
    seen = set()
    timestamps = []
    while pending:
        conversation_id = pending.pop()
        if conversation_id in seen:
            raise ValueError("reference child graph is cyclic or has shared children")
        seen.add(conversation_id)
        item = by_id.get(conversation_id)
        if item is None:
            raise ValueError(f"missing reference child metadata: {conversation_id}")
        if conversation_id != root_id and getattr(item, "is_root", True):
            raise ValueError("reference child also appears as a sampleable root")
        for turn in item.turns:
            timestamp = turn.timestamp_ms
            if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("every reference request must have a finite nonnegative timestamp")
            timestamps.append(timestamp)
        for branch in item.branches:
            pending.extend(branch.child_conversation_ids)
    if not timestamps:
        raise ValueError("reference play has no request timestamps")
    return min(timestamps), max(timestamps)


def export_snapshot(metadata, source_play_map, *, lanes, seed, trajectory_type, trace_format, clock_evidence=None):
    """Invoke the reference selector and retain its actual shared cursor."""
    if type(lanes) is not int or lanes <= 0 or type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("lanes must be positive and seed must be an unsigned 64-bit integer")
    if trace_format not in ("weka", "agentic_mooncake"):
        raise ValueError("trace_format must explicitly select weka or agentic_mooncake")
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
    selections = []
    for item in source.trajectories:
        start, end = _source_time_bounds(metadata, item.conversation_id)
        cut = item.snapshot.t_star_ms
        if type(cut) not in (int, float) or not math.isfinite(cut) or not start <= cut <= end:
            raise ValueError("reference cut is outside its source request-start bounds")
        # Weka lower_trace subtracts the earliest request in the entire play,
        # including child streams. Agentic Mooncake retains its source clock.
        origin = start if trace_format == "weka" else 0
        selections.append({"source_play_id": source_play_map[item.conversation_id], "t_star_ms": cut - origin})
        if clock_evidence is not None:
            clock_evidence.append(
                {
                    "reference_root_id": item.conversation_id,
                    "source_play_id": source_play_map[item.conversation_id],
                    "source_origin_ms": start,
                    "subtracted_origin_ms": origin,
                    "reference_t_star_ms": cut,
                    "imported_t_star_ms": cut - origin,
                }
            )
    return {
        "seed": seed,
        "initial_selections": selections,
        "corpus_cursor": sampler.cursor,
    }


def write_export(result, provenance, *, output, provenance_out):
    """Persist provenance before publishing usable config; never overwrite artifacts."""
    if output is not None:
        if provenance_out is None:
            provenance_out = output.with_name(output.name + ".provenance.json")
        if output.resolve() == provenance_out.resolve():
            raise ValueError("output and provenance paths must differ")
        if output.exists():
            raise FileExistsError(output)
    elif provenance_out is None:
        raise ValueError("stdout export requires --provenance-out")
    result_json = json.dumps(result, indent=2, allow_nan=False) + "\n"
    provenance = {**provenance, "snapshot_options_sha256": hashlib.sha256(result_json.encode()).hexdigest()}
    with provenance_out.open("x") as stream:
        stream.write(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    if output is None:
        print(result_json, end="")
    else:
        with output.open("x") as stream:
            stream.write(result_json)


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
    parser.add_argument("--trace-format", choices=("weka", "agentic_mooncake"), required=True)
    parser.add_argument(
        "--output", type=Path, help="Write config and a sibling .provenance.json sidecar; never overwrite"
    )
    parser.add_argument("--provenance-out", type=Path, help="Required for stdout; overrides the --output sidecar path")
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
    clock_evidence = []
    result = export_snapshot(
        metadata,
        json.loads(mapping_bytes),
        lanes=args.lanes,
        seed=args.seed,
        trajectory_type=module.TrajectorySource,
        trace_format=args.trace_format,
        clock_evidence=clock_evidence,
    )
    write_export(
        result,
        {
            "reference_revision": head,
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "source_play_map_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
            "source_order": [
                item.conversation_id for item in metadata.conversations if getattr(item, "is_root", True) is not False
            ],
            "sampling_strategy": "sequential",
            "start_ratio": [0.25, 0.75],
            "trace_format": args.trace_format,
            "input_clock": "reference_source_ms",
            "output_clock": "per_play_relative_ms" if args.trace_format == "weka" else "source_ms",
            "clock_conversions": clock_evidence,
        },
        output=args.output,
        provenance_out=args.provenance_out,
    )


if __name__ == "__main__":
    main()
