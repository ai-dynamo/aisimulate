#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve point 446063 inputs and derive a measured-output-length replay corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

DATASET_SHA256 = "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276"
DATASET_BYTES = 1_847_151_435
SOURCE_URL = (
    "https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/resolve/"
    "23f152f6f0f9399a85901b89a6458def0ef16729/traces.jsonl"
)
RequestKey = tuple[str, int, int | None]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def request_key(metadata: dict[str, Any]) -> RequestKey:
    return (
        metadata["source_trace_id"],
        metadata["source_outer_idx"],
        metadata.get("source_inner_idx"),
    )


def key_record(key: RequestKey) -> dict[str, Any]:
    return dict(zip(("source_trace_id", "source_outer_idx", "source_inner_idx"), key, strict=True))


def observed_lengths(
    profile: Path,
) -> tuple[dict[RequestKey, int], list[dict[str, Any]], Counter]:
    lengths: dict[RequestKey, int] = {}
    excluded: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for line in profile.read_text().splitlines():
        row = json.loads(line)
        metadata = row["metadata"]
        counts["all_reference_rows"] += 1
        if metadata["benchmark_phase"] != "profiling":
            counts["warmup_rows_not_modified"] += 1
            continue
        counts["profile_reference_rows"] += 1
        metrics = row.get("metrics", {})
        length = metrics.get("output_sequence_length", {}).get("value")
        if length is None:
            length = metrics.get("usage_completion_tokens", {}).get("value")
        key = request_key(metadata)
        if length is None:
            excluded.append(
                {
                    **key_record(key),
                    "reason": "No per-request output or usage completion token count",
                    "error": row.get("error"),
                }
            )
            continue
        if isinstance(length, bool) or not isinstance(length, (int, float)):
            raise ValueError(f"Non-numeric observed output length for {key}: {length!r}")
        if length < 0 or int(length) != length:
            raise ValueError(f"Invalid observed output length for {key}: {length!r}")
        length = int(length)
        if key in lengths and lengths[key] != length:
            raise ValueError(f"Conflicting observed output lengths for {key}")
        lengths[key] = length
        counts["profile_rows_with_observed_lengths"] += 1
        counts["observed_profile_output_tokens"] += length
    if counts["all_reference_rows"] != 337 or len(lengths) != 324:
        raise ValueError(f"Unexpected point 446063 reference coverage: {dict(counts)}")
    return lengths, excluded, counts


def preserve_sources(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    trace = source / "traces.jsonl"
    target = destination / "traces.jsonl"
    if target.exists():
        if not target.samefile(trace) and digest(target) != DATASET_SHA256:
            raise ValueError(f"Existing preserved corpus has a different hash: {target}")
    else:
        try:
            os.link(trace, target)
        except OSError:
            shutil.copy2(trace, target)
    names = [
        "source-manifest.json",
        "point.json",
        "recipe.yaml",
        "glm5.3_mtp.yaml",
        "run.json",
        "dataset-download.json",
        "dataset-README.md",
        "dataset-stats.txt",
        "initial-selection.json",
        "reference-requests.json",
        "first-lane-source-map.json",
        "source_namespace_map.json",
        "source_namespace_map.provenance.json",
        "request-timeline.json",
        "request-chart-data.json",
        "trace-server-metrics.json",
        "agentic/benchmark_command.txt",
        "agentic/aiperf_artifacts/profile_export.jsonl",
        "agentic/aiperf_artifacts/profile_export_aiperf.json",
    ]
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or not target.samefile(source / name):
            shutil.copy2(source / name, target)


def prepare(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the original source directory")
    profile = source / "agentic/aiperf_artifacts/profile_export.jsonl"
    lengths, excluded, counts = observed_lengths(profile)
    trace = source / "traces.jsonl"
    if trace.stat().st_size != DATASET_BYTES:
        raise ValueError("The source corpus is incomplete or differs from the pinned dataset")
    derived = output / "observed-input"
    derived.mkdir(parents=True, exist_ok=True)
    temporary = derived / "traces.jsonl.tmp"
    original_hash = hashlib.sha256()
    derived_hash = hashlib.sha256()
    matched: set[RequestKey] = set()
    changes: list[dict[str, Any]] = []
    play_ids: list[str] = []
    try:
        with trace.open("rb") as src, temporary.open("wb") as dst:
            for line in src:
                original_hash.update(line)
                play = json.loads(line)
                play_ids.append(play["id"])
                changed = False
                for outer, request in enumerate(play["requests"]):
                    if request["type"] == "subagent":
                        requests = enumerate(request["requests"])
                    else:
                        requests = [(None, request)]
                    for inner, item in requests:
                        counts["source_model_requests"] += 1
                        key = (play["id"], outer, inner)
                        if key not in lengths:
                            continue
                        if key in matched:
                            raise ValueError(f"Duplicate source request identity: {key}")
                        matched.add(key)
                        old, new = item["out"], lengths[key]
                        counts["matched_source_output_tokens"] += old
                        if old != new:
                            changes.append({**key_record(key), "old_out": old, "new_out": new})
                            item["out"] = new
                            changed = True
                # All source fields, including complete hashes and timing, remain intact.
                encoded = (
                    (json.dumps(play, separators=(",", ":"), ensure_ascii=False) + "\n").encode() if changed else line
                )
                if changed:
                    original = json.loads(line)
                    restored = json.loads(encoded)
                    for change in changes:
                        if change["source_trace_id"] != play["id"]:
                            continue
                        item = restored["requests"][change["source_outer_idx"]]
                        if change["source_inner_idx"] is not None:
                            item = item["requests"][change["source_inner_idx"]]
                        item["out"] = change["old_out"]
                    if original != restored:
                        raise ValueError(f"A field other than output length changed in {play['id']}")
                dst.write(encoded)
                derived_hash.update(encoded)
        if original_hash.hexdigest() != DATASET_SHA256:
            raise ValueError("Source SHA256 does not match the pinned dataset")
        if len(play_ids) != 393 or counts["source_model_requests"] != 98_827:
            raise ValueError("Source corpus counts do not match the pinned dataset")
        if matched != lengths.keys():
            raise ValueError(f"Reference requests missing from corpus: {lengths.keys() - matched}")
        if len(changes) != 134 or counts["observed_profile_output_tokens"] != 318_405:
            raise ValueError("Derived output counts differ from the audited reference")
        preserve_sources(source, output / "source")
        target = derived / "traces.jsonl"
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "point_id": 446063,
        "variant": "observed_profile_output_lengths",
        "source_url": SOURCE_URL,
        "source_sha256": DATASET_SHA256,
        "source_bytes": DATASET_BYTES,
        "source_license": "Apache-2.0",
        "source_attribution": "SemiAnalysisAI cc-traces-weka-062126; see ../source/dataset-README.md",
        "modification": "Only observed profiling output lengths were changed; original warmup lengths remain",
        "reference_profile_sha256": digest(profile),
        "derived_sha256": derived_hash.hexdigest(),
        "derived_bytes": target.stat().st_size,
        "plays": len(play_ids),
        "play_order_sha256": hashlib.sha256("\n".join(play_ids).encode()).hexdigest(),
        "counts": dict(counts),
        "changed_requests": len(changes),
        "unchanged_matched_requests": len(matched) - len(changes),
        "changes": changes,
        "excluded_profile_rows": excluded,
        "preserved": ["input lengths", "all prefix hashes", "source order", "timing", "model metadata"],
        "limitations": [
            "Two no-content errors have no per-request output count; their source output lengths are unchanged",
            "Actual measured prompt lengths differ slightly from source hash-block input lengths",
            "Unobserved requests retain original output lengths; this is not an exact HTTP payload replay",
            "No cache-hit values or timing estimates were calibrated",
        ],
    }
    (derived / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.source_dir, args.output_dir)
    print(json.dumps({key: manifest[key] for key in ("plays", "changed_requests", "derived_sha256")}))


if __name__ == "__main__":
    main()
