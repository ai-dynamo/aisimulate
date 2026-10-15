# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAFD MoE stage profiles: public exports and the builder from raw benchmark results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from aisimulate_core.sdk.fastafd_profile import (
    FASTAFD_PROFILE_SCHEMA,
    FastAFDMoEStageKey,
    FastAFDMoEStageMeasurement,
    FastAFDMoEStageProfile,
    FastAFDProfileSource,
)

# Raw result schemas written by FastAFD's MegaMoE model benchmarks, and the stage each one measures.
_BENCHMARK_STAGES = {
    "fastafd.megamoe-colocated-model-benchmark.v2": "agg",
    "fastafd.moe-m2n-model-benchmark.v2": "afd",
}
_BACKENDS = {"mega": "megamoe", "megamoe": "megamoe"}


def build_profile(
    results_dir: str | Path,
    *,
    repository: str,
    routing: str = "balanced",
) -> tuple[dict[str, Any], Counter[str]]:
    """Convert one directory of raw MegaMoE benchmark results into a profile document.

    Every file must come from the same benchmark, source commit, and system. Results
    with another routing distribution, an unstable timing, or a failed matched-output
    check are skipped and counted in the returned summary.
    """

    files = sorted(Path(results_dir).glob("*.json"))
    if not files:
        raise ValueError(f"no benchmark results in {results_dir}")
    summary: Counter[str] = Counter()
    source: dict[str, str] | None = None
    entries: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], str] = {}
    for path in files:
        raw = path.read_bytes()
        payload = json.loads(raw)
        stage = _BENCHMARK_STAGES.get(payload.get("schema"))
        if stage is None:
            raise ValueError(f"{path.name}: unsupported benchmark schema {payload.get('schema')!r}")
        current = {
            "repository": repository,
            "commit": payload["source"]["commit"],
            "benchmark": payload["schema"],
            "boundary": payload["measurement_boundary"],
        }
        if source is None:
            source = current
        elif current != source:
            raise ValueError(f"{path.name}: mixed benchmark, commit, or boundary in {results_dir}")
        workload = payload["workload"]
        if workload["routing"] != routing:
            summary["skipped_routing"] += 1
            continue
        if payload["stable"] is not True:
            summary["skipped_unstable"] += 1
            continue
        correctness = payload.get("correctness_passed")
        if correctness is False:
            summary["skipped_correctness"] += 1
            continue
        key = _key(stage, payload)
        identity = tuple(sorted(key.items()))
        if identity in seen:
            raise ValueError(f"{path.name} repeats the measurement in {seen[identity]}")
        seen[identity] = path.name
        stats = payload["stage_cuda"]
        stack = payload["provenance_by_rank"][0]
        entries.append(
            {
                "key": key,
                "model_profile": payload["model_profile"]["key"],
                "latency_ms": stats["p50_ms"],
                "measurement": {
                    "scope": "complete_moe_stage",
                    "method": "cuda_event",
                    "method_version": f"torch {stack['torch']}; cuda {stack['cuda_runtime']}",
                    "statistic": "p50",
                    "sample_count": stats["samples"],
                    "cv_percent": stats["cv_percent"],
                    "routing": routing,
                    "source_tree_sha256": payload["source"]["source_tree_sha256"],
                    "raw_artifact": path.name,
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                },
                "validation": {"stable": True, "correctness": correctness},
            }
        )
        summary["accepted"] += 1
    if not entries:
        raise ValueError(f"no qualified {routing} measurements in {results_dir}")
    return {
        "schema": FASTAFD_PROFILE_SCHEMA,
        "lookup_policy": "exact-only",
        "source": source,
        "entries": entries,
    }, summary


def _key(stage: str, payload: dict[str, Any]) -> dict[str, Any]:
    model = payload["model_profile"]
    topology = payload["topology"]
    workload = payload["workload"]
    if stage == "agg":
        shape = f"ep{topology['ep_size']}"
        batch = workload["logical_tokens_per_rank"]
        microbatches = workload.get("microbatches") or 1
        backend = _BACKENDS[payload["primary_backend"]]
    else:
        shape = f"{topology['ag_size']}A{topology['eg_size']}F"
        batch = workload["sequences_per_ag_rank"]
        microbatches = workload["microbatches"]
        backend = _BACKENDS[payload["moe_backend"]]
    return {
        "model_path": model["model_path"],
        "system": payload["system_label"],
        "stage": stage,
        "topology": shape,
        "logical_batch_per_source_rank": batch,
        "mtp_nextn": workload["mtp_nextn"],
        "microbatches": microbatches,
        "moe_layers": workload["layers"],
        "routed_topk": model["routed_top_k"],
        "moe_precision": model["moe_quant_contract"],
        "moe_backend": backend,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aisimulate.fastafd_profile",
        description="Build an AISimulate FastAFD MoE stage profile from raw MegaMoE benchmark results.",
    )
    parser.add_argument("results_dir", help="directory of raw benchmark result JSON files")
    parser.add_argument("--repository", required=True, help="git repository URL that produced the results")
    parser.add_argument("--routing", default="balanced", help="routing distribution to keep (default: balanced)")
    parser.add_argument("--output", required=True, help="profile JSON path to write")
    args = parser.parse_args(argv)
    profile, summary = build_profile(args.results_dir, repository=args.repository, routing=args.routing)
    output = Path(args.output)
    output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    FastAFDMoEStageProfile.load(output)
    print(f"wrote {output}: {summary['accepted']} entries", file=sys.stderr)
    for reason in ("skipped_routing", "skipped_unstable", "skipped_correctness"):
        if summary[reason]:
            print(f"  {reason}: {summary[reason]}", file=sys.stderr)
    return 0


__all__ = [
    "FASTAFD_PROFILE_SCHEMA",
    "FastAFDMoEStageKey",
    "FastAFDMoEStageMeasurement",
    "FastAFDMoEStageProfile",
    "FastAFDProfileSource",
    "build_profile",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
