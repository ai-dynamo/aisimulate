# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate versioned GLM candidate points; no runtime admission is inferred.

The two point files use the existing planner's schema 3. Inventory and
qualification-template JSON are planning receipts, never collector perf data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from aisimulate_core.sdk.glm53flash import BACKEND_REVISIONS, MODEL_REVISIONS

VERSION = "glm53flash_sampling_v1"
ROLES = ("calibration", "holdout")
MAX_CONTEXT = 131072
MAX_BATCH = 32
MAX_PREFILL = 8192
CORPUS_DIR = Path(__file__).with_name("glm53flash_corpora")


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class SamplingOptions:
    batches: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    contexts: tuple[int, ...] = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
    cached_queries: tuple[int, ...] = (32, 256)
    full_prefill_totals: tuple[int, ...] = (128, 512, 2048, 8192)
    edge_batches: tuple[int, ...] = (1, 4, 32)
    block_anchors: tuple[int, ...] = (128, 4352)
    chunk_anchors: tuple[int, ...] = (2048, 8192)
    graph_batch_anchors: tuple[int, ...] = (4, 8, 16, 32)

    def validate(self):
        for name, values in asdict(self).items():
            if not values or any(type(value) is not int or value <= 0 for value in values):
                raise ValueError(f"{name} requires nonempty positive integer values")
            if len(set(values)) != len(values) or tuple(sorted(values)) != values:
                raise ValueError(f"{name} must be unique and ascending")
        if self.batches != (1, 2, 4, 8, 16, 32):
            raise ValueError("formal batch anchors must retain B1/2/4/8/16/32")
        if not set(self.edge_batches) <= set(self.batches):
            raise ValueError("edge batches must be drawn from formal batch anchors")
        if any(total % batch for total in self.full_prefill_totals for batch in self.batches):
            raise ValueError("full prefill totals must support homogeneous formal batches")
        if 1024 not in self.contexts or max(self.contexts) != MAX_CONTEXT or 65536 not in self.contexts:
            raise ValueError("context anchors must retain 1K, 64K, and inclusive128K coverage")
        if max(self.full_prefill_totals) > MAX_PREFILL or max(self.cached_queries) * max(self.batches) > MAX_PREFILL:
            raise ValueError("declared prefill axes exceed the 8192 new-token budget")
        if max(self.cached_queries) >= min(self.contexts):
            raise ValueError("cached-prefill context anchors must include a positive prefix")
        if any(value > MAX_BATCH for value in self.graph_batch_anchors):
            raise ValueError("graph batch anchors cannot exceed batch32")


def point_key(phase: str, batch: int, query: int, prefix: int) -> tuple:
    return phase, batch, 0 if phase == "decode" else batch * query, batch * prefix


def payload_point(key: tuple) -> dict:
    phase, batch, query, prefix = key
    return {
        "batch_size": batch,
        "total_kv_read_tokens": prefix,
        **({"total_prefill_tokens": query} if phase == "prefill" else {}),
    }


def context_length(key: tuple) -> int:
    phase, batch, query, prefix = key
    return (prefix + (batch if phase == "decode" else query)) // batch


def generate(options: SamplingOptions | None = None) -> dict:
    options = options or SamplingOptions()
    options.validate()
    calibration, lines, excluded = {}, [], []

    def add_line(phase, batch, query, prefix, axis, values, family):
        accepted = []
        for value in sorted(set(values)):
            q, p = (value, prefix) if axis == "query" else (query, value)
            key = point_key(phase, batch, q, p)
            reason = None
            if not 1 <= batch <= MAX_BATCH:
                reason = "declared_batch_limit"
            elif q < 1 or p < 0 or (phase == "decode" and p < 1):
                reason = "declared_positive_query_nonnegative_prefix"
            elif q + p > MAX_CONTEXT:
                reason = "declared_inclusive_context_limit"
            elif phase == "prefill" and q * batch > MAX_PREFILL:
                reason = "declared_total_new_token_limit"
            if reason:
                excluded.append(
                    {"family": family, "phase": phase, "batch_size": batch, "query": q, "prefix": p, "reason": reason}
                )
                continue
            calibration.setdefault(key, set()).add(family)
            accepted.append((value, key))
        lines.append((phase, batch, query, prefix, axis, accepted, family))

    for batch in options.batches:
        add_line("decode", batch, 1, 0, "prefix", [length - 1 for length in options.contexts], "context_band")
        for query in options.cached_queries:
            add_line(
                "prefill",
                batch,
                query,
                0,
                "prefix",
                [length - query for length in options.contexts],
                "cached_prefill_context_band",
            )
        add_line(
            "prefill",
            batch,
            0,
            0,
            "query",
            [total // batch for total in options.full_prefill_totals],
            "full_prefill_new_token_axis",
        )

    # These are declared probe anchors, not assertions about a particular
    # backend's active block/chunk/graph configuration. Runtime receipts decide.
    offsets = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
    for batch in options.edge_batches:
        for block in options.block_anchors:
            for phase, query in (("decode", 1), ("prefill", 32)):
                add_line(
                    phase,
                    batch,
                    query,
                    0,
                    "prefix",
                    [block + offset for offset in offsets],
                    f"prefix_block_{block}_indexpool_mod4",
                )
        for total in options.chunk_anchors:
            if total % batch:
                raise ValueError("chunk anchors must be divisible by edge batches")
            add_line(
                "prefill",
                batch,
                0,
                0,
                "query",
                [total // batch + offset for offset in offsets],
                f"new_tokens_chunk_{total}",
            )
    for anchor in options.graph_batch_anchors:
        for batch in (anchor - 1, anchor, anchor + 1):
            add_line("decode", batch, 1, 0, "prefix", (1023, 4095), f"decode_graph_batch_{anchor}")

    # Every holdout is strictly between two actual calibration points along
    # one physical axis, with batch and other coordinates unchanged. Checking
    # against the complete calibration set prevents cross-family leakage.
    holdout, no_interior = {}, []
    for phase, batch, query, prefix, axis, accepted, family in lines:
        for (lower, lower_key), (upper, upper_key) in pairwise(accepted):
            available = [
                value
                for value in range(lower + 1, upper)
                if point_key(phase, batch, value if axis == "query" else query, prefix if axis == "query" else value)
                not in calibration
            ]
            if not available:
                no_interior.append({"family": family, "lower": lower_key, "upper": upper_key})
                continue
            midpoint = (lower + upper) // 2
            value = min(available, key=lambda candidate: (abs(candidate - midpoint), candidate))
            key = point_key(phase, batch, value if axis == "query" else query, prefix if axis == "query" else value)
            record = holdout.setdefault(key, {"families": set(), "brackets": []})
            record["families"].add(family)
            record["brackets"].append({"axis": axis, "lower": lower_key, "upper": upper_key})
    if set(calibration) & set(holdout):
        raise AssertionError("calibration/holdout geometry collision")
    corpora = {}
    for role in ROLES:
        raw = (CORPUS_DIR / f"{role}.txt").read_bytes()
        if len(raw) < 1000:
            raise ValueError("original corpus is unexpectedly short")
        corpora[role] = {"file": f"{role}.txt", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    if corpora["calibration"]["sha256"] == corpora["holdout"]["sha256"]:
        raise ValueError("calibration and holdout corpora must be distinct")
    candidate_id = lambda role, key: role + "-" + digest([VERSION, role, key])[:20]
    inventory, payloads = {}, {}
    for role, points in (("calibration", calibration), ("holdout", holdout)):
        payloads[role] = {"schema_version": 3, "prefill": [], "decode": []}
        inventory[role] = []
        for key in sorted(points):
            point = payload_point(key)
            payloads[role][key[0]].append(point)
            families = calibration[key] if role == "calibration" else holdout[key]["families"]
            brackets = (
                []
                if role == "calibration"
                else [
                    {
                        "axis": pair["axis"],
                        "lower": candidate_id("calibration", pair["lower"]),
                        "upper": candidate_id("calibration", pair["upper"]),
                    }
                    for pair in holdout[key]["brackets"]
                ]
            )
            inventory[role].append(
                {
                    "candidate_id": candidate_id(role, key),
                    "phase": key[0],
                    "point": point,
                    "inclusive_context": context_length(key),
                    "families": sorted(families),
                    "calibration_brackets": brackets,
                    "qualification_status": "NOT_EVALUATED",
                }
            )
    pins = {"model_revisions": MODEL_REVISIONS, "backend_revisions": BACKEND_REVISIONS}
    campaign_id = digest(
        {"version": VERSION, "options": asdict(options), "corpora": corpora, "pins": pins, "payloads": payloads}
    )
    cells = []
    for backend in ("vllm", "sglang"):
        for quant in ("fp8", "nvfp4"):
            for tp in (2, 4):
                cell_id = f"{backend}-{quant}-tp{tp}"
                cells.append(
                    {
                        "cell_id": cell_id,
                        "backend": backend,
                        "checkpoint_format": quant,
                        "tp": tp,
                        "dp": 1,
                        "pp": 1,
                        "cp": 1,
                        "ep": 1,
                    }
                )
    return {
        "schema": VERSION,
        "campaign_id": campaign_id,
        "status": "CANDIDATES_NOT_QUALIFIED",
        "accuracy_acceptance": "NOT_EVALUATED",
        "pins": pins,
        "options": asdict(options),
        "corpora": corpora,
        "request_namespaces": {role: f"glm53-{campaign_id[:16]}-{role}" for role in ROLES},
        "cells": cells,
        "points": inventory,
        "payloads": payloads,
        "payload_sha256": {role: digest(payloads[role]) for role in ROLES},
        "excluded_by_declared_bounds": excluded,
        "calibration_intervals_without_disjoint_integer_holdout": no_interior,
    }


def write_bundle(destination: Path, result: dict) -> None:
    """Write immutable candidate inputs plus unfilled receipt slots; never admit."""
    destination.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (destination / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")

    inventory = {key: value for key, value in result.items() if key != "payloads"}
    write("candidate-inventory.json", inventory)
    for role in ROLES:
        # These exact schema3 bytes are accepted by FPMCollectionOptions.
        (destination / f"{role}-points.json").write_text(canonical(result["payloads"][role]))
        (destination / f"{role}.txt").write_bytes((CORPUS_DIR / f"{role}.txt").read_bytes())
    write(
        "qualification-template.json",
        {
            "schema": "glm53flash_qualification_references_v1",
            "campaign_id": result["campaign_id"],
            "candidate_inventory_sha256": hashlib.sha256(
                (destination / "candidate-inventory.json").read_bytes()
            ).hexdigest(),
            "status": "NOT_EVALUATED",
            "receipt_format": {"path": "<actual native or capacity receipt path>", "sha256": "<actual bytes SHA256>"},
            "cells": [
                {
                    **cell,
                    "capacity_receipts": [],
                    "runtime_configuration_receipt": None,
                    "candidates": [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "role": role,
                            "status": "NOT_EVALUATED",
                            "native_schedule_receipts": [],
                            "failure_receipts": [],
                        }
                        for role in ROLES
                        for candidate in result["points"][role]
                    ],
                }
                for cell in result["cells"]
            ],
        },
    )


def int_list(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for name, default in asdict(SamplingOptions()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int_list, default=default)
    args = parser.parse_args(argv)
    result = generate(SamplingOptions(**{name: getattr(args, name) for name in asdict(SamplingOptions())}))
    write_bundle(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "campaign_id": result["campaign_id"],
                "status": result["status"],
                "points_per_cell": {role: len(result["points"][role]) for role in ROLES},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
