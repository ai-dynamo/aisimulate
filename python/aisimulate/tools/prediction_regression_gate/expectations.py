# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact, temporary AIC-2004 DSA data gaps; never change prediction results.

The CSV is a reviewed list of individual snapshot identities, not a pattern.
Both the identity and the native missing-op diagnostic must match. Restored
coverage blocks as XPASS until the data follow-up disables the waiver, keeping
the same case identities as required passing checks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

from tools.prediction_regression_gate import compare

DEFAULT_PATH = Path(__file__).with_name("expected_data_misses.csv")
ISSUE = "https://linear.app/nvidia/issue/AIC-2004"
SYSTEMS = frozenset({"b300_sxm", "gb200", "gb300", "h100_sxm", "h200_sxm"})
MODELS = frozenset({"zai-org/GLM-5-FP8", "zai-org/GLM-5.1-FP8", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3-FP8"})
HEADER = ("combo", *compare.KEY_FIELDS)


@dataclass(frozen=True)
class ExpectedDataMiss:
    combo: str
    key: tuple[str, ...]
    expected_failure_enabled: bool = True

    def matches_cause(self, row: dict) -> bool:
        phase = {"ctx": "context", "gen": "generation"}[self.key[7]]
        diagnostic = (
            f"perf database error: {phase} DSA module data missing for DsaKey {{ "
            'architecture: "GlmMoeDsaForCausalLM", fmha_quant: "bfloat16", '
            'kv_quant: "fp8", gemm_quant: "fp8" }'
        )
        return (
            row.get("status") == "DATA_MISS"
            and row.get("err", "") == ""
            and row.get("value_ms", "") == ""
            and re.fullmatch(re.escape(diagnostic) + r"(?: at [^\r\n]+)?", row.get("data_miss_detail", "")) is not None
        )


def load_expectations(path: Path = DEFAULT_PATH) -> list[ExpectedDataMiss]:
    """Reject duplicate, wildcard, malformed and out-of-authorized-scope entries."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(HEADER):
            raise ValueError(f"{path}: expected exact columns {HEADER}")
        result = []
        seen = set()
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: malformed expectation row")
            combo = row["combo"].split("/")
            key = tuple(row[k] for k in compare.KEY_FIELDS)
            if (
                len(combo) != 3
                or combo[0] not in SYSTEMS
                or combo[1:] != ["vllm", "0.24.0.csv"]
                or row["model"] not in MODELS
                or row["quant"] != "fp8"
                or row["phase"] not in {"ctx", "gen"}
                or any(
                    not row[k].isdigit() or int(row[k]) <= 0
                    for k in ("tp", "pp", "adp", "moe_tp", "moe_ep", "bs", "isl")
                )
            ):
                raise ValueError(f"{path}: expectation outside the authorized vLLM DSA FP8 scope: {row}")
            identity = (row["combo"], key)
            if identity in seen:
                raise ValueError(f"{path}: duplicate expectation {identity}")
            seen.add(identity)
            result.append(ExpectedDataMiss(*identity))
    manifest_path = path.with_suffix(".json")
    if result or path == DEFAULT_PATH or manifest_path.is_file():
        if not manifest_path.is_file():
            raise ValueError(f"{path}: non-empty expectations require a reviewed manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["case_count"] != len(result)
            or manifest["csv_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError(f"{path}: reviewed expectation manifest count/hash mismatch")
        enabled = manifest.get("expected_failure_enabled")
        if not isinstance(enabled, bool):
            raise ValueError(f"{path}: expected_failure_enabled must be an explicit boolean")
        result = [replace(entry, expected_failure_enabled=enabled) for entry in result]
    return result


def apply_expectations(
    results: list[compare.ComboResult], old_dir: Path, new_dir: Path, expectations: list[ExpectedDataMiss]
) -> None:
    """Annotate exact misses, preserving raw status/cause and every other failure.

    With exceptions enabled, a combo absent on both sides is outside a scoped
    local run. Reopened cases require their candidate even when the whole combo
    is missing. Within an observed combo, a missing row always blocks.
    After the fix merges, DATA_MISS -> DATA_MISS stays explicitly visible; it
    must retain this same native DSA diagnostic on both sides.
    """
    by_combo = {result.combo: result for result in results}
    snapshots = {}
    for expectation in expectations:
        combo, key = expectation.combo, expectation.key
        if combo not in by_combo:
            if expectation.expected_failure_enabled:
                continue
            result = compare.ComboResult(combo)
            by_combo[combo] = result
            results.append(result)
        if combo not in snapshots:
            snapshots[combo] = tuple(
                compare.load_rows(path) if path.is_file() else {} for path in (old_dir / combo, new_dir / combo)
            )
        old, new = (rows.get(key) for rows in snapshots[combo])
        result = by_combo[combo]
        fields = {
            f"{side}_{field}": (row or {}).get(field, "")
            for side, row in (("old", old), ("new", new))
            for field in ("status", "err", "data_miss_detail")
        }
        if not expectation.expected_failure_enabled:
            try:
                value = float((new or {}).get("value_ms", ""))
            except (TypeError, ValueError):
                value = float("nan")
            if (
                new is not None
                and new["status"] == "OK"
                and new.get("err", "") == ""
                and new.get("data_miss_detail", "") == ""
                and math.isfinite(value)
                and value > 0
            ):
                # Reopened cases require measured coverage. Normal comparison
                # still reports GAIN/DRIFT; restored rows are no longer XPASS.
                continue
            category = "REOPENED_FAILURE"
            detail = f"DSA FP8 data follow-up requires candidate OK with valid latency; exception disabled ({ISSUE})"
        elif new is not None and new["status"] == "OK":
            category = "XPASS"
            detail = f"DSA FP8 coverage restored; disable the expected-failure waiver, retaining this case ({ISSUE})"
        elif old is None or new is None:
            category = "EXPECTATION_MISMATCH"
            detail = "Expected DSA case is absent from a snapshot; restore its explicit coverage"
        elif (
            expectation.matches_cause(new)
            and old.get("err", "") == ""
            and (old["status"] == "OK" or expectation.matches_cause(old))
        ):
            category = "EXPECTED_DATA_MISS"
            detail = f"{old['status']} -> DATA_MISS: awaiting this GPU's measured DSA FP8 data ({ISSUE})"
            # Remove only this row's exact missing-data transition. Never
            # suppress INVALID, another missing op, or a tier-2 regression.
            result.diffs[:] = [d for d in result.diffs if not (d.key == key and d.category == "REGRESSION")]
        else:
            category = "EXPECTATION_MISMATCH"
            detail = "Status/error/cause differs from the reviewed DSA FP8 data gap; no exception applied"
        result.diffs.append(compare.Diff(combo, key, category, detail, **fields))
