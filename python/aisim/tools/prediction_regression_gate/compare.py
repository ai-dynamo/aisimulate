# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Old-vs-new snapshot comparison for the prediction-regression-gate gate.

Pure CSV diffing — no SDK imports, so the comparison itself is fast and
testable. "Old" is the snapshot collected on the PR's base revision, "new"
the one collected on the PR head; each side runs its own copy of the
collectors, so grid/config changes show up as added/removed rows.

Diff categories:

  REGRESSION   OK -> DATA_MISS / INVALID (a working combo stopped working).
               The only blocking category.
  GAIN         DATA_MISS / INVALID -> OK (coverage gained)
  DRIFT        OK -> OK but |rel change| > rtol
  STATUS_CHANGE  non-OK status or error type changed
  ROWS_ADDED / ROWS_REMOVED  grid, model list, or config list changed
  RELABELED    old rows of a declared model rename, compared under the new
               identifier (one entry per rename and combo)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

KEY_FIELDS = ("model", "tp", "pp", "adp", "moe_tp", "moe_ep", "quant", "phase", "bs", "isl")
DEFAULT_RTOL = 1e-4

# Only "was working, stopped working" blocks the gate; everything else is
# reported for review. Intentional modeling changes therefore need no
# acknowledgment ritual — the report itself is the review artifact.
BLOCKING_CATEGORIES = {"REGRESSION"}

# Declared renames of bundled model identifiers (old -> new) for a PR that
# changes which identifier names a checkpoint without changing how that
# checkpoint is modeled. Old rows are diffed against the same model under its
# new identifier, so every regression and drift check still applies; without
# this, the rename itself would read as OK -> DATA_MISS on whichever combos
# lack data for the precision the identifier now names. A table applies to a
# combo only while the old side has every retired identifier and none of the
# introduced ones, so it goes inert once the rename reaches the base and its
# entries can then be deleted. A new side that lacks an identifier the table
# maps onto is rejected, so an incomplete snapshot cannot hide behind the rename.
MODEL_RELABELS: dict[str, str] = {
    # AISIM-2124: zai-org/GLM-5.3 is Z.AI's FP8 release and the BF16 one is
    # zai-org/GLM-5.3-BF16; there is no zai-org/GLM-5.3-FP8.
    "zai-org/GLM-5.3-FP8": "zai-org/GLM-5.3",
    "zai-org/GLM-5.3": "zai-org/GLM-5.3-BF16",
}


@dataclass
class Diff:
    combo: str
    key: tuple
    category: str
    detail: str

    def render(self) -> str:
        key_text = "/".join(str(part) for part in self.key)
        return f"[{self.category}] {self.combo} :: {key_text} :: {self.detail}"


@dataclass
class ComboResult:
    combo: str
    diffs: list[Diff] = field(default_factory=list)
    rows_compared: int = 0

    @property
    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for diff in self.diffs:
            counts[diff.category] = counts.get(diff.category, 0) + 1
        return counts


def load_rows(path: Path) -> dict[tuple, dict]:
    with path.open(newline="") as f:
        rows = {}
        for row in csv.DictReader(f):
            key = tuple(row[k] for k in KEY_FIELDS)
            if key in rows:
                raise ValueError(f"{path}: duplicate snapshot identity {key}")
            rows[key] = row
        return rows


def relabel_old_rows(
    old: dict[tuple, dict], new: dict[tuple, dict], relabels: dict[str, str]
) -> tuple[dict[tuple, dict], dict[str, int]]:
    """Re-key old rows by a declared rename table, if it applies to this pair.

    Returns the (possibly re-keyed) old rows and the number of old rows moved
    per renamed identifier; an empty count means the table did not apply.
    Raises ValueError when the table applies but the new side lacks one of the
    identifiers it maps onto: an incomplete snapshot must not pass as a rename.
    """
    if len(set(relabels.values())) != len(relabels):
        raise ValueError(f"model relabels must be one-to-one: {relabels}")
    old_models = {key[0] for key in old}
    new_models = {key[0] for key in new}
    retired = relabels.keys() - set(relabels.values())
    introduced = set(relabels.values()) - relabels.keys()
    if not retired <= old_models or retired & new_models or introduced & old_models:
        return old, {}
    missing = set(relabels.values()) - new_models
    if missing:
        raise ValueError(f"declared model rename is incomplete: the new side lacks {sorted(missing)}")

    relabeled: dict[tuple, dict] = {}
    moved: dict[str, int] = {}
    for key, row in old.items():
        model = key[0]
        if model in relabels:
            moved[model] = moved.get(model, 0) + 1
            key = (relabels[model], *key[1:])
        relabeled[key] = row
    return relabeled, moved


def compare_combo(
    combo: str,
    old_path: Path,
    new_path: Path,
    rtol: float = DEFAULT_RTOL,
    relabels: dict[str, str] | None = None,
) -> ComboResult:
    result = ComboResult(combo=combo)
    old = load_rows(old_path)
    new = load_rows(new_path)

    relabels = MODEL_RELABELS if relabels is None else relabels
    old, moved = relabel_old_rows(old, new, relabels)
    for model, count in sorted(moved.items()):
        result.diffs.append(Diff(combo, (model,), "RELABELED", f"{count} old rows compared as {relabels[model]}"))

    for key in old.keys() - new.keys():
        result.diffs.append(Diff(combo, key, "ROWS_REMOVED", "row present on old side, absent on new"))
    for key in new.keys() - old.keys():
        result.diffs.append(Diff(combo, key, "ROWS_ADDED", "row produced on new side, absent on old"))

    for key in old.keys() & new.keys():
        result.rows_compared += 1
        result.diffs.extend(_diff_statuses(combo, key, old[key], new[key], value_fields=("value_ms",), rtol=rtol))
    return result


def compare_tier2(old_path: Path, new_path: Path, rtol: float = DEFAULT_RTOL) -> ComboResult:
    """Diff two tier-2 snapshots (tier2.csv: id,status,ttft_ms,tpot_ms)."""
    combo = "tier2"
    result = ComboResult(combo=combo)

    def load(path: Path) -> dict[tuple, dict]:
        with path.open(newline="") as f:
            return {(row["id"],): row for row in csv.DictReader(f)}

    old = load(old_path)
    new = load(new_path)

    for key in old.keys() - new.keys():
        result.diffs.append(Diff(combo, key, "ROWS_REMOVED", "config present on old side, absent on new"))
    for key in new.keys() - old.keys():
        result.diffs.append(Diff(combo, key, "ROWS_ADDED", "config produced on new side, absent on old"))

    for key in old.keys() & new.keys():
        result.rows_compared += 1
        result.diffs.extend(
            _diff_statuses(combo, key, old[key], new[key], value_fields=("ttft_ms", "tpot_ms"), rtol=rtol)
        )
    return result


def _diff_statuses(
    combo: str, key: tuple, old: dict, new: dict, value_fields: tuple[str, ...], rtol: float
) -> list[Diff]:
    """Shared status/value diff logic for one row pair.

    Tier-1 rows carry err (exception type) and one value; tier-2 rows encode
    the exception type in status itself and carry two values.
    """
    old_status, new_status = old["status"], new["status"]
    old_err, new_err = old.get("err", ""), new.get("err", "")
    diffs: list[Diff] = []

    if old_status == "OK" and new_status != "OK":
        old_values = ", ".join(f"{old[f]} ms" for f in value_fields)
        diffs.append(Diff(combo, key, "REGRESSION", f"OK ({old_values}) -> {new_status} {new_err}".rstrip()))
    elif old_status != "OK" and new_status == "OK":
        new_values = ", ".join(f"{new[f]} ms" for f in value_fields)
        diffs.append(Diff(combo, key, "GAIN", f"{old_status} {old_err} -> OK ({new_values})".replace("  ", " ")))
    elif old_status == "OK" and new_status == "OK":
        for value_field in value_fields:
            old_value, new_value = float(old[value_field]), float(new[value_field])
            denom = max(abs(old_value), 1e-9)
            rel = abs(new_value - old_value) / denom
            if rel > rtol:
                diffs.append(
                    Diff(
                        combo,
                        key,
                        "DRIFT",
                        f"{value_field} {old_value:.6f} -> {new_value:.6f} ms (rel {rel:.2%})",
                    )
                )
    elif old_status != new_status or old_err != new_err:
        diffs.append(
            Diff(combo, key, "STATUS_CHANGE", f"{old_status} {old_err} -> {new_status} {new_err}".replace("  ", " "))
        )
    return diffs


def summarize(results: list[ComboResult], max_examples: int = 20) -> str:
    total_counts: dict[str, int] = {}
    for result in results:
        for category, count in result.counts.items():
            total_counts[category] = total_counts.get(category, 0) + count
    lines = [f"compared {sum(r.rows_compared for r in results)} rows across {len(results)} combos"]
    if not total_counts:
        lines.append("no differences")
        return "\n".join(lines)
    lines.append(f"differences: {total_counts}")
    shown = 0
    for result in results:
        for diff in result.diffs:
            if shown >= max_examples:
                lines.append(f"... and {sum(total_counts.values()) - shown} more (see report artifact)")
                return "\n".join(lines)
            lines.append(diff.render())
            shown += 1
    return "\n".join(lines)


def write_report(results: list[ComboResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["combo", "key", "category", "detail"])
        for result in results:
            for diff in result.diffs:
                writer.writerow([diff.combo, "/".join(str(p) for p in diff.key), diff.category, diff.detail])
