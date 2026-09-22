# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict reader for FastAFD measured MoE-stage profiles.

The JSON contract is adapted from ``liz-badada/FastAFD`` at commit
``0b9bce2bdbee04ace2673cfc3118572fec484fe4`` (MIT), original path
``scripts/experiments/afd/summarize_megamoe_model_results.py``:
https://github.com/liz-badada/FastAFD/blob/0b9bce2bdbee04ace2673cfc3118572fec484fe4/scripts/experiments/afd/summarize_megamoe_model_results.py
This module intentionally does not turn a profile into silicon data: a
profile is an external, complete stage measurement whose topology and
provenance must match exactly.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PROFILE_SCHEMA = "aic.afd-moe-stage-profile.v3"
Stage = Literal["agg", "afd"]
_AFD_TOPOLOGY = re.compile(r"^[1-9][0-9]*A[1-9][0-9]*F$")
_AGG_TOPOLOGY = re.compile(r"^ep[1-9][0-9]*$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FastAFDMoEStageKey:
    """All inputs that must match before using a measured stage time."""

    model_path: str
    system: str
    stage: Stage
    topology: str
    logical_batch_per_source_rank: int
    mtp_nextn: int
    microbatches: int
    moe_layers: int
    routed_topk: int
    moe_precision: str
    moe_backend: str


@dataclass(frozen=True)
class FastAFDMoEStageMeasurement:
    """One externally measured stage and its immutable provenance."""

    key: FastAFDMoEStageKey
    model_profile: str
    latency_ms: float
    stable: bool
    correctness: bool | None
    evidence: str
    source_commit: str
    source_tree_sha256: str
    source_result: str


class FastAFDMoEStageProfile:
    """Exact-only index over FastAFD full-stage measurements.

    The profile has no interpolation or hardware transfer policy.  Those are
    separate modeling decisions and must not be silently inferred from a B200
    measurement.
    """

    def __init__(self, entries: tuple[FastAFDMoEStageMeasurement, ...], *, source: Path) -> None:
        self.entries = entries
        self.source = source
        self._by_key: dict[FastAFDMoEStageKey, FastAFDMoEStageMeasurement] = {}
        for entry in entries:
            if entry.key in self._by_key:
                raise ValueError(f"duplicate FastAFD MoE stage profile key: {entry.key}")
            self._by_key[entry.key] = entry

    @classmethod
    def load(cls, path: str | Path) -> FastAFDMoEStageProfile:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("FastAFD MoE stage profile root must be an object")
        if payload.get("schema") != PROFILE_SCHEMA:
            raise ValueError(f"unsupported FastAFD MoE stage profile schema: {payload.get('schema')!r}")
        if payload.get("lookup_policy") != "exact-only":
            raise ValueError("FastAFD MoE stage profile lookup_policy must be 'exact-only'")
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError("FastAFD MoE stage profile entries must be a non-empty list")
        return cls(tuple(_parse_entry(raw, index) for index, raw in enumerate(raw_entries)), source=source)

    def find(self, key: FastAFDMoEStageKey) -> FastAFDMoEStageMeasurement | None:
        """Return a measurement only for an exact workload and topology match."""

        return self._by_key.get(key)

    def require(self, key: FastAFDMoEStageKey) -> FastAFDMoEStageMeasurement:
        measurement = self.find(key)
        if measurement is None:
            raise KeyError(f"no exact FastAFD MoE stage measurement for {key}")
        return measurement


def _parse_entry(raw: Any, index: int) -> FastAFDMoEStageMeasurement:
    if not isinstance(raw, dict):
        raise TypeError(f"profile entry {index} must be an object")

    stage = _string(raw, "stage", index)
    if stage not in {"agg", "afd"}:
        raise ValueError(f"profile entry {index} has unsupported stage {stage!r}")
    topology = _string(raw, "topology", index)
    expected_pattern = _AGG_TOPOLOGY if stage == "agg" else _AFD_TOPOLOGY
    if not expected_pattern.fullmatch(topology):
        raise ValueError(f"profile entry {index} has invalid {stage} topology {topology!r}")

    validation = raw.get("validation")
    if not isinstance(validation, dict):
        raise TypeError(f"profile entry {index} validation must be an object")
    stable = validation.get("stable")
    if stable is not True:
        raise ValueError(f"profile entry {index} must have validation.stable=true")
    correctness = validation.get("correctness")
    if correctness is not None and not isinstance(correctness, bool):
        raise TypeError(f"profile entry {index} validation.correctness must be bool or null")

    source = raw.get("source")
    if not isinstance(source, dict):
        raise TypeError(f"profile entry {index} source must be an object")
    source_commit = _string(source, "commit", index, prefix="source.")
    source_tree_sha256 = _string(source, "source_tree_sha256", index, prefix="source.")
    if not _SHA1.fullmatch(source_commit):
        raise ValueError(f"profile entry {index} source.commit must be a full lowercase SHA-1")
    if not _SHA256.fullmatch(source_tree_sha256):
        raise ValueError(f"profile entry {index} source.source_tree_sha256 must be a lowercase SHA-256")

    key = FastAFDMoEStageKey(
        model_path=_string(raw, "model_path", index),
        system=_string(raw, "system", index),
        stage=stage,
        topology=topology,
        logical_batch_per_source_rank=_positive_int(raw, "logical_batch_per_source_rank", index),
        mtp_nextn=_nonnegative_int(raw, "mtp_nextn", index),
        microbatches=_positive_int(raw, "microbatches", index),
        moe_layers=_positive_int(raw, "moe_layers", index),
        routed_topk=_positive_int(raw, "routed_topk", index),
        moe_precision=_string(raw, "moe_precision", index),
        moe_backend=_string(raw, "moe_backend", index),
    )
    latency_ms = _positive_float(raw, "latency_ms", index)
    return FastAFDMoEStageMeasurement(
        key=key,
        model_profile=_string(raw, "model_profile", index),
        latency_ms=latency_ms,
        stable=True,
        correctness=correctness,
        evidence=_string(validation, "evidence", index, prefix="validation."),
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        source_result=_string(source, "result", index, prefix="source."),
    )


def _string(mapping: dict[str, Any], field: str, index: int, *, prefix: str = "") -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"profile entry {index} {prefix}{field} must be a non-empty string")
    return value


def _positive_int(mapping: dict[str, Any], field: str, index: int) -> int:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"profile entry {index} {field} must be a positive integer")
    return value


def _nonnegative_int(mapping: dict[str, Any], field: str, index: int) -> int:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"profile entry {index} {field} must be a non-negative integer")
    return value


def _positive_float(mapping: dict[str, Any], field: str, index: int) -> float:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"profile entry {index} {field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"profile entry {index} {field} must be a finite positive number")
    return result


__all__ = [
    "PROFILE_SCHEMA",
    "FastAFDMoEStageKey",
    "FastAFDMoEStageMeasurement",
    "FastAFDMoEStageProfile",
]
