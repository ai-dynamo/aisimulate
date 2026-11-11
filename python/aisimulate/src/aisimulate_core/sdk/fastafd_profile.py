# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read externally measured FastAFD MoE stage profiles."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

FASTAFD_PROFILE_SCHEMA = "aisimulate.fastafd-moe-stage.v1"
FastAFDStage = Literal["agg", "afd"]

_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_AFD_TOPOLOGY = re.compile(r"[1-9][0-9]*A[1-9][0-9]*F\Z")
_AGG_TOPOLOGY = re.compile(r"ep[1-9][0-9]*\Z")
_KEY_FIELDS = {
    "model_path",
    "system",
    "stage",
    "topology",
    "logical_batch_per_source_rank",
    "mtp_nextn",
    "microbatches",
    "moe_layers",
    "routed_topk",
    "moe_precision",
    "moe_backend",
}
_MEASUREMENT_FIELDS = {
    "scope",
    "method",
    "method_version",
    "statistic",
    "sample_count",
    "cv_percent",
    "routing",
    "source_tree_sha256",
    "raw_artifact",
    "raw_sha256",
}


@dataclass(frozen=True)
class FastAFDProfileSource:
    """The code that produced every measurement in a profile."""

    repository: str
    commit: str
    benchmark: str
    boundary: str


@dataclass(frozen=True)
class FastAFDMoEStageKey:
    model_path: str
    system: str
    stage: FastAFDStage
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
    key: FastAFDMoEStageKey
    model_profile: str
    latency_ms: float
    correctness: bool | None
    source: FastAFDProfileSource
    method: str
    method_version: str
    statistic: str
    sample_count: int
    cv_percent: float
    routing: str
    source_tree_sha256: str
    raw_artifact: str
    raw_sha256: str

    def provenance(self) -> dict[str, Any]:
        return {
            "provider": "fastafd",
            "schema": FASTAFD_PROFILE_SCHEMA,
            "repository": self.source.repository,
            "source_commit": self.source.commit,
            "benchmark": self.source.benchmark,
            "boundary": self.source.boundary,
            "model_profile": self.model_profile,
            "method": self.method,
            "method_version": self.method_version,
            "statistic": self.statistic,
            "sample_count": self.sample_count,
            "cv_percent": self.cv_percent,
            "routing": self.routing,
            "source_tree_sha256": self.source_tree_sha256,
            "raw_artifact": self.raw_artifact,
            "raw_sha256": self.raw_sha256,
            "correctness": self.correctness,
        }


class FastAFDMoEStageProfile:
    def __init__(
        self,
        entries: tuple[FastAFDMoEStageMeasurement, ...],
        *,
        source: FastAFDProfileSource,
        path: Path,
        profile_sha256: str,
    ) -> None:
        self.entries = entries
        self.source = source
        self.path = path
        self.profile_sha256 = profile_sha256
        self._entries = {entry.key: entry for entry in entries}
        if len(self._entries) != len(entries):
            raise ValueError("duplicate FastAFD MoE stage key")

    @classmethod
    def load(cls, path: str | Path) -> FastAFDMoEStageProfile:
        resolved = Path(path).expanduser().resolve()
        raw = resolved.read_bytes()
        payload = json.loads(raw, object_pairs_hook=_unique_object)
        root = _object(payload, {"schema", "source", "lookup_policy", "entries"}, "profile")
        if root["schema"] != FASTAFD_PROFILE_SCHEMA:
            raise ValueError(f"unsupported FastAFD profile schema: {root['schema']!r}")
        if root["lookup_policy"] != "exact-only":
            raise ValueError("FastAFD profile lookup_policy must be 'exact-only'")
        origin = _object(root["source"], {"repository", "commit", "benchmark", "boundary"}, "source")
        source = FastAFDProfileSource(
            repository=_text(origin["repository"], "source.repository"),
            commit=_digest(origin["commit"], _SHA1, "source.commit"),
            benchmark=_text(origin["benchmark"], "source.benchmark"),
            boundary=_text(origin["boundary"], "source.boundary"),
        )
        if not isinstance(root["entries"], list) or not root["entries"]:
            raise ValueError("FastAFD profile entries must be a non-empty list")
        entries = tuple(_entry(value, source, index) for index, value in enumerate(root["entries"]))
        return cls(entries, source=source, path=resolved, profile_sha256=hashlib.sha256(raw).hexdigest())

    def find(self, key: FastAFDMoEStageKey) -> FastAFDMoEStageMeasurement | None:
        return self._entries.get(key)

    def require(self, key: FastAFDMoEStageKey) -> FastAFDMoEStageMeasurement:
        measurement = self.find(key)
        if measurement is None:
            raise KeyError(f"no exact FastAFD MoE stage measurement for {key}")
        return measurement


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly {sorted(fields)}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _digest(value: Any, pattern: re.Pattern[str], name: str) -> str:
    text = _text(value, name)
    if pattern.fullmatch(text) is None:
        raise ValueError(f"{name} has an invalid digest")
    return text


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float, exclusive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (exclusive and value == minimum):
        raise ValueError(f"{name} must be {'>' if exclusive else '>='} {minimum}")
    return float(value)


def _entry(value: Any, source: FastAFDProfileSource, index: int) -> FastAFDMoEStageMeasurement:
    label = f"entry {index}"
    row = _object(value, {"key", "model_profile", "latency_ms", "measurement", "validation"}, label)
    fields = _object(row["key"], _KEY_FIELDS, f"{label}.key")
    stage = _text(fields["stage"], f"{label}.key.stage")
    if stage not in {"agg", "afd"}:
        raise ValueError(f"{label}.key.stage is unsupported")
    topology = _text(fields["topology"], f"{label}.key.topology")
    if (_AGG_TOPOLOGY if stage == "agg" else _AFD_TOPOLOGY).fullmatch(topology) is None:
        raise ValueError(f"{label}.key.topology is invalid")
    key = FastAFDMoEStageKey(
        model_path=_text(fields["model_path"], f"{label}.key.model_path"),
        system=_text(fields["system"], f"{label}.key.system"),
        stage=stage,
        topology=topology,
        logical_batch_per_source_rank=_integer(
            fields["logical_batch_per_source_rank"], f"{label}.key.logical_batch_per_source_rank"
        ),
        mtp_nextn=_integer(fields["mtp_nextn"], f"{label}.key.mtp_nextn", minimum=0),
        microbatches=_integer(fields["microbatches"], f"{label}.key.microbatches"),
        moe_layers=_integer(fields["moe_layers"], f"{label}.key.moe_layers"),
        routed_topk=_integer(fields["routed_topk"], f"{label}.key.routed_topk"),
        moe_precision=_text(fields["moe_precision"], f"{label}.key.moe_precision"),
        moe_backend=_text(fields["moe_backend"], f"{label}.key.moe_backend"),
    )
    capture = _object(row["measurement"], _MEASUREMENT_FIELDS, f"{label}.measurement")
    if capture["scope"] != "complete_moe_stage":
        raise ValueError(f"{label}.measurement.scope must be complete_moe_stage")
    if capture["statistic"] != "p50":
        raise ValueError(f"{label}.measurement.statistic must be p50")
    check = _object(row["validation"], {"stable", "correctness"}, f"{label}.validation")
    if check["stable"] is not True or check["correctness"] is False:
        raise ValueError(f"{label} is not a qualified measurement")
    if check["correctness"] is not None and check["correctness"] is not True:
        raise ValueError(f"{label}.validation.correctness must be true, false, or null")
    return FastAFDMoEStageMeasurement(
        key=key,
        model_profile=_text(row["model_profile"], f"{label}.model_profile"),
        latency_ms=_number(row["latency_ms"], f"{label}.latency_ms", minimum=0.0, exclusive=True),
        correctness=check["correctness"],
        source=source,
        method=_text(capture["method"], f"{label}.measurement.method"),
        method_version=_text(capture["method_version"], f"{label}.measurement.method_version"),
        statistic="p50",
        sample_count=_integer(capture["sample_count"], f"{label}.measurement.sample_count"),
        cv_percent=_number(capture["cv_percent"], f"{label}.measurement.cv_percent", minimum=0.0, exclusive=False),
        routing=_text(capture["routing"], f"{label}.measurement.routing"),
        source_tree_sha256=_digest(capture["source_tree_sha256"], _SHA256, f"{label}.measurement.source_tree_sha256"),
        raw_artifact=_text(capture["raw_artifact"], f"{label}.measurement.raw_artifact"),
        raw_sha256=_digest(capture["raw_sha256"], _SHA256, f"{label}.measurement.raw_sha256"),
    )


__all__ = [
    "FASTAFD_PROFILE_SCHEMA",
    "FastAFDMoEStageKey",
    "FastAFDMoEStageMeasurement",
    "FastAFDMoEStageProfile",
    "FastAFDProfileSource",
]
