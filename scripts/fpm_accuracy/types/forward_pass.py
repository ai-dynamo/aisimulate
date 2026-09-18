# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Forward-pass telemetry types and the no-target-leak predictor boundary."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fpm_accuracy.exceptions import DataError

FPM_SCHEMA_VERSION = 1


class WorkloadKind(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"
    EMPTY = "empty"


def classify_workload(
    *,
    sum_prefill_tokens: int,
    num_decode_requests: int,
    sum_decode_kv_tokens: int,
) -> WorkloadKind:
    """Classify one rank exactly as AISim's ``IterationFeatures`` does."""

    has_prefill = sum_prefill_tokens > 0
    has_decode = num_decode_requests > 0 or sum_decode_kv_tokens > 0
    if has_prefill and has_decode:
        return WorkloadKind.MIXED
    if has_prefill:
        return WorkloadKind.PREFILL
    if has_decode:
        return WorkloadKind.DECODE
    return WorkloadKind.EMPTY


def workload_load_score(
    *,
    sum_prefill_tokens: int,
    num_decode_requests: int,
    sum_decode_kv_tokens: int,
) -> int | None:
    """Return AISim's max-rank comparison score, or ``None`` for an empty rank."""

    kind = classify_workload(
        sum_prefill_tokens=sum_prefill_tokens,
        num_decode_requests=num_decode_requests,
        sum_decode_kv_tokens=sum_decode_kv_tokens,
    )
    if kind is WorkloadKind.EMPTY:
        return None
    if kind is WorkloadKind.PREFILL:
        return sum_prefill_tokens
    if kind is WorkloadKind.DECODE:
        return num_decode_requests + sum_decode_kv_tokens
    return sum_prefill_tokens + sum_decode_kv_tokens


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    var_prefill_length: float = 0.0
    sum_prefill_kv_tokens: int = 0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0
    var_decode_kv_tokens: float = 0.0

    def __post_init__(self) -> None:
        integer_values = (
            self.num_prefill_requests,
            self.sum_prefill_tokens,
            self.sum_prefill_kv_tokens,
            self.num_decode_requests,
            self.sum_decode_kv_tokens,
        )
        if any(value < 0 for value in integer_values):
            raise DataError("forward-pass request counts and token sums must be non-negative")
        variances = (self.var_prefill_length, self.var_decode_kv_tokens)
        if any(not math.isfinite(value) or value < 0 for value in variances):
            raise DataError("forward-pass request variances must be finite and non-negative")
        if self.num_prefill_requests == 0 and (self.sum_prefill_tokens or self.sum_prefill_kv_tokens):
            raise DataError("prefill token sums require num_prefill_requests > 0")
        if self.num_decode_requests == 0 and self.sum_decode_kv_tokens:
            raise DataError("decode KV token sum requires num_decode_requests > 0")

    def to_aic_dict(self, *, queued: bool = False) -> dict[str, int | float]:
        result: dict[str, int | float] = {
            "num_prefill_requests": self.num_prefill_requests,
            "sum_prefill_tokens": self.sum_prefill_tokens,
            "var_prefill_length": self.var_prefill_length,
            "num_decode_requests": self.num_decode_requests,
            "sum_decode_kv_tokens": self.sum_decode_kv_tokens,
            "var_decode_kv_tokens": self.var_decode_kv_tokens,
        }
        if not queued:
            result["sum_prefill_kv_tokens"] = self.sum_prefill_kv_tokens
        return result


@dataclass(frozen=True, slots=True)
class ForwardPassMetric:
    fpm_id: int
    configuration_id: str
    version: int
    worker_id: str
    dp_rank: int
    counter_id: int
    wall_time_s: float
    scheduled: RequestMetrics
    queued: RequestMetrics
    event_time: str = ""
    source_cluster_id: str = ""
    capture_id: str = ""
    stream_id: str = ""
    pass_type: str = ""
    record_kind: str = ""
    metric_validity: str = ""
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version != FPM_SCHEMA_VERSION:
            raise DataError(f"FPM {self.fpm_id} uses schema v{self.version}; expected v{FPM_SCHEMA_VERSION}")
        if self.fpm_id < 0 or self.dp_rank < 0 or self.counter_id < 0:
            raise DataError("fpm_id, dp_rank, and counter_id must be non-negative")
        if not self.configuration_id:
            raise DataError(f"FPM {self.fpm_id} is missing configuration_id")
        if not math.isfinite(self.wall_time_s) or self.wall_time_s < 0:
            raise DataError(f"FPM {self.fpm_id} wall_time_s must be finite and non-negative")

    def source_metadata(self) -> Mapping[str, Any]:
        """Return optional publisher metadata as a mapping when it is valid JSON."""

        raw = self.extra_metadata.get("source_metadata")
        if isinstance(raw, Mapping):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except ValueError:
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    @property
    def workload_kind(self) -> WorkloadKind:
        # A scheduled prefill request with zero freshly computed tokens is idle
        # at this abstraction.
        return classify_workload(
            sum_prefill_tokens=self.scheduled.sum_prefill_tokens,
            num_decode_requests=self.scheduled.num_decode_requests,
            sum_decode_kv_tokens=self.scheduled.sum_decode_kv_tokens,
        )

    @property
    def load_score(self) -> int | None:
        return workload_load_score(
            sum_prefill_tokens=self.scheduled.sum_prefill_tokens,
            num_decode_requests=self.scheduled.num_decode_requests,
            sum_decode_kv_tokens=self.scheduled.sum_decode_kv_tokens,
        )

    def to_aic_dict(self, *, include_observation: bool) -> dict[str, Any]:
        """Return AISim's FPM v1 payload, excluding the target on prediction."""

        result: dict[str, Any] = {
            "version": self.version,
            "worker_id": self.worker_id,
            "dp_rank": self.dp_rank,
            "counter_id": self.counter_id,
            "scheduled_requests": self.scheduled.to_aic_dict(),
            "queued_requests": self.queued.to_aic_dict(queued=True),
        }
        if include_observation:
            result["wall_time"] = self.wall_time_s
        return result


@dataclass(frozen=True, slots=True)
class ForwardPassInput:
    """Target-blind workload passed to a predictor.

    It deliberately has no observed latency field. Predictor implementations
    therefore cannot train on or otherwise inspect the label they are being
    asked to predict.
    """

    configuration_id: str
    iteration_id: str
    fpm_ids: tuple[int, ...]
    workload_kind: WorkloadKind
    rank_payloads: tuple[Mapping[str, Any], ...]

    def aic_payload(self) -> list[dict[str, Any]]:
        return [dict(rank) for rank in self.rank_payloads]


@dataclass(frozen=True, slots=True)
class ForwardPassIteration:
    """One engine iteration, represented by one metric per attention-DP rank."""

    ranks: tuple[ForwardPassMetric, ...]

    def __post_init__(self) -> None:
        if not self.ranks:
            raise DataError("a forward-pass iteration must contain at least one rank")
        configuration_ids = {rank.configuration_id for rank in self.ranks}
        if len(configuration_ids) != 1:
            raise DataError("all ranks in an iteration must share configuration_id")
        dp_ranks = [rank.dp_rank for rank in self.ranks]
        if len(dp_ranks) != len(set(dp_ranks)):
            raise DataError("an iteration may not contain duplicate dp_rank values")

    @classmethod
    def single_rank(cls, metric: ForwardPassMetric) -> ForwardPassIteration:
        return cls((metric,))

    @property
    def configuration_id(self) -> str:
        return self.ranks[0].configuration_id

    @property
    def iteration_id(self) -> str:
        first = self.ranks[0]
        identity = [
            first.source_cluster_id,
            first.capture_id,
            first.stream_id,
            first.worker_id,
            str(first.counter_id),
        ]
        compact = "/".join(part for part in identity if part)
        return compact or f"fpm:{first.fpm_id}"

    @property
    def fpm_ids(self) -> tuple[int, ...]:
        return tuple(rank.fpm_id for rank in self.ranks)

    def source_metadata(self) -> Mapping[str, Any]:
        """Return optional publisher metadata from the representative rank."""

        return self.ranks[0].source_metadata()

    @property
    def representative_rank(self) -> ForwardPassMetric:
        """Rank whose scheduled load AISim uses to classify and fit the iteration.

        Rust's ``Iterator::max_by`` keeps the later value on an exact tie. Ranks
        are stored in ascending ``dp_rank`` order, so the index tie-break below
        mirrors that behavior.
        """

        non_empty = [(index, rank) for index, rank in enumerate(self.ranks) if rank.load_score is not None]
        if not non_empty:
            return self.ranks[-1]
        return max(non_empty, key=lambda item: (item[1].load_score, item[0]))[1]

    @property
    def workload_kind(self) -> WorkloadKind:
        return self.representative_rank.workload_kind

    @property
    def observed_time_ms(self) -> float | None:
        positive = [rank.wall_time_s for rank in self.ranks if rank.wall_time_s > 0]
        return max(positive) * 1000.0 if positive else None

    def prediction_payload(self) -> list[dict[str, Any]]:
        return [rank.to_aic_dict(include_observation=False) for rank in self.ranks]

    def prediction_input(self) -> ForwardPassInput:
        return ForwardPassInput(
            configuration_id=self.configuration_id,
            iteration_id=self.iteration_id,
            fpm_ids=self.fpm_ids,
            workload_kind=self.workload_kind,
            rank_payloads=tuple(rank.to_aic_dict(include_observation=False) for rank in self.ranks),
        )

    def tuning_payload(self) -> list[dict[str, Any]]:
        return [rank.to_aic_dict(include_observation=True) for rank in self.ranks]
