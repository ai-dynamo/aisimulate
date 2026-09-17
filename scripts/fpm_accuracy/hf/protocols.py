# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Versioned parsers for measurement protocols published by the HF dataset."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TextIO

from fpm_accuracy.exceptions import DataError
from fpm_accuracy.hf.models import MeasurementFile, MeasurementIssue, MeasurementState
from fpm_accuracy.types.forward_pass import ForwardPassIteration, ForwardPassMetric, RequestMetrics, WorkloadKind


@dataclass(frozen=True, slots=True)
class ParsedObservation:
    source_file: MeasurementFile
    source_row: int
    identity_hint: str
    iteration: ForwardPassIteration
    event_time: str | None = None
    chronology_key: tuple[int | float | str, ...] | None = None


@dataclass(frozen=True, slots=True)
class ParseResult:
    observations: tuple[ParsedObservation, ...]
    warnings: tuple[str, ...] = ()
    issues: tuple[MeasurementIssue, ...] = ()


ProtocolParser = Callable[[str, Sequence[MeasurementFile], int | None], ParseResult]


@dataclass(frozen=True, slots=True)
class ProtocolAdapter:
    policy_id: str
    parse: ProtocolParser


SUPPORTED_PROTOCOL_IDS = frozenset({"forward-pass-measurement-v1", "forward-pass-record-v1"})
SUPPORTED_EVIDENCE_FORMAT_IDS = frozenset({"wave-summary-v1", "bucket-summary-v1"})
_MEASUREMENT_STREAM_SUFFIXES = (".jsonl", ".jsonl.gz")
_SYNCHRONIZED_ITERATION_SUFFIXES = (".json", ".json.gz")
_FORWARD_PASS_RECORD_SUFFIXES = (".csv", ".csv.gz")
_FORWARD_PASS_RECORD_FIELDS = frozenset(
    {
        "measurement_id",
        "phase",
        "batch_size",
        "total_prefill_tokens",
        "total_kv_read_tokens",
        "truth_latency_ms",
    }
)
_TIME_UNIT_SECONDS = {
    "s": Decimal(1),
    "ms": Decimal("0.001"),
    "us": Decimal("0.000001"),
    "ns": Decimal("0.000000001"),
}
_CANONICAL_ITERATION_FIELDS = frozenset(
    {
        "source_kind",
        "iteration_id",
        "producer",
        "grouping",
        "expected_dp_ranks",
        "complete",
        "max_rank_wall_time",
        "rank_measurements",
    }
)


@dataclass(frozen=True, slots=True)
class _RankMeasurement:
    version: int
    worker_id: str
    dp_rank: int
    counter_id: int
    wall_time_s: float
    observed_at_unix_ms: float | None
    scheduled: RequestMetrics
    queued: RequestMetrics


def adapter_for(protocol_id: str | None) -> ProtocolAdapter | None:
    if protocol_id == "forward-pass-measurement-v1":
        return ProtocolAdapter(
            "aisim-fpm/forward-pass-measurement-v1/max-rank-v3",
            _parse_forward_pass_measurements,
        )
    if protocol_id == "forward-pass-record-v1":
        return ProtocolAdapter(
            "aisim-fpm/forward-pass-record-v1",
            _parse_evaluation_records,
        )
    return None


def parser_for(protocol_id: str | None) -> ProtocolParser | None:
    adapter = adapter_for(protocol_id)
    return adapter.parse if adapter is not None else None


def _parse_forward_pass_measurements(
    configuration_id: str,
    files: Sequence[MeasurementFile],
    attention_dp_size: int | None,
) -> ParseResult:
    stream_files: list[MeasurementFile] = []
    benchmark_files: list[MeasurementFile] = []
    pre_grouped_files: list[MeasurementFile] = []
    for file in files:
        if file.representation == "pre_grouped_rank_lists":
            pre_grouped_files.append(file)
        elif file.local_path.name.endswith(_MEASUREMENT_STREAM_SUFFIXES):
            stream_files.append(file)
        elif file.local_path.name.endswith(_SYNCHRONIZED_ITERATION_SUFFIXES):
            benchmark_files.append(file)
        else:
            raise DataError(f"measurement protocol forward-pass-measurement-v1 does not support file {file.path!r}")
    return _combine(
        _parse_fpm_streams(configuration_id, stream_files, attention_dp_size),
        _parse_benchmark_files(configuration_id, benchmark_files, attention_dp_size),
        _parse_pre_grouped_files(configuration_id, pre_grouped_files, attention_dp_size),
    )


def _parse_pre_grouped_files(
    configuration_id: str,
    files: Sequence[MeasurementFile],
    attention_dp_size: int | None,
) -> ParseResult:
    if attention_dp_size is None or attention_dp_size <= 0:
        raise DataError("cannot validate pre-grouped measurements without attention_dp_size")
    expected_ranks = list(range(attention_dp_size))
    observations: list[ParsedObservation] = []
    for file in files:
        iteration_count = 0
        rank_record_count = 0
        seen: set[tuple[str, int, int]] = set()
        for source_row, group in enumerate(_iter_json_array(file), start=1):
            if not isinstance(group, list) or not group:
                raise DataError(f"measurement file {file.path} iteration {source_row} must be a non-empty rank array")
            if not all(isinstance(value, Mapping) for value in group):
                raise DataError(f"measurement file {file.path} iteration {source_row} contains a non-object rank")
            ranks = [_rank_measurement(value, file, source_row) for value in group]
            _validate_pre_grouped_selection(group, file, source_row)
            rank_ids = [rank.dp_rank for rank in ranks]
            if rank_ids != expected_ranks:
                raise DataError(
                    f"measurement file {file.path} iteration {source_row} ranks do not match expected_dp_ranks"
                )
            for rank in ranks:
                identity = (rank.worker_id, rank.dp_rank, rank.counter_id)
                if identity in seen:
                    raise DataError(f"measurement file {file.path} reuses a rank observation")
                seen.add(identity)
                if rank.wall_time_s <= 0:
                    raise DataError(f"measurement file {file.path} iteration {source_row} has non-positive latency")
            if all(_workload(rank.scheduled) is WorkloadKind.EMPTY for rank in ranks):
                raise DataError(f"measurement file {file.path} iteration {source_row} has no scheduled work")
            metrics = tuple(
                _metric(
                    configuration_id=configuration_id,
                    file=file,
                    source_row=source_row,
                    rank_index=rank_index,
                    rank=rank,
                )
                for rank_index, rank in enumerate(ranks)
            )
            observations.append(
                ParsedObservation(
                    source_file=file,
                    source_row=source_row,
                    identity_hint=f"pre-grouped:{file.measurement_file_id}:{source_row - 1}",
                    iteration=ForwardPassIteration(metrics),
                )
            )
            iteration_count += 1
            rank_record_count += len(ranks)
        if iteration_count != file.iteration_count or rank_record_count != file.rank_record_count:
            raise DataError(f"measurement file {file.path} counts disagree with its manifest")
    return ParseResult(tuple(observations))


def _validate_pre_grouped_selection(
    group: Sequence[Mapping[str, Any]],
    file: MeasurementFile,
    source_row: int,
) -> None:
    grouping = file.grouping or {}
    selection = grouping.get("selection_window")
    if selection is None:
        return
    if not isinstance(selection, Mapping):
        raise DataError(f"measurement file {file.path} has invalid grouping selection_window")
    start = selection.get("start_inclusive_ns")
    end = selection.get("end_exclusive_ns")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start >= end
    ):
        raise DataError(f"measurement file {file.path} has invalid grouping selection_window")
    for rank in group:
        received_at = rank.get("received_at_ns")
        if isinstance(received_at, bool) or not isinstance(received_at, int) or not start <= received_at < end:
            raise DataError(f"measurement file {file.path} iteration {source_row} violates grouping selection_window")


def _iter_json_array(file: MeasurementFile) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    try:
        with _open_text(file.local_path) as handle:
            buffer = ""
            position = 0
            eof = False

            def token() -> str:
                nonlocal buffer, position, eof
                while True:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position < len(buffer):
                        return buffer[position]
                    if eof:
                        raise DataError(f"unexpected end of JSON array in {file.path}")
                    chunk = handle.read(65536)
                    buffer = buffer[position:] + chunk
                    position = 0
                    eof = not chunk

            if token() != "[":
                raise DataError(f"measurement file {file.path} must contain an outer iteration array")
            position += 1
            first = True
            while True:
                next_token = token()
                if next_token == "]":
                    position += 1
                    if buffer[position:].strip() or handle.read().strip():
                        raise DataError(f"measurement file {file.path} has trailing JSON content")
                    return
                if not first:
                    if next_token != ",":
                        raise DataError(f"measurement file {file.path} is missing an iteration separator")
                    position += 1
                    if token() == "]":
                        raise DataError(f"measurement file {file.path} has a trailing comma")
                while True:
                    try:
                        value, end = decoder.raw_decode(buffer, position)
                        position = end
                        break
                    except json.JSONDecodeError as exc:
                        if eof:
                            raise DataError(f"cannot parse measurement JSON {file.path}: {exc}") from exc
                        chunk = handle.read(65536)
                        buffer = buffer[position:] + chunk
                        position = 0
                        eof = not chunk
                first = False
                yield value
    except (OSError, UnicodeError) as exc:
        raise DataError(f"cannot parse measurement JSON {file.path}: {exc}") from exc


def _combine(*results: ParseResult) -> ParseResult:
    return ParseResult(
        observations=tuple(observation for result in results for observation in result.observations),
        warnings=tuple(warning for result in results for warning in result.warnings),
        issues=tuple(issue for result in results for issue in result.issues),
    )


def _parse_fpm_streams(
    configuration_id: str,
    files: Sequence[MeasurementFile],
    attention_dp_size: int | None,
) -> ParseResult:
    observations: list[ParsedObservation] = []
    issues: list[MeasurementIssue] = []
    for file in files:
        excluded: Counter[str] = Counter()
        unavailable: Counter[str] = Counter()
        stream_layout: str | None = None
        with _open_text(file.local_path) as handle:
            for source_row, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = _json_line(line, file, source_row)
                record_layout = (
                    "canonical_iteration" if _looks_like_canonical_iteration(payload) else "rank_observation"
                )
                if stream_layout is None:
                    stream_layout = record_layout
                elif stream_layout != record_layout:
                    raise DataError(f"measurement file {file.path} mixes rank and canonical iteration records")
                if record_layout == "canonical_iteration":
                    observation, issue = _canonical_iteration(
                        configuration_id,
                        file,
                        source_row,
                        payload,
                        attention_dp_size,
                    )
                    if observation is not None:
                        observations.append(observation)
                    elif issue is not None:
                        state, reason = issue
                        (excluded if state is MeasurementState.EXCLUDED else unavailable)[reason] += 1
                    continue

                rank = _rank_measurement(payload, file, source_row)
                if attention_dp_size is None or attention_dp_size <= 0:
                    raise DataError(f"cannot validate rank stream {file.path} without attention_dp_size")
                # A rank-local counter is not a cross-rank iteration key. Raw
                # multi-rank streams must be materialized by the dataset using
                # their declared synchronization rule before Gym can score them.
                if attention_dp_size > 1:
                    unavailable["unsynchronized_attention_dp_stream"] += 1
                    continue
                if rank.dp_rank != 0:
                    unavailable["unexpected_dp_rank"] += 1
                    continue
                if rank.wall_time_s == 0:
                    excluded["non_positive_latency"] += 1
                    continue
                if _workload(rank.scheduled) is WorkloadKind.EMPTY:
                    excluded["empty_scheduler_heartbeat"] += 1
                    continue
                chronology_key: tuple[int | float | str, ...] | None = None
                event_time: str | None = None
                if rank.observed_at_unix_ms is not None:
                    chronology_key = (rank.observed_at_unix_ms, rank.counter_id)
                    event_time = str(int(rank.observed_at_unix_ms))
                metric = _metric(
                    configuration_id=configuration_id,
                    file=file,
                    source_row=source_row,
                    rank_index=0,
                    rank=rank,
                )
                observations.append(
                    ParsedObservation(
                        source_file=file,
                        source_row=source_row,
                        identity_hint=f"{rank.worker_id}:{rank.counter_id}:rank{rank.dp_rank}",
                        iteration=ForwardPassIteration.single_rank(metric),
                        event_time=event_time,
                        chronology_key=chronology_key,
                    )
                )
        issues.extend(_issues(file, excluded, MeasurementState.EXCLUDED))
        issues.extend(_issues(file, unavailable, MeasurementState.MEASUREMENT_UNAVAILABLE))
    return ParseResult(tuple(observations), issues=tuple(issues))


def _looks_like_canonical_iteration(payload: Mapping[str, Any]) -> bool:
    return "rank_measurements" in payload or ("iteration_id" in payload and "grouping" in payload)


def _canonical_iteration(
    configuration_id: str,
    file: MeasurementFile,
    source_row: int,
    payload: Mapping[str, Any],
    attention_dp_size: int | None,
) -> tuple[ParsedObservation | None, tuple[MeasurementState, str] | None]:
    _require_fields(payload, {"version", *_CANONICAL_ITERATION_FIELDS}, file, source_row)
    version = _json_nonnegative_int(payload["version"], file, source_row, "version")
    if version != 1:
        raise DataError(f"measurement file {file.path} row {source_row} field version must be 1")
    if attention_dp_size is None or attention_dp_size <= 0:
        raise DataError(f"cannot validate synchronized ranks for {file.path} without attention_dp_size")

    source_kind = payload["source_kind"]
    if source_kind not in {"self_benchmark", "rank_event_stream"}:
        raise DataError(f"measurement file {file.path} row {source_row} has invalid source_kind")
    iteration_id = _json_nonempty_string(payload["iteration_id"], file, source_row, "iteration_id")
    _validate_producer(payload["producer"], file, source_row)
    if "collector" in payload:
        _validate_collector(payload["collector"], file, source_row)

    expected_ranks = _json_unique_nonnegative_ints(payload["expected_dp_ranks"], file, source_row, "expected_dp_ranks")
    complete = payload["complete"]
    if not isinstance(complete, bool):
        raise DataError(f"measurement file {file.path} row {source_row} field complete must be boolean")

    rank_payloads = payload["rank_measurements"]
    if not isinstance(rank_payloads, list):
        raise DataError(f"measurement file {file.path} row {source_row} rank_measurements must be an array")
    ranks = [_rank_measurement(rank, file, source_row) for rank in rank_payloads if isinstance(rank, Mapping)]
    if len(ranks) != len(rank_payloads):
        raise DataError(f"measurement file {file.path} row {source_row} contains a non-object rank measurement")
    rank_ids = [rank.dp_rank for rank in ranks]
    if rank_ids != sorted(rank_ids) or len(rank_ids) != len(set(rank_ids)):
        raise DataError(
            f"measurement file {file.path} row {source_row} rank_measurements must have unique ascending dp_rank"
        )
    if any(dp_rank not in expected_ranks for dp_rank in rank_ids):
        raise DataError(
            f"measurement file {file.path} row {source_row} rank_measurements contain an unexpected dp_rank"
        )

    grouping = payload["grouping"]
    if not isinstance(grouping, Mapping):
        raise DataError(f"measurement file {file.path} row {source_row} grouping must be an object")
    event_time, chronology_key = _validate_grouping(
        payload,
        grouping,
        source_kind,
        expected_ranks,
        ranks,
        rank_payloads,
        file,
        source_row,
    )

    cached_max = payload["max_rank_wall_time"]
    if not complete:
        if cached_max is not None:
            raise DataError(
                f"measurement file {file.path} row {source_row} incomplete iteration must have null max_rank_wall_time"
            )
        return None, (MeasurementState.MEASUREMENT_UNAVAILABLE, "incomplete_synchronized_iteration")
    if not ranks:
        raise DataError(f"measurement file {file.path} row {source_row} complete iteration has no rank measurements")
    if rank_ids != sorted(expected_ranks):
        raise DataError(
            f"measurement file {file.path} row {source_row} complete iteration ranks do not match expected_dp_ranks"
        )
    official_wall = _json_nonnegative_number(cached_max, file, source_row, "max_rank_wall_time")
    if official_wall != max(rank.wall_time_s for rank in ranks):
        raise DataError(
            f"measurement file {file.path} row {source_row} max_rank_wall_time does not match rank measurements"
        )
    configured_ranks = list(range(attention_dp_size))
    if sorted(expected_ranks) != configured_ranks:
        return None, (MeasurementState.MEASUREMENT_UNAVAILABLE, "unexpected_attention_dp_ranks")
    if official_wall == 0:
        return None, (MeasurementState.EXCLUDED, "non_positive_latency")
    if all(_workload(rank.scheduled) is WorkloadKind.EMPTY for rank in ranks):
        return None, (MeasurementState.EXCLUDED, "empty_scheduler_heartbeat")

    metrics = tuple(
        _metric(
            configuration_id=configuration_id,
            file=file,
            source_row=source_row,
            rank_index=rank_index,
            rank=rank,
        )
        for rank_index, rank in enumerate(ranks)
    )
    return (
        ParsedObservation(
            source_file=file,
            source_row=source_row,
            identity_hint=f"iteration:{iteration_id}",
            iteration=ForwardPassIteration(metrics),
            event_time=event_time,
            chronology_key=chronology_key,
        ),
        None,
    )


def _validate_grouping(
    iteration: Mapping[str, Any],
    grouping: Mapping[str, Any],
    source_kind: Any,
    expected_ranks: list[int],
    ranks: list[_RankMeasurement],
    rank_payloads: list[Any],
    file: MeasurementFile,
    source_row: int,
) -> tuple[str | None, tuple[int | float | str, ...] | None]:
    method = grouping.get("method")
    if method == "single_rank":
        _require_fields(grouping, {"method", "authority"}, file, source_row, prefix="grouping.")
        if source_kind != "rank_event_stream" or grouping["authority"] != "producer":
            raise DataError(f"measurement file {file.path} row {source_row} has invalid single_rank grouping")
        if "key" in grouping:
            _json_nonempty_string(grouping["key"], file, source_row, "grouping.key")
        if len(expected_ranks) != 1 or len(ranks) != 1 or ranks[0].dp_rank != expected_ranks[0]:
            raise DataError(
                f"measurement file {file.path} row {source_row} single_rank grouping must contain one matching rank"
            )
        if ranks[0].observed_at_unix_ms is None:
            return None, None
        event_time = str(int(ranks[0].observed_at_unix_ms))
        return event_time, (ranks[0].observed_at_unix_ms, ranks[0].counter_id)

    if method == "benchmark_id":
        _require_fields(
            grouping,
            {"method", "authority", "key", "benchmark_id", "point"},
            file,
            source_row,
            prefix="grouping.",
        )
        if source_kind != "self_benchmark" or grouping["authority"] != "producer":
            raise DataError(f"measurement file {file.path} row {source_row} has invalid benchmark_id grouping")
        benchmark_id = _json_nonnegative_int(grouping["benchmark_id"], file, source_row, "grouping.benchmark_id")
        key = _json_nonempty_string(grouping["key"], file, source_row, "grouping.key")
        if key != str(benchmark_id):
            raise DataError(f"measurement file {file.path} row {source_row} grouping.key must equal benchmark_id")
        _validate_benchmark_point(grouping["point"], benchmark_id, file, source_row, prefix="grouping.point.")
        return None, None

    if method == "listener_receive_time_window":
        return _validate_listener_grouping(
            iteration,
            grouping,
            source_kind,
            expected_ranks,
            ranks,
            rank_payloads,
            file,
            source_row,
        )
    raise DataError(f"measurement file {file.path} row {source_row} has invalid grouping.method")


def _validate_listener_grouping(
    iteration: Mapping[str, Any],
    grouping: Mapping[str, Any],
    source_kind: Any,
    expected_ranks: list[int],
    ranks: list[_RankMeasurement],
    rank_payloads: list[Any],
    file: MeasurementFile,
    source_row: int,
) -> tuple[str, tuple[int | float | str, ...]]:
    required = {
        "method",
        "authority",
        "performed_by",
        "key",
        "engine_role",
        "source_field",
        "source_unit",
        "clock_correction",
        "window",
        "window_start",
    }
    _require_fields(grouping, required, file, source_row, prefix="grouping.")
    if source_kind != "rank_event_stream" or grouping["authority"] != "derived":
        raise DataError(
            f"measurement file {file.path} row {source_row} has invalid listener_receive_time_window grouping"
        )
    for field in ("performed_by", "key", "engine_role"):
        _json_nonempty_string(grouping[field], file, source_row, f"grouping.{field}")
    if grouping["source_field"] != "_recv_ts":
        raise DataError(f"measurement file {file.path} row {source_row} grouping.source_field must be _recv_ts")
    source_unit = _json_time_unit(grouping["source_unit"], file, source_row, "grouping.source_unit")

    if "collector" not in iteration:
        raise DataError(f"measurement file {file.path} row {source_row} listener grouping requires collector")
    offsets_s = _validate_clock_correction(grouping["clock_correction"], expected_ranks, file, source_row)

    window = grouping["window"]
    if not isinstance(window, Mapping):
        raise DataError(f"measurement file {file.path} row {source_row} grouping.window must be an object")
    _require_fields(
        window,
        {"size", "unit", "anchor", "boundary", "distinct_by"},
        file,
        source_row,
        prefix="grouping.window.",
    )
    window_size = _json_decimal(window["size"], file, source_row, "grouping.window.size")
    if window_size <= 0:
        raise DataError(f"measurement file {file.path} row {source_row} grouping.window.size must be positive")
    window_unit = _json_time_unit(window["unit"], file, source_row, "grouping.window.unit")
    if (
        window["anchor"] != "earliest_corrected_receive_timestamp"
        or window["boundary"] != "half_open"
        or window["distinct_by"] != "dp_rank"
    ):
        raise DataError(f"measurement file {file.path} row {source_row} has invalid grouping.window policy")

    window_start_text = _json_nonempty_string(grouping["window_start"], file, source_row, "grouping.window_start")
    window_start_s = (
        _json_decimal(window_start_text, file, source_row, "grouping.window_start") * _TIME_UNIT_SECONDS[source_unit]
    )
    if window_start_s < 0:
        raise DataError(f"measurement file {file.path} row {source_row} grouping.window_start must be non-negative")
    window_size_s = window_size * _TIME_UNIT_SECONDS[window_unit]
    corrected_times: list[Decimal] = []
    for rank, rank_payload in zip(ranks, rank_payloads, strict=True):
        if rank.dp_rank not in offsets_s or not isinstance(rank_payload, Mapping) or "_recv_ts" not in rank_payload:
            raise DataError(
                f"measurement file {file.path} row {source_row} listener rank {rank.dp_rank} lacks join evidence"
            )
        received_s = (
            _json_decimal(rank_payload["_recv_ts"], file, source_row, "rank_measurements[]._recv_ts")
            * _TIME_UNIT_SECONDS[source_unit]
        )
        corrected_times.append(received_s + offsets_s[rank.dp_rank])
    if corrected_times:
        if min(corrected_times) != window_start_s:
            raise DataError(
                f"measurement file {file.path} row {source_row} grouping.window_start is not the earliest "
                "corrected receive time"
            )
        if any(time < window_start_s or time - window_start_s >= window_size_s for time in corrected_times):
            raise DataError(f"measurement file {file.path} row {source_row} contains a rank outside grouping.window")
    return window_start_text, (float(window_start_s),)


def _validate_producer(value: Any, file: MeasurementFile, source_row: int) -> None:
    if not isinstance(value, Mapping):
        raise DataError(f"measurement file {file.path} row {source_row} producer must be an object")
    _require_fields(value, {"component"}, file, source_row, prefix="producer.")
    _json_nonempty_string(value["component"], file, source_row, "producer.component")
    for field in ("version", "source_revision"):
        if field in value:
            _json_nonempty_string(value[field], file, source_row, f"producer.{field}")


def _validate_collector(value: Any, file: MeasurementFile, source_row: int) -> None:
    if not isinstance(value, Mapping):
        raise DataError(f"measurement file {file.path} row {source_row} collector must be an object")
    _require_fields(value, {"kind"}, file, source_row, prefix="collector.")
    if value["kind"] not in {"custom_script", "service"}:
        raise DataError(f"measurement file {file.path} row {source_row} has invalid collector.kind")
    for field in ("component", "path", "source_revision"):
        if field in value:
            _json_nonempty_string(value[field], file, source_row, f"collector.{field}")


def _validate_clock_correction(
    value: Any,
    expected_ranks: list[int],
    file: MeasurementFile,
    source_row: int,
) -> dict[int, Decimal]:
    if not isinstance(value, Mapping):
        raise DataError(f"measurement file {file.path} row {source_row} grouping.clock_correction must be an object")
    _require_fields(
        value,
        {"method", "reference_clock_id", "offset_unit", "groups"},
        file,
        source_row,
        prefix="grouping.clock_correction.",
    )
    _json_nonempty_string(value["method"], file, source_row, "grouping.clock_correction.method")
    reference = _json_nonempty_string(
        value["reference_clock_id"], file, source_row, "grouping.clock_correction.reference_clock_id"
    )
    offset_unit = _json_time_unit(value["offset_unit"], file, source_row, "grouping.clock_correction.offset_unit")
    groups = value["groups"]
    if not isinstance(groups, list) or not groups:
        raise DataError(
            f"measurement file {file.path} row {source_row} grouping.clock_correction.groups must be non-empty"
        )

    clock_offsets: dict[str, Decimal] = {}
    rank_offsets: dict[int, Decimal] = {}
    for group in groups:
        if not isinstance(group, Mapping):
            raise DataError(f"measurement file {file.path} row {source_row} contains a non-object clock group")
        _require_fields(group, {"clock_id", "dp_ranks", "offset"}, file, source_row, prefix="clock_group.")
        clock_id = _json_nonempty_string(group["clock_id"], file, source_row, "clock_group.clock_id")
        if clock_id in clock_offsets:
            raise DataError(f"measurement file {file.path} row {source_row} contains duplicate clock_id")
        offset = _json_decimal(group["offset"], file, source_row, "clock_group.offset")
        clock_offsets[clock_id] = offset
        dp_ranks = _json_unique_nonnegative_ints(group["dp_ranks"], file, source_row, "clock_group.dp_ranks")
        for dp_rank in dp_ranks:
            if dp_rank in rank_offsets:
                raise DataError(f"measurement file {file.path} row {source_row} assigns a rank to multiple clocks")
            rank_offsets[dp_rank] = offset * _TIME_UNIT_SECONDS[offset_unit]
    if sorted(rank_offsets) != sorted(expected_ranks):
        raise DataError(f"measurement file {file.path} row {source_row} clock groups must partition expected_dp_ranks")
    if reference not in clock_offsets or clock_offsets[reference] != 0:
        raise DataError(f"measurement file {file.path} row {source_row} reference clock must exist with zero offset")
    return rank_offsets


def _parse_benchmark_files(
    configuration_id: str,
    files: Sequence[MeasurementFile],
    attention_dp_size: int | None,
) -> ParseResult:
    observations: list[ParsedObservation] = []
    issues: list[MeasurementIssue] = []
    for file in files:
        payload = _read_json(file)
        if not isinstance(payload, Mapping):
            raise DataError(f"measurement file {file.path} must contain a JSON object")
        groups = payload.get("iteration_groups")
        if groups is None:
            issues.append(
                MeasurementIssue(
                    MeasurementState.EXCLUDED,
                    "truth_file_has_no_latency_records",
                    file.measurement_file_id,
                    file.path,
                )
            )
            continue
        if not isinstance(groups, list):
            raise DataError(f"measurement file {file.path} field iteration_groups must be an array")

        unavailable = 0
        for source_row, row in enumerate(groups, start=1):
            if not isinstance(row, Mapping):
                raise DataError(f"measurement file {file.path} record {source_row} must be an object")
            _validate_group_record(row, file, source_row)
            fpms = _complete_group_fpms(row, attention_dp_size, file, source_row)
            if fpms is None:
                unavailable += 1
                continue
            ranks: list[ForwardPassMetric] = []
            for rank_index, fpm in enumerate(fpms):
                rank = _rank_measurement(fpm, file, source_row)
                ranks.append(
                    _metric(
                        configuration_id=configuration_id,
                        file=file,
                        source_row=source_row,
                        rank_index=rank_index,
                        rank=rank,
                    )
                )
            expected_ranks = list(range(attention_dp_size)) if attention_dp_size is not None else None
            if expected_ranks is None or [rank.dp_rank for rank in ranks] != expected_ranks:
                unavailable += 1
                continue
            if max(rank.wall_time_s for rank in ranks) <= 0 or all(
                _workload(rank.scheduled) is WorkloadKind.EMPTY for rank in ranks
            ):
                unavailable += 1
                continue
            official_wall = _json_positive_number(row["wall_time"], file, source_row, "wall_time")
            if official_wall != max(rank.wall_time_s for rank in ranks):
                unavailable += 1
                continue
            benchmark_id = row["benchmark_id"]
            observations.append(
                ParsedObservation(
                    source_file=file,
                    source_row=source_row,
                    identity_hint=f"benchmark:{benchmark_id if benchmark_id is not None else source_row}",
                    iteration=ForwardPassIteration(tuple(sorted(ranks, key=lambda rank: rank.dp_rank))),
                )
            )
        if unavailable:
            issues.append(
                MeasurementIssue(
                    MeasurementState.MEASUREMENT_UNAVAILABLE,
                    "incomplete_or_invalid_benchmark_point",
                    file.measurement_file_id,
                    file.path,
                    unavailable,
                )
            )
        for missing in payload.get("missing_phases") or ():
            issues.append(
                MeasurementIssue(
                    MeasurementState.MEASUREMENT_UNAVAILABLE,
                    f"missing_phase:{missing}",
                    file.measurement_file_id,
                    file.path,
                )
            )
        skipped = payload.get("skipped_points") or ()
        if skipped:
            issues.append(
                MeasurementIssue(
                    MeasurementState.MEASUREMENT_UNAVAILABLE,
                    "benchmark_skipped_point",
                    file.measurement_file_id,
                    file.path,
                    len(skipped),
                )
            )
    return ParseResult(tuple(observations), issues=tuple(issues))


def _parse_evaluation_records(
    configuration_id: str,
    files: Sequence[MeasurementFile],
    _attention_dp_size: int | None,
) -> ParseResult:
    observations: list[ParsedObservation] = []
    for file in files:
        if not file.local_path.name.endswith(_FORWARD_PASS_RECORD_SUFFIXES):
            raise DataError(f"measurement protocol forward-pass-record-v1 does not support file {file.path!r}")
        with _open_text(file.local_path) as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or ()
            if len(fieldnames) != len(set(fieldnames)):
                raise DataError(f"derived truth {file.path} contains duplicate columns")
            if not _FORWARD_PASS_RECORD_FIELDS.issubset(fieldnames):
                missing = sorted(_FORWARD_PASS_RECORD_FIELDS - set(fieldnames))
                raise DataError(f"derived truth {file.path} is missing columns: {missing}")
            for source_row, row in enumerate(reader, start=2):
                measurement_id = row["measurement_id"]
                if not isinstance(measurement_id, str) or not measurement_id:
                    raise DataError(f"derived truth {file.path} row {source_row} has an empty measurement_id")
                phase = row["phase"]
                if phase not in {"prefill", "decode"}:
                    raise DataError(f"derived truth {file.path} row {source_row} has unknown phase {phase!r}")
                batch = _positive_csv_int(row.get("batch_size"), file, source_row, "batch_size")
                prefill_tokens = _csv_nonnegative_int_or_zero(
                    row.get("total_prefill_tokens"), file, source_row, "total_prefill_tokens"
                )
                kv_tokens = _csv_nonnegative_int_or_zero(
                    row.get("total_kv_read_tokens"), file, source_row, "total_kv_read_tokens"
                )
                if phase == "prefill":
                    if prefill_tokens == 0:
                        raise DataError(
                            f"derived truth {file.path} row {source_row} has non-positive total_prefill_tokens"
                        )
                    scheduled = RequestMetrics(
                        num_prefill_requests=batch,
                        sum_prefill_tokens=prefill_tokens,
                        sum_prefill_kv_tokens=kv_tokens,
                    )
                else:
                    if prefill_tokens != 0:
                        raise DataError(
                            f"derived truth {file.path} row {source_row} must use zero total_prefill_tokens for decode"
                        )
                    scheduled = RequestMetrics(num_decode_requests=batch, sum_decode_kv_tokens=kv_tokens)
                _validate_optional_csv_provenance(row, fieldnames, file, source_row)
                truth_latency_ms = _csv_finite_number(row.get("truth_latency_ms"), file, source_row, "truth_latency_ms")
                if truth_latency_ms <= 0:
                    raise DataError(f"derived truth {file.path} row {source_row} has non-positive truth_latency_ms")
                rank = _RankMeasurement(
                    version=1,
                    worker_id="derived-truth",
                    dp_rank=0,
                    counter_id=source_row - 2,
                    wall_time_s=truth_latency_ms / 1000.0,
                    observed_at_unix_ms=None,
                    scheduled=scheduled,
                    queued=RequestMetrics(),
                )
                metric = _metric(
                    configuration_id=configuration_id,
                    file=file,
                    source_row=source_row,
                    rank_index=0,
                    rank=rank,
                )
                observations.append(
                    ParsedObservation(
                        source_file=file,
                        source_row=source_row,
                        identity_hint=measurement_id,
                        iteration=ForwardPassIteration.single_rank(metric),
                    )
                )
    return ParseResult(tuple(observations))


def _complete_group_fpms(
    group: Mapping[str, Any],
    attention_dp_size: int | None,
    file: MeasurementFile,
    source_row: int,
) -> list[Any] | None:
    """Return one synchronized FPM per exact attention-DP rank.

    A grouped benchmark is usable only when its declared ranks, outer rank
    wrappers, and inner FPM ranks all agree with the selected configuration.
    Rank-local records are not enough to reconstruct an engine iteration.
    """

    if attention_dp_size is None or attention_dp_size <= 0:
        raise DataError(f"cannot validate benchmark ranks for {file.path} without attention_dp_size")
    expected = list(range(attention_dp_size))
    declared = group.get("expected_dp_ranks")
    rank_results = group.get("rank_results")
    if (
        not isinstance(declared, list)
        or any(isinstance(rank, bool) or not isinstance(rank, int) for rank in declared)
        or declared != expected
        or not isinstance(rank_results, list)
        or len(rank_results) != attention_dp_size
    ):
        return None

    by_rank: dict[int, Mapping[str, Any]] = {}
    for rank_result in rank_results:
        if not isinstance(rank_result, Mapping):
            return None
        wrapper_rank = _optional_nonnegative_int(rank_result.get("dp_rank"))
        fpms = rank_result.get("fpms")
        if wrapper_rank is None or not isinstance(fpms, list) or len(fpms) != 1:
            return None
        fpm = fpms[0]
        if not isinstance(fpm, Mapping):
            return None
        inner_rank = _optional_nonnegative_int(fpm.get("dp_rank"))
        if inner_rank != wrapper_rank or wrapper_rank in by_rank:
            return None
        by_rank[wrapper_rank] = fpm
    if sorted(by_rank) != expected:
        return None

    fpms = [by_rank[rank] for rank in expected]
    benchmark_id = _optional_nonnegative_int(group.get("benchmark_id"))
    point = group.get("point")
    point_id = _optional_nonnegative_int(point.get("benchmark_id")) if isinstance(point, Mapping) else None
    counter_ids = {_optional_nonnegative_int(fpm.get("counter_id")) for fpm in fpms}
    worker_ids = {str(fpm.get("worker_id") or "") for fpm in fpms}
    if (
        benchmark_id is None
        or point_id != benchmark_id
        or counter_ids != {benchmark_id}
        or len(worker_ids) != 1
        or "" in worker_ids
    ):
        return None
    return fpms


def _validate_group_record(
    group: Mapping[str, Any],
    file: MeasurementFile,
    source_row: int,
) -> None:
    required = {"benchmark_id", "point", "expected_dp_ranks", "complete", "wall_time", "rank_results"}
    _require_fields(group, required, file, source_row)
    benchmark_id = _json_nonnegative_int(group["benchmark_id"], file, source_row, "benchmark_id")
    if group["complete"] is not True:
        raise DataError(f"measurement file {file.path} row {source_row} field complete must be true")
    _json_positive_number(group["wall_time"], file, source_row, "wall_time")

    _validate_benchmark_point(group["point"], benchmark_id, file, source_row, prefix="point.")

    expected = group["expected_dp_ranks"]
    if not isinstance(expected, list) or not expected:
        raise DataError(f"measurement file {file.path} row {source_row} expected_dp_ranks must be a non-empty array")
    parsed_expected = [_json_nonnegative_int(value, file, source_row, "expected_dp_ranks") for value in expected]
    if len(parsed_expected) != len(set(parsed_expected)):
        raise DataError(f"measurement file {file.path} row {source_row} expected_dp_ranks must be unique")

    rank_results = group["rank_results"]
    if not isinstance(rank_results, list) or not rank_results:
        raise DataError(f"measurement file {file.path} row {source_row} rank_results must be a non-empty array")
    for rank_result in rank_results:
        if not isinstance(rank_result, Mapping):
            raise DataError(f"measurement file {file.path} row {source_row} contains a non-object rank result")
        _require_fields(rank_result, {"dp_rank", "fpms"}, file, source_row, prefix="rank_results[].")
        _json_nonnegative_int(rank_result["dp_rank"], file, source_row, "rank_results[].dp_rank")
        fpms = rank_result["fpms"]
        if not isinstance(fpms, list) or len(fpms) != 1:
            raise DataError(
                f"measurement file {file.path} row {source_row} rank_results[].fpms must contain one record"
            )
        if not isinstance(fpms[0], Mapping):
            raise DataError(f"measurement file {file.path} row {source_row} contains a non-object FPM")
        _rank_measurement(fpms[0], file, source_row)


def _validate_benchmark_point(
    value: Any,
    benchmark_id: int,
    file: MeasurementFile,
    source_row: int,
    *,
    prefix: str,
) -> None:
    if not isinstance(value, Mapping):
        raise DataError(f"measurement file {file.path} row {source_row} field {prefix[:-1]} must be an object")
    _require_fields(
        value,
        {"point_type", "benchmark_id", "batch_size", "total_prefill_tokens", "total_kv_read_tokens"},
        file,
        source_row,
        prefix=prefix,
    )
    if value["point_type"] not in {"prefill", "decode", "mixed"}:
        raise DataError(f"measurement file {file.path} row {source_row} has invalid {prefix}point_type")
    point_id = _json_nonnegative_int(value["benchmark_id"], file, source_row, f"{prefix}benchmark_id")
    if point_id != benchmark_id:
        raise DataError(f"measurement file {file.path} row {source_row} has inconsistent benchmark IDs")
    _json_positive_int(value["batch_size"], file, source_row, f"{prefix}batch_size")
    _json_nonnegative_int(value["total_prefill_tokens"], file, source_row, f"{prefix}total_prefill_tokens")
    _json_nonnegative_int(value["total_kv_read_tokens"], file, source_row, f"{prefix}total_kv_read_tokens")


def _require_fields(
    value: Mapping[str, Any],
    fields: set[str],
    file: MeasurementFile,
    source_row: int,
    *,
    prefix: str = "",
) -> None:
    missing = sorted(field for field in fields if field not in value)
    if missing:
        names = [f"{prefix}{field}" for field in missing]
        raise DataError(f"measurement file {file.path} row {source_row} is missing fields: {names}")


def _rank_measurement(
    payload: Mapping[str, Any],
    file: MeasurementFile,
    source_row: int,
) -> _RankMeasurement:
    _require_fields(
        payload,
        {"version", "worker_id", "dp_rank", "counter_id", "wall_time", "scheduled_requests", "queued_requests"},
        file,
        source_row,
    )
    version = _json_nonnegative_int(payload["version"], file, source_row, "version")
    if version != 1:
        raise DataError(f"measurement file {file.path} row {source_row} field version must be 1")
    worker_id = payload["worker_id"]
    if not isinstance(worker_id, str) or not worker_id:
        raise DataError(f"measurement file {file.path} row {source_row} has invalid worker_id")
    observed_at_unix_ms = None
    if "observed_at_unix_ms" in payload:
        observed_at_unix_ms = _json_nonnegative_number(
            payload["observed_at_unix_ms"], file, source_row, "observed_at_unix_ms"
        )
    return _RankMeasurement(
        version=version,
        worker_id=worker_id,
        dp_rank=_json_nonnegative_int(payload["dp_rank"], file, source_row, "dp_rank"),
        counter_id=_json_nonnegative_int(payload["counter_id"], file, source_row, "counter_id"),
        wall_time_s=_json_nonnegative_number(payload["wall_time"], file, source_row, "wall_time"),
        observed_at_unix_ms=observed_at_unix_ms,
        scheduled=_request_metrics(payload["scheduled_requests"], file, source_row, "scheduled_requests"),
        queued=_request_metrics(payload["queued_requests"], file, source_row, "queued_requests"),
    )


def _metric(
    *,
    configuration_id: str,
    file: MeasurementFile,
    source_row: int,
    rank_index: int,
    rank: _RankMeasurement,
) -> ForwardPassMetric:
    digest = hashlib.sha256(
        f"{configuration_id}|{file.measurement_file_id}|{source_row}|{rank_index}".encode()
    ).digest()
    # Keep compatibility with the existing OOP model while the public evidence
    # identity remains the stable string carried by MeasurementObservation.
    fpm_id = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return ForwardPassMetric(
        fpm_id=fpm_id,
        configuration_id=configuration_id,
        version=rank.version,
        worker_id=rank.worker_id,
        dp_rank=rank.dp_rank,
        counter_id=rank.counter_id,
        wall_time_s=rank.wall_time_s,
        scheduled=rank.scheduled,
        queued=rank.queued,
        event_time=str(rank.observed_at_unix_ms) if rank.observed_at_unix_ms is not None else "",
        capture_id=file.measurement_file_id,
        stream_id=file.path,
        extra_metadata={"hf_measurement_file_id": file.measurement_file_id, "hf_source_row": source_row},
    )


def _request_metrics(value: Any, file: MeasurementFile, source_row: int, field: str) -> RequestMetrics:
    if not isinstance(value, Mapping):
        raise DataError(f"measurement file {file.path} row {source_row} field {field} must be an object")
    try:
        return RequestMetrics(
            num_prefill_requests=_request_integer(value, "num_prefill_requests"),
            sum_prefill_tokens=_request_integer(value, "sum_prefill_tokens"),
            var_prefill_length=_request_number(value, "var_prefill_length"),
            sum_prefill_kv_tokens=_request_integer(value, "sum_prefill_kv_tokens"),
            num_decode_requests=_request_integer(value, "num_decode_requests"),
            sum_decode_kv_tokens=_request_integer(value, "sum_decode_kv_tokens"),
            var_decode_kv_tokens=_request_number(value, "var_decode_kv_tokens"),
        )
    except (TypeError, ValueError, DataError) as exc:
        raise DataError(f"invalid {field} in {file.path} row {source_row}: {exc}") from exc


def _workload(metrics: RequestMetrics) -> WorkloadKind:
    if metrics.sum_prefill_tokens > 0 and (metrics.num_decode_requests > 0 or metrics.sum_decode_kv_tokens > 0):
        return WorkloadKind.MIXED
    if metrics.sum_prefill_tokens > 0:
        return WorkloadKind.PREFILL
    if metrics.num_decode_requests > 0 or metrics.sum_decode_kv_tokens > 0:
        return WorkloadKind.DECODE
    return WorkloadKind.EMPTY


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _read_json(file: MeasurementFile) -> Any:
    try:
        with _open_text(file.local_path) as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot parse measurement JSON {file.path}: {exc}") from exc


def _json_line(line: str, file: MeasurementFile, source_row: int) -> Mapping[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DataError(f"cannot parse {file.path} row {source_row}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DataError(f"measurement file {file.path} row {source_row} must be an object")
    return payload


def _json_nonempty_string(value: Any, file: MeasurementFile, source_row: int, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DataError(f"measurement file {file.path} row {source_row} has invalid {field}")
    return value


def _json_time_unit(value: Any, file: MeasurementFile, source_row: int, field: str) -> str:
    if value not in _TIME_UNIT_SECONDS:
        raise DataError(f"measurement file {file.path} row {source_row} has invalid {field}")
    return str(value)


def _json_decimal(value: Any, file: MeasurementFile, source_row: int, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DataError(f"measurement file {file.path} row {source_row} has invalid {field}")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise DataError(f"measurement file {file.path} row {source_row} has invalid {field}") from exc
    if not result.is_finite():
        raise DataError(f"measurement file {file.path} row {source_row} has non-finite {field}")
    return result


def _json_unique_nonnegative_ints(
    value: Any,
    file: MeasurementFile,
    source_row: int,
    field: str,
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise DataError(f"measurement file {file.path} row {source_row} field {field} must be a non-empty array")
    result = [_json_nonnegative_int(item, file, source_row, field) for item in value]
    if len(result) != len(set(result)):
        raise DataError(f"measurement file {file.path} row {source_row} field {field} must contain unique values")
    return result


def _required_int(value: Any, file: MeasurementFile, source_row: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise DataError(f"measurement file {file.path} row {source_row} has invalid {field}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DataError(f"measurement file {file.path} row {source_row} has invalid {field}") from exc
    if result < 0:
        raise DataError(f"measurement file {file.path} row {source_row} has negative {field}")
    return result


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _positive_csv_int(value: Any, file: MeasurementFile, source_row: int, field: str) -> int:
    result = _required_int(value, file, source_row, field)
    if result <= 0:
        raise DataError(f"measurement file {file.path} row {source_row} has non-positive {field}")
    return result


def _csv_nonnegative_int_or_zero(
    value: Any,
    file: MeasurementFile,
    source_row: int,
    field: str,
) -> int:
    if value in (None, ""):
        return 0
    return _required_int(value, file, source_row, field)


def _csv_finite_number(value: Any, file: MeasurementFile, source_row: int, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataError(f"measurement file {file.path} row {source_row} has invalid {field}") from exc
    if not math.isfinite(result):
        raise DataError(f"measurement file {file.path} row {source_row} has non-finite {field}")
    return result


def _validate_optional_csv_provenance(
    row: Mapping[str, Any],
    fieldnames: Sequence[str],
    file: MeasurementFile,
    source_row: int,
) -> None:
    if "source_file" in fieldnames:
        source_file = row.get("source_file")
        if not isinstance(source_file, str) or not source_file:
            raise DataError(f"derived truth {file.path} row {source_row} has an empty source_file")
    if "source_row" in fieldnames:
        _positive_csv_int(row.get("source_row"), file, source_row, "source_row")


def _json_nonnegative_int(value: Any, file: MeasurementFile, source_row: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataError(f"measurement file {file.path} row {source_row} has invalid {field}")
    if value < 0:
        raise DataError(f"measurement file {file.path} row {source_row} has negative {field}")
    return value


def _json_positive_int(value: Any, file: MeasurementFile, source_row: int, field: str) -> int:
    result = _json_nonnegative_int(value, file, source_row, field)
    if result == 0:
        raise DataError(f"measurement file {file.path} row {source_row} has non-positive {field}")
    return result


def _json_nonnegative_number(value: Any, file: MeasurementFile, source_row: int, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError(f"measurement file {file.path} row {source_row} has invalid {field}")
    result = float(value)
    if not math.isfinite(result):
        raise DataError(f"measurement file {file.path} row {source_row} has non-finite {field}")
    if result < 0:
        raise DataError(f"measurement file {file.path} row {source_row} has negative {field}")
    return result


def _json_positive_number(value: Any, file: MeasurementFile, source_row: int, field: str) -> float:
    result = _json_nonnegative_number(value, file, source_row, field)
    if result == 0:
        raise DataError(f"measurement file {file.path} row {source_row} has non-positive {field}")
    return result


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("value must be an integer")
    result = value
    if result < 0:
        raise ValueError("value must be non-negative")
    return result


def _nonnegative_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value must be finite")
    if result < 0:
        raise ValueError("value must be non-negative")
    return result


def _request_integer(value: Mapping[str, Any], field: str) -> int:
    return _nonnegative_int(value[field]) if field in value else 0


def _request_number(value: Mapping[str, Any], field: str) -> float:
    return _nonnegative_number(value[field]) if field in value else 0.0


def _issues(file: MeasurementFile, counts: Counter[str], state: MeasurementState) -> Iterator[MeasurementIssue]:
    for reason, count in sorted(counts.items()):
        yield MeasurementIssue(state, reason, file.measurement_file_id, file.path, count)
