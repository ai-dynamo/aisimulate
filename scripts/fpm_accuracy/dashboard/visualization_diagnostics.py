# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified from AISim FPM Gym f934c030afc3a03cb04d8f3ff4709194f7445c98.
# See ../README.md for upstream paths and modifications.
"""Visualization-only diagnostic grouping; never added to evaluation cases."""

from collections import Counter, defaultdict

from scripts.fpm_accuracy.hf.models import MeasurementCase, MeasurementObservation
from scripts.fpm_accuracy.hf.protocols import _metric, _rank_measurement, _stream_records
from scripts.fpm_accuracy.types.forward_pass import ForwardPassIteration, WorkloadKind

POLICY = "diagnostic-file-worker-counter-complete-dp-v1"


def diagnostic_observations(case: MeasurementCase) -> tuple[list[MeasurementObservation], dict[str, int]]:
    """Only reconsider raw streams explicitly rejected as unsynchronized.

    Shared rank validation is reused, but this grouping is not a protocol adapter
    and cannot change accepted measurement membership, evaluation or history.
    A duplicate or unexpected rank invalidates the entire counter group.
    """
    ids = {issue.source_file_id for issue in case.issues if issue.reason == "unsynchronized_attention_dp_stream"}
    observations = []
    counts = Counter()
    expected = set(range(case.configuration.worker_config_record.config.parallelism.attention_dp_size))
    for file in case.truth_files:
        if file.measurement_file_id not in ids:
            continue
        grouped = defaultdict(list)
        for source_row, payload in _stream_records(file):
            rank = _rank_measurement(payload, file, source_row)
            grouped[(rank.worker_id, rank.counter_id)].append((source_row, rank))
        for (worker, counter), rows in sorted(grouped.items()):
            if len(rows) != len(expected) or {rank.dp_rank for _, rank in rows} != expected:
                counts["incomplete_duplicate_or_unexpected_rank_group"] += 1
                continue
            ranks = tuple(
                _metric(
                    configuration_id=case.configuration_id,
                    file=file,
                    source_row=line,
                    rank_index=rank.dp_rank,
                    rank=rank,
                )
                for line, rank in sorted(rows, key=lambda row: row[1].dp_rank)
            )
            iteration = ForwardPassIteration(ranks)
            if iteration.workload_kind is WorkloadKind.EMPTY or iteration.observed_time_ms is None:
                counts["empty_or_nonpositive_latency_group"] += 1
                continue
            observations.append(
                MeasurementObservation(
                    observation_id=f"{POLICY}:{file.measurement_file_id}:{worker}:{counter}",
                    configuration_id=case.configuration_id,
                    order=len(observations),
                    source_file_id=file.measurement_file_id,
                    source_path=file.path,
                    source_row=min(line for line, _ in rows),
                    iteration=iteration,
                )
            )
    return observations, dict(counts)
