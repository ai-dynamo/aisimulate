# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep source timestamp spelling and expose protocol-normalized milliseconds."""

import json
from dataclasses import replace
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_hf_dataset import (
    CONFIGURATION_PATH,
    _build_dataset,
    _build_pre_grouped_dataset,
    _fpm_payload,
    _listener_iteration,
)

from scripts.fpm_accuracy.exceptions import DataError
from scripts.fpm_accuracy.hf.dataset import HfDataset, _membership_sha256
from scripts.fpm_accuracy.hf.protocols import adapter_for


def _case(tmp_path, rows, *, dp=1):
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        include_fpm=False,
        dp=dp,
        files=[("truth", "truth.jsonl", "\n".join(map(json.dumps, rows)).encode())],
    )
    return dataset.measurement_case(CONFIGURATION_PATH)


def test_rank_timestamps_preserve_submillisecond_precision_and_ties(tmp_path):
    rows = [
        {**_fpm_payload(counter=index), "observed_at_unix_ms": value}
        for index, value in enumerate((6000.75, 1000.25, 6000.75, 0.0))
    ]
    observations = _case(tmp_path, rows).observations
    assert [item.event_time_ms for item in observations] == [
        0.0,
        1000.25,
        6000.75,
        6000.75,
    ]
    assert [item.event_time for item in observations] == ["0", "1000", "6000", "6000"]
    assert all("event_time_ms" not in rank for item in observations for rank in item.prediction_input().rank_payloads)


@pytest.mark.parametrize("unit,factor", [("s", 1), ("ms", 1000), ("us", 1000000), ("ns", 1000000000)])
def test_listener_timestamp_unit_is_normalized_by_protocol_not_guessed(tmp_path, unit, factor):
    row = _listener_iteration()
    row["grouping"]["source_unit"] = unit
    row["grouping"]["window_start"] = str(Decimal(row["grouping"]["window_start"]) * factor)
    for rank in row["rank_measurements"]:
        rank["_recv_ts"] = str(Decimal(rank["_recv_ts"]) * factor)
    observation = _case(tmp_path, [row], dp=2).observations[0]
    assert observation.event_time_ms == 1000001.0
    assert observation.event_time == row["grouping"]["window_start"]


def test_missing_and_unbound_group_timestamps_remain_unavailable(tmp_path):
    row = _fpm_payload()
    del row["observed_at_unix_ms"]
    assert _case(tmp_path / "raw", [row]).observations[0].event_time_ms is None
    grouped = _build_pre_grouped_dataset(
        tmp_path / "grouped", [[_fpm_payload(rank=0), _fpm_payload(rank=1)]]
    ).measurement_case(CONFIGURATION_PATH)
    # Rank-local timestamps do not establish an authoritative group timestamp.
    assert grouped.observations[0].event_time_ms is None


def test_normalized_timestamp_does_not_change_existing_observation_or_membership_identity(
    tmp_path,
):
    case = _case(tmp_path, [_fpm_payload()])
    adapter = adapter_for(case.protocol_id)
    parsed = adapter.parse(case.configuration_id, case.truth_files, 1).observations[0]
    with_time = HfDataset._materialize_observations(case.configuration_id, [parsed])
    without_time = HfDataset._materialize_observations(case.configuration_id, [replace(parsed, chronology_key=None)])
    assert with_time[0].event_time_ms == 1001.0 and without_time[0].event_time_ms is None
    assert with_time[0].observation_id == without_time[0].observation_id
    assert _membership_sha256(with_time) == _membership_sha256(without_time) == case.measurement_membership_sha256


@pytest.mark.parametrize(
    "key",
    [
        (),
        (True, 0),
        ("1000", 0),
        (None, 0),
        (-1, 0),
        (float("nan"), 0),
        (float("inf"), 0),
    ],
)
def test_materialization_rejects_invalid_normalized_chronology(tmp_path, key):
    case = _case(tmp_path, [_fpm_payload()])
    parsed = adapter_for(case.protocol_id).parse(case.configuration_id, case.truth_files, 1).observations[0]
    with pytest.raises(DataError, match="normalized|chronology"):
        HfDataset._materialize_observations(case.configuration_id, [replace(parsed, chronology_key=key)])


def test_columnar_rank_storage_retains_same_normalized_time_and_identity(tmp_path):
    row = {**_fpm_payload(), "observed_at_unix_ms": 6000.125}
    case = _case(tmp_path / "source", [row])
    source = case.truth_files[0]
    record = {
        **row,
        "received_at_ns": 6000125000,
        "extensions_json": "{}",
        "queued_requests": {"num_decode_requests": 0},
    }
    table = pa.Table.from_pylist(
        [
            {
                "source_row": 1,
                "layout": "rank_observation",
                "rank_measurements": [record],
            }
        ]
    ).replace_schema_metadata({b"schema_id": b"fpm-iterations-parquet-v1", b"schema_version": b"1"})
    parquet = tmp_path / "rank.parquet"
    pq.write_table(table, parquet)
    file = replace(
        source,
        local_path=parquet,
        storage_format="parquet",
        storage_schema="fpm-iterations-parquet-v1",
        source_layout="stream",
        logical_row_count=1,
    )
    parsed = adapter_for(case.protocol_id).parse(case.configuration_id, [file], 1)
    observation = HfDataset._materialize_observations(case.configuration_id, parsed.observations)[0]
    assert observation.event_time_ms == 6000.125
    assert observation.observation_id == case.observations[0].observation_id
