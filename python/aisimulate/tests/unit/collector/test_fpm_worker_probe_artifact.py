# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from collector.fpm_forward.native_artifact import _rank_artifacts

pytestmark = pytest.mark.unit


def test_worker_probe_sidecar_is_not_a_native_rank(tmp_path):
    rank = {"artifact_type": "rank", "dp": {"rank": 0}}
    (tmp_path / "benchmark_dp0.json").write_text(json.dumps(rank))
    (tmp_path / "benchmark_merged.json").write_text(json.dumps({"artifact_type": "merged"}))
    sidecar = tmp_path / "benchmark_merged_worker_probe.json"
    sidecar.write_text(json.dumps({"schema": "dynamo.fpm.benchmark_worker_probe", "schema_version": 1}))
    assert _rank_artifacts(tmp_path) == [(tmp_path / "benchmark_dp0.json", rank)]
    assert sidecar.exists()


@pytest.mark.parametrize("payload", [{}, {"schema": "dynamo.fpm.benchmark_worker_probe", "schema_version": 2}])
def test_unrecognized_worker_probe_envelope_fails(tmp_path, payload):
    (tmp_path / "benchmark_merged_worker_probe.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="worker probe sidecar"):
        _rank_artifacts(tmp_path)
