# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aisim-e2e/v1: InferenceX and developer fixtures share one evaluator loader."""

import gzip
import json

import pyarrow.parquet as pq
import pytest

from aisim_core.data import check_manifest
from aisim_core.data.e2e import (
    LOG_CAP_BYTES,
    E2eManifestBody,
    MetricRow,
    ProvenanceFile,
    Setting,
    SnapshotAcquisition,
    SnapshotMember,
    aiperf_to_configuration,
    evaluation_points,
    inferencex_to_configurations,
    load_e2e,
    make_metric,
    public_release_problems,
    read_provenance,
    read_raw_record,
    read_snapshot,
    redact,
    store_provenance,
    throughput_per_device,
    to_canonical,
    write_configuration,
    write_snapshot,
)
from aisim_core.data.loaders import LoaderError
from aisim_core.data.parquet import read_parquet_schema_id

pytestmark = pytest.mark.unit


def _infx_record(**over):
    # Same shape as the InferenceX cache fixture in tests/test_e2e_hf_measurements.py.
    rec = dict(
        model="test-model",
        hardware="h100",
        framework="vllm",
        precision="fp8",
        spec_method="none",
        disagg=False,
        is_multinode=False,
        prefill_tp=1,
        prefill_ep=1,
        prefill_dp_attention=False,
        prefill_num_workers=0,
        decode_tp=8,
        decode_ep=1,
        decode_dp_attention=False,
        decode_num_workers=1,
        num_prefill_gpu=0,
        num_decode_gpu=8,
        id=123,
        date="2026-10-17",
        isl=1024,
        osl=128,
        conc=32,
        metrics={
            "mean_ttft": 0.1,
            "median_ttft": 0.09,
            "p99_ttft": 0.3,
            "mean_tpot": 0.01,
            "median_tpot": 0.0,
            "mean_e2el": 1.5,
            "tput_per_gpu": 1200.0,
            "output_tput_per_gpu": 150.0,
            "mean_intvty": 100.0,
            "weird_metric": 7,
        },
        image="test-image",
        benchmark_type="single_turn",
        run_url="https://github.com/SemiAnalysisAI/InferenceX/actions/runs/987/attempts/2",
    )
    rec.update(over)
    return rec


def _dev_body():
    return E2eManifestBody(
        model_id="test-model",
        deployment="aggregated",
        hardware="h100",
        framework="vllm",
        framework_version="0.11.0",
        parallelism={"aggregated": {"tp": 8, "ep": 1}},
        parallelism_complete=True,
        precision={"weights": "fp8"},
        settings={"max_num_seqs": Setting(value=256, authority="observed")},
        source_kind="developer",
    )


def _aiperf_summary(**over):
    s = {
        "is_complete": True,
        "was_cancelled": False,
        "request_count": {"avg": 2},
        "error_request_count": {"avg": 0},
        "time_to_first_token": {"unit": "ms", "avg": 100.0, "p50": 90.0, "p90": 150.0, "p99": 300.0},
        "inter_token_latency": {"unit": "ms", "avg": 10.0, "p50": 9.0, "p90": 14.0},
        "request_latency": {"unit": "seconds", "avg": 1.5, "p50": 1.4, "p90": 2.0},
        "output_token_throughput": {"unit": "tokens/second", "avg": 9600.0},
    }
    s.update(over)
    return s


def _aiperf_records():
    return [
        {
            "request_id": i,
            "error": None,
            "metrics": {
                "input_sequence_length": {"value": 1024, "unit": "tokens"},
                "output_sequence_length": {"value": 128, "unit": "tokens"},
                "time_to_first_token": {"value": 0.1, "unit": "seconds"},
                "inter_token_latency": {"value": 10, "unit": "ms"},
            },
        }
        for i in range(2)
    ]


def _dev_config(**kw):
    return aiperf_to_configuration(
        _aiperf_summary(), _aiperf_records(), _dev_body(), observation_id="dev:1", device_count=8, **kw
    )


# ---------------------------------------------------------------- metrics


def test_missing_is_never_zero():
    assert make_metric("o", "ttft", "mean", None, "s").status == "missing"
    zero = make_metric("o", "ttft", "mean", 0, "s")
    assert zero.status == "valid" and zero.value == 0.0
    for bad in (float("nan"), float("inf"), -1.0, True, "1"):
        row = make_metric("o", "ttft", "mean", bad, "s")
        assert row.status == "invalid" and row.value is None
    with pytest.raises(ValueError):
        MetricRow(observation_id="o", metric="ttft", statistic="mean", unit="ms", status="missing", value=0.0)
    with pytest.raises(ValueError):
        MetricRow(observation_id="o", metric="ttft", statistic="mean", unit="ms", status="valid")


def test_unit_conversions():
    assert to_canonical(0.25, "s", "ms") == 250.0
    assert to_canonical(250, "milliseconds", "ms") == 250
    assert to_canonical(500, "us", "ms") == pytest.approx(0.5)
    assert to_canonical(3, "tok/s", "tokens/s") == 3
    with pytest.raises(ValueError):
        to_canonical(1, "furlongs", "ms")
    with pytest.raises(ValueError):
        to_canonical(1, "tokens/s", "ms")
    row = make_metric("o", "ttft", "p90", 0.2, "s")
    assert (row.value, row.unit, row.source_value, row.source_unit) == (200.0, "ms", 0.2, "s")


def test_per_device_throughput_is_explicit():
    per = make_metric("o", "output_throughput", "value", 150.0, "tokens/s", normalization="per_device", device_count=8)
    total = make_metric("o", "output_throughput", "value", 1200.0, "tokens/s", normalization="total", device_count=8)
    assert throughput_per_device(per) == 150.0 and throughput_per_device(total) == 150.0
    unknown = total.model_copy(update={"device_count": None})
    with pytest.raises(ValueError, match="device_count"):
        throughput_per_device(unknown)
    with pytest.raises(ValueError):
        throughput_per_device(make_metric("o", "ttft", "mean", 1, "ms"))


# ---------------------------------------------------------------- round trip through one loader


def test_inferencex_and_developer_share_the_evaluator_loader(tmp_path):
    (infx,) = inferencex_to_configurations([_infx_record()])
    dev = _dev_config(log=b"server log\n" * 10, log_redacted=True)
    points = {}
    for name, cfg in (("infx", infx), ("dev", dev)):
        path = write_configuration(tmp_path, cfg)
        check = load_e2e(tmp_path, path, config=None)
        assert check.manifest.category == "e2e" and check.manifest.configuration_id == cfg.body.configuration_id()
        (point,) = evaluation_points(check)
        points[name] = point
        for artifact in check.manifest.artifacts:
            if artifact.format == "parquet":
                assert read_parquet_schema_id(tmp_path / artifact.path) == artifact.artifact_schema
                assert pq.ParquetFile(tmp_path / artifact.path).metadata.row_group(0).column(0).compression == "ZSTD"
    i, d = points["infx"], points["dev"]
    # Canonical units regardless of source: seconds and ms both become ms.
    assert i.metrics[("ttft", "mean")] == pytest.approx(100.0) and d.metrics[("ttft", "mean")] == pytest.approx(100.0)
    assert i.metrics[("e2el", "mean")] == pytest.approx(1500.0) and d.metrics[("e2el", "mean")] == pytest.approx(1500.0)
    assert d.metrics[("ttft", "p90")] == 150.0
    assert ("ttft", "p90") in i.unavailable  # InferenceX did not report P90
    assert ("itl", "mean") in d.unavailable  # AIPerf has no separate ITL
    assert (
        i.per_device_throughput["output_throughput"] == 150.0 and d.per_device_throughput["output_throughput"] == 1200.0
    )
    assert i.per_device_throughput["total_throughput"] == 1200.0
    assert set(i.__dataclass_fields__) == set(d.__dataclass_fields__)


def test_inferencex_metrics_every_statistic_zero_vs_missing(tmp_path):
    (cfg,) = inferencex_to_configurations([_infx_record()])
    path = write_configuration(tmp_path, cfg)
    ds = load_e2e(tmp_path, path)
    by_key = {(m.metric, m.statistic): m for m in ds.metrics}
    assert by_key[("ttft", "p99")].value == pytest.approx(300.0)  # extra statistic stored
    assert by_key[("tpot", "median")].status == "valid" and by_key[("tpot", "median")].value == 0.0
    assert by_key[("itl", "mean")].status == "missing" and by_key[("itl", "mean")].value is None
    for metric in ("ttft", "tpot", "itl", "e2el"):
        for stat in ("mean", "median", "p90"):
            assert (metric, stat) in by_key
    assert by_key[("output_throughput", "value")].normalization == "per_device"
    assert by_key[("output_throughput", "value")].device_count == 8
    assert by_key[("interactivity", "mean")].normalization == "per_user"
    # Unmapped metrics are not invented; they stay in the raw artifact and are noted verbatim.
    assert ds.body.source_fields["unmapped_metrics"] == ["weird_metric"]
    raw = read_raw_record(tmp_path, ds, ds.observations[0].observation_id)
    assert b"weird_metric" in raw
    assert json.loads(ds.observations[0].unmapped_metrics_json) == {"weird_metric": 7}  # typed, verbatim
    assert ds.body.dp_attention == {"decode": False, "prefill": False}  # booleans never become degrees


def test_inferencex_disaggregated_groups_and_identity():
    configs = inferencex_to_configurations(
        [
            _infx_record(),
            _infx_record(id=124, conc=64),
            _infx_record(id=125, disagg=True, prefill_tp=4, prefill_ep=2, prefill_num_workers=1, num_prefill_gpu=4),
        ]
    )
    assert sorted(len(c.observations) for c in configs) == [1, 2]
    dis = next(c for c in configs if c.body.deployment == "disaggregated")
    assert dis.body.parallelism == {"decode": {"tp": 8}, "prefill": {"tp": 4, "ep": 2}}
    assert dis.observations[0].device_count == 12
    assert dis.body.configuration_id() != next(c for c in configs if c is not dis).body.configuration_id()


def test_open_settings_do_not_change_identity():
    base = _dev_body()
    more = base.model_copy(update={"settings": {**base.settings, "new_flag": Setting(value=True, authority="unknown")}})
    assert base.configuration_id() == more.configuration_id()
    other = base.model_copy(update={"parallelism": {"aggregated": {"tp": 4}}})
    assert base.configuration_id() != other.configuration_id()


def test_requests_table_and_developer_observation(tmp_path):
    cfg = _dev_config()
    ds = load_e2e(tmp_path, write_configuration(tmp_path, cfg))
    assert len(ds.requests) == 2 and ds.requests[0].ttft_ms == pytest.approx(100.0)
    assert ds.requests[0].itl_ms is None  # not measured is null, not zero
    obs = ds.observations[0]
    assert (obs.input_tokens, obs.output_tokens, obs.requests_total, obs.outcome) == (1024, 128, 2, "complete")
    assert check_manifest is not None


def test_loader_rejects_tamper_and_wrong_schema(tmp_path):
    cfg = _dev_config()
    path = write_configuration(tmp_path, cfg)
    table = next(tmp_path.rglob("metrics.parquet"))
    table.write_bytes(table.read_bytes() + b"x")
    with pytest.raises(ValueError):
        load_e2e(tmp_path, path)


# ---------------------------------------------------------------- provenance


def test_provenance_cap_redaction_and_hash_link(tmp_path):
    big = b"A" * 100 + b"x" * (LOG_CAP_BYTES + 10) + b"Z" * 100
    artifact, info = store_provenance(tmp_path, "d", ProvenanceFile("log", big, "o"))
    assert info.truncated and info.tier == 3 and info.full_bytes == len(big)
    assert info.full_sha256 != artifact.sha256
    kept = read_provenance(tmp_path, artifact)
    assert kept.startswith(b"AAAA") and kept.endswith(b"ZZZZ") and len(kept) < len(big)
    small, small_info = store_provenance(tmp_path, "d", ProvenanceFile("log", b"hello", "o", name="s"))
    assert not small_info.truncated and gzip.decompress((tmp_path / small.path).read_bytes()) == b"hello"
    with pytest.raises(ValueError, match="never truncated"):
        store_provenance(tmp_path, "d", ProvenanceFile("raw-output", b"x" * 11, "o", cap_bytes=10))
    assert b"hf_" not in redact(b"token=hf_" + b"a" * 30)
    body = _dev_body().model_copy(update={"provenance_artifacts": [info]})
    assert public_release_problems(body) == [f"unredacted tier-3 artifact: {info.artifact_id}"]


def test_provenance_manifest_mismatch_detected(tmp_path):
    cfg = _dev_config(log=b"log")
    path = write_configuration(tmp_path, cfg)
    ds = load_e2e(tmp_path, path)
    assert ds.observations[0].log_artifact_id in {p.artifact_id for p in ds.body.provenance_artifacts}


# ---------------------------------------------------------------- snapshot


def test_snapshot_round_trip_and_tamper(tmp_path):
    members = [
        SnapshotMember(
            configuration_id="cfg-" + "a" * 64, observation_id="o2", record_sha256="b" * 64, manifest_path="m"
        ),
        SnapshotMember(
            configuration_id="cfg-" + "a" * 64, observation_id="o1", record_sha256="c" * 64, manifest_path="m"
        ),
    ]
    snap = write_snapshot(
        tmp_path,
        "snap.json",
        members,
        source_scope={"kind": "api-current-model-view"},
        selection_policy="p",
        completeness="complete",
        acquisition=[SnapshotAcquisition(observation_id="o1", source_uri="https://x")],
    )
    got, rows = read_snapshot(tmp_path, "snap.json")
    assert got == snap and [r.observation_id for r in rows] == ["o1", "o2"]
    data = (tmp_path / "snap.json").read_text().replace('"complete"', '"partial"')
    (tmp_path / "snap.json").write_text(data)
    assert read_snapshot(tmp_path, "snap.json")[0].completeness == "partial"
    (tmp_path / "snap.json").write_text(data.replace(snap.snapshot_id, "0" * 64))
    with pytest.raises(ValueError, match="snapshot ID"):
        read_snapshot(tmp_path, "snap.json")


def test_loader_refuses_foreign_table_schema(tmp_path):
    from aisim_core.data.loaders import E2eLoader

    path = write_configuration(tmp_path, _dev_config())
    with pytest.raises(LoaderError, match="unsupported table schema"):
        E2eLoader(accepted_schemas=("aisim-e2e-requests/v1",)).load(tmp_path, path)
