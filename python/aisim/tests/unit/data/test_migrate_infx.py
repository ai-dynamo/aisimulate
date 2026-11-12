# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-time migration of the private InferenceX cache to aisim-e2e/v1 (synthetic cache)."""

import gzip
import hashlib
import json
import shutil
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisim_core.data.e2e import inferencex_to_configurations
from aisim_core.data.migrate_infx import (
    MigrationError,
    canonical,
    migrate_infx_cache,
    read_infx_cache,
    verify_migrated_dataset,
)
from aisim_core.data.migrate_infx_parity import EXPLAINED, compare_cache_to_dataset

pytestmark = pytest.mark.unit

RUN = "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/{n}/attempts/1"
RUNS = {
    RUN.format(n=n): {
        "github_run_id": str(n),
        "run_attempt": 1,
        "html_url": RUN.format(n=n),
        "head_sha": "a" * 40,
        "head_branch": "main",
        "run_started_at": "2026-10-01T00:00:00Z",
        "status": "completed",
        "conclusion": "success",
    }
    for n in (1, 2, 3, 4)
}


def _record(rid, **over):
    rec = dict(
        id=str(rid),
        model="dsr1",
        hardware="b200",
        framework="sglang",
        precision="fp8",
        spec_method="none",
        disagg=False,
        is_multinode=False,
        prefill_tp=8,
        prefill_ep=1,
        prefill_dp_attention=False,
        prefill_num_workers=0,
        decode_tp=8,
        decode_ep=1,
        decode_dp_attention=False,
        decode_num_workers=0,
        num_prefill_gpu=8,
        num_decode_gpu=8,
        offload_mode="off",
        workers=None,
        benchmark_type="single_turn",
        isl=1024,
        osl=128,
        conc=4,
        date="2026-10-01",
        image="img:1",
        recipe_fingerprint=None,
        run_url=RUN.format(n=1),
        metrics={
            "mean_ttft": 0.1,
            "median_ttft": 0.09,
            "p99.9_ttft": 0.4,
            "mean_tpot": 0.01,
            "mean_e2el": 1.5,
            "mean_intvty": 100.0,
            "tput_per_gpu": 1200.0,
            "output_tput_per_gpu": 150.0,
            "avg_power_w": 500.0,
            "pp": 1,
            "dcp_size": 1,
        },
    )
    rec.update(over)
    return rec


def _cache_metric_rows(raw):
    """Rows as the cache normalizer wrote them (no interactivity, no unmapped keys)."""
    rows = []
    names = {"ttft": "ttft", "tpot": "tpot", "itl": "itl", "e2el": "e2e_latency"}
    for key, value in raw["metrics"].items():
        stat, _, metric = key.partition("_")
        if metric in names:
            percentile = None
            if stat.startswith("p"):
                percentile, stat = float(stat[1:]), "percentile"
            rows.append(
                dict(
                    observation_id="inferencex:" + raw["id"],
                    record_sha256="x",
                    name=names[metric],
                    statistic=stat,
                    percentile=percentile,
                    value=value * 1000.0,
                    unit="ms",
                    normalization="none",
                    timing_boundary="unknown",
                    status="valid",
                    reason="Source API does not declare timing boundary",
                    source_metric=key,
                )
            )
        elif key in ("tput_per_gpu", "output_tput_per_gpu"):
            rows.append(
                dict(
                    observation_id="inferencex:" + raw["id"],
                    record_sha256="x",
                    name="total_throughput" if key == "tput_per_gpu" else "output_throughput",
                    statistic="rate",
                    percentile=None,
                    value=value,
                    unit="tokens/s",
                    normalization="per_device",
                    timing_boundary="unknown",
                    status="valid",
                    reason="Source API does not declare timing boundary",
                    source_metric=key,
                )
            )
    return rows


def _write_parquet(path, rows, schema_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows).replace_schema_metadata({b"schema_id": schema_id.encode()})
    pq.write_table(table, path, compression="zstd")
    data = path.read_bytes()
    return {
        "bytes": len(data),
        "path": None,
        "rows": len(rows),
        "schema": schema_id,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _make_cache(root, records):
    partitions = []
    by_part = {}
    for r in records:
        by_part.setdefault((r["model"], r["hardware"], r["framework"]), []).append(r)
    for (model, hw, fw), recs in sorted(by_part.items()):
        base = f"data/e2e/inferencex/{model}-x/{hw}-x/{fw}-x"
        obs_rows = [
            dict(
                observation_id="inferencex:" + r["id"],
                record_sha256=hashlib.sha256(canonical(r)).hexdigest(),
                source_kind="inferencex",
                source_benchmark_id=r["id"],
                source_model=model,
                display_model="Display",
                hardware=hw,
                framework=fw,
                measurement_date=r["date"],
                producer_run_url=r["run_url"],
                source_config_id=None,
                raw_record_json=json.dumps(r, sort_keys=True),
            )
            for r in recs
        ]
        met_rows = [m for r in recs for m in _cache_metric_rows(r)]
        obs = _write_parquet(root / base / "observations-a.parquet", obs_rows, "inferencex-observations-v1")
        met = _write_parquet(root / base / "metrics-a.parquet", met_rows, "inferencex-metrics-v1")
        obs["path"], met["path"] = base + "/observations-a.parquet", base + "/metrics-a.parquet"
        partitions.append(
            {
                "artifacts": [obs, met],
                "display_model": "Display",
                "framework": fw,
                "hardware": hw,
                "source_model": model,
            }
        )
    body = gzip.compress(b'{"data":[]}')
    resp_path = "data/e2e/inferencex/responses/" + hashlib.sha256(body).hexdigest() + ".json.gz"
    (root / resp_path).parent.mkdir(parents=True, exist_ok=True)
    (root / resp_path).write_bytes(body)
    snapshot = {
        "acquisitions": [
            {
                "artifact": {
                    "bytes": len(body),
                    "path": resp_path,
                    "schema": "inferencex-api-json/v1",
                    "sha256": hashlib.sha256(body).hexdigest(),
                },
                "body_sha256": "b" * 64,
                "model": "Display",
                "retrieved_at": "2026-10-19T17:25:30+00:00",
                "url": "https://inferencex.example/api/v1/benchmarks?model=Display",
            }
        ],
        "completeness": "complete",
        "counts": {"observations_distinct": len(records)},
        "model_mapping": [],
        "partitions": partitions,
        "publication": {"redistribution_status": "unresolved"},
        "qualification": "archival-only",
        "schema_version": "inferencex-input-snapshot/v1",
        "selection_policy": "archive-all-returned-records-v1",
        "snapshot_id": "c" * 64,
        "source_scope": {"kind": "api-current-model-view", "models": ["Display"]},
    }
    snap_path = "catalog/e2e/snapshots/" + "c" * 64 + ".json"
    (root / snap_path).parent.mkdir(parents=True, exist_ok=True)
    snap_bytes = json.dumps(snapshot, sort_keys=True).encode()
    (root / snap_path).write_bytes(snap_bytes)
    index = {
        "partitions": partitions,
        "schema_version": "e2e-source-index/v1",
        "snapshot": {
            "bytes": len(snap_bytes),
            "path": snap_path,
            "schema": "inferencex-input-snapshot/v1",
            "sha256": hashlib.sha256(snap_bytes).hexdigest(),
        },
        "snapshot_id": "c" * 64,
        "source_kind": "inferencex",
    }
    (root / "catalog/e2e/inferencex.json").write_text(json.dumps(index, sort_keys=True))


def _records():
    return [
        _record(1),
        _record(2, conc=8, metrics={**_record(2)["metrics"], "mean_ttft": 0.2}),
        # Differs from record 1 only in dp-attention: a different deployment, so a different identity.
        _record(3, decode_dp_attention=True, run_url=RUN.format(n=2)),
        # Differs only in GPU telemetry: same configuration, kept as a separate source configuration.
        _record(4, workers=[{"avg_power_w": 1.0, "role": "decode"}], run_url=RUN.format(n=3)),
        # pp and dcp are reported under metrics: part of the identity.
        _record(5, metrics={**_record(5)["metrics"], "pp": 2, "dcp_size": 4}, run_url=RUN.format(n=3)),
        # Degenerate source row: a zero degree is unknown, never a degree.
        _record(
            6,
            disagg=True,
            prefill_tp=8,
            prefill_num_workers=1,
            decode_tp=0,
            decode_ep=0,
            num_decode_gpu=0,
            run_url=RUN.format(n=4),
        ),
    ]


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / "cache"
    _make_cache(root, _records())
    return root


@pytest.fixture
def migrated(tmp_path, cache):
    loaded = read_infx_cache(cache)
    result = migrate_infx_cache(loaded, tmp_path / "out", runs=RUNS, source={"revision": "r" * 40})
    return loaded, result


def test_migration_round_trips_through_the_production_loader(migrated):
    cache, result = migrated
    assert (result.observations, result.metric_rows > 0) == (6, True)
    check = verify_migrated_dataset(result.output)
    assert check.problems == []
    assert (check.configurations, check.observations) == (result.configurations, 6)
    assert check.snapshot_id == result.snapshot_id != result.legacy_snapshot_id


def test_identity_keeps_dp_attention_and_pp_dcp_but_not_telemetry(migrated):
    _, result = migrated
    cids = {o: p.split("/")[-2] for o, p in result.observation_manifest.items()}
    assert cids["inferencex:1"] == cids["inferencex:2"] == cids["inferencex:4"]  # telemetry only
    assert len({cids["inferencex:1"], cids["inferencex:3"], cids["inferencex:5"], cids["inferencex:6"]}) == 4


def test_deployment_facts_are_first_class_identity_fields(migrated):
    _, result = migrated
    from aisim_core.data.migrate_infx_parity import load_dataset

    datasets, _ = load_dataset(result.output)
    for ds in datasets:
        identity = ds.manifest.identity
        assert (identity.dp_attention, identity.multinode, identity.offload_mode) == (
            ds.body.dp_attention,
            ds.body.multinode,
            ds.body.offload_mode,
        )
        assert identity.multinode is False and identity.offload_mode == "off"
        assert set(ds.body.identity_settings) == {"spec_method", "workers"}
        assert ds.manifest.configuration_id.startswith("cfg-")
    assert {tuple(sorted(d.body.dp_attention.items())) for d in datasets} == {
        (("decode", False), ("prefill", False)),
        (("decode", True), ("prefill", False)),
    }


def test_raw_records_are_bundled_per_configuration_with_row_level_audit(migrated):
    from aisim_core.data.e2e import read_raw_records
    from aisim_core.data.migrate_infx_parity import load_dataset

    cache, result = migrated
    datasets, _ = load_dataset(result.output)
    raw_of = {"inferencex:" + r["id"]: r for r in cache.records}
    for ds in datasets:
        bundles = [a for a in ds.manifest.artifacts if a.role == "raw-output-bundle"]
        assert len(bundles) == 1  # one raw artifact file per configuration, not per observation
        assert not [a for a in ds.manifest.artifacts if a.role == "raw-output"]
        records = read_raw_records(result.output, ds.manifest, ds.observations)
        for obs in ds.observations:
            assert obs.raw_artifact_id == bundles[0].artifact_id
            assert records[obs.observation_id] == canonical(raw_of[obs.observation_id])
            assert hashlib.sha256(records[obs.observation_id]).hexdigest() == obs.record_sha256
    assert result.files < 3 * len(raw_of) + 10 * len(datasets)


def test_bundle_row_audit_detects_a_changed_row_range(migrated):
    from aisim_core.data.e2e import read_raw_records
    from aisim_core.data.loaders import LoaderError
    from aisim_core.data.migrate_infx_parity import load_dataset

    _, result = migrated
    datasets, _ = load_dataset(result.output)
    ds = next(d for d in datasets if len(d.observations) > 1)
    first, second = ds.observations[:2]
    swapped = [first.model_copy(update={"raw_offset": second.raw_offset, "raw_length": second.raw_length})]
    with pytest.raises(LoaderError, match="hash mismatch"):
        read_raw_records(result.output, ds.manifest, swapped)
    outside = [first.model_copy(update={"raw_offset": 10**9})]
    with pytest.raises(LoaderError, match="invalid raw record range"):
        read_raw_records(result.output, ds.manifest, outside)


def test_typed_columns_carry_fields_without_a_canonical_home(migrated):
    cache, result = migrated
    from aisim_core.data.migrate_infx_parity import load_dataset

    datasets, _ = load_dataset(result.output)
    obs = {o.observation_id: o for ds in datasets for o in ds.observations}
    five = obs["inferencex:5"]
    # pp / dcp / power are verbatim source metrics without a metric row.
    assert json.loads(five.unmapped_metrics_json) == {"avg_power_w": 500.0, "dcp_size": 4, "pp": 2}
    assert json.loads(obs["inferencex:4"].source_config_json)["workers"] == [{"avg_power_w": 1.0, "role": "decode"}]
    assert json.loads(obs["inferencex:1"].source_config_json)["workers"] is None
    zero = json.loads(obs["inferencex:6"].source_config_json)
    assert (zero["decode_tp"], zero["decode_ep"]) == (0, 0)  # the source value survives the "unknown" identity


def test_zero_degree_is_unknown_not_a_degree():
    [data] = inferencex_to_configurations([_records()[5]])
    assert data.body.parallelism == {"decode": {}, "prefill": {"tp": 8}}
    assert {"field": "parallelism.decode.tp", "reason": "source reported 0"} in data.body.evidence_gaps


def test_metric_parallelism_and_p99_9_are_mapped():
    [data] = inferencex_to_configurations([_records()[4]], RUNS)
    assert data.body.parallelism == {"aggregated": {"dcp": 4, "pp": 2, "tp": 8}}
    stats = {(m.metric, m.statistic) for m in data.metrics if m.status == "valid"}
    assert ("ttft", "p99.9") in stats
    assert data.body.source_fields["unmapped_metrics"] == ["avg_power_w"]  # pp/dcp are not "unmapped"


def test_producer_run_metadata_is_kept_for_freshness_decisions(migrated):
    cache, result = migrated
    from aisim_core.data.migrate_infx_parity import load_dataset

    datasets, _ = load_dataset(result.output)
    runs = {}
    for ds in datasets:
        runs.update(ds.body.source_fields.get("producer_runs", {}))
    assert runs[RUN.format(n=1)]["run_started_at"] == "2026-10-01T00:00:00Z"


def test_parity_report_has_no_unexplained_difference(migrated):
    cache, result = migrated
    report = compare_cache_to_dataset(cache, result.output)
    assert report["unexplained"] == {}
    assert report["summary"]["max_relative_value_difference"] == 0.0
    # Merged telemetry configuration and split pp/dcp configuration are both reported and explained.
    assert report["identities"]["identity.configuration_merge_telemetry"]["status"] == "explained"
    assert set(report["metrics"]) <= set(EXPLAINED)


def test_parity_report_flags_a_changed_metric(migrated):
    cache, result = migrated
    # Corrupt the cache side: a changed cached value must surface as an unexplained mismatch.
    tampered = [dict(row) for row in cache.metric_rows]
    tampered[0]["value"] += 1.0
    report = compare_cache_to_dataset(replace(cache, metric_rows=tampered), result.output)
    assert "metrics:metric.value_mismatch" in report["unexplained"]


def test_verified_reader_rejects_tampering(tmp_path, cache):
    target = next((cache / "data/e2e/inferencex").glob("*/*/*/observations-a.parquet"))
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(MigrationError, match="checksum"):
        read_infx_cache(cache)


def test_reader_rejects_incomplete_snapshot(tmp_path, cache):
    index_path = cache / "catalog/e2e/inferencex.json"
    index = json.loads(index_path.read_text())
    index["snapshot_id"] = "d" * 64
    index_path.write_text(json.dumps(index))
    with pytest.raises(MigrationError, match="inconsistent"):
        read_infx_cache(cache)


def test_migration_refuses_a_non_empty_output(tmp_path, cache):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep").write_text("x")
    with pytest.raises(MigrationError, match="empty"):
        migrate_infx_cache(read_infx_cache(cache), out)
    shutil.rmtree(out)
