# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentX arithmetic, complete accounting, and public publication boundaries."""

import hashlib
import io
import json
import posixpath
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.e2e_accuracy import agentx_predict
from scripts.e2e_accuracy.agentx_contract import (
    SUMMARY_SCHEMA,
    aggregate,
    canonical,
    cohort,
    driver_digest,
    native_completion_qualified,
    score,
    validate_result,
)
from scripts.e2e_accuracy.run_agentx_accuracy import artifact, normalize_native
from scripts.pages.agentx_pages import unpack, validate_bundle


def observation(key="a", **values):
    return {
        "key": key,
        "phase": "profile",
        "status": "completed",
        "start_ms": 0,
        "end_ms": 1000,
        "input_length": 10,
        "output_length": 100,
        "ttft_ms": 100,
        "tpot_ms": 10,
        **values,
    }


def result(**changes):
    value = {
        "id": "agentx-446063",
        "category": "agentx",
        "model": "nvidia/GLM-5.3-NVFP4",
        "hardware": "B300",
        "backend": "sglang",
        "measured_backend_version": "0.5.20",
        "precision": "NVFP4",
        "tensor_parallel": 8,
        "lanes": 1,
        "initial_input_tokens": None,
        "cache_state": "recorded",
        "repeat": 0,
        "benchmark_url": None,
        "source_run_url": None,
        "boundary": "successful_profile_span",
        "measured_requests": 1,
        "outcome": "completed",
        "reason": None,
        "prediction_backend_version": "0.5.21",
        "cache_alignment": None,
        "config_summary": [],
        "report": None,
        **score(
            [observation()],
            [observation(ttft_ms=120, tpot_ms=15)],
            "successful_profile_span",
        ),
    }
    value.update(changes)
    return value


def simulation_record(identity="agentx-446063", revision="a" * 40, dataset="b" * 40):
    from scripts.e2e_accuracy.simulation_features import unavailable

    record = unavailable(
        {"benchmark_id": identity, "evaluated_revision": revision, "dataset_revision": dataset}, result(), "disabled"
    )
    record.update(execution_mode="offline", configuration_sha256="c" * 64)
    record["roles"][0].update(backend_version="0.5.21", compiled_sha256="d" * 64)
    return record


def bundle(run=None):
    run = result() if run is None else run
    snapshot = {
        "branch": "main",
        "commit_sha": "a" * 40,
        "dataset_revision": "b" * 40,
        "wheel_sha256": "c" * 64,
        "driver_sha256": "e" * 64,
        "cohort_sha256": "d" * 64,
        "run_id": "123",
        "run_attempt": 1,
        "preview": False,
        "completed_at": "2026-11-02T00:00:00+00:00",
    }
    summary_run = {k: v for k, v in run.items() if k not in {"pairs", "report"}}
    summary_run["detail_path"] = f"details/{run['id']}.json"
    files = {
        "summary.json": canonical({"schema": SUMMARY_SCHEMA, "snapshot": snapshot, "runs": [summary_run]}),
        summary_run["detail_path"]: canonical({"schema": SUMMARY_SCHEMA, "snapshot": snapshot, **run}),
    }
    seal(files, snapshot)
    return files


def seal(files, snapshot=None):
    snapshot = snapshot or json.loads(files["summary.json"])["snapshot"]
    files["qualification.json"] = canonical(
        {
            "schema": "aisimulate.agentx.qualification.v1",
            "status": "complete",
            "snapshot": snapshot,
            "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items() if k != "qualification.json"},
        }
    )


def test_macro_points_micro_requests_and_no_error_of_means():
    a = score([observation()], [observation(ttft_ms=200)], "successful_profile_span")
    b = score(
        [observation(str(i), ttft_ms=1000) for i in range(3)],
        [observation(str(i), ttft_ms=1000) for i in range(3)],
        "successful_profile_span",
    )
    combined = aggregate([a, b])
    assert combined["point_errors"]["ttft_ms"] == 50
    assert combined["request_errors"]["ttft_ms"] == {
        "mape": 25,
        "count": 4,
        "sum_ape": 100,
    }
    assert combined["coverage"]["measured"] == 4


@pytest.mark.parametrize("backend,version", [("sglang", "0.5.21"), ("vllm", "0.31.0")])
def test_reviewed_database_pin_overrides_current_slot_and_never_borrows_data(tmp_path, monkeypatch, backend, version):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "engine": {
                    "backend": backend,
                    "backend_version": "measured",
                    "hardware": "B300",
                },
                "prediction_database_version": version,
            }
        )
    )
    captured = []

    def parsed(value):
        captured.append(value)
        return value

    def unsupported(spec):
        raise NotImplementedError("Older wheel has no replay capability")

    api = SimpleNamespace(
        ReplayOutputRequirements=lambda **kwargs: kwargs,
        CorePredictionConfig=SimpleNamespace(model_validate=parsed),
        EngineReplayRunnerFactory=lambda: SimpleNamespace(
            capabilities=lambda: SimpleNamespace(require_compatible=unsupported)
        ),
    )
    modules = {
        "aisim": api,
        "aisim.compiler": SimpleNamespace(prediction_to_replay_spec=lambda value: value),
        # No get_latest_database_version: the reviewed pin must take precedence.
        "aisim.sdk.perf_database": SimpleNamespace(),
        "aisim.replay.dashboard": SimpleNamespace(),
    }
    monkeypatch.setattr(
        agentx_predict,
        "importlib",
        SimpleNamespace(
            import_module=modules.__getitem__,
            resources=SimpleNamespace(files=lambda package: tmp_path),
        ),
    )
    missing = agentx_predict.predict(config, tmp_path)
    assert missing["reason"] == "database_unavailable"
    assert missing["prediction_backend_version"] == version
    assert not captured

    def unavailable_resources(_):
        raise ModuleNotFoundError("Older wheel has no core resources")

    monkeypatch.setattr(agentx_predict.importlib.resources, "files", unavailable_resources)
    assert agentx_predict.predict(config, tmp_path) == missing
    monkeypatch.setattr(agentx_predict.importlib.resources, "files", lambda _: tmp_path)
    relative = (
        f"sparse_attention/sglang/{version}/dsa_generation_module_perf.parquet"
        if backend == "sglang"
        else f"attention/vllm/{version}/context_attention_perf.parquet"
    )
    data = tmp_path / "systems/data/b300_sxm" / relative
    data.parent.mkdir(parents=True)
    data.write_bytes(b"fixture data")
    result = agentx_predict.predict(config, tmp_path)
    assert result["reason"] == "replay_capability_unavailable"
    assert captured[0]["engine"]["backend_version"] == version
    assert "prediction_database_version" not in captured[0]


@pytest.mark.parametrize("missing", ["utils", "common", "lookup"])
def test_optional_wheel_feature_capabilities_report_unsupported(tmp_path, monkeypatch, missing):
    parsed = SimpleNamespace(engine=SimpleNamespace(model="test/model"), model_dump=lambda **_: {})
    api = SimpleNamespace(
        ReplayOutputRequirements=lambda **_: {},
        CorePredictionConfig=SimpleNamespace(model_validate=lambda _: parsed),
        EngineReplayRunnerFactory=lambda: SimpleNamespace(
            capabilities=lambda: SimpleNamespace(require_compatible=lambda _: None)
        ),
    )
    modules = {
        "aisim": api,
        "aisim.compiler": SimpleNamespace(prediction_to_replay_spec=lambda _: {}),
        "aisim.sdk.perf_database": SimpleNamespace(),
        "aisim.replay.dashboard": SimpleNamespace(),
        "aisim_core.sdk.utils": SimpleNamespace()
        if missing == "lookup"
        else SimpleNamespace(get_model_config_from_model_path=lambda _: {}),
        "aisim_core.sdk.common": SimpleNamespace(),
    }

    def import_module(name):
        if name == "aisim_core.sdk." + missing:
            raise ModuleNotFoundError(name)
        return modules[name]

    monkeypatch.setattr(
        agentx_predict,
        "importlib",
        SimpleNamespace(import_module=import_module, resources=SimpleNamespace(files=lambda _: tmp_path)),
    )
    monkeypatch.setattr(agentx_predict, "asdict", lambda value: value)
    data = tmp_path / "systems/data/b300_sxm/attention/vllm/0.31.0/context_attention_perf.parquet"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"synthetic op-data")
    config = tmp_path / "config.json"
    config.write_bytes(
        canonical({"engine": {"backend": "vllm", "hardware": "B300"}, "prediction_database_version": "0.31.0"})
    )
    (tmp_path / "feature-context.json").write_text("{}")
    assert agentx_predict.predict(config, tmp_path) == {
        "outcome": "unsupported",
        "reason": "replay_telemetry_unavailable",
        "prediction_backend_version": "0.31.0",
    }


def test_historical_driver_digest_preserves_the_original_file_closure():
    names = (
        "agentx_contract.py",
        "agentx_predict.py",
        "run_agentx_accuracy.py",
        "run_e2e_accuracy.py",
    )
    files = {"scripts/e2e_accuracy/" + name: name.encode() for name in names}
    expected = hashlib.sha256(
        canonical({name: hashlib.sha256(name.encode()).hexdigest() for name in names})
    ).hexdigest()
    assert driver_digest(files.__getitem__) == expected


@pytest.mark.parametrize("historical", [False, True])
def test_portable_driver_digest_tracks_cache_validation_location(historical):
    names = (
        "agentx_contract.py",
        "agentx_predict.py",
        "run_agentx_accuracy.py",
        "run_e2e_accuracy.py",
        "agentx_inputs.py",
    )
    files = {"scripts/e2e_accuracy/" + name: name.encode() for name in names}
    files["scripts/e2e_accuracy/agentx_contract.py"] = b'"agentx_inputs.py"'
    if historical:
        names += ("../../benchmarks/evidence/accuracy/qwen-weka/cache_probe.py",)
        files["scripts/e2e_accuracy/agentx_inputs.py"] = b"importlib.util.spec_from_file_location"
        files["benchmarks/evidence/accuracy/qwen-weka/cache_probe.py"] = b"historical geometry"
    expected = hashlib.sha256(
        canonical(
            {
                name: hashlib.sha256(files[posixpath.normpath("scripts/e2e_accuracy/" + name)]).hexdigest()
                for name in names
            }
        )
    ).hexdigest()
    assert driver_digest(files.__getitem__) == expected


def cutoff_native():
    return {
        "agentic_profile": {
            "profile_start_ms": 0,
            "admission_cutoff_ms": 600_000,
            "response_grace_deadline_ms": 630_000,
            "finished_at_ms": 630_000,
            "cancel_drain_deadline_ms": 640_000,
            "admission_closed": True,
            "cancel_drain_timed_out": False,
            "client_in_flight_requests": 0,
            "issued_requests": 2,
            "successful_responses": 1,
            "canceled_requests": 1,
            "server_unsettled_requests": 1,
            "unsettled_server_requests": 1,
        },
        "telemetry": {
            "completion_status": "server_unsettled",
            "phase_boundaries": [{"phase": "profile", "end_ms": 630_000}],
            "requests": [
                {
                    "phase": "profile",
                    "uuid": "ok",
                    "dispatched_at_ms": 1,
                    "terminal_time_ms": 100,
                    "terminal_status": "completed",
                },
                {
                    "phase": "profile",
                    "uuid": "cancel",
                    "dispatched_at_ms": 590_000,
                    "terminal_time_ms": 630_000,
                    "terminal_status": "canceled",
                },
            ],
            "lifecycle": [
                {"phase": "profile", "request_uuid": "ok", "event": "quiescent"},
                {
                    "phase": "profile",
                    "request_uuid": "cancel",
                    "event": "causal_terminal",
                    "status": "canceled",
                },
            ],
        },
    }


def test_profile_cutoff_accepts_accounted_client_cancellation_without_hiding_server_work():
    native = cutoff_native()
    assert native_completion_qualified(native, "successful_profile_span")
    assert native["telemetry"]["completion_status"] == "server_unsettled"
    assert native["agentic_profile"]["server_unsettled_requests"] == 1
    assert not native_completion_qualified(native, "complete_finite_cohort")
    assert native_completion_qualified({"telemetry": {"completion_status": "completed"}}, "complete_finite_cohort")


@pytest.mark.parametrize(
    "field,value",
    [
        ("admission_closed", False),
        ("cancel_drain_timed_out", True),
        ("client_in_flight_requests", 1),
        ("issued_requests", 3),
        ("canceled_requests", 0),
        ("successful_responses", 2),
        ("server_unsettled_requests", 2),
        ("unsettled_server_requests", 0),
        ("finished_at_ms", 629_999),
        ("finished_at_ms", 640_001),
    ],
)
def test_unsettled_profile_requires_complete_accounting_and_bounded_deadlines(field, value):
    native = cutoff_native()
    native["agentic_profile"][field] = value
    assert not native_completion_qualified(native, "successful_profile_span")


@pytest.mark.parametrize(
    "case",
    [
        "missing_profile",
        "unacknowledged",
        "unsettled_completed",
        "early_cancel",
        "late_admission",
        "failure",
        "timeout",
        "failed_preparation",
    ],
)
def test_unsettled_profile_rejects_unexplained_work_and_incomplete_replays(case):
    native = cutoff_native()
    telemetry = native["telemetry"]
    if case == "missing_profile":
        native.pop("agentic_profile")
    elif case == "unacknowledged":
        telemetry["lifecycle"].pop()
    elif case == "unsettled_completed":
        telemetry["lifecycle"].pop(0)
    elif case == "early_cancel":
        telemetry["requests"][1]["terminal_time_ms"] = 629_999
    elif case == "late_admission":
        telemetry["requests"][1]["dispatched_at_ms"] = 600_000
    elif case == "failure":
        telemetry["requests"][1]["terminal_status"] = "failed"
    elif case == "timeout":
        telemetry["completion_status"] = "time_limited"
    else:
        telemetry["phase_boundaries"].append({"phase": "preparation", "completion_status": "aborted"})
    assert not native_completion_qualified(native, "successful_profile_span")


def test_every_measured_request_accounted_and_metric_denominators():
    actual = [observation(str(i)) for i in range(7)] + [observation("warm", phase="preparation")]
    actual[1]["status"] = "failed"
    actual[6].update(ttft_ms=0, tpot_ms=None)
    predicted = [
        observation("0", input_length=20),
        observation("1"),
        observation("2"),
        observation("2"),
        observation("4", status="failed"),
        observation("5", output_length=99),
        observation("6"),
        observation("extra"),
    ]
    x = score(actual, predicted, "successful_profile_span")
    assert x["coverage"] == {
        "measured": 7,
        "predicted": 8,
        "matched": 2,
        "measurement_failed": 1,
        "ambiguous_prediction": 1,
        "missing_prediction": 1,
        "prediction_failed": 1,
        "output_length_mismatch": 1,
        "unmatched_predictions": 1,
        "input_length_mismatch": 1,
    }
    assert x["request_errors"]["ttft_ms"]["count"] == x["request_errors"]["tpot_ms"]["count"] == 1
    assert len(x["pairs"]) == 7
    assert x["pairs"][6]["ttft_ms_ape"] is None


def test_units_finite_cohort_delays_and_profile_boundary():
    rows = [
        observation("1", start_ms=5000, end_ms=6000),
        observation("2", start_ms=6100, end_ms=7100),
    ]
    assert cohort(rows)["output_tokens_s"] == pytest.approx(200000 / 2100)
    assert cohort(rows[:1], complete=True, expected=2)["output_tokens_s"] is None
    assert cohort([observation(end_ms=0)])["output_tokens_s"] is None
    x = score(
        [observation(), observation("warm", phase="preparation", ttft_ms=9000)],
        [],
        "complete_finite_cohort",
    )
    assert x["silicon"]["ttft_ms"] == 100
    assert all(v is None for v in x["point_errors"].values())
    assert x["request_errors"]["ttft_ms"]["count"] == 0
    ambiguous = score([observation(), observation()], [observation()], "successful_profile_span")
    assert ambiguous["coverage"]["ambiguous_measurement"] == 2
    assert ambiguous["coverage"]["measured"] == len(ambiguous["pairs"]) == 2
    assert ambiguous["request_errors"]["ttft_ms"]["count"] == 0
    unknown = score([observation(identity_valid=False)], [observation()], "successful_profile_span")
    assert unknown["coverage"]["missing_measurement_identity"] == 1
    assert unknown["request_errors"]["ttft_ms"]["count"] == 0
    excluded = score(
        [observation(status="failed", failure_reason="context_overflow")],
        [],
        "successful_profile_span",
    )
    assert excluded["coverage"]["context_overflow"] == 1
    assert excluded["coverage"]["measured"] == 1


def test_qwen_preparation_relabel_and_native_milliseconds():
    native = {
        "telemetry": {
            "requests": [
                {
                    "request_id": "warm",
                    "phase": "profile",
                    "terminal_status": "completed",
                    "arrival_time_ms": 100,
                    "terminal_time_ms": 400,
                    "input_length": 10,
                    "output_length": 1,
                    "ttft_ms": 300,
                    "itl_ms": 0,
                }
            ]
        }
    }
    rows = normalize_native(native, [observation("warm", phase="preparation")], "self_collected")
    assert rows[0]["phase"] == "preparation"
    assert rows[0]["ttft_ms"] == 300
    assert rows[0]["tpot_ms"] is None


def test_valid_public_bundle_reconciles():
    validate_result(result())
    assert validate_bundle(bundle())["runs"][0]["coverage"]["measured"] == 1


@pytest.mark.parametrize(
    "mutation",
    ["missing", "mixed", "private", "denominator", "coverage", "path", "checksum"],
)
def test_publication_rejects_invalid_bundles(mutation):
    files = bundle()
    name = "details/agentx-446063.json"
    detail = json.loads(files[name])
    if mutation == "missing":
        del files[name]
    elif mutation == "path":
        files["../secret.json"] = b"{}"
    else:
        if mutation == "mixed":
            detail["snapshot"]["dataset_revision"] = "e" * 40
        if mutation == "private":
            detail["report"] = {"prompt": "private words"}
        if mutation == "denominator":
            detail["pairs"][0]["ttft_ms_ape"] = 0
        if mutation == "coverage":
            detail["pairs"] = []
        if mutation == "checksum":
            detail["reason"] = "changed"
        files[name] = canonical(detail)
    if mutation != "checksum":
        seal(files)
    with pytest.raises((ValueError, KeyError)):
        validate_bundle(files)


def test_private_fields_cannot_hide_in_numerical_payload():
    r = result()
    r["pairs"][0]["silicon_ttft_ms"] = {"value": "private prompt"}
    with pytest.raises(ValueError):
        validate_result(r)
    r = result(config_summary=[{"name": "Cache", "value": [1, 2, 3]}])
    with pytest.raises(ValueError):
        validate_result(r)


def test_archive_rejects_duplicate_paths_and_symlink():
    for symlink in (False, True):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            if symlink:
                info = zipfile.ZipInfo("summary.json")
                info.external_attr = 0o120777 << 16
                archive.writestr(info, "target")
            else:
                archive.writestr("summary.json", "{}")
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr("summary.json", "{}")
        with pytest.raises(ValueError):
            unpack(stream.getvalue())


def test_checksum_failure_before_replay(tmp_path):
    (tmp_path / "data.json").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        artifact(
            tmp_path,
            {"artifacts": {"trace": {"path": "data.json", "sha256": "a" * 64, "bytes": 7}}},
            "trace",
        )


def silicon_inputs(tmp_path, run, features=None):
    from scripts.e2e_accuracy.agentx_contract import IDENTITY

    source = {k: run[k] for k in IDENTITY}
    source.update(
        schema="aisimulate.agentx.benchmark.v1",
        configuration_id="cfg-" + "a" * 64,
        corpus_id=None,
    )
    from scripts.e2e_accuracy.benchmark_features import SILICON_FEATURES

    features = (
        features
        if features is not None
        else {
            "schema": "aisimulate.benchmark.features.v1",
            "benchmark_id": run["id"],
            "taxonomy": None,
            "features": [
                {
                    "id": key,
                    "state": "unknown",
                    "observed": {"state": "unknown", "count": None},
                    "evidence": [],
                    "parameters": (
                        [{"name": "scope", "value": "unknown"}]
                        if key in {"cache.g2", "cache.g3"}
                        else [
                            {"name": "method", "value": "unknown"},
                            {"name": "draft_length", "value": None},
                        ]
                        if key == "speculation"
                        else []
                    ),
                }
                for key in sorted(SILICON_FEATURES)
            ],
        }
    )
    category = "official" if run["category"] == "agentx" else "self-collected"
    base = (
        tmp_path
        / f"data/e2e/agentx/{category}/model/B300/sglang/version/tp8/{source['configuration_id']}/active/{run['id']}"
    )
    base.mkdir(parents=True)
    source["artifacts"] = {}
    for role, value in {
        "feature_receipt": {"settings": {"synthetic": True}},
        "feature_metadata": features,
        "configuration": {
            "schema": "aisimulate.agentx.silicon-config.v1",
            "identity": {
                k: run[k]
                for k in (
                    "model",
                    "hardware",
                    "backend",
                    "measured_backend_version",
                    "precision",
                    "tensor_parallel",
                )
            },
        },
        "workload": {
            "schema": "aisimulate.agentx.workload.v1",
            "benchmark_id": run["id"],
            "corpus_id": None,
        },
        "observations": [observation()],
    }.items():
        content = canonical(value)
        (base / (role + ".json")).write_bytes(content)
        source["artifacts"][role] = {
            "path": role + ".json",
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    content = canonical(source)
    path = base / "manifest.json"
    path.write_bytes(content)
    entry = {k: source[k] for k in ("id", "category", "configuration_id")}
    entry["manifest"] = {
        "path": path.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }
    catalog = {"schema": "aisimulate.agentx.inputs.v1", "runs": [entry]}
    catalog["snapshot_id"] = hashlib.sha256(canonical(catalog)).hexdigest()
    catalog_path = tmp_path / "catalog/e2e/agentx.json"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_bytes(canonical(catalog))
    return catalog


def test_qualification_requires_exact_catalog_identities(tmp_path):
    from argparse import Namespace

    from scripts.e2e_accuracy.agentx_contract import (
        driver_digest,
    )
    from scripts.e2e_accuracy.run_agentx_accuracy import qualify

    run = result()
    run["simulation_features"] = simulation_record()
    catalog = silicon_inputs(tmp_path, run)
    snapshot = json.loads(bundle()["summary.json"])["snapshot"]
    del snapshot["completed_at"]
    snapshot.update(cohort_sha256=catalog["snapshot_id"], driver_sha256=driver_digest())
    args = Namespace(inputs=tmp_path, shards=tmp_path, shard_count=1, output=tmp_path / "public")
    for selected in ([], [run, run], [{**run, "model": "wrong/model"}]):
        (tmp_path / "shard-0.json").write_bytes(canonical({"snapshot": snapshot, "runs": selected}))
        with pytest.raises(ValueError):
            qualify(args)
    (tmp_path / "shard-0.json").write_bytes(canonical({"snapshot": snapshot, "runs": [run]}))
    qualify(args)
    files = {p.relative_to(args.output).as_posix(): p.read_bytes() for p in args.output.rglob("*.json")}
    assert validate_bundle(files)["runs"][0]["id"] == run["id"]


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "missing_metadata",
        "unknown",
        "na_without_evidence",
        "broad_inventory",
        "sub_block",
        "hybrid",
        "host",
        "storage",
        "scope",
        "speculation_method",
        "speculation_length",
        "disabled_speculation",
        "unretained",
        "pointer",
        "result_drift",
    ],
)
def test_new_self_collected_publication_requires_retained_complete_receipts(tmp_path, problem):
    from argparse import Namespace

    from test_benchmark_features import metadata, silicon_metadata

    from scripts.e2e_accuracy.run_agentx_accuracy import qualify

    identity = "fresh-silicon"
    features = silicon_metadata(identity)
    indexed = {row["id"]: row for row in features["features"]}
    if problem == "unknown":
        for row in features["features"]:
            row.update(state="unknown", evidence=[])
        indexed["speculation"]["parameters"][0]["value"] = "unknown"
    elif problem == "na_without_evidence":
        for row in features["features"]:
            row.update(state="not_applicable", evidence=[])
    elif problem == "broad_inventory":
        features = metadata()
        features["benchmark_id"] = identity
    elif problem in {"sub_block", "hybrid", "host", "storage"}:
        key = {
            "sub_block": "cache.sub_block_prefix_reuse",
            "hybrid": "cache.hybrid_state",
            "host": "cache.g2",
            "storage": "cache.g3",
        }[problem]
        indexed[key]["state"] = "enabled"
        if problem in {"host", "storage"}:
            indexed[key]["parameters"] = [{"name": "scope", "value": "local"}]
    elif problem == "scope":
        indexed["cache.g2"]["parameters"][0]["value"] = "unknown"
    elif problem in {"speculation_method", "speculation_length"}:
        indexed["speculation"].update(
            state="enabled",
            parameters=[
                {"name": "method", "value": "unknown" if problem == "speculation_method" else "mtp"},
                {"name": "draft_length", "value": None if problem == "speculation_length" else 3},
            ],
        )
    elif problem == "disabled_speculation":
        indexed["speculation"]["parameters"] = [
            {"name": "method", "value": "mtp"},
            {"name": "draft_length", "value": 3},
        ]
    elif problem in {"unretained", "pointer"}:
        indexed["cache.g1_prefix"]["evidence"][0].update(
            {"sha256": "f" * 64} if problem == "unretained" else {"pointer": "/missing"}
        )
    run = result(
        id=identity,
        category="self_collected",
        feature_metadata=features,
        simulation_features=simulation_record(identity),
    )
    if problem == "missing_metadata":
        del run["feature_metadata"]
    catalog = silicon_inputs(tmp_path, run, features)
    if problem == "result_drift":
        run["feature_metadata"] = json.loads(json.dumps(features))
        run["feature_metadata"]["features"][0]["state"] = "not_applicable"
    snapshot = json.loads(bundle()["summary.json"])["snapshot"]
    snapshot.update(cohort_sha256=catalog["snapshot_id"], driver_sha256=driver_digest())
    args = Namespace(inputs=tmp_path, shards=tmp_path, shard_count=1, output=tmp_path / "public")
    (tmp_path / "shard-0.json").write_bytes(canonical({"snapshot": snapshot, "runs": [run]}))
    if problem is not None:
        with pytest.raises(ValueError):
            qualify(args)
        # A checksum-valid publication cannot bypass the public receipt contract.
        if problem not in {"unretained", "pointer", "result_drift"}:
            with pytest.raises(ValueError):
                validate_bundle(bundle(run))
    else:
        qualify(args)
        files = {p.relative_to(args.output).as_posix(): p.read_bytes() for p in args.output.rglob("*.json")}
        published = validate_bundle(files)["runs"][0]["feature_metadata"]
        assert published == features
        assert all(row["observed"] == {"state": "unknown", "count": None} for row in published["features"])


def test_publisher_requires_successful_trusted_job_and_exact_driver(tmp_path, monkeypatch):
    from scripts.e2e_accuracy.agentx_contract import driver_digest
    from scripts.pages import agentx_pages as publisher

    files = bundle()
    for name in ("summary.json", "details/agentx-446063.json"):
        value = json.loads(files[name])
        value["snapshot"]["driver_sha256"] = driver_digest()
        files[name] = canonical(value)
    seal(files)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    run = {
        "id": 123,
        "run_attempt": 1,
        "event": "schedule",
        "head_branch": "main",
        "head_sha": "a" * 40,
        "path": ".github/workflows/e2e-accuracy.yml",
        "conclusion": "success",
        "repository": {"full_name": "ai-dynamo/aisim"},
        "head_repository": {"full_name": "ai-dynamo/aisim"},
    }
    key = publisher.artifact_key("main")
    job = {
        "name": f"Qualify AgentX accuracy ({key})",
        "status": "completed",
        "conclusion": "success",
        "run_id": 123,
        "head_sha": "a" * 40,
    }

    def api(path, **kwargs):
        if path.endswith("/zip"):
            return archive.getvalue()
        if "workflows/" in path:
            return {"workflow_runs": [run]}
        return run

    def items(path, _):
        if path.endswith("/artifacts"):
            return [{"id": 9, "name": f"e2e-agentx-web-{key}", "expired": False}]
        return [job]

    monkeypatch.setattr(publisher, "api", api)
    monkeypatch.setattr(publisher, "api_items", items)
    monkeypatch.setattr(publisher, "ancestor", lambda *args: True)

    def git(args, **kwargs):
        if "for-each-ref" in args:
            return ""
        return (Path(__file__).resolve().parents[1] / args[-1].split(":", 1)[1]).read_bytes()

    monkeypatch.setattr(publisher.subprocess, "check_output", git)
    publisher.prepare(tmp_path, tmp_path / "accepted")
    assert (tmp_path / "accepted" / key / "summary.json").exists()
    job["conclusion"] = "failure"
    publisher.prepare(tmp_path, tmp_path / "failed")
    assert not (tmp_path / "failed" / key).exists()
    assert json.loads((tmp_path / "failed/updates.json").read_text())["main"]["status"] == "failed"
    job["head_sha"] = "e" * 40
    with pytest.raises(ValueError, match="Untrusted"):
        publisher.prepare(tmp_path, tmp_path / "foreign")
