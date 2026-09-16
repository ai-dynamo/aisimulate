# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_e2e_accuracy_overview.py"
SPEC = importlib.util.spec_from_file_location("build_e2e_accuracy_overview", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
OVERVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OVERVIEW)


def _row(
    *,
    model: str,
    config_id: int,
    concurrency: int,
    silicon_ttft: float,
    silicon_tpot: float,
    aic_ttft: float,
    aic_tpot: float,
    aisimulate_ttft: float | None,
    aisimulate_tpot: float | None,
    status: str,
    tp_size: int = 1,
) -> dict[str, object]:
    return {
        "config_id": config_id,
        "silicon_model": model.lower(),
        "display_name": model,
        "hf_model_path": f"org/{model}",
        "hardware": "h200",
        "framework": "vllm",
        "aic_backend": "vllm",
        "precision": "fp8",
        "spec_method": "none",
        "disagg": False,
        "is_multinode": tp_size > 8,
        "isl": 1024,
        "osl": 1024,
        "conc": concurrency,
        "tp_size": tp_size,
        "pp_size": 1,
        "attention_dp_size": 1,
        "moe_ep_size": None,
        "moe_tp_size": None,
        "silicon_ttft_ms": silicon_ttft,
        "silicon_tpot_ms": silicon_tpot,
        "aic_ttft_ms": aic_ttft,
        "aic_tpot_ms": aic_tpot,
        "dynamo_ttft_ms": aisimulate_ttft,
        "dynamo_tpot_ms": aisimulate_tpot,
        "aisimulate_status": status,
        "aisimulate_runner": "aisimulate.engine_replay",
    }


def _inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    rows = [
        _row(
            model="Alpha",
            config_id=1,
            concurrency=1,
            silicon_ttft=100,
            silicon_tpot=10,
            aic_ttft=110,
            aic_tpot=12,
            aisimulate_ttft=90,
            aisimulate_tpot=11,
            status="success",
        ),
        _row(
            model="Alpha",
            config_id=1,
            concurrency=2,
            silicon_ttft=200,
            silicon_tpot=20,
            aic_ttft=220,
            aic_tpot=18,
            aisimulate_ttft=210,
            aisimulate_tpot=22,
            status="success",
        ),
        _row(
            model="Beta",
            config_id=2,
            concurrency=1,
            silicon_ttft=100,
            silicon_tpot=10,
            aic_ttft=120,
            aic_tpot=11,
            aisimulate_ttft=None,
            aisimulate_tpot=None,
            status="unsupported",
        ),
        _row(
            model="Gamma",
            config_id=3,
            concurrency=1,
            silicon_ttft=100,
            silicon_tpot=10,
            aic_ttft=100,
            aic_tpot=10,
            aisimulate_ttft=None,
            aisimulate_tpot=None,
            status="failed",
            tp_size=16,
        ),
    ]
    predictions = {
        "release_tag": "db-dump/fixture",
        "aic_commit_sha": "a" * 40,
        "generated_at": "2026-08-26T00:00:00Z",
        "rows": rows,
    }
    metadata = {
        "release_tag": "db-dump/fixture",
        "point_count": len(rows),
        "sha256": "b" * 64,
        "aic_commit_sha": "a" * 40,
        "aisimulate_run": {
            "status": "complete",
            "selected": 4,
            "success": 2,
            "unsupported": 1,
            "failed": 1,
            "method": "randomized_synthetic_engine_replay",
            "completed_at": "2026-08-26T01:00:00Z",
            "runtime": {"packages": {"aisimulate": "0.12.0", "aisimulate-core": "0.12.0"}},
        },
    }
    coverage = {
        "release_tag": "db-dump/fixture",
        "aic_commit_sha": "a" * 40,
        "dump_max_date": "2026-08-25",
        "final_unique_groups": len(rows),
    }
    return predictions, metadata, coverage


def _summary() -> dict[str, object]:
    predictions, metadata, coverage = _inputs()
    return OVERVIEW.build_summary(
        predictions,
        metadata,
        coverage,
        predictions_sha256="c" * 64,
        source_url=("https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/fixture"),
    )


def _qualified_inputs(branch: str = "main") -> tuple[dict, dict, dict]:
    predictions, metadata, coverage = _inputs()
    source = {"branch": branch, "commit_sha": "d" * 40, "clean": True}
    metadata["aisimulate_run"]["runtime"]["source_checkout"] = source
    predictions["aisimulate_run"] = deepcopy(metadata["aisimulate_run"])
    run = {
        "status": "complete",
        "runtime": {
            "source_checkout": {**source, "repository": "https://github.com/ai-dynamo/aisimulate"},
            "cli_entry_point": "aiconfigurator.main:main",
        },
    }
    for document in (predictions, metadata, coverage):
        document["aic_commit_sha"] = source["commit_sha"]
        document["aic_run"] = deepcopy(run)
    return predictions, metadata, coverage


def test_summary_separates_coverage_accuracy_and_multinode_scope() -> None:
    summary = _summary()

    assert summary["scope"] == {
        "measurement_scope": "end_to_end",
        "latency_scope": "client_observed",
        "metrics": ["TTFT", "TPOT"],
        "multinode": "excluded",
        "raw_rows": 4,
        "published_rows": 3,
        "excluded_multinode_rows": 1,
        "claim": (
            "Accuracy applies only to the exact measured operating points in this "
            "snapshot; it is not universal model, hardware, or deployment support."
        ),
    }
    totals = summary["totals"]
    assert totals["models"] == 2
    assert totals["aic"]["points"] == 3
    assert totals["aic"]["ttft_mape_pct"] == pytest.approx(13.33)
    assert totals["aic"]["tpot_mape_pct"] == pytest.approx(13.33)
    assert totals["aisimulate"]["points"] == 2
    assert totals["aisimulate"]["coverage_pct"] == pytest.approx(66.67)
    assert totals["aisimulate"]["status_counts"] == {
        "success": 2,
        "unsupported": 1,
        "failed": 0,
        "unknown": 0,
    }


def test_summary_matches_existing_mape_and_shape_error_semantics() -> None:
    alpha = _summary()["models"][0]

    assert alpha["model"] == "Alpha"
    assert alpha["aic"]["ttft_mape_pct"] == pytest.approx(10.0)
    assert alpha["aic"]["tpot_mape_pct"] == pytest.approx(15.0)
    assert alpha["aic"]["ttft_shape_error_pct"] == pytest.approx(0.0)
    assert alpha["aic"]["tpot_shape_error_pct"] == pytest.approx(25.0)
    assert alpha["aisimulate"]["ttft_mape_pct"] == pytest.approx(7.5)
    assert alpha["aisimulate"]["tpot_mape_pct"] == pytest.approx(10.0)
    assert alpha["aisimulate"]["ttft_shape_error_pct"] == pytest.approx(16.67)
    assert alpha["aisimulate"]["tpot_shape_error_pct"] == pytest.approx(0.0)


def test_workload_labels_match_the_overview_dashboard() -> None:
    assert OVERVIEW._workload_label("1024:1024") == "1k1k"
    assert OVERVIEW._workload_label("1024:8192") == "1k8k"
    assert OVERVIEW._workload_label("8192:1024") == "8k1k"


def test_public_summary_omits_raw_measurements_and_internal_provenance() -> None:
    serialized = str(_summary())

    assert "silicon_ttft_ms" not in serialized
    assert "silicon_workflow_run_id" not in serialized
    assert "gitlab-master.nvidia.com" not in serialized
    assert "linear.app/nvidia" not in serialized


def test_inconsistent_snapshot_fails_closed() -> None:
    predictions, metadata, coverage = _inputs()
    metadata["release_tag"] = "db-dump/other"

    with pytest.raises(OVERVIEW.SnapshotError, match="release_tag"):
        OVERVIEW.build_summary(
            predictions,
            metadata,
            coverage,
            predictions_sha256="c" * 64,
            source_url=("https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/fixture"),
        )


def test_source_url_must_match_validated_release_tag() -> None:
    predictions, metadata, coverage = _inputs()

    with pytest.raises(OVERVIEW.SnapshotError, match="source URL"):
        OVERVIEW.build_summary(
            predictions,
            metadata,
            coverage,
            predictions_sha256="c" * 64,
            source_url=("https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/other"),
        )


def test_unknown_hardware_requires_explicit_multinode_scope() -> None:
    predictions, metadata, coverage = _inputs()
    predictions["rows"][0]["hardware"] = "rtx_6000_ada"
    predictions["rows"][0].pop("is_multinode")

    with pytest.raises(OVERVIEW.SnapshotError, match="unknown hardware family"):
        OVERVIEW.build_summary(
            predictions,
            metadata,
            coverage,
            predictions_sha256="c" * 64,
            source_url=("https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/fixture"),
        )


def test_checked_in_public_snapshot_is_consistent_and_internal_link_free() -> None:
    public_dir = ROOT / "python" / "aisimulate" / "docs" / "e2e-accuracy"
    summary = json.loads((public_dir / "summary.json").read_text())

    assert summary["schema_version"] == OVERVIEW.SCHEMA_VERSION
    assert summary["snapshot"]["measurement_source_url"].startswith("https://github.com/")
    assert summary["scope"]["measurement_scope"] == "end_to_end"
    assert summary["scope"]["latency_scope"] == "client_observed"
    assert summary["scope"]["multinode"] == "excluded"
    assert sum(model["rows"] for model in summary["models"]) == summary["totals"]["rows"]

    statuses = summary["totals"]["aisimulate"]["status_counts"]
    assert sum(statuses.values()) == summary["totals"]["rows"]
    assert statuses["success"] == summary["totals"]["aisimulate"]["points"]

    for path in public_dir.iterdir():
        if path.is_file():
            content = path.read_text()
            for fragment in OVERVIEW.FORBIDDEN_PUBLIC_FRAGMENTS:
                assert fragment not in content, f"{path} contains {fragment}"

    for model in summary["models"]:
        for workload in model["workloads"]:
            assert sum(gpu["rows"] for gpu in workload["gpus"]) == workload["rows"]
            assert sorted(gpu["gpu"] for gpu in workload["gpus"]) == workload["gpu_skus"]


def test_public_page_has_no_e2e_gym_navigation_or_payload() -> None:
    public_dir = ROOT / "python" / "aisimulate" / "docs" / "e2e-accuracy"
    page = (public_dir / "index.html").read_text()
    script = (public_dir / "app.js").read_text()

    assert "E2E Gym" not in page
    assert "predictors" not in page
    assert 'fetch("./branches.json")' in script


def test_public_page_prioritizes_aisimulate_over_aic_baseline() -> None:
    public_dir = ROOT / "python" / "aisimulate" / "docs" / "e2e-accuracy"
    page = (public_dir / "index.html").read_text()
    script = (public_dir / "app.js").read_text()

    assert page.index("AISim CLI TPOT MAPE") < page.index("AIC CLI TPOT MAPE")
    assert script.index('accuracyCard("AISim CLI (new) Error"') < script.index('accuracyCard("AIC CLI (legacy) Error"')
    assert "data-series" not in page


def test_public_page_uses_compact_dashboard_structure() -> None:
    page = (ROOT / "python" / "aisimulate" / "docs" / "e2e-accuracy" / "index.html").read_text()

    assert '<html lang="en" data-theme="dark">' in page
    assert 'class="top-header"' in page
    header = page.split("<header", 1)[1].split("</header>", 1)[0]
    assert 'id="page-title"' in header
    assert "E2E Accuracy Overview" in header
    assert 'id="branch-select"' in header
    assert 'class="tab-nav"' not in page
    assert 'class="summary-grid"' in page
    assert 'class="matrix-panel"' in page
    assert 'class="hero"' not in page


def test_public_page_validates_snapshot_urls_and_nested_schema() -> None:
    script = (ROOT / "python" / "aisimulate" / "docs" / "e2e-accuracy" / "app.js").read_text()

    assert "isSafeHttpsUrl(snapshot.measurement_source_url)" in script
    assert "!Array.isArray(model.workloads)" in script


def test_drilldown_partitions_topologies_and_uses_one_measured_anchor() -> None:
    predictions, metadata, coverage = _inputs()
    predictions["rows"][1]["aisimulate_status"] = "failed"
    predictions["rows"][1]["dynamo_ttft_ms"] = None
    predictions["rows"][1]["dynamo_tpot_ms"] = None
    metadata["aisimulate_run"].update(success=1, failed=2)
    # Internal/free-form data must never be spread into the public payload.
    predictions["rows"][0]["silicon_workflow_run_id"] = "private-run"
    predictions["rows"][0]["infx_config"] = {"private": "secret"}
    result = OVERVIEW.build_summary(
        predictions,
        metadata,
        coverage,
        predictions_sha256="c" * 64,
        source_url=OVERVIEW.INFERENCEX_RELEASE_URL_PREFIX + predictions["release_tag"],
    )
    gpu = result["models"][0]["workloads"][0]["gpus"][0]
    topology = gpu["topologies"][0]
    assert topology["rows"] == 2
    assert topology["aisimulate"]["status_counts"]["failed"] == 1
    first, second = topology["points"]
    assert first["measured"]["ttft_relative"] == 1
    assert first["aisimulate"]["ttft_relative"] == 0.9
    assert first["aisimulate"]["ttft_error_pct"] == 10
    assert second["measured"]["ttft_relative"] == 2
    assert second["aic"]["ttft_relative"] == 2.2
    assert second["aisimulate"]["ttft_relative"] is None
    assert second["aisimulate"]["ttft_error_pct"] is None
    assert "private-run" not in json.dumps(result)
    assert "secret" not in json.dumps(result)
    assert "silicon_ttft_ms" not in json.dumps(result)


def test_distinct_frameworks_and_parallelism_do_not_share_curves() -> None:
    predictions, metadata, coverage = _inputs()
    predictions["rows"][1]["framework"] = "sglang"
    result = OVERVIEW.build_summary(
        predictions,
        metadata,
        coverage,
        predictions_sha256="c" * 64,
        source_url=OVERVIEW.INFERENCEX_RELEASE_URL_PREFIX + predictions["release_tag"],
    )
    topologies = result["models"][0]["workloads"][0]["gpus"][0]["topologies"]
    assert len(topologies) == 2
    assert len({item["id"] for item in topologies}) == 2
    assert {item["framework"] for item in topologies} == {"vllm", "sglang"}
    assert all(len(item["points"]) == 1 for item in topologies)


@pytest.mark.parametrize("branch", ["main", "release/0.12.0", "release/0.13.0/rc1"])
def test_branch_publication_records_evaluated_revision(branch: str) -> None:
    predictions, metadata, coverage = _qualified_inputs(branch)
    result = OVERVIEW.build_summary(
        predictions,
        metadata,
        coverage,
        predictions_sha256="c" * 64,
        branch=branch,
        source_url=OVERVIEW.INFERENCEX_RELEASE_URL_PREFIX + predictions["release_tag"],
    )
    assert result["snapshot"]["evaluated_revision"] == {"branch": branch, "commit_sha": "d" * 40}
    assert result["snapshot"]["aic_source"] == {
        "repository": "https://github.com/ai-dynamo/aisimulate",
        "branch": branch,
        "commit_sha": "d" * 40,
    }


@pytest.mark.parametrize("defect", ["missing", "repository", "revision", "dirty", "incomplete", "metadata"])
def test_branch_publication_rejects_wrong_legacy_cli_source(defect: str) -> None:
    predictions, metadata, coverage = _qualified_inputs()
    run = predictions["aic_run"]
    source = run["runtime"]["source_checkout"]
    if defect == "repository":
        source["repository"] = "https://github.com/ai-dynamo/aiconfigurator"
    elif defect == "revision":
        source["commit_sha"] = "e" * 40
        for document in (predictions, metadata, coverage):
            document["aic_commit_sha"] = "e" * 40
    elif defect == "dirty":
        source["clean"] = False
    elif defect == "incomplete":
        run["status"] = "running"
    for document in (predictions, metadata, coverage):
        document["aic_run"] = deepcopy(run)
        if defect == "missing":
            del document["aic_run"]
    if defect == "metadata":
        metadata["aic_run"]["runtime"]["source_checkout"]["commit_sha"] = "e" * 40
    with pytest.raises(OVERVIEW.SnapshotError, match="AIC"):
        OVERVIEW.build_summary(
            predictions,
            metadata,
            coverage,
            predictions_sha256="c" * 64,
            branch="main",
            source_url=OVERVIEW.INFERENCEX_RELEASE_URL_PREFIX + predictions["release_tag"],
        )


@pytest.mark.parametrize(
    "source",
    [
        None,
        {},
        {"branch": "main", "commit_sha": "d" * 40, "clean": False},
        {"branch": "release/0.12.0", "commit_sha": "d" * 40, "clean": True},
        {"branch": "main", "commit_sha": "short", "clean": True},
    ],
)
def test_branch_labels_cannot_relabel_unqualified_results(source: dict | None) -> None:
    predictions, metadata, coverage = _inputs()
    metadata["aisimulate_run"]["runtime"]["source_checkout"] = source
    with pytest.raises(OVERVIEW.SnapshotError, match="source_checkout"):
        OVERVIEW.build_summary(
            predictions,
            metadata,
            coverage,
            predictions_sha256="c" * 64,
            branch="main",
            source_url=OVERVIEW.INFERENCEX_RELEASE_URL_PREFIX + predictions["release_tag"],
        )


@pytest.mark.parametrize(
    "field,value", [("conc", 0), ("conc", float("nan")), ("silicon_ttft_ms", 0), ("silicon_tpot_ms", -1)]
)
def test_invalid_curve_inputs_fail_closed(field: str, value: float) -> None:
    predictions, metadata, coverage = _inputs()
    predictions["rows"][0][field] = value
    with pytest.raises(OVERVIEW.SnapshotError, match="positive finite"):
        OVERVIEW.build_summary(
            predictions,
            metadata,
            coverage,
            predictions_sha256="c" * 64,
            source_url=OVERVIEW.INFERENCEX_RELEASE_URL_PREFIX + predictions["release_tag"],
        )


@pytest.mark.parametrize("mixed", [False, True])
def test_branch_publication_rejects_mismatched_or_mixed_runs(mixed: bool) -> None:
    predictions, metadata, coverage = _inputs()
    metadata["aisimulate_run"]["runtime"]["source_checkout"] = {
        "branch": "main",
        "commit_sha": "d" * 40,
        "clean": True,
    }
    predictions["aisimulate_run"] = deepcopy(metadata["aisimulate_run"])
    if mixed:
        metadata["aisimulate_run"]["incremental_refreshes"] = [{"selected": 1}]
        message = "one complete run"
    else:
        predictions["aisimulate_run"]["runtime"]["source_checkout"]["commit_sha"] = "e" * 40
        message = "source_checkout disagree"
    with pytest.raises(OVERVIEW.SnapshotError, match=message):
        OVERVIEW.build_summary(
            predictions,
            metadata,
            coverage,
            predictions_sha256="c" * 64,
            branch="main",
            source_url=OVERVIEW.INFERENCEX_RELEASE_URL_PREFIX + predictions["release_tag"],
        )


def test_successful_replay_rejects_zero_latency() -> None:
    predictions, metadata, coverage = _inputs()
    predictions["rows"][0]["dynamo_ttft_ms"] = 0
    with pytest.raises(OVERVIEW.SnapshotError, match="successful AISimulate latencies must be positive"):
        OVERVIEW.build_summary(
            predictions,
            metadata,
            coverage,
            predictions_sha256="c" * 64,
            source_url=OVERVIEW.INFERENCEX_RELEASE_URL_PREFIX + predictions["release_tag"],
        )


def test_page_explains_cli_migration_and_aic_deprecation() -> None:
    page = (ROOT / "python/aisimulate/docs/e2e-accuracy/index.html").read_text()
    assert "new AISim CLI with the legacy AIC CLI" in page
    assert "confidence" in page
    assert "deprecate the AIC CLI" in page
    assert "different prediction coverage" in page
