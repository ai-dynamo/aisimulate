# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
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
            "runtime": {
                "packages": {"aisimulate": "0.12.0", "aisimulate-core": "0.12.0"}
            },
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
        source_url=(
            "https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/"
            "db-dump/fixture"
        ),
    )


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
            source_url=(
                "https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/"
                "db-dump/fixture"
            ),
        )


def test_source_url_must_match_validated_release_tag() -> None:
    predictions, metadata, coverage = _inputs()

    with pytest.raises(OVERVIEW.SnapshotError, match="source URL"):
        OVERVIEW.build_summary(
            predictions,
            metadata,
            coverage,
            predictions_sha256="c" * 64,
            source_url=(
                "https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/"
                "db-dump/other"
            ),
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
            source_url=(
                "https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/"
                "db-dump/fixture"
            ),
        )


def test_checked_in_public_snapshot_is_consistent_and_internal_link_free() -> None:
    public_dir = ROOT / "python" / "aisimulate" / "docs" / "e2e-accuracy"
    summary = json.loads((public_dir / "summary.json").read_text())

    assert summary["schema_version"] == OVERVIEW.SCHEMA_VERSION
    assert summary["snapshot"]["measurement_source_url"].startswith(
        "https://github.com/"
    )
    assert summary["scope"]["measurement_scope"] == "end_to_end"
    assert summary["scope"]["latency_scope"] == "client_observed"
    assert summary["scope"]["multinode"] == "excluded"
    assert (
        sum(model["rows"] for model in summary["models"]) == summary["totals"]["rows"]
    )

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
            assert (
                sorted(gpu["gpu"] for gpu in workload["gpus"]) == workload["gpu_skus"]
            )


def test_public_page_has_no_e2e_gym_navigation_or_payload() -> None:
    public_dir = ROOT / "python" / "aisimulate" / "docs" / "e2e-accuracy"
    page = (public_dir / "index.html").read_text()
    script = (public_dir / "app.js").read_text()

    assert "E2E Gym" not in page
    assert "predictors" not in page
    assert 'fetch("./summary.json")' in script


def test_public_page_prioritizes_aisimulate_over_aic_baseline() -> None:
    public_dir = ROOT / "python" / "aisimulate" / "docs" / "e2e-accuracy"
    page = (public_dir / "index.html").read_text()
    script = (public_dir / "app.js").read_text()

    assert page.index("AISimulate TPOT MAPE") < page.index("AIC TPOT MAPE")
    assert script.index('accuracyCard("Average AISimulate Error"') < script.index(
        'accuracyCard("Average AIC Error"'
    )
    assert "data-series" not in page


def test_public_page_uses_compact_dashboard_structure() -> None:
    page = (
        ROOT / "python" / "aisimulate" / "docs" / "e2e-accuracy" / "index.html"
    ).read_text()

    assert '<html lang="en" data-theme="dark">' in page
    assert 'class="top-header"' in page
    assert 'class="tab-nav"' in page
    assert 'class="summary-grid"' in page
    assert 'class="matrix-panel"' in page
    assert 'class="hero"' not in page


def test_public_page_validates_snapshot_urls_and_nested_schema() -> None:
    script = (
        ROOT / "python" / "aisimulate" / "docs" / "e2e-accuracy" / "app.js"
    ).read_text()

    assert "isSafeHttpsUrl(snapshot.measurement_source_url)" in script
    assert "!Array.isArray(model.workloads)" in script
