# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FPM accuracy dataset on the unified ``data/fpm/`` layout (AISIM-2338).

The unified tree is produced from the legacy fixture by the library's own migration, so the legacy
spelling inside moved files (logical keys) is exercised exactly as in the published dataset.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aisim_core.data import fpm_migration
from scripts.fpm_accuracy.exceptions import ConfigurationError, DataError
from scripts.fpm_accuracy.hf import HfDataset
from scripts.fpm_accuracy.hf import dataset as dataset_module
from tests.fpm_accuracy.test_hf_dataset import CONFIGURATION_PATH, REVISION, _build_dataset, _fpm_payload

REPO = "ai-dynamo/aisim-data"


def _legacy_and_unified(tmp_path: Path) -> tuple[Path, Path]:
    legacy = tmp_path / "legacy"
    content = f"{json.dumps(_fpm_payload())}\n".encode()
    _build_dataset(
        legacy,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
    )
    (legacy / "catalog/fpm.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    {
                        "path": f"{CONFIGURATION_PATH}/fpm/fpm.parquet",
                        "schema_name": "aic_fpm_forward_perf",
                        "schema_version": 6,
                    }
                ],
            }
        )
    )
    unified = tmp_path / "unified"
    fpm_migration.migrate(legacy, unified, revision=REVISION, repo=REPO)
    return legacy, unified


def _signature(case) -> dict:
    configuration = case.configuration
    return {
        "case_id": case.case_id,
        "membership": case.measurement_membership_sha256,
        "status": case.status.value,
        "configuration": configuration.configuration_id,
        "configuration_path": configuration.configuration_path,
        "truth": [(f.measurement_file_id, f.path, f.sha256) for f in case.truth_files],
        "helpers": [(f.measurement_file_id, f.path, f.sha256) for f in case.helper_files],
        "fpm": [(a.artifact_id, a.path, a.sha256, a.row_count) for a in case.fpm_artifacts],
        "observations": [(o.observation_id, o.source_path, o.actual_ms) for o in case.observations],
    }


def test_unified_resolution_is_identical_to_legacy(tmp_path: Path) -> None:
    legacy_root, unified_root = _legacy_and_unified(tmp_path)
    legacy = HfDataset.from_local(legacy_root, revision=REVISION, repo_id=REPO)
    unified = HfDataset.from_local(unified_root, revision=REVISION, repo_id=REPO)

    assert (legacy.layout, unified.layout) == ("legacy", "unified")
    assert [c.configuration_id for c in unified.configurations()] == [
        c.configuration_id for c in legacy.configurations()
    ]
    assert _signature(unified.measurement_case(CONFIGURATION_PATH)) == _signature(
        legacy.measurement_case(CONFIGURATION_PATH)
    )


def test_links_use_the_physical_path_and_identity_keeps_the_logical_key(tmp_path: Path) -> None:
    _, unified_root = _legacy_and_unified(tmp_path)
    case = HfDataset.from_local(unified_root, revision=REVISION, repo_id=REPO).measurement_case(CONFIGURATION_PATH)

    assert case.configuration.configuration_path == CONFIGURATION_PATH  # data/<model>/..., unchanged
    assert all("/data/fpm/Org--Model/" in url for url in case.provenance_urls)
    assert all(f"/{REPO}/blob/{REVISION}/" in url for url in case.provenance_urls)


def test_leaf_manifest_tamper_is_caught_by_the_envelope_hash(tmp_path: Path) -> None:
    _, unified_root = _legacy_and_unified(tmp_path)
    leaf = unified_root / "data/fpm/Org--Model/h100-sxm/vllm/1.0/single/manifest.json"
    leaf.write_text(leaf.read_text() + " ")
    with pytest.raises(DataError, match="hash mismatch"):
        HfDataset.from_local(unified_root, revision=REVISION, repo_id=REPO)


def test_selection_filters_before_any_artifact_is_needed(tmp_path: Path) -> None:
    _, unified_root = _legacy_and_unified(tmp_path)
    by_model = HfDataset.from_local(unified_root, revision=REVISION, repo_id=REPO, models=["Org/Model"])
    assert len(by_model.configurations()) == 1
    assert (
        len(HfDataset.from_local(unified_root, revision=REVISION, repo_id=REPO, hardware=["h100_sxm"]).configurations())
        == 1
    )
    by_path = HfDataset.from_local(
        unified_root, revision=REVISION, repo_id=REPO, configuration_paths=[CONFIGURATION_PATH]
    )
    assert len(by_path.configurations()) == 1
    with pytest.raises(ConfigurationError, match="no FPM configuration matches"):
        HfDataset.from_local(unified_root, revision=REVISION, repo_id=REPO, models=["Other/Model"])


class _FakeTransport:
    """Serves a local unified tree; records every opened path."""

    root: Path
    opened: list[str]

    def __init__(self, token: str | None = None) -> None:
        self.opened = type(self).opened

    def resolve_revision(self, repo: str, revision: str) -> str:
        assert repo == REPO
        return REVISION

    def open_file(self, repo: str, sha: str, path: str) -> Iterator[bytes]:
        self.opened.append(path)
        yield (type(self).root / path).read_bytes()


def test_from_hub_selects_before_downloading_and_skips_excluded_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, unified_root = _legacy_and_unified(tmp_path)
    provenance = "data/fpm/Org--Model/h100-sxm/vllm/1.0/single/fpm/provenance/source.tar.gz"
    _FakeTransport.root, _FakeTransport.opened = unified_root, []
    monkeypatch.setattr(dataset_module, "HfTransport", _FakeTransport)

    dataset = HfDataset.from_hub(REPO, revision="main", cache_dir=tmp_path / "cache", models=["Org/Model"])

    assert dataset.revision == REVISION
    assert len(dataset.measurement_case(CONFIGURATION_PATH).observations) == 1
    assert "catalog/dataset.json" in _FakeTransport.opened
    assert provenance not in _FakeTransport.opened
    assert not any(path.startswith("data/") and "Org--Model" not in path for path in _FakeTransport.opened)
    with pytest.raises(ConfigurationError, match="no FPM configuration matches"):
        HfDataset.from_hub(REPO, revision="main", cache_dir=tmp_path / "cache", models=["Missing/Model"])


def test_parity_script_passes_on_a_migrated_fixture_and_detects_a_tampered_tree(tmp_path: Path, monkeypatch) -> None:
    from scripts.fpm_accuracy import parity_unified_layout as parity

    _, unified = _legacy_and_unified(tmp_path)
    report = tmp_path / "report.json"
    argv = ["parity", "--tree", str(unified), "--repo-id", REPO, "--revision", REVISION, "--skip-pins"]
    monkeypatch.setattr("sys.argv", [*argv, "--workdir", str(tmp_path / "w1"), "--report", str(report)])
    assert parity.main() == 0
    checks = json.loads(report.read_text())["checks"]
    assert checks["truth membership, hashes, FPM artifacts, case IDs, provenance links"]["observations"] == 1
    assert all(item["ok"] for item in checks.values())

    # Changed truth bytes are refused by the library's size/hash verification before evaluation sees them.
    truth = unified / "data/fpm/Org--Model/h100-sxm/vllm/1.0/single/measurements/truth.jsonl"
    truth.write_text(truth.read_text() + "\n")
    monkeypatch.setattr("sys.argv", [*argv, "--workdir", str(tmp_path / "w2"), "--report", str(report)])
    with pytest.raises(DataError, match="cannot fetch"):
        parity.main()
