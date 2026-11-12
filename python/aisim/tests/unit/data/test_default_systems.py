# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default systems root is composed from the pinned dataset, never from packaged tables."""

import importlib.util
import re
from pathlib import Path

import pytest

from aisim_core.data import default_systems as ds
from aisim_core.data.ensure import MissingDataError

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location("op_publish_fixtures", Path(__file__).with_name("test_op_publish.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
tree, _build = _fixtures.tree, _fixtures._build

SHA = "b" * 40


class DirTransport:
    """Serves a local publish tree as if it were the Hub repository at ``SHA``."""

    def __init__(self, root: Path):
        self.root, self.opened = root, []

    def resolve_revision(self, repo, revision):
        return SHA

    def open_file(self, repo, sha, path):
        self.opened.append(path)
        yield (self.root / path).read_bytes()


@pytest.fixture
def published(tree, tmp_path):
    out = tmp_path / "published"
    _build(tree, out)
    return out


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for name in (ds.DATASET_ENV, ds.LOCAL_ENV, "AISIM_DATA_OFFLINE", "AISIM_DATA_MAX_BYTES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AISIM_DATA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(ds, "op_latency_pin", lambda: ("o/n", SHA))
    ds.clear_default_systems_cache()
    yield
    ds.clear_default_systems_cache()


def test_pin_file_names_a_full_commit_sha():
    import json

    pin = json.loads(ds.PIN_FILE.read_text())
    assert pin["repo_id"] == "ai-dynamo/aisim-data"
    assert re.fullmatch(r"[0-9a-f]{40}", pin["revision"])


def test_composes_root_from_the_pinned_revision_and_reuses_it(published):
    transport = DirTransport(published)
    first = Path(ds.default_systems_path(transport=transport))
    assert (first / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet").is_file()
    assert (first / "query_versions.yaml").is_file() and (first / "perf_data_reuse_manifest.yaml").is_file()
    assert (first / "h200_sxm.yaml").is_file()  # packaged specs travel with the data
    assert transport.opened

    ds.clear_default_systems_cache()
    again = DirTransport(published)
    assert Path(ds.default_systems_path(transport=again)) == first
    assert again.opened == []  # warm cache: no network


def test_offline_with_cold_cache_fails_instead_of_falling_back(monkeypatch):
    monkeypatch.setenv("AISIM_DATA_OFFLINE", "1")
    with pytest.raises(MissingDataError, match="no fallback"):
        ds.default_systems_path()


def test_local_override_needs_no_network(published, monkeypatch):
    monkeypatch.setenv(ds.LOCAL_ENV, str(published))
    root = Path(ds.default_systems_path())
    assert (root / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet").is_file()


def test_local_override_that_is_not_a_dataset_fails(tmp_path, monkeypatch):
    monkeypatch.setenv(ds.LOCAL_ENV, str(tmp_path))
    with pytest.raises(MissingDataError, match="dataset-layout"):
        ds.default_systems_path()


def test_explicit_systems_paths_skip_the_dataset(tmp_path):
    from aisim_core.sdk import perf_database as pdb

    explicit = tmp_path / "mine"
    explicit.mkdir()
    assert pdb._normalize_systems_paths([str(explicit)]) == [str(explicit)]  # nothing fetched, no pin read
