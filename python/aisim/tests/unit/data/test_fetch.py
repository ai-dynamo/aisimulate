# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for resolve/plan/fetch using a fake transport."""

import hashlib
import json

import pyarrow as pa
import pytest

from aisim_core.data import Identity, configuration_id
from aisim_core.data.fetch import (
    BudgetExceededError,
    Cache,
    IntegrityError,
    OfflineMissingError,
    RevisionError,
    Selection,
    SelectionError,
    TransferError,
    fetch,
    load_category_index,
    plan,
)
from aisim_core.data.parquet import write_parquet

pytestmark = pytest.mark.unit

REPO = "org/dataset"
SHA = "a" * 40
SHA2 = "b" * 40


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact_json(artifact_id: str, path: str, data: bytes, **extra) -> dict:
    return {
        "artifact_id": artifact_id,
        "path": path,
        "role": "runtime",
        "format": extra.pop("format", "parquet"),
        "schema": extra.pop("schema", "aisim.test/v1"),
        "sha256": _sha(data),
        "bytes": len(data),
        **extra,
    }


def _identity(model: str, hardware: str = "h200") -> Identity:
    return Identity(
        hardware=hardware,
        framework="vllm",
        framework_version="0.1",
        operation_family=None,
        model=model,
        effective_settings={},
        unresolved_identity=None,
    )


def build_repo(tmp_path, models=("m1", "m2"), rows=3):
    """Return {path: bytes} for a small fpm dataset with one manifest per model."""
    files: dict[str, bytes] = {}
    entries = []
    for model in models:
        table = pa.table({"x": list(range(rows))})
        pq_path = tmp_path / f"{model}.parquet"
        write_parquet(table, pq_path, "aisim.test/v1")
        data = pq_path.read_bytes()
        art_path = f"fpm/{model}/data.parquet"
        files[art_path] = data
        identity = _identity(model)
        cfg = configuration_id("fpm", "execution-profile", identity)
        manifest = {
            "schema_version": "aisim-artifact-manifest/v1",
            "category": "fpm",
            "purpose": "execution-profile",
            "configuration_id": cfg,
            "identity": identity.model_dump(),
            "requested_settings": None,
            "artifacts": [_artifact_json("a", art_path, data, rows=rows)],
            "provenance": {},
            "evidence_gaps": [],
        }
        mpath = f"fpm/{model}/manifest.json"
        files[mpath] = json.dumps(manifest).encode()
        entries.append(
            {
                "configuration_id": cfg,
                "purpose": "execution-profile",
                "identity": identity.model_dump(),
                "manifest": _artifact_json(
                    "m", mpath, files[mpath], format="json", schema="aisim-artifact-manifest/v1"
                ),
            }
        )
    index = {"schema_version": "aisim-category-index/v1", "category": "fpm", "entries": entries}
    files["catalog/fpm.json"] = json.dumps(index).encode()
    dataset = {
        "schema_version": "aisim-dataset-index/v1",
        "dataset_id": REPO,
        "categories": {
            "fpm": {
                "root": "fpm",
                "layout": "category",
                "index": _artifact_json(
                    "i", "catalog/fpm.json", files["catalog/fpm.json"], format="json", schema="aisim-category-index/v1"
                ),
            }
        },
    }
    files["catalog/dataset.json"] = json.dumps(dataset).encode()
    return files


class FakeTransport:
    def __init__(self, files, refs=None, fail_after=None):
        self.files = files
        self.refs = refs or {"main": SHA}
        self.fail_after = fail_after  # {path: bytes yielded before raising}
        self.resolves = 0
        self.opened: list[str] = []

    def resolve_revision(self, repo, revision):
        self.resolves += 1
        return self.refs[revision]

    def open_file(self, repo, sha, path):
        self.opened.append(path)
        data = self.files[path]
        cut = (self.fail_after or {}).get(path)
        if cut is not None:
            yield data[:cut]
            raise ConnectionError("link dropped")
        for i in range(0, len(data), 7):
            yield data[i : i + 7]


class NoNetwork:
    def resolve_revision(self, *a):
        raise AssertionError("network call in offline mode")

    def open_file(self, *a):
        raise AssertionError("network call in offline mode")


SEL = Selection(category="fpm", model="m1", hardware="h200")


@pytest.fixture
def files(tmp_path):
    return build_repo(tmp_path)


def test_plan_lists_closure_and_bytes(files, tmp_path):
    t = FakeTransport(files)
    p = plan(SEL, repo=REPO, revision="main", transport=t, cache=Cache(tmp_path / "c"))
    assert p.revision == SHA
    assert {f.path for f in p.files} == {
        "catalog/dataset.json",
        "catalog/fpm.json",
        "fpm/m1/manifest.json",
        "fpm/m1/data.parquet",
    }
    assert "fpm/m2/data.parquet" not in t.opened  # unrelated model never touched
    assert p.total_bytes == len(files["fpm/m1/data.parquet"]) + len(files["fpm/m1/manifest.json"]) + len(
        files["catalog/fpm.json"]
    )  # dataset.json has no declared size
    assert p.uncached_bytes == len(files["fpm/m1/data.parquet"])
    json.dumps(p.to_dict())


def test_resolve_once_and_pinned(files, tmp_path):
    t = FakeTransport(files)
    r = fetch(SEL, repo=REPO, revision="main", transport=t, cache=Cache(tmp_path / "c"))
    assert t.resolves == 1
    assert r.plan.revision == SHA and SHA in str(r.root)


def test_full_sha_needs_no_resolution(files, tmp_path):
    t = FakeTransport(files)
    fetch(SEL, repo=REPO, revision=SHA, transport=t, cache=Cache(tmp_path / "c"))
    assert t.resolves == 0


def test_cache_keyed_by_commit(files, tmp_path):
    c = Cache(tmp_path / "c")
    t = FakeTransport(files, refs={"main": SHA, "next": SHA2})
    fetch(SEL, repo=REPO, revision="main", transport=t, cache=c)
    t.opened.clear()
    fetch(SEL, repo=REPO, revision="next", transport=t, cache=c)
    assert "fpm/m1/data.parquet" in t.opened  # different commit, separate cache entry


def test_idempotent_repeat(files, tmp_path):
    c = Cache(tmp_path / "c")
    t = FakeTransport(files)
    first = fetch(SEL, repo=REPO, revision=SHA, transport=t, cache=c)
    assert first.downloaded_bytes == len(files["fpm/m1/data.parquet"])
    t.opened.clear()
    again = fetch(SEL, repo=REPO, revision=SHA, transport=t, cache=c)
    assert again.downloaded_bytes == 0 and t.opened == []
    assert again.plan.uncached_bytes == 0
    assert again.paths == first.paths


def test_interrupted_transfer_leaves_no_partial_and_resumes(files, tmp_path):
    c = Cache(tmp_path / "c")
    art = "fpm/m1/data.parquet"
    t = FakeTransport(files, fail_after={art: 20})
    with pytest.raises(TransferError):
        fetch(SEL, repo=REPO, revision=SHA, transport=t, cache=c)
    target = c.path_for(REPO, SHA, art)
    assert not target.exists()
    assert not list(target.parent.glob("*.part"))
    t.fail_after = None
    r = fetch(SEL, repo=REPO, revision=SHA, transport=t, cache=c)
    assert r.paths[art].read_bytes() == files[art]


def test_corrupt_download_rejected_and_not_cached(files, tmp_path):
    art = "fpm/m1/data.parquet"
    bad = dict(files)
    bad[art] = files[art][:-1] + bytes([files[art][-1] ^ 1])
    c = Cache(tmp_path / "c")
    with pytest.raises(IntegrityError, match="SHA-256"):
        fetch(SEL, repo=REPO, revision=SHA, transport=FakeTransport(bad), cache=c)
    assert not c.path_for(REPO, SHA, art).exists()


def test_schema_mismatch_rejected(tmp_path):
    files = build_repo(tmp_path)
    # Rewrite the artifact with a different schema ID but keep digest consistent.
    other = tmp_path / "other.parquet"
    write_parquet(pa.table({"x": [0, 1, 2]}), other, "aisim.other/v9")
    art = "fpm/m1/data.parquet"
    files[art] = other.read_bytes()
    manifest = json.loads(files["fpm/m1/manifest.json"])
    manifest["artifacts"][0].update(sha256=_sha(files[art]), bytes=len(files[art]))
    files["fpm/m1/manifest.json"] = json.dumps(manifest).encode()
    index = json.loads(files["catalog/fpm.json"])
    index["entries"][0]["manifest"].update(
        sha256=_sha(files["fpm/m1/manifest.json"]), bytes=len(files["fpm/m1/manifest.json"])
    )
    files["catalog/fpm.json"] = json.dumps(index).encode()
    dataset = json.loads(files["catalog/dataset.json"])
    dataset["categories"]["fpm"]["index"].update(
        sha256=_sha(files["catalog/fpm.json"]), bytes=len(files["catalog/fpm.json"])
    )
    files["catalog/dataset.json"] = json.dumps(dataset).encode()
    with pytest.raises(IntegrityError, match="schema ID"):
        fetch(SEL, repo=REPO, revision=SHA, transport=FakeTransport(files), cache=Cache(tmp_path / "c"))


def test_offline_missing_file_report(files, tmp_path):
    c = Cache(tmp_path / "c")
    with pytest.raises(OfflineMissingError) as err:
        fetch(SEL, repo=REPO, revision=SHA, transport=NoNetwork(), cache=c, offline=True)
    assert err.value.missing == ["catalog/dataset.json"]
    assert err.value.to_dict()["error"] == "offline-missing"

    # Warm the metadata and artifact, then remove only the artifact.
    fetch(SEL, repo=REPO, revision=SHA, transport=FakeTransport(files), cache=c)
    c.discard(REPO, SHA, "fpm/m1/data.parquet")
    with pytest.raises(OfflineMissingError) as err:
        fetch(SEL, repo=REPO, revision=SHA, transport=NoNetwork(), cache=c, offline=True)
    assert err.value.missing == ["fpm/m1/data.parquet"]


def test_offline_succeeds_from_cache_with_zero_network(files, tmp_path):
    c = Cache(tmp_path / "c")
    fetch(SEL, repo=REPO, revision=SHA, transport=FakeTransport(files), cache=c)
    r = fetch(SEL, repo=REPO, revision=SHA, transport=None, cache=c, offline=True)
    assert r.downloaded_bytes == 0 and r.plan.missing == []


def test_offline_rejects_unresolved_ref(files, tmp_path):
    with pytest.raises(RevisionError):
        fetch(SEL, repo=REPO, revision="main", transport=NoNetwork(), cache=Cache(tmp_path / "c"), offline=True)


def test_over_budget_refused_before_artifact_transfer(files, tmp_path):
    t = FakeTransport(files)
    size = len(files["fpm/m1/data.parquet"])
    with pytest.raises(BudgetExceededError) as err:
        fetch(SEL, repo=REPO, revision=SHA, transport=t, cache=Cache(tmp_path / "c"), max_bytes=size - 1)
    assert err.value.plan.uncached_bytes == size
    assert "fpm/m1/data.parquet" not in t.opened
    # At the limit, and with explicit opt-in, it proceeds.
    fetch(SEL, repo=REPO, revision=SHA, transport=t, cache=Cache(tmp_path / "c1"), max_bytes=size)
    fetch(SEL, repo=REPO, revision=SHA, transport=t, cache=Cache(tmp_path / "c2"), max_bytes=0, allow_over_budget=True)


def test_cached_bytes_do_not_count_against_budget(files, tmp_path):
    c = Cache(tmp_path / "c")
    fetch(SEL, repo=REPO, revision=SHA, transport=FakeTransport(files), cache=c)
    fetch(SEL, repo=REPO, revision=SHA, transport=FakeTransport(files), cache=c, max_bytes=0)


@pytest.mark.parametrize(
    "selection",
    [
        Selection(category="fpm"),
        Selection(category="fpm", hardware="h200"),  # fpm needs a model
        Selection(category="fpm", model="m1"),  # needs hardware
    ],
)
def test_selectors_required(selection, files, tmp_path):
    t = FakeTransport(files)
    with pytest.raises(SelectionError):
        fetch(selection, repo=REPO, revision=SHA, transport=t, cache=Cache(tmp_path / "c"))
    assert t.opened == []


def test_empty_match_is_error(files, tmp_path):
    with pytest.raises(SelectionError, match="no catalog entries"):
        fetch(
            Selection(category="fpm", model="nope", hardware="h200"),
            repo=REPO,
            revision=SHA,
            transport=FakeTransport(files),
            cache=Cache(tmp_path / "c"),
        )


def test_configuration_id_selector(files, tmp_path):
    cfg = configuration_id("fpm", "execution-profile", _identity("m2"))
    r = fetch(
        Selection(category="fpm", configuration_ids=(cfg,)),
        repo=REPO,
        revision=SHA,
        transport=FakeTransport(files),
        cache=Cache(tmp_path / "c"),
    )
    assert "fpm/m2/data.parquet" in r.paths and "fpm/m1/data.parquet" not in r.paths


def test_corrupt_cache_entry_is_refetched(files, tmp_path):
    c = Cache(tmp_path / "c")
    r = fetch(SEL, repo=REPO, revision=SHA, transport=FakeTransport(files), cache=c)
    r.paths["fpm/m1/data.parquet"].write_bytes(b"junk")
    again = fetch(SEL, repo=REPO, revision=SHA, transport=FakeTransport(files), cache=c)
    assert again.paths["fpm/m1/data.parquet"].read_bytes() == files["fpm/m1/data.parquet"]
    with pytest.raises(OfflineMissingError):
        r.paths["fpm/m1/data.parquet"].write_bytes(b"junk")
        fetch(SEL, repo=REPO, revision=SHA, transport=None, cache=c, offline=True)


def test_exclude_roles_leaves_those_artifacts_out_of_plan_and_download(files, tmp_path):
    t = FakeTransport(files)
    selection = Selection(category="fpm", model="m1", hardware="h200", exclude_roles=("runtime",))
    result = fetch(selection, repo=REPO, revision="main", transport=t, cache=Cache(tmp_path / "c"))
    assert {f.path for f in result.plan.files} == {"catalog/dataset.json", "catalog/fpm.json", "fpm/m1/manifest.json"}
    assert "fpm/m1/data.parquet" not in t.opened
    assert result.downloaded_bytes == 0


def test_load_category_index_reads_only_the_indexes(files, tmp_path):
    t = FakeTransport(files)
    sha, index, dataset = load_category_index(
        "fpm", repo=REPO, revision="main", transport=t, cache=Cache(tmp_path / "c")
    )
    assert sha == SHA and dataset.dataset_id == REPO
    assert {e.identity.model for e in index.entries} == {"m1", "m2"}
    assert sorted(t.opened) == ["catalog/dataset.json", "catalog/fpm.json"]  # no manifest, no artifact
    with pytest.raises(OfflineMissingError):
        load_category_index("fpm", repo=REPO, revision=SHA, cache=Cache(tmp_path / "empty"), offline=True)
    with pytest.raises(SelectionError):
        load_category_index("e2e", repo=REPO, revision="main", transport=t, cache=Cache(tmp_path / "c"))


def test_fpm_layout_rule_round_trips():
    from aisim_core.data.fpm_layout import logical_path, physical_path

    assert physical_path("data/M--m/h/f/1/tp1/manifest.json") == "data/fpm/M--m/h/f/1/tp1/manifest.json"
    assert physical_path("data/history/M--m/x") == "data/fpm/history/M--m/x"
    assert physical_path("data/fpm/M--m/x") == "data/fpm/M--m/x"
    assert logical_path("data/fpm/history/M--m/x") == "data/history/M--m/x"
    with pytest.raises(ValueError):
        physical_path("catalog/dataset.json")
    with pytest.raises(ValueError):
        logical_path("data/M--m/x")
