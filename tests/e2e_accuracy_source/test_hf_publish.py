# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import EntryNotFoundError

from scripts.e2e_accuracy import hf_cache as cache
from scripts.e2e_accuracy import hf_publish as publisher


def bundle(root):
    def acquisition(value):
        return cache.Acquisition(
            "https://inferencex.semianalysis.com/api/v1/benchmarks",
            cache.canonical(value),
            "2026-10-17",
        )

    return cache.prepare_snapshot(
        root,
        acquisition({"models": [{"name": "Example"}]}),
        {
            "Example": acquisition(
                [
                    {
                        "id": "1",
                        "model": "example",
                        "hardware": "h200",
                        "framework": "sglang",
                        "metrics": {"mean_ttft": 0.5},
                    }
                ]
            )
        },
        normalizer_revision="a" * 40,
        publication={
            "redistribution_status": "unresolved",
            "attribution": "InferenceX",
            "terms_request": "https://example.org/request",
        },
    )


class Api:
    token = False

    def __init__(self, private=True, siblings=()):
        self.private = private
        self.siblings = siblings
        self.commits = []

    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(private=self.private, sha="a" * 40, siblings=self.siblings)

    def get_repo_discussions(self, *args, **kwargs):
        return []

    def create_commit(self, *args, **kwargs):
        self.commits.append(kwargs)
        return SimpleNamespace(oid="b" * 40, pr_url="https://example.org/candidate")


def missing(*args, **kwargs):
    raise EntryNotFoundError("missing")


def test_inferencex_cannot_publish_publicly(tmp_path):
    bundle(tmp_path)
    api = Api(private=False)
    with pytest.raises(ValueError, match="private dataset"):
        publisher.publish_snapshot(tmp_path, "org/cache", signoff="Test <test@example.org>", api=api)
    assert not api.commits


def test_unchanged_snapshot_does_not_commit(tmp_path, monkeypatch):
    bundle(tmp_path)
    monkeypatch.setattr(publisher, "hf_hub_download", lambda *a, **k: str(tmp_path / cache.INDEX_PATH))
    api = Api()
    result = publisher.publish_snapshot(tmp_path, "org/cache", signoff="Test <test@example.org>", api=api)
    assert result["status"] == "unchanged" and not api.commits


def test_delta_skips_existing_payloads_and_opens_candidate(tmp_path, monkeypatch):
    index = bundle(tmp_path)
    entry = index["partitions"][0]["artifacts"][0]
    api = Api(siblings=[SimpleNamespace(rfilename=entry["path"], lfs=SimpleNamespace(sha256=entry["sha256"]))])
    monkeypatch.setattr(publisher, "hf_hub_download", missing)
    result = publisher.publish_snapshot(tmp_path, "org/cache", signoff="Test <test@example.org>", api=api)
    assert result["status"] == "candidate_created"
    commit = api.commits[0]
    assert commit["create_pr"] and commit["parent_commit"] == "a" * 40
    assert "Signed-off-by: Test <test@example.org>" in commit["commit_description"]
    assert entry["path"] not in {operation.path_in_repo for operation in commit["operations"]}


def test_other_source_indexes_are_preserved(tmp_path, monkeypatch):
    bundle(tmp_path)
    category = tmp_path / "remote-category.json"
    category.write_text(
        json.dumps(
            {
                "schema_version": "e2e-index/v1",
                "sources": {"developer": {"path": "kept.json"}},
            }
        )
    )

    def download(repo, path, **kwargs):
        if path == "catalog/e2e/index.json":
            return str(category)
        return missing()

    monkeypatch.setattr(publisher, "hf_hub_download", download)
    publisher.publish_snapshot(tmp_path, "org/cache", signoff="Test <test@example.org>", api=Api())
    assert json.loads((tmp_path / "catalog/e2e/index.json").read_text())["sources"]["developer"] == {
        "path": "kept.json"
    }
