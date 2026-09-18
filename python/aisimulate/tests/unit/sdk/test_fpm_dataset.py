# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage trust boundary tests; native schema/cell checks remain in Rust."""

import hashlib
import io
import json

import pytest

from aisimulate_core.sdk import fpm_dataset

pytestmark = pytest.mark.unit


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    bodies = {
        "gb300.yaml": b"data_dir: data/gb300\n",
        "data/gb300/sglang/dev-example/fpm_forward_perf.parquet": b"test parquet bytes",
        "data/gb300/sglang/dev-example/fpm_forward_perf.metadata.json": b'{"schema_version":7}',
    }
    manifest = {
        "format_version": 1,
        "repo_id": "nvidia/aisimulate-fpm-dataset",
        "revision": "a" * 40,
        "profiles": {
            "gb300-full": {
                "admission": "serving",
                "files": [
                    {"path": "archive/" + path, "target": path, "sha256": hashlib.sha256(body).hexdigest()}
                    for path, body in bodies.items()
                ],
            }
        },
    }
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(manifest))
    calls = []

    def fetch(request, **kwargs):
        url = request.full_url
        calls.append(url)
        prefix = f"https://huggingface.co/datasets/{manifest['repo_id']}/resolve/{'a' * 40}/archive/"
        assert url.startswith(prefix)
        return io.BytesIO(bodies[url.removeprefix(prefix)])

    monkeypatch.setattr(fpm_dataset, "urlopen", fetch)
    return path, manifest, bodies, calls, tmp_path / "cache"


def test_pinned_fetch_and_offline_cache(dataset):
    path, _, bodies, calls, cache = dataset
    staged = fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    assert len(calls) == 3
    for target, body in bodies.items():
        assert (staged / target).read_bytes() == body
    assert fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache, local_files_only=True) == staged
    assert len(calls) == 3


def test_offline_missing_never_downloads(dataset):
    path, _, _, calls, cache = dataset
    with pytest.raises(FileNotFoundError):
        fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache, local_files_only=True)
    assert not calls


@pytest.mark.parametrize("mutation", ["revision", "path", "target", "duplicate", "diagnostic", "metadata"])
def test_bad_manifests_fail_before_network(dataset, mutation):
    path, manifest, _, calls, cache = dataset
    entry = manifest["profiles"]["gb300-full"]
    if mutation == "revision":
        manifest["revision"] = "main"
    elif mutation in ("path", "target"):
        entry["files"][0][mutation] = "../escape"
    elif mutation == "duplicate":
        entry["files"].append(entry["files"][0])
    elif mutation == "diagnostic":
        entry["admission"] = "quarantined"
    else:
        entry["files"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    assert not calls


def test_corrupt_download_is_not_published(dataset):
    path, _, bodies, _, cache = dataset
    bodies["gb300.yaml"] = b"corruption"
    with pytest.raises(ValueError, match="SHA256"):
        fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    assert list(cache.iterdir()) == []


def test_corrupt_cache_fails_without_redownload(dataset):
    path, _, _, calls, cache = dataset
    staged = fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    (staged / "gb300.yaml").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="SHA256"):
        fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    assert len(calls) == 3


def test_symlink_cache_file_rejected(dataset, tmp_path):
    path, _, bodies, _, cache = dataset
    staged = fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    outside = tmp_path / "outside.yaml"
    outside.write_bytes(bodies["gb300.yaml"])
    (staged / "gb300.yaml").unlink()
    (staged / "gb300.yaml").symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe"):
        fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)


@pytest.mark.parametrize("admission", ["development", "quarantined"])
def test_historical_reproduction_requires_explicit_opt_in(dataset, admission):
    path, manifest, _, calls, cache = dataset
    manifest["profiles"]["gb300-full"]["admission"] = admission
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not admitted"):
        fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    assert not calls
    with pytest.warns(UserWarning, match="historical reproduction only"):
        staged = fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache, allow_unqualified=True)
    assert staged.is_dir()
    with pytest.raises(ValueError, match="not admitted"):
        fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)


def test_hf_login_and_environment_credentials_do_not_follow_redirects(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    token_path = tmp_path / "token"
    token_path.write_text("test-login-secret\n")
    monkeypatch.setenv("HF_TOKEN_PATH", str(token_path))
    url = "https://huggingface.co/datasets/org/repo/resolve/" + "a" * 40 + "/file"
    request = fpm_dataset._download_request(url)
    assert request.get_header("Authorization") == "Bearer test-login-secret"
    redirected = fpm_dataset._HubRedirectHandler().redirect_request(
        request, None, 302, "Found", {}, "https://cdn.example/file"
    )
    assert redirected.get_header("Authorization") is None
    monkeypatch.setenv("HF_TOKEN", "test-environment-secret")
    assert fpm_dataset._download_request(url).get_header("Authorization") == "Bearer test-environment-secret"
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    assert fpm_dataset._download_request(url).get_header("Authorization") is None


@pytest.mark.parametrize(
    ("target", "authenticated"),
    [
        ("https://huggingface.co/api/resolve-cache/file", True),
        ("https://huggingface.co:443/api/resolve-cache/file", True),
        ("https://cdn.example/file", False),
        ("https://huggingface.co:444/file", False),
        ("http://huggingface.co/file", False),
    ],
)
def test_hub_cache_redirects_preserve_auth_only_within_same_https_origin(monkeypatch, target, authenticated):
    monkeypatch.setenv("HF_TOKEN", "test-private-dataset-token")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    request = fpm_dataset._download_request("https://huggingface.co/datasets/org/repo/resolve/commit/file")
    redirected = fpm_dataset._HubRedirectHandler().redirect_request(request, None, 302, "Found", {}, target)
    assert redirected.get_header("Authorization") == ("Bearer test-private-dataset-token" if authenticated else None)
    # A second cross-origin hop must drop same-origin retained auth too.
    second = fpm_dataset._HubRedirectHandler().redirect_request(
        redirected, None, 302, "Found", {}, "https://cdn.example/second"
    )
    assert second.get_header("Authorization") is None


def test_concurrent_downloaders_publish_one_complete_profile(dataset, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, local

    path, _, bodies, _, cache = dataset
    download = fpm_dataset.urlopen
    barrier = Barrier(4)
    state = local()

    def simultaneous_download(request, **kwargs):
        if not getattr(state, "started", False):
            state.started = True
            barrier.wait(timeout=10)
        return download(request, **kwargs)

    monkeypatch.setattr(fpm_dataset, "urlopen", simultaneous_download)
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(
            workers.map(lambda _: fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache), range(4))
        )
    assert len(set(results)) == 1
    assert list(cache.iterdir()) == [results[0]]
    for target, body in bodies.items():
        assert (results[0] / target).read_bytes() == body
