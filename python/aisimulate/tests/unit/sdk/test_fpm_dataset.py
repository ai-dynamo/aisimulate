# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage trust boundary tests; native schema/cell checks remain in Rust."""

import hashlib
import io
import json
import sys
from pathlib import Path, PurePosixPath

import pytest
import yaml

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


def test_admission_reason_is_listed_and_explains_rejection_and_opt_in(dataset):
    path, manifest, _, calls, cache = dataset
    entry = manifest["profiles"]["gb300-full"]
    entry.update(admission="quarantined", admission_reason="Artifact-only: unavailable to the native FPM consumer")
    path.write_text(json.dumps(manifest))
    assert fpm_dataset.list_fpm_profiles(path)[0]["admission_reason"] == entry["admission_reason"]
    with pytest.raises(ValueError, match=entry["admission_reason"]):
        fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    assert not calls
    with pytest.warns(UserWarning, match=entry["admission_reason"]):
        staged = fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache, allow_unqualified=True)
    assert staged.is_dir()


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


@pytest.fixture
def packaged_dataset(dataset, monkeypatch, tmp_path):
    path, manifest, bodies, _, _ = dataset
    systems = tmp_path / "installed-systems"
    systems.mkdir()
    (systems / "gb300.yaml").write_bytes(bodies["gb300.yaml"])
    entry = manifest["profiles"]["gb300-full"]
    entry["files"].pop(0)
    entry["packaged_system"] = "gb300"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(fpm_dataset, "_SYSTEMS_DIRECTORY", systems)
    return systems


def test_installed_system_yaml_is_copied_without_global_policy_files(dataset, packaged_dataset):
    path, _, bodies, calls, cache = dataset
    for policy in ("query_versions.yaml", "attention_lane_defaults.yaml"):
        (packaged_dataset / policy).write_text("installed global policy")
    staged = fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    assert (staged / "gb300.yaml").read_bytes() == bodies["gb300.yaml"]
    assert not (staged / "query_versions.yaml").exists()
    assert not (staged / "attention_lane_defaults.yaml").exists()
    assert len(calls) == 2
    assert fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache, local_files_only=True) == staged
    assert len(calls) == 2


def test_installed_system_changes_select_a_new_cache_without_changing_old_roots(dataset, packaged_dataset):
    path, _, bodies, calls, cache = dataset
    original = fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    changed = bodies["gb300.yaml"] + b"# updated hardware configuration\n"
    (packaged_dataset / "gb300.yaml").write_bytes(changed)
    with pytest.raises(FileNotFoundError, match="not cached"):
        fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache, local_files_only=True)
    assert len(calls) == 2
    updated = fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    assert updated != original
    assert (original / "gb300.yaml").read_bytes() == bodies["gb300.yaml"]
    assert (updated / "gb300.yaml").read_bytes() == changed
    assert fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache, local_files_only=True) == updated
    assert len(calls) == 4


def test_copied_system_yaml_is_verified_on_cache_reuse(dataset, packaged_dataset):
    path, _, _, calls, cache = dataset
    staged = fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    (staged / "gb300.yaml").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="SHA256"):
        fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache, local_files_only=True)
    assert len(calls) == 2


@pytest.mark.parametrize("mutation", ["missing", "escape", "collision"])
def test_invalid_packaged_system_fails_before_download(dataset, packaged_dataset, tmp_path, mutation):
    path, manifest, bodies, calls, cache = dataset
    source = packaged_dataset / "gb300.yaml"
    if mutation == "missing":
        source.unlink()
    elif mutation == "escape":
        outside = tmp_path / "outside.yaml"
        outside.write_bytes(bodies["gb300.yaml"])
        source.unlink()
        source.symlink_to(outside)
    else:
        manifest["profiles"]["gb300-full"]["files"].append(
            {
                "path": "archive/gb300.yaml",
                "target": "gb300.yaml",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    assert not calls
    assert not cache.exists()


@pytest.mark.parametrize("system", ["../gb300", "nested/gb300", r"nested\gb300", "gb300.yaml", 123])
def test_packaged_system_must_be_a_system_name(dataset, packaged_dataset, system):
    path, manifest, _, calls, cache = dataset
    manifest["profiles"]["gb300-full"]["packaged_system"] = system
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="packaged_system"):
        fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    assert not calls


def test_profiles_with_same_loader_target_remain_isolated(dataset):
    path, manifest, bodies, _, cache = dataset
    first = fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache)
    target = "data/gb300/sglang/dev-example/fpm_forward_perf.parquet"
    original = bodies[target]
    bodies[target] = b"different deployment table"
    other = json.loads(json.dumps(manifest["profiles"]["gb300-full"]))
    other["files"][1]["sha256"] = hashlib.sha256(bodies[target]).hexdigest()
    manifest["profiles"]["gb300-other"] = other
    path.write_text(json.dumps(manifest))
    second = fpm_dataset.materialize_fpm_profile(path, "gb300-other", cache_dir=cache)
    assert first != second
    assert (first / target).read_bytes() == original
    assert (second / target).read_bytes() == bodies[target]


def test_discovery_lists_admission_without_download_or_cache(dataset, packaged_dataset):
    path, manifest, _, calls, cache = dataset
    manifest["profiles"]["gb300-full"]["admission"] = "development"
    builtin = packaged_dataset / "profiles/example_fpm/hf_dataset.json"
    builtin.parent.mkdir(parents=True)
    builtin.write_text(json.dumps(manifest))
    secondary = builtin.with_name("four_gpu_hf_dataset.json")
    secondary.write_text(json.dumps(manifest))
    expected = {
        "manifest": str(builtin),
        "profile": "gb300-full",
        "admission": "development",
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
    }
    assert fpm_dataset.list_fpm_profiles("example_fpm") == [expected]
    assert fpm_dataset.list_fpm_profiles(builtin) == [expected]
    assert fpm_dataset.list_fpm_profiles("example_fpm/four_gpu_hf_dataset.json") == [
        {**expected, "manifest": str(secondary)}
    ]
    assert fpm_dataset.list_fpm_profiles() == [{**expected, "manifest": str(secondary)}, expected]
    assert not calls
    assert not cache.exists()


def test_packaged_manifest_materializes_by_id(dataset, packaged_dataset):
    path, _, _, calls, cache = dataset
    builtin = packaged_dataset / "profiles/example_fpm/hf_dataset.json"
    builtin.parent.mkdir(parents=True)
    builtin.write_bytes(path.read_bytes())
    by_id = fpm_dataset.materialize_fpm_profile("example_fpm", "gb300-full", cache_dir=cache)
    assert fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache, local_files_only=True) == by_id
    assert len(calls) == 2


def test_local_directory_does_not_shadow_packaged_manifest_id(dataset, packaged_dataset, monkeypatch):
    path, _, _, _, _ = dataset
    builtin = packaged_dataset / "profiles/example_fpm/hf_dataset.json"
    builtin.parent.mkdir(parents=True)
    builtin.write_bytes(path.read_bytes())
    monkeypatch.chdir(builtin.parent.parent)
    assert fpm_dataset.list_fpm_profiles("example_fpm")[0]["manifest"] == str(builtin)


def test_legacy_identity_retains_existing_offline_cache_key(dataset):
    path, manifest, bodies, calls, cache = dataset
    entry = manifest["profiles"]["gb300-full"]
    entry["identity"] = {"model_path": "example/model", "tp": 2}
    path.write_text(json.dumps(manifest))
    fingerprint = json.dumps(
        {"repo": manifest["repo_id"], "revision": manifest["revision"], "profile": entry}, sort_keys=True
    ).encode()
    existing = cache / hashlib.sha256(fingerprint).hexdigest()
    for target, body in bodies.items():
        file = existing / target
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(body)
    assert fpm_dataset.materialize_fpm_profile(path, "gb300-full", cache_dir=cache, local_files_only=True) == existing
    assert not calls


def test_local_manifest_precedes_packaged_id(dataset, packaged_dataset, monkeypatch):
    path, _, _, _, _ = dataset
    monkeypatch.chdir(path.parent)
    assert fpm_dataset.list_fpm_profiles(path.name)[0]["manifest"] == str(path)


@pytest.mark.parametrize("name", ["missing_fpm", "../outside", r"example_fpm\hf_dataset.json"])
def test_invalid_packaged_manifest_id(dataset, packaged_dataset, name):
    with pytest.raises((FileNotFoundError, ValueError)):
        fpm_dataset.list_fpm_profiles(name)


def test_discovery_cli(dataset, monkeypatch, capsys):
    path, _, _, calls, _ = dataset
    monkeypatch.setattr(sys, "argv", ["fpm_dataset", str(path), "--list"])
    fpm_dataset.main()
    assert json.loads(capsys.readouterr().out) == fpm_dataset.list_fpm_profiles(path)
    assert not calls


def test_discovery_cli_lists_shipped_artifact_only_profiles(monkeypatch, capsys, tmp_path):
    def no_network(*args, **kwargs):
        pytest.fail("listing or rejecting a quarantined profile must not download")

    monkeypatch.setattr(fpm_dataset, "urlopen", no_network)
    monkeypatch.setattr(sys, "argv", ["fpm_dataset", "--list", "minimax_m27_fpm"])
    fpm_dataset.main()
    listed = {entry["profile"]: entry for entry in json.loads(capsys.readouterr().out)}
    for variant in ("on", "off"):
        key = f"h200_sxm-vllm-fp8-block-tp4-ep4-kv-{variant}"
        assert listed[key]["admission"] == "quarantined"
        assert "Artifact-only: unavailable to the native FPM consumer" in listed[key]["admission_reason"]
        with pytest.raises(ValueError, match="Artifact-only"):
            fpm_dataset.materialize_fpm_profile("minimax_m27_fpm", key, cache_dir=tmp_path / "cache")
    assert listed["h200_sxm-vllm-fp8-block-tp2-ep2"]["admission"] == "development"
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("args", [[], ["example_fpm"], ["example_fpm", "profile", "--list"]])
def test_cli_requires_unambiguous_action(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["fpm_dataset", *args])
    with pytest.raises(SystemExit) as exc:
        fpm_dataset.main()
    assert exc.value.code == 2


def test_shipped_profiles_reference_installed_systems():
    for row in fpm_dataset.list_fpm_profiles():
        manifest = json.loads(Path(row["manifest"]).read_text())
        entry = manifest["profiles"][row["profile"]]
        if entry.get("packaged_system") is not None:
            spec = yaml.safe_load(fpm_dataset._packaged_system_bytes(entry["packaged_system"]))
            data_dir = PurePosixPath(spec["data_dir"])
            tables = [PurePosixPath(item["target"]) for item in entry["files"] if item["target"].endswith(".parquet")]
            assert tables
            for table in tables:
                assert table.is_relative_to(data_dir), (row, data_dir, table)
                # The native loader expects <data_dir>/<backend>/<version>/<table>.
                assert len(table.relative_to(data_dir).parts) == 3, (row, table)
