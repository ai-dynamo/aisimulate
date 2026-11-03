# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise immutable wheel bundles and retry/cross-run identity without credentials."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / ".github/actions/wheel-storage"
spec = importlib.util.spec_from_file_location("wheel_transfer", ACTION / "transfer.py")
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    source = tmp_path / "build"
    source.mkdir()
    output = tmp_path / "installed"
    manifest = tmp_path / "wheel-manifest.json"
    env = {
        "STORAGE_MODE": "upload",
        "STORAGE_NAME": "simulation-wheel-base",
        "STORAGE_PATH": str(source),
        "STORAGE_URL": "https://artifactory.invalid/artifactory",
        "STORAGE_REPO": "ci-wheels-local",
        "STORAGE_TOKEN": "fixture-write-token",
        "STORAGE_RUN_ID": "101",
        "STORAGE_SOURCE_SHA": "a" * 40,
        "STORAGE_ARCHITECTURE": "amd64",
        "STORAGE_MANIFEST": str(manifest),
        "GITHUB_RUN_ID": "101",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_REPOSITORY": "ai-dynamo/aisimulate",
        "GITHUB_REPOSITORY_ID": "1330347449",
        "GITHUB_SHA": "b" * 40,
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    objects, requests = {}, []

    def request(url, token, destination, *, upload=False, checksum=""):
        requests.append((url, token, upload))
        if upload:
            assert token == "fixture-write-token"
            objects[url] = destination.read_bytes()
            assert transfer.digest(destination) == checksum
        elif url not in objects:
            raise transfer.HTTPError(404)
        else:
            destination.write_bytes(objects[url])

    monkeypatch.setattr(transfer, "transfer", request)
    return SimpleNamespace(
        source=source,
        output=output,
        manifest=manifest,
        objects=objects,
        requests=requests,
        env=env,
        request=request,
        monkeypatch=monkeypatch,
    )


def download(handoff, *, run="101", attempt="2"):
    handoff.monkeypatch.setenv("STORAGE_MODE", "download")
    handoff.monkeypatch.setenv("STORAGE_PATH", str(handoff.output))
    handoff.monkeypatch.setenv("STORAGE_TOKEN", "fixture-read-token")
    handoff.monkeypatch.setenv("GITHUB_RUN_ID", run)
    handoff.monkeypatch.setenv("GITHUB_RUN_ATTEMPT", attempt)
    transfer.main()


@pytest.mark.parametrize("package", ["aisim", "aisimulate"])
def test_bundle_survives_consumer_only_rerun_and_cross_run(handoff, package):
    files = {
        f"{package}-0.13.0-cp311-abi3-manylinux_2_28_x86_64.whl": b"product",
        "ai_dynamo-1.6.0-py3-none-any.whl": b"adapter",
        "ai_dynamo_runtime-1.6.0.whl": b"runtime",
        "provenance.json": b'{"source":"fixture"}',
        "requirements.txt": b"numpy==2.2.0\n",
    }
    for name, data in files.items():
        (handoff.source / name).write_bytes(data)
    transfer.main()
    manifest = json.loads(handoff.manifest.read_text())
    assert manifest["attempt"] == "1"
    assert manifest["source_sha"] == "a" * 40
    assert manifest["commit"] == "b" * 40
    assert {entry["path"]: entry["size"] for entry in manifest["files"]} == {n: len(b) for n, b in files.items()}
    first = dict(handoff.objects)
    transfer.main()  # Same producer attempt is idempotent and reads back every object.
    assert handoff.objects == first
    assert sum(upload for _, _, upload in handoff.requests) == len(files)
    download(handoff, attempt="7")  # Successful producer was not rerun.
    assert {p.name: p.read_bytes() for p in handoff.output.iterdir()} == files
    handoff.monkeypatch.setenv("GITHUB_SHA", "c" * 40)
    download(handoff, run="202", attempt="3")  # Consumer workflow revision may differ.
    handoff.monkeypatch.setenv("GITHUB_REPOSITORY", "ai-dynamo/aisim")
    download(handoff, run="303")
    assert all("/1330347449/101/1/" in url for url, _, _ in handoff.requests)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha", "c" * 40),
        ("run_id", "404"),
        ("repository_id", "different"),
        ("name", "other"),
        ("architecture", "arm64"),
        ("commit", "d" * 40),
        ("attempt", "../bad"),
        ("schema", 1),
    ],
)
def test_wrong_producer_is_rejected_before_authenticated_download(handoff, field, value):
    (handoff.source / "aisim-0.13.0.whl").write_bytes(b"wheel")
    transfer.main()
    manifest = json.loads(handoff.manifest.read_text())
    manifest[field] = value
    handoff.manifest.write_text(json.dumps(manifest))
    handoff.requests.clear()
    with pytest.raises(ValueError):
        download(handoff)
    assert handoff.requests == []


@pytest.mark.parametrize("change", ["traversal", "object", "digest", "size", "duplicate", "symlink"])
def test_manifest_paths_and_objects_are_validated_before_download(handoff, change):
    (handoff.source / "aisim-0.13.0.whl").write_bytes(b"wheel")
    transfer.main()
    manifest = json.loads(handoff.manifest.read_text())
    entry = manifest["files"][0]
    if change == "traversal":
        entry["path"] = "../aisim.whl"
    elif change == "object":
        entry["object"] = "2/other/aisim.whl"
    elif change == "digest":
        entry["sha256"] = "not-a-digest"
    elif change == "size":
        entry["size"] = True
    elif change == "duplicate":
        manifest["files"].append(dict(entry))
    elif change == "symlink":
        handoff.output.mkdir()
        (handoff.output / entry["path"]).symlink_to(handoff.source / entry["path"])
    handoff.manifest.write_text(json.dumps(manifest))
    handoff.requests.clear()
    with pytest.raises(ValueError):
        download(handoff)
    assert handoff.requests == []


@pytest.mark.parametrize("failure", ["missing", "corrupt", "truncated"])
def test_bad_objects_never_replace_a_previous_wheel(handoff, failure):
    name = "aisim-0.13.0.whl"
    (handoff.source / name).write_bytes(b"wheel")
    transfer.main()
    url = next(iter(handoff.objects))
    if failure == "missing":
        del handoff.objects[url]
    else:
        handoff.objects[url] = b"corru" if failure == "corrupt" else b"w"
    handoff.output.mkdir()
    (handoff.output / name).write_bytes(b"previous verified wheel")
    with pytest.raises((ValueError, transfer.HTTPError)):
        download(handoff)
    assert (handoff.output / name).read_bytes() == b"previous verified wheel"


@pytest.mark.parametrize("status", [401, 403, 500])
def test_failed_upload_does_not_publish_completion_manifest(handoff, status):
    (handoff.source / "aisim-0.13.0.whl").write_bytes(b"wheel")

    def fail(*_args, **_kwargs):
        raise transfer.HTTPError(status)

    handoff.monkeypatch.setattr(transfer, "transfer", fail)
    with pytest.raises(transfer.HTTPError):
        transfer.main()
    assert not handoff.manifest.exists()


def test_partial_bundle_upload_has_no_manifest_and_rerun_recovers(handoff):
    for name in ("aisim-0.13.0.whl", "z-runtime.whl"):
        (handoff.source / name).write_bytes(name.encode())

    def fail_second(url, *args, **kwargs):
        if url.endswith("z-runtime.whl"):
            raise transfer.HTTPError(500)
        return handoff.request(url, *args, **kwargs)

    handoff.monkeypatch.setattr(transfer, "transfer", fail_second)
    with pytest.raises(transfer.HTTPError):
        transfer.main()
    assert len(handoff.objects) == 1 and not handoff.manifest.exists()
    handoff.monkeypatch.setattr(transfer, "transfer", handoff.request)
    transfer.main()
    assert len(json.loads(handoff.manifest.read_text())["files"]) == 2
    assert sum(upload for _, _, upload in handoff.requests) == 2


def test_collect_preserves_multi_path_artifact_layout(tmp_path):
    wheel = tmp_path / "fpe-release-wheel/aisim.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"wheel")
    probes = tmp_path / "fpe-release-probes.json"
    probes.write_bytes(b"{}")
    assert transfer.collect(f"{wheel.parent}\n{probes}") == {
        "fpe-release-wheel/aisim.whl": wheel,
        "fpe-release-probes.json": probes,
    }
    assert transfer.collect(f"{wheel.parent}/*.whl") == {"aisim.whl": wheel}


def test_curl_uses_private_config_without_redirects_or_token_arguments(tmp_path, monkeypatch):
    token = 'fixture-"token\\value'
    payload = tmp_path / "wheel"
    payload.write_bytes(b"wheel")

    def run(command, **kwargs):
        assert token not in str(command)
        assert "--location" not in command and "-L" not in command
        config = Path(command[command.index("--config") + 1])
        assert config.stat().st_mode & 0o077 == 0
        assert 'Authorization: Bearer fixture-\\"token\\\\value' in config.read_text()
        return SimpleNamespace(stdout="200", returncode=0, check_returncode=lambda: None)

    monkeypatch.setattr(transfer.subprocess, "run", run)
    transfer.transfer("https://artifactory.invalid/wheel", token, payload, upload=True, checksum="a" * 64)


def test_wheel_workflows_only_upload_manifests_and_preserve_credentials_boundary():
    found = 0
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        for job in yaml.safe_load(path.read_text()).get("jobs", {}).values():
            for step in job.get("steps", []):
                uses, settings = step.get("uses", ""), step.get("with", {})
                if "upload-artifact@" in uses:
                    assert not any(
                        word in str(settings.get("path", "")) for word in ("*.whl", "nightly-dist/", "fpe-wheel/")
                    )
                if uses.endswith("/wheel-storage"):
                    found += 1
                    assert job["environment"] in ("pr-wheel-staging", "automated-release")
                    assert settings["source-sha"] and settings["architecture"]
                    expected = (
                        "WHEEL_STORAGE_WRITE_TOKEN" if settings["mode"] == "upload" else "WHEEL_STORAGE_READ_TOKEN"
                    )
                    assert settings["token"] == "${{ secrets." + expected + " }}"
                    assert "run_attempt" not in str(settings)
    assert found == 22
    action = yaml.safe_load((ACTION / "action.yml").read_text())
    upload = next(step for step in action["runs"]["steps"] if "upload-artifact@" in step.get("uses", ""))
    assert upload["with"]["name"] == "${{ inputs.name }}-manifest"
    assert upload["with"]["path"].endswith("/wheel-manifest.json")


def test_rerun_producer_keeps_old_objects_and_publishes_a_new_attempt(handoff):
    name = "aisim-0.13.0.whl"
    (handoff.source / name).write_bytes(b"first build")
    transfer.main()
    first_objects = dict(handoff.objects)
    first_manifest = handoff.manifest.read_bytes()
    handoff.monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    (handoff.source / name).write_bytes(b"rebuilt wheel")
    transfer.main()
    assert json.loads(handoff.manifest.read_text())["attempt"] == "2"
    assert all(handoff.objects[url] == payload for url, payload in first_objects.items())
    download(handoff, attempt="3")
    assert (handoff.output / name).read_bytes() == b"rebuilt wheel"
    handoff.manifest.write_bytes(first_manifest)
    download(handoff, attempt="4")
    assert (handoff.output / name).read_bytes() == b"first build"
