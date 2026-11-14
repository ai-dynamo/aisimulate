# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise immutable wheel bundles and retry/cross-run identity without credentials."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
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
        "STORAGE_SCOPE": "official",
        "STORAGE_PR_NUMBER": "",
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
        ("storage_repo", "wrong-storage-repo"),
        ("name", "other"),
        ("architecture", "arm64"),
        ("commit", "d" * 40),
        ("attempt", "../bad"),
        ("schema", 1),
        ("scope", "pr"),
        ("pr_number", "199"),
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
    expected = {
        ("ci.yml", "application-test-wheel", "upload", "application-test-wheel-${{ matrix.arch }}"),
        ("ci.yml", "application-tests", "download", "application-test-wheel-${{ matrix.arch }}"),
        ("ci.yml", "application-wheel", "download", "application-test-wheel-${{ matrix.arch }}"),
        ("ci.yml", "stage-application-wheel", "download", "application-test-wheel-${{ matrix.arch }}"),
        ("nightly-ci.yml", "build-artifacts", "upload", "nightly-dist-${{ matrix.arch }}"),
        ("fpe-support-matrix.yml", "prepare-wheel", "download", "${{ inputs.wheel_artifact }}"),
        ("fpe-support-matrix.yml", "prepare-wheel", "upload", "fpe-qualified-wheel"),
        ("fpe-support-matrix.yml", "discover-shards", "download", "fpe-qualified-wheel"),
        ("fpe-support-matrix.yml", "generate", "download", "fpe-qualified-wheel"),
        ("fpe-support-matrix.yml", "build-web-matrix", "download", "fpe-qualified-wheel"),
        ("fpe-release-qualify.yml", "prepare", "upload", "fpe-release-wheel-${{ inputs.release }}"),
        ("fpe-release-qualify.yml", "generate", "download", "fpe-release-wheel-${{ inputs.release }}"),
        ("fpe-release-qualify.yml", "qualify", "download", "fpe-release-wheel-${{ inputs.release }}"),
        ("fpm-accuracy-branch.yml", "wheel", "download", "nightly-dist-amd64"),
        ("fpm-accuracy-branch.yml", "wheel", "upload", "fpm-accuracy-wheel-${{ inputs.artifact_key }}"),
        ("fpm-accuracy-branch.yml", "campaign", "download", "fpm-accuracy-wheel-${{ inputs.artifact_key }}"),
        ("e2e-accuracy-branch.yml", "wheel", "download", "nightly-dist-amd64"),
        ("e2e-accuracy-branch.yml", "wheel", "upload", "e2e-accuracy-wheel-${{ inputs.artifact_key }}"),
        ("e2e-accuracy-branch.yml", "shards", "download", "e2e-accuracy-wheel-${{ inputs.artifact_key }}"),
        ("e2e-accuracy-branch.yml", "agentx", "download", "e2e-accuracy-wheel-${{ inputs.artifact_key }}"),
        ("simulation-performance.yml", "build", "upload", "simulation-wheel-${{ matrix.side }}-${{ matrix.sha }}"),
        (
            "simulation-performance.yml",
            "compare",
            "download",
            "simulation-wheel-base-${{ needs.select.outputs.base_sha }}",
        ),
        (
            "simulation-performance.yml",
            "compare",
            "download",
            "simulation-wheel-head-${{ needs.select.outputs.head_sha }}",
        ),
    }
    found = set()
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        for job_name, job in yaml.safe_load(path.read_text()).get("jobs", {}).items():
            for step in job.get("steps", []):
                uses, settings = step.get("uses", ""), step.get("with", {})
                if "upload-artifact@" in uses:
                    assert not any(
                        word in str(settings.get("path", "")) for word in ("*.whl", "nightly-dist/", "fpe-wheel/")
                    )
                if uses.endswith("/wheel-storage"):
                    found.add((path.name, job_name, settings["mode"], settings["name"]))
                    assert "environment" in job
                    assert settings["source-sha"] and settings["architecture"]
                    assert settings["token"] == "${{ secrets.ARTIFACTORY_TOKEN }}"
                    assert "run_attempt" not in str(settings)
    assert found == expected
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


def selected_environment(job, *, ref="refs/heads/main", preview=False, pr_number=""):
    expression = job["environment"]
    if not expression.startswith("${{"):
        return expression
    expression = expression[3:-2].strip()
    expression = expression.replace("github.ref", repr(ref)).replace("inputs.preview", repr(preview))
    expression = expression.replace("inputs.pr_number", repr(pr_number))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    return eval(expression, {"__builtins__": {}, "startsWith": str.startswith})


@pytest.mark.parametrize("workflow", ["ci.yml", "fpe-support-matrix.yml"])
@pytest.mark.parametrize(
    "ref,expected",
    [
        ("refs/heads/main", "automated-release"),
        ("refs/heads/release/0.13.0", "automated-release"),
        ("refs/heads/pull-request/199", "pr-wheel-staging"),
        ("refs/heads/fix-handoff", "pr-wheel-staging"),
    ],
)
def test_ci_and_fpe_handoffs_select_storage_by_workflow_ref(workflow, ref, expected):
    jobs = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())["jobs"]
    for name, job in jobs.items():
        transfers = [s for s in job.get("steps", []) if s.get("uses", "").endswith("/wheel-storage")]
        if not transfers or name == "stage-application-wheel":
            continue
        assert selected_environment(job, ref=ref) == expected
        assert all(s["with"]["repo"] == "${{ vars.WHEEL_STORAGE_REPO }}" for s in transfers)


@pytest.mark.parametrize(
    "workflow,expected",
    [
        ("nightly-ci.yml", "automated-release"),
        ("fpe-release-qualify.yml", "automated-release"),
        ("fpm-accuracy-branch.yml", "automated-release"),
        ("simulation-performance.yml", "pr-wheel-staging"),
    ],
)
def test_official_campaigns_and_pr_simulations_keep_their_storage_scope(workflow, expected):
    jobs = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())["jobs"]
    for job in jobs.values():
        if any(s.get("uses", "").endswith("/wheel-storage") for s in job.get("steps", [])):
            # Simulation can be manually launched from main, but still evaluates a PR.
            assert selected_environment(job) == expected


@pytest.mark.parametrize("preview,expected", [(False, "automated-release"), (True, "pr-wheel-staging")])
def test_accuracy_previews_are_separate_from_official_nightly_reuse(preview, expected):
    jobs = yaml.safe_load((ROOT / ".github/workflows/e2e-accuracy-branch.yml").read_text())["jobs"]
    for job in jobs.values():
        if any(s.get("uses", "").endswith("/wheel-storage") for s in job.get("steps", [])):
            assert selected_environment(job, preview=preview) == expected


@pytest.mark.parametrize("repo", ["sw-dynamo-aisimulate-pr-pypi-local", "sw-dynamo-aisimulate-pypi-local"])
def test_manifest_pins_the_producer_storage_repository(handoff, repo):
    handoff.monkeypatch.setenv("STORAGE_REPO", repo)
    (handoff.source / "aisim-0.13.0.whl").write_bytes(b"wheel")
    transfer.main()
    assert json.loads(handoff.manifest.read_text())["storage_repo"] == repo
    download(handoff, run="202")
    assert all(f"/{repo}/ci-wheels/" in url for url, _, _ in handoff.requests)
    handoff.monkeypatch.setenv("STORAGE_REPO", "sw-dynamo-dev-generic-local")
    handoff.requests.clear()
    with pytest.raises(ValueError, match="producer mismatch"):
        download(handoff, run="202")
    assert not handoff.requests


@pytest.mark.parametrize("scope,number,prefix", [("pr", "199", "pr/199"), ("preview", "", "preview")])
@pytest.mark.parametrize("source", ["a" * 40, "c" * 40])
def test_pr_and_preview_paths_bind_actual_source_and_consumer_context(handoff, scope, number, prefix, source):
    handoff.monkeypatch.setenv("STORAGE_SCOPE", scope)
    handoff.monkeypatch.setenv("STORAGE_PR_NUMBER", number)
    handoff.monkeypatch.setenv("STORAGE_SOURCE_SHA", source)
    (handoff.source / "aisim-0.14.0.whl").write_bytes(b"wheel")
    transfer.main()
    manifest = json.loads(handoff.manifest.read_text())
    assert manifest["scope"] == scope and manifest["pr_number"] == number
    assert manifest["source_sha"] == source and manifest["commit"] == "b" * 40
    assert all(f"/ci-wheels-local/{prefix}/{source}/101/1/" in url for url in handoff.objects)
    assert all("/ci-wheels/" not in url for url in handoff.objects)
    download(handoff, run="202", attempt="7")
    assert (handoff.output / "aisim-0.14.0.whl").read_bytes() == b"wheel"
    handoff.requests.clear()
    # A different PR or non-PR category must fail before authenticated requests.
    handoff.monkeypatch.setenv("STORAGE_PR_NUMBER", "200" if scope == "pr" else "199")
    with pytest.raises(ValueError):
        download(handoff, run="202")
    assert handoff.requests == []


@pytest.mark.parametrize("number", ["", "0", "01", "-1", "199/../200", "not-a-pr"])
def test_invalid_pr_identity_is_rejected_before_authentication(handoff, number):
    handoff.monkeypatch.setenv("STORAGE_SCOPE", "pr")
    handoff.monkeypatch.setenv("STORAGE_PR_NUMBER", number)
    (handoff.source / "aisim.whl").write_bytes(b"wheel")
    with pytest.raises(ValueError, match="explicit positive PR number"):
        transfer.main()
    assert handoff.requests == []


@pytest.mark.parametrize("scope", ["official", "preview"])
def test_non_pr_categories_cannot_carry_a_pr_identity(handoff, scope):
    handoff.monkeypatch.setenv("STORAGE_SCOPE", scope)
    handoff.monkeypatch.setenv("STORAGE_PR_NUMBER", "199")
    with pytest.raises(ValueError, match="cannot carry a PR number"):
        transfer.main()
    assert handoff.requests == []


def test_explicit_pr_number_selects_pr_environment_for_manual_fpe_from_main():
    jobs = yaml.safe_load((ROOT / ".github/workflows/fpe-support-matrix.yml").read_text())["jobs"]
    for job in jobs.values():
        if any(step.get("uses", "").endswith("/wheel-storage") for step in job.get("steps", [])):
            assert selected_environment(job, pr_number="199") == "pr-wheel-staging"


def test_pr_identity_is_propagated_from_verified_ci_and_preview_selectors():
    workflows = ROOT / ".github/workflows"
    ci = yaml.safe_load((workflows / "ci.yml").read_text())["jobs"]
    assert ci["verify-target"]["outputs"]["pr-number"] == "${{ steps.pr-target.outputs.number }}"
    verify = next(step for step in ci["verify-target"]["steps"] if step.get("id") == "pr-target")
    assert '"${pr_head}" != "${RUN_SHA}"' in verify["run"]
    assert "number=%s" in verify["run"]
    assert ci["select-full-ci"]["outputs"]["pr-number"] == "${{ needs.verify-target.outputs.pr-number }}"
    for name in ("application-wheel", "application-test-wheel", "application-tests"):
        for step in ci[name]["steps"]:
            if step.get("uses", "").endswith("/wheel-storage"):
                assert step["with"]["pr-number"] == "${{ needs.select-full-ci.outputs.pr-number }}"
    simulation = yaml.safe_load((workflows / "simulation-performance.yml").read_text())["jobs"]
    for job in simulation.values():
        for step in job.get("steps", []):
            if step.get("uses", "").endswith("/wheel-storage"):
                assert step["with"]["scope"] == "pr"
                assert step["with"]["pr-number"] == "${{ needs.select.outputs.number }}"
    accuracy = (workflows / "e2e-accuracy.yml").read_text()
    assert "pr_number: prNumber" in accuracy
    assert "pr_number: ${{ matrix.pr_number || '' }}" in accuracy


@pytest.mark.parametrize(
    "ref,number,allowed",
    [
        ("refs/heads/pull-request/199", "199", True),
        ("refs/heads/pull-request/199", "", False),
        ("refs/heads/pull-request/199", "200", False),
        ("refs/heads/pull-request/199", "0199", False),
        ("refs/heads/experiment", "", True),
        ("refs/heads/main", "", True),
        ("refs/heads/main", "199", True),
        ("refs/heads/main", "199/../200", False),
    ],
)
def test_standalone_fpe_requires_explicit_matching_copied_pr_identity(tmp_path, ref, number, allowed):
    job = yaml.safe_load((ROOT / ".github/workflows/fpe-support-matrix.yml").read_text())["jobs"]["prepare-wheel"]
    guard = next(step for step in job["steps"] if step.get("name") == "Validate wheel storage context")
    result = subprocess.run(
        ["bash", "-e", "-c", guard["run"]],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={**os.environ, "GITHUB_REF": ref, "PR_NUMBER": number},
    )
    assert (result.returncode == 0) is allowed, result.stderr
    first_transfer = next(i for i, step in enumerate(job["steps"]) if step.get("uses", "").endswith("/wheel-storage"))
    assert job["steps"].index(guard) < first_transfer


@pytest.mark.parametrize("missing", ["missing.json", "distant/missing.json"])
def test_collect_rejects_missing_literal_companion_before_upload(handoff, missing):
    wheel = handoff.source / "aisim.whl"
    wheel.write_bytes(b"wheel")
    handoff.monkeypatch.setenv("STORAGE_PATH", f"{wheel}\n{handoff.source.parent / missing}")
    with pytest.raises(ValueError, match="path does not exist"):
        transfer.main()
    assert not handoff.requests
    assert not handoff.manifest.exists()
    # Without the invalid companion the wheel keeps its original archive root.
    assert transfer.collect(str(wheel)) == {"aisim.whl": wheel}


@pytest.mark.parametrize("code,reason", [(6, "DNS"), (60, "TLS peer certificate"), (28, "timed out")])
def test_curl_transport_diagnostics_do_not_expose_private_stderr(tmp_path, monkeypatch, code, reason):
    token = 'secret-"token\\value'
    escaped = token.replace("\\", "\\\\").replace('"', '\\"')

    def run(command, **kwargs):
        config = command[command.index("--config") + 1]
        return SimpleNamespace(
            stdout="000",
            returncode=code,
            stderr=f"curl: ({code}) diagnostic {token} {escaped} {config} "
            "Authorization: Bearer credentials https://user:password@example.invalid/path?secret=query",
        )

    monkeypatch.setattr(transfer.subprocess, "run", run)
    with pytest.raises(RuntimeError, match=reason) as error:
        transfer.transfer("https://artifactory.invalid/wheel", token, tmp_path / "wheel")
    message = str(error.value)
    assert f"curl exit {code}" in message
    for private in (token, escaped, "wheel-curl-", "credentials", "password", "query"):
        assert private not in message


@pytest.mark.parametrize("status", [404, 401, 403, 500])
def test_curl_http_failure_preserves_status(tmp_path, monkeypatch, status):
    monkeypatch.setattr(
        transfer.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout=str(status), returncode=22),
    )
    with pytest.raises(transfer.HTTPError) as error:
        transfer.transfer("https://artifactory.invalid/wheel", "secret", tmp_path / "wheel")
    assert error.value.status == status


@pytest.mark.parametrize("corrupt", [False, True])
def test_documented_local_release_restore_executes_transfer_and_verifies_bytes(handoff, corrupt):
    files = {
        "fpe-release-wheel/aisim-0.14.0-cp311-abi3-manylinux_2_28_x86_64.whl": b"fixture wheel",
        "fpe-release-wheel/provenance.json": b'{"source_sha":"' + b"a" * 40 + b'"}',
        "fpe-release-shards.json": b"[]",
        "fpe-release-probes.json": b"{}",
    }
    for name, data in files.items():
        path = handoff.source / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
    handoff.monkeypatch.setenv("STORAGE_NAME", "fpe-release-wheel-0.14.0")
    handoff.monkeypatch.setenv("GITHUB_REPOSITORY", "ai-dynamo/aisim")
    transfer.main()
    if corrupt:
        url = next(url for url in handoff.objects if url.endswith(".whl"))
        handoff.objects[url] = b"corrupt wheel"  # Same length: SHA-256 must catch this.
    objects = handoff.source.parent / "objects.json"
    objects.write_text(json.dumps({url: data.hex() for url, data in handoff.objects.items()}))
    manifest = handoff.manifest
    # A local shell has no Actions STORAGE_* or GITHUB_* environment. Run the
    # documented command verbatim with only network endpoints replaced by stubs.
    workspace = handoff.output
    script = workspace / ".github/actions/wheel-storage/transfer.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(ACTION / "transfer.py", script)
    bin_dir = workspace / "bin"
    bin_dir.mkdir()
    stubs = {
        "gh": (
            "import os, pathlib, shutil, sys\n"
            "args = sys.argv[1:]\n"
            "if args[:2] == ['run', 'download']:\n"
            "    assert args == ['run', 'download', '101', '--repo', 'ai-dynamo/aisim',\n"
            "                    '--name', 'fpe-release-wheel-0.14.0-manifest', '--dir', 'release-wheel-manifest']\n"
            "    destination = pathlib.Path(args[-1])\n"
            "    destination.mkdir()\n"
            "    shutil.copyfile(os.environ['FIXTURE_MANIFEST'], destination / 'wheel-manifest.json')\n"
            "else:\n"
            "    assert args == ['api', 'repos/ai-dynamo/aisim', '--jq', '.id']\n"
            "    print('1330347449')\n"
        ),
        "curl": (
            "import json, os, pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "assert '--upload-file' not in args\n"
            "objects = json.loads(pathlib.Path(os.environ['FIXTURE_OBJECTS']).read_text())\n"
            "pathlib.Path(args[args.index('--output') + 1]).write_bytes(bytes.fromhex(objects[args[-1]]))\n"
            "print('200', end='')\n"
        ),
    }
    for name, body in stubs.items():
        executable = bin_dir / name
        executable.write_text(f"#!{sys.executable}\n{body}")
        executable.chmod(0o755)
    (bin_dir / "python3").symlink_to(sys.executable)
    documentation = (ROOT / "docs/ci/accuracy.md").read_text().split("To reproduce a release result,", 1)[1]
    command = re.search(r"```bash\n(.*?)```", documentation, re.DOTALL).group(1)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("GITHUB_", "STORAGE_"))}
    env.update(
        PATH=f"{bin_dir}{os.pathsep}{env['PATH']}",
        FPE_RELEASE="0.14.0",
        FPE_SOURCE_SHA="a" * 40,
        FPE_TOOLING_SHA="b" * 40,
        FPE_RUN_ID="101",
        WHEEL_STORAGE_URL=handoff.env["STORAGE_URL"],
        WHEEL_STORAGE_REPO=handoff.env["STORAGE_REPO"],
        ARTIFACTORY_TOKEN="fixture-read-token",
        FIXTURE_MANIFEST=str(manifest),
        FIXTURE_OBJECTS=str(objects),
    )
    result = subprocess.run(["bash", "-euc", command], cwd=workspace, env=env, capture_output=True, text=True)
    if corrupt:
        assert result.returncode != 0
        assert "downloaded Artifactory file checksum mismatch" in result.stderr
        assert not list((workspace / "fpe-release-wheel").glob("*.whl"))
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert {name: (workspace / name).read_bytes() for name in files} == files


@pytest.mark.parametrize("package", ["aisim", "aisimulate"])
@pytest.mark.parametrize("corrupt", ["aisim-wheel", "dynamo-build.patch"])
def test_complete_simulation_handoff_reaches_final_consumer_and_detects_corruption(handoff, package, corrupt):
    from scripts.performance.simulation_perf_artifact import sha256, verify

    for side, revision in [("base", "a" * 40), ("head", "c" * 40)]:
        source = handoff.source / side
        source.mkdir()
        wheel_name = f"{package}-0.14.0-cp311-abi3-manylinux_2_28_x86_64.whl"
        files = {
            wheel_name: b"product",
            "ai_dynamo_runtime-1.6.0.whl": b"runtime",
            "ai_dynamo-1.6.0-py3-none-any.whl": b"adapter",
            "requirements.txt": b"numpy==2.2.0\n",
            "dynamo-requirements.txt": b"numpy==2.2.0\n",
            "dynamo-build.json": json.dumps({"aisimulate_sha": revision}).encode(),
            "dynamo-build.patch": b"patch",
            "dynamo-Cargo.lock": b"lock",
        }
        for name, data in files.items():
            (source / name).write_bytes(data)
        provenance = {
            "side": side,
            "source_sha": revision,
            "dynamo": {"aisimulate_sha": revision},
            "files": {name: sha256(source / name) for name in files},
        }
        (source / "provenance.json").write_text(json.dumps(provenance))
        for key, value in {
            "STORAGE_MODE": "upload",
            "STORAGE_PATH": str(source),
            "STORAGE_SOURCE_SHA": revision,
            "STORAGE_NAME": f"simulation-wheel-{side}",
            "STORAGE_TOKEN": "fixture-write-token",
        }.items():
            handoff.monkeypatch.setenv(key, value)
        transfer.main()
        handoff.output = source.parent / f"downloaded-{side}"
        download(handoff)
        assert verify(handoff.output, side, revision) == provenance
        damaged = wheel_name if corrupt == "aisim-wheel" else corrupt
        (handoff.output / damaged).write_bytes(b"corrupt after transfer")
        with pytest.raises(ValueError, match="checksum mismatch"):
            verify(handoff.output, side, revision)


@pytest.mark.parametrize(
    "ref,allowed",
    [
        ("refs/heads/main", True),
        ("refs/heads/release/0.14.0", True),
        ("refs/heads/pull-request/199", True),
        ("refs/heads/feature", False),
        ("refs/tags/v0.14.0", False),
        ("refs/pull/199/head", False),
        ("refs/heads/release/", False),
        ("refs/heads/release/../main", False),
        ("refs/heads/pull-request/0", False),
        ("refs/heads/pull-request/0199", False),
        ("refs/heads/pull-request/199/extra", False),
    ],
)
def test_manual_full_ci_rejects_unadmitted_refs_early(ref, allowed):
    # BaseLoader preserves GitHub's `on` and shell text without YAML 1.1 coercion.
    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    script = workflow["jobs"]["verify-target"]["steps"][0]["run"]
    result = subprocess.run(
        ["bash", "-eu", "-c", script],
        env={
            **os.environ,
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": ref,
            "EXPECTED_SHA": "a" * 40,
            "RUN_SHA": "a" * 40,
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == allowed, result.stdout + result.stderr
    if not allowed:
        assert "/ok to test <sha>" in result.stdout
