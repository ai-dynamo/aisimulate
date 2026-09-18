# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import sync_required_main_checks as rules_sync
from scripts.check_prediction_numerics import check_results, resolve_baseline, validate_cases

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA = json.loads((ROOT / ".github/prediction-numerical-sentinels.json").read_text())["baseline_source_sha"]


@pytest.fixture
def case():
    return {
        "id": "dense-prefill",
        "method": "predict_prefill_latency",
        "expected_ms": 10.0,
        "rtol": 0.02,
        "atol_ms": 0.0001,
    }


def test_small_roundoff_passes_and_large_numerical_change_fails(case):
    assert not check_results([case], [{"id": case["id"], "status": "PASS", "latency_ms": 10.01}])
    assert check_results([case], [{"id": case["id"], "status": "PASS", "latency_ms": 100.0}])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1, True, None])
def test_nonfinite_or_invalid_prediction_cannot_pass(case, value):
    assert check_results([case], [{"id": case["id"], "status": "PASS", "latency_ms": value}])


def test_missing_duplicate_and_skipped_sentinels_fail(case):
    row = {"id": case["id"], "status": "PASS", "latency_ms": 10.0}
    assert check_results([case], [])
    assert check_results([case], [row, row])
    assert check_results([case], [{**row, "status": "SKIP"}])


@pytest.mark.parametrize(
    "field,value", [("rtol", 10), ("atol_ms", float("inf")), ("expected_ms", float("nan")), ("method", "from_spec")]
)
def test_invalid_tolerances_or_query_rejected(case, field, value):
    case[field] = value
    with pytest.raises(ValueError):
        validate_cases({"schema_version": 1, "baseline_source_sha": BASELINE_SHA, "cases": [case]})


@pytest.mark.parametrize("baseline", [None, "", "main", "a" * 39, "z" * 40, "0" * 40])
def test_invalid_or_unresolved_baseline_commit_fails(case, baseline):
    with pytest.raises(ValueError, match="baseline_source_sha"):
        validate_cases({"schema_version": 1, "baseline_source_sha": baseline, "cases": [case]})


def test_valid_baseline_commit_is_accepted(case):
    assert validate_cases({"schema_version": 1, "baseline_source_sha": BASELINE_SHA, "cases": [case]}) == [case]


def test_fetch_historical_baseline_preserves_checkout_and_works_offline_afterward(tmp_path):
    upstream = tmp_path / "upstream"
    checkout = tmp_path / "checkout"
    upstream.mkdir()

    def git(root, *args):
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()

    git(upstream, "init", "-q", "-b", "main")
    git(upstream, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-qm", "main")
    git(upstream, "checkout", "-qb", "historical-baseline")
    git(
        upstream,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        "baseline",
    )
    baseline = git(upstream, "rev-parse", "HEAD")
    git(upstream, "checkout", "main")
    git(tmp_path, "clone", "--depth=1", "--single-branch", upstream.as_uri(), str(checkout))
    head = git(checkout, "rev-parse", "HEAD")
    manifest = {"baseline_source_sha": baseline}
    with pytest.raises(ValueError, match="does not resolve"):
        resolve_baseline(manifest, repository_root=checkout)
    assert resolve_baseline(manifest, fetch=True, repository_root=checkout) == baseline
    assert git(checkout, "rev-parse", "HEAD") == head
    assert git(checkout, "branch", "--show-current") == "main"
    git(checkout, "remote", "remove", "origin")
    assert resolve_baseline(manifest, fetch=True, repository_root=checkout) == baseline
    assert manifest == {"baseline_source_sha": baseline}


@pytest.mark.parametrize("baseline", [None, "main", "--upload-pack=invalid", "a" * 39, "z" * 40])
def test_fetch_baseline_rejects_non_sha_before_git(tmp_path, baseline):
    with pytest.raises(ValueError, match="full commit SHA"):
        resolve_baseline({"baseline_source_sha": baseline}, fetch=True, repository_root=tmp_path)


def test_fetch_baseline_fails_when_origin_cannot_supply_commit(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(tmp_path / "missing")], cwd=tmp_path, check=True)
    with pytest.raises(subprocess.CalledProcessError):
        resolve_baseline({"baseline_source_sha": "0" * 40}, fetch=True, repository_root=tmp_path)


def test_fetch_baseline_cli_does_not_require_prediction_cases_or_write_results(tmp_path):
    baseline = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"baseline_source_sha": baseline}))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_prediction_numerics.py"),
            "--manifest",
            str(manifest),
            "--fetch-baseline-only",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"Numerical baseline available: {baseline}"
    assert set(tmp_path.iterdir()) == {manifest}


@pytest.mark.parametrize("arguments", [[], ["--fetch-baseline-only", "--output", "results.json"]])
def test_numerical_cli_requires_exactly_one_mode(tmp_path, arguments):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_prediction_numerics.py"), *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--fetch-baseline-only" in result.stderr
    assert "--output" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("base", ["", "runner:latest", "runner:2.0", "runner@sha256:abc", "runner@sha256:" + "x" * 64])
def test_image_builder_rejects_unpinned_base_before_docker(tmp_path, base):
    result, log = _build_image(tmp_path, base)
    assert result.returncode == 2
    assert log == ""


def test_image_builder_preserves_digest_and_builds_both_architectures(tmp_path):
    base = "registry.example:5000/runner@sha256:" + "a" * 64
    result, log = _build_image(tmp_path, base)
    assert result.returncode == 0, result.stderr
    assert f"BASE_IMAGE={base}" in log.splitlines()
    assert "linux/amd64,linux/arm64" in log.splitlines()


def _build_image(tmp_path, base):
    binary = tmp_path / "docker"
    log = tmp_path / "docker-calls"
    binary.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$AUDIT_DOCKER_LOG"\n')
    binary.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", str(ROOT / "scripts/build_ci_image.sh")],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "AISIM_BASE_IMAGE_BY_DIGEST": base,
            "AISIM_BUILD_IMAGE_TAG": "registry.example/aisim-test:ci",
            "AUDIT_DOCKER_LOG": str(log),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, log.read_text() if log.exists() else ""


def test_manifest_retains_dense_moe_prefill_and_decode():
    manifest = json.loads((ROOT / ".github/prediction-numerical-sentinels.json").read_text())
    cases = validate_cases(manifest)
    assert len(cases) == 8
    assert {(c["compile"]["model_path"], c["method"], c["arguments"]["isl"]) for c in cases} == {
        (model, method, isl)
        for model in ("Qwen/Qwen3-32B", "MiniMaxAI/MiniMax-M2.5")
        for method in ("predict_prefill_latency", "predict_decode_latency")
        for isl in (1024, 8192)
    }
    duplicated = copy.deepcopy(manifest)
    duplicated["cases"].append(duplicated["cases"][0])
    with pytest.raises(ValueError, match="unique"):
        validate_cases(duplicated)


def _setup(tmp_path, *, failures: int, preinstalled: bool = False):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "calls"
    for name, body in {
        "id": "echo 0",
        "rm": 'echo cleanup >> "$AUDIT_LOG"',
        "sleep": 'echo retry >> "$AUDIT_LOG"',
        "apt-get": """
echo "$*" >> "$AUDIT_LOG"
if [[ "$*" == *update ]]; then
  count=0
  [[ ! -f "$AUDIT_COUNT" ]] || count=$(/bin/cat "$AUDIT_COUNT")
  count=$((count + 1))
  echo "$count" > "$AUDIT_COUNT"
  ((count > AUDIT_FAILURES)) || exit 100
else
  for name in cc c++ make; do /bin/ln -s /usr/bin/true "$PATH/$name"; done
fi
""",
    }.items():
        path = binaries / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    if preinstalled:
        for name in ("cc", "c++", "make"):
            (binaries / name).symlink_to("/usr/bin/true")
    result = subprocess.run(
        ["/bin/bash", str(ROOT / "scripts/ci_install_build_tools.sh")],
        env={
            **os.environ,
            "PATH": str(binaries),
            "AUDIT_LOG": str(log),
            "AUDIT_COUNT": str(tmp_path / "count"),
            "AUDIT_FAILURES": str(failures),
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    return result, log.read_text() if log.exists() else ""


def test_preinstalled_tools_need_no_network(tmp_path):
    result, log = _setup(tmp_path, failures=99, preinstalled=True)
    assert result.returncode == 0, result.stderr
    assert log == ""


def test_transient_apt_failure_refetches_and_recovers(tmp_path):
    result, log = _setup(tmp_path, failures=1)
    assert result.returncode == 0, result.stderr
    assert log.count("update") == 2
    assert log.count("cleanup") == 1
    assert "--allow-unauthenticated" not in log


def test_permanent_apt_failure_is_bounded_and_red(tmp_path):
    result, log = _setup(tmp_path, failures=99)
    assert result.returncode != 0
    assert log.count("update") == 3
    assert log.count("retry") == 2


def test_required_main_checks_are_additive_and_bound_to_actions():
    ruleset = json.loads((ROOT / ".github/required-main-checks.json").read_text())
    assert ruleset["conditions"]["ref_name"] == {"include": ["~DEFAULT_BRANCH"], "exclude": []}
    checks = ruleset["rules"][0]["parameters"]
    assert checks["strict_required_status_checks_policy"] is True
    assert {c["context"] for c in checks["required_status_checks"]} == {
        "Fast CI Success",
        "Full CI Success",
        "codeowners",
    }
    assert all(c["integration_id"] == 15368 for c in checks["required_status_checks"])


@pytest.fixture
def ruleset_sync():
    desired = json.loads(rules_sync.PAYLOAD.read_text())
    current = {**copy.deepcopy(desired), "id": 42, "source_type": "Repository", "source": "owner/repo"}
    calls = []

    def api(endpoint, *, payload=None):
        calls.append((endpoint, payload))
        if endpoint == "repos/owner/repo":
            return {"default_branch": "main"}
        assert endpoint == "repos/owner/repo/rulesets/42"
        if payload is not None:
            current.update(copy.deepcopy(payload))
        return copy.deepcopy(current)

    return desired, current, calls, api


def test_ruleset_sync_defaults_to_reads_and_reports_drift(ruleset_sync, capsys):
    desired, current, calls, api = ruleset_sync
    current["rules"][0]["parameters"]["required_status_checks"].pop()
    assert rules_sync.synchronize("owner/repo", 42, desired, api=api) == 1
    assert all(payload is None for _, payload in calls)
    assert "codeowners" in capsys.readouterr().out


def test_ruleset_sync_ignores_order_and_never_writes_an_unchanged_rule(ruleset_sync):
    desired, current, calls, api = ruleset_sync
    current["rules"][0]["parameters"]["required_status_checks"].reverse()
    assert rules_sync.synchronize("owner/repo", 42, desired, apply=True, api=api) == 0
    assert all(payload is None for _, payload in calls)


def test_ruleset_sync_applies_exact_file_and_preserves_backup(ruleset_sync, tmp_path):
    desired, current, calls, api = ruleset_sync
    current["rules"][0]["parameters"]["strict_required_status_checks_policy"] = False
    before = copy.deepcopy(current)
    backup = tmp_path / "before.json"
    assert rules_sync.synchronize("owner/repo", 42, desired, apply=True, backup=backup, api=api) == 0
    assert json.loads(backup.read_text()) == before
    writes = [(endpoint, payload) for endpoint, payload in calls if payload is not None]
    assert writes == [("repos/owner/repo/rulesets/42", desired)]
    assert rules_sync.normalized(current) == rules_sync.normalized(desired)
    assert calls[-1] == ("repos/owner/repo/rulesets/42", None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", 43),
        ("source_type", "Organization"),
        ("source", "other/repo"),
        ("name", "Human review"),
        ("target", "tag"),
        ("conditions", {"ref_name": {"include": ["refs/heads/release"], "exclude": []}}),
        ("rules", [{"type": "pull_request"}]),
        ("bypass_actors", [{"actor_type": "OrganizationAdmin", "bypass_mode": "always"}]),
    ],
)
def test_ruleset_sync_refuses_unrelated_or_bypassed_targets(ruleset_sync, tmp_path, field, value):
    desired, current, calls, api = ruleset_sync
    current[field] = value
    with pytest.raises(ValueError):
        rules_sync.synchronize("owner/repo", 42, desired, apply=True, backup=tmp_path / "before.json", api=api)
    assert all(payload is None for _, payload in calls)
    assert not list(tmp_path.iterdir())


def test_ruleset_sync_cannot_treat_hidden_bypasses_as_empty(ruleset_sync):
    desired, current, calls, api = ruleset_sync
    del current["bypass_actors"]
    with pytest.raises(ValueError, match="hides bypass settings"):
        rules_sync.synchronize("owner/repo", 42, desired, api=api)
    assert all(payload is None for _, payload in calls)


def test_ruleset_sync_preserves_unknown_rule_parameters(ruleset_sync):
    desired, current, calls, api = ruleset_sync
    current["rules"][0]["parameters"]["future_requirement"] = True
    with pytest.raises(ValueError, match="mixed-purpose"):
        rules_sync.synchronize("owner/repo", 42, desired, apply=True, api=api)
    assert all(payload is None for _, payload in calls)


@pytest.mark.parametrize("backup_exists", [False, True])
def test_ruleset_sync_requires_a_new_backup_before_writing(ruleset_sync, tmp_path, backup_exists):
    desired, current, calls, api = ruleset_sync
    current["enforcement"] = "evaluate"
    backup = tmp_path / "before.json" if backup_exists else None
    if backup:
        backup.write_text("prior evidence")
    with pytest.raises((ValueError, FileExistsError)):
        rules_sync.synchronize("owner/repo", 42, desired, apply=True, backup=backup, api=api)
    assert all(payload is None for _, payload in calls)
    if backup:
        assert backup.read_text() == "prior evidence"


@pytest.mark.parametrize("change", ["ruleset", "default_branch"])
def test_ruleset_sync_refuses_concurrent_changes(ruleset_sync, tmp_path, change):
    desired, current, calls, api = ruleset_sync
    current["enforcement"] = "evaluate"

    def changing_api(endpoint, **kwargs):
        response = api(endpoint, **kwargs)
        if calls.count((endpoint, None)) > 1:
            if change == "ruleset" and endpoint.endswith("/42"):
                response["enforcement"] = "disabled"
            if change == "default_branch" and endpoint == "repos/owner/repo":
                response["default_branch"] = "release"
        return response

    with pytest.raises(ValueError, match="changed during inspection"):
        rules_sync.synchronize("owner/repo", 42, desired, apply=True, backup=tmp_path / "before.json", api=changing_api)
    assert all(payload is None for _, payload in calls)


@pytest.mark.parametrize("failure", ["write_denied", "readback_mismatch"])
def test_ruleset_sync_reports_uncertain_updates_and_keeps_backup(ruleset_sync, tmp_path, failure):
    desired, current, calls, api = ruleset_sync
    current["enforcement"] = "evaluate"

    def failed_api(endpoint, *, payload=None):
        if payload is not None:
            if failure == "write_denied":
                raise subprocess.CalledProcessError(1, ["gh", "api"])
            return {}  # Simulate a write that did not take effect.
        return api(endpoint)

    backup = tmp_path / "before.json"
    with pytest.raises(ValueError, match="settings may already have changed"):
        rules_sync.synchronize("owner/repo", 42, desired, apply=True, backup=backup, api=failed_api)
    assert json.loads(backup.read_text())["enforcement"] == "evaluate"


@pytest.mark.parametrize("invalid", ["bypass", "non_strict", "wrong_app", "duplicate", "empty", "extra_rule"])
def test_ruleset_sync_rejects_invalid_source_before_any_api_call(ruleset_sync, invalid):
    desired, _, calls, api = ruleset_sync
    parameters = desired["rules"][0]["parameters"]
    checks = parameters["required_status_checks"]
    if invalid == "bypass":
        desired["bypass_actors"] = [{"actor_type": "OrganizationAdmin"}]
    elif invalid == "non_strict":
        parameters["strict_required_status_checks_policy"] = False
    elif invalid == "wrong_app":
        checks[0]["integration_id"] = 1
    elif invalid == "duplicate":
        checks.append(checks[0])
    elif invalid == "empty":
        checks.clear()
    else:
        desired["rules"].append({"type": "pull_request"})
    with pytest.raises(ValueError):
        rules_sync.synchronize("owner/repo", 42, desired, apply=True, api=api)
    assert calls == []


@pytest.mark.parametrize("payload", [None, {"rules": []}])
def test_ruleset_sync_transport_only_puts_when_given_payload(monkeypatch, payload):
    def run(command, **kwargs):
        assert command[:6] == ["gh", "api", "--hostname", "github.com", "--method", "GET" if payload is None else "PUT"]
        assert kwargs["check"] and kwargs["timeout"] == 60
        if payload is None:
            assert "--input" not in command and kwargs["input"] is None
        else:
            assert command[-2:] == ["--input", "-"]
            assert json.loads(kwargs["input"]) == payload
        return subprocess.CompletedProcess(command, 0, stdout="{}")

    monkeypatch.setattr(subprocess, "run", run)
    assert rules_sync.github_api("repos/owner/repo/rulesets/42", payload=payload) == {}
