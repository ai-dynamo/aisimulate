# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Negative rollout cases for the read-only effective-rule verifier."""

import copy
import json
import subprocess

import pytest

from scripts import check_required_main_checks as checker


def _rules():
    payload = json.loads(checker.PAYLOAD.read_text())
    return [
        {**payload["rules"][0], "ruleset_id": 42},
        {
            "type": "pull_request",
            "parameters": {
                "required_approving_review_count": 1,
                "require_code_owner_review": True,
                "required_review_thread_resolution": True,
            },
        },
        {"type": "deletion"},
        {"type": "non_fast_forward"},
    ]


def _inspect(rules, *, detail=None, changed_rules=None, changed_detail=None):
    calls = []

    def api(endpoint):
        calls.append(endpoint)
        if endpoint.endswith("/rulesets/42"):
            if changed_detail is not None and calls.count(endpoint) > 1:
                return [changed_detail]
            return [detail if detail is not None else {"id": 42, "enforcement": "active", "bypass_actors": []}]
        if "/rules/branches/" in endpoint:
            if changed_rules is not None and calls.count(endpoint) > 1:
                return [changed_rules]
            # Exercise aggregation across result pages.
            return [rules[:1], rules[1:]]
        return [{"default_branch": "main"}]

    return checker.inspect_repository("owner/repo", api=api)


def test_activated_configuration_preserves_checks_and_human_protections():
    report = _inspect(_rules())
    assert report["configuration_verified"]
    assert report["errors"] == []
    assert report["expected_checks"] == [
        {"context": "Fast CI Success", "integration_id": 15368},
        {"context": "Full CI Success", "integration_id": 15368},
        {"context": "codeowners", "integration_id": 15368},
    ]
    assert report["effective_rules"] == _rules()
    assert report["ci_rulesets"][0]["id"] == 42


@pytest.mark.parametrize("index", range(3))
def test_each_missing_status_fails(index):
    rules = _rules()
    missing = rules[0]["parameters"]["required_status_checks"].pop(index)
    report = _inspect(rules)
    assert not report["configuration_verified"]
    assert any(missing["context"] in error for error in report["errors"])


@pytest.mark.parametrize("integration_id", [None, 0, 123, "15368"])
def test_wrong_or_unbound_application_fails(integration_id):
    rules = _rules()
    rules[0]["parameters"]["required_status_checks"][0]["integration_id"] = integration_id
    assert not _inspect(rules)["configuration_verified"]


@pytest.mark.parametrize("parameter", ["strict_required_status_checks_policy", "do_not_enforce_on_create"])
def test_weakened_ci_policy_fails(parameter):
    rules = _rules()
    rules[0]["parameters"][parameter] = not rules[0]["parameters"][parameter]
    assert not _inspect(rules)["configuration_verified"]


@pytest.mark.parametrize("index", range(4))
def test_missing_ci_review_or_branch_protection_fails(index):
    rules = _rules()
    rules.pop(index)
    assert not _inspect(rules)["configuration_verified"]


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("required_approving_review_count", 0),
        ("require_code_owner_review", False),
        ("required_review_thread_resolution", False),
    ],
)
def test_weakened_review_policy_fails(parameter, value):
    rules = _rules()
    rules[1]["parameters"][parameter] = value
    assert not _inspect(rules)["configuration_verified"]


@pytest.mark.parametrize("bypasses", [None, [{"actor_type": "OrganizationAdmin", "bypass_mode": "always"}]])
def test_hidden_or_configured_ci_bypasses_fail(bypasses):
    detail = {"id": 42, "enforcement": "active", "bypass_actors": bypasses}
    assert not _inspect(_rules(), detail=detail)["configuration_verified"]


def test_disabled_source_or_rules_changed_during_inspection_fails():
    detail = {"id": 42, "enforcement": "disabled", "bypass_actors": []}
    assert not _inspect(_rules(), detail=detail)["configuration_verified"]
    assert not _inspect(_rules(), changed_rules=[])["configuration_verified"]


def test_new_bypass_actor_fails_even_when_effective_rules_do_not_change():
    after = {
        "id": 42,
        "enforcement": "active",
        "bypass_actors": [{"actor_type": "OrganizationAdmin", "bypass_mode": "always"}],
    }
    report = _inspect(_rules(), changed_detail=after)
    assert not report["configuration_verified"]
    assert report["errors"] == ["CI ruleset 42 changed during inspection; rerun the verifier"]


@pytest.mark.parametrize("response", [[], [None], [{}], [[None]], [{"default_branch": "other"}]])
def test_missing_or_malformed_api_evidence_fails(response):
    report = checker.inspect_repository("owner/repo", api=lambda endpoint: response)
    assert not report["configuration_verified"]
    assert report["errors"]


def test_api_failure_preserves_negative_evidence():
    def api(endpoint):
        raise subprocess.CalledProcessError(1, ["gh", "api", endpoint])

    report = checker.inspect_repository("owner/repo", api=api)
    assert not report["configuration_verified"]
    assert "Cannot verify configuration" in report["errors"][0]


def test_github_api_uses_only_paginated_reads(monkeypatch):
    def run(command, **kwargs):
        assert command == ["gh", "api", "--method", "GET", "--paginate", "--slurp", "repos/owner/repo"]
        assert kwargs["check"] and kwargs["timeout"] == 60
        return subprocess.CompletedProcess(command, 0, stdout='[[{"type": "deletion"}], []]')

    monkeypatch.setattr(subprocess, "run", run)
    assert checker.github_api("repos/owner/repo") == [[{"type": "deletion"}], []]


@pytest.mark.parametrize("success", [True, False])
def test_cli_exit_code_and_saved_evidence(monkeypatch, tmp_path, capsys, success):
    report = _inspect(_rules() if success else [])
    output = tmp_path / "rules.json"
    repositories = []

    def inspect(repository):
        repositories.append(repository)
        return copy.deepcopy(report)

    monkeypatch.setattr(checker, "inspect_repository", inspect)
    assert checker.main(["--repository", "owner/repo", "--output", str(output)]) == (0 if success else 1)
    assert repositories == ["owner/repo"]
    assert json.loads(output.read_text()) == report
    assert json.loads(capsys.readouterr().out) == report
