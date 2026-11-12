# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trust boundary of the credentialed E2E accuracy job (AISIM-2335).

The private HF read token may only reach steps that run main's code (``trusted/``) in a virtualenv that
never receives the evaluated wheel. Preview and PR-controlled code never shares a step, interpreter or
environment variable with it. The nightly page keeps publishing from default-branch artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
NAMES = ("e2e-accuracy.yml", "e2e-accuracy-branch.yml")
TRUSTED_PYTHON = '"${RUNNER_TEMP}/trusted-venv/bin/python" trusted/'
PIN = re.compile(r"[^@]+@[0-9a-f]{40}( #.*)?$")
ENVIRONMENT = "${{ inputs.preview && 'e2e-accuracy-preview' || 'e2e-accuracy-private' }}"


def load(name: str) -> dict:
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def steps(name: str):
    for job_id, job in load(name)["jobs"].items():
        for step in job.get("steps", []):
            yield job_id, job, step


def references_secret(value) -> bool:
    return "secrets." in json.dumps(value)


@pytest.mark.parametrize("name", NAMES)
def test_token_steps_run_only_main_code_in_the_trusted_environment(name):
    seen = 0
    for job_id, _, step in steps(name):
        if not references_secret(step):
            continue
        seen += 1
        assert "uses" not in step, f"{name}:{job_id}: a secret must not reach an action ({step.get('name')})"
        lines = [ln.strip() for ln in step["run"].splitlines() if ln.strip() and not ln.strip().startswith("#")]
        launches = [ln for ln in lines if "python" in ln]
        assert launches, f"{name}:{job_id}: token step {step.get('name')!r} runs nothing"
        for line in launches:
            assert line.startswith(TRUSTED_PYTHON), f"{name}:{job_id}: {line}"
        assert "${{" not in step["run"], "expressions must go through env, never into the shell text"
        assert not step["env"].keys() - {"HF_TOKEN", "HF_HOME", "AISIM_DATA_CACHE", "GITHUB_TOKEN"}
    assert seen


@pytest.mark.parametrize("name", NAMES)
def test_no_other_step_receives_a_secret_and_secrets_are_not_job_or_workflow_wide(name):
    document = load(name)
    assert "secrets." not in json.dumps(document.get("env", {}))
    for job_id, job, step in steps(name):
        assert "secrets." not in json.dumps(job.get("env", {})), job_id
        if references_secret(step):
            assert step["env"].get("HF_TOKEN", "").startswith("${{ secrets.HF_READ_AISIM_INFX_CACHE"), step
    # The wheel is built from the evaluated revision and so runs PR-controlled build code: no secret, ever.
    wheel = load("e2e-accuracy-branch.yml")["jobs"]["wheel"]
    assert not references_secret(wheel)


def test_credentialed_jobs_declare_the_environment_and_check_out_main():
    for name, job_ids in (
        ("e2e-accuracy.yml", ["resolve"]),
        ("e2e-accuracy-branch.yml", ["measurements", "shards", "campaign", "agentx", "qualify-agentx"]),
    ):
        document = load(name)
        for job_id in job_ids:
            job = document["jobs"][job_id]
            assert job["environment"] == ENVIRONMENT, job_id
            trusted = [s for s in job["steps"] if s.get("with", {}).get("path") == "trusted"]
            assert len(trusted) == 1, job_id
            assert trusted[0]["with"] == {"ref": "main", "path": "trusted", "persist-credentials": "false"}
            # Every token step comes after the trusted checkout and the trusted venv exists.
            names = [s.get("name", s.get("uses")) for s in job["steps"]]
            first_token = next(i for i, s in enumerate(job["steps"]) if references_secret(s))
            assert job["steps"].index(trusted[0]) < first_token, job_id
            venv = next(
                i for i, s in enumerate(job["steps"]) if "trusted-venv" in s.get("run", "") and "venv" in s["run"]
            )
            assert venv < first_token, (job_id, names)


def test_the_trusted_venv_never_sees_the_evaluated_wheel():
    for name in NAMES:
        for job_id, _, step in steps(name):
            run = step.get("run", "")
            if "trusted-venv/bin/pip" in run:
                assert "accuracy-wheel" not in run and "*.whl" not in run
            if "accuracy-wheel" in run and "pip install" in run:
                assert "trusted-venv" not in run, (job_id, step.get("name"))


def test_secrets_flow_only_through_the_default_branch_or_exact_preview_gate():
    document = load("e2e-accuracy.yml")
    assert set(document["on"]) == {"schedule", "workflow_dispatch"}  # never pull_request(_target) or workflow_run
    resolve = document["jobs"]["resolve"]
    assert "github.ref == 'refs/heads/main'" in resolve["if"]
    assert "startsWith(github.ref, 'refs/heads/pull-request/')" in resolve["if"]
    assert "inputs.preview" in resolve["if"]
    branches = document["jobs"]["branches"]
    assert branches["secrets"] == {"HF_READ_AISIM_INFX_CACHE": "${{ secrets.HF_READ_AISIM_INFX_CACHE }}"}
    assert branches["uses"] == "./.github/workflows/e2e-accuracy-branch.yml"
    assert document["permissions"] == {"contents": "read"}


@pytest.mark.parametrize("name", NAMES)
def test_actions_are_sha_pinned_and_checkouts_drop_credentials(name):
    for job_id, _, step in steps(name):
        if "uses" in step and not step["uses"].startswith("./"):  # local actions only in the secret-free wheel job
            assert PIN.match(step["uses"]), (name, job_id, step["uses"])
        if step.get("uses", "").startswith("./"):
            assert job_id == "wheel", (name, job_id)
        if step.get("uses", "").startswith("actions/checkout@"):
            assert step["with"]["persist-credentials"] == "false"


def test_preview_never_publishes_and_the_nightly_page_publishes_from_main_without_secrets():
    branch = load("e2e-accuracy-branch.yml")
    names = [s.get("with", {}).get("name", "") for _, _, s in steps("e2e-accuracy-branch.yml")]
    assert any("e2e-accuracy-preview" in n and "e2e-accuracy-web" in n and "inputs.preview" in n for n in names)
    pages = load("pages.yml")
    assert "secrets." not in json.dumps(pages)
    assert pages["jobs"]["deploy"]["environment"]["name"] == "github-pages"
    assert "github.ref == 'refs/heads/main'" in pages["jobs"]["deploy"]["if"]
    assert "E2E Accuracy Matrix" in pages["on"]["workflow_run"]["workflows"]
    assert "campaign" in branch["jobs"]


def test_the_credentialed_dataset_pin_is_read_from_main_and_names_a_supported_layout():
    for _, _, step in steps("e2e-accuracy.yml"):
        if references_secret(step):
            assert "--manifest trusted/.github/e2e-accuracy-dataset.json" in step["run"]
    pin = json.loads((ROOT / ".github/e2e-accuracy-dataset.json").read_text())
    assert pin.get("layout", "legacy-inferencex") in {"legacy-inferencex", "aisim-e2e/v1"}
    assert re.fullmatch(r"[0-9a-f]{40}", pin["revision"])
