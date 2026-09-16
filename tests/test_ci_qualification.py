# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.check_prediction_numerics import check_results, validate_cases

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
