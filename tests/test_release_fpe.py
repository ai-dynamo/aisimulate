# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release package isolation and producer-to-publisher contract tests."""

import copy
import csv
import importlib.util
import io
import json
import shlex
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / {"run_release_fpe": "fpe", "prepare_fpe_pages": "pages"}[name] / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RELEASE = load_script("run_release_fpe")
PAGES = load_script("prepare_fpe_pages")
SHA = "a" * 40
TOOLING = "b" * 40
WHEEL = "c" * 64
IDENTITY = {"schema_version": 1, "source_branch": "release/0.12.0", "source_sha": SHA, "tooling_sha": TOOLING}


@pytest.fixture
def probe_plan():
    selectors = json.loads((ROOT / ".github/fpe-required-probes.json").read_text())["probes"]
    return {
        **IDENTITY,
        "wheel_sha256": WHEEL,
        "selectors": selectors,
        "probes": [{**selector, "backend_version": "1.0"} for selector in selectors],
    }


@pytest.mark.parametrize("version", ["1.0", "2.0"])
def test_required_probes_follow_selected_source_current_version(probe_plan, version):
    expected = [{**p, "backend_version": version} for p in probe_plan["probes"]]
    combinations = [tuple(p[k] for k in ("model", "system", "backend", "backend_version")) for p in expected]
    # A higher development version is also queryable. The source's current
    # resolver, rather than lexical ordering or report successes, decides.
    combinations += [(m, s, b, "99.0") for m, s, b, _ in combinations]
    resolver = Mock(return_value=version)
    receipt = {**IDENTITY, "wheel_sha256": WHEEL}
    plan = RELEASE.discover_required_probes(combinations, receipt, resolver)
    assert plan == {**receipt, "selectors": probe_plan["selectors"], "probes": expected}
    assert resolver.call_args_list == [call(p["system"], p["backend"], "current") for p in expected]
    assert RELEASE.validate_required_probes(plan, receipt) == expected


def test_required_probe_discovery_rejects_missing_current_version(probe_plan):
    receipt = {**IDENTITY, "wheel_sha256": WHEEL}
    resolver = Mock(side_effect=ValueError("no queryable versions defined"))
    with pytest.raises(ValueError, match="no queryable versions"):
        RELEASE.discover_required_probes([], receipt, resolver)


@pytest.mark.parametrize("missing_index", range(4))
def test_required_probe_discovery_cannot_drop_an_absent_identity(probe_plan, missing_index):
    combinations = [
        tuple(p[k] for k in ("model", "system", "backend", "backend_version")) for p in probe_plan["probes"]
    ]
    combinations.pop(missing_index)
    with pytest.raises(ValueError, match="absent from the selected-source inventory"):
        RELEASE.discover_required_probes(combinations, {**IDENTITY, "wheel_sha256": WHEEL}, Mock(return_value="1.0"))


@pytest.mark.parametrize("field", ["source_sha", "tooling_sha", "source_branch", "schema_version", "wheel_sha256"])
def test_resolved_probe_plan_rejects_mixed_provenance(probe_plan, field):
    probe_plan[field] = "different"
    with pytest.raises(ValueError, match="provenance or policy"):
        RELEASE.validate_required_probes(probe_plan, {**IDENTITY, "wheel_sha256": WHEEL})


@pytest.mark.parametrize("version", [None, "", " ", "current", "previous", "next", True])
def test_resolved_probe_plan_requires_concrete_versions(probe_plan, version):
    probe_plan["probes"][0]["backend_version"] = version
    with pytest.raises(ValueError, match="invalid resolved FPE probe"):
        RELEASE.validate_required_probes(probe_plan, {**IDENTITY, "wheel_sha256": WHEEL})


@pytest.mark.parametrize("change", ["missing", "empty", "duplicate", "wrong_model", "wrong_policy"])
def test_resolved_probe_plan_cannot_change_required_coverage(probe_plan, change):
    if change == "missing":
        probe_plan["probes"].pop()
    elif change == "empty":
        probe_plan["probes"] = []
    elif change == "duplicate":
        probe_plan["probes"][1] = probe_plan["probes"][0]
    elif change == "wrong_model":
        probe_plan["probes"][0]["model"] = "test/unrelated"
    else:
        probe_plan["selectors"][0]["backend_version"] = "previous"
    with pytest.raises(ValueError, match="every selector|invalid resolved|provenance or policy"):
        RELEASE.validate_required_probes(probe_plan, {**IDENTITY, "wheel_sha256": WHEEL})


def test_discovery_automatically_includes_future_releases_and_pins_tips(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True, stderr=subprocess.DEVNULL).strip()

    git("init", "-b", "main")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "first")
    first = git("rev-parse", "HEAD")
    assert RELEASE.list_releases(tmp_path) == []
    git("update-ref", "refs/remotes/origin/release/0.12.0", first)
    git("update-ref", "refs/remotes/origin/feature/skip", first)
    git("update-ref", "refs/heads/release/local-only", first)
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "second")
    second = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/release/0.13.0", second)
    git("update-ref", "refs/remotes/origin/release/0.14.0-rc1", second)
    captured = RELEASE.list_releases(tmp_path)
    assert captured == [
        {"version": "0.12.0", "source_sha": first},
        {"version": "0.13.0", "source_sha": second},
        {"version": "0.14.0-rc1", "source_sha": second},
    ]
    git("update-ref", "refs/remotes/origin/release/0.12.0", second)
    assert captured[0]["source_sha"] == first
    assert RELEASE.list_releases(tmp_path)[0]["source_sha"] == second


@pytest.mark.parametrize("version,sha", [("nested/branch", SHA), ("bad*glob", SHA), ("0.13.0", "not-a-sha")])
def test_discovery_rejects_unsafe_matrix_values(tmp_path, version, sha):
    with (
        patch.object(
            RELEASE.subprocess, "check_output", return_value=f"refs/remotes/origin/release/{version}\0{sha}\n"
        ),
        pytest.raises(ValueError, match="invalid release branch or commit"),
    ):
        RELEASE.list_releases(tmp_path)


def test_discovery_does_not_silently_truncate_releases(tmp_path):
    refs = "".join(f"refs/remotes/origin/release/{n}\0{SHA}\n" for n in range(257))
    with (
        patch.object(RELEASE.subprocess, "check_output", return_value=refs),
        pytest.raises(ValueError, match="matrix limit"),
    ):
        RELEASE.list_releases(tmp_path)


def test_harness_uses_release_inventory_and_current_probe_code(tmp_path):
    source = tmp_path / "release"
    inventory = source / RELEASE.PROBES / "support_matrix.py"
    inventory.parent.mkdir(parents=True)
    inventory.write_text("release inventory\n")
    destination = RELEASE.prepare_harness(source, tmp_path / "harness")
    copied = destination / "tools/support_matrix"
    assert (copied / "support_matrix.py").read_text() == "release inventory\n"
    assert (copied / "fpe_support_matrix.py").read_bytes() == (
        ROOT / RELEASE.PROBES / "fpe_support_matrix.py"
    ).read_bytes()
    assert not (copied / "src").exists()
    with pytest.raises(ValueError, match="already exist"):
        RELEASE.prepare_harness(source, destination)


@pytest.mark.parametrize("branch", ["feature/test", "release/../main", "release/", "release/a/b"])
def test_release_identity_rejects_unsafe_branches(tmp_path, branch):
    with pytest.raises(ValueError, match="release/<version>"):
        RELEASE.identity(tmp_path, SHA, TOOLING, branch)


def test_identity_rejects_wrong_source_and_modified_checkouts(tmp_path):
    with patch.object(RELEASE, "revision", return_value=TOOLING), pytest.raises(ValueError, match="source checkout"):
        RELEASE.identity(tmp_path, SHA, TOOLING, "release/0.12.0")
    with (
        patch.object(RELEASE, "revision", side_effect=[SHA, TOOLING]),
        patch.object(RELEASE.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "git diff")),
        pytest.raises(subprocess.CalledProcessError),
    ):
        RELEASE.identity(tmp_path, SHA, TOOLING, "release/0.12.0")


def test_wheel_requires_exactly_one_package(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        RELEASE.wheel_identity(tmp_path)
    (tmp_path / "aisimulate-one.whl").write_bytes(b"one")
    (tmp_path / "aisimulate-two.whl").write_bytes(b"two")
    with pytest.raises(ValueError, match="exactly one"):
        RELEASE.wheel_identity(tmp_path)


def test_installed_bytes_and_active_imports_must_belong_to_release_wheel(tmp_path):
    names = [
        "aisimulate/__init__.py",
        "aisimulate/_runtime.so",
        "aisimulate_core/__init__.py",
        "aisimulate_core/_native.py",
    ]
    wheel = tmp_path / "aisimulate-test.whl"
    site = tmp_path / "site"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in names:
            archive.writestr(name, "release bytes")
            path = site / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("release bytes")
    dist = SimpleNamespace(files=names, locate_file=lambda p: site / p)

    def find_spec(name):
        path = "aisimulate/_runtime.so" if name == "aisimulate._runtime" else name + "/__init__.py"
        return SimpleNamespace(origin=str(site / path))

    with patch.object(RELEASE.importlib.util, "find_spec", side_effect=find_spec):
        RELEASE.verify_installed_wheel(wheel, distribution=dist)
        (site / names[0]).write_text("different source")
        with pytest.raises(ValueError, match="differs"):
            RELEASE.verify_installed_wheel(wheel, distribution=dist)
        (site / names[0]).write_text("release bytes")
    with (
        patch.object(RELEASE.importlib.util, "find_spec", return_value=SimpleNamespace(origin="/other/__init__.py")),
        pytest.raises(ValueError, match="not imported"),
    ):
        RELEASE.verify_installed_wheel(wheel, distribution=dist)


def reports(root, required):
    groups = defaultdict(list)
    for probe in required:
        row = {
            **probe,
            "architecture": "TestForCausalLM",
            "roles": "agg",
            "source_sha": SHA,
            "source_version": "0.12.0",
            "status": "PASS",
            "latency_ms": 1.0,
            "source": "silicon",
            "tp_size": 1,
            "pp_size": 1,
            "attention_dp_size": 1,
            "moe_tp_size": 1,
            "moe_ep_size": 1,
            "cp_size": 1,
            "gemm_quant_mode": "half",
            "moe_quant_mode": "half",
            "kvcache_quant_mode": "half",
            "fmha_quant_mode": "half",
            "comm_quant_mode": "half",
            "nextn": 0,
            "attention_backend": None,
        }
        groups[(probe["system"], probe["backend"])].extend(
            {**row, "phase": phase} for phase in ["prefill", "decode_start", "decode_end", "mixed"]
        )
    for (system, backend), rows in groups.items():
        path = root / system / backend / "fpe_support_matrix.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "schema_version": 1,
                        "source_sha": SHA,
                        "source_version": "0.12.0",
                        "wheel_sha256": WHEEL,
                        "workload": {"isl": 256},
                        "plan_count": len(rows) // 4,
                    },
                    "results": rows,
                }
            )
        )
    return [{"system": s, "backend": b} for s, b in sorted(groups)]


def test_complete_release_reports_produce_publishable_ci_artifact(tmp_path, probe_plan):
    source = tmp_path / "reports"
    shards = reports(source, probe_plan["probes"])
    destination = tmp_path / "web"
    RELEASE.package_reports(source, IDENTITY, WHEEL, shards, destination, probe_plan=probe_plan)
    qualification = json.loads((destination / "fpe-qualification.json").read_text())
    assert qualification["source_sha"] == SHA
    assert qualification["tooling_sha"] == TOOLING
    assert qualification["source_branch"] == "release/0.12.0"
    assert qualification["required_probe_count"] == 4
    assert qualification["required_probes"] == probe_plan["probes"]
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for path in destination.rglob("*"):
            if path.is_file():
                archive.writestr(str(path.relative_to(destination)), path.read_bytes())
    selected = PAGES.qualified_files(output.getvalue(), SHA)
    assert selected is not None
    rows = list(csv.DictReader(io.StringIO(selected["b200_sxm.csv"].decode())))
    assert len(rows) == 4
    assert all(
        shlex.split(row["Command"])[3:6]
        == ["release-source/python/aisimulate/.venv/bin/python", "scripts/fpe/run_release_fpe.py", "probe"]
        for row in rows
    )
    assert all(SHA in row["Command"] and TOOLING in row["Command"] for row in rows)
    assert all(row["SourceSHA"] == SHA for row in rows)


def test_incomplete_release_cannot_produce_a_web_artifact(tmp_path, probe_plan):
    source = tmp_path / "reports"
    shards = reports(source, probe_plan["probes"])
    next(source.rglob("fpe_support_matrix.json")).unlink()
    destination = tmp_path / "web"
    with pytest.raises(ValueError, match="missing shards"):
        RELEASE.package_reports(source, IDENTITY, WHEEL, shards, destination, probe_plan=probe_plan)
    assert not destination.exists()


def test_passing_previous_version_cannot_replace_a_failed_current_probe(tmp_path, probe_plan):
    source = tmp_path / "reports"
    shards = reports(source, probe_plan["probes"])
    for path in source.rglob("fpe_support_matrix.json"):
        payload = json.loads(path.read_text())
        previous_rows = [{**row, "backend_version": "0.9"} for row in payload["results"]]
        for row in payload["results"]:
            row.update(status="PERF_DATA_MISSING", latency_ms=None)
        payload["results"].extend(previous_rows)
        payload["metadata"]["plan_count"] *= 2
        path.write_text(json.dumps(payload))
    destination = tmp_path / "web"
    with pytest.raises(ValueError, match="required known-good probes no longer pass"):
        RELEASE.package_reports(source, IDENTITY, WHEEL, shards, destination, probe_plan=probe_plan)
    assert not destination.exists()


def test_packaging_rejects_another_wheels_probe_plan_before_creating_output(tmp_path, probe_plan):
    source = tmp_path / "reports"
    shards = reports(source, probe_plan["probes"])
    wrong_plan = copy.deepcopy(probe_plan)
    wrong_plan["wheel_sha256"] = "d" * 64
    destination = tmp_path / "web"
    with pytest.raises(ValueError, match="provenance or policy"):
        RELEASE.package_reports(source, IDENTITY, WHEEL, shards, destination, probe_plan=wrong_plan)
    assert not destination.exists()


def test_discovery_plan_round_trips_through_packaging_cli(tmp_path, monkeypatch, probe_plan):
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    receipt = {**IDENTITY, "wheel_sha256": WHEEL}
    (wheel_dir / "provenance.json").write_text(json.dumps(receipt))
    report_dir = tmp_path / "reports"
    reports(report_dir, probe_plan["probes"])
    combinations = [
        tuple(p[k] for k in ("model", "system", "backend", "backend_version")) for p in probe_plan["probes"]
    ]
    verified = Mock()

    def resolve_version(system, backend, alias):
        verified.assert_called_once()
        assert alias == "current"
        return "1.0"

    # The source harness exports its own SDK (including older release package
    # names); discovery must resolve through that module after wheel validation.
    selected_source = SimpleNamespace(
        SupportMatrix=lambda: SimpleNamespace(generate_combinations=lambda: combinations),
        perf_database=SimpleNamespace(resolve_query_version=resolve_version),
    )
    monkeypatch.setitem(sys.modules, "tools.support_matrix.support_matrix", selected_source)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(RELEASE, "identity", Mock(return_value=IDENTITY))
    monkeypatch.setattr(RELEASE, "wheel_identity", Mock(return_value=(wheel_dir / "test.whl", WHEEL)))
    monkeypatch.setattr(RELEASE, "verify_installed_wheel", verified)
    monkeypatch.setattr(RELEASE, "prepare_harness", Mock(return_value=tmp_path / "harness"))
    for name, value in (("FPE_SOURCE_SHA", SHA), ("FPE_TOOLING_SHA", TOOLING), ("FPE_BRANCH", "release/0.12.0")):
        monkeypatch.setenv(name, value)
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    destination = tmp_path / "web"
    plan_path = tmp_path / "resolved-probes.json"
    arguments = [
        "--wheel-dir",
        str(wheel_dir),
        "--shards",
        str(tmp_path / "shards.json"),
        "--required-probes",
        str(plan_path),
        "--reports",
        str(report_dir),
        "--destination",
        str(destination),
    ]
    monkeypatch.setattr(sys, "argv", ["run_release_fpe.py", "discover", *arguments])
    RELEASE.main()
    assert json.loads(plan_path.read_text()) == probe_plan
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert json.loads(outputs["required_probes"]) == probe_plan

    # The packaging job has no installed SDK; it must consume the captured plan.
    monkeypatch.delitem(sys.modules, "tools.support_matrix.support_matrix")
    verified.reset_mock()
    monkeypatch.setattr(sys, "argv", ["run_release_fpe.py", "package", *arguments])
    RELEASE.main()
    verified.assert_not_called()
    result = json.loads((destination / "fpe-qualification.json").read_text())
    assert result["required_probes"] == probe_plan["probes"]


@pytest.mark.parametrize("branch", ["main", "release/0.12.0"])
def test_current_harness_keeps_selected_source_and_tooling_identities_distinct(tmp_path, branch):
    with patch.object(RELEASE, "revision", side_effect=[SHA, TOOLING]), patch.object(RELEASE.subprocess, "run"):
        assert RELEASE.identity(tmp_path, SHA, TOOLING, branch) == {
            "schema_version": 1,
            "source_branch": branch,
            "source_sha": SHA,
            "tooling_sha": TOOLING,
        }
