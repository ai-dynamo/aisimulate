# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release package isolation and producer-to-publisher contract tests."""

import csv
import importlib.util
import io
import json
import shlex
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RELEASE = load_script("run_release_fpe")
PAGES = load_script("prepare_fpe_pages")
SHA = "a" * 40
TOOLING = "b" * 40
WHEEL = "c" * 64
IDENTITY = {"schema_version": 1, "source_branch": "release/0.12.0", "source_sha": SHA, "tooling_sha": TOOLING}


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


@pytest.mark.parametrize("branch", ["main", "release/../main", "release/", "release/a/b"])
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
        "aiconfigurator/__init__.py",
        "aiconfigurator_core/__init__.py",
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


def reports(root):
    required = json.loads((ROOT / ".github/fpe-required-probes.json").read_text())["probes"]
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


def test_complete_release_reports_produce_publishable_ci_artifact(tmp_path):
    source = tmp_path / "reports"
    shards = reports(source)
    destination = tmp_path / "web"
    RELEASE.package_reports(source, IDENTITY, WHEEL, shards, destination)
    qualification = json.loads((destination / "fpe-qualification.json").read_text())
    assert qualification["source_sha"] == SHA
    assert qualification["tooling_sha"] == TOOLING
    assert qualification["source_branch"] == "release/0.12.0"
    assert qualification["required_probe_count"] == 4
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
        == ["release-source/python/aisimulate/.venv/bin/python", "scripts/run_release_fpe.py", "probe"]
        for row in rows
    )
    assert all(SHA in row["Command"] and TOOLING in row["Command"] for row in rows)
    assert all(row["SourceSHA"] == SHA for row in rows)


def test_incomplete_release_cannot_produce_a_web_artifact(tmp_path):
    source = tmp_path / "reports"
    shards = reports(source)
    next(source.rglob("fpe_support_matrix.json")).unlink()
    destination = tmp_path / "web"
    with pytest.raises(ValueError, match="missing shards"):
        RELEASE.package_reports(source, IDENTITY, WHEEL, shards, destination)
    assert not destination.exists()
