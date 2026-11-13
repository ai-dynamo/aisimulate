# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation is atomic and cannot bless changed source or native binaries."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location("afd_generation", Path(__file__).parents[1] / "scripts/afd_parity.py")
parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parity)


@pytest.mark.parametrize("module", ["test_afd_parity.py", "test_afd_execution_plan_parity.py"])
def test_standalone_parity_import_keeps_the_installed_package_first(tmp_path, module):
    # Model a non-editable wheel: its runtime lives beside its __init__.py,
    # while the checkout is only the source of the parity tests and scripts.
    package = tmp_path / "site-packages/aisim"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from . import _runtime\n")
    (package / "_runtime.py").write_text("installed = True\n")
    target = parity.ROOT / "crates/core/parity_tests/perfmodel" / module
    code = (
        "import runpy, sys\n"
        "from pathlib import Path\n"
        "target, installed = map(Path, sys.argv[1:])\n"
        "sys.path.insert(0, str(target.parent))\n"
        "runpy.run_path(str(target))\n"
        "import aisim\n"
        "assert Path(aisim.__file__).parent == installed\n"
        "assert aisim._runtime.installed\n"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(target), str(package)],
        env={**os.environ, "PYTHONPATH": str(package.parent)},
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("stale", ["source", "binary"])
def test_build_identity_rejects_stale_input(tmp_path, monkeypatch, stale):
    source = tmp_path / "source.rs"
    source.write_text("checked source")
    manifest = {
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "data_tree": "c" * 40,
        "binary_sha256": "wrong binary",
        "source_files": {"source.rs": parity._sha256(source)},
    }
    if stale == "source":
        source.write_text("changed after build")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(parity, "ROOT", tmp_path)
    with pytest.raises(
        ValueError,
        match="source changed" if stale == "source" else "native binary differs",
    ):
        parity._verified_build(path)


@pytest.mark.parametrize("fail_after_run", [False, True])
def test_partial_refresh_is_atomic_and_keeps_other_case_identity(tmp_path, monkeypatch, fail_after_run):
    fixture = {
        "cases": {
            name: {
                "config": f"{name}.yaml",
                "summary": {"completed_requests": 1},
                "generation": {"source_commit": "original"},
            }
            for name in ["selected", "untouched"]
        }
    }
    path = tmp_path / "afd_e2e.json"
    path.write_text(json.dumps(fixture))
    before = path.read_bytes()
    (tmp_path / "selected.yaml").write_text("input")
    monkeypatch.setattr(parity, "ROOT", tmp_path)
    monkeypatch.setattr(parity, "GOLDEN_DIR", tmp_path)
    calls = []

    def verify(_):
        calls.append(1)
        if len(calls) > 1 and fail_after_run:
            raise ValueError("source changed after build")
        return {"source_commit": "verified"}

    def run(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        (output / "prediction.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "completed_requests": 1,
                        "prediction_provenance": {"data_root": "/machine/local"},
                    }
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(parity, "_verified_build", verify)
    monkeypatch.setattr(parity.subprocess, "run", run)
    if fail_after_run:
        with pytest.raises(ValueError, match="source changed"):
            parity._refresh_e2e(["selected"], "explained change", build_manifest=tmp_path / "manifest")
        assert path.read_bytes() == before
    else:
        parity._refresh_e2e(["selected"], "explained change", build_manifest=tmp_path / "manifest")
        actual = json.loads(path.read_text())
        assert actual["cases"]["untouched"] == fixture["cases"]["untouched"]
        assert actual["cases"]["selected"]["generation"]["source_commit"] == "verified"
        assert actual["cases"]["selected"]["historical_generations"] == [fixture["cases"]["selected"]["generation"]]
        assert "prediction_provenance" not in actual["cases"]["selected"]["summary"]


def test_layout_refresh_preserves_previous_generation_and_unselected_cases(tmp_path, monkeypatch):
    fixture = {
        "cases": {
            name: {
                "config": {},
                "trace": False,
                "expected": {"completed_requests": 1, "duration_ms": 1},
                "generation": {"source_commit": "original"},
            }
            for name in ["selected", "untouched"]
        }
    }
    path = tmp_path / "afd_execution_plan.json"
    path.write_text(json.dumps(fixture))
    monkeypatch.setattr(parity, "GOLDEN_DIR", tmp_path)
    monkeypatch.setattr(parity, "layout_config", lambda record, directory: {})
    monkeypatch.setattr(parity, "_verified_build", lambda manifest: {"source_commit": "new"})

    def run(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        (output / "prediction.json").write_text(json.dumps({"summary": {"completed_requests": 1, "duration_ms": 2}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(parity.subprocess, "run", run)
    parity._refresh_layouts(["selected"], "explained change", tmp_path / "manifest")
    result = json.loads(path.read_text())
    assert result["cases"]["untouched"] == fixture["cases"]["untouched"]
    selected = result["cases"]["selected"]
    assert selected["historical_generations"] == [fixture["cases"]["selected"]["generation"]]
    assert selected["generation"]["source_commit"] == "new"
    assert selected["expected"]["duration_ms"] == 2


def test_golden_job_runs_both_afd_suites_with_installed_python():
    import yaml

    root = Path(__file__).parents[1]
    job = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())["jobs"]["engine-golden-regression"]
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "test_afd_parity.py" in commands
    assert "test_afd_execution_plan_parity.py" in commands
    source = (root / "crates/core/parity_tests/perfmodel/test_afd_parity.py").read_text()
    assert 'AISIM_CLI = [sys.executable, "-m", "aisim"]' in source


def test_snapshot_checks_archive_contents_and_digest(tmp_path):
    import gzip
    import hashlib
    import io
    import tarfile

    archive = tmp_path / "source.tar.gz"
    content = b"immutable source"
    files = {"new.rs": hashlib.sha256(content).hexdigest()}
    with gzip.open(archive, "wb") as zipped, tarfile.open(fileobj=zipped, mode="w") as tar:
        entry = tarfile.TarInfo("new.rs")
        entry.size = len(content)
        tar.addfile(entry, io.BytesIO(content))
    manifest = {
        "source_snapshot": archive.name,
        "source_snapshot_sha256": parity._sha256(archive),
        "source_files": files,
        "source_inventory_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }
    parity._verify_snapshot(manifest, tmp_path / "manifest.json")
    files["new.rs"] = "wrong"
    manifest["source_inventory_sha256"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError, match="snapshot contents differ"):
        parity._verify_snapshot(manifest, tmp_path / "manifest.json")
    archive.write_bytes(b"changed archive")
    with pytest.raises(ValueError, match="snapshot digest mismatch"):
        parity._verify_snapshot(manifest, tmp_path / "manifest.json")


def test_source_inventory_includes_untracked_runtime_inputs(tmp_path, monkeypatch):
    (tmp_path / "Cargo.toml").write_text("workspace manifest")
    new = tmp_path / "crates/core/src/new.rs"
    new.parent.mkdir(parents=True)
    new.write_text("new runtime source")
    new_python = tmp_path / "python/aisim/src/aisim/new.py"
    new_python.parent.mkdir(parents=True)
    new_python.write_text("new_runtime = True")
    monkeypatch.setattr(parity, "ROOT", tmp_path)
    monkeypatch.setattr(parity.subprocess, "check_output", lambda *args, **kwargs: b"Cargo.toml\0")
    assert {
        "Cargo.toml",
        "crates/core/src/new.rs",
        "python/aisim/src/aisim/new.py",
    } <= parity._source_inputs()


@pytest.fixture
def built_workspace(tmp_path, monkeypatch):
    """A tiny Git checkout and extension whose identity was captured at build time."""
    import aisim

    root = tmp_path.resolve()
    contents = {
        ".gitignore": "runtime.so\n",
        "Cargo.toml": "workspace manifest",
        "Cargo.lock": "resolved dependencies",
        "crates/core/Cargo.toml": "crate manifest",
        "crates/core/build.rs": "build script",
        "crates/core/src/lib.rs": "pub fn original() {}",
        "crates/core/src/obsolete.rs": "pub fn obsolete() {}",
        "python/aisim/src/aisim/current.py": "value = 1",
        "python/aisim/src/aisim_core/systems/data/control.json": "{}",
    }
    for name, content in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    monkeypatch.setattr(parity, "ROOT", root)
    built = json.dumps({name: parity._sha256(root / name) for name in parity._native_source_inputs()})
    binary = root / "runtime.so"
    binary.write_bytes(b"compiled extension")
    runtime = SimpleNamespace(__file__=str(binary), _build_source_files_json=lambda: built)
    monkeypatch.setattr(aisim, "_runtime", runtime, raising=False)
    return root, runtime


@pytest.mark.parametrize("change", ["none", "python", "staged", "untracked"])
def test_snapshot_accepts_matching_build_and_current_python(built_workspace, change):
    root, _ = built_workspace
    if change in {"python", "staged"}:
        (root / "python/aisim/src/aisim/current.py").write_text("value = 2")
        if change == "staged":
            subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    elif change == "untracked":
        (root / "python/aisim/src/aisim/new.py").write_text("value = 3")
    output = root / "evidence/manifest.json"
    manifest = parity._write_build_snapshot(output)
    assert manifest["workspace_dirty"] is (change != "none")
    # The new untracked evidence must not change the recorded pre-snapshot state.
    assert "?? evidence/" in subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    verified = parity._verified_build(output)
    assert verified["native_source_inventory_sha256"] == manifest["native_source_inventory_sha256"]
    parity._verify_snapshot(manifest, output)


@pytest.mark.parametrize(
    "change", ["edit", "add", "delete", "rename", "Cargo.toml", "Cargo.lock", "crates/core/build.rs"]
)
def test_snapshot_rejects_native_changes_since_build_without_writing(built_workspace, change):
    root, _ = built_workspace
    source = root / "crates/core/src/obsolete.rs"
    if change == "edit":
        source.write_text("this Rust source cannot compile")
    elif change == "add":
        source.with_name("new.rs").write_text("new source")
    elif change == "delete":
        source.unlink()
    elif change == "rename":
        source.rename(source.with_name("renamed.rs"))
    else:
        (root / change).write_text("changed build input")
    output = root / "evidence/manifest.json"
    with pytest.raises(ValueError, match="built from different sources; rebuild"):
        parity._write_build_snapshot(output)
    assert not output.parent.exists()


def test_snapshot_rejects_an_extension_without_build_identity(built_workspace, capsys):
    root, runtime = built_workspace
    del runtime._build_source_files_json
    output = root / "evidence/manifest.json"
    assert parity.main(["snapshot-build", "--output", str(output)]) == 1
    assert "no build-time source identity; rebuild" in capsys.readouterr().err
    assert not output.parent.exists()


def test_refresh_verification_rejects_recomputed_manifest_for_stale_binary(built_workspace):
    root, _ = built_workspace
    output = root / "evidence/manifest.json"
    manifest = parity._write_build_snapshot(output)
    source = root / "crates/core/src/lib.rs"
    source.write_text("this Rust source cannot compile")
    # Rehashing the new source into a manifest must not bless the old binary,
    # even though its checksum and all current source checksums still match.
    manifest["source_files"]["crates/core/src/lib.rs"] = parity._sha256(source)
    output.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="built from different sources; rebuild"):
        parity._verified_build(output)


@pytest.mark.parametrize("identity", [None, "incorrect"])
def test_refresh_requires_recorded_native_identity(built_workspace, identity):
    root, _ = built_workspace
    output = root / "evidence/manifest.json"
    manifest = parity._write_build_snapshot(output)
    if identity is None:
        del manifest["native_source_inventory_sha256"]
    else:
        manifest["native_source_inventory_sha256"] = identity
    output.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing or inconsistent native source identity; rebuild"):
        parity._verified_build(output)
