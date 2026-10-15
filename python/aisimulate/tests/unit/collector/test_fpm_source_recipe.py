# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source archive integrity and real subprocess checks; no GPU qualification."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
MODULE = Path(__file__).resolve().parents[3] / "collector/fpm_forward/recipe.py"
spec = importlib.util.spec_from_file_location("fpm_source_recipe", MODULE)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


def make_archive(tmp_path, *, mutate=None, extra=None):
    files = {
        "run.py": b"import sys\nfrom helper import main\nraise SystemExit(main(sys.argv[1:]))\n",
        "helper.py": (
            b"import json\nfrom pathlib import Path\ndef main(args):\n"
            b" Path(args[0]).write_text(json.dumps(args[2:]))\n return int(args[1])\n"
        ),
        "LICENSE": b"Original test fixture, Apache-2.0.\n",
        "README.md": b"CPU integrity fixture. No model or measurement acceptance.\n",
    }
    manifest = {
        "schema": recipe.SCHEMA,
        "entrypoint": {"interpreter": "python", "path": "run.py"},
        "licenses": ["LICENSE"],
        "readme": "README.md",
        "files": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in files.items()},
    }
    if mutate:
        mutate(manifest, files)
    files["recipe.json"] = json.dumps(manifest).encode()
    archive = tmp_path / "recipe.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        for name, raw in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            target.addfile(info, io.BytesIO(raw))
        if extra:
            extra(target)
    pin = {
        "repo_id": "fixture/fpm",
        "revision": "a" * 40,
        "filename": "recipes/source.tar.gz",
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    return archive, pin


def test_materialize_import_and_run_exact_source(tmp_path):
    archive, pin = make_archive(tmp_path)
    destination = tmp_path / "materialized"
    root = recipe.materialize(pin, destination, archive=archive)
    output = tmp_path / "observed.json"
    literal = "$(not-a-shell-command)\nexact argument"
    assert recipe.run_recipe(destination, pin, [str(output), "0", literal]) == 0
    assert json.loads(output.read_text()) == [literal]
    assert recipe.verify_materialized(destination, pin)["entrypoint"]["path"] == "run.py"
    assert (root / "LICENSE").read_bytes() == b"Original test fixture, Apache-2.0.\n"
    receipt = json.loads(next(destination.glob("execution-*.json")).read_text())
    assert receipt["exit_code"] == 0
    assert receipt["measurement_admission"] == "owned by the archived native recipe"


def test_nonzero_native_exit_is_not_success(tmp_path):
    archive, pin = make_archive(tmp_path)
    destination = tmp_path / "materialized"
    recipe.materialize(pin, destination, archive=archive)
    assert recipe.run_recipe(destination, pin, [str(tmp_path / "output"), "23"]) == 23
    assert json.loads(next(destination.glob("execution-*.json")).read_text())["exit_code"] == 23


def test_wrong_archive_preserves_failure_and_never_publishes_source(tmp_path):
    archive, pin = make_archive(tmp_path)
    pin["sha256"] = "0" * 64
    destination = tmp_path / "failed"
    with pytest.raises(ValueError, match="immutable pin"):
        recipe.materialize(pin, destination, archive=archive)
    assert not (destination / "source").exists()
    assert json.loads((destination / "materialization.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("mutation", ["digest", "size", "missing", "extra", "entry", "licenses"])
def test_incomplete_or_inconsistent_source_manifest_rejected(tmp_path, mutation):
    def mutate(manifest, files):
        if mutation == "digest":
            manifest["files"]["helper.py"]["sha256"] = "0" * 64
        elif mutation == "size":
            manifest["files"]["helper.py"]["bytes"] += 1
        elif mutation == "missing":
            del files["helper.py"]
        elif mutation == "extra":
            files["unbound.py"] = b""
        elif mutation == "entry":
            manifest["entrypoint"]["path"] = "unbound.py"
        else:
            manifest["licenses"] = []

    archive, pin = make_archive(tmp_path, mutate=mutate)
    with pytest.raises(ValueError):
        recipe.materialize(pin, tmp_path / "failed", archive=archive)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "device", "directory", "duplicate", "traversal", "absolute"])
def test_tar_nonfiles_aliases_and_escape_paths_rejected(tmp_path, kind):
    def extra(target):
        names = {"duplicate": "run.py", "traversal": "../outside", "absolute": "/outside"}
        info = tarfile.TarInfo(names.get(kind, "link"))
        types = {
            "symlink": tarfile.SYMTYPE,
            "hardlink": tarfile.LNKTYPE,
            "device": tarfile.CHRTYPE,
            "directory": tarfile.DIRTYPE,
        }
        info.type = types.get(kind, tarfile.REGTYPE)
        info.linkname = "run.py" if kind == "hardlink" else "../outside"
        target.addfile(info, io.BytesIO(b""))

    archive, pin = make_archive(tmp_path, extra=extra)
    with pytest.raises(ValueError):
        recipe.materialize(pin, tmp_path / "failed", archive=archive)
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("change", ["source", "extra", "symlink", "manifest_and_receipt", "archive", "pin"])
def test_materialized_tampering_fails_before_execution(tmp_path, change):
    archive, pin = make_archive(tmp_path)
    destination = tmp_path / "materialized"
    root = recipe.materialize(pin, destination, archive=archive)
    if change == "source":
        (root / "helper.py").chmod(0o600)
        (root / "helper.py").write_text("raise RuntimeError('modified')")
    elif change == "extra":
        (root / "extra.py").touch()
    elif change == "symlink":
        (root / "helper.py").unlink()
        (root / "helper.py").symlink_to(root / "run.py")
    elif change == "manifest_and_receipt":
        (root / "recipe.json").chmod(0o600)
        (root / "recipe.json").write_bytes((root / "recipe.json").read_bytes() + b" ")
        receipt_path = destination / "materialization.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["recipe_sha256"] = hashlib.sha256((root / "recipe.json").read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt))
    elif change == "archive":
        (destination / "archive.tar").chmod(0o600)
        (destination / "archive.tar").write_bytes(b"different")
    else:
        pin["revision"] = "b" * 40
    with pytest.raises(ValueError):
        recipe.run_recipe(destination, pin, [])
    assert not list(destination.glob("execution-*.json"))


def test_existing_destination_is_never_overwritten(tmp_path):
    archive, pin = make_archive(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "user-file").write_text("preserve")
    with pytest.raises(FileExistsError):
        recipe.materialize(pin, existing, archive=archive)
    assert (existing / "user-file").read_text() == "preserve"


@pytest.mark.parametrize("revision", ["main", "a" * 7, "refs/pr/12", None])
def test_mutable_hf_revision_rejected(tmp_path, revision):
    archive, pin = make_archive(tmp_path)
    pin["revision"] = revision
    with pytest.raises(ValueError):
        recipe.materialize(pin, tmp_path / "unused", archive=archive)
    assert not (tmp_path / "unused").exists()


def test_hf_download_uses_exact_commit_and_existing_auth(tmp_path, monkeypatch):
    from types import SimpleNamespace

    archive, pin = make_archive(tmp_path)
    commands = []

    def download(argv, **kwargs):
        commands.append(argv)
        assert kwargs["check"] is True
        assert kwargs["stdin"] == recipe.subprocess.DEVNULL
        return SimpleNamespace(stdout=str(archive) + "\n")

    monkeypatch.setattr(recipe.subprocess, "run", download)
    recipe.materialize(pin, tmp_path / "downloaded")
    assert commands == [
        [
            "hf",
            "download",
            pin["repo_id"],
            pin["filename"],
            "--repo-type",
            "dataset",
            "--revision",
            pin["revision"],
            "--quiet",
        ]
    ]


def test_cli_materialize_and_explicit_run(tmp_path):
    archive, pin = make_archive(tmp_path)
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin))
    output = tmp_path / "result.json"
    assert (
        recipe.main(
            [
                "--pin",
                str(pin_path),
                "--archive",
                str(archive),
                "--destination",
                str(tmp_path / "cli"),
                "--run",
                "--",
                str(output),
                "0",
                "literal",
            ]
        )
        == 0
    )
    assert json.loads(output.read_text()) == ["literal"]
