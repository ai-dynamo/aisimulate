# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``aisim data fetch`` tests with a fake transport (no network)."""

import json

import pytest

from aisim import data_cli
from aisim.main import main
from aisim_core.data.fetch import Cache, Selection, fetch

from .test_fetch import REPO, SHA, FakeTransport, NoNetwork, build_repo

pytestmark = pytest.mark.unit


@pytest.fixture
def files(tmp_path):
    return build_repo(tmp_path)


@pytest.fixture
def run(monkeypatch, tmp_path, capsys, files):
    transport = FakeTransport(files)
    monkeypatch.setattr(data_cli, "make_transport", lambda: transport)
    cache = tmp_path / "cache"

    def _run(*extra, cache_dir=True):
        argv = ["data", "fetch", "--repo", REPO, *extra]
        if cache_dir:
            argv += ["--cache-dir", str(cache)]
        status = main(argv)
        out = capsys.readouterr()
        return status, out.out, out.err

    _run.transport = transport
    _run.cache = cache
    return _run


BASE = ("--category", "fpm", "--model", "m1", "--hardware", "h200", "--purpose", "full-evidence")


def test_fetch_downloads_selection(run):
    status, out, _ = run(*BASE)
    payload = json.loads(out)
    assert status == 0 and payload["dry_run"] is False
    assert payload["plan"]["revision"] == SHA
    assert "fpm/m1/data.parquet" in payload["paths"] and "fpm/m2/data.parquet" not in payload["paths"]
    assert payload["downloaded_bytes"] > 0


def test_prediction_purpose_maps_to_entry_purpose(run):
    status, out, _ = run("--category", "fpm", "--model", "m1", "--hardware", "h200", "--purpose", "prediction")
    assert status == 0 and "fpm/m1/data.parquet" in json.loads(out)["paths"]


@pytest.mark.parametrize(
    "extra",
    [
        ("--category", "systems", "--hardware", "h200"),
        ("--category", "bogus", "--hardware", "h200"),
        ("--category", "fpm", "--model", "m1", "--hardware", "h200", "--purpose", "validation"),
        ("--category", "fpm", "--model", "m1", "--hardware", "h200", "--purpose", "results"),
        ("--category", "e2e", "--model", "m1", "--hardware", "h200"),
    ],
)
def test_unsupported_is_explicit(run, extra):
    status, out, err = run(*extra)
    assert status == 2 and out == ""
    assert "not supported" in err or "no data" in err
    assert json.loads(err.strip().splitlines()[-1])["error"] == "unsupported"


@pytest.mark.parametrize(
    "extra",
    [
        ("--category", "fpm", "--model", "m1", "--purpose", "full-evidence"),  # no hardware or configuration
        ("--category", "fpm", "--hardware", "h200", "--purpose", "full-evidence"),  # fpm needs a model
    ],
)
def test_selector_validation(run, extra):
    status, out, err = run(*extra)
    assert status == 2 and out == ""
    assert json.loads(err.strip().splitlines()[-1])["error"] == "invalid-selection"
    assert run.transport.opened == []


def test_invalid_purpose_and_missing_category_are_usage_errors(run):
    with pytest.raises(SystemExit) as exc:
        run("--category", "fpm", "--purpose", "everything")
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        run("--model", "m1")
    assert exc.value.code == 2


def test_empty_selection_is_error_not_full_download(run):
    status, out, err = run("--category", "fpm", "--model", "nope", "--hardware", "h200", "--purpose", "full-evidence")
    assert status == 2 and out == ""
    assert "matched no catalog entries" in err
    assert "fpm/m1/data.parquet" not in run.transport.opened


def test_invalid_repo(run, capsys):
    assert main(["data", "fetch", "--repo", "not a repo", *BASE]) == 2


def test_revision_pin(run, files):
    run.transport.refs = {"main": SHA, "v1": "c" * 40}
    status, out, _ = run(*BASE, "--revision", "v1")
    payload = json.loads(out)
    assert status == 0
    assert payload["plan"]["revision"] == "c" * 40 and payload["plan"]["requested_revision"] == "v1"
    assert run.transport.resolves == 1
    run.transport.resolves = 0
    status, out, _ = run(*BASE, "--revision", SHA)
    assert status == 0 and run.transport.resolves == 0


def test_unresolvable_revision_fails_nonzero(run):
    run.transport.refs = {}
    status, out, err = run(*BASE, "--revision", "missing")
    assert status == 1 and out == "" and "KeyError" in err


def test_dry_run_plans_without_artifact_download(run):
    status, out, _ = run(*BASE, "--dry-run")
    payload = json.loads(out)
    assert status == 0 and payload["dry_run"] is True
    assert payload["total_bytes"] > 0 and payload["uncached_bytes"] > 0
    assert any(f["path"] == "fpm/m1/data.parquet" and not f["cached"] for f in payload["files"])
    assert "fpm/m1/data.parquet" not in run.transport.opened
    assert not (run.cache / REPO.replace("/", "__") / SHA / "fpm" / "m1" / "data.parquet").exists()


def test_offline_empty_cache_fails_listing_missing(run, monkeypatch):
    monkeypatch.setattr(data_cli, "make_transport", lambda: NoNetwork())
    status, out, err = run(*BASE, "--offline", "--revision", SHA)
    assert status == 1 and out == ""
    assert json.loads(err.strip().splitlines()[-1])["error"] == "offline-missing"


def test_offline_branch_revision_fails(run, monkeypatch):
    monkeypatch.setattr(data_cli, "make_transport", lambda: NoNetwork())
    status, _, err = run(*BASE, "--offline")
    assert status == 1
    assert json.loads(err.strip().splitlines()[-1])["error"] == "revision-unresolved"


def test_offline_after_online_fetch_uses_cache(run, monkeypatch):
    assert run(*BASE, "--revision", SHA)[0] == 0
    monkeypatch.setattr(data_cli, "make_transport", lambda: NoNetwork())
    status, out, _ = run(*BASE, "--offline", "--revision", SHA)
    assert status == 0 and json.loads(out)["downloaded_bytes"] == 0


def test_offline_dry_run_lists_missing(run):
    status, out, _ = run(*BASE, "--offline", "--dry-run", "--revision", SHA)
    payload = json.loads(out)
    assert status == 0 and payload["missing"] == ["catalog/dataset.json"]
    assert run.transport.opened == []


def test_integrity_failure_exits_nonzero(run, files):
    files["fpm/m1/data.parquet"] = files["fpm/m1/data.parquet"][:-1] + b"\x00"
    status, out, err = run(*BASE)
    assert status == 1 and out == ""
    assert json.loads(err.strip().splitlines()[-1])["error"] == "integrity"


def test_transfer_failure_exits_nonzero(run):
    run.transport.fail_after = {"fpm/m1/data.parquet": 5}
    status, _, err = run(*BASE)
    assert status == 1
    assert json.loads(err.strip().splitlines()[-1])["error"] == "transfer-failed"


def test_idempotent_repeat(run):
    first = json.loads(run(*BASE)[1])
    run.transport.opened.clear()
    status, out, _ = run(*BASE)
    second = json.loads(out)
    assert status == 0 and second["downloaded_bytes"] == 0
    assert "fpm/m1/data.parquet" not in run.transport.opened
    assert second["paths"] == first["paths"]


def test_configuration_flag_selects_by_id(run, files):
    index = json.loads(files["catalog/fpm.json"])
    cfg = index["entries"][1]["configuration_id"]
    status, out, _ = run("--category", "fpm", "--configuration", cfg, "--purpose", "full-evidence")
    paths = json.loads(out)["paths"]
    assert status == 0 and "fpm/m2/data.parquet" in paths and "fpm/m1/data.parquet" not in paths


def test_parity_with_library(run, files, tmp_path):
    status, out, _ = run(*BASE, "--revision", SHA)
    direct = fetch(
        Selection(category="fpm", model="m1", hardware="h200"),
        repo=REPO,
        revision=SHA,
        transport=FakeTransport(files),
        cache=Cache(tmp_path / "direct"),
    ).to_dict()
    cli = json.loads(out)
    assert cli["plan"]["files"] == direct["plan"]["files"]
    assert cli["downloaded_bytes"] == direct["downloaded_bytes"]
    assert set(cli["paths"]) == set(direct["paths"])
