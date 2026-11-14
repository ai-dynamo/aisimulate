# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Old and new public FPM repository names both resolve (AISIM-2339).

Historical pins name ``nvidia/aisimulate-fpm-dataset`` + a commit SHA and must keep resolving there,
byte for byte. Current pins name ``ai-dynamo/aisim-data`` and the ``data/fpm/`` paths. A symbolic
reference to the retired name follows the successor and says so; a SHA is never rewritten.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from aisim_core.data import repos
from aisim_core.data.ensure import pin_from_environment
from aisim_core.data.fpm_layout import physical_path
from aisim_core.sdk import fpm_dataset

pytestmark = pytest.mark.unit

NEW, OLD = repos.PUBLIC_REPO, repos.RETIRED_FPM_REPO
SHA = "a" * 40
SYSTEMS = Path(fpm_dataset.__file__).resolve().parents[1] / "systems"
BODIES = {
    "gb300.yaml": b"data_dir: data/gb300\n",
    "data/gb300/sglang/dev-example/fpm_forward_perf.parquet": b"test parquet bytes",
    "data/gb300/sglang/dev-example/fpm_forward_perf.metadata.json": b'{"schema_version":7}',
}


def test_names_are_the_documented_pair():
    assert (NEW, OLD) == ("ai-dynamo/aisim-data", "nvidia/aisimulate-fpm-dataset")
    assert repos.SUCCESSORS == {OLD: NEW}
    assert repos.is_retired(OLD) and not repos.is_retired(NEW)


@pytest.mark.parametrize(
    ("repo", "revision", "expected"),
    [
        (OLD, SHA, (OLD, SHA, "verbatim")),  # historical pin: never rewritten
        (NEW, SHA, (NEW, SHA, "verbatim")),
        (NEW, "main", (NEW, "main", "verbatim")),
        (OLD, "main", (NEW, "main", "successor-alias")),
        (OLD, "v1", (NEW, "v1", "successor-alias")),
        (OLD, "", (NEW, "main", "successor-alias")),
        (OLD, None, (NEW, "main", "successor-alias")),
        (OLD, "A" * 40, (NEW, "A" * 40, "successor-alias")),  # not a lowercase SHA: a name
        ("someone/else", "main", ("someone/else", "main", "verbatim")),
    ],
)
def test_resolve_ref(repo, revision, expected):
    resolved = repos.resolve_ref(repo, revision)
    assert (resolved.repo, resolved.revision, resolved.via) == expected


def test_paths_follow_the_alias_only_when_the_repository_does():
    historical, symbolic = repos.resolve_ref(OLD, SHA), repos.resolve_ref(OLD, "main")
    assert historical.path("data/m/h/vllm/1/tp4/manifest.json") == "data/m/h/vllm/1/tp4/manifest.json"
    assert symbolic.path("data/m/h/vllm/1/tp4/manifest.json") == "data/fpm/m/h/vllm/1/tp4/manifest.json"
    assert symbolic.path("data/fpm/m/x") == "data/fpm/m/x"
    assert symbolic.path("data/history/r/manifest.json") == physical_path("data/history/r/manifest.json")


def test_environment_pin_follows_symbolic_refs_but_not_shas():
    assert (pin := pin_from_environment({"AISIM_DATASET": f"{OLD}@main"})) and (pin.repo, pin.revision) == (NEW, "main")
    assert (pin := pin_from_environment({"AISIM_DATASET": f"{OLD}@{SHA}"})) and (pin.repo, pin.revision) == (OLD, SHA)
    assert (pin := pin_from_environment({"AISIM_DATASET": f"{NEW}@{SHA}"})) and (pin.repo, pin.revision) == (NEW, SHA)


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """Materialize the same bytes through a pin naming ``repo``; the fake hub serves what that repo holds."""
    calls: list[str] = []

    def run(repo: str, prefix: str, *, gone: bool = False):
        manifest = {
            "format_version": 1,
            "repo_id": repo,
            "revision": SHA,
            "profiles": {
                "gb300-full": {
                    "admission": "serving",
                    "files": [
                        {"path": prefix + path, "target": path, "sha256": hashlib.sha256(body).hexdigest()}
                        for path, body in BODIES.items()
                    ],
                }
            },
        }
        pin = tmp_path / f"{repo.replace('/', '_')}.json"
        pin.write_text(json.dumps(manifest))

        def fetch(request, **kwargs):
            url = request.full_url
            calls.append(url)
            head = f"https://huggingface.co/datasets/{repo}/resolve/{SHA}/{prefix}"
            if gone or not url.startswith(head):
                raise HTTPError(url, 404, "Not Found", {}, io.BytesIO())
            return io.BytesIO(BODIES[url.removeprefix(head)])

        monkeypatch.setattr(fpm_dataset, "urlopen", fetch)
        return fpm_dataset.materialize_fpm_profile(
            pin, "gb300-full", cache_dir=tmp_path / f"cache-{repo.replace('/', '_')}"
        )

    run.calls = calls
    return run


def test_old_pin_and_new_pin_materialize_identical_bytes(staged):
    old_root = staged(OLD, "archive/")  # retired layout: data/<rest> logical paths
    new_root = staged(NEW, "data/fpm/archive/")  # successor layout
    assert old_root != new_root  # the repository is part of the cache identity
    for target, body in BODIES.items():
        assert (old_root / target).read_bytes() == (new_root / target).read_bytes() == body
    assert all(f"/datasets/{OLD}/resolve/{SHA}/archive/" in u for u in staged.calls[:3])
    assert all(f"/datasets/{NEW}/resolve/{SHA}/data/fpm/archive/" in u for u in staged.calls[3:])


def test_retired_repo_that_stops_serving_a_pin_fails_with_the_migration(staged):
    with pytest.raises(ValueError, match=r"retired.*ai-dynamo/aisim-data.*never rewritten"):
        staged(OLD, "archive/", gone=True)


def test_current_repo_failure_has_no_retirement_hint(staged):
    with pytest.raises(HTTPError):
        staged(NEW, "data/fpm/archive/", gone=True)


def test_packaged_pins_old_and_new_all_parse():
    """The checked-in history pin keeps the retired name; every other pin names the successor."""
    old = fpm_dataset._read_manifest(SYSTEMS / "dsv41_fpm_hf.json")
    assert old["repo_id"] == OLD and all(
        f["path"].startswith("data/history/") for e in old["profiles"].values() for f in e["files"]
    )
    pins = sorted((SYSTEMS / "profiles").glob("*/*hf_dataset.json"))
    assert len(pins) >= 8
    for path in pins:
        manifest = fpm_dataset._read_manifest(path)
        assert manifest["repo_id"] == NEW, path
        for entry in manifest["profiles"].values():
            assert all(f["path"].startswith("data/fpm/") for f in fpm_dataset._profile_files(entry)), path
