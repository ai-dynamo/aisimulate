# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "python/aisimulate/tools/artifactory_wheel_handoff.sh"

FAKE_CURL = r"""#!/usr/bin/env python3
import hashlib
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

args = sys.argv[1:]
head = False
output = None
dump_header = None
upload_file = None
write_out = None
url = None
value_options = {
    "--connect-timeout", "--max-time", "--retry", "--output",
    "--dump-header", "--write-out", "--header", "--upload-file",
}
i = 0
while i < len(args):
    arg = args[i]
    if arg == "--head":
        head = True
        i += 1
    elif arg in value_options:
        value = args[i + 1]
        if arg == "--output":
            output = value
        elif arg == "--dump-header":
            dump_header = value
        elif arg == "--upload-file":
            upload_file = value
        elif arg == "--write-out":
            write_out = value
        i += 2
    elif arg.startswith("-"):
        i += 1
    else:
        url = arg
        i += 1

if url is None:
    raise SystemExit("fake curl received no URL")
target = Path(os.environ["FAKE_ARTIFACTORY_ROOT"]) / urlsplit(url).path.lstrip("/")

if head:
    status = "200" if target.is_file() else "404"
    if dump_header:
        headers = f"HTTP/1.1 {status}\r\n"
        if target.is_file():
            headers += f"X-Checksum-Sha256: {hashlib.sha256(target.read_bytes()).hexdigest()}\r\n"
        Path(dump_header).write_text(headers)
    if write_out:
        print(status, end="")
    raise SystemExit(0)

if upload_file:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(upload_file, target)
    with Path(os.environ["FAKE_CURL_LOG"]).open("a") as log:
        log.write(target.name + "\n")
    raise SystemExit(0)

if not target.is_file():
    raise SystemExit(22)
if output and output != "/dev/null":
    shutil.copyfile(target, output)
raise SystemExit(0)
"""


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text(FAKE_CURL)
    curl.chmod(0o755)
    remote = tmp_path / "remote"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_ARTIFACTORY_ROOT": str(remote),
        "FAKE_CURL_LOG": str(tmp_path / "curl.log"),
        "ARTIFACTORY_URL": "https://artifactory.example/artifactory",
        "ARTIFACTORY_TOKEN": "test-token",
        "ARTIFACTORY_PYPI_REPO_NAME": "test-pypi-local",
        "ARTIFACTORY_SUBPATH": "ci/source/run/application-test/amd64",
        "EXPECTED_WHEEL_SOURCE_SHA": "a" * 40,
        "GITHUB_SHA": "a" * 40,
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1",
        "RUNNER_TEMP": str(tmp_path),
    }
    return env, remote


def _run(
    mode: str, directory: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(HANDOFF), mode, str(directory)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_artifactory_handoff_uploads_manifest_last_and_downloads_exact_wheel(
    tmp_path: Path,
) -> None:
    env, remote = _environment(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    filename = "aisimulate-0.12.0-cp311-abi3-manylinux_2_28_x86_64.whl"
    payload = b"wheel payload"
    (source / filename).write_bytes(payload)

    uploaded = _run("upload", source, env)
    assert uploaded.returncode == 0, uploaded.stdout + uploaded.stderr

    remote_dir = remote / "artifactory/test-pypi-local" / env["ARTIFACTORY_SUBPATH"]
    manifest = json.loads((remote_dir / "_WHEEL.json").read_text())
    assert manifest["filename"] == filename
    assert manifest["source_sha"] == env["GITHUB_SHA"]
    assert manifest["size"] == len(payload)
    assert (remote_dir / filename).read_bytes() == payload
    assert Path(env["FAKE_CURL_LOG"]).read_text().splitlines() == [
        filename,
        "_WHEEL.json",
    ]

    destination = tmp_path / "destination"
    downloaded = _run("download", destination, env)
    assert downloaded.returncode == 0, downloaded.stdout + downloaded.stderr
    assert (destination / filename).read_bytes() == payload


def test_artifactory_handoff_records_explicit_source_sha(tmp_path: Path) -> None:
    env, remote = _environment(tmp_path)
    env["WHEEL_SOURCE_SHA"] = "c" * 40
    env["EXPECTED_WHEEL_SOURCE_SHA"] = env["WHEEL_SOURCE_SHA"]
    source = tmp_path / "source"
    source.mkdir()
    filename = "aisimulate-0.12.0-cp311-abi3-manylinux_2_28_x86_64.whl"
    (source / filename).write_bytes(b"release branch wheel")

    uploaded = _run("upload", source, env)
    assert uploaded.returncode == 0, uploaded.stdout + uploaded.stderr

    remote_dir = remote / "artifactory/test-pypi-local" / env["ARTIFACTORY_SUBPATH"]
    manifest = json.loads((remote_dir / "_WHEEL.json").read_text())
    assert manifest["source_sha"] == env["WHEEL_SOURCE_SHA"]
    assert _run("download", tmp_path / "destination", env).returncode == 0


def test_artifactory_handoff_rejects_overwrite_and_wrong_source(tmp_path: Path) -> None:
    env, remote = _environment(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    filename = "aisimulate-0.12.0-cp311-abi3-manylinux_2_28_x86_64.whl"
    wheel = source / filename
    wheel.write_bytes(b"first")
    assert _run("upload", source, env).returncode == 0

    wheel.write_bytes(b"different")
    overwrite = _run("upload", source, env)
    assert overwrite.returncode != 0
    assert "different checksum" in overwrite.stdout + overwrite.stderr

    remote_dir = remote / "artifactory/test-pypi-local" / env["ARTIFACTORY_SUBPATH"]
    manifest_path = remote_dir / "_WHEEL.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha"] = "b" * 40
    manifest_path.write_text(json.dumps(manifest))
    wrong_source = _run("download", tmp_path / "destination", env)
    assert wrong_source.returncode != 0
    assert "wheel source mismatch" in wrong_source.stdout + wrong_source.stderr
