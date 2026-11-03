# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream run-scoped files to Artifactory, with a checksum-pinned Actions manifest."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def collect(patterns: str) -> dict[str, Path]:
    files = set()
    roots = []
    for pattern in patterns.splitlines():
        pattern = pattern.strip()
        if not pattern:
            continue
        # Match upload-artifact's least common ancestor layout. The directory
        # before the first wildcard is the root for glob uploads.
        wildcard = re.search(r"[*?\[]", pattern)
        root = Path(os.path.dirname(pattern[: wildcard.start()])) if wildcard else Path(pattern)
        if not wildcard and not root.is_dir():
            root = root.parent
        roots.append(root.resolve())
        for match in glob.glob(pattern, recursive=True):
            candidate = Path(match)
            if candidate.is_dir():
                files.update(p.resolve() for p in candidate.rglob("*") if p.is_file() and not p.is_symlink())
            elif candidate.is_file() and not candidate.is_symlink():
                files.add(candidate.resolve())
    if not files:
        raise ValueError("no files found for wheel handoff")
    if not any(path.suffix == ".whl" for path in files):
        raise ValueError("wheel handoff must contain a wheel")
    base = Path(os.path.commonpath(roots))
    return {path.relative_to(base).as_posix(): path for path in sorted(files)}


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or not path.parts
        or re.search(r"[\x00-\x1f]", value)
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or "\\" in value
    ):
        raise ValueError("unsafe manifest path")
    return value


class HTTPError(RuntimeError):
    def __init__(self, status: int):
        super().__init__(f"Artifactory request failed (HTTP {status})")
        self.status = status


def transfer(url: str, token: str, destination: Path, *, upload: bool = False, checksum: str = "") -> None:
    # Keep credentials out of process arguments and disable curl's implicit
    # config. No redirects: an Authorization header must stay on this host.
    with tempfile.NamedTemporaryFile(mode="w", prefix="wheel-curl-", delete=True) as config:
        escaped = token.replace("\\", "\\\\").replace('"', '\\"')
        config.write(f'header = "Authorization: Bearer {escaped}"\n')
        config.flush()
        command = [
            "curl",
            "-q",
            "--config",
            config.name,
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--connect-timeout",
            "30",
            "--max-time",
            "600",
            "--proto",
            "=https",
        ]
        if upload:
            command += [
                "--header",
                f"X-Checksum-Sha256: {checksum}",
                "--upload-file",
                str(destination),
                "--output",
                os.devnull,
            ]
        else:
            command += ["--output", str(destination)]
        result = subprocess.run(
            [*command, "--write-out", "%{http_code}", url],
            capture_output=True,
            text=True,
        )
        status = int(result.stdout or "0")
        if result.returncode == 22 or (result.returncode == 0 and not 200 <= status < 300):
            raise HTTPError(status)
        result.check_returncode()


def main() -> None:
    env = os.environ
    for key in ("STORAGE_URL", "STORAGE_REPO", "STORAGE_TOKEN"):
        if not env.get(key):
            raise ValueError(f"{key} is not configured")
    endpoint = urlsplit(env["STORAGE_URL"])
    if (
        endpoint.scheme != "https"
        or not endpoint.netloc
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("Artifactory URL must be an HTTPS service root without credentials")
    repo = env["STORAGE_REPO"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", repo):
        raise ValueError("invalid Artifactory repository key")
    if "\n" in env["STORAGE_TOKEN"] or "\r" in env["STORAGE_TOKEN"]:
        raise ValueError("invalid Artifactory credential")
    run_id = env["STORAGE_RUN_ID"]
    if not run_id.isdigit():
        raise ValueError("invalid producer run ID")
    name = env["STORAGE_NAME"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or ".." in name:
        raise ValueError("invalid wheel handoff name")
    repository = env["GITHUB_REPOSITORY"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid repository identity")
    repository_id = env["GITHUB_REPOSITORY_ID"]
    if not repository_id.isdigit():
        raise ValueError("invalid repository ID")
    source_sha = env["STORAGE_SOURCE_SHA"]
    workflow_sha = env["GITHUB_SHA"]
    if any(not re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (source_sha, workflow_sha)):
        raise ValueError("source and workflow revisions must be full commit SHAs")
    architecture = env["STORAGE_ARCHITECTURE"]
    if architecture not in ("amd64", "arm64"):
        raise ValueError("unsupported wheel architecture")
    base = f"{env['STORAGE_URL'].rstrip('/')}/{repo}/ci-wheels/{repository_id}/{run_id}/"
    manifest_path = Path(env["STORAGE_MANIFEST"])
    mode = env["STORAGE_MODE"]
    if mode == "upload":
        if run_id != env["GITHUB_RUN_ID"]:
            raise ValueError("upload run must be the current run")
        attempt = env["GITHUB_RUN_ATTEMPT"]
        if not attempt.isdigit():
            raise ValueError("invalid run attempt")
        manifest_path.unlink(missing_ok=True)
        entries = []
        for relative, file in collect(env["STORAGE_PATH"]).items():
            if relative.endswith(".whl") and file.stat().st_size == 0:
                raise ValueError("cannot publish an empty wheel")
            checksum = digest(file)
            # Content-addressed filenames also make deploy-only reruns safe.
            object_path = f"{attempt}/{name}/{checksum}/{safe_relative(relative)}"
            url = base + quote(object_path, safe="/")
            # Verify existing bytes or upload and read back before publishing the manifest.
            with tempfile.TemporaryDirectory() as temp:
                readback = Path(temp) / "readback"
                try:
                    transfer(url, env["STORAGE_TOKEN"], readback)
                except HTTPError as error:
                    if error.status != 404:
                        raise
                    transfer(url, env["STORAGE_TOKEN"], file, upload=True, checksum=checksum)
                    transfer(url, env["STORAGE_TOKEN"], readback)
                if digest(readback) != checksum:
                    raise ValueError("Artifactory readback checksum mismatch")
            entries.append({"path": relative, "object": object_path, "sha256": checksum, "size": file.stat().st_size})
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": 2,
                    "repository": repository,
                    "repository_id": repository_id,
                    "run_id": run_id,
                    "attempt": attempt,
                    "name": name,
                    "commit": workflow_sha,
                    "source_sha": source_sha,
                    "architecture": architecture,
                    "files": entries,
                },
                indent=2,
            )
            + "\n"
        )
    elif mode == "download":
        if manifest_path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("wheel manifest exceeds size limit")
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise ValueError("invalid wheel manifest")
        if (
            manifest.get("schema") != 2
            or manifest.get("repository_id") != repository_id
            or manifest.get("run_id") != run_id
            or manifest.get("name") != name
            or manifest.get("source_sha") != source_sha
            or manifest.get("architecture") != architecture
            or (run_id == env["GITHUB_RUN_ID"] and manifest.get("commit") != env["GITHUB_SHA"])
        ):
            raise ValueError("wheel manifest producer mismatch")
        attempt = manifest["attempt"]
        if not isinstance(attempt, str) or not attempt.isdigit():
            raise ValueError("invalid manifest attempt")
        output = Path(env["STORAGE_PATH"]).resolve()
        entries = manifest["files"]
        if not isinstance(entries, list) or not 0 < len(entries) <= 10000:
            raise ValueError("invalid manifest file list")
        if not any(entry["path"].endswith(".whl") for entry in entries):
            raise ValueError("manifest has no wheel")
        seen = set()
        # Validate the whole manifest before making authenticated requests.
        for entry in entries:
            relative = safe_relative(entry["path"])
            checksum = entry["sha256"]
            if relative in seen or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise ValueError("invalid manifest file identity")
            size = entry["size"]
            if type(size) is not int or size < 0 or (relative.endswith(".whl") and size == 0):
                raise ValueError("invalid manifest file size")
            seen.add(relative)
            if entry["object"] != f"{attempt}/{name}/{checksum}/{relative}":
                raise ValueError("manifest object is outside the producer handoff")
            target = (output / relative).resolve()
            if not target.is_relative_to(output):
                raise ValueError("download destination escapes through a symlink")
        for entry in entries:
            target = output / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                transfer(
                    base + quote(entry["object"], safe="/"),
                    env["STORAGE_TOKEN"],
                    temporary,
                )
                if temporary.stat().st_size != entry["size"] or digest(temporary) != entry["sha256"]:
                    raise ValueError("downloaded Artifactory file checksum mismatch")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    else:
        raise ValueError("storage mode must be upload or download")


if __name__ == "__main__":
    main()
