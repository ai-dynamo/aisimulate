# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize immutable Hugging Face FPM profiles for the native systems loader.

Storage verification does not replace native schema, provenance or exact-cell
validation. Downloads happen only on this explicit API/CLI call, never during an
ordinary op-level query. Unqualified profiles require an explicit reproduction-only opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _HubRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        old, new = urlsplit(request.full_url), urlsplit(newurl)
        same_https_origin = (
            old.scheme == new.scheme == "https"
            and old.hostname == new.hostname
            and (old.port or 443) == (new.port or 443)
        )
        authorization = request.get_header("Authorization")
        if redirected is not None and same_https_origin and authorization:
            redirected.add_unredirected_header("Authorization", authorization)
        return redirected


urlopen = build_opener(_HubRedirectHandler()).open


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("dataset paths must be nonempty relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"unsafe dataset path: {value!r}")
    return value


def _download_request(url: str) -> Request:
    """Use HF login credentials without forwarding them to signed CDN redirects."""
    request = Request(url)
    if os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "").upper() in ("1", "ON", "YES", "TRUE"):
        return request
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        default_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface"
        hf_home = Path(os.environ.get("HF_HOME", default_home))
        token_path = Path(os.environ.get("HF_TOKEN_PATH", hf_home / "token"))
        try:
            token = token_path.read_text().strip()
        except FileNotFoundError:
            token = None
    if token:
        # Preserve auth on Hub cache redirects only. Signed CDN redirects do not
        # need our credential and the handler never copies it across origins.
        request.add_unredirected_header("Authorization", f"Bearer {token.strip()}")
    return request


def _verify(directory: Path, files: list[dict]) -> None:
    for item in files:
        path = directory / item["target"]
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"missing or unsafe cached FPM file: {item['target']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"FPM SHA256 mismatch: {item['target']}")


def materialize_fpm_profile(
    manifest_path: str | Path,
    profile: str,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    allow_unqualified: bool = False,
) -> Path:
    """Return a verified ``systems_path`` pinned by a checked-in manifest.

    The manifest has ``format_version: 1``, ``repo_id``, a 40-character commit
    ``revision`` and ``profiles``. Each admitted profile contains ``admission:
    serving`` and ``files`` entries with HF ``path``, relative overlay ``target``
    and ``sha256``. Branches, tags, diagnostic profiles and corrupt caches fail
    closed; offline mode never attempts a network request. ``allow_unqualified``
    permits historical diagnostic reproduction, without granting serving admission.
    """
    manifest = json.loads(Path(manifest_path).read_text(), object_pairs_hook=_unique_object)
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported FPM dataset manifest format")
    repo = manifest.get("repo_id", "")
    revision = manifest.get("revision", "")
    if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("invalid Hugging Face dataset repository")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("FPM dataset revision must be an immutable commit SHA")
    entry = manifest["profiles"][profile]
    admission = entry.get("admission")
    if admission not in ("serving", "development", "quarantined"):
        raise ValueError(f"unknown FPM admission: {admission!r}")
    if admission != "serving":
        if not allow_unqualified:
            raise ValueError(f"FPM profile {profile!r} is not admitted for serving")
        warnings.warn(
            f"FPM profile {profile!r} is {admission}: historical reproduction only, not serving-admitted",
            UserWarning,
            stacklevel=2,
        )
    files = entry["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("FPM profile must contain files")
    targets = set()
    for item in files:
        _relative_path(item["path"])
        target = _relative_path(item["target"])
        if target in targets:
            raise ValueError(f"duplicate FPM target: {target}")
        targets.add(target)
        if not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError(f"invalid SHA256 for {target}")
    parquets = {target for target in targets if target.endswith("/fpm_forward_perf.parquet")}
    if not parquets or not any(target.endswith(".yaml") for target in targets):
        raise ValueError("FPM profile requires a system YAML and a parquet table")
    for target in parquets:
        if target.removesuffix(".parquet") + ".metadata.json" not in targets:
            raise ValueError(f"missing metadata pair for {target}")
    identity = json.dumps({"repo": repo, "revision": revision, "profile": entry}, sort_keys=True).encode()
    digest = hashlib.sha256(identity).hexdigest()
    root = (
        Path(cache_dir)
        if cache_dir is not None
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "aisimulate" / "fpm"
    )
    destination = root / digest
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise ValueError("FPM cache directory must not be a symbolic link")
        _verify(destination, files)
        return destination
    if local_files_only:
        raise FileNotFoundError(f"pinned FPM profile is not cached: {profile}")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".download-", dir=root))
    try:
        for item in files:
            output = temporary / item["target"]
            output.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(item['path'], safe='/')}"
            with urlopen(_download_request(url), timeout=120) as response, output.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        _verify(temporary, files)
        try:
            temporary.rename(destination)
        except OSError:
            # A concurrent reader may have completed the same immutable profile.
            if not destination.is_dir() or destination.is_symlink():
                raise
            _verify(destination, files)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("profile")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--allow-unqualified",
        action="store_true",
        help="historical reproduction only; does not grant serving admission",
    )
    args = parser.parse_args()
    print(
        materialize_fpm_profile(
            args.manifest,
            args.profile,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            allow_unqualified=args.allow_unqualified,
        )
    )


if __name__ == "__main__":
    main()
