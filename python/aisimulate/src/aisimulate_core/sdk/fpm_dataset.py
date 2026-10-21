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
_SYSTEMS_DIRECTORY = Path(__file__).resolve().parents[1] / "systems"


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


def _resolve_manifest_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    if path.is_file():
        return path.resolve()
    if path.is_absolute():
        raise FileNotFoundError(f"FPM manifest not found: {manifest_path}")
    # Existing local files take precedence over packaged manifest identifiers.
    relative = _relative_path(str(manifest_path))
    if "/" not in relative:
        relative += "/hf_dataset.json"
    profiles = _SYSTEMS_DIRECTORY / "profiles"
    packaged = (profiles / relative).resolve()
    if packaged.is_relative_to(profiles.resolve()) and packaged.is_file():
        return packaged
    raise FileNotFoundError(f"FPM manifest not found: {manifest_path}")


def _read_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(), object_pairs_hook=_unique_object)
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported FPM dataset manifest format")
    repo = manifest.get("repo_id", "")
    revision = manifest.get("revision", "")
    if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("invalid Hugging Face dataset repository")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("FPM dataset revision must be an immutable commit SHA")
    if not isinstance(manifest.get("profiles"), dict) or not manifest["profiles"]:
        raise ValueError("FPM manifest must contain profiles")
    return manifest


def _profile_files(entry: dict) -> list[dict]:
    admission = entry.get("admission")
    if admission not in ("serving", "development", "quarantined"):
        raise ValueError(f"unknown FPM admission: {admission!r}")
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
    packaged_system = entry.get("packaged_system")
    if packaged_system is not None:
        if not isinstance(packaged_system, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", packaged_system):
            raise ValueError("packaged_system must name an installed system")
        if f"{packaged_system}.yaml" in targets:
            raise ValueError("packaged_system must not replace a dataset system YAML")
    parquets = {target for target in targets if target.endswith("/fpm_forward_perf.parquet")}
    if not parquets or (packaged_system is None and not any(target.endswith(".yaml") for target in targets)):
        raise ValueError("FPM profile requires a system YAML and a parquet table")
    for target in parquets:
        if target.removesuffix(".parquet") + ".metadata.json" not in targets:
            raise ValueError(f"missing metadata pair for {target}")
    return files


def list_fpm_profiles(manifest_path: str | Path | None = None) -> list[dict]:
    """List pinned profiles and admission without downloading or touching caches.

    Omit ``manifest_path`` to discover the manifests shipped in this package.
    A manifest may be a local file, a packaged directory ID (``kimi_k3_fpm``),
    or a packaged relative path (``dsv41_fpm/four_gpu_hf_dataset.json``).
    """
    if manifest_path is None:
        profiles = _SYSTEMS_DIRECTORY / "profiles"
        paths = sorted(set(profiles.glob("*/hf_dataset.json")) | set(profiles.glob("*/*_hf_dataset.json")))
    else:
        paths = [_resolve_manifest_path(manifest_path)]
    result = []
    for path in paths:
        manifest = _read_manifest(path)
        for key, entry in sorted(manifest["profiles"].items()):
            _profile_files(entry)
            record = {
                "manifest": str(path.resolve()),
                "profile": key,
                "admission": entry["admission"],
                "repo_id": manifest["repo_id"],
                "revision": manifest["revision"],
            }
            if entry.get("admission_reason"):
                record["admission_reason"] = entry["admission_reason"]
            result.append(record)
    return result


def _packaged_system_bytes(system: str) -> bytes:
    source = (_SYSTEMS_DIRECTORY / f"{system}.yaml").resolve()
    if not source.is_relative_to(_SYSTEMS_DIRECTORY.resolve()) or not source.is_file():
        raise ValueError(f"missing or unsafe packaged system: {system}")
    return source.read_bytes()


def materialize_fpm_profile(
    manifest_path: str | Path,
    profile: str,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    allow_unqualified: bool = False,
) -> Path:
    """Return a verified ``systems_path`` pinned by a local or packaged manifest.

    Manifests pin an immutable dataset revision and each file's SHA256. An
    explicit ``packaged_system`` uses the installed hardware YAML when collection
    YAML is unavailable. Its actual bytes enter the cache key and verification;
    they remain a reference configuration, not collection evidence. Legacy
    ``identity`` is descriptive metadata only; native schema and query checks
    enforce execution identity.

    Offline mode never downloads. ``allow_unqualified`` permits diagnostic
    reproduction without granting serving admission. The complete manifest entry
    remains part of the cache key, including legacy metadata for compatibility.
    """
    manifest = _read_manifest(_resolve_manifest_path(manifest_path))
    repo, revision = manifest["repo_id"], manifest["revision"]
    entry = manifest["profiles"][profile]
    files = _profile_files(entry)
    admission = entry["admission"]
    if admission != "serving":
        reason = f"; {entry['admission_reason']}" if entry.get("admission_reason") else ""
        if not allow_unqualified:
            raise ValueError(f"FPM profile {profile!r} is not admitted for serving{reason}")
        warnings.warn(
            f"FPM profile {profile!r} is {admission}: historical reproduction only, not serving-admitted{reason}",
            UserWarning,
            stacklevel=2,
        )
    cache_identity = {"repo": repo, "revision": revision, "profile": entry}
    system_file = None
    if entry.get("packaged_system") is not None:
        system_bytes = _packaged_system_bytes(entry["packaged_system"])
        system_file = {
            "target": f"{entry['packaged_system']}.yaml",
            "sha256": hashlib.sha256(system_bytes).hexdigest(),
        }
        cache_identity["packaged_system_sha256"] = system_file["sha256"]
        files = [*files, system_file]
    identity = json.dumps(cache_identity, sort_keys=True).encode()
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
        if system_file is not None:
            (temporary / system_file["target"]).write_bytes(system_bytes)
        for item in entry["files"]:
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
    parser.add_argument("manifest", nargs="?", help="local manifest path or packaged manifest ID")
    parser.add_argument("profile", nargs="?")
    parser.add_argument("--list", action="store_true", help="list profiles as JSON without downloading")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--allow-unqualified",
        action="store_true",
        help="historical reproduction only; does not grant serving admission",
    )
    args = parser.parse_args()
    if args.list:
        if args.profile is not None:
            parser.error("--list accepts a manifest, but not a profile key")
        print(json.dumps(list_fpm_profiles(args.manifest), indent=2))
        return
    if args.manifest is None or args.profile is None:
        parser.error("manifest and profile are required unless --list is used")
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
