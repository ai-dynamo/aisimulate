#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require provenance for model configurations in source trees and exact wheels.

This is an offline inventory/integrity check, not license approval or automatic
inference of a file's origin. Unknown pre-existing files remain visible under a
frozen, byte-addressed baseline; additions and changes cannot use that exception.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = Path("scripts/ci/bundled_configs.json")
BASELINE = Path(__file__).with_name("bundled_config_legacy.json")
# Changes to this frozen exception set require an explicit checker/policy review.
BASELINE_SHA256 = "8c283262bfd1a904f5b3ad9544c492c7a7b2f2c682408989eedb5ead4492545b"
CONFIG_PREFIX = "aisimulate_core/model_configs/"
CONFIG_SUFFIXES = {".json", ".yaml", ".yml", ".toml"}
FIELDS = ("dependency_type", "name", "version", "spdx_license")


class InventoryError(ValueError):
    """A bundled asset is undeclared, changed, or has incomplete attribution."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_digest(text: str) -> str:
    """Ignore layout-only whitespace when pinning a complete legal text."""
    return digest(" ".join(text.split()).encode())


def relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InventoryError(f"invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or str(path) != value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise InventoryError(f"invalid relative path: {value!r}")
    return value


def nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InventoryError(f"missing {field}")
    return value


def load_json(data: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InventoryError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(data, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise InventoryError("expected a JSON object")
    return value


def legacy_baseline() -> dict[str, str]:
    data = BASELINE.read_bytes()
    if digest(data) != BASELINE_SHA256:
        raise InventoryError("frozen unresolved baseline changed; policy review is required")
    return load_json(data)["files"]


def inventory_entries(path: Path) -> dict[str, dict]:
    inventory = load_json(path.read_bytes())
    if inventory.get("schema_version") != 1 or not isinstance(inventory.get("files"), list):
        raise InventoryError("unsupported bundled-config inventory schema")
    licenses = inventory.get("licenses")
    if not isinstance(licenses, dict):
        raise InventoryError("missing license text registry")
    result = {}
    for entry in inventory["files"]:
        if not isinstance(entry, dict):
            raise InventoryError("inventory file entry must be an object")
        name = relative_path(entry.get("path"))
        if name in result:
            raise InventoryError(f"duplicate inventory path: {name}")
        license_id = entry.get("license")
        if entry.get("origin") != "legacy_unresolved" and license_id != "UNKNOWN":
            registered = licenses.get(license_id)
            if not isinstance(registered, dict):
                raise InventoryError(f"{name}: unregistered license identifier: {license_id!r}")
            text_hashes = registered.get("license_text_sha256", [])
            if not isinstance(text_hashes, list) or not text_hashes:
                raise InventoryError(f"{name}: license has no registered complete text")
            notice = entry.get("notice")
            if isinstance(notice, dict) and notice.get("license_text_sha256") not in text_hashes:
                raise InventoryError(f"{name}: license identifier does not match the registered legal text")
        result[name] = entry
    return result


def model_config(name: str, data: bytes) -> bool:
    path = PurePosixPath(name)
    if path.suffix.lower() not in CONFIG_SUFFIXES:
        return False
    if "model_configs" in path.parts:
        return True
    # Also catch Hugging Face configs copied outside the canonical directory.
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(data)
        except (ValueError, UnicodeError):
            return False
        return isinstance(value, dict) and (
            isinstance(value.get("architectures"), list) or isinstance(value.get("model_type"), str)
        )
    return False


def source_configs(root: Path, declared: dict[str, dict]) -> dict[str, bytes]:
    result = {}
    # These directories contribute Python/package data to the unified wheel.
    roots = (("python/aisimulate/src", ""), ("python/aisimulate/collector/fpm_forward", "collector/fpm_forward/"))
    for source, prefix in roots:
        directory = root / source
        for path in directory.rglob("*"):
            if path.suffix.lower() not in CONFIG_SUFFIXES or not path.is_file():
                continue
            name = prefix + path.relative_to(directory).as_posix()
            data = path.read_bytes()
            if name in declared or model_config(name, data):
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                    raise InventoryError(f"configuration must be a regular in-tree file: {name}")
                result[name] = data
    if not any(name.startswith(CONFIG_PREFIX) for name in result):
        raise InventoryError("model configuration source directory is missing or empty")
    return result


def notice_text(root: Path, name: str, wheel: zipfile.ZipFile | None) -> str:
    if name not in {"LICENSE", "THIRD_PARTY_NOTICES.md"}:
        raise InventoryError(f"unsupported canonical legal file: {name}")
    source = (root / name).read_bytes()
    if wheel is not None:
        matches = [n for n in wheel.namelist() if n.endswith(f".dist-info/licenses/{name}")]
        if len(matches) != 1 or wheel.read(matches[0]) != source:
            raise InventoryError(f"wheel's packaged {name} is missing, duplicated, or differs from its source")
    return source.decode("utf-8")


def check_notice(root: Path, entry: dict, wheel: zipfile.ZipFile | None) -> None:
    notice = entry.get("notice")
    if not isinstance(notice, dict):
        raise InventoryError("missing canonical notice reference")
    text = notice_text(root, notice.get("file"), wheel)
    section = notice.get("section")
    if section:
        matches = re.findall(r"^## " + re.escape(section) + r"\s*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
        if len(matches) != 1:
            raise InventoryError(f"missing or duplicate notice section: {section}")
        text = matches[0]
    expected = notice.get("license_text_sha256")
    if notice.get("license_file"):
        candidates = [notice_text(root, notice["license_file"], wheel)]
    else:
        candidates = re.findall(r"```text\n(.*?)\n```", text, re.S) if section else [text]
    if not expected or expected not in {text_digest(candidate) for candidate in candidates}:
        raise InventoryError("required complete license text is missing or changed")
    if section and Path(entry["path"]).name not in text:
        raise InventoryError("canonical notice does not identify this configuration file")
    source = entry.get("source", {})
    if section and source and source["revision"] not in text:
        raise InventoryError("canonical notice does not identify this upstream revision")
    if entry["copyright_owner"] not in text:
        raise InventoryError("canonical notice does not retain the recorded copyright owner")
    prose = " ".join(re.sub(r"```.*?```", "", text, flags=re.S).split())
    for required in notice.get("required_attribution", []):
        if " ".join(nonempty(required, "required attribution").split()) not in prose:
            raise InventoryError("required attribution outside the license text is missing")


def check_entry(entry: dict, data: bytes, baseline: dict[str, str], root: Path, wheel: zipfile.ZipFile | None) -> None:
    name = entry["path"]
    actual = digest(data)
    if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256"))) or actual != entry["sha256"]:
        raise InventoryError("configuration bytes changed; update and review provenance")
    origin = entry.get("origin")
    if origin == "legacy_unresolved":
        if baseline.get(name) != actual or entry.get("license") != "UNKNOWN":
            raise InventoryError("new or modified configuration cannot use the unresolved baseline")
        nonempty(entry.get("unresolved_reason"), "unresolved_reason")
        return
    nonempty(entry.get("license"), "license")
    if entry["license"] == "UNKNOWN":
        raise InventoryError("new configurations require an identified license")
    nonempty(entry.get("copyright_owner"), "copyright_owner")
    if origin == "upstream":
        source = entry.get("source")
        if not isinstance(source, dict):
            raise InventoryError("missing upstream source")
        revision = source.get("revision")
        if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
            raise InventoryError("upstream revision must be an immutable full commit SHA")
        relative_path(source.get("path"))
        url = urlsplit(nonempty(source.get("url"), "upstream URL"))
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise InventoryError("upstream URL must be credential-free HTTPS")
        if revision not in url.path or not url.path.endswith("/" + source["path"]):
            raise InventoryError("upstream URL must identify the recorded revision and original path")
        if not isinstance(entry.get("modified"), bool):
            raise InventoryError("upstream modification status must be explicit")
        if entry["modified"]:
            nonempty(entry.get("modifications"), "modifications")
    elif origin in {"project", "generated"}:
        nonempty(entry.get("description"), "project/generated origin description")
        if origin == "generated":
            if not isinstance(entry.get("inputs"), list) or not entry["inputs"]:
                raise InventoryError("generated configuration requires recorded inputs")
            for source in entry["inputs"]:
                nonempty(source, "generation input provenance")
    else:
        raise InventoryError(f"unknown origin classification: {origin!r}")
    check_notice(root, entry, wheel)


def validate(
    root: Path, inventory: Path, *, wheel_path: Path | None = None, allow_subset: bool = False
) -> tuple[list[dict], str | None]:
    entries = inventory_entries(inventory)
    baseline = legacy_baseline()
    source = source_configs(root, entries)
    undeclared = sorted(source.keys() - entries.keys())
    if undeclared:
        raise InventoryError("configuration has no provenance entry: " + ", ".join(undeclared))
    stale = sorted(entries.keys() - source.keys())
    if stale and not allow_subset:
        raise InventoryError("inventory names missing source configurations: " + ", ".join(stale))
    # A source check remains mandatory even when a wheel is supplied.
    for name, data in source.items():
        try:
            check_entry(entries[name], data, baseline, root, None)
        except (InventoryError, KeyError, TypeError) as error:
            raise InventoryError(f"{name}: {error}") from error
    wheel_sha = None
    if wheel_path is not None:
        wheel_sha = digest(wheel_path.read_bytes())
        with zipfile.ZipFile(wheel_path) as wheel:
            names = wheel.namelist()
            if len(names) != len(set(names)):
                raise InventoryError("wheel has duplicate archive members")
            actual = {}
            for name in names:
                relative_path(name.removesuffix("/"))
                if PurePosixPath(name).suffix.lower() in CONFIG_SUFFIXES:
                    data = wheel.read(name)
                    if name in entries or model_config(name, data):
                        actual[name] = data
            if actual.keys() != source.keys():
                raise InventoryError(
                    "wheel configuration inventory differs from source; "
                    f"unexpected={sorted(actual.keys() - source.keys())}, "
                    f"missing={sorted(source.keys() - actual.keys())}"
                )
            for name, data in actual.items():
                try:
                    check_entry(entries[name], data, baseline, root, wheel)
                except (InventoryError, KeyError, TypeError) as error:
                    raise InventoryError(f"{name}: {error}") from error
            # Verify canonical legal payloads even if every config is unresolved.
            for legal_file in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
                notice_text(root, legal_file, wheel)
    return [entries[name] for name in sorted(source)], wheel_sha


def write_reports(directory: Path, entries: list[dict], wheel_sha: str | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "bundled-deps.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for entry in entries:
            # Configs have no package version. Content identity makes byte changes
            # visible to the existing nightly added/removed dependency diff.
            writer.writerow(
                dict(
                    dependency_type="bundled_model_config",
                    name=entry["path"],
                    version="sha256:" + entry["sha256"],
                    spdx_license=entry["license"],
                )
            )
    report = dict(schema_version=1, wheel_sha256=wheel_sha, license_approval_asserted=False, files=entries)
    (directory / "bundled-provenance.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Exact source checkout being built")
    parser.add_argument("--inventory", type=Path, help="Explicit trusted inventory for historical sources")
    parser.add_argument("--allow-subset", action="store_true", help="Permit historical sources to omit catalog entries")
    parser.add_argument("--wheel", type=Path, help="Inspect the exact repaired wheel without importing it")
    parser.add_argument("--output-dir", type=Path, help="Write CSV inventory and complete provenance evidence")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.allow_subset and args.inventory is None:
        parser.error("--allow-subset requires an explicit trusted --inventory")
    inventory = args.inventory or root / INVENTORY
    try:
        entries, wheel_sha = validate(root, inventory, wheel_path=args.wheel, allow_subset=args.allow_subset)
        if args.output_dir:
            write_reports(args.output_dir, entries, wheel_sha)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"::error::Bundled configuration provenance check failed: {error}", file=sys.stderr)
        return 1
    unresolved = sum(entry["origin"] == "legacy_unresolved" for entry in entries)
    custom = sum(entry["license"].startswith("LicenseRef-") for entry in entries)
    print(
        f"Verified {len(entries)} bundled configs; {custom} custom-licensed; {unresolved} legacy provenance unresolved"
    )
    if unresolved:
        print("::warning::Pre-existing unresolved configs are byte-pinned, not license-approved; see provenance report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
