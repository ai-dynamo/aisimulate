# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that one ``aisimulate`` wheel owns every Python runtime payload."""

from __future__ import annotations

import argparse
import re
import subprocess
import zipfile
from email import message_from_bytes
from email.message import Message
from pathlib import Path

PAYLOAD_SUFFIXES = {
    ".css",
    ".csv",
    ".j2",
    ".js",
    ".json",
    ".md",
    ".parquet",
    ".py",
    ".pyi",
    ".rule",
    ".txt",
    ".typed",
    ".yaml",
}
LEGAL_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _wheel_files(wheel: Path) -> tuple[set[str], Message]:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise RuntimeError(f"{wheel.name}: expected one METADATA file, found {metadata_paths}")
        metadata = message_from_bytes(archive.read(metadata_paths[0]))
    return names, metadata


def _payload_files(names: set[str]) -> set[str]:
    return {name for name in names if ".dist-info/" not in name and not name.endswith("/")}


def _spica_entries(names: set[str]) -> list[str]:
    """Return every stale Spica archive member, regardless of type or suffix."""
    return sorted(name for name in names if name.startswith("spica/"))


def _infra_entries(names: set[str]) -> list[str]:
    """Return gap-analysis, skill, tool, dataset, report, and web payloads."""
    forbidden_roots = (".agents/", "tools/", "datasets/", "reports/", "web/", "webapp/")
    forbidden_package_roots = (
        "aiconfigurator/datasets/",
        "aiconfigurator/gap_analysis/",
        "aiconfigurator/reports/",
        "aiconfigurator/skills/",
        "aiconfigurator/tools/",
        "aiconfigurator/web/",
        "aiconfigurator/webapp/",
    )
    return sorted(
        name
        for name in names
        if name.startswith(forbidden_roots + forbidden_package_roots)
        or any(
            segment in name
            for segment in (
                "/datasets/",
                "/gap_analysis/",
                "/reports/",
                "/skills/",
                "/tools/",
                "/web/",
                "/webapp/",
            )
        )
        or "auto_gap_analysis" in name
        or "auto-gap-analysis" in name
    )


def _one_wheel(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one wheel matching {pattern!r}, found {[path.name for path in matches]}")
    return matches[0]


def _verify_legal_files(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for legal_file in LEGAL_FILES:
            matches = [name for name in names if name.endswith(f".dist-info/licenses/{legal_file}")]
            if len(matches) != 1:
                raise RuntimeError(f"{wheel.name}: expected one packaged {legal_file}, found {matches}")
            expected = (PACKAGE_ROOT / legal_file).read_bytes()
            if archive.read(matches[0]) != expected:
                raise RuntimeError(f"{wheel.name}: packaged {legal_file} differs from the project copy")


def _add_source_tree(expected: set[str], source_root: Path, package_root: str) -> None:
    for path in source_root.rglob("*"):
        if path.is_file() and path.suffix in PAYLOAD_SUFFIXES:
            expected.add((Path(package_root) / path.relative_to(source_root)).as_posix())


def _source_payloads() -> set[str]:
    """Return package payloads that the sole wheel must own."""
    source_root = Path(__file__).resolve().parents[1] / "src"
    expected: set[str] = set()
    for package in ("aisimulate", "aisimulate_core", "aiconfigurator", "aiconfigurator_core"):
        _add_source_tree(expected, source_root / package, package)
    expected.discard("aiconfigurator/sdk/config_adapter/README.md")
    collector_root = Path(__file__).resolve().parents[1] / "collector"
    expected.update({"collector/__init__.py", "collector/model_cases.py"})
    for pattern in ("cases/**/*.yaml", "fpm_forward/**/*.py", "fpm_forward/runtime/fpm_exec.sh"):
        expected.update(
            (Path("collector") / path.relative_to(collector_root)).as_posix()
            for path in collector_root.glob(pattern)
            if path.is_file()
        )
    return expected


def _verify_rust_crate_package() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    manifest = repository_root / "crates" / "core" / "Cargo.toml"
    result = subprocess.run(
        ["cargo", "package", "--list", "--allow-dirty", "--manifest-path", str(manifest)],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    entries = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    forbidden = sorted(
        entry
        for entry in entries
        if entry.startswith((".agents/", "aiconfigurator/", "datasets/", "reports/", "tools/", "web/", "webapp/"))
        or "config_adapter" in entry
        or "gap_analysis" in entry
        or "auto-gap-analysis" in entry
    )
    if forbidden:
        raise RuntimeError(f"Rust crate contains Python application or infra payload: {forbidden}")
    if not any(entry.startswith("src/") for entry in entries):
        raise RuntimeError("Rust crate package list contains no core source files")


def _requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if match is None:
        return ""
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def _verify_wheel(wheel: Path, expected_payload: set[str]) -> set[str]:
    names, metadata = _wheel_files(wheel)
    payload = _payload_files(names)
    removed = _spica_entries(names)
    if removed:
        raise RuntimeError(f"{wheel.name}: removed Spica payload is still present: {removed}")
    infra = _infra_entries(names)
    if infra:
        raise RuntimeError(f"{wheel.name}: infra-only payload must not be packaged: {infra}")

    required = {
        "aisimulate/__init__.py",
        "aisimulate_core/__init__.py",
        "aiconfigurator/__init__.py",
        "aiconfigurator/cli/main.py",
        "aiconfigurator/generator/api.py",
        "aiconfigurator/sdk/config_adapter/schemas/estimate-request-v1.schema.json",
        "aiconfigurator_core/__init__.py",
        "aiconfigurator_core/_aiconfigurator_core.py",
        "aiconfigurator_core/_aiconfigurator_core.pyi",
        "aiconfigurator_core/model_configs/meta-llama--Meta-Llama-3.1-8B_config.json",
        "aiconfigurator_core/sdk/engine.py",
        "aiconfigurator_core/systems/h100_sxm.yaml",
    }
    missing = sorted(required - payload)
    if missing:
        raise RuntimeError(f"{wheel.name}: missing unified package payload: {missing}")

    missing_source = sorted(expected_payload - payload)
    if missing_source:
        raise RuntimeError(f"{wheel.name}: missing source-tree payload: {missing_source}")

    checks = {
        "unified native extension": any(
            name.startswith("aisimulate/_runtime.") and name.endswith((".so", ".pyd")) for name in payload
        ),
        "nested performance data": any(
            name.startswith("aiconfigurator_core/systems/data/") and name.endswith(".parquet") for name in payload
        ),
        "Rust SBOM": any(".dist-info/sboms/" in name and name.endswith(".json") for name in names),
    }
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{wheel.name}: missing {', '.join(failed)}")

    _verify_legal_files(wheel)

    requirements = metadata.get_all("Requires-Dist", [])
    split_dependencies = sorted(
        requirement
        for requirement in requirements
        if _requirement_name(requirement) in {"aisimulate-core", "aiconfigurator-core"}
    )
    if split_dependencies:
        raise RuntimeError(f"{wheel.name}: split Python core dependency remains: {split_dependencies}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()

    wheel = _one_wheel(args.dist_dir, "aisimulate-*.whl")
    payload = _verify_wheel(wheel, _source_payloads())
    _verify_rust_crate_package()
    print(f"Verified unified {wheel.name}: {len(payload)} application, SDK, data, and native payload files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
