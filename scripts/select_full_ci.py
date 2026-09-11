# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the expensive Full CI components required by changed paths."""

from __future__ import annotations

import argparse
import base64
import json
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

COMPONENTS = (
    "platform_wheels",
    "collector_data",
    "prediction_regression",
    "cargo_deny",
    "rust",
    "rust_feature_modes",
    "public_api_rust",
    "application_wheel",
    "application_tests",
    "python_compatibility",
    "engine_golden_regression",
    "release_artifact_contract",
)

PYTHON_PACKAGE_COMPONENTS = {
    "platform_wheels",
    "application_wheel",
    "application_tests",
    "python_compatibility",
    "release_artifact_contract",
}
# Rust is the implementation behind the Python package and prediction engine,
# so Rust changes exercise every compiled and consumer boundary except the
# collector-data validator. Cargo policy is added separately for manifest and
# dependency changes.
RUST_COMPONENTS = {
    "platform_wheels",
    "prediction_regression",
    "rust",
    "rust_feature_modes",
    "public_api_rust",
    "application_wheel",
    "application_tests",
    "python_compatibility",
    "engine_golden_regression",
    "release_artifact_contract",
}
# Prediction inputs and implementations must exercise both before/after output
# comparison and the lower-level engine parity goldens.
PREDICTION_COMPONENTS = {
    "prediction_regression",
    "engine_golden_regression",
}
# Collector source and performance data affect the data validator as well as
# the prediction and parity consumers that read the resulting records.
COLLECTOR_COMPONENTS = {
    "collector_data",
    "prediction_regression",
    "engine_golden_regression",
}

ROOT_DOCUMENTATION = {
    "AGENTS.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "CONTRIBUTORS.md",
    "DEVELOPMENT.md",
    "README.md",
    "REVIEW.md",
}
ROOT_POLICY_ONLY = {
    ".coderabbit.yaml",
    ".gitignore",
    "CODEOWNERS",
}
PACKAGE_CONTENT_FILES = {
    ".dockerignore",
    ".gitattributes",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "python/aisimulate/.dockerignore",
    "python/aisimulate/.gitattributes",
    "python/aisimulate/THIRD_PARTY_NOTICES.md",
}
PYTHON_DEPENDENCY_FILES = {
    "python/aisimulate/pyproject.toml",
    "python/aisimulate/uv.lock",
}
RUST_DEPENDENCY_FILES = {
    "Cargo.lock",
    "Cargo.toml",
    "deny.toml",
}


def _normalize(paths: Iterable[str]) -> list[str]:
    normalized = []
    for raw in paths:
        path = raw
        if not path:
            continue
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or any(ord(character) < 32 or ord(character) == 127 for character in path)
        ):
            raise ValueError(f"changed path is not repository-relative: {raw!r}")
        normalized.append(str(pure))
    return sorted(set(normalized))


def _is_documentation(path: str) -> bool:
    return path in ROOT_DOCUMENTATION or path.startswith("docs/")


def _is_policy_only(path: str) -> bool:
    return (
        path in ROOT_POLICY_ONLY
        or path == ".github/pull_request_template.md"
        or path.startswith(".github/ISSUE_TEMPLATE/")
        or path.startswith(".github/codeowners/")
    )


def _is_prediction_path(path: str) -> bool:
    return path.startswith(
        (
            "crates/core/parity_tests/perfmodel/",
            "crates/core/perfmodel/",
            "crates/core/src/engine/",
            "crates/core/src/perfmodel/",
            "crates/core/tests/perfmodel/",
            "python/aisimulate/src/aiconfigurator/",
            "python/aisimulate/src/aiconfigurator_core/",
            "python/aisimulate/src/aisimulate/aic.py",
            "python/aisimulate/tools/accuracy_regression_testing/",
            "python/aisimulate/tools/accuracy_tracking/",
            "python/aisimulate/tools/prediction_regression_gate/",
            "tests/test_aic.py",
        )
    )


def _is_collector_path(path: str) -> bool:
    return path.startswith(
        (
            "python/aisimulate/collector/",
            "python/aisimulate/tools/perf_database/",
            "python/aisimulate/tests/unit/collector/",
            "python/aisimulate/docs/perf_database/",
        )
    ) or path.startswith("python/aisimulate/src/aiconfigurator_core/systems/data/")


def _all(reason: str, paths: list[str]) -> dict[str, object]:
    return {
        "run_all": True,
        "reason": reason,
        "paths": paths,
        "components": dict.fromkeys(COMPONENTS, True),
    }


def select_components(paths: Iterable[str], *, force_all: bool = False) -> dict[str, object]:
    """Return a fail-closed Full CI plan for repository-relative paths."""

    changed = _normalize(paths)
    if force_all:
        return _all("full matrix required by workflow event", changed)
    if not changed:
        return _all("changed-path set is empty", changed)

    selected: set[str] = set()
    unknown: list[str] = []
    reasons: set[str] = set()

    for path in changed:
        if path.startswith((".github/workflows/", ".github/actions/")) or path in {
            ".github/actionlint.yaml",
            ".github/copy-pr-bot.yaml",
            "scripts/select_full_ci.py",
            "tests/test_ci_workflow_contracts.py",
        }:
            return _all(f"CI execution contract changed: {path!r}", changed)

        if _is_documentation(path) or _is_policy_only(path):
            reasons.add("documentation or review policy")
            continue

        if path in PYTHON_DEPENDENCY_FILES:
            return _all(f"Python dependency contract changed: {path!r}", changed)

        if path == "python/aisimulate/pytest.ini":
            return _all(f"shared test configuration changed: {path!r}", changed)

        if path in PACKAGE_CONTENT_FILES or path.startswith("python/aisimulate/docker/"):
            selected.update(PYTHON_PACKAGE_COMPONENTS)
            reasons.add("package or distribution metadata")
            continue

        if path in RUST_DEPENDENCY_FILES or (path.startswith("crates/") and path.endswith(("Cargo.toml", "deny.toml"))):
            selected.update(RUST_COMPONENTS)
            selected.add("cargo_deny")
            reasons.add("Rust dependency or package contract")
            continue

        if path.startswith("crates/"):
            selected.update(RUST_COMPONENTS)
            reasons.add("Rust implementation or tests")
            continue

        if path.startswith("python/aisimulate/"):
            selected.update(PYTHON_PACKAGE_COMPONENTS)
            reasons.add("Python package implementation or tests")
            if _is_prediction_path(path):
                selected.update(PREDICTION_COMPONENTS)
                reasons.add("prediction or engine behavior")
            if _is_collector_path(path):
                selected.update(COLLECTOR_COMPONENTS)
                reasons.add("collector or performance data")
            continue

        if path.startswith("tests/"):
            selected.add("application_tests")
            reasons.add("repository application tests")
            if _is_prediction_path(path):
                selected.update(PREDICTION_COMPONENTS)
            continue

        if path.startswith("examples/"):
            selected.add("application_tests")
            reasons.add("executable examples")
            continue

        if path.startswith("scripts/") or path in {"pytest.ini", "SECURITY.md"}:
            return _all(f"shared repository contract changed: {path!r}", changed)

        unknown.append(path)

    if unknown:
        return _all(f"unclassified path: {unknown[0]!r}", changed)

    reason = ", ".join(sorted(reasons)) if reasons else "documentation-only change"
    return {
        "run_all": False,
        "reason": reason,
        "paths": changed,
        "components": {component: component in selected for component in COMPONENTS},
    }


def _write_github_output(path: Path, plan: dict[str, object]) -> None:
    components = plan["components"]
    assert isinstance(components, dict)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"run_all={str(plan['run_all']).lower()}\n")
        for component in COMPONENTS:
            handle.write(f"{component}={str(components[component]).lower()}\n")


def _write_summary(path: Path, plan: dict[str, object]) -> None:
    components = plan["components"]
    assert isinstance(components, dict)
    selected = [name for name in COMPONENTS if components[name]]
    skipped = [name for name in COMPONENTS if not components[name]]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("### Full CI selection\n\n")
        handle.write(f"Reason: {plan['reason']}\n\n")
        handle.write(f"Selected: {', '.join(selected) if selected else 'none'}\n\n")
        handle.write(f"Explicitly N/A: {', '.join(skipped) if skipped else 'none'}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base64-paths-file", type=Path)
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    paths = []
    if args.base64_paths_file:
        encoded_paths = args.base64_paths_file.read_text(encoding="ascii").splitlines()
        paths = [base64.b64decode(encoded, validate=True).decode("utf-8") for encoded in encoded_paths]
    plan = select_components(paths, force_all=args.force_all)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.github_output:
        _write_github_output(args.github_output, plan)
    if args.summary:
        _write_summary(args.summary, plan)


if __name__ == "__main__":
    main()
