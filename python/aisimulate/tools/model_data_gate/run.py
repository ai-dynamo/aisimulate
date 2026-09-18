#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-one shadow Model Data Quality Gate. Incomplete evidence is a failure."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

APP = "python/aisimulate"
SYSTEMS = f"{APP}/src/aiconfigurator_core/systems"
DATA = f"{SYSTEMS}/data/"
STAGES = ("Artifact integrity", "Numerical sanity", "Production reachability", "Behavior and parity")
# Unknown source paths fail open for *selection* (run the gate), never for quality.
UNRELATED_PREFIXES = ("docs/", "python/aisimulate/docs/")
UNRELATED_FILES = {"README.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "CONTRIBUTORS.md", "LICENSE"}


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, timeout=120).stdout


def resolve_sha(repo: Path, sha: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("base and head must be full lowercase commit SHAs")
    if git(repo, "rev-parse", f"{sha}^{{commit}}").decode().strip() != sha:
        raise ValueError(f"not an exact commit: {sha}")
    return sha


def changed_paths(repo: Path, base: str, head: str) -> list[str]:
    # Two-dot comparison deliberately compares the supplied trees, not a hidden
    # merge base. --no-renames makes both sides of moves/deletions observable.
    raw = git(repo, "diff", "--no-renames", "--name-only", "-z", base, head, "--")
    return sorted({item.decode("utf-8") for item in raw.split(b"\0") if item})


def applicable(paths: list[str]) -> bool:
    return any(
        path not in UNRELATED_FILES
        and not (path.startswith(UNRELATED_PREFIXES) and Path(path).suffix in {".md", ".rst"})
        for path in paths
    )


def export_snapshot(repo: Path, sha: str, destination: Path) -> None:
    paths = (SYSTEMS, f"{APP}/collector/op_backend_catalog.yaml")
    entries = git(repo, "ls-tree", "-r", "-z", sha, "--", *paths).split(b"\0")
    blobs = []
    for entry in filter(None, entries):
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        relative = PurePosixPath(raw_path.decode())
        if relative.is_absolute() or ".." in relative.parts or kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"unsupported snapshot entry (including symlinks): {relative}")
        blobs.append((oid, relative))
    if not blobs:
        raise ValueError(f"{sha}: systems data and operation catalog are missing")
    # Read Git blobs directly. git archive honors export-ignore/export-subst
    # attributes, which can omit or rewrite input and break exact-tree evidence.
    response = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repo,
        check=True,
        capture_output=True,
        input="".join(f"{oid}\n" for oid, _ in blobs).encode(),
        timeout=120,
    ).stdout
    offset = 0
    for expected_oid, relative in blobs:
        header_end = response.index(b"\n", offset)
        oid, kind, raw_size = response[offset:header_end].decode().split()
        size = int(raw_size)
        if oid != expected_oid or kind != "blob" or size < 0:
            raise ValueError("invalid Git object response")
        offset = header_end + 1
        content = response[offset : offset + size]
        if len(content) != size or response[offset + size : offset + size + 1] != b"\n":
            raise ValueError("truncated Git object response")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        offset += size + 1
    if offset != len(response):
        raise ValueError("unexpected trailing Git object response")


def result(status: str, meaning: str, **details) -> dict:
    return {"status": status, "meaning": meaning, **details}


def guarded_stage(callback) -> dict:
    try:
        value = callback()
        if not isinstance(value, dict) or value.get("status") not in {"PASS", "FAIL", "INCOMPLETE"}:
            raise ValueError("required stage returned no valid result")
        if not isinstance(value.get("meaning"), str) or not value["meaning"]:
            raise ValueError("required stage returned no explanation")
        # Reject NaN/infinity and unserializable data at the stage boundary so
        # another stage can still run and the final failure report is written.
        json.dumps(value, allow_nan=False)
        return value
    except Exception as error:
        return result(
            "FAIL", "Required stage crashed or returned invalid evidence", error=f"{type(error).__name__}: {error}"
        )


def finish(report: dict, out: Path) -> int:
    stages = report["stages"]
    if not report.get("applicable") and report.get("classification_complete"):
        report["conclusion"] = "Not applicable"
    elif set(stages) == set(STAGES) and all(stage["status"] == "PASS" for stage in stages.values()):
        report["conclusion"] = "Passed"
    else:
        report["conclusion"] = "Failed"
    report["exit_code"] = int(report["conclusion"] == "Failed")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("<", "&lt;")

    lines = [
        f"# Model Data Quality Gate — {report['conclusion']}",
        "",
        "**Phase-one shadow report. INCOMPLETE is a failing result, not successful validation.**",
        "",
        f"Base: `{cell(report['base_sha'])}`  ",
        f"Head: `{cell(report['head_sha'])}`",
        "",
        "| Stage | Result | Meaning |",
        "| --- | --- | --- |",
    ]
    for name in STAGES:
        stage = stages.get(name, result("FAIL", "Required stage result is missing"))
        lines.append(f"| {name} | {stage['status']} | {cell(stage['meaning'])} |")
    for name, stage in stages.items():
        findings = stage.get("findings", [])
        if findings or stage.get("error"):
            lines.extend(["", f"## {name}", ""])
            if stage.get("error"):
                lines.append(cell(stage["error"]))
            for finding in findings[:30]:
                lines.append(f"- {cell(json.dumps(finding, sort_keys=True, ensure_ascii=True))}")
            if len(findings) > 30:
                lines.append(f"- {len(findings) - 30} more findings in report.json.")
    lines.extend(["", "Full diagnostics, scope and coverage gaps are in `report.json`. No hardware accuracy claim."])
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return report["exit_code"]


def run_gate(repo: Path, base: str, head: str, out: Path, *, classify_only: bool = False) -> int:
    report = {
        "schema_version": 1,
        "rollout": "shadow_phase_one",
        "base_sha": base,
        "head_sha": head,
        "applicable": True,
        "classification_complete": False,
        "stages": {name: result("INCOMPLETE", "Required stage has not completed") for name in STAGES},
    }
    try:
        resolve_sha(repo, base)
        resolve_sha(repo, head)
        paths = changed_paths(repo, base, head)
        report.update(changed_paths=paths, applicable=applicable(paths), classification_complete=True)
        if not report["applicable"]:
            report["stages"] = {name: result("SKIP", "Explicitly unrelated change set") for name in STAGES}
        elif not classify_only:
            from tools.model_data_gate.stages import artifact_integrity, numerical_sanity

            with tempfile.TemporaryDirectory(prefix="model-data-gate-") as temporary:
                roots = {side: Path(temporary) / side for side in ("base", "head")}
                export_snapshot(repo, base, roots["base"])
                export_snapshot(repo, head, roots["head"])
                report["stages"][STAGES[0]] = guarded_stage(lambda: artifact_integrity(roots, paths, out))
                report["stages"][STAGES[1]] = guarded_stage(lambda: numerical_sanity(roots, paths))
                report["stages"][STAGES[2]] = result(
                    "INCOMPLETE",
                    "Native exact-key, boundary, source-classification and warm/cold probes are not integrated",
                    coverage_gaps=[
                        "packaged-but-unreachable data",
                        "unintended fallback",
                        "per-coordinate SILICON source",
                    ],
                )
                report["stages"][STAGES[3]] = result(
                    "INCOMPLETE",
                    "Exact-base/head affected predictions and native parity evidence are not integrated",
                    coverage_gaps=[
                        "supported-to-error transitions",
                        "prediction discontinuities",
                        "Rust/Python parity",
                    ],
                    existing_workflow=".github/workflows/prediction-regression-gate.yml",
                )
    except Exception as error:
        report["stages"][STAGES[0]] = result(
            "FAIL", "Cannot establish exact comparison evidence", error=f"{type(error).__name__}: {error}"
        )
    exit_code = finish(report, out)
    # Classification writes provisional failing evidence for applicable changes;
    # it is a planning command and must not be used as the validation exit code.
    return 0 if classify_only and report["classification_complete"] else exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--classify-only", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    try:
        if git(repo, "rev-parse", "HEAD").decode().strip() != args.head:
            raise ValueError("checkout HEAD must equal --head; do not execute another revision's gate")
        if git(repo, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError("tracked checkout differs from the exact head; use a clean worktree")
    except Exception as error:
        report = {
            "schema_version": 1,
            "rollout": "shadow_phase_one",
            "base_sha": args.base,
            "head_sha": args.head,
            "applicable": True,
            "classification_complete": False,
            "stages": {name: result("FAIL", "Exact checkout provenance failed", error=str(error)) for name in STAGES},
        }
        return finish(report, args.out)
    return run_gate(repo, args.base, args.head, args.out, classify_only=args.classify_only)


if __name__ == "__main__":
    raise SystemExit(main())
