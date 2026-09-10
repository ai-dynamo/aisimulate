# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package actual comparison bytes, with separate model and analysis provenance."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

import render_report as render

SOURCE_FILES = {
    "model": {
        "engine_compiler": "python/aisimulate/src/aiconfigurator_core/sdk/engine.py",
        "native_python_bridge": "python/aisimulate/src/aiconfigurator_core/sdk/rust_engine_step.py",
        "model_graph": "python/aisimulate/src/aiconfigurator_core/sdk/models/deepseek_v41.py",
    },
    "analysis": {
        "trial_statistics": "analyze_e2e.py",
        "forward_identity": "compare_forward.py",
        "trace_qualification": "compare_trace.py",
        "axis_bridge": "normalize_fpm.py",
        "e2e_comparison": "compare_e2e.py",
        "segment_comparison": "compare_segments.py",
    },
}


def original_bytes(path):
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == ".gz" else raw


def private_strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from private_strings(key)
            yield from private_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from private_strings(child)
    elif isinstance(value, str) and re.search(
        r"/(?:Users|tmp|private|local|home|lustre|mnt)/|ssh://|"
        r"\b(?:login|compute|worker|node)[-_]?\d{2,}\b|"
        r"\b[a-z0-9][a-z0-9.-]*\.(?:internal|corp|local)\b",
        value,
        re.IGNORECASE,
    ):
        yield value


def commit_file(repo, commit, path):
    render.require(re.fullmatch(r"[0-9a-f]{40}", commit), "immutable source commit required")
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=repo)


def bind_sources(report, model_repo, model_commit, analysis_repo, analysis_commit):
    expected = report["prediction_sources"]
    manifest = {}
    for group, files in SOURCE_FILES.items():
        repo, commit = (model_repo, model_commit) if group == "model" else (analysis_repo, analysis_commit)
        for key, name in files.items():
            path = name if group == "model" else "data/experimental/deepseek-v41/verification-plan/" + name
            checksum = hashlib.sha256(commit_file(repo, commit, path)).hexdigest()
            render.require(checksum == expected[key], f"actual comparison source differs from committed {key}")
            manifest[key] = {"commit": commit, "path": path, "sha256": checksum}
    return manifest


def bind_archive(path, segment):
    """Hash the original archive and identify it from actual in-archive receipts."""
    sha = render.checksum(path)
    run = segment["physical_run_id"]
    expected = segment["receipt"]["source_inputs_sha256"]
    with tarfile.open(path, "r:*") as archive:
        members = archive.getmembers()
        render.require(len({m.name for m in members}) == len(members), "duplicate archive member")
        executions = []
        for member in members:
            if PurePosixPath(member.name).name != "execution.json" or not member.isfile():
                continue
            raw = archive.extractfile(member).read()
            if json.loads(raw).get("run_id") == run:
                executions.append(member)
        render.require(len(executions) == 1, "archive does not identify this physical lifecycle uniquely")
        parent = PurePosixPath(executions[0].name).parent
        bindings = {}
        for key, filename in (
            ("execution", "execution.json"),
            ("closed_audit", "closed-segment-audit.json"),
            ("combined_summary", "combined-summary.json"),
            ("status", "status.json"),
        ):
            name = str(parent / filename)
            member = next((m for m in members if m.name.removeprefix("./") == name.removeprefix("./")), None)
            render.require(member is not None and member.isfile(), "archive omits a bound physical closure input")
            checksum = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            render.require(checksum == expected[key], "archive closure bytes differ from admitted segment")
            bindings[key] = checksum
    render.require(render.checksum(path) == sha, "archive changed during binding")
    return {"physical_run_id": run, "sha256": sha, "admitted_closure_inputs_sha256": bindings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "sol",
        "hybrid",
        "silicon",
        "logical-plan",
        "main-budget",
        "pilot-statistics",
        "model-repository",
        "analysis-repository",
        "native-build-receipt",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model-source-commit", required=True)
    parser.add_argument("--analysis-source-commit", required=True)
    parser.add_argument("--raw-archive", type=Path, action="append", required=True)
    parser.add_argument("--paired-outputs", type=Path)
    parser.add_argument("--content-controls", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    paths = {mode: getattr(args, mode) for mode in render.MODES}
    paths.update(
        logical_plan=args.logical_plan,
        main_budget=args.main_budget,
        pilot_statistics=args.pilot_statistics,
        native_build_receipt=args.native_build_receipt,
    )
    if args.paired_outputs:
        paths["paired_outputs"] = args.paired_outputs
    if args.content_controls:
        paths["content_controls"] = args.content_controls
    hashes = {key: render.checksum(path) for key, path in paths.items()}
    reports = {mode: render.read(paths[mode]) for mode in render.MODES}
    plan, budget = render.read(args.logical_plan), render.read(args.main_budget)
    render.validate_reports(reports, plan, allow_partial=args.allow_partial)
    render.require(
        budget["stage_trials"] == 100 and budget["required_trials"] == 150,
        "the original capped ON sample budget changed",
    )
    first = reports["hybrid"]
    build = render.read(args.native_build_receipt)
    render.require(
        build["native_extension_sha256"] == first["prediction_sources"]["native_extension"],
        "original SILICON native build differs from actual comparison",
    )
    render.require(re.fullmatch(r"[0-9a-f]{40}", build["native_compiled_source"]), "native source commit absent")
    archives = args.raw_archive
    render.require(len(archives) == len(first["segments"]), "one original archive is required per physical segment")
    archive_receipts = [bind_archive(path, s) for path, s in zip(archives, first["segments"], strict=True)]
    sources = bind_sources(
        first, args.model_repository, args.model_source_commit, args.analysis_repository, args.analysis_source_commit
    )
    payloads = {f"comparison-{mode}.json.gz": original_bytes(paths[mode]) for mode in render.MODES}
    payloads["logical-plan.json.gz"] = original_bytes(args.logical_plan)
    payloads["main-budget.json"] = original_bytes(args.main_budget)
    payloads["pilot-statistics.json"] = original_bytes(args.pilot_statistics)
    if args.paired_outputs:
        render.validate_paired_outputs(render.read(args.paired_outputs), first)
        payloads["paired-output-ids.json"] = original_bytes(args.paired_outputs)
    if args.content_controls:
        render.validate_content_controls(render.read(args.content_controls), first, plan)
        payloads["content-controls.json"] = original_bytes(args.content_controls)
    for name, raw in payloads.items():
        render.require(not list(private_strings(json.loads(raw))), f"private operational text in {name}")
    provenance = {
        "schema": "dsv41.public.segment.report.provenance.v1",
        "logical_run_id": plan["run_id"],
        "silicon_model_source_commit": args.model_source_commit,
        "fpm_analysis_source_commit": args.analysis_source_commit,
        "native_compiled_source_commit": build["native_compiled_source"],
        "native_extension_sha256": build["native_extension_sha256"],
        "source_files": sources,
        "native_build_receipt_sha256": hashes["native_build_receipt"],
        "original_input_files_sha256": hashes,
        "physical_raw_archives": archive_receipts,
        "model_or_calibration_fitted_to_this_run": False,
        "sampling_observations_removed": False,
        "cross_lifecycle_confidence_intervals": None,
        "model_vs_analysis_source_note": (
            "SILICON model/native binary and FPM-branch analysis are separate immutable sources."
        ),
    }
    render.require(
        hashes == {key: render.checksum(path) for key, path in paths.items()}, "source changed during packaging"
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        (out / name).write_bytes(gzip.compress(raw, mtime=0) if name.endswith(".gz") else raw)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for name in ("render_report.py", "package_report.py", "compare_paired_outputs.py", "test_report.py"):
        shutil.copyfile(Path(__file__).with_name(name), out / name)
    print(
        json.dumps(
            {
                "output": str(out),
                "coverage_complete": first["summary"]["coverage_complete"],
                "physical_segments": len(first["segments"]),
            }
        )
    )


if __name__ == "__main__":
    main()
