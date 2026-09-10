# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Privately revalidate raw GB200 holdouts before exporting public exact artifacts.

The actual allocation/image/source verification receipt remains private. Its
identity and hash must be provided from the completed staging verification,
along with the actual resolved worker snapshots. No timings are synthesized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_bytes())


def text(path):
    # Preserve exact original UTF-8 bytes; no universal-newline conversion.
    return Path(path).read_bytes().decode("utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def probe_intervals(runtime):
    wanted = set(runtime["qualification_step_ids"])
    found = []
    seen = set()
    for row in csv.DictReader(io.StringIO(runtime["scheduler_step_timeline_utc"]), delimiter="|"):
        if row["JobID"] not in wanted:
            continue
        require(row["JobID"] not in seen, "duplicate runtime probe interval")
        seen.add(row["JobID"])
        require(row["State"] == "COMPLETED" and row["ExitCode"] == "0:0", "runtime probe did not finish")
        start, end = utc(row["Start"]), utc(row["End"])
        require(start <= end, "invalid runtime probe interval")
        # The scheduler reports whole seconds. Include the entire reported end
        # second rather than inferring sub-second separation from rounded data.
        found.append((start, end + timedelta(seconds=1)))
    require(seen == wanted and len(found) == len(wanted) == 3, "missing or duplicate runtime probe interval")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repository",
        "collection-root",
        "checkpoint",
        "original-admission",
        "calibration-dir",
        "calibration-raw",
        "runtime-receipt",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    repo = args.repository.resolve(strict=True)
    sys.path[:0] = [str(repo / "data/experimental/deepseek-v41/verification-plan"), str(Path(__file__).parent)]
    comparator = importlib.import_module("compare_fpm_holdout")
    validator = importlib.import_module("admit_parity")
    calibration = comparator.load_calibration(args.calibration_dir)
    require(
        Path(validator.__file__).resolve() == Path(__file__).with_name("admit_parity.py").resolve(),
        "wrong private validator",
    )

    plan_path = args.collection_root / "collection-plan.json"
    plan = read(plan_path)
    original = read(args.original_admission)
    replayed_admission = validator.admission(args.collection_root, args.checkpoint)
    require(replayed_admission == original, "independent original Collector admission differs")
    frozen = read(repo / "data/experimental/deepseek-v41/verification-plan/heldout.json")
    require(plan["options"]["benchmark_points"]["payload"] == frozen, "plan is not the frozen holdout manifest")
    require(plan["system"] == "gb200" and plan["backend"] == "vllm", "wrong measured backend/system")
    require(plan["sha256"] != calibration["admission"]["collector_plan_sha256"], "calibration relabeled as holdout")

    runtime = read(args.runtime_receipt)
    # This explicit receipt is emitted only after recomputing the allocation's
    # staged image and installed source digests, not from the requested config.
    require(
        runtime["schema"] == "dsv41.gb200.actual-runtime.verification.v1" and runtime["valid"] is True,
        "actual runtime verification is absent",
    )
    require(
        runtime["prepared_image_sha256"] == calibration["collection"]["prepared_runtime_image_sha256"],
        "actual prepared image differs from calibration",
    )
    require(runtime["model_revision"] == calibration["collection"]["model_revision"], "actual checkpoint differs")
    require(runtime["gpu_count"] == 4 and runtime["gpu_names"] == ["NVIDIA GB200"] * 4, "wrong actual GPUs")
    require(
        runtime["source_verification_complete"] is True and runtime["staged_files_verified"] >= 67,
        "actual source or staging verification incomplete",
    )
    runtime_evidence_root = args.runtime_receipt.parent.parent
    for item in runtime["evidence"].values():
        path = runtime_evidence_root / item["path"]
        require(
            path.resolve().is_relative_to(runtime_evidence_root.resolve())
            and sha(path) == item["sha256"]
            and read(path) == item["receipt"],
            "runtime verification differs from its original evidence",
        )
    probes = probe_intervals(runtime)

    sources = [plan_path, args.checkpoint, args.original_admission, args.runtime_receipt]
    artifacts = []
    runtime_evidence = {}
    interference = {}
    calibration_cells = {c["workload_kind"]: c for c in read(args.calibration_raw / "collection-plan.json")["cells"]}
    for cell in plan["cells"]:
        phase = cell["workload_kind"]
        directory = args.collection_root / "cells" / cell["cell_id"]
        cell_path = directory / "cell.json"
        require(read(cell_path) == cell, "original cell file differs from collection plan")
        ranks = [p for p in (directory / "raw").glob("**/benchmark*.json") if read(p).get("artifact_type") == "rank"]
        require(len(ranks) == 1, "DP1 must have one native rank artifact")
        native_path = ranks[0]
        native = read(native_path)
        window = (utc(native["timing"]["started_at"]), utc(native["timing"]["completed_at"]))
        require(window[0] <= window[1], "invalid native measurement interval")
        overlap = any(a < window[1] and b > window[0] for a, b in probes)
        require(not overlap, "runtime probes overlap benchmark; preserve this attempt as diagnostic")
        interference[phase] = dict(
            native_benchmark_started_at=native["timing"]["started_at"],
            native_benchmark_completed_at=native["timing"]["completed_at"],
            readonly_runtime_probes_overlap=False,
            probe_timestamp_resolution_ns=1_000_000_000,
            probe_window="reported start through reported end plus one second; half-open",
        )
        sidecar_path = native_path.with_name(native["input_provenance"]["token_stream_manifest"]["file"])
        require(sidecar_path.parent == native_path.parent, "unsafe native sidecar path")
        provenance_path = native_path.with_name("collector-provenance.json")
        resolved_path = native_path.with_name("resolved-config-node0.json")
        preflight_path = native_path.with_name("runtime-preflight.json")
        log_path = native_path.with_name("engine.stdout.log")
        resolved = read(resolved_path)
        preflight = read(preflight_path)
        reference_raw = args.calibration_raw / "cells" / calibration_cells[phase]["cell_id"] / "raw/node0000"
        reference = read(reference_raw / "resolved-config-node0.json")
        manifest_path = reference_raw.parent.parent / "slurm-runtime/runtime-source-sha256.json"
        require(
            sha(manifest_path)
            == runtime["source_manifest_sha256"]
            == calibration["collection"]["phases"][phase]["producer"]["runtime_source_manifest_sha256"],
            "runtime source manifest differs from calibration",
        )
        actual_sources = runtime["evidence"]["source"]["receipt"]["actual_sources"]
        require(
            {key: item["sha256"] for key, item in actual_sources.items()} == read(manifest_path),
            "actual installed source hashes differ from calibration",
        )
        sources.append(manifest_path)
        require(
            preflight["status"] == "passed" and preflight["missing_fields"] == preflight["missing_methods"] == [],
            "native preflight failed",
        )
        require(
            resolved["gpu_info"]["count"] == 4
            and [g["name"] for g in resolved["gpu_info"]["gpus"]] == ["NVIDIA GB200"] * 4,
            "actual native worker GPUs differ",
        )
        require(resolved["installed_packages"] == reference["installed_packages"], "actual package set changed")
        require(
            resolved["config"]["engine_args"] == reference["config"]["engine_args"],
            "actual engine arguments changed from calibration",
        )
        # Generator materialization may explicitly add null rows/partition.
        # Compare the complete ordered geometry using the same reviewed axes.
        actual_points = resolved["config"]["_benchmark_points"]
        require(actual_points["schema_version"] == frozen["schema_version"], "actual workload schema changed")
        for point_phase in ("prefill", "decode"):
            require(
                [comparator.geometry(point_phase, p) for p in actual_points[point_phase]]
                == [comparator.geometry(point_phase, p) for p in frozen[point_phase]],
                "actual worker point geometry changed",
            )
        require(resolved["config"]["benchmark_mode"] == phase, "actual worker phase differs")
        require(resolved["vllm_version"] == calibration["collection"]["actual_vllm_package"], "actual vLLM differs")
        kernel = calibration["collection"]["actual_moe_kernel"]
        require(kernel in text(log_path), "actual MoE selection is not recorded")
        sources.extend([cell_path, native_path, sidecar_path, provenance_path, resolved_path, preflight_path, log_path])
        sources.append(reference_raw / "resolved-config-node0.json")
        artifacts.append(
            dict(
                phase=phase,
                native_json=text(native_path),
                token_stream_jsonl=text(sidecar_path),
                collector_provenance_json=text(provenance_path),
                cell_json=text(cell_path),
            )
        )
        runtime_evidence[phase] = {p.name: sha(p) for p in (resolved_path, preflight_path, log_path)}

    source_hashes = {str(p.resolve()): sha(p) for p in sources}
    payload = dict(
        schema="dsv41.fpm.holdout.v1",
        role="independent_holdout",
        status="accepted",
        runtime_identity={k: calibration["collection"][k] for k in comparator.RUNTIME_KEYS},
        artifacts=artifacts,
    )
    payload_bytes = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    receipt = dict(
        schema="dsv41.fpm.holdout.admission.v1",
        valid=True,
        normalized_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        normalizer_source_sha256=sha(__file__),
        validator_source_sha256=original["validator_source_sha256"],
        collector_plan_sha256=original["collector_plan_sha256"],
        collector_plan_file_sha256=original["plan_file_sha256"],
        collector_checkpoint_sha256=original["checkpoint_sha256"],
        calibration_files_sha256=calibration["files_sha256"],
        calibration_manifest_sha256=sha(repo / "data/experimental/deepseek-v41/verification-plan/calibration.json"),
        holdout_manifest_sha256=sha(repo / "data/experimental/deepseek-v41/verification-plan/heldout.json"),
        original_admission_json=text(args.original_admission),
        original_admission_sha256=sha(args.original_admission),
        actual_runtime_receipt_sha256=sha(args.runtime_receipt),
        actual_runtime_evidence_sha256=runtime_evidence,
        runtime_probe_interference=interference,
    )
    checked = comparator.qualify_holdout(
        payload, receipt, calibration, normalized_sha256=receipt["normalized_sha256"], normalizer_sha256=sha(__file__)
    )
    require(len(checked) == 38, "comparator did not independently admit all 38")
    require(source_hashes == {str(p.resolve()): sha(p) for p in sources}, "raw evidence changed during export")
    require(
        calibration["files_sha256"] == comparator.load_calibration(args.calibration_dir)["files_sha256"],
        "calibration changed during export",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "holdout.json").write_bytes(payload_bytes)
    (args.output_dir / "admission-receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "private-source-inventory.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    print(canonical(dict(valid=True, holdout_points=len(checked), normalized_sha256=receipt["normalized_sha256"])))


if __name__ == "__main__":
    main()
