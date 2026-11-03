#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh, local-only qualification of a pinned Gym evaluator and native binary.

The selected evaluator owns construction, splitting, scoring and admissions.
This runner only validates/loads existing inputs, optionally records predictions,
and writes evidence. Run baseline and candidate in separate interpreters with
their own installed native binaries and identical dependency versions. No old
result is imported. Trace runs are accuracy evidence; use --trace=none for cost.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import resource
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import fields, replace
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, value):
    pending = path.with_suffix(".partial")
    pending.write_bytes(encoded(value) + b"\n")
    pending.replace(path)


def verified(path, digest, base):
    path = Path(path)
    path = path if path.is_absolute() else base / path
    if not path.is_file() or sha(path) != digest:
        raise ValueError(f"Input hash mismatch: {path}")
    return path.resolve()


def capture_specifications(path, digest, selected_ids=()):
    """Validate all selected record shapes before parsing large capture streams."""
    manifest = verified(path, digest, Path.cwd())
    document = json.loads(manifest.read_text())
    # The original pinned private index predates the envelope version field.
    version = document.get("schema_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError("Unsupported capture specification")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Capture specification requires a nonempty cases list")
    required = {
        "case_id",
        "configuration_id",
        "observations_path",
        "observations_sha256",
        "worker_config",
        "model_id",
        "system",
        "gpu_family",
        "framework",
        "framework_version",
        "parallelism",
        "parallel_strategy",
        "worker_role",
        "ordering",
    }
    identifiers = set()
    for case in cases:
        if not isinstance(case, dict) or required - case.keys():
            raise ValueError("Capture specification is missing required case fields")
        if case["case_id"] in identifiers:
            raise ValueError("Capture specification repeats a case ID")
        identifiers.add(case["case_id"])
        for artifact in case.get("artifacts", []):
            if {
                "artifact_id",
                "local_path",
                "local_metadata_path",
                "sha256",
                "metadata_sha256",
                "row_count",
            } - artifact.keys():
                raise ValueError(f"Capture artifact has missing fields: {case['case_id']}")
    if selected_ids and not set(selected_ids) <= identifiers:
        raise ValueError("Unknown capture case")
    selected = [case for case in cases if not selected_ids or case["case_id"] in selected_ids]
    return manifest, selected


def load_capture(spec, manifest):
    """Transport verified native rank records without aggregating their inputs."""
    from scripts.fpm_accuracy.hf.models import (
        CaseStatus,
        ConfigurationSnapshot,
        FpmArtifact,
        MeasurementCase,
        MeasurementFile,
        MeasurementObservation,
        MeasurementReference,
        OrderingKind,
    )
    from scripts.fpm_accuracy.types.forward_pass import ForwardPassIteration, ForwardPassMetric, RequestMetrics
    from scripts.fpm_accuracy.types.worker_config import WorkerConfig, WorkerConfigRecord

    source = verified(spec["observations_path"], spec["observations_sha256"], manifest.parent)
    expected_dp = spec.get("expected_dp_ranks")
    if expected_dp is None:
        from scripts.fpm_accuracy.models.aic_config import map_worker_config_to_aic

        worker = WorkerConfigRecord(spec["configuration_id"], 1, WorkerConfig.model_validate(spec["worker_config"]))
        expected_dp = list(range(map_worker_config_to_aic(worker)["attention_dp_size"]))
    request_fields = {item.name for item in fields(RequestMetrics)}
    observations, seen, rank_ids = [], set(), set()
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt") as stream:
        for index, line in enumerate(stream):
            record = json.loads(line)
            identifier = record["observation_id"]
            if identifier in seen:
                raise ValueError(f"Duplicate observation: {identifier}")
            seen.add(identifier)
            if [row["dp_rank"] for row in record["ranks"]] != expected_dp:
                raise ValueError(f"Incorrect complete rank membership: {identifier}")
            ranks = []
            for rank_index, row in enumerate(record["ranks"]):
                rank_identity = (row["worker_id"], row["dp_rank"], row["counter_id"])
                if rank_identity in rank_ids:
                    raise ValueError(f"Reused rank observation: {rank_identity}")
                rank_ids.add(rank_identity)
                extensions = {
                    section: {key: value for key, value in row.get(section, {}).items() if key not in request_fields}
                    for section in ("scheduled_requests", "queued_requests")
                }
                ranks.append(
                    ForwardPassMetric(
                        fpm_id=index * len(expected_dp) + rank_index,
                        configuration_id=spec["configuration_id"],
                        version=row.get("version", 1),
                        worker_id=str(row["worker_id"]),
                        dp_rank=row["dp_rank"],
                        counter_id=row["counter_id"],
                        wall_time_s=row["wall_time"],
                        scheduled=RequestMetrics(
                            **{k: v for k, v in row["scheduled_requests"].items() if k in request_fields}
                        ),
                        queued=RequestMetrics(
                            **{k: v for k, v in row.get("queued_requests", {}).items() if k in request_fields}
                        ),
                        capture_id=spec["case_id"],
                        event_time=record.get("event_time") or "",
                        stream_id=record.get("stream_id", ""),
                        source_cluster_id=spec.get("source_cluster_id", ""),
                        extra_metadata={
                            "source_metadata": record.get("metadata", {}),
                            "source_request_extensions": extensions,
                            **row.get("extra_metadata", {}),
                        },
                    )
                )
            iteration = ForwardPassIteration(tuple(ranks))
            actual = iteration.observed_time_ms
            if actual is None or iteration.workload_kind == "empty":
                raise ValueError(f"Ineligible observation: {identifier}")
            if "actual_ms" in record and not math.isclose(actual, record["actual_ms"], rel_tol=1e-12):
                raise ValueError(f"Changed aggregate target: {identifier}")
            observations.append(
                MeasurementObservation(
                    observation_id=identifier,
                    configuration_id=spec["configuration_id"],
                    order=index,
                    source_file_id=spec.get("source_file_id", spec["case_id"]),
                    source_path=record.get("source_path", str(source)),
                    source_row=record.get("source_row", index + 1),
                    iteration=iteration,
                    event_time=record.get("event_time"),
                    event_time_ms=record.get("event_time_ms"),
                )
            )
    if not observations or len(observations) != spec.get("observation_count", len(observations)):
        raise ValueError("Observation count differs from the pinned case specification")
    artifacts = []
    for item in spec.get("artifacts", []):
        data = verified(item["local_path"], item["sha256"], manifest.parent)
        metadata = verified(item["local_metadata_path"], item["metadata_sha256"], manifest.parent)
        artifacts.append(
            FpmArtifact(
                artifact_id=item["artifact_id"],
                path=item.get("path", str(data)),
                metadata_path=item.get("metadata_path", str(metadata)),
                sha256=item["sha256"],
                metadata_sha256=item["metadata_sha256"],
                role=item.get("role", "self_benchmark"),
                phases=tuple(item.get("phases", ["prefill", "decode"])),
                row_count=item["row_count"],
                local_path=data,
                local_metadata_path=metadata,
                provenance_url=item.get("provenance_url", str(data)),
                metadata_provenance_url=item.get("metadata_provenance_url", str(metadata)),
            )
        )
    artifacts = tuple(artifacts)
    config = ConfigurationSnapshot(
        configuration_id=spec["configuration_id"],
        configuration_path=spec.get("configuration_path", spec["configuration_id"]),
        snapshot_id=spec.get("snapshot_id", spec["case_id"]),
        snapshot_status="local_capture",
        manifest_path=str(manifest),
        manifest_sha256=sha(manifest),
        manifest_provenance_url=str(manifest),
        model_id=spec["model_id"],
        model_revision=spec.get("model_revision"),
        system=spec["system"],
        gpu_family=spec["gpu_family"],
        framework=spec["framework"],
        framework_version=spec["framework_version"],
        parallelism=spec["parallelism"],
        parallel_strategy=spec["parallel_strategy"],
        fpm_artifacts=artifacts,
        measurements=MeasurementReference(
            spec["case_id"], "forward-pass-measurement-v1", None, str(manifest), sha(manifest), 1, str(manifest)
        ),
        worker_config_record=WorkerConfigRecord(
            spec["configuration_id"], 1, WorkerConfig.model_validate(spec["worker_config"])
        ),
        aisim_commit=spec.get("aisim_commit"),
    )
    return MeasurementCase(
        case_id=spec["case_id"],
        measurement_membership_sha256=spec["observations_sha256"],
        configuration=config,
        protocol_id="forward-pass-measurement-v1",
        status=CaseStatus.READY,
        observations=tuple(observations),
        truth_files=(
            MeasurementFile(spec["case_id"], str(source), spec["observations_sha256"], "truth", source, str(source)),
        ),
        helper_files=(),
        fpm_artifacts=artifacts,
        worker_role=spec["worker_role"],
        ordering=OrderingKind(spec["ordering"]),
        override_applied=False,
        override_sha256=None,
        override_effects=(),
        warnings=tuple(spec.get("warnings", [])),
        collection_runs=tuple(spec.get("collection_runs", [])),
        parser_policy_id="verified-native-jsonl-v1",
    )


class TracePredictor:
    """Optional one-record buffer around the unchanged public predictor contract."""

    def __init__(self, predictor, testing, stream, worker_role):
        self.inner, self.testing, self.stream = predictor, iter(testing), stream
        self.worker_role = worker_role
        self.pending = None

    def _flush(self):
        if self.pending is not None:
            self.stream.write(encoded(self.pending) + b"\n")
            self.pending = None

    def predict(self, features):
        self._flush()
        from aisimulate_core.sdk.fpm_learned import classify_workload

        observed = next(self.testing)
        self.pending = {
            "observation_id": observed.observation_id,
            "event_time_ms": observed.event_time_ms,
            "actual_ms": observed.actual_ms,
            "workload_kind": str(observed.workload_kind),
            "worker_id": observed.iteration.ranks[0].worker_id,
            "native_workload_kind": classify_workload(features.aic_payload(), self.worker_role),
            "update_submitted": False,
        }
        try:
            prediction = self.inner.predict(features)
            self.pending.update(predicted_ms=prediction.value_ms, prediction_metadata=prediction.metadata)
            return prediction
        except Exception as exc:
            self.pending.update(predicted_ms=None, prediction_error=str(exc))
            raise

    def tune(self, observations):
        # Called only by score_scenario after prediction and scoring.
        self.pending["update_submitted"] = True
        self.pending["submitted_rows_sha256"] = hashlib.sha256(
            encoded([row.tuning_payload() for row in observations])
        ).hexdigest()
        try:
            self.inner.tune(observations)
        except Exception as exc:
            self.pending["update_error"] = str(exc)
            raise

    def diagnostics(self):
        return self.inner.diagnostics()

    def close(self):
        try:
            self._flush()
        finally:
            self.inner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--evaluator-revision", required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture-spec", type=Path)
    source.add_argument("--hf-snapshot", type=Path)
    parser.add_argument("--source-sha256", required=True, help="Capture spec or HF catalog/index.json hash")
    parser.add_argument("--hf-revision")
    parser.add_argument("--case", action="append", default=[], help="Exact capture ID or HF configuration path")
    parser.add_argument("--scenario", action="append", choices=["a", "b", "c", "ac", "bc"], default=[])
    parser.add_argument("--train-percent", type=int, default=50)
    parser.add_argument("--prefix-seconds", type=float)
    parser.add_argument("--trace", choices=["none", "predictions"], default="none")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            parser.error(f"Set {name}=1 before starting Python")
    root = args.evaluator_root.resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != args.evaluator_revision:
        parser.error("Evaluator HEAD differs from the supplied immutable revision")
    sys.path.insert(0, str(root))
    from scripts.fpm_accuracy.models.scenario_predictors import create_scenario_predictor
    from scripts.fpm_accuracy.scenario_contract import PROTOCOL_ID, PROTOCOL_METADATA
    from scripts.fpm_accuracy.scenarios import (
        _has_event_chronology,
        _validate_observations,
        score_scenario,
        split_observations,
    )

    native = importlib.import_module("aisimulate._runtime")
    native_path = Path(native.__file__).resolve()
    if sha(native_path) != args.native_sha256:
        parser.error("Loaded native binary differs from the expected SHA-256")
    sources = {str(path.relative_to(root)): sha(path) for path in sorted((root / "scripts/fpm_accuracy").rglob("*.py"))}
    dependencies = {item.metadata["Name"].lower(): item.version for item in importlib.metadata.distributions()}
    runtime_sources = {}
    for package in ("aisimulate", "aisimulate_core"):
        package_root = Path(importlib.import_module(package).__file__).resolve().parent
        runtime_sources[package] = {
            "root": str(package_root),
            "files": {str(path.relative_to(package_root)): sha(path) for path in sorted(package_root.rglob("*.py"))},
        }
    identity = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "evaluator_revision": head,
        "evaluator_sources": sources,
        "evaluator_sources_sha256": hashlib.sha256(encoded(sources)).hexdigest(),
        "harness_sha256": sha(__file__),
        "native_path": str(native_path),
        "native_sha256": sha(native_path),
        "dependencies": dict(sorted(dependencies.items())),
        "runtime_python_sources": runtime_sources,
        "protocol_id": PROTOCOL_ID,
        "protocol_metadata": PROTOCOL_METADATA,
        "timing_scope": "Complete score_scenario call plus its fit/predict/update counters",
        "instrumented_accuracy_run": args.trace != "none",
        "publication_qualified": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    run = {"identity": identity, "complete": False, "cases": [], "cells": [], "failed": []}
    write(args.output / "run.json", run)
    if args.capture_spec:
        manifest, selected = capture_specifications(args.capture_spec, args.source_sha256, args.case)
        cases = (load_capture(item, manifest) for item in selected)
    else:
        from scripts.fpm_accuracy.hf.dataset import DEFAULT_REPO_ID, HfDataset, _hub_cache_blob_roots
        from scripts.fpm_accuracy.hf.overrides import HfOverrides

        snapshot = args.hf_snapshot.resolve()
        verified(snapshot / "catalog/index.json", args.source_sha256, Path.cwd())
        if snapshot.name != args.hf_revision or snapshot.parent.name != "snapshots":
            raise ValueError("HF input must be the exact pinned local cache snapshot")
        dataset = HfDataset(
            snapshot,
            repo_id=DEFAULT_REPO_ID,
            revision=args.hf_revision,
            overrides=HfOverrides(version=1),
            overrides_sha256=None,
            allowed_file_roots=_hub_cache_blob_roots(snapshot, args.hf_revision),
        )
        configurations = [
            item for item in dataset.configurations() if not args.case or item.configuration_path in args.case
        ]
        if args.case and {item.configuration_path for item in configurations} != set(args.case):
            raise ValueError("Unknown HF configuration")
        cases = (
            dataset.measurement_case(item.configuration_path, snapshot_id=item.snapshot_id) for item in configurations
        )
    for case in cases:
        _validate_observations(case)
        if not _has_event_chronology(case):
            raise ValueError(f"Case has no valid complete chronology: {case.case_id}")
        full_count = len(case.observations)
        if args.prefix_seconds is not None:
            if args.prefix_seconds <= 0:
                raise ValueError("prefix-seconds must be positive")
            start = case.observations[0].event_time_ms
            case = replace(
                case,
                observations=tuple(
                    row for row in case.observations if row.event_time_ms < start + args.prefix_seconds * 1000
                ),
            )
        case_evidence = {
            "case_id": case.case_id,
            "configuration_id": case.configuration_id,
            "source_count": full_count,
            "selected_count": len(case.observations),
            "membership_sha256": case.measurement_membership_sha256,
            "selected_ids_sha256": hashlib.sha256(
                encoded([row.observation_id for row in case.observations])
            ).hexdigest(),
            "first_event_time_ms": case.observations[0].event_time_ms,
            "last_event_time_ms": case.observations[-1].event_time_ms,
        }
        run["cases"].append(case_evidence)
        print(json.dumps(case_evidence), flush=True)
        for scenario in [] if args.validate_only else args.scenario or ["c"]:
            cell_id = f"{case.case_id}:{scenario}:{args.train_percent}"
            key = hashlib.sha256(cell_id.encode()).hexdigest()[:20]
            directory = args.output / key
            directory.mkdir()
            testing = split_observations(case.observations, scenario, train_percent=args.train_percent)[1]
            with (
                gzip.open(directory / "predictions.jsonl.gz", "wb", compresslevel=1)
                if args.trace != "none"
                else nullcontext(None)
            ) as trace:

                def factory(*positional, **kwargs):
                    model = create_scenario_predictor(*positional, **kwargs)
                    return TracePredictor(model, testing, trace, case.worker_role) if trace else model

                started = time.perf_counter()
                result = score_scenario(case, "gbdt", scenario, train_percent=args.train_percent, factory=factory)
                wall_seconds = time.perf_counter() - started
            record = {
                "cell_id": cell_id,
                "result": result,
                "wall_seconds": wall_seconds,
                "process_peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "process_peak_rss_unit": "bytes" if sys.platform == "darwin" else "KiB",
                "trace_sha256": sha(directory / "predictions.jsonl.gz") if trace else None,
            }
            write(directory / "result.json", record)
            run["cells"].append(
                {
                    "cell_id": cell_id,
                    "path": str(directory.relative_to(args.output) / "result.json"),
                    "sha256": sha(directory / "result.json"),
                }
            )
            if (
                result["status"] != "evaluated"
                or result["metrics"]["all"]["error_count"]
                or result["metrics"]["all"]["tuning_error_count"]
            ):
                run["failed"].append(cell_id)
            write(args.output / "run.json", run)
            print(
                json.dumps({"cell_id": cell_id, "status": result["status"], "metrics": result["metrics"]["all"]}),
                flush=True,
            )
            gc.collect()
        write(args.output / "run.json", run)
    run["complete"] = not run["failed"]
    run["validation_only"] = args.validate_only
    write(args.output / "run.json", run)
    if run["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
