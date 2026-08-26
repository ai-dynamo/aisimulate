# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the public AISimulate E2E accuracy overview from a validated snapshot.

The source ``predictions.json`` retains historical ``dynamo_*`` field names for
frontend compatibility. Rows with ``aisimulate_runner`` provenance are emitted
as AISimulate results. The public output contains aggregate errors and identity
dimensions only; it deliberately omits raw measurements and internal run IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
FORBIDDEN_PUBLIC_FRAGMENTS = (
    "gitlab-master.nvidia.com",
    "linear.app/nvidia",
    "slack.com/archives",
    "silicon_workflow_run_id",
)
GPUS_PER_NODE_BY_FAMILY = {
    "a100": 8,
    "b200": 8,
    "b300": 8,
    "gb200": 4,
    "gb300": 4,
    "h100": 8,
    "h200": 8,
}


class SnapshotError(ValueError):
    """Raised when the input snapshot is incomplete or inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SnapshotError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SnapshotError(f"missing required text field {field}")
    return value.strip()


def _round_metric(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else round(value, 2)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _absolute_percentage_error(prediction: float, measured: float) -> float:
    if measured == 0:
        raise SnapshotError("accuracy inputs must not contain a zero measured value")
    return abs((prediction - measured) / measured) * 100


def _normalized_hardware(hardware: str) -> str:
    return "".join(character for character in hardware.lower() if character.isalnum())


def _total_gpus(row: dict[str, Any]) -> int:
    if row.get("disagg"):
        for field in ("aisimulate_total_gpus", "dynamo_total_gpus"):
            value = _finite(row.get(field))
            if value is not None and value > 0:
                return int(value)
        raw = row.get("aic_raw")
        if isinstance(raw, dict):
            value = _finite(raw.get("num_total_gpus"))
            if value is not None and value > 0:
                return int(value)

    factors = []
    for field in ("tp_size", "pp_size", "attention_dp_size"):
        value = _finite(row.get(field))
        if value is None or value <= 0:
            raise SnapshotError(f"row is missing a positive {field}")
        factors.append(int(value))
    return math.prod(factors)


def _is_multinode(row: dict[str, Any]) -> bool:
    hardware = _required_text(row.get("hardware"), "hardware")
    normalized = _normalized_hardware(hardware)
    gpus_per_node = next(
        (
            count
            for family, count in GPUS_PER_NODE_BY_FAMILY.items()
            if normalized.startswith(family)
        ),
        None,
    )
    if gpus_per_node is None:
        return bool(row.get("is_multinode"))
    return _total_gpus(row) > gpus_per_node


def _topology_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("silicon_model"),
        row.get("isl"),
        row.get("osl"),
        row.get("hardware"),
        row.get("framework"),
        row.get("precision"),
        row.get("spec_method"),
        bool(row.get("disagg")),
        row.get("config_id"),
        row.get("tp_size"),
        row.get("pp_size"),
        row.get("attention_dp_size"),
        row.get("moe_ep_size"),
        row.get("moe_tp_size"),
    )


def _aggregate_shape_error(
    rows: list[dict[str, Any]], measured_field: str, prediction_field: str
) -> float | None:
    topologies: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        topologies[_topology_key(row)].append(row)

    weighted_error = 0.0
    comparison_count = 0
    for topology_rows in topologies.values():
        points: list[tuple[float, float, float]] = []
        for row in topology_rows:
            concurrency = _finite(row.get("conc"))
            measured = _finite(row.get(measured_field))
            prediction = _finite(row.get(prediction_field))
            if (
                concurrency is None
                or measured is None
                or prediction is None
                or measured == 0
                or prediction == 0
            ):
                continue
            points.append((concurrency, measured, prediction))
        points.sort(key=lambda point: point[0])
        if len(points) < 2:
            continue

        _, anchor_measured, anchor_prediction = points[0]
        errors = []
        for _, measured, prediction in points[1:]:
            measured_shape = measured / anchor_measured
            prediction_shape = prediction / anchor_prediction
            errors.append(
                abs((prediction_shape - measured_shape) / measured_shape) * 100
            )
        weighted_error += sum(errors)
        comparison_count += len(errors)

    if comparison_count == 0:
        return None
    return weighted_error / comparison_count


def _series_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    ttft_errors: list[float] = []
    tpot_errors: list[float] = []
    eligible_rows: list[dict[str, Any]] = []

    for row in rows:
        measured_ttft = _finite(row.get("silicon_ttft_ms"))
        measured_tpot = _finite(row.get("silicon_tpot_ms"))
        predicted_ttft = _finite(row.get(f"{prefix}_ttft_ms"))
        predicted_tpot = _finite(row.get(f"{prefix}_tpot_ms"))
        if None in (measured_ttft, measured_tpot, predicted_ttft, predicted_tpot):
            continue
        assert measured_ttft is not None
        assert measured_tpot is not None
        assert predicted_ttft is not None
        assert predicted_tpot is not None
        if measured_ttft == 0 or measured_tpot == 0:
            raise SnapshotError(
                "accuracy inputs must not contain zero measured latency"
            )
        eligible_rows.append(row)
        ttft_errors.append(_absolute_percentage_error(predicted_ttft, measured_ttft))
        tpot_errors.append(_absolute_percentage_error(predicted_tpot, measured_tpot))

    return {
        "points": len(eligible_rows),
        "ttft_mape_pct": _round_metric(_mean(ttft_errors)),
        "tpot_mape_pct": _round_metric(_mean(tpot_errors)),
        "ttft_shape_error_pct": _round_metric(
            _aggregate_shape_error(rows, "silicon_ttft_ms", f"{prefix}_ttft_ms")
        ),
        "tpot_shape_error_pct": _round_metric(
            _aggregate_shape_error(rows, "silicon_tpot_ms", f"{prefix}_tpot_ms")
        ),
    }


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"success": 0, "unsupported": 0, "failed": 0, "unknown": 0}
    for row in rows:
        status = row.get("aisimulate_status")
        if status not in counts:
            status = "unknown"
        counts[status] += 1
    return counts


def _identity_summary(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "gpu_skus": sorted(
            {_required_text(row.get("hardware"), "hardware") for row in rows}
        ),
        "frameworks": sorted(
            {_required_text(row.get("framework"), "framework") for row in rows}
        ),
        "precisions": sorted(
            {_required_text(row.get("precision"), "precision") for row in rows}
        ),
        "workloads": sorted(
            {
                f"{int(row['isl'])}:{int(row['osl'])}"
                for row in rows
                if _finite(row.get("isl")) is not None
                and _finite(row.get("osl")) is not None
            },
            key=lambda value: tuple(int(part) for part in value.split(":")),
        ),
    }


def _workload_label(workload: str) -> str:
    def short_length(value: int) -> str:
        return (
            f"{value // 1024}k" if value >= 1024 and value % 1024 == 0 else str(value)
        )

    input_tokens, output_tokens = (int(part) for part in workload.split(":"))
    return f"{short_length(input_tokens)}{short_length(output_tokens)}"


def _model_summary(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    identities = _identity_summary(rows)
    workloads = []
    for workload in identities["workloads"]:
        input_tokens, output_tokens = (int(part) for part in workload.split(":"))
        workload_rows = [
            row
            for row in rows
            if row.get("isl") == input_tokens and row.get("osl") == output_tokens
        ]
        statuses = _status_counts(workload_rows)
        gpu_summaries = []
        for gpu in sorted({str(row["hardware"]) for row in workload_rows}):
            gpu_rows = [row for row in workload_rows if str(row["hardware"]) == gpu]
            gpu_statuses = _status_counts(gpu_rows)
            gpu_summaries.append(
                {
                    "gpu": gpu,
                    "rows": len(gpu_rows),
                    "precisions": sorted({str(row["precision"]) for row in gpu_rows}),
                    "aic": _series_metrics(gpu_rows, "aic"),
                    "aisimulate": {
                        **_series_metrics(gpu_rows, "dynamo"),
                        "status_counts": gpu_statuses,
                        "coverage_pct": _round_metric(
                            gpu_statuses["success"] / len(gpu_rows) * 100
                            if gpu_rows
                            else None
                        ),
                    },
                }
            )
        workloads.append(
            {
                "identity": workload,
                "label": _workload_label(workload),
                "rows": len(workload_rows),
                "gpu_skus": sorted({str(row["hardware"]) for row in workload_rows}),
                "precisions": sorted({str(row["precision"]) for row in workload_rows}),
                "aic": _series_metrics(workload_rows, "aic"),
                "aisimulate": {
                    **_series_metrics(workload_rows, "dynamo"),
                    "status_counts": statuses,
                    "coverage_pct": _round_metric(
                        statuses["success"] / len(workload_rows) * 100
                        if workload_rows
                        else None
                    ),
                },
                "gpus": gpu_summaries,
            }
        )

    statuses = _status_counts(rows)
    hf_paths = sorted(
        {
            str(row["hf_model_path"])
            for row in rows
            if isinstance(row.get("hf_model_path"), str) and row["hf_model_path"]
        }
    )
    return {
        "model": model,
        "hf_model_paths": hf_paths,
        "rows": len(rows),
        **identities,
        "aic": _series_metrics(rows, "aic"),
        "aisimulate": {
            **_series_metrics(rows, "dynamo"),
            "status_counts": statuses,
            "coverage_pct": _round_metric(statuses["success"] / len(rows) * 100),
        },
        "workloads": workloads,
    }


def _validate_inputs(
    predictions: dict[str, Any],
    metadata: dict[str, Any],
    coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    release_tag = _required_text(
        predictions.get("release_tag"), "predictions.release_tag"
    )
    for name, document in (("metadata", metadata), ("coverage", coverage)):
        if document.get("release_tag") != release_tag:
            raise SnapshotError(
                f"{name}.release_tag does not match predictions.release_tag"
            )

    rows = predictions.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SnapshotError("predictions.rows must be a non-empty list")
    if not all(isinstance(row, dict) for row in rows):
        raise SnapshotError("every predictions row must be an object")
    if metadata.get("point_count") != len(rows):
        raise SnapshotError("metadata.point_count does not match predictions.rows")
    if coverage.get("final_unique_groups") != len(rows):
        raise SnapshotError(
            "coverage.final_unique_groups does not match predictions.rows"
        )
    if metadata.get("aic_commit_sha") != predictions.get("aic_commit_sha"):
        raise SnapshotError(
            "metadata.aic_commit_sha does not match predictions.aic_commit_sha"
        )
    if coverage.get("aic_commit_sha") != predictions.get("aic_commit_sha"):
        raise SnapshotError(
            "coverage.aic_commit_sha does not match predictions.aic_commit_sha"
        )

    status_counts = {"success": 0, "unsupported": 0, "failed": 0}
    for row in rows:
        for field in (
            "silicon_ttft_ms",
            "silicon_tpot_ms",
            "aic_ttft_ms",
            "aic_tpot_ms",
        ):
            if _finite(row.get(field)) is None:
                raise SnapshotError(f"row is missing a finite {field}")
        status = row.get("aisimulate_status")
        if status not in status_counts:
            raise SnapshotError(f"row has unknown aisimulate_status: {status!r}")
        status_counts[status] += 1

        has_ttft = _finite(row.get("dynamo_ttft_ms")) is not None
        has_tpot = _finite(row.get("dynamo_tpot_ms")) is not None
        if has_ttft != has_tpot:
            raise SnapshotError(
                "AISimulate TTFT and TPOT must be present or absent together"
            )
        if status == "success" and not has_ttft:
            raise SnapshotError("successful AISimulate row is missing latency metrics")
        if status != "success" and has_ttft:
            raise SnapshotError("non-success AISimulate row contains latency metrics")
        if (
            status == "success"
            and row.get("aisimulate_runner") != "aisimulate.engine_replay"
        ):
            raise SnapshotError(
                "successful row does not identify the public AISimulate runner"
            )

    aisimulate_run = metadata.get("aisimulate_run")
    if not isinstance(aisimulate_run, dict):
        raise SnapshotError("metadata.aisimulate_run is required")
    expected_counts = {"selected": len(rows), **status_counts}
    for field, expected in expected_counts.items():
        if aisimulate_run.get(field) != expected:
            raise SnapshotError(
                f"metadata.aisimulate_run.{field} does not match prediction rows"
            )
    return rows


def build_summary(
    predictions: dict[str, Any],
    metadata: dict[str, Any],
    coverage: dict[str, Any],
    *,
    predictions_sha256: str,
    source_url: str,
    exclude_multinode: bool = True,
) -> dict[str, Any]:
    if not source_url.startswith("https://"):
        raise SnapshotError("source URL must use https")
    all_rows = _validate_inputs(predictions, metadata, coverage)
    scoped_rows = [
        row for row in all_rows if not (exclude_multinode and _is_multinode(row))
    ]
    if not scoped_rows:
        raise SnapshotError("no rows remain after applying the publication scope")

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scoped_rows:
        by_model[_required_text(row.get("display_name"), "display_name")].append(row)

    statuses = _status_counts(scoped_rows)
    identities = _identity_summary(scoped_rows)
    aisimulate_run = metadata.get("aisimulate_run")
    if (
        not isinstance(aisimulate_run, dict)
        or aisimulate_run.get("status") != "complete"
    ):
        raise SnapshotError("metadata.aisimulate_run must be complete")
    runtime = aisimulate_run.get("runtime")
    if not isinstance(runtime, dict):
        raise SnapshotError("metadata.aisimulate_run.runtime is required")
    packages = runtime.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise SnapshotError("metadata.aisimulate_run.runtime.packages is required")

    result = {
        "schema_version": SCHEMA_VERSION,
        "title": "AISimulate E2E Accuracy Overview",
        "snapshot": {
            "release_tag": predictions["release_tag"],
            "measurement_source": "SemiAnalysis InferenceX",
            "measurement_source_url": source_url,
            "measurement_date_through": coverage.get("dump_max_date"),
            "prediction_generated_at": predictions.get("generated_at"),
            "predictions_sha256": predictions_sha256,
            "aic_commit_sha": predictions.get("aic_commit_sha"),
            "aisimulate_completed_at": aisimulate_run.get("completed_at"),
            "aisimulate_method": aisimulate_run.get("method"),
            "aisimulate_packages": dict(sorted(packages.items())),
            "aisimulate_sot_sha256": metadata.get("sha256"),
            "corrections": sorted(
                correction["id"]
                for correction in aisimulate_run.get("corrections", [])
                if isinstance(correction, dict)
                and isinstance(correction.get("id"), str)
            ),
        },
        "scope": {
            "measurement_scope": "end_to_end",
            "latency_scope": "client_observed",
            "metrics": ["TTFT", "TPOT"],
            "multinode": "excluded" if exclude_multinode else "included",
            "raw_rows": len(all_rows),
            "published_rows": len(scoped_rows),
            "excluded_multinode_rows": len(all_rows) - len(scoped_rows),
            "claim": (
                "Accuracy applies only to the exact measured operating points in this "
                "snapshot; it is not universal model, hardware, or deployment support."
            ),
        },
        "totals": {
            "models": len(by_model),
            **identities,
            "rows": len(scoped_rows),
            "aic": _series_metrics(scoped_rows, "aic"),
            "aisimulate": {
                **_series_metrics(scoped_rows, "dynamo"),
                "status_counts": statuses,
                "coverage_pct": _round_metric(
                    statuses["success"] / len(scoped_rows) * 100
                ),
            },
        },
        "models": [
            _model_summary(model, by_model[model])
            for model in sorted(by_model, key=str.casefold)
        ],
    }
    serialized = json.dumps(result, sort_keys=True)
    for fragment in FORBIDDEN_PUBLIC_FRAGMENTS:
        if fragment in serialized:
            raise SnapshotError(
                f"public output contains forbidden fragment: {fragment}"
            )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-multinode",
        action="store_true",
        help="Include multi-node rows. The public overview excludes them by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = build_summary(
        _load_json(args.predictions),
        _load_json(args.metadata),
        _load_json(args.coverage),
        predictions_sha256=_sha256(args.predictions),
        source_url=args.source_url,
        exclude_multinode=not args.include_multinode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
