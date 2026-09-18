# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free public FPM overview contract shared by producer and publisher."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import PurePosixPath

REPO = "ai-dynamo/aisimulate"
HF_REPO = "nvidia/aisimulate-fpm-dataset"
METHODS = ("warmup", "nowarmup", "regression")
WORKLOADS = ("all", "prefill", "decode", "mixed")
COUNTS = ("measured_count", "predicted_count", "unavailable_count", "error_count", "tuning_error_count")


def eligible_branch(branch: str) -> bool:
    if branch == "main":
        return True
    match = re.fullmatch(r"release/(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", branch)
    return match is not None and tuple(map(int, match.groups())) >= (0, 12, 0)


def artifact_key(branch: str) -> str:
    return hashlib.sha256(branch.encode()).hexdigest()[:16]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value, length=40):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value), "invalid SHA")


def keys(value, expected):
    require(isinstance(value, dict) and set(value) == set(expected), "unexpected or missing public fields")


def text(value):
    require(isinstance(value, str) and 0 < len(value) <= 1024, "invalid public text")
    require(not any(ord(ch) < 32 for ch in value), "control characters in public text")
    require(
        not any(fragment in value.lower() for fragment in ("gitlab-master", "slack.com", "linear.app", "file://")),
        "internal provenance",
    )


def path(value):
    text(value)
    require(not value.startswith("/") and "\\" not in value and ":" not in value, "unsafe HF path")
    require(all(part not in {"", ".", ".."} for part in value.split("/")), "unsafe HF path")
    require(PurePosixPath(value).parts[0] == "data", "unexpected HF path")


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("non-finite JSON constant")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def validate_summary(summary):
    keys(summary, ("schema_version", "snapshot", "methods", "rows"))
    require(type(summary["schema_version"]) is int and summary["schema_version"] == 1, "unsupported schema")
    require(summary["methods"] == list(METHODS), "unexpected predictors")
    snapshot = summary["snapshot"]
    keys(
        snapshot,
        (
            "branch",
            "commit_sha",
            "hf_repo",
            "hf_revision",
            "evaluator_sha",
            "wheel_sha256",
            "completed_at",
            "run_id",
            "run_attempt",
            "configuration_count",
            "complete",
        ),
    )
    require(isinstance(snapshot["branch"], str) and eligible_branch(snapshot["branch"]), "unsupported branch")
    for name in ("commit_sha", "hf_revision", "evaluator_sha"):
        sha(snapshot[name])
    sha(snapshot["wheel_sha256"], 64)
    require(snapshot["hf_repo"] == HF_REPO and snapshot["complete"] is True, "incomplete campaign")
    for name in ("run_id", "run_attempt"):
        require(
            isinstance(snapshot[name], str) and re.fullmatch(r"[1-9][0-9]*", snapshot[name]), "invalid run identity"
        )
    text(snapshot["completed_at"])
    require(
        datetime.fromisoformat(snapshot["completed_at"].replace("Z", "+00:00")).tzinfo is not None,
        "timestamp requires timezone",
    )
    rows = summary["rows"]
    require(isinstance(rows, list) and bool(rows), "empty campaign")
    require(
        type(snapshot["configuration_count"]) is int and snapshot["configuration_count"] == len(rows),
        "incomplete configurations",
    )
    identities = set()
    for row in rows:
        keys(
            row,
            (
                "configuration_id",
                "configuration_path",
                "snapshot_id",
                "model",
                "gpu",
                "framework",
                "framework_version",
                "parallelism",
                "worker_role",
                "status",
                "protocol_id",
                "parser_policy_id",
                "ordering",
                "membership_sha256",
                "measurement_count",
                "skipped_count",
                "configuration_manifest",
                "measurement_manifest",
                "results",
            ),
        )
        for name in (
            "configuration_id",
            "snapshot_id",
            "model",
            "gpu",
            "framework",
            "framework_version",
            "parallelism",
            "worker_role",
            "ordering",
        ):
            text(row[name])
        for name in ("protocol_id", "parser_policy_id"):
            if row[name] is not None:
                text(row[name])
        for name in ("configuration_path", "configuration_manifest", "measurement_manifest"):
            path(row[name])
        sha(row["membership_sha256"], 64)
        identity = (row["configuration_id"], row["snapshot_id"])
        require(identity not in identities, "duplicate configuration")
        identities.add(identity)
        require(
            row["status"] in {"ready", "no_measurements", "unsupported_protocol", "supporting_evidence_only"},
            "unknown case status",
        )
        for name in ("measurement_count", "skipped_count"):
            require(type(row[name]) is int and row[name] >= 0, "invalid measurement count")
        keys(row["results"], METHODS if row["status"] == "ready" else ())
        for method, result in row["results"].items():
            keys(result, ("status", "artifact", "metrics"))
            require(
                result["status"] in {"evaluated", "predictor_error", "no_fpm_input", "unsupported_predictor"},
                "invalid result status",
            )
            artifact = result["artifact"]
            if artifact is not None:
                require(method != "regression", "regression has no FPM input")
                keys(artifact, ("id", "path", "sha256", "metadata_path", "metadata_sha256"))
                text(artifact["id"])
                for name in ("path", "metadata_path"):
                    path(artifact[name])
                for name in ("sha256", "metadata_sha256"):
                    sha(artifact[name], 64)
                require(
                    ("nowarmup" if ".kv-off." in artifact["path"].lower() else "warmup") == method,
                    "wrong FPM input mode",
                )
            keys(result["metrics"], WORKLOADS)
            for metric in result["metrics"].values():
                keys(metric, (*COUNTS, "mape_pct"))
                for name in COUNTS:
                    require(type(metric[name]) is int and metric[name] >= 0, "invalid metric count")
                require(
                    metric["measured_count"]
                    == sum(metric[name] for name in ("predicted_count", "unavailable_count", "error_count")),
                    "invalid coverage denominator",
                )
                require(metric["tuning_error_count"] <= metric["measured_count"], "invalid tuning error count")
                mape = metric["mape_pct"]
                if metric["predicted_count"]:
                    require(type(mape) in (int, float) and math.isfinite(mape) and mape >= 0, "invalid MAPE")
                else:
                    require(mape is None, "missing predictions cannot have MAPE")
            metrics = result["metrics"]
            require(metrics["all"]["measured_count"] == row["measurement_count"], "different predictor membership")
            for name in COUNTS:
                require(
                    metrics["all"][name] == sum(metrics[workload][name] for workload in WORKLOADS[1:]),
                    "inconsistent workload totals",
                )
    return summary
