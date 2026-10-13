# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free validation for public dashboard assets."""

import gzip
import hashlib
import io
import json
import math
import re
import zipfile

from scripts.fpm_accuracy.contract import COUNTS, HF_REPO, METHODS, keys, require, sha, strict_json, validate_summary

BASELINE = "8dad9634735b6875e22a90927216e542e73ba237"
MAX_FILE = 64 * 1024 * 1024
MAX_BUNDLE = 900 * 1024 * 1024


def archive_files(archive):
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        infos = bundle.infolist()
        names = [info.filename for info in infos]
        require(len(set(names)) == len(names) and len(names) <= 100000, "duplicate or excessive assets")
        require(sum(info.file_size for info in infos) <= MAX_BUNDLE, "oversized dashboard archive")
        for info in infos:
            require(re.fullmatch(r"[a-zA-Z0-9_.-]+", info.filename) is not None, "unsafe asset path")
            require(info.filename not in {".", ".."}, "unsafe asset name")
            require(info.file_size <= MAX_FILE, "oversized dashboard asset")
            require((info.external_attr >> 16) & 0o170000 != 0o120000, "symlink asset")
        return {name: bundle.read(name) for name in names}


def validate_heatmap(value, measured, *, errors=False):
    keys(value, ("x_label", "y_label", "x_bins", "y_bins", "cells"))
    for axis in ("x_bins", "y_bins"):
        bins = value[axis]
        require(isinstance(bins, list) and 0 < len(bins) <= 8, "invalid heatmap bins")
        previous = -math.inf
        for item in bins:
            keys(item, ("label", "lower", "upper"))
            require(isinstance(item["label"], str), "invalid bin label")
            require(
                all(type(item[k]) in (int, float) and math.isfinite(item[k]) for k in ("lower", "upper")),
                "invalid bin boundary",
            )
            require(previous < item["lower"] <= item["upper"], "overlapping heatmap bins")
            previous = item["upper"]
    seen = set()
    total = 0
    for cell in value["cells"]:
        keys(
            cell,
            ("x_index", "y_index", *COUNTS, "mape_pct")
            if errors
            else ("x_index", "y_index", "measured_count", "predicted_count", "mape_pct"),
        )
        x, y = cell["x_index"], cell["y_index"]
        require(
            type(x) is int and type(y) is int and 0 <= x < len(value["x_bins"]) and 0 <= y < len(value["y_bins"]),
            "invalid heatmap coordinate",
        )
        require((x, y) not in seen, "duplicate heatmap cell")
        seen.add((x, y))
        n, p = cell["measured_count"], cell["predicted_count"]
        require(type(n) is int and type(p) is int and 0 <= p <= n, "invalid heatmap counts")
        mape = cell["mape_pct"]
        require(
            (p == 0 and mape is None) or (p > 0 and type(mape) in (int, float) and math.isfinite(mape) and mape >= 0),
            "invalid cell MAPE",
        )
        if errors:
            for key in ("error_count", "unavailable_count", "tuning_error_count"):
                require(type(cell[key]) is int and 0 <= cell[key] <= n, "invalid cell outcome count")
            require(n == p + cell["error_count"] + cell["unavailable_count"], "invalid cell coverage")
        total += n
    require(total == measured, "heatmap measurement mismatch")


def validate_collection(collection, row):
    keys(collection, ("runs", "unattributed_measurements"))
    require(isinstance(collection["runs"], list), "invalid collection runs")
    require(
        type(collection["unattributed_measurements"]) is int and collection["unattributed_measurements"] >= 0,
        "invalid unattributed count",
    )
    ids = set()
    total = collection["unattributed_measurements"]
    for run in collection["runs"]:
        keys(
            run,
            (
                "id",
                "collection_type",
                "benchmark_preset",
                "benchmark_id",
                "started_at",
                "collector",
                "replay_mode",
                "dataset",
                "workload",
                "serving",
                "measurement_count",
                "availability",
                "reason",
                "charts",
            ),
        )
        require(isinstance(run["id"], str) and run["id"] not in ids, "duplicate collection run")
        ids.add(run["id"])
        require(type(run["measurement_count"]) is int and run["measurement_count"] > 0, "noncontributing run")
        total += run["measurement_count"]
        require(run["availability"] in ("available", "unavailable", "not_applicable"), "invalid chart availability")
        if run["availability"] != "available":
            require(
                run["charts"] is None and isinstance(run["reason"], str) and bool(run["reason"]),
                "missing chart unavailability reason",
            )
            continue
        charts = run["charts"]
        keys(charts, ("request_count", "stage_counts", "outcome_counts", "input", "output", "ttft", "interactivity"))
        population = charts["request_count"]
        require(type(population) is int and population >= 0, "invalid chart population")
        require(sum(charts["outcome_counts"].values()) == population, "request outcome count mismatch")
        require(charts["stage_counts"].get("profiling", 0) == population, "profiling count mismatch")
        for metric in ("input", "output"):
            histogram = charts[metric]
            keys(histogram, ("count", "bins", "p50", "p90"))
            require(0 <= histogram["count"] <= population and len(histogram["bins"]) <= 32, "invalid histogram count")
            require(sum(b["count"] for b in histogram["bins"]) == histogram["count"], "histogram population mismatch")
            for bucket in histogram["bins"]:
                keys(bucket, ("lower", "upper", "count"))
                require(
                    0 <= bucket["lower"] <= bucket["upper"] and type(bucket["count"]) is int and bucket["count"] >= 0,
                    "invalid histogram bucket",
                )
        for metric in ("ttft", "interactivity"):
            series = charts[metric]
            keys(series, ("count", "excluded", "points", "rolling_p90"))
            require(
                series["count"] == len(series["points"]) == len(series["rolling_p90"]), "time chart population mismatch"
            )
            require(series["count"] + series["excluded"] == population, "time chart exclusion mismatch")
            for points in (series["points"], series["rolling_p90"]):
                previous = -1
                for point in points:
                    require(
                        isinstance(point, list)
                        and len(point) == 2
                        and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in point),
                        "invalid chart point",
                    )
                    require(point[0] >= previous, "time chart is unordered")
                    previous = point[0]
    require(total == row["measurement_count"], "collection membership mismatch")
    if "collection" in row:
        require(len(ids) == row["collection"]["run_count"], "collection summary count mismatch")
        require(
            collection["unattributed_measurements"] == row["collection"]["unattributed_measurements"],
            "collection summary membership mismatch",
        )


def validate_details(details, summary):
    keys(details, ("schema_version", "snapshot", "rows"))
    require(
        details["schema_version"] in (1, 2) and details["snapshot"] == summary["snapshot"], "detail identity mismatch"
    )
    require(len(details["rows"]) == len(summary["rows"]), "missing detail rows")
    for detail, row in zip(details["rows"], summary["rows"], strict=True):
        require(details["schema_version"] != 2 or "collection" in detail, "missing normalized collection details")
        if "collection" in detail:
            validate_collection(detail["collection"], row)
        keys(
            detail,
            (
                "configuration_id",
                "snapshot_id",
                "membership_sha256",
                "workload_heatmaps",
                "methods",
                *(("collection",) if "collection" in detail else ()),
            ),
        )
        for key in ("configuration_id", "snapshot_id", "membership_sha256"):
            require(detail[key] == row[key], "detail membership mismatch")
        maps = detail["workload_heatmaps"]
        require(set(maps) <= {"prefill", "decode", "mixed"}, "invalid heatmap phase")
        for heatmap in maps.values():
            validate_heatmap(heatmap, sum(c["measured_count"] for c in heatmap["cells"]))
        require(
            sum(c["measured_count"] for h in maps.values() for c in h["cells"]) == row["measurement_count"],
            "wrong workload population",
        )
        keys(detail["methods"], row["results"])
        for method, candidates in detail["methods"].items():
            require(isinstance(candidates, list) and candidates, "missing method variants")
            found = False
            for candidate in candidates:
                keys(candidate, ("status", "artifact", "metrics", "_heatmaps"))
                result = {k: v for k, v in candidate.items() if k != "_heatmaps"}
                check = {**summary, "snapshot": dict(summary["snapshot"])}
                check["rows"] = [dict(row, results={**row["results"], method: result})]
                check["snapshot"]["configuration_count"] = 1
                validate_summary(check)
                found |= result == row["results"][method]
                require(candidate["_heatmaps"].keys() == maps.keys(), "missing predictor heatmap")
                for phase, heatmap in candidate["_heatmaps"].items():
                    require(
                        all(heatmap[k] == maps[phase][k] for k in ("x_label", "y_label", "x_bins", "y_bins")),
                        "predictor heatmap geometry differs",
                    )
                    require(
                        {(c["x_index"], c["y_index"]): c["measured_count"] for c in heatmap["cells"]}
                        == {(c["x_index"], c["y_index"]): c["measured_count"] for c in maps[phase]["cells"]},
                        "predictor heatmap population differs",
                    )
                    metric = candidate["metrics"][phase]
                    validate_heatmap(heatmap, metric["measured_count"], errors=True)
                    predicted = sum(c["predicted_count"] for c in heatmap["cells"])
                    require(predicted == metric["predicted_count"], "heatmap prediction mismatch")
                    if predicted:
                        weighted = (
                            sum((c["mape_pct"] or 0) * c["predicted_count"] for c in heatmap["cells"]) / predicted
                        )
                        require(
                            math.isclose(weighted, metric["mape_pct"], rel_tol=1e-9, abs_tol=1e-9),
                            "heatmap MAPE mismatch",
                        )
            require(found, "selected variant absent from detail")
    return details


def validate_visualization(files, revision, summary=None):
    manifest = strict_json(files["manifest.json"])
    require(
        manifest["schema_version"] == 1 and manifest["hf_revision"] == revision and manifest["repo_id"] == HF_REPO,
        "visualization identity mismatch",
    )
    require(set(files) == {*manifest["files"], "manifest.json", "qualification.json"}, "unexpected visualization files")
    lengths = {}
    for name, digest in manifest["files"].items():
        sha(digest, 64)
        require(hashlib.sha256(files[name]).hexdigest() == digest, "visualization checksum mismatch")
        if name.endswith(".gz"):
            with gzip.GzipFile(fileobj=io.BytesIO(files[name])) as stream:
                raw = stream.read(MAX_FILE + 1)
            require(len(raw) <= MAX_FILE, "oversized decompressed asset")
            document = strict_json(raw)
        else:
            document = strict_json(files[name])
        if name != "catalog.json":
            require(name == digest + (".json.gz" if name.endswith(".gz") else ".json"), "wrong point asset name")
            prefix = "" if name.endswith(".gz") else "sample_"
            fields = [prefix + field for field in ("points", "axis_values", "rank_details", "iteration_ids")]
            keys(document, fields)
            length = len(document[fields[0]])
            require(
                all(isinstance(document[field], list) and len(document[field]) == length for field in fields),
                "inconsistent point arrays",
            )
            lengths[name] = length
    catalog = strict_json(files["catalog.json"])
    require(
        catalog["hf_revision"] == revision
        and catalog["repo_id"] == HF_REPO
        and catalog["policy"] == manifest["policy"],
        "wrong visualization catalog",
    )
    groups = {group["id"]: group for group in catalog["groups"]}
    require(len(groups) == len(catalog["groups"]), "duplicate visualization group")
    configurations = {(item["configuration_id"], item["snapshot_id"]): item for item in catalog["catalog"]}
    require(len(configurations) == len(catalog["catalog"]), "duplicate visualization configuration")
    if summary is not None:
        for row in summary["rows"]:
            item = configurations.get((row["configuration_id"], row["snapshot_id"]))
            require(
                item is not None
                and item["membership_digest"] == row["membership_sha256"]
                and item["measured"] == row["measurement_count"]
                and item["parser_policy_id"] == row["parser_policy_id"],
                "visualization/evaluation membership mismatch",
            )
    for item in catalog["catalog"]:
        require(all(key in groups for key in item["groups"]), "missing visualization group")
        require(
            sum(groups[key]["n"] for key in item["groups"]) == item["measured"] + item["diagnostic_count"],
            "wrong visualization population",
        )
    for group in catalog["groups"]:
        require(
            all(name in manifest["files"] for name in [group["sample_file"], *group["all_files"]]),
            "unbound point assets",
        )
        require(
            lengths[group["sample_file"]] == len(group["sample_indices"])
            and sum(lengths[name] for name in group["all_files"]) == group["n"],
            "wrong visualization point count",
        )
    return catalog


def population(summary):
    """Same code can have distinct dataset populations and FPM input revisions."""
    values = sorted(
        (
            row["configuration_id"],
            row["snapshot_id"],
            row["membership_sha256"],
            row["parser_policy_id"],
            row["ordering"],
            [(method, row["results"].get(method, {}).get("artifact")) for method in METHODS],
        )
        for row in summary["rows"]
    )
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
