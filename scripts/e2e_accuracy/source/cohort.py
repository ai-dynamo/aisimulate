# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join immutable workflow provenance and select the gym reference cohort."""

from collections import Counter
from dataclasses import asdict, fields

from .filter import apply_filter_rules
from .schema import SiliconRow
from .staleness import apply_config_staleness, apply_image_coherence, dedupe_rows

POLICY = "gym-resolved-config-v3"


def select_points(tables, max_age_days=180, *, include_mtp=False):
    configs = {row["id"]: row for row in tables["configs"]}
    runs = {row["id"]: row for row in tables["workflow_runs"]}
    benches = {row["id"]: row for row in tables["benchmark_results"]}
    if any(
        len(index) != len(tables[name])
        for index, name in ((configs, "configs"), (runs, "workflow_runs"), (benches, "benchmark_results"))
    ):
        raise ValueError("duplicate source IDs")
    rows = []
    excluded = Counter()
    names = {field.name for field in fields(SiliconRow)}
    for bench in tables["benchmark_results"]:
        config = configs.get(bench["config_id"])
        if config is None:
            excluded["orphaned_measurement"] += 1
            continue
        run = runs.get(bench["workflow_run_id"], {})
        values = {**config, **bench}
        values.update(
            run_started_at=run.get("run_started_at"),
            silicon_model=config["model"],
            bench_id=bench["id"],
            github_run_id=str(run["github_run_id"]) if run.get("github_run_id") is not None else None,
            run_attempt=run.get("run_attempt"),
            head_sha=run.get("head_sha"),
            head_branch=run.get("head_branch"),
            workflow_url=run.get("html_url"),
            infx_config=config,
            metrics=bench.get("metrics") or {},
            server_log_id=str(bench["server_log_id"]) if bench.get("server_log_id") is not None else None,
        )
        rows.append(SiliconRow(**{name: value for name, value in values.items() if name in names}))
    for stage, operation in (
        ("source_filter", apply_filter_rules),
        ("superseded_row", dedupe_rows),
        ("superseded_image", apply_image_coherence),
    ):
        rows, drops = (
            apply_filter_rules(rows, spec_method="all" if include_mtp else "none")
            if stage == "source_filter"
            else operation(rows)
        )
        excluded[stage] += len(drops)
    # Same reference as gym's default: newest row after dedupe/image coherence.
    fresh = []
    # Keep STP unchanged; age MTP against the newest eligible measurement.
    reference_date = max((row.date for row in rows), default=None)
    for method in ("none", "mtp"):
        group = [row for row in rows if row.spec_method == method]
        if group:
            kept, drops = apply_config_staleness(
                group, max_age_days=max_age_days, dump_max_date=reference_date if method == "mtp" else None
            )
            fresh.extend(kept)
            excluded["stale"] += len(drops)
    rows = fresh
    log_uses = Counter(str(b["server_log_id"]) for b in benches.values() if b.get("server_log_id") is not None)
    points = [
        {"config": configs[row.config_id], "benchmark": benches[row.bench_id], "source_row": asdict(row)}
        for row in rows
    ]
    for point in points:
        if point["config"]["spec_method"] == "mtp":
            point["server_log_uses"] = log_uses.get(point["source_row"]["server_log_id"], 0)
    return points, {
        "measurement_date_through": max(bench["date"] for bench in benches.values()),
        "selected": len(points),
        "excluded": {reason: count for reason, count in sorted(excluded.items()) if count},
    }
