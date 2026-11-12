#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cohort-level parity between the current InferenceX cache path and ``aisim-e2e/v1``.

The current path is ``hf_measurements.to_tables`` over the cache records followed by
``source.cohort.select_points`` (the ``hf-current-view-config-v2`` policy). The new path
rebuilds the same ``tables`` from the migrated dataset and runs the *same* selection code.

Two views of the migrated dataset are compared with the current path:

* **typed**: only manifest, observation and metric tables (what a source-agnostic evaluator
  reads), including the observation columns that carry source fields without a canonical home
  (``recipe_fingerprint``, ``source_config_json``, ``unmapped_metrics_json``). Any field the
  typed tables cannot supply surfaces as an explicit difference.
* **raw-backed**: the same tables rebuilt from the hash-linked raw-output bundle; the reference
  that shows the typed columns lose nothing.

Reads local files only. ``--check-live`` re-reads the pinned HF revision (read-only) and
confirms the local cache copy equals what the current path downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python" / "aisim" / "src"))

from aisim_core.data.migrate_infx import PRODUCER_URL, canonical, read_infx_cache
from aisim_core.data.migrate_infx_parity import load_dataset
from scripts.e2e_accuracy.source import cohort
from scripts.e2e_accuracy.source.filter import apply_filter_rules
from scripts.e2e_accuracy.source.hf_bindings import reviewed_binding, row_digest
from scripts.e2e_accuracy.source.schema import SiliconRow
from scripts.e2e_accuracy.source.staleness import (
    apply_config_staleness,
    apply_image_coherence,
    dedupe_rows,
)

CONFIG_FIELDS = (
    "model",
    "hardware",
    "framework",
    "precision",
    "spec_method",
    "disagg",
    "is_multinode",
    "prefill_tp",
    "prefill_ep",
    "prefill_dp_attention",
    "prefill_num_workers",
    "decode_tp",
    "decode_ep",
    "decode_dp_attention",
    "decode_num_workers",
    "num_prefill_gpu",
    "num_decode_gpu",
)
MAX_AGE_DAYS = 180


def to_tables(records: list[dict], runs: dict[str, dict]) -> dict[str, list]:
    """``hf_measurements.to_tables`` (the current path), imported lazily: it needs huggingface_hub."""
    from scripts.e2e_accuracy import hf_measurements

    return hf_measurements.to_tables(records, runs)


# --------------------------------------------------------------------------- new-path tables


def tables_from_dataset(root: Path, *, raw_backed: bool) -> dict[str, list]:
    """The ``tables`` the selection code consumes, rebuilt from an ``aisim-e2e/v1`` dataset.

    Typed (``raw_backed=False``): only manifest, observation and metric rows (a source-agnostic
    evaluator's view): the metric rows plus the observation's typed ``unmapped_metrics_json``,
    ``source_config_json`` and ``recipe_fingerprint`` columns. Raw-backed: the same tables rebuilt from
    each observation's hash-checked raw record in the per-configuration bundle.
    """
    from aisim_core.data.e2e import read_raw_records

    datasets, _ = load_dataset(root)
    configs: dict[str, dict] = {}
    workflows: dict[str, dict] = {}
    benches = []
    for ds in datasets:
        body = ds.body
        metrics_by_obs: dict[str, dict[str, float]] = defaultdict(dict)
        for row in ds.metrics:
            if row.status == "valid" and row.source_name is not None and row.scope == "run":
                metrics_by_obs[row.observation_id][row.source_name] = row.source_value
        runs = body.source_fields.get("producer_runs", {})
        raw_records = read_raw_records(root, ds.manifest, ds.observations) if raw_backed else {}
        for obs in ds.observations:
            if raw_backed:
                raw = json.loads(raw_records[obs.observation_id])
                config = {key: raw[key] for key in CONFIG_FIELDS}
                config.update({key: raw[key] for key in ("offload_mode", "workers") if key in raw})
                metrics = dict(raw["metrics"])
                recipe_fingerprint = raw.get("recipe_fingerprint")
            else:
                config = json.loads(obs.source_config_json)
                metrics = {**metrics_by_obs[obs.observation_id], **json.loads(obs.unmapped_metrics_json or "{}")}
                recipe_fingerprint = obs.recipe_fingerprint
            config_id = obs.source_config_id
            config = {"id": config_id, **config}
            if configs.setdefault(config_id, config) != config:
                configs[config_id] = {**configs[config_id], "_conflict": True}
            url = obs.producer_id
            run_id = url if PRODUCER_URL.fullmatch(url or "") else f"unknown-producer:{obs.source_observation_id}"
            workflows.setdefault(run_id, {"id": run_id, **runs.get(url or "", {})})
            benches.append(
                {
                    "date": obs.collected_at,
                    "isl": obs.input_tokens,
                    "osl": obs.output_tokens,
                    "conc": obs.concurrency,
                    "metrics": metrics,
                    "image": obs.image,
                    "benchmark_type": obs.benchmark_type,
                    "id": int(obs.source_observation_id),
                    "config_id": config_id,
                    "workflow_run_id": run_id,
                    "recipe_fingerprint": recipe_fingerprint,
                }
            )
    return {
        "configs": [configs[k] for k in sorted(configs)],
        "workflow_runs": [workflows[k] for k in sorted(workflows)],
        "benchmark_results": sorted(benches, key=lambda b: b["id"]),
    }


# --------------------------------------------------------------------------- comparison helpers


def _examples(items: list, limit: int = 5) -> list:
    return items[:limit]


def compare_tables(old: dict[str, list], new: dict[str, list]) -> dict[str, Any]:
    """Field-level differences between two ``tables`` dicts (counts and examples)."""
    report: dict[str, Any] = {}
    for name in ("configs", "workflow_runs", "benchmark_results"):
        o = {r["id"]: r for r in old[name]}
        n = {r["id"]: r for r in new[name]}
        fields: Counter[str] = Counter()
        examples: dict[str, list] = defaultdict(list)
        for key in sorted(set(o) & set(n), key=str):
            for field in sorted(set(o[key]) | set(n[key])):
                a, b = o[key].get(field, "<absent>"), n[key].get(field, "<absent>")
                if name == "benchmark_results" and field == "metrics":
                    a_keys, b_keys = set(a), set(b)
                    for missing in sorted(a_keys - b_keys):
                        fields[f"metrics.{missing}:missing_in_new"] += 1
                    for extra in sorted(b_keys - a_keys):
                        fields[f"metrics.{extra}:extra_in_new"] += 1
                    for shared in a_keys & b_keys:
                        if a[shared] != b[shared]:
                            fields[f"metrics.{shared}:value_differs"] += 1
                            if len(examples[f"metrics.{shared}"]) < 3:
                                examples[f"metrics.{shared}"].append([key, a[shared], b[shared]])
                    continue
                if a != b:
                    fields[field] += 1
                    if len(examples[field]) < 3:
                        examples[field].append([key, a, b])
        report[name] = {
            "old_count": len(o),
            "new_count": len(n),
            "only_in_old": _examples(sorted(set(o) - set(n), key=str)),
            "only_in_new": _examples(sorted(set(n) - set(o), key=str)),
            "n_only_in_old": len(set(o) - set(n)),
            "n_only_in_new": len(set(n) - set(o)),
            "field_differences": dict(sorted(fields.items())),
            "examples": dict(examples),
        }
    return report


def silicon_rows(tables: dict[str, list]) -> list[SiliconRow]:
    """The ``SiliconRow``s ``cohort.select_points`` builds before filtering (same construction)."""
    from dataclasses import fields

    configs = {row["id"]: row for row in tables["configs"]}
    runs = {row["id"]: row for row in tables["workflow_runs"]}
    names = {f.name for f in fields(SiliconRow)}
    rows = []
    for bench in tables["benchmark_results"]:
        config = configs[bench["config_id"]]
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
        rows.append(SiliconRow(**{k: v for k, v in values.items() if k in names}))
    return rows


def decisions(tables: dict[str, list], *, include_mtp: bool = True) -> dict[int, str]:
    """Per-benchmark fate under the cohort pipeline: ``selected`` or ``<stage>:<reason class>``.

    Mirrors ``cohort.select_points`` stage by stage; the caller checks that its selected set
    equals ``select_points``' own.
    """
    rows = silicon_rows(tables)
    fate: dict[int, str] = {}
    by_id = {r.bench_id: r for r in rows}

    def drop(dropped, stage):
        # DropRecord carries no bench id; recover the dropped rows by set difference below.
        for d in dropped:
            fate_key = (d.config_id, d.isl, d.osl, d.conc)
            pending.setdefault(fate_key, []).append(stage)

    pending: dict[tuple, list[str]] = {}
    kept = rows
    for stage, operation in (
        ("source_filter", lambda r: apply_filter_rules(r, spec_method="all" if include_mtp else "none")),
        ("superseded_row", dedupe_rows),
        ("superseded_image", apply_image_coherence),
    ):
        before = {r.bench_id for r in kept}
        kept, drops = operation(kept)
        after = {r.bench_id for r in kept}
        for bench_id in before - after:
            fate[bench_id] = stage
        assert len(before - after) == len(drops), stage
    reference = max((r.date for r in kept), default=None)
    final = []
    for method in ("none", "mtp"):
        group = [r for r in kept if r.spec_method == method]
        if group:
            keep, drops = apply_config_staleness(
                group, max_age_days=MAX_AGE_DAYS, dump_max_date=reference if method == "mtp" else None
            )
            for r in set(g.bench_id for g in group) - {k.bench_id for k in keep}:
                fate[r] = "stale"
            final.extend(keep)
    for r in final:
        fate[r.bench_id] = "selected"
    for bench_id in by_id:
        fate.setdefault(bench_id, "unaccounted")
    return fate


def selection_report(old_tables: dict, new_tables: dict, label: str) -> dict[str, Any]:
    old_points, old_selection = cohort.select_points(old_tables, MAX_AGE_DAYS, include_mtp=True)
    new_points, new_selection = cohort.select_points(new_tables, MAX_AGE_DAYS, include_mtp=True)
    old_fate, new_fate = decisions(old_tables), decisions(new_tables)
    for tables, points, fate in ((old_tables, old_points, old_fate), (new_tables, new_points, new_fate)):
        chosen = {p["benchmark"]["id"] for p in points}
        assert chosen == {b for b, f in fate.items() if f == "selected"}, "decision replica diverges from cohort"
    point_id = lambda p: hashlib.sha256(json.dumps([p["benchmark"]["id"], p["config"]["id"]]).encode()).hexdigest()
    old_ids = {point_id(p): p for p in old_points}
    new_ids = {point_id(p): p for p in new_points}
    fate_diff = Counter()
    fate_examples = []
    for bench_id in sorted(set(old_fate) | set(new_fate)):
        a, b = old_fate.get(bench_id), new_fate.get(bench_id)
        if a != b:
            fate_diff[f"{a}->{b}"] += 1
            if len(fate_examples) < 5:
                fate_examples.append([bench_id, a, b])
    row_diffs: Counter[str] = Counter()
    row_examples: dict[str, list] = defaultdict(list)
    for pid in sorted(set(old_ids) & set(new_ids)):
        a_row, b_row = old_ids[pid]["source_row"], new_ids[pid]["source_row"]
        for key in sorted(set(a_row) | set(b_row)):
            if a_row.get(key) != b_row.get(key):
                row_diffs[key] += 1
                if len(row_examples[key]) < 2:
                    row_examples[key].append([pid[:12], str(a_row.get(key))[:200], str(b_row.get(key))[:200]])
    binding_old = sum(
        1 for p in old_points if p["config"]["spec_method"] == "mtp" and reviewed_binding(SiliconRow(**p["source_row"]))
    )
    binding_new = sum(
        1 for p in new_points if p["config"]["spec_method"] == "mtp" and reviewed_binding(SiliconRow(**p["source_row"]))
    )
    digest = lambda points: hashlib.sha256(
        json.dumps(sorted((point_id(p), row_digest(SiliconRow(**p["source_row"]))) for p in points)).encode()
    ).hexdigest()
    return {
        "view": label,
        "selected": {"old": len(old_points), "new": len(new_points)},
        "selection": {"old": old_selection, "new": new_selection},
        "point_ids_only_in_old": len(set(old_ids) - set(new_ids)),
        "point_ids_only_in_new": len(set(new_ids) - set(old_ids)),
        "decision_differences": dict(fate_diff),
        "decision_examples": fate_examples,
        "decision_counts": {"old": dict(Counter(old_fate.values())), "new": dict(Counter(new_fate.values()))},
        "selected_row_field_differences": dict(sorted(row_diffs.items())),
        "selected_row_examples": dict(row_examples),
        "reviewed_mtp_bindings": {"old": binding_old, "new": binding_new},
        "cohort_input_digest": {"old": digest(old_points), "new": digest(new_points)},
        "cohort_input_digest_equal": digest(old_points) == digest(new_points),
        "excluded_by_stage_equal": old_selection["excluded"] == new_selection["excluded"],
    }


def source_config_decisions(old_tables: dict, new_tables: dict) -> dict[str, Any]:
    """Freshness (config staleness) and image-coherence decisions per upstream source configuration."""
    out = {}
    for label, tables in (("old", old_tables), ("new", new_tables)):
        rows = silicon_rows(tables)
        rows, _ = apply_filter_rules(rows, spec_method="all")
        rows, _ = dedupe_rows(rows)
        images_before = defaultdict(set)
        for r in rows:
            images_before[r.config_id].add(r.image)
        after, drops = apply_image_coherence(rows)
        images_after = defaultdict(set)
        for r in after:
            images_after[r.config_id].add(r.image)
        out[label] = {
            "configs_with_mixed_images": sorted(c for c, v in images_before.items() if len({i for i in v if i}) > 1),
            "locked_image": {
                c: sorted(i for i in v if i) for c, v in images_after.items() if len(images_before[c]) > 1
            },
        }
    return {
        "mixed_image_configs": {
            "old": len(out["old"]["configs_with_mixed_images"]),
            "new": len(out["new"]["configs_with_mixed_images"]),
        },
        "mixed_image_configs_equal": out["old"]["configs_with_mixed_images"] == out["new"]["configs_with_mixed_images"],
        "locked_image_equal": out["old"]["locked_image"] == out["new"]["locked_image"],
    }


def point_id(point: dict) -> str:
    return hashlib.sha256(json.dumps([point["benchmark"]["id"], point["config"]["id"]]).encode()).hexdigest()


def resolution_report(
    tables: dict[str, list], cache_dir: Path, workers: int, *, limit: int | None = None
) -> dict[str, dict]:
    """Select the cohort and resolve every point's deployment evidence the way the campaign does.

    Returns ``point_id -> {"digest", "status", "stage"}``. The digest covers the point's source row,
    server-log use count, resolved deployment, evidence and resolution issues, i.e. everything the
    campaign's ``cohort_sha256`` and the predictors consume. Needs network access for source recipes.
    """
    from scripts.e2e_accuracy.run_e2e_accuracy import resolve_points

    points, _ = cohort.select_points(tables, MAX_AGE_DAYS, include_mtp=True)
    for point in points:
        point["id"] = point_id(point)
    points.sort(key=lambda p: p["id"])
    if limit:
        points = points[:limit]
    resolved = resolve_points(points, cache_dir, workers, allow_estimated_defaults=True)
    out = {}
    for point in resolved:
        issues = point.get("resolution_error") or []
        out[point["id"]] = {
            "digest": hashlib.sha256(
                json.dumps(
                    {
                        "source_row": point["source_row"],
                        "server_log_uses": point.get("server_log_uses"),
                        "deployment": point.get("deployment"),
                        "evidence": point.get("evidence"),
                        "issues": issues,
                    },
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest(),
            "status": "unresolved" if issues and not point.get("deployment") else "resolved",
            "stage": issues[0].get("stage") if issues else None,
            "spec_method": point["config"]["spec_method"],
        }
    return out


def compare_resolution(old: dict[str, dict], new: dict[str, dict]) -> dict[str, Any]:
    diff = [pid for pid in sorted(set(old) & set(new)) if old[pid]["digest"] != new[pid]["digest"]]
    status_diff = [pid for pid in sorted(set(old) & set(new)) if old[pid]["status"] != new[pid]["status"]]
    return {
        "points": {"old": len(old), "new": len(new)},
        "only_in_old": len(set(old) - set(new)),
        "only_in_new": len(set(new) - set(old)),
        "resolution_digest_differences": len(diff),
        "resolution_status_differences": len(status_diff),
        "difference_examples": [
            {
                "point": p[:12],
                "old": old[p]["status"] + ":" + str(old[p]["stage"]),
                "new": new[p]["status"] + ":" + str(new[p]["stage"]),
            }
            for p in (status_diff or diff)[:5]
        ],
        "coverage": {
            "old": dict(Counter(f"{v['spec_method']}:{v['status']}:{v['stage']}" for v in old.values())),
            "new": dict(Counter(f"{v['spec_method']}:{v['status']}:{v['stage']}" for v in new.values())),
        },
    }


def legacy_report(
    db_tables: dict[str, list], hf_tables: dict[str, list], new_tables: dict[str, list]
) -> dict[str, Any]:
    """Frozen legacy cohort (database dump, policy ``gym-resolved-config-v3``) against both HF views.

    Shared benchmark IDs are compared field by field with the reviewed binding contract
    (``bind_row``: only transport IDs, run ordering metadata and workflow URLs may differ).
    """
    from scripts.e2e_accuracy.source.hf_bindings import bind_row

    db_points, db_selection = cohort.select_points(db_tables, MAX_AGE_DAYS, include_mtp=True)
    out: dict[str, Any] = {"database_selection": db_selection}
    db_rows = {r.bench_id: r for r in silicon_rows(db_tables)}
    db_fate = decisions(db_tables)
    db_selected = {p["benchmark"]["id"] for p in db_points}
    for label, tables in (("hf_current_path", hf_tables), ("aisim_e2e_raw_backed", new_tables)):
        points, selection = cohort.select_points(tables, MAX_AGE_DAYS, include_mtp=True)
        rows = {r.bench_id: r for r in silicon_rows(tables)}
        fate = decisions(tables)
        selected = {p["benchmark"]["id"] for p in points}
        shared = sorted(set(db_rows) & set(rows))
        mismatched = []
        for bench_id in shared:
            try:
                bind_row(rows[bench_id], db_rows[bench_id], 0)
            except ValueError:
                mismatched.append(bench_id)
        both_view_fate = Counter(f"db={db_fate[b]} view={fate[b]}" for b in shared if db_fate[b] != fate[b])
        out[label] = {
            "hf_view_observations": len(rows),
            "database_observations": len(db_rows),
            "shared_observations": len(shared),
            "database_only_observations": len(set(db_rows) - set(rows)),
            "view_only_observations": len(set(rows) - set(db_rows)),
            "shared_with_non_transport_field_difference": len(mismatched),
            "difference_examples": mismatched[:5],
            "selected": {"database": len(db_selected), "view": len(selected)},
            "selected_in_both": len(db_selected & selected),
            "selected_database_only": len(db_selected - selected),
            "selected_view_only": len(selected - db_selected),
            "database_only_selected_fate_in_view": dict(
                Counter(fate.get(b, "not_in_view") for b in db_selected - selected)
            ),
            "view_only_selected_fate_in_database": dict(
                Counter(db_fate.get(b, "not_in_database") for b in selected - db_selected)
            ),
            "shared_decision_differences": dict(both_view_fate),
            "view_selection": selection,
        }
    return out


def score_inputs_report(old_tables: dict[str, list], dataset: Path) -> dict[str, Any]:
    """Measured values the accuracy scores divide by, from the old tables and the canonical metric rows.

    The campaign scores ``mean_ttft``, ``mean_tpot`` and ``mean_e2el`` (seconds, times 1000 to ms) and
    per-GPU throughput. The migrated dataset stores the same values in canonical units.
    """
    from aisim_core.data.migrate_infx_parity import load_dataset

    points, _ = cohort.select_points(old_tables, MAX_AGE_DAYS, include_mtp=True)
    canonical_ms: dict[tuple[str, str], float] = {}
    for ds in load_dataset(dataset)[0]:
        for row in ds.metrics:
            if row.status == "valid" and row.source_name is not None:
                canonical_ms[(row.observation_id, row.source_name)] = row.value
    checked = Counter()
    differences = []
    for point in points:
        oid = "inferencex:" + str(point["benchmark"]["id"])
        metrics = point["benchmark"]["metrics"]
        for key, scale in (
            ("mean_ttft", 1000),
            ("mean_tpot", 1000),
            ("mean_e2el", 1000),
            ("tput_per_gpu", 1),
            ("output_tput_per_gpu", 1),
        ):
            if metrics.get(key) is None:
                checked[f"{key}:absent_in_source"] += 1
                continue
            new = canonical_ms.get((oid, key))
            checked[key] += 1
            if new != metrics[key] * scale:
                differences.append([oid, key, metrics[key] * scale, new])
    return {
        "selected_points": len(points),
        "checked": dict(checked),
        "differences": len(differences),
        "examples": differences[:5],
    }


def check_live(manifest_path: Path, local_cache: Path, runs: dict) -> dict[str, Any]:
    """Run the current HF reader against the pinned revision (read only) and compare with the local copy."""
    import tempfile

    from scripts.e2e_accuracy import hf_measurements

    manifest = json.loads(manifest_path.read_text())
    hf_measurements.validate_manifest(manifest)
    hf_measurements.producer_metadata = lambda url: runs[url]  # same GitHub metadata for both sides
    with tempfile.TemporaryDirectory() as tmp:
        target = hf_measurements.fetch(manifest, Path(tmp))
        live = json.loads(target.read_text())
    cache = read_infx_cache(local_cache)
    local = hf_measurements.to_tables(cache.records, runs)
    return {
        "revision": manifest["revision"],
        "live_equals_local_cache": canonical(live) == canonical(local),
        "live_counts": {k: len(v) for k, v in live.items()},
        "local_counts": {k: len(v) for k, v in local.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="local copy of the pinned cache revision")
    parser.add_argument("--dataset", type=Path, required=True, help="migrated aisim-e2e/v1 dataset root")
    parser.add_argument("--runs", type=Path, required=True, help="GitHub run metadata JSON (url -> metadata)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resolve-cache", type=Path, help="resolve every selected point's deployment evidence (network)"
    )
    parser.add_argument("--resolve-workers", type=int, default=8)
    parser.add_argument("--resolve-limit", type=int)
    parser.add_argument("--legacy-tables", type=Path, help="tables.json of the frozen database dump")
    parser.add_argument("--live-manifest", type=Path, help=".github/e2e-accuracy-dataset.json; enables --check-live")
    args = parser.parse_args()
    runs = json.loads(args.runs.read_text())
    cache = read_infx_cache(args.cache)
    old = to_tables(cache.records, runs)
    typed = tables_from_dataset(args.dataset, raw_backed=False)
    backed = tables_from_dataset(args.dataset, raw_backed=True)
    report: dict[str, Any] = {
        "tables": {"typed": compare_tables(old, typed), "raw_backed": compare_tables(old, backed)},
        "selection": {
            "typed": selection_report(old, typed, "typed"),
            "raw_backed": selection_report(old, backed, "raw_backed"),
        },
        "source_config_decisions": {
            "typed": source_config_decisions(old, typed),
            "raw_backed": source_config_decisions(old, backed),
        },
    }
    report["score_inputs"] = score_inputs_report(old, args.dataset)
    if args.legacy_tables:
        report["legacy_cohort"] = legacy_report(json.loads(args.legacy_tables.read_text()), old, backed)
    if args.resolve_cache:
        args.resolve_cache.mkdir(parents=True, exist_ok=True)
        resolved_old = resolution_report(old, args.resolve_cache, args.resolve_workers, limit=args.resolve_limit)
        report["resolution"] = {
            "raw_backed": compare_resolution(
                resolved_old,
                resolution_report(backed, args.resolve_cache, args.resolve_workers, limit=args.resolve_limit),
            ),
            "typed": compare_resolution(
                resolved_old,
                resolution_report(typed, args.resolve_cache, args.resolve_workers, limit=args.resolve_limit),
            ),
        }
    if args.live_manifest:
        report["live_current_path"] = check_live(args.live_manifest, args.cache, runs)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report["selection"]["typed"].items() if k.startswith(("selected", "cohort", "excluded"))}
        )
    )


if __name__ == "__main__":
    main()
