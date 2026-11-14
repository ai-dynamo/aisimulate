# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity between the private InferenceX cache and its ``aisim-e2e/v1`` migration.

:func:`compare_cache_to_dataset` compares *data*: identities, membership, metrics,
units and provenance, observation by observation. It reads only local files. Every
difference carries a code; a code listed in :data:`EXPLAINED` has a written reason,
every other code is reported as unexplained and must be resolved or accepted by a
person. Cohort selection, freshness/image decisions, predictions and scores need the
accuracy scripts and are compared by ``scripts/e2e_accuracy/migrate_infx_parity.py``.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aisim_core.data import CategoryIndex, DatasetIndex
from aisim_core.data.e2e import (
    INFERENCEX_CONFIG_FIELDS,
    E2eDataset,
    load_e2e,
    read_raw_records,
    read_snapshot,
)
from aisim_core.data.migrate_infx import PRODUCER_URL, SNAPSHOT_DIR, InfxCache, _sha, canonical

VALUE_REL_TOL = 1e-12
PARALLELISM_KEY = re.compile(r"(?:(?:prefill|decode)_)?(?:pp|pcp_size|dcp_size)")
EXAMPLE_LIMIT = 5

# code -> why the difference is expected. Anything else is "unexplained".
EXPLAINED: dict[str, str] = {
    "metric.timing_boundary_null": (
        "The cache stamped timing_boundary='unknown' with a reason; aisim-e2e/v1 stores unknown as null."
    ),
    "metric.explicit_missing_rows": (
        "aisim-e2e/v1 stores every required (metric, statistic) the source did not report as a "
        "'missing' row (never zero); the cache simply had no row."
    ),
    "metric.name_and_statistic_renamed": (
        "Canonical names: e2e_latency -> e2el, stddev -> std, rate -> value, percentile columns -> pNN."
    ),
    "metric.value_ulp": "Latency seconds*1000 vs the cache's own conversion differ in the last bits only.",
    "metric.interactivity_added": (
        "The cache normalizer dropped the *_intvty (tokens/s/user) statistics; aisim-e2e/v1 maps them "
        "(normalization per_user) with the source values unchanged."
    ),
    "metric.source_key_parallelism": (
        "pp / pcp_size / dcp_size (and prefill_/decode_ variants) are deployment degrees the source reports under "
        "metrics. They are now parallelism dimensions pp / cp / dcp in the configuration identity, not metric rows; "
        "the cache had no row for them either."
    ),
    "metric.source_key_unmapped": (
        "The source metric has no aisim-e2e/v1 mapping yet (power, qps, token counts, cache hit rates, ...); "
        "it stays in the observation's typed unmapped_metrics_json column and the hash-linked raw-output "
        "bundle. The cache had no row for it either."
    ),
    "identity.configuration_merge_telemetry": (
        "Source configurations that differ only in the GPU-telemetry list `workers` share one configuration_id; "
        "per-observation source_config_id keeps the upstream grouping."
    ),
    "identity.configuration_split_parallelism": (
        "One source configuration spans several configuration_ids because pp/cp/dcp degrees are reported per "
        "observation (in metrics); per-observation source_config_id keeps the upstream grouping."
    ),
    "identity.configuration_ids_new": "aisim-e2e/v1 configuration_id (cfg-<sha256>) replaces the cache's hf:<sha256>.",
}


class Diffs:
    """Difference counters with a few examples each."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[Any]] = defaultdict(list)

    def add(self, code: str, example: Any = None, n: int = 1) -> None:
        self.counts[code] += n
        if example is not None and len(self.examples[code]) < EXAMPLE_LIMIT:
            self.examples[code].append(example)

    def report(self) -> dict[str, Any]:
        return {
            code: {
                "count": count,
                "status": "explained" if code in EXPLAINED else "unexplained",
                "reason": EXPLAINED.get(code),
                "examples": self.examples.get(code, []),
            }
            for code, count in sorted(self.counts.items())
        }


# canonical name/statistic for a cache metric row, per the documented renames
_NAME = {"e2e_latency": "e2el"}
_STAT = {"stddev": "std", "rate": "value"}


def _cache_stat(row: Mapping[str, Any]) -> str:
    if row["percentile"] is not None:
        pct = row["percentile"]
        return "p" + (str(int(pct)) if float(pct).is_integer() else repr(float(pct)))
    return _STAT.get(row["statistic"], row["statistic"])


def load_dataset(root: Path) -> tuple[list[E2eDataset], dict[str, str]]:
    """Every configuration of the migrated dataset and its manifest path by configuration_id."""
    dataset_index = DatasetIndex.model_validate_json((root / "catalog/dataset.json").read_bytes())
    index = CategoryIndex.model_validate_json((root / dataset_index.categories["e2e"].index.path).read_bytes())
    datasets, paths = [], {}
    for entry in index.entries:
        datasets.append(load_e2e(root, entry.manifest.path))
        paths[entry.configuration_id] = entry.manifest.path
    return datasets, paths


def compare_cache_to_dataset(cache: InfxCache, root: Path | str) -> dict[str, Any]:
    """Compare a verified cache with its migrated dataset; returns a JSON-serializable report."""
    root = Path(root)
    datasets, manifest_paths = load_dataset(root)
    new_obs = {}
    body_of = {}
    new_metrics: dict[str, list] = defaultdict(list)
    for ds in datasets:
        for o in ds.observations:
            new_obs[o.observation_id] = o
            body_of[o.observation_id] = ds
        for m in ds.metrics:
            new_metrics[m.observation_id].append(m)

    old_obs = {r["observation_id"]: r for r in cache.observation_rows}
    raw_of = {"inferencex:" + str(r["id"]): r for r in cache.records}
    identities, membership, metrics, units, provenance = (Diffs() for _ in range(5))

    # ---- identities ---------------------------------------------------------------------------
    for oid in sorted(set(old_obs) - set(new_obs)):
        identities.add("observation_missing_in_new", oid)
    for oid in sorted(set(new_obs) - set(old_obs)):
        identities.add("observation_extra_in_new", oid)
    source_to_cids: dict[str, set[str]] = defaultdict(set)
    cid_to_sources: dict[str, set[str]] = defaultdict(set)
    for oid in sorted(set(old_obs) & set(new_obs)):
        old, new, raw = old_obs[oid], new_obs[oid], raw_of[oid]
        if old["record_sha256"] != new.record_sha256:
            identities.add("record_sha256_mismatch", oid)
        if str(old["source_benchmark_id"]) != new.source_observation_id:
            identities.add("source_observation_id_mismatch", oid)
        config = {k: raw[k] for k in INFERENCEX_CONFIG_FIELDS}
        config.update({k: raw[k] for k in ("offload_mode", "workers") if k in raw})
        expected_source_config = "hf:" + _sha(canonical(config))
        if new.source_config_id != expected_source_config:
            identities.add("source_config_id_mismatch", oid)
        source_to_cids[expected_source_config].add(new.configuration_id)
        cid_to_sources[new.configuration_id].add(expected_source_config)
    identities.add("identity.configuration_ids_new", n=len(cid_to_sources))
    for sources in cid_to_sources.values():
        if len(sources) > 1:
            identities.add("identity.configuration_merge_telemetry", sorted(sources)[:2])
    for source, cids in source_to_cids.items():
        if len(cids) > 1:
            identities.add("identity.configuration_split_parallelism", {source: sorted(cids)[:2]})

    # ---- membership ---------------------------------------------------------------------------
    old_part = Counter(cache.partition_of[o] for o in old_obs)
    new_part = Counter(
        (raw_of[o]["model"], body_of[o].body.hardware, body_of[o].body.framework) for o in new_obs if o in raw_of
    )
    for key in sorted(set(old_part) | set(new_part)):
        if old_part.get(key, 0) != new_part.get(key, 0):
            membership.add(
                "partition_count_mismatch", {"partition": key, "old": old_part.get(key), "new": new_part.get(key)}
            )
    for o in new_obs:
        body = body_of[o].body
        raw = raw_of.get(o)
        if raw and (
            body.model_id != raw["model"] or body.hardware != raw["hardware"] or body.framework != raw["framework"]
        ):
            membership.add("body_identity_differs_from_record", o)
    snapshots = sorted((root / SNAPSHOT_DIR).glob("*/snapshot.json"))
    snapshot_report: dict[str, Any] = {}
    if snapshots:
        snapshot, members = read_snapshot(root, str(snapshots[0].relative_to(root)))
        if {m.observation_id for m in members} != set(old_obs):
            membership.add("snapshot_membership_mismatch")
        for m in members:
            if new_obs[m.observation_id].record_sha256 != m.record_sha256:
                membership.add("snapshot_record_hash_mismatch", m.observation_id)
        if snapshot.source_scope.get("legacy", {}).get("snapshot_id") != cache.snapshot["snapshot_id"]:
            membership.add("legacy_snapshot_id_not_recorded")
        snapshot_report = {
            "snapshot_id": snapshot.snapshot_id,
            "legacy_snapshot_id": cache.snapshot["snapshot_id"],
            "members": len(members),
            "completeness": snapshot.completeness,
            "selection_policy": snapshot.selection_policy,
        }
    else:
        membership.add("snapshot_missing")

    # ---- metrics and units --------------------------------------------------------------------
    cache_by_obs: dict[str, list] = defaultdict(list)
    for row in cache.metric_rows:
        cache_by_obs[row["observation_id"]].append(row)
    name_map: dict[tuple, set] = defaultdict(set)
    cache_keys_mapped, new_keys_mapped = Counter(), Counter()
    max_rel = 0.0
    compared = 0
    for oid in sorted(set(old_obs) & set(new_obs)):
        new_rows = {m.source_name: m for m in new_metrics[oid] if m.source_name is not None}
        explicit_missing = [m for m in new_metrics[oid] if m.source_name is None]
        if explicit_missing:
            metrics.add(
                "metric.explicit_missing_rows",
                (oid, [(m.metric, m.statistic) for m in explicit_missing][:3]),
                len(explicit_missing),
            )
        old_rows = {r["source_metric"]: r for r in cache_by_obs[oid]}
        for key, old in old_rows.items():
            cache_keys_mapped[key] += 1
            new = new_rows.get(key)
            if new is None:
                metrics.add("metric.old_row_without_new", (oid, key))
                continue
            compared += 1
            name_map[(old["name"], old["statistic"], old["percentile"])].add((new.metric, new.statistic))
            if (_NAME.get(old["name"], old["name"]), _cache_stat(old)) != (new.metric, new.statistic):
                metrics.add("metric.name_and_statistic_renamed", (oid, key, old["name"], new.metric))
            if old["unit"] != new.unit:
                units.add("unit_mismatch", (oid, key, old["unit"], new.unit))
            if old["normalization"] != new.normalization:
                units.add("normalization_mismatch", (oid, key, old["normalization"], new.normalization))
            if old["status"] != new.status:
                metrics.add("metric.status_mismatch", (oid, key, old["status"], new.status))
            elif old["status"] == "valid":
                diff = abs(old["value"] - new.value)
                rel = diff / max(abs(old["value"]), 1e-300)
                max_rel = max(max_rel, rel)
                if diff != 0 and rel > VALUE_REL_TOL:
                    metrics.add("metric.value_mismatch", (oid, key, old["value"], new.value))
                elif diff != 0:
                    metrics.add("metric.value_ulp", None)
            if old["timing_boundary"] != new.timing_boundary:
                if old["timing_boundary"] == "unknown" and new.timing_boundary is None:
                    units.add("metric.timing_boundary_null")
                else:
                    units.add("timing_boundary_mismatch", (oid, old["timing_boundary"], new.timing_boundary))
        for key, new in new_rows.items():
            new_keys_mapped[key] += 1
            if key not in old_rows:
                if new.metric == "interactivity":
                    metrics.add("metric.interactivity_added")
                else:
                    metrics.add("metric.new_row_without_old", (oid, key, new.status))
        mapped_by_new = set(new_rows)
        mapped_by_old = set(old_rows)
        for key in raw_of[oid]["metrics"]:
            if key not in mapped_by_new and key not in mapped_by_old:
                if PARALLELISM_KEY.fullmatch(key):
                    metrics.add("metric.source_key_parallelism")
                else:
                    metrics.add("metric.source_key_unmapped")
    unmapped_keys = Counter(
        key
        for oid in old_obs
        for key in raw_of[oid]["metrics"]
        if key not in {m.source_name for m in new_metrics[oid]}
        and key not in {r["source_metric"] for r in cache_by_obs[oid]}
    )
    units_seen = {
        "cache": sorted({(r["name"], r["unit"], r["normalization"]) for r in cache.metric_rows}),
        "new": sorted({(m.metric, m.unit, m.normalization) for ds in datasets for m in ds.metrics}),
    }

    # ---- provenance ---------------------------------------------------------------------------
    raw_cache: dict[str, bytes] = {}  # observation_id -> verified raw record, filled one configuration at a time
    for oid in sorted(set(old_obs) & set(new_obs)):
        old, new, raw = old_obs[oid], new_obs[oid], raw_of[oid]
        pairs = {
            "producer_run_url": (old["producer_run_url"], new.producer_id),
            "measurement_date": (old["measurement_date"], new.collected_at),
            "image": (raw.get("image"), new.image),
            "benchmark_type": (raw.get("benchmark_type"), new.benchmark_type),
            "isl": (raw.get("isl"), new.input_tokens),
            "osl": (raw.get("osl"), new.output_tokens),
            "conc": (raw.get("conc"), new.concurrency),
        }
        for name, (a, b) in pairs.items():
            if a != b:
                provenance.add(f"{name}_mismatch", (oid, a, b))
        mapped = {m.source_name for m in new_metrics[oid]}
        extras = {k: v for k, v in raw["metrics"].items() if k not in mapped}
        config = {k: raw[k] for k in INFERENCEX_CONFIG_FIELDS}
        config.update({k: raw[k] for k in ("offload_mode", "workers") if k in raw})
        if new.recipe_fingerprint != raw.get("recipe_fingerprint"):
            provenance.add("recipe_fingerprint_mismatch", oid)
        if new.source_config_json != canonical(config).decode():
            provenance.add("source_config_json_mismatch", oid)
        if (new.unmapped_metrics_json or "{}") != canonical(extras).decode():
            provenance.add("unmapped_metrics_json_mismatch", oid)
        ds = body_of[oid]
        info = next((p for p in ds.body.provenance_artifacts if p.artifact_id == new.raw_artifact_id), None)
        if info is None:
            provenance.add("raw_artifact_missing", oid)
            continue
        if info.truncated:
            provenance.add("raw_artifact_hash_or_truncation", oid)
        if new.record_sha256 != old["record_sha256"]:
            provenance.add("raw_artifact_hash_or_truncation", oid)
        if oid not in raw_cache:
            raw_cache.update(read_raw_records(root, ds.manifest, ds.observations))
        if raw_cache[oid] != canonical(raw):
            provenance.add("raw_artifact_differs_from_record", oid)
        url = raw.get("run_url")
        runs = ds.body.source_fields.get("producer_runs", {})
        if url and PRODUCER_URL.fullmatch(url) and url not in runs:
            provenance.add("producer_run_metadata_missing", (oid, url))
    redacted = Counter(p.redacted for ds in datasets for p in ds.body.provenance_artifacts)

    summary = {
        "cache": {
            "observations": len(old_obs),
            "metric_rows": len(cache.metric_rows),
            "partitions": len(cache.index["partitions"]),
        },
        "dataset": {
            "configurations": len(datasets),
            "observations": len(new_obs),
            "metric_rows": sum(len(v) for v in new_metrics.values()),
            "source_configurations": len(source_to_cids),
            "redacted_raw_artifacts": dict(redacted),
        },
        "metrics_compared": compared,
        "max_relative_value_difference": max_rel,
        "metric_name_map": {f"{a}/{b}/{c}": sorted(v) for (a, b, c), v in sorted(name_map.items(), key=str)},
        "units": units_seen,
        "unmapped_source_keys": dict(unmapped_keys.most_common()),
        "snapshot": snapshot_report,
    }
    sections = {
        "identities": identities.report(),
        "membership": membership.report(),
        "metrics": metrics.report(),
        "units": units.report(),
        "provenance": provenance.report(),
    }
    unexplained = {
        f"{section}:{code}": item["count"]
        for section, codes in sections.items()
        for code, item in codes.items()
        if item["status"] == "unexplained"
    }
    return {"summary": summary, **sections, "unexplained": unexplained}
