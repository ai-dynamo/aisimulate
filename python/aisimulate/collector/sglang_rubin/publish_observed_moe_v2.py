# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish one fixed composite profile from two qualified native GPU campaigns.

This offline publisher generates no GPU timings. Its identity binds both complete
native archives, producers, captures, reviews, and per-anchor cache provenance.
The v1 publisher and its source closure remain unchanged. Logical admission and
native bucket lookup belong to the separate consumer, never to perf-row keys.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
import statistics
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from collector.sglang_rubin import publish_observed_moe as v1
from collector.sglang_rubin.publish_observed_moe import _inventory, _read, _record, _require

MODULE = "collector.sglang_rubin.publish_observed_moe_v2"
PROFILE = "observed_glm52_nvfp4_decode_composite_v2"
IDENTITY_PATH = Path(__file__).with_name("observed_moe_v2_identity.json")
TOKENS = (1, 4, 8, 32)
LOGICAL = (3, 29, 31)
PHYSICAL = {3: 4, 29: 32, 31: 32}


def _approved_base_inventory(root, identity):
    """Admit only the fixed systems tree qualified by both strict source readers."""
    files = _inventory(root)
    tree = {
        "files": files,
        "directories": sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_dir()),
    }
    digest = hashlib.sha256(json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _require(digest == identity["base"]["systems_tree_sha256"], "Unapproved base systems tree")
    return files


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value) and value > 0, "Invalid positive finite number")
    return value


def _same_number(actual, expected):
    _require(math.isclose(_number(actual), expected, rel_tol=1e-12), "Recorded aggregate differs from raw values")


def _percent(numerator, denominator):
    return 100 * (_number(numerator) / _number(denominator) - 1)


def _tail_case_id(identity, rank, block, logical, ordinal):
    return (
        f"archive={identity['native_archive']['sha256']}/tail/rank={rank}/block={block}/isl=32768/"
        f"logical_tokens={logical}/physical_tokens={PHYSICAL[logical]}/ordinal={ordinal}/layers=3..77"
    )


def _reviews(old_review, tail_review, old, tail):
    _require(_record(old_review) == old["actual_data_review"], "Unapproved v1 actual-data review")
    first = _read(old_review)
    _require(
        first["verdict"] == "CLEAN"
        and first["status"] == old["actual_data_review_status"]
        and first["primary_fixed_mixture_timing_qualified"] is True
        and first["complete_primary_data_validated"] is True
        and first["native_archive"] == old["native_archive"]
        and first["producer_source_manifest"] == old["producer_source_manifest"],
        "Review does not qualify the exact v1 primary campaign",
    )
    _require(_record(tail_review) == tail["actual_data_review"], "Unapproved tail actual-data review")
    second = _read(tail_review)
    _require(
        second["verdict"] == "CLEAN"
        and second["status"] == tail["actual_data_review_status"]
        and second["actual_data_qualified"] is True
        and second["timing_qualified"] is True
        and second["measurement_controls_qualified"] is True
        and second["capture_semantics_valid"] is True
        and second["identities"]["producer_source_manifest_sha256"] == tail["producer_source_manifest"]["sha256"]
        and {k: second["identities"]["native_archive"][k] for k in ("sha256", "size_bytes")} == tail["native_archive"],
        "Review does not qualify the exact tail campaign",
    )


def _tail_primary(root, identity, old_root):
    """Recompute every fixed tail case and control; only L3/P4 supplies a row."""
    campaign = _read(root / "campaign.json")
    _require(campaign["status"] == "complete" and "error" not in campaign, "Incomplete tail campaign")
    for field in (
        "observer_controls_qualified",
        "structural_complete",
        "capture_observer_qualified",
        "operator_observer_drift_qualified",
        "cross_block_stable",
        "tp_mean_representative",
    ):
        _require(campaign[field] is True, f"Unqualified tail {field}")
    capture = campaign["capture_input"]
    expected_capture = identity["capture"]
    _require(
        capture["archive_identity"]
        == {"sha256": expected_capture["archive_sha256"], "size_bytes": expected_capture["archive_size_bytes"]}
        and capture["manifest_sha256"] == expected_capture["manifest_sha256"]
        and capture["review_identity"]["sha256"] == expected_capture["review_sha256"],
        "Tail capture mismatch",
    )
    order = [
        (block, logical, ordinal)
        for block in range(3)
        for logical in LOGICAL[block:] + LOGICAL[:block]
        for ordinal in v1.ORDINALS[block * 2 :] + v1.ORDINALS[: block * 2]
    ]
    expected_plan = [
        {
            "block": block,
            "isl": 32768,
            "cohort": f"isl32768-c{logical}-l{logical}-p{PHYSICAL[logical]}",
            "logical": logical,
            "physical": PHYSICAL[logical],
            "ordinal": ordinal,
        }
        for block, logical, ordinal in order
    ]
    _require(
        campaign["plan"]["primary_rank_local_cases"] == expected_plan
        and campaign["plan"]["capture_schema_version"] == 4
        and campaign["plan"]["image"] == identity["runtime"]["image_digest"],
        "Wrong tail physical plan or image",
    )
    expected_runtime = _read(root / "source/expected-runtime.json")
    values, case_ids, contributing, controls = {}, [], [], []
    for rank in range(4):
        folder = root / f"rank-{rank}"
        result = _read(folder / "result.json")
        _require(
            result["status"] == "complete"
            and result["rank"] == rank
            and result["native_boundary"] == identity["producer_boundary"]
            and result["weights_before_after_equal"] is True,
            "Incomplete or wrong tail rank",
        )
        for field, expected in {
            "primary_cases": 72,
            "helper_results": 216,
            "case_observer_checks": 144,
            "cache_before_after_equal": True,
            "inputs_before_after_equal": True,
            "one_cohort_resident": True,
            "legacy_subset_or_per_layer_diagnostics_executed": False,
        }.items():
            _require(result["coverage"][field] == expected, "Incomplete tail coverage")
        weights = _read(folder / "weights-before.json")
        _require(
            weights == _read(folder / "weights-after.json") == _read(old_root / f"rank-{rank}/weights-before.json"),
            "Tail weights changed or differ from v1",
        )
        cache = _record(folder / "saved-native-cache.json")
        _require(cache == identity["rank_caches"][str(rank)], "Tail saved cache mismatch")
        preflight = _read(folder / "preflight.json")
        _require(
            preflight["machine"] == "aarch64"
            and preflight["versions"] == expected_runtime["package_versions"]
            and preflight["versions"]["sglang"] == identity["runtime"]["version"]
            and preflight["torch_cuda_version"] == expected_runtime["torch_cuda_version"],
            "Tail runtime mismatch",
        )
        rows = _read(folder / "primary-measurements.json")
        _require(
            [tuple(row[k] for k in ("block", "logical", "ordinal")) for row in rows] == order,
            "Missing, duplicate or reordered tail case",
        )
        previous_end = 0
        for row, case, (block, logical, ordinal) in zip(rows, expected_plan, order, strict=True):
            _require(
                {k: row[k] for k in ("block", "isl", "cohort", "logical", "physical", "ordinal")} == case
                and row["rank"] == rank
                and row["native_rank_cache_sha256"] == cache["sha256"],
                "Tail logical/physical/rank/cache identity mismatch",
            )
            captured = row["captured_input"]
            _require(
                captured["capture_source_manifest_sha256"] == expected_capture["source_manifest_sha256"]
                and captured["capture_archive_sha256"] == expected_capture["archive_sha256"]
                and captured["ordinal"] == captured["eligible_ordinal"] == ordinal
                and captured["actual_logical_tokens"] == logical
                and captured["actual_physical_graph_tokens"] == PHYSICAL[logical]
                and captured["valid_row_range"] == [0, logical]
                and captured["padding_range"] == [logical, PHYSICAL[logical]]
                and captured["padding_tokens"] == PHYSICAL[logical] - logical,
                "Tail native padding/input identity mismatch",
            )
            durations = []
            for phase in ("before", "observed", "after"):
                helper = row[phase]
                _require(
                    type(helper["host_start_ns"]) is int
                    and type(helper["host_end_ns"]) is int
                    and previous_end < helper["host_start_ns"] < helper["host_end_ns"],
                    "Tail helper order mismatch",
                )
                previous_end = helper["host_end_ns"]
                raw = helper["raw"]
                _require(
                    raw["used_cuda_graph"] is True
                    and raw["num_runs_executed"] == 10
                    and raw["throttled"] is False
                    and raw["power_stats"] is None,
                    "Unqualified tail timing method",
                )
                durations.append(_number(raw["latency_ms"]))
            before, observed, after = durations
            mean = statistics.fmean((before, after))
            _same_number(row["plain_mean_ms"], mean)
            _same_number(row["normalized_ms_per_layer"], mean / 75)
            _require(
                row["observed"]["external_event_count"] == 2
                and row["observed"]["event_interval_substituted_for_primary"] is False
                and row["control"]["qualified"] is True
                and row["control"]["limit_percent"] == 5
                and row["control"]["overhead_subtracted"] is False,
                "Wrong observer control or substituted instrumented timing",
            )
            controls.extend((_percent(after, before), _percent(observed, mean)))
            values[rank, block, logical, ordinal] = mean / 75
            case_id = _tail_case_id(identity, rank, block, logical, ordinal)
            case_ids.append(case_id)
            if logical == 3:
                contributing.append(case_id)
    _require(len(values) == len(set(case_ids)) == 288 and len(controls) == 576, "Incomplete tail case controls")
    adjacent = [
        _percent(values[r, b + 1, n, o], values[r, b, n, o])
        for n, r, o, b in itertools.product(LOGICAL, range(4), v1.ORDINALS, range(2))
    ]
    spreads = [
        _percent(max(v), statistics.fmean(v))
        for n, b, o in itertools.product(LOGICAL, range(3), v1.ORDINALS)
        for v in [[values[r, b, n, o] for r in range(4)]]
    ]
    _require(len(adjacent) == 192 and len(spreads) == 72, "Incomplete stability controls")
    _require(all(abs(v) <= 5 + 1e-12 for v in controls + adjacent + spreads), "Tail controls exceed fixed 5% gate")
    means = {n: statistics.fmean(v for key, v in values.items() if key[2] == n) for n in LOGICAL}
    for n in LOGICAL:
        _same_number(
            campaign["summary"]["cohorts"][f"isl32768-c{n}-l{n}-p{PHYSICAL[n]}"]["mean_ms_per_layer"], means[n]
        )
    return (
        means[3],
        case_ids,
        contributing,
        {
            "case_controls": {"count": 576, "maximum_absolute_percent": max(map(abs, controls))},
            "adjacent_controls": {"count": 192, "maximum_absolute_percent": max(map(abs, adjacent))},
            "rank_spreads": {"count": 72, "maximum_percent": max(spreads)},
            "complete_cohort_means_ms": means,
            "validation_only_logical_cohorts": [29, 31],
        },
    )


def publish(
    *,
    base_systems,
    output_systems,
    v1_archive,
    v1_review,
    tail_archive,
    tail_review,
    validate_only=False,
    publisher_source=None,
):
    """Replay the retained qualified composite publisher without changing its pins."""
    from collector.sglang_rubin.replay_observed_moe import replay

    return replay(
        version="v2",
        publisher_source=publisher_source,
        base_systems=base_systems,
        output_systems=output_systems,
        v1_archive=v1_archive,
        v1_review=v1_review,
        tail_archive=tail_archive,
        tail_review=tail_review,
        validate_only=validate_only,
    )


def _publish_current(
    *, base_systems, output_systems, v1_archive, v1_review, tail_archive, tail_review, validate_only=False
):
    """Validate complete pinned inputs and atomically append a distinct profile."""
    import pyarrow.parquet as pq
    import yaml

    from collector import helper, provenance

    identity, old = _read(IDENTITY_PATH), _read(v1.IDENTITY_PATH)
    _require(
        identity["module"] == MODULE
        and identity["distribution"] == PROFILE
        and identity["physical_tokens"] == list(TOKENS)
        and identity["case_roles"] == {"total": 1152, "row_contributing": 960, "validation_only": 192},
        "Wrong composite publication identity",
    )
    package = Path(__file__).resolve().parents[2]
    closures = provenance.load_closures(package / "collector/hash_closures.yaml")
    _require(
        _record(v1.IDENTITY_PATH) == identity["v1_identity_file"]
        and provenance.collector_hash(v1.MODULE, package, closures) == identity["v1_collector_hash"],
        "v1 publisher source closure changed",
    )
    module_hash = provenance.collector_hash(MODULE, package, closures)
    base, output = Path(base_systems), Path(output_systems)
    _require(base.is_dir() and not base.is_symlink(), "Unsafe base systems root")
    _require(not output.exists() and not output.is_symlink(), "Output systems root already exists")
    _require(not output.resolve().is_relative_to(base.resolve()), "Output cannot be inside base systems")
    before = _approved_base_inventory(base, identity)
    relative = Path(identity["table_relative_path"])
    _require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe table path")
    table = base / relative
    for name in ("moe_perf.parquet", "collection_meta.yaml"):
        _require(_record(table / name) == identity["base"][name], "Unapproved 84-row base")
    old_table = pq.read_table(table / "moe_perf.parquet")
    old_rows = old_table.to_pylist()
    old_meta = yaml.safe_load((table / "collection_meta.yaml").read_text())
    provenance.validate_collection_meta_for_update(old_meta)
    _require(
        len(old_rows) == identity["base"]["rows"] == old_meta["tables"]["moe_perf"]["rows"] == 84
        and old_meta["schema_version"] == 2
        and old_meta["runtime"] == identity["runtime"] == old["runtime"],
        "Wrong base count/schema/runtime",
    )
    history = old_meta["tables"]["moe_perf"]["collections"]
    _require(
        sum(event["rows"] for event in history) == 84
        and all(
            history[-1].get(key) == value
            for key, value in {
                "collector_ref": v1.MODULE,
                "collector_hash": identity["v1_collector_hash"],
                "case_plan_hash": old["case_plan_hash"],
                "rows": 3,
                "source_campaign_rows": 864,
                "source_campaign_status": "complete",
                "status": "complete",
                "runtime": old["runtime"],
            }.items()
        ),
        "Incomplete or mismatched original collection history",
    )
    _require(all(row["distribution"] != PROFILE for row in old_rows), "Composite profile already present")
    old_profile = [row for row in old_rows if row["distribution"] == v1.PROFILE]
    _require(len(old_profile) == 3 and [row["num_tokens"] for row in old_profile] == [1, 8, 32], "Wrong v1 anchors")
    for row in old_rows:
        _number(row["latency"])
        _require(all(row[key] == value for key, value in identity["row_fields"].items()), "Base row identity mismatch")
    keys = [json.dumps({k: v for k, v in row.items() if k != "latency"}, sort_keys=True) for row in old_rows]
    _require(len(keys) == len(set(keys)), "Duplicate base row keys")
    _reviews(Path(v1_review), Path(tail_review), old, identity["tail"])
    with tempfile.TemporaryDirectory(prefix="aisim-observed-moe-v2-", dir=output.parent) as temporary:
        work = Path(temporary)
        natives = {}
        for role, archive, spec in (("v1", v1_archive, old), ("tail", tail_archive, identity["tail"])):
            retained = work / f"{role}.tar.gz"
            shutil.copyfile(archive, retained)
            natives[role] = v1._verify_native(retained, work / role, spec)
        means, old_cases = v1._primary(natives["v1"], old)
        _require(
            all(row["latency"] == means[row["num_tokens"]] for row in old_profile),
            "v1 anchor values differ from independently recomputed original campaign",
        )
        n4, tail_cases, contributing, controls = _tail_primary(natives["tail"], identity["tail"], natives["v1"])
        old_cases = [f"archive={old['native_archive']['sha256']}/v1/{case}" for case in old_cases]
        cases = old_cases + tail_cases
        _require(
            len(cases) == len(set(cases)) == 1152
            and len(old_cases) + len(contributing) == 960
            and provenance.case_plan_hash(cases) == identity["case_plan_hash"],
            "Composite case plan or role count mismatch",
        )
        means[4] = n4
        rows = [
            {**identity["row_fields"], "num_tokens": n, "distribution": PROFILE, "latency": means[n]} for n in TOKENS
        ]
        _require(means[4] > means[8], "Expected measured nonmonotonicity changed")
        result = {"rows": rows, "case_plan_hash": identity["case_plan_hash"], "controls": controls}
        if validate_only:
            _require(_approved_base_inventory(base, identity) == before, "Base systems changed during validation")
            return {"status": "VALIDATED_ONLY", **result}
        staged = work / "systems"
        shutil.copytree(base, staged)
        _require(_approved_base_inventory(staged, identity) == before, "Base systems changed during copy")
        target = staged / relative
        evidence = target / "evidence" / PROFILE
        evidence.mkdir(parents=True, exist_ok=False)
        originals = evidence / "original-table"
        originals.mkdir()
        for name in ("moe_perf.parquet", "collection_meta.yaml"):
            shutil.copyfile(table / name, originals / name)
            _require(_record(originals / name) == identity["base"][name], "Original table evidence changed")
        for role, review, spec in (("v1", v1_review, old), ("tail", tail_review, identity["tail"])):
            dest = evidence / role
            dest.mkdir()
            shutil.move(natives[role], dest / "native")
            shutil.move(work / f"{role}.tar.gz", dest / "native.tar.gz")
            shutil.copyfile(review, dest / "actual-data-review.json")
            _require(_record(dest / "actual-data-review.json") == spec["actual_data_review"], "Review changed")
        closure = {MODULE.replace(".", "/") + ".py", *provenance.SHARED_CORE, *closures[MODULE]}
        publisher_sources = {}
        for name in sorted(closure):
            dest = evidence / "publisher-source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(package / name, dest)
            publisher_sources[name] = _record(dest)
        _require(
            _read(evidence / "publisher-source/collector/sglang_rubin/observed_moe_v2_identity.json") == identity,
            "Composite publisher identity changed",
        )
        expected_paths = set(before) | {str(p.relative_to(staged)) for p in evidence.rglob("*") if p.is_file()}
        expected_paths.update(
            str((evidence / name).relative_to(staged)) for name in ("publication.json", "file-manifest.json")
        )
        merge_lock = target / "moe_perf.parquet.mergelock"
        _require(not merge_lock.exists() and not merge_lock.is_symlink(), "Preexisting merge lock")
        labels = {k: rows[0][k] for k in ("framework", "version", "op_name", "kernel_source")}
        items = [{k: v for k, v in row.items() if k not in {*labels, "device"}} for row in rows]
        csv = target / "moe_perf.txt"
        helper.log_perf(items, **labels, device_name=rows[0]["device"], perf_filename=str(csv))
        finalized = helper.finalize_perf_files([csv], merge_existing=True)
        _require(finalized == [target / "moe_perf.parquet"], "Missing finalized canonical table")
        # The finalizer released the lock; only this private, unexposed tree is
        # owned by this publisher. Shared collector locks are never removed.
        _require(_record(merge_lock)["size_bytes"] == 0, "Unexpected private merge lock")
        merge_lock.unlink()
        merged = pq.read_table(finalized[0]).cast(old_table.schema, safe=True)
        merged_rows = merged.to_pylist()
        _require(len(merged_rows) == 88, "Wrong published row count")
        _require([row for row in merged_rows if row["distribution"] != PROFILE] == old_rows, "Old rows changed")
        _require([row for row in merged_rows if row["distribution"] == PROFILE] == rows, "New rows changed")
        pq.write_table(merged, finalized[0])
        _require(pq.read_table(finalized[0]).equals(merged, check_metadata=True), "Final Arrow readback mismatch")
        event = {
            "collector_ref": MODULE,
            "collector_hash": module_hash,
            "case_plan_hash": identity["case_plan_hash"],
            "collected_at": identity["collected_at"],
            "rows": 4,
            "status": "complete",
            "source_campaign_rows": 1152,
            "source_campaign_status": "complete",
            "runtime": identity["runtime"],
        }
        entry = provenance.append_collection_event(
            old_meta["tables"]["moe_perf"], event, table="moe_perf", merged_rows=88
        )
        provenance.write_collection_meta(target, identity["runtime"], {**old_meta["tables"], "moe_perf": entry})
        metadata = yaml.safe_load((target / "collection_meta.yaml").read_text())
        provenance.validate_collection_meta_for_update(metadata)
        _require(
            metadata["tables"]["moe_perf"]["collections"][:-1] == old_meta["tables"]["moe_perf"]["collections"],
            "Original collection history changed",
        )
        publication = {
            "publisher_role": (
                "Offline four-row derivation; separate native GPU producers retained "
                "in v1/native/source and tail/native/source"
            ),
            "identity": identity,
            "publisher_collector_hash": module_hash,
            "publisher_sources": publisher_sources,
            "event": event,
            "case_ids": cases,
            "row_contributing_case_ids": old_cases + contributing,
            "validation_only_case_ids": [case for case in tail_cases if case not in contributing],
            "rows": rows,
            "controls": controls,
            "original_systems_inventory": before,
            "published_at": datetime.now(UTC).isoformat(),
        }
        (evidence / "publication.json").write_text(json.dumps(publication, indent=2, allow_nan=False) + "\n")
        (evidence / "file-manifest.json").write_text(json.dumps(_inventory(evidence), indent=2) + "\n")
        _require(_approved_base_inventory(base, identity) == before, "Base systems mutated")
        after = _inventory(staged)
        _require(set(after) == expected_paths, "Unexpected file in private publication staging")
        changed = {str(relative / name) for name in ("moe_perf.parquet", "collection_meta.yaml")}
        _require(
            all(after[name] == record for name, record in before.items() if name not in changed), "Old evidence changed"
        )
        _require(provenance.collector_hash(MODULE, package, closures) == module_hash, "Publisher source changed")
        helper._rename_noreplace(staged, output)
    return {"status": "PUBLISHED", **result, "event": event, "output_systems": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "base-systems",
        "output-systems",
        "v1-archive",
        "v1-review",
        "tail-archive",
        "tail-review",
        "publisher-source",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    print(json.dumps(publish(**vars(parser.parse_args())), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
