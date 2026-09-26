# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline publication of one independently qualified native MoE campaign.

This module did not generate GPU timings. Its authored closure binds the exact
native producer, capture and data-review identities in observed_moe_identity.json.
The complete native archive/source and original table/sidecar are retained beside
the published data. No framework imports, GPU execution or fitted values occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

MODULE = "collector.sglang_rubin.publish_observed_moe"
PROFILE = "observed_glm52_nvfp4_decode_1ab2c747975e_v1"
IDENTITY_PATH = Path(__file__).with_name("observed_moe_identity.json")
ISLS = (1024, 8192, 32768)
TOKENS = (1, 8, 32)
ORDINALS = (8, 64, 128, 192, 256, 320, 384, 448)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=nonfinite)


def _record(path):
    _require(path.is_file() and not path.is_symlink(), f"Expected regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _inventory(root):
    result = {}
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"Symlink in evidence tree: {path}")
        if not path.is_dir():
            result[str(path.relative_to(root))] = _record(path)
    return result


def _case_id(rank, block, isl, batch, ordinal):
    return f"observed_moe/rank={rank}/block={block}/isl={isl}/physical_tokens={batch}/ordinal={ordinal}/layers=3..77"


def _verify_native(archive, directory, identity):
    _require(_record(archive) == identity["native_archive"], "Unapproved native archive")
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        _require(len(members) == identity["native_archive_members"], "Incomplete native archive")
        _require(len(members) == len({m.name for m in members}), "Duplicate archive members")
        _require(sum(m.size for m in members) == identity["native_expanded_bytes"], "Native archive size mismatch")
        for member in members:
            name = PurePosixPath(member.name)
            _require(
                not name.is_absolute()
                and ".." not in name.parts
                and str(name) == member.name
                and bool(name.parts)
                and name.parts[0] == identity["native_archive_root"]
                and (member.isdir() or member.isfile()),
                "Unsafe native archive member",
            )
        handle.extractall(directory, filter="data")
    root = directory / identity["native_archive_root"]
    manifest_path = root / "file-manifest.json"
    _require(_record(manifest_path) == identity["native_file_manifest"], "Native file manifest mismatch")
    manifest = _read(manifest_path)
    actual = _inventory(root)
    _require(len(actual) == identity["native_archive_files"], "Incomplete native evidence")
    _require(actual == {**manifest, "file-manifest.json": _record(manifest_path)}, "Native evidence hash mismatch")
    _require(
        _record(root / "staging-manifest.json") == identity["producer_source_manifest"], "Producer source mismatch"
    )
    source = _read(root / "staging-manifest.json")
    _require(source["files"] == identity["producer_source_files"], "Producer source closure mismatch")
    for name, record in source["files"].items():
        _require(_record(root / "source" / name) == record, "Executed producer file mismatch")
    return root


def _primary(root, identity):
    from collector.provenance import case_plan_hash

    campaign = _read(root / "campaign.json")
    _require(campaign["status"] == "complete" and "error" not in campaign, "Incomplete producer campaign")
    _require(campaign["summary"]["layers"] == list(range(3, 78)), "Incomplete 75-layer sweep")
    _require(campaign["summary"]["count"] == 864, "Incomplete primary campaign")
    capture = campaign["capture_input"]
    _require(capture["archive_identity"]["sha256"] == identity["capture"]["archive_sha256"], "Capture archive mismatch")
    _require(capture["manifest_sha256"] == identity["capture"]["manifest_sha256"], "Capture manifest mismatch")
    _require(capture["review_identity"]["sha256"] == identity["capture"]["review_sha256"], "Capture review mismatch")
    expected_runtime = _read(root / "source/expected-runtime.json")
    values = {tokens: [] for tokens in TOKENS}
    case_ids = []
    workloads = [(isl, tokens) for isl in ISLS for tokens in TOKENS]
    order = [
        (block, isl, tokens, ordinal)
        for block in range(3)
        for isl, tokens in workloads[block * 3 :] + workloads[: block * 3]
        for ordinal in ORDINALS[block * 2 :] + ORDINALS[: block * 2]
    ]
    for rank in range(4):
        folder = root / f"rank-{rank}"
        result = _read(folder / "result.json")
        _require(result["status"] == "complete" and result["rank"] == rank, "Incomplete rank")
        _require(result["native_boundary"] == identity["producer_boundary"], "Native timing boundary mismatch")
        _require(result["weights_before_after_equal"] is True, "Unqualified weight inventory")
        _require(
            _read(folder / "weights-before.json") == _read(folder / "weights-after.json"), "Changed native weights"
        )
        preflight = _read(folder / "preflight.json")
        _require(
            preflight["machine"] == "aarch64"
            and preflight["versions"] == expected_runtime["package_versions"]
            and preflight["versions"]["sglang"] == identity["runtime"]["version"]
            and preflight["torch_cuda_version"] == expected_runtime["torch_cuda_version"],
            "Native runtime mismatch",
        )
        coverage = result["coverage"]
        _require(
            all(
                coverage[key] == value
                for key, value in {
                    "primary_blocks": 216,
                    "layer_diagnostic_blocks": 72,
                    "dual_stream_blocks": 18,
                    "subset_blocks": 48,
                    "cache_before_after_equal": True,
                    "inputs_before_after_equal": True,
                }.items()
            ),
            "Incomplete measurements or input/cache guards",
        )
        diagnostics = _read(folder / "observer-diagnostics.json")
        _require(
            len(diagnostics) == 72
            and sum("dual_stream" in row for row in diagnostics) == 18
            and len(_read(folder / "subset-measurements.json")) == 48,
            "Incomplete retained diagnostic records",
        )
        rows = _read(folder / "primary-measurements.json")
        _require(
            [tuple(row[key] for key in ("block", "isl", "batch", "ordinal")) for row in rows] == order,
            "Missing, duplicate or reordered primary case",
        )
        for row in rows:
            raw = row["raw"]
            value = raw["latency_ms"]
            _require(row["rank"] == rank, "Primary rank mismatch")
            _require(type(value) in (int, float) and math.isfinite(value) and value > 0, "Invalid measured latency")
            _require(
                raw["used_cuda_graph"] is True
                and raw["num_runs_executed"] == 10
                and raw["throttled"] is False
                and raw["power_stats"] is None,
                "Unqualified primary timing method",
            )
            _require(
                row["captured_input"]["capture_source_manifest_sha256"]
                == identity["capture"]["source_manifest_sha256"],
                "Primary capture source mismatch",
            )
            values[row["batch"]].append(value / 75)
            case_ids.append(_case_id(rank, *(row[key] for key in ("block", "isl", "batch", "ordinal"))))
    _require(len(case_ids) == len(set(case_ids)) == 864, "Incomplete native case plan")
    _require(case_plan_hash(case_ids) == identity["case_plan_hash"], "Case-plan identity mismatch")
    means = {tokens: statistics.fmean(values[tokens]) for tokens in TOKENS}
    for tokens, mean in means.items():
        _require(len(values[tokens]) == 288, "Unequal fixed-mixture coverage")
        _require(
            math.isclose(
                mean, campaign["summary"]["summaries"][str(tokens)]["mean_rank_local_ms_per_layer"], rel_tol=1e-12
            ),
            "Native aggregate mismatch",
        )
    return means, case_ids


def publish(*, base_systems, output_systems, archive, review, validate_only=False, publisher_source=None):
    """Replay the retained qualified publisher; current source is not its identity."""
    from collector.sglang_rubin.replay_observed_moe import replay

    return replay(
        version="v1",
        publisher_source=publisher_source,
        base_systems=base_systems,
        output_systems=output_systems,
        archive=archive,
        review=review,
        validate_only=validate_only,
    )


def _publish_current(*, base_systems, output_systems, archive, review, validate_only=False):
    """Validate exact approved input; publish only to an absent fresh systems root."""
    import pyarrow.parquet as pq
    import yaml

    from collector import helper, provenance

    identity = _read(IDENTITY_PATH)
    base_systems, output_systems, archive, review = map(Path, (base_systems, output_systems, archive, review))
    _require(identity["module"] == MODULE, "Publication identity module mismatch")
    _require(
        identity["distribution"] == PROFILE and identity["physical_tokens"] == list(TOKENS),
        "Publication profile mismatch",
    )
    _require(not base_systems.is_symlink() and base_systems.is_dir(), "Unsafe source systems root")
    _require(not output_systems.exists() and not output_systems.is_symlink(), "Output systems root already exists")
    _require(
        not output_systems.resolve().is_relative_to(base_systems.resolve()), "Output cannot be inside source systems"
    )
    _require(_record(review) == identity["actual_data_review"], "Unapproved actual-data review")
    verdict = _read(review)
    _require(
        verdict["verdict"] == "CLEAN"
        and verdict["status"] == identity["actual_data_review_status"]
        and verdict["primary_fixed_mixture_timing_qualified"] is True
        and verdict["complete_primary_data_validated"] is True
        and verdict["native_archive"] == identity["native_archive"]
        and verdict["producer_source_manifest"] == identity["producer_source_manifest"],
        "Review does not qualify this exact native primary campaign",
    )
    relative = Path(identity["table_relative_path"])
    _require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe table path")
    table = base_systems / relative
    before = _inventory(base_systems)
    for name in ("moe_perf.parquet", "collection_meta.yaml"):
        _require(_record(table / name) == identity["base"][name], "Unapproved original table or metadata")
    old_meta = yaml.safe_load((table / "collection_meta.yaml").read_text())
    provenance.validate_collection_meta_for_update(old_meta, tables_to_update={"moe_perf"})
    _require(all(old_meta["runtime"].get(k) == v for k, v in identity["runtime"].items()), "Base runtime mismatch")
    old_table = pq.read_table(table / "moe_perf.parquet")
    old_rows = old_table.to_pylist()
    _require(len(old_rows) == identity["base"]["rows"] == 81, "Original 81 rows required")
    _require(old_meta["tables"]["moe_perf"]["rows"] == 81, "Original event count mismatch")
    _require(
        all(
            all(row[key] == value for key, value in identity["row_fields"].items())
            and type(row["latency"]) in (int, float)
            and math.isfinite(row["latency"])
            and row["latency"] > 0
            for row in old_rows
        ),
        "Original row runtime, shape or latency mismatch",
    )
    _require(all(row["distribution"] != identity["distribution"] for row in old_rows), "Profile already exists")
    keys = [{k: v for k, v in row.items() if k != "latency"} for row in old_rows]
    _require(len({json.dumps(row, sort_keys=True) for row in keys}) == len(keys), "Duplicate original row keys")
    with tempfile.TemporaryDirectory(prefix="aisim-observed-moe-", dir=output_systems.parent) as work:
        work = Path(work)
        retained_archive = work / "native.tar.gz"
        shutil.copyfile(archive, retained_archive)
        native = _verify_native(retained_archive, work / "raw", identity)
        means, case_ids = _primary(native, identity)
        new_rows = []
        for tokens in TOKENS:
            row = {**identity["row_fields"], "num_tokens": tokens, "distribution": identity["distribution"]}
            new_rows.append({**row, "latency": means[tokens]})
        if validate_only:
            return {"status": "VALIDATED_ONLY", "rows": new_rows, "case_plan_hash": identity["case_plan_hash"]}
        staged = work / "systems"
        shutil.copytree(base_systems, staged)
        _require(_inventory(staged) == before, "Source systems changed during copy")
        target = staged / relative
        evidence = target / "evidence" / identity["distribution"]
        evidence.mkdir(parents=True, exist_ok=False)
        originals = evidence / "original-table"
        originals.mkdir()
        for name in ("moe_perf.parquet", "collection_meta.yaml"):
            shutil.copyfile(table / name, originals / name)
            _require(_record(originals / name) == identity["base"][name], "Original evidence changed")
        shutil.move(native, evidence / "native")
        shutil.move(retained_archive, evidence / "native.tar.gz")
        shutil.copyfile(review, evidence / "actual-data-review.json")
        _require(_record(evidence / "actual-data-review.json") == identity["actual_data_review"], "Review changed")
        package_root = Path(__file__).resolve().parents[2]
        closures = provenance.load_closures(package_root / "collector/hash_closures.yaml")
        module_hash = provenance.collector_hash(MODULE, package_root, closures)
        closure = {MODULE.replace(".", "/") + ".py", *provenance.SHARED_CORE, *closures[MODULE]}
        publisher_sources = {}
        for name in sorted(closure):
            dest = evidence / "publisher-source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(package_root / name, dest)
            publisher_sources[name] = _record(dest)
        _require(
            _read(evidence / "publisher-source/collector/sglang_rubin/observed_moe_identity.json") == identity,
            "Publisher identity changed",
        )
        expected_paths = set(before) | {str(path.relative_to(staged)) for path in evidence.rglob("*") if path.is_file()}
        expected_paths.update(
            str((evidence / name).relative_to(staged)) for name in ("publication.json", "file-manifest.json")
        )
        merge_lock = target / "moe_perf.parquet.mergelock"
        _require(not merge_lock.exists() and not merge_lock.is_symlink(), "Preexisting merge lock in source systems")
        items = [
            {
                key: row[key]
                for key in (
                    "moe_dtype",
                    "num_tokens",
                    "hidden_size",
                    "inter_size",
                    "topk",
                    "num_experts",
                    "moe_tp_size",
                    "moe_ep_size",
                    "distribution",
                    "latency",
                )
            }
            for row in new_rows
        ]
        csv = target / "moe_perf.txt"
        helper.log_perf(
            items,
            **{k: new_rows[0][k] for k in ("framework", "version", "op_name", "kernel_source")},
            device_name=new_rows[0]["device"],
            perf_filename=str(csv),
        )
        finalized = helper.finalize_perf_files([csv], merge_existing=True)
        _require(finalized == [target / "moe_perf.parquet"], "Missing finalized canonical table")
        # Finalization has released its locks. This unexposed temporary tree is
        # exclusively owned by the publisher; shared collector locks stay intact.
        _require(_record(merge_lock)["size_bytes"] == 0, "Unexpected private finalization lock")
        merge_lock.unlink()
        merged = pq.read_table(finalized[0])
        # The finalizer's pandas round trip may widen Arrow string offsets.
        # Retain the original physical schema, then verify every row unchanged.
        merged = merged.cast(old_table.schema, safe=True)
        merged_rows = merged.to_pylist()
        _require(len(merged_rows) == 84, "Duplicate or missing published rows")
        _require(
            [r for r in merged_rows if r["distribution"] != identity["distribution"]] == old_rows,
            "Original rows changed",
        )
        _require(
            [r for r in merged_rows if r["distribution"] == identity["distribution"]] == new_rows,
            "Published aggregate mismatch",
        )
        pq.write_table(merged, finalized[0])
        event = {
            "collector_ref": MODULE,
            "collector_hash": module_hash,
            "case_plan_hash": identity["case_plan_hash"],
            "collected_at": identity["collected_at"],
            "rows": 3,
            "status": "complete",
            "source_campaign_rows": 864,
            "source_campaign_status": "complete",
            "runtime": identity["runtime"],
        }
        entry = provenance.append_collection_event(
            old_meta["tables"]["moe_perf"], event, table="moe_perf", merged_rows=84
        )
        provenance.write_collection_meta(target, identity["runtime"], {**old_meta["tables"], "moe_perf": entry})
        provenance.validate_collection_meta_for_update(yaml.safe_load((target / "collection_meta.yaml").read_text()))
        publication = {
            "publisher_role": "offline derivation; GPU timing producer source is retained under native/source",
            "identity": identity,
            "publisher_collector_hash": module_hash,
            "publisher_sources": publisher_sources,
            "event": event,
            "case_ids": case_ids,
            "rows": new_rows,
            "original_systems_inventory": before,
            "published_at": datetime.now(UTC).isoformat(),
        }
        (evidence / "publication.json").write_text(json.dumps(publication, indent=2, allow_nan=False) + "\n")
        (evidence / "file-manifest.json").write_text(json.dumps(_inventory(evidence), indent=2) + "\n")
        _require(_inventory(base_systems) == before, "Source systems mutated")
        after = _inventory(staged)
        _require(set(after) == expected_paths, "Unexpected file in private publication staging")
        for name, record in before.items():
            if name not in {str(relative / "moe_perf.parquet"), str(relative / "collection_meta.yaml")}:
                _require(after[name] == record, "Unrelated systems file changed")
        _require(provenance.collector_hash(MODULE, package_root, closures) == module_hash, "Publisher source changed")
        helper._rename_noreplace(staged, output_systems)
    return {"status": "PUBLISHED", "rows": new_rows, "event": event, "output_systems": str(output_systems)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-systems", type=Path, required=True)
    parser.add_argument("--output-systems", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument(
        "--publisher-source", type=Path, required=True, help="Retained qualified publisher-source directory"
    )
    parser.add_argument("--validate-only", action="store_true")
    args = vars(parser.parse_args())
    print(json.dumps(publish(**args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
