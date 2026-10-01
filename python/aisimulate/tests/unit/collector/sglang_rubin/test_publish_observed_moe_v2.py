# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite offline publication; native fixtures are synthetic, never GPU data."""

import copy
import hashlib
import io
import json
import re
import shutil
import statistics
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from collector import helper, provenance
from collector.sglang_rubin import publish_observed_moe as v1
from collector.sglang_rubin import publish_observed_moe_v2 as publisher
from test_publish_observed_moe import Campaign as V1Campaign
from test_publish_observed_moe import record, write_json

pytestmark = pytest.mark.unit


def test_identity_declares_physical_anchors_and_complete_case_roles():
    identity = json.loads(publisher.IDENTITY_PATH.read_text())
    assert identity["distribution"] == "observed_glm52_nvfp4_decode_composite_v2"
    assert identity["physical_tokens"] == [1, 4, 8, 32]
    assert identity["case_roles"] == {"total": 1152, "row_contributing": 960, "validation_only": 192}
    assert identity["v1_collector_hash"] == "sha256:067ad7797474518eab028911d4f0d6f314e1dd0456b08be5bae45de267f3e332"


class Composite:
    """Two complete synthetic campaigns; no fixture is a native measurement."""

    def __init__(self, tmp_path, monkeypatch):
        actual_module = Path(publisher.__file__)
        self.identity = json.loads(publisher.IDENTITY_PATH.read_text())
        self.old = V1Campaign(tmp_path, monkeypatch)
        self.old.run()
        self.base = self.old.output
        self.output = tmp_path / "composite"
        self.package = self.old.package
        dest = self.package / "collector/sglang_rubin/publish_observed_moe_v2.py"
        shutil.copyfile(actual_module, dest)
        self.identity_path = dest.with_name("observed_moe_v2_identity.json")
        monkeypatch.setattr(publisher, "__file__", str(dest))
        monkeypatch.setattr(publisher, "IDENTITY_PATH", self.identity_path)
        self.table = self.base / self.identity["table_relative_path"]
        self.old_table = pq.read_table(self.table / "moe_perf.parquet")
        self.old_meta = yaml.safe_load((self.table / "collection_meta.yaml").read_text())
        self.native = tmp_path / "synthetic-tail"
        (self.native / "source").mkdir(parents=True)
        for name in ("expected-runtime.json", "producer.py"):
            shutil.copyfile(self.old.native / "source" / name, self.native / "source" / name)
        self.tail = self.identity["tail"]
        self.tail["native_archive_root"] = self.native.name
        sources = {p.name: record(p) for p in (self.native / "source").iterdir()}
        write_json(self.native / "staging-manifest.json", {"files": sources})
        self.tail["producer_source_manifest"] = record(self.native / "staging-manifest.json")
        self.tail["producer_source_files"] = sources
        self.plan = []
        logicals = (3, 29, 31)
        for block in range(3):
            for logical in logicals[block:] + logicals[:block]:
                for ordinal in v1.ORDINALS[2 * block :] + v1.ORDINALS[: 2 * block]:
                    physical = 4 if logical == 3 else 32
                    self.plan.append(
                        {
                            "block": block,
                            "isl": 32768,
                            "cohort": f"isl32768-c{logical}-l{logical}-p{physical}",
                            "logical": logical,
                            "physical": physical,
                            "ordinal": ordinal,
                        }
                    )
        capture = self.tail["capture"]
        cohort_values = {n: [] for n in logicals}
        for rank in range(4):
            folder = self.native / f"rank-{rank}"
            folder.mkdir()
            for name in ("weights-before.json", "weights-after.json", "preflight.json"):
                shutil.copyfile(self.old.native / f"rank-{rank}" / name, folder / name)
            write_json(folder / "saved-native-cache.json", {"synthetic_rank": rank})
            self.tail["rank_caches"][str(rank)] = record(folder / "saved-native-cache.json")
            rows = []
            for index, case in enumerate(self.plan):
                logical = case["logical"]
                latency = {3: 0.011, 29: 0.033, 31: 0.032}[logical] * 75
                # Deliberate small rank/block variation exercises every aggregation axis.
                latency *= 1 + rank / 10000 + case["block"] / 100000
                row = {**case, "rank": rank}
                for phase_index, phase in enumerate(("before", "observed", "after")):
                    row[phase] = {
                        "raw": {
                            "latency_ms": latency,
                            "used_cuda_graph": True,
                            "num_runs_executed": 10,
                            "throttled": False,
                            "power_stats": None,
                        },
                        "host_start_ns": index * 100 + phase_index * 10 + 1,
                        "host_end_ns": index * 100 + phase_index * 10 + 2,
                    }
                row["observed"].update(external_event_count=2, event_interval_substituted_for_primary=False)
                row.update(
                    plain_mean_ms=latency,
                    normalized_ms_per_layer=latency / 75,
                    control={"qualified": True, "limit_percent": 5, "overhead_subtracted": False},
                    native_rank_cache_sha256=self.tail["rank_caches"][str(rank)]["sha256"],
                    captured_input={
                        "capture_source_manifest_sha256": capture["source_manifest_sha256"],
                        "capture_archive_sha256": capture["archive_sha256"],
                        "ordinal": case["ordinal"],
                        "eligible_ordinal": case["ordinal"],
                        "actual_logical_tokens": logical,
                        "actual_physical_graph_tokens": case["physical"],
                        "valid_row_range": [0, logical],
                        "padding_range": [logical, case["physical"]],
                        "padding_tokens": case["physical"] - logical,
                    },
                )
                rows.append(row)
                cohort_values[logical].append(latency / 75)
            write_json(folder / "primary-measurements.json", rows)
            write_json(
                folder / "result.json",
                {
                    "status": "complete",
                    "rank": rank,
                    "native_boundary": self.tail["producer_boundary"],
                    "weights_before_after_equal": True,
                    "coverage": {
                        "primary_cases": 72,
                        "helper_results": 216,
                        "case_observer_checks": 144,
                        "cache_before_after_equal": True,
                        "inputs_before_after_equal": True,
                        "one_cohort_resident": True,
                        "legacy_subset_or_per_layer_diagnostics_executed": False,
                    },
                },
            )
        self.expected_means = dict(self.old.expected_means)
        self.expected_means[4] = statistics.fmean(cohort_values[3])
        write_json(
            self.native / "campaign.json",
            {
                "status": "complete",
                "observer_controls_qualified": True,
                "structural_complete": True,
                "capture_observer_qualified": True,
                "operator_observer_drift_qualified": True,
                "cross_block_stable": True,
                "tp_mean_representative": True,
                "capture_input": {
                    "archive_identity": {
                        "sha256": capture["archive_sha256"],
                        "size_bytes": capture["archive_size_bytes"],
                    },
                    "manifest_sha256": capture["manifest_sha256"],
                    "review_identity": {"sha256": capture["review_sha256"]},
                },
                "plan": {
                    "primary_rank_local_cases": self.plan,
                    "capture_schema_version": 4,
                    "image": self.identity["runtime"]["image_digest"],
                },
                "summary": {
                    "cohorts": {
                        f"isl32768-c{n}-l{n}-p{4 if n == 3 else 32}": {"mean_ms_per_layer": statistics.fmean(values)}
                        for n, values in cohort_values.items()
                    }
                },
            },
        )
        self.archive = tmp_path / "synthetic-tail.tar.gz"
        self.review_path = tmp_path / "synthetic-tail-review.json"
        self.review = {
            "verdict": "CLEAN",
            "status": self.tail["actual_data_review_status"],
            "actual_data_qualified": True,
            "timing_qualified": True,
            "measurement_controls_qualified": True,
            "capture_semantics_valid": True,
            "identities": {"producer_source_manifest_sha256": self.tail["producer_source_manifest"]["sha256"]},
        }
        self.seal()

    def bind(self):
        self.identity["v1_identity_file"] = record(self.old.identity_path)
        closures = provenance.load_closures(self.package / "collector/hash_closures.yaml")
        self.identity["v1_collector_hash"] = provenance.collector_hash(v1.MODULE, self.package, closures)
        for name in ("moe_perf.parquet", "collection_meta.yaml"):
            self.identity["base"][name] = record(self.table / name)
        base_tree = {
            "files": publisher._inventory(self.base),
            "directories": sorted(str(p.relative_to(self.base)) for p in self.base.rglob("*") if p.is_dir()),
        }
        self.identity["base"]["systems_tree_sha256"] = hashlib.sha256(
            json.dumps(base_tree, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        old_cases = [
            f"archive={self.old.identity['native_archive']['sha256']}/v1/{v1._case_id(r, b, isl, n, o)}"
            for r in range(4)
            for b in range(3)
            for isl in v1.ISLS
            for n in v1.TOKENS
            for o in v1.ORDINALS
        ]
        tail_cases = [
            f"archive={self.tail['native_archive']['sha256']}/tail/rank={r}/block={b}/isl=32768/"
            f"logical_tokens={n}/physical_tokens={4 if n == 3 else 32}/ordinal={o}/layers=3..77"
            for r in range(4)
            for b in range(3)
            for n in (3, 29, 31)
            for o in v1.ORDINALS
        ]
        self.identity["case_plan_hash"] = provenance.case_plan_hash(old_cases + tail_cases)
        write_json(self.review_path, self.review)
        self.tail["actual_data_review"] = record(self.review_path)
        write_json(self.identity_path, self.identity)

    def seal(self, extra=None):
        manifest = {
            str(p.relative_to(self.native)): record(p)
            for p in self.native.rglob("*")
            if p.is_file() and p.name != "file-manifest.json"
        }
        write_json(self.native / "file-manifest.json", manifest)
        self.tail["native_file_manifest"] = record(self.native / "file-manifest.json")
        with tarfile.open(self.archive, "w:gz") as handle:
            handle.add(self.native, arcname=self.native.name)
            if extra:
                info, content = extra
                handle.addfile(info, io.BytesIO(content))
        with tarfile.open(self.archive) as handle:
            members = handle.getmembers()
        self.tail.update(
            native_archive=record(self.archive),
            native_archive_members=len(members),
            native_archive_files=len(manifest) + 1,
            native_expanded_bytes=sum(m.size for m in members),
        )
        self.review["identities"]["native_archive"] = self.tail["native_archive"]
        self.bind()

    def mutate(self, name, change):
        path = self.native / name
        data = json.loads(path.read_text())
        change(data)
        write_json(path, data)
        self.seal()

    def run(self, **kwargs):
        # Fixed-identity public replay is covered separately with real archives.
        return publisher._publish_current(
            base_systems=self.base,
            output_systems=self.output,
            v1_archive=self.old.archive,
            v1_review=self.old.review_path,
            tail_archive=self.archive,
            tail_review=self.review_path,
            **kwargs,
        )


@pytest.fixture
def composite(tmp_path, monkeypatch):
    return Composite(tmp_path, monkeypatch)


def rejected(composite, match, *, errors=ValueError, **kwargs):
    before = publisher._inventory(composite.base)
    with pytest.raises(errors, match=match):
        composite.run(**kwargs)
    assert not composite.output.exists()
    assert publisher._inventory(composite.base) == before


@pytest.mark.parametrize("validate_only", [True, False])
@pytest.mark.parametrize(
    "kind",
    [
        "extra_table",
        "missing_metadata",
        "corrupt_metadata",
        "missing_table_coverage",
        "wrong_runtime",
        "changed_table",
        "changed_unrelated",
        "missing_unrelated",
        "renamed_unrelated",
        "changed_source",
        "extra_reuse",
        "empty_version_directory",
        "unowned_lock",
    ],
)
def test_unapproved_effective_source_tree_rejects_before_work(composite, monkeypatch, kind, validate_only):
    metadata = composite.table / "collection_meta.yaml"
    unrelated = composite.base / "unrelated.yaml"
    if kind == "extra_table":
        shutil.copyfile(composite.table / "moe_perf.parquet", composite.table / "unrecorded_perf.parquet")
    elif kind == "missing_metadata":
        metadata.unlink()
    elif kind == "corrupt_metadata":
        metadata.write_text("tables: [unterminated")
    elif kind in ("missing_table_coverage", "wrong_runtime"):
        value = yaml.safe_load(metadata.read_text())
        if kind == "missing_table_coverage":
            value["tables"] = {}
        else:
            value["runtime"]["image_digest"] = "sha256:" + "0" * 64
        metadata.write_text(yaml.safe_dump(value))
    elif kind == "changed_table":
        with (composite.table / "moe_perf.parquet").open("ab") as stream:
            stream.write(b"changed")
    elif kind == "changed_unrelated":
        unrelated.write_text("changed: true\n")
    elif kind == "missing_unrelated":
        unrelated.unlink()
    elif kind == "renamed_unrelated":
        unrelated.rename(unrelated.with_name("renamed.yaml"))
    elif kind == "changed_source":
        next(composite.base.rglob("producer.py")).write_text("# substituted source\n")
    elif kind == "extra_reuse":
        (composite.table / "reuse.yaml").write_text("source: changed\n")
    elif kind == "empty_version_directory":
        (composite.table.parent / "0.0.0").mkdir()
    else:
        (composite.table / "moe_perf.parquet.mergelock").write_text("unowned")

    def no_archive_work(*args, **kwargs):
        pytest.fail("Invalid input reached archive extraction or publication staging")

    monkeypatch.setattr(v1, "_verify_native", no_archive_work)
    rejected(composite, "Unapproved base systems tree", validate_only=validate_only)


@pytest.mark.parametrize("validate_only", [True, False])
@pytest.mark.parametrize("kind", ["missing", "mismatch", "wrong_type"])
def test_fixed_base_identity_is_required(composite, kind, validate_only):
    if kind == "missing":
        del composite.identity["base"]["systems_tree_sha256"]
    else:
        composite.identity["base"]["systems_tree_sha256"] = "0" * 64 if kind == "mismatch" else None
    write_json(composite.identity_path, composite.identity)
    if kind == "missing":
        rejected(composite, "^'systems_tree_sha256'$", errors=KeyError, validate_only=validate_only)
    else:
        rejected(composite, "Unapproved base systems tree", validate_only=validate_only)


def test_complete_union_preserves_exact_rows_schema_history_and_source_evidence(composite):
    from aisimulate_core.sdk.perf_database import _validate_collection_meta_v2

    before = publisher._inventory(composite.base)
    result = composite.run()
    target = composite.output / composite.identity["table_relative_path"]
    actual = pq.read_table(target / "moe_perf.parquet")
    assert actual.schema.equals(composite.old_table.schema, check_metadata=True)
    assert actual.to_pylist()[:84] == composite.old_table.to_pylist()
    assert len(actual) == 88
    new = actual.to_pylist()[84:]
    assert [row["num_tokens"] for row in new] == [1, 4, 8, 32]
    assert [row["latency"] for row in new] == pytest.approx([composite.expected_means[n] for n in (1, 4, 8, 32)])
    assert new[1]["latency"] > new[2]["latency"]
    for row in (new[0], new[2], new[3]):
        assert row["latency"] == next(
            r["latency"]
            for r in composite.old_table.to_pylist()
            if r["distribution"] == v1.PROFILE and r["num_tokens"] == row["num_tokens"]
        )
    metadata = yaml.safe_load((target / "collection_meta.yaml").read_text())
    assert metadata["runtime"] == composite.old_meta["runtime"]
    assert metadata["tables"]["moe_perf"]["collections"] == composite.old_meta["tables"]["moe_perf"]["collections"] + [
        result["event"]
    ]
    provenance.validate_collection_meta_for_update(metadata)
    _validate_collection_meta_v2(metadata, str(target / "collection_meta.yaml"))
    event = result["event"]
    assert event["rows"] == 4 and event["source_campaign_rows"] == 1152
    assert event["collector_ref"] == publisher.MODULE
    assert event["case_plan_hash"] == composite.identity["case_plan_hash"]
    evidence = target / "evidence" / publisher.PROFILE
    publication = json.loads((evidence / "publication.json").read_text())
    assert len(set(publication["case_ids"])) == 1152
    assert len(set(publication["row_contributing_case_ids"])) == 960
    assert len(set(publication["validation_only_case_ids"])) == 192
    assert set(publication["row_contributing_case_ids"]).isdisjoint(publication["validation_only_case_ids"])
    assert set(publication["case_ids"]) == set(
        publication["row_contributing_case_ids"] + publication["validation_only_case_ids"]
    )
    for role, native, archive, review in (
        ("v1", composite.old.native, composite.old.archive, composite.old.review_path),
        ("tail", composite.native, composite.archive, composite.review_path),
    ):
        assert publisher._inventory(evidence / role / "native") == publisher._inventory(native)
        assert record(evidence / role / "native.tar.gz") == record(archive)
        assert record(evidence / role / "actual-data-review.json") == record(review)
    manifest = json.loads((evidence / "file-manifest.json").read_text())
    assert manifest == {k: v for k, v in publisher._inventory(evidence).items() if k != "file-manifest.json"}
    assert record(evidence / "publisher-source/collector/sglang_rubin/observed_moe_identity.json") == record(
        composite.old.identity_path
    )
    assert publisher._inventory(composite.base) == before
    assert {p.name for p in target.iterdir()} == {"moe_perf.parquet", "collection_meta.yaml", "evidence"}


def test_validate_only_and_both_strict_source_resolvers(composite):
    import aisimulate_core as core
    from aisimulate_core.sdk.perf_database import _check_strict_provenance_for_request

    assert composite.run(validate_only=True)["status"] == "VALIDATED_ONLY"
    assert not composite.output.exists()
    composite.run()
    target = composite.output / composite.identity["table_relative_path"]
    system = composite.output / "data/vr200_hecate"
    _check_strict_provenance_for_request([str(target)], "sglang", str(system), strict=True)
    report = json.loads(
        core.resolve_op_sources_report_json(
            str(composite.output),
            str(system),
            "sglang",
            composite.identity["runtime"]["version"],
            "moe_perf.parquet",
            enable_shared_layer=True,
            strict=True,
        )
    )
    assert report["warnings"] == [] and len(report["records"]) == 1


@pytest.mark.parametrize("role", ["v1", "tail"])
@pytest.mark.parametrize("kind", ["missing", "duplicate", "reorder"])
def test_complete_schedule_is_mandatory(composite, role, kind):
    def change(rows):
        if kind == "missing":
            rows.pop()
        elif kind == "duplicate":
            rows[-1] = copy.deepcopy(rows[0])
        else:
            rows.reverse()

    if role == "tail":
        composite.mutate("rank-3/primary-measurements.json", change)
    else:
        composite.old.mutate_json("rank-3/primary-measurements.json", change)
        # Rebind this synthetic attestation so this test reaches the native
        # schedule guard instead of stopping at the changed identity hash.
        meta = copy.deepcopy(composite.old_meta)
        closures = provenance.load_closures(composite.package / "collector/hash_closures.yaml")
        meta["tables"]["moe_perf"]["collections"][-1]["collector_hash"] = provenance.collector_hash(
            v1.MODULE, composite.package, closures
        )
        (composite.table / "collection_meta.yaml").write_text(yaml.safe_dump(meta))
        composite.bind()
    rejected(composite, "Missing, duplicate or reordered")


@pytest.mark.parametrize("cohort", [3, 29, 31])
@pytest.mark.parametrize(
    "kind,match",
    [
        ("physical", "Tail logical/physical/rank/cache identity mismatch"),
        ("logical", "Missing, duplicate or reordered tail case"),
        ("padding", "Tail native padding/input identity mismatch"),
        ("ordinal", "Tail native padding/input identity mismatch"),
        ("cache", "Tail logical/physical/rank/cache identity mismatch"),
        ("capture", "Tail native padding/input identity mismatch"),
        ("graph", "Unqualified tail timing method"),
        ("substitution", "Wrong observer control or substituted instrumented timing"),
        ("boolean", "Invalid positive finite number"),
        ("nonfinite", "Nonfinite JSON number: NaN"),
        ("chronology", "Tail helper order mismatch"),
        ("observer", "Tail controls exceed fixed 5% gate"),
        ("drift", "Tail controls exceed fixed 5% gate"),
    ],
)
def test_every_cohort_rejects_bad_identity_timing_and_controls(composite, cohort, kind, match):
    def change(rows):
        row = next(r for r in rows if r["logical"] == cohort)
        if kind in ("physical", "logical"):
            row[kind] += 1
        elif kind == "padding":
            row["captured_input"]["padding_range"] = [0, 0]
        elif kind == "ordinal":
            row["captured_input"]["eligible_ordinal"] += 1
        elif kind == "cache":
            row["native_rank_cache_sha256"] = "0" * 64
        elif kind == "capture":
            row["captured_input"]["capture_archive_sha256"] = "0" * 64
        elif kind == "graph":
            row["before"]["raw"]["used_cuda_graph"] = False
        elif kind == "substitution":
            row["observed"]["event_interval_substituted_for_primary"] = True
        elif kind in ("boolean", "nonfinite"):
            row["before"]["raw"]["latency_ms"] = True if kind == "boolean" else float("nan")
        elif kind == "chronology":
            row["after"]["host_start_ns"] = row["before"]["host_start_ns"]
        else:
            row["observed" if kind == "observer" else "after"]["raw"]["latency_ms"] *= 1.2
            row["plain_mean_ms"] = statistics.fmean(row[p]["raw"]["latency_ms"] for p in ("before", "after"))
            row["normalized_ms_per_layer"] = row["plain_mean_ms"] / 75

    composite.mutate("rank-0/primary-measurements.json", change)
    rejected(composite, match)


@pytest.mark.parametrize("kind", ["adjacent", "rank_spread"])
def test_stability_controls_are_recomputed_from_all_raw_a_brackets(composite, kind):
    for rank in range(4):

        def change(rows, rank=rank):
            for row in rows:
                if (kind == "adjacent" and row["block"] == 1) or (kind == "rank_spread" and rank == 3):
                    for phase in ("before", "observed", "after"):
                        row[phase]["raw"]["latency_ms"] *= 1.2
                    row["plain_mean_ms"] *= 1.2
                    row["normalized_ms_per_layer"] *= 1.2

        composite.mutate(f"rank-{rank}/primary-measurements.json", change)
    rejected(composite, "Tail controls exceed")


@pytest.mark.parametrize(
    "name,change,match",
    [
        ("rank-1/result.json", lambda x: x["coverage"].update(primary_cases=71), "Incomplete tail coverage"),
        (
            "rank-1/result.json",
            lambda x: x["coverage"].update(cache_before_after_equal=False),
            "Incomplete tail coverage",
        ),
        (
            "rank-1/weights-after.json",
            lambda x: x.update(fixture_weight_sha256="0" * 64),
            "Tail weights changed or differ from v1",
        ),
        ("rank-1/saved-native-cache.json", lambda x: x.update(unapproved_tactic=1), "Tail saved cache mismatch"),
        ("rank-1/preflight.json", lambda x: x["versions"].update(sglang="wrong"), "Tail runtime mismatch"),
        ("source/producer.py", None, "Executed producer file mismatch"),
        ("campaign.json", lambda x: x.update(tp_mean_representative=False), "Unqualified tail tp_mean_representative"),
        ("campaign.json", lambda x: x["plan"].update(capture_schema_version=3), "Wrong tail physical plan or image"),
        ("campaign.json", lambda x: x["capture_input"].update(manifest_sha256="0" * 64), "Tail capture mismatch"),
    ],
)
def test_runtime_weights_source_cache_and_qualification_fail_closed(composite, name, change, match):
    if change is None:
        (composite.native / name).write_text("# changed native producer\n")
        composite.seal()
    else:
        composite.mutate(name, change)
    rejected(composite, match)


@pytest.mark.parametrize("role", ["v1", "tail"])
@pytest.mark.parametrize("which", ["archive", "review", "unqualified"])
def test_exact_archive_and_review_admission(composite, role, which):
    obj = composite.old if role == "v1" else composite
    if which == "unqualified":
        obj.review["verdict"] = "FAILED_UNQUALIFIED"
        if role == "v1":
            obj.save_identity()
            meta = copy.deepcopy(composite.old_meta)
            closures = provenance.load_closures(composite.package / "collector/hash_closures.yaml")
            meta["tables"]["moe_perf"]["collections"][-1]["collector_hash"] = provenance.collector_hash(
                v1.MODULE, composite.package, closures
            )
            (composite.table / "collection_meta.yaml").write_text(yaml.safe_dump(meta))
        composite.bind()
    else:
        with (obj.archive if which == "archive" else obj.review_path).open("ab") as stream:
            stream.write(b"unapproved")
    rejected(composite, "Review does not qualify" if which == "unqualified" else "Unapproved")


@pytest.mark.parametrize("kind", ["absolute", "traversal", "symlink", "duplicate"])
def test_unsafe_native_archive_cannot_escape(composite, tmp_path, kind):
    name = {
        "absolute": str(tmp_path / "escaped"),
        "traversal": "../escaped",
        "symlink": composite.native.name + "/link",
        "duplicate": composite.native.name + "/campaign.json",
    }[kind]
    info = tarfile.TarInfo(name)
    if kind == "symlink":
        info.type, info.linkname = tarfile.SYMTYPE, str(tmp_path / "escaped")
    composite.seal((info, b""))
    rejected(composite, "Unsafe native archive|Duplicate archive")
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize(
    "kind,match",
    [
        ("v1_identity", "v1 publisher source closure changed"),
        ("v1_source", "v1 publisher source closure changed"),
        ("case_plan", "Composite case plan or role count mismatch"),
        ("wrong_physical_label", "Wrong composite publication identity"),
        ("base_hash", "Unapproved base systems tree"),
        ("duplicate", "Duplicate base row keys"),
        ("nonfinite", "Invalid positive finite number"),
        ("history", "Incomplete or mismatched original collection history"),
        ("runtime", "Wrong base count/schema/runtime"),
    ],
)
def test_preserved_source_and_base_identity_are_mandatory(composite, kind, match):
    if kind == "v1_identity":
        composite.old.identity_path.write_text("{}")
    elif kind == "v1_source":
        with Path(v1.__file__).open("a") as stream:
            stream.write("\n# unexpected change\n")
    elif kind in ("case_plan", "wrong_physical_label"):
        if kind == "case_plan":
            composite.identity["case_plan_hash"] = "sha256:" + "0" * 64
        else:
            composite.identity["physical_tokens"] = [1, 3, 8, 32]
        write_json(composite.identity_path, composite.identity)
    elif kind in ("base_hash", "duplicate", "nonfinite"):
        rows = composite.old_table.to_pylist()
        if kind == "duplicate":
            rows[1] = rows[0]
        else:
            rows[0]["latency"] = float("nan") if kind == "nonfinite" else 99.0
        pq.write_table(
            pa.Table.from_pylist(rows, schema=composite.old_table.schema), composite.table / "moe_perf.parquet"
        )
        if kind != "base_hash":
            composite.bind()
    else:
        meta = copy.deepcopy(composite.old_meta)
        if kind == "history":
            meta["tables"]["moe_perf"]["collections"].pop()
        else:
            meta["runtime"]["version"] = "wrong"
        (composite.table / "collection_meta.yaml").write_text(yaml.safe_dump(meta))
        composite.bind()
    rejected(composite, match)


@pytest.mark.parametrize(
    "kind,match",
    [
        ("unknown", "Unexpected file in private publication staging"),
        ("nonempty_lock", "Unexpected private merge lock"),
        ("directory_lock", r"Expected regular file: .*moe_perf\.parquet\.mergelock"),
        ("symlink_lock", r"Expected regular file: .*moe_perf\.parquet\.mergelock"),
        ("missing_lock", r"Expected regular file: .*moe_perf\.parquet\.mergelock"),
        ("finalizer_failure", "synthetic finalizer failure"),
        ("source_race", "Publisher source changed"),
    ],
)
def test_failed_private_staging_never_promotes(composite, monkeypatch, kind, match):
    finalize = helper.finalize_perf_files

    def failure(*args, **kwargs):
        assert not composite.output.exists()
        if kind == "finalizer_failure":
            raise ValueError("synthetic finalizer failure")
        result = finalize(*args, **kwargs)
        target = result[0].parent
        lock = target / "moe_perf.parquet.mergelock"
        if kind == "unknown":
            (target / "untracked.tmp").write_text("unexpected")
        elif kind == "source_race":
            with Path(publisher.__file__).open("a") as stream:
                stream.write("\n# changed during publication\n")
        elif kind == "nonempty_lock":
            lock.write_text("unexpected")
        else:
            lock.unlink()
            if kind == "directory_lock":
                lock.mkdir()
            elif kind == "symlink_lock":
                lock.symlink_to(composite.base / "unrelated.yaml")
        return result

    monkeypatch.setattr(helper, "finalize_perf_files", failure)
    rejected(composite, match)


def test_private_lock_is_released_before_removal(composite, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    finalize = helper.finalize_perf_files

    def check(*args, **kwargs):
        paths = finalize(*args, **kwargs)
        with paths[0].with_name("moe_perf.parquet.mergelock").open("rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(stream, fcntl.LOCK_UN)
        assert not composite.output.exists()
        return paths

    monkeypatch.setattr(helper, "finalize_perf_files", check)
    composite.run()
    assert not list(composite.output.rglob("*.mergelock"))


def test_preexisting_output_and_unowned_lock_are_preserved(composite):
    composite.output.mkdir()
    (composite.output / "keep").write_text("unchanged")
    with pytest.raises(ValueError, match="already exists"):
        composite.run()
    assert (composite.output / "keep").read_text() == "unchanged"
    shutil.rmtree(composite.output)
    lock = composite.table / "moe_perf.parquet.mergelock"
    lock.write_text("unowned")
    composite.bind()
    rejected(composite, "Preexisting merge lock")
    assert lock.read_text() == "unowned"


@pytest.mark.parametrize(
    "value,match",
    [
        (0, "Invalid positive finite number"),
        (-1, "Invalid positive finite number"),
        (float("inf"), "Nonfinite JSON number: Infinity"),
        (float("-inf"), "Nonfinite JSON number: -Infinity"),
        ("0.1", "Invalid positive finite number"),
    ],
)
def test_nonpositive_nonfinite_and_nonnumeric_times_reject(composite, value, match):
    composite.mutate("rank-0/primary-measurements.json", lambda rows: rows[0]["before"]["raw"].update(latency_ms=value))
    rejected(composite, match)


@pytest.mark.parametrize(
    "name", ["rank-0/result.json", "rank-3/primary-measurements.json", "rank-2/weights-before.json"]
)
def test_missing_native_file_is_not_a_partial_publication(composite, name):
    (composite.native / name).unlink()
    composite.seal()
    rejected(composite, re.escape(name), errors=FileNotFoundError)


def test_duplicate_json_key_rejects(composite):
    (composite.native / "campaign.json").write_text('{"status":"complete","status":"complete"}')
    composite.seal()
    rejected(composite, "Duplicate JSON key")


def test_atomic_promotion_does_not_overwrite_a_racing_destination(composite, monkeypatch):
    finalize = helper.finalize_perf_files

    def create_destination(*args, **kwargs):
        paths = finalize(*args, **kwargs)
        composite.output.mkdir()
        (composite.output / "keep").write_text("concurrent owner")
        return paths

    monkeypatch.setattr(helper, "finalize_perf_files", create_destination)
    before = publisher._inventory(composite.base)
    with pytest.raises(FileExistsError):
        composite.run()
    assert (composite.output / "keep").read_text() == "concurrent owner"
    assert publisher._inventory(composite.base) == before
