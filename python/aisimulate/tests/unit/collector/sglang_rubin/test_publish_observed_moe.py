# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline publisher contract tests; all native measurements here are synthetic."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from collector import helper, provenance
from collector.sglang_rubin import publish_observed_moe as publisher

pytestmark = pytest.mark.unit


def record(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


class Campaign:
    """An isolated source closure and complete small synthetic native archive."""

    def __init__(self, tmp_path, monkeypatch):
        actual_module = Path(publisher.__file__)
        package = actual_module.parents[2]
        self.identity = json.loads(publisher.IDENTITY_PATH.read_text())
        self.package = tmp_path / "python"
        closures = provenance.load_closures(package / "collector/hash_closures.yaml")
        files = {
            "collector/hash_closures.yaml",
            publisher.MODULE.replace(".", "/") + ".py",
            *provenance.SHARED_CORE,
            *closures[publisher.MODULE],
        }
        for name in files:
            dest = self.package / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(package / name, dest)
        self.identity_path = self.package / "collector/sglang_rubin/observed_moe_identity.json"
        monkeypatch.setattr(publisher, "__file__", str(self.package / actual_module.relative_to(package)))
        monkeypatch.setattr(publisher, "IDENTITY_PATH", self.identity_path)
        self.base = tmp_path / "base"
        self.output = tmp_path / "output"
        self.table = self.base / self.identity["table_relative_path"]
        self.table.mkdir(parents=True)
        self.old_rows = [
            {**self.identity["row_fields"], "num_tokens": n, "distribution": dist, "latency": n / 1000}
            for dist in ("uniform", "power_law_1.01", "power_law_1.5")
            for n in range(1, 28)
        ]
        # Use the canonical column order emitted by log_perf, with exact old values.
        columns = [
            "framework",
            "version",
            "device",
            "op_name",
            "kernel_source",
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
        ]
        self.old_rows = [{key: row[key] for key in columns} for row in self.old_rows]
        pq.write_table(pa.Table.from_pylist(self.old_rows), self.table / "moe_perf.parquet")
        self.old_event = {
            "collector_ref": "unknown",
            "collector_hash": "sha256:" + "a" * 64,
            "case_plan_hash": "sha256:" + "b" * 64,
            "collected_at": "2026-09-18",
            "rows": 81,
            "status": "complete",
        }
        self.old_meta = {
            "schema_version": 1,
            "runtime": {**self.identity["runtime"], "source_commit": "c" * 40},
            "tables": {"moe_perf": self.old_event},
        }
        (self.table / "collection_meta.yaml").write_text(yaml.safe_dump(self.old_meta))
        (self.base / "unrelated.yaml").write_text("keep: exact\n")
        self.bind_base()
        self.native = tmp_path / self.identity["native_archive_root"]
        self.native.mkdir()
        runtime = {
            "package_versions": {"sglang": self.identity["runtime"]["version"], "torch": "fixture"},
            "torch_cuda_version": "fixture",
        }
        write_json(self.native / "source/expected-runtime.json", runtime)
        (self.native / "source/producer.py").write_text("# Synthetic producer fixture, never executed.\n")
        source_files = {
            str(p.relative_to(self.native / "source")): record(p) for p in (self.native / "source").iterdir()
        }
        write_json(self.native / "staging-manifest.json", {"files": source_files})
        self.identity["producer_source_files"] = source_files
        self.identity["producer_source_manifest"] = record(self.native / "staging-manifest.json")
        self.expected_means = {n: n / 1000 + 0.00015 + 0.00001 for n in (1, 8, 32)}
        capture = self.identity["capture"]
        write_json(
            self.native / "campaign.json",
            {
                "status": "complete",
                "capture_input": {
                    "archive_identity": {"sha256": capture["archive_sha256"]},
                    "manifest_sha256": capture["manifest_sha256"],
                    "review_identity": {"sha256": capture["review_sha256"]},
                },
                "summary": {
                    "layers": list(range(3, 78)),
                    "count": 864,
                    "summaries": {
                        str(n): {"mean_rank_local_ms_per_layer": value} for n, value in self.expected_means.items()
                    },
                },
            },
        )
        workloads = [(isl, n) for isl in (1024, 8192, 32768) for n in (1, 8, 32)]
        ordinals = [8, 64, 128, 192, 256, 320, 384, 448]
        for rank in range(4):
            folder = self.native / f"rank-{rank}"
            rows = []
            for block in range(3):
                for isl, n in workloads[block * 3 :] + workloads[: block * 3]:
                    for ordinal in ordinals[block * 2 :] + ordinals[: block * 2]:
                        rows.append(
                            {
                                "rank": rank,
                                "block": block,
                                "isl": isl,
                                "batch": n,
                                "ordinal": ordinal,
                                "raw": {
                                    "latency_ms": 75 * (n / 1000 + rank / 10000 + block / 100000),
                                    "used_cuda_graph": True,
                                    "num_runs_executed": 10,
                                    "throttled": False,
                                    "power_stats": None,
                                },
                                "captured_input": {"capture_source_manifest_sha256": capture["source_manifest_sha256"]},
                            }
                        )
            write_json(folder / "primary-measurements.json", rows)
            write_json(
                folder / "result.json",
                {
                    "status": "complete",
                    "rank": rank,
                    "native_boundary": self.identity["producer_boundary"],
                    "weights_before_after_equal": True,
                    "coverage": {
                        "primary_blocks": 216,
                        "layer_diagnostic_blocks": 72,
                        "dual_stream_blocks": 18,
                        "subset_blocks": 48,
                        "cache_before_after_equal": True,
                        "inputs_before_after_equal": True,
                    },
                },
            )
            for name in ("weights-before.json", "weights-after.json"):
                write_json(folder / name, {"fixture_weight_sha256": "d" * 64})
            write_json(folder / "observer-diagnostics.json", [{"dual_stream": {}}] * 18 + [{}] * 54)
            write_json(folder / "subset-measurements.json", [{}] * 48)
            write_json(
                folder / "preflight.json",
                {"machine": "aarch64", "versions": runtime["package_versions"], "torch_cuda_version": "fixture"},
            )
        self.archive = tmp_path / "native.tar.gz"
        self.review_path = tmp_path / "review.json"
        self.review = {
            "verdict": "CLEAN",
            "status": self.identity["actual_data_review_status"],
            "primary_fixed_mixture_timing_qualified": True,
            "complete_primary_data_validated": True,
            "producer_source_manifest": self.identity["producer_source_manifest"],
        }
        self.seal()

    def bind_base(self):
        for name in ("moe_perf.parquet", "collection_meta.yaml"):
            self.identity["base"][name] = record(self.table / name)

    def seal(self, extra=None):
        manifest = {
            str(p.relative_to(self.native)): record(p)
            for p in self.native.rglob("*")
            if p.is_file() and p != self.native / "file-manifest.json"
        }
        write_json(self.native / "file-manifest.json", manifest)
        self.identity["native_file_manifest"] = record(self.native / "file-manifest.json")
        with tarfile.open(self.archive, "w:gz") as handle:
            handle.add(self.native, arcname=self.native.name)
            if extra is not None:
                info, content = extra
                handle.addfile(info, io.BytesIO(content))
        with tarfile.open(self.archive) as handle:
            members = handle.getmembers()
        self.identity["native_archive"] = record(self.archive)
        self.identity["native_archive_members"] = len(members)
        self.identity["native_archive_files"] = len(manifest) + 1
        self.identity["native_expanded_bytes"] = sum(m.size for m in members)
        self.review["native_archive"] = self.identity["native_archive"]
        self.save_identity()

    def save_identity(self):
        write_json(self.review_path, self.review)
        self.identity["actual_data_review"] = record(self.review_path)
        write_json(self.identity_path, self.identity)

    def mutate_json(self, name, change):
        path = self.native / name
        data = json.loads(path.read_text())
        change(data)
        write_json(path, data)
        self.seal()

    def run(self, **kwargs):
        return publisher.publish(
            base_systems=self.base,
            output_systems=self.output,
            archive=self.archive,
            review=self.review_path,
            **kwargs,
        )


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    return Campaign(tmp_path, monkeypatch)


def test_publication_preserves_rows_history_sources_and_both_metadata_contracts(campaign):
    from aisimulate_core.sdk.perf_database import _validate_collection_meta_v2

    base_before = publisher._inventory(campaign.base)
    result = campaign.run()
    target = campaign.output / campaign.identity["table_relative_path"]
    table = pq.read_table(target / "moe_perf.parquet")
    assert table.schema.equals(pq.read_table(campaign.table / "moe_perf.parquet").schema)
    rows = table.to_pylist()
    assert rows[:81] == campaign.old_rows
    assert len(rows) == 84
    assert [r["num_tokens"] for r in rows[81:]] == [1, 8, 32]
    assert [r["latency"] for r in rows[81:]] == pytest.approx(list(campaign.expected_means.values()))
    assert {r["distribution"] for r in rows[81:]} == {publisher.PROFILE}
    meta = yaml.safe_load((target / "collection_meta.yaml").read_text())
    assert meta["schema_version"] == 2
    assert meta["tables"]["moe_perf"]["collections"] == [campaign.old_event, result["event"]]
    assert meta["runtime"] == campaign.identity["runtime"]
    provenance.validate_collection_meta_for_update(meta)
    _validate_collection_meta_v2(meta, str(target / "collection_meta.yaml"))
    event = result["event"]
    assert event["rows"] == 3 and event["source_campaign_rows"] == 864
    assert event["source_campaign_status"] == event["status"] == "complete"
    closures = provenance.load_closures(campaign.package / "collector/hash_closures.yaml")
    assert event["collector_hash"] == provenance.collector_hash(publisher.MODULE, campaign.package, closures)
    assert event["case_plan_hash"] == "sha256:dd11934097a3bc664e82c7081af5ceac3161666eb689e736b47765d7ea955cde"
    evidence = target / "evidence" / publisher.PROFILE
    assert record(evidence / "native.tar.gz") == record(campaign.archive)
    assert record(evidence / "actual-data-review.json") == record(campaign.review_path)
    assert publisher._inventory(evidence / "native") == publisher._inventory(campaign.native)
    assert yaml.safe_load((evidence / "original-table/collection_meta.yaml").read_text()) == campaign.old_meta
    manifest = json.loads((evidence / "file-manifest.json").read_text())
    assert manifest == {
        name: value for name, value in publisher._inventory(evidence).items() if name != "file-manifest.json"
    }
    assert publisher._inventory(campaign.base) == base_before
    assert (campaign.output / "unrelated.yaml").read_bytes() == (campaign.base / "unrelated.yaml").read_bytes()
    assert not list(campaign.output.rglob("*.txt"))


def test_validation_does_not_publish(campaign):
    assert campaign.run(validate_only=True)["status"] == "VALIDATED_ONLY"
    assert not campaign.output.exists()


@pytest.mark.parametrize("reader", ["python", "rust"])
def test_published_directory_passes_strict_source_resolution(campaign, reader):
    campaign.run()
    target = campaign.output / campaign.identity["table_relative_path"]
    if reader == "python":
        from aisimulate_core.sdk.perf_database import _check_strict_provenance_for_request

        _check_strict_provenance_for_request(
            [str(target)], "sglang", str(campaign.output / "data/vr200_hecate"), strict=True
        )
    else:
        import aisimulate_core as core

        report = json.loads(
            core.resolve_op_sources_report_json(
                str(campaign.output),
                str(campaign.output / "data/vr200_hecate"),
                "sglang",
                campaign.identity["runtime"]["version"],
                "moe_perf.parquet",
                enable_shared_layer=True,
                strict=True,
            )
        )
        assert report["warnings"] == []
        assert len(report["records"]) == 1
        assert report["records"][0]["path"] == str(target / "moe_perf.parquet")
    assert {path.name for path in target.iterdir()} == {"moe_perf.parquet", "collection_meta.yaml", "evidence"}


def test_only_private_released_finalization_lock_is_removed(campaign, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    original = helper.finalize_perf_files
    observed = []

    def finalize(*args, **kwargs):
        paths = original(*args, **kwargs)
        lock = paths[0].with_name("moe_perf.parquet.mergelock")
        assert lock.is_file() and lock.stat().st_size == 0
        with lock.open("rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(stream, fcntl.LOCK_UN)
        assert not campaign.output.exists()
        observed.append(lock)
        return paths

    monkeypatch.setattr(helper, "finalize_perf_files", finalize)
    campaign.run()
    assert len(observed) == 1
    assert not list(campaign.output.rglob("*.mergelock"))


@pytest.mark.parametrize(
    "name",
    [
        "unexpected.tmp",
        "moe_perf.txt",
        "evidence/new.bin",
        f"evidence/{publisher.PROFILE}/untracked.bin",
        "../foreign.csv",
    ],
)
def test_unknown_finalization_outputs_abort_before_promotion(campaign, monkeypatch, name):
    original = helper.finalize_perf_files

    def finalize(*args, **kwargs):
        paths = original(*args, **kwargs)
        extra = paths[0].parent / name
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text("unexpected helper artifact")
        return paths

    monkeypatch.setattr(helper, "finalize_perf_files", finalize)
    before = publisher._inventory(campaign.base)
    with pytest.raises(ValueError, match="Unexpected file in private publication staging"):
        campaign.run()
    assert not campaign.output.exists()
    assert publisher._inventory(campaign.base) == before


@pytest.mark.parametrize("kind", ["missing", "nonempty", "symlink", "directory"])
def test_unexpected_merge_lock_identity_aborts(campaign, monkeypatch, kind):
    original = helper.finalize_perf_files

    def finalize(*args, **kwargs):
        paths = original(*args, **kwargs)
        lock = paths[0].with_name("moe_perf.parquet.mergelock")
        if kind == "nonempty":
            lock.write_text("unexpected")
        else:
            lock.unlink()
            if kind == "symlink":
                lock.symlink_to(campaign.base / "unrelated.yaml")
            elif kind == "directory":
                lock.mkdir()
        return paths

    monkeypatch.setattr(helper, "finalize_perf_files", finalize)
    with pytest.raises(ValueError, match="Expected regular file|Unexpected private finalization lock"):
        campaign.run()
    assert not campaign.output.exists()
    assert (campaign.base / "unrelated.yaml").read_text() == "keep: exact\n"


def test_preexisting_lock_is_not_treated_as_owned_staging_artifact(campaign):
    lock = campaign.table / "moe_perf.parquet.mergelock"
    lock.write_text("preexisting")
    before = publisher._inventory(campaign.base)
    with pytest.raises(ValueError, match="Preexisting merge lock"):
        campaign.run()
    assert not campaign.output.exists()
    assert publisher._inventory(campaign.base) == before


@pytest.mark.parametrize("which", ["archive", "review", "base"])
def test_exact_input_hashes_reject_unapproved_bytes(campaign, which):
    path = {"archive": campaign.archive, "review": campaign.review_path, "base": campaign.table / "moe_perf.parquet"}[
        which
    ]
    with path.open("ab") as stream:
        stream.write(b"unapproved")
    with pytest.raises(ValueError, match="Unapproved"):
        campaign.run()
    assert not campaign.output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("verdict", "FAILED"),
        ("primary_fixed_mixture_timing_qualified", False),
        ("complete_primary_data_validated", False),
        ("status", "DIAGNOSTICS_ONLY"),
        ("producer_source_manifest", {"sha256": "wrong", "size_bytes": 1}),
    ],
)
def test_mismatched_review_cannot_authorize_publication(campaign, field, value):
    campaign.review[field] = value
    campaign.save_identity()
    with pytest.raises(ValueError, match="Review does not qualify"):
        campaign.run()
    assert not campaign.output.exists()


@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        ("campaign.json", lambda d: d.update(status="partial"), "Incomplete producer"),
        ("campaign.json", lambda d: d["summary"].update(count=863), "Incomplete primary"),
        ("campaign.json", lambda d: d["summary"]["layers"].pop(), "Incomplete 75-layer"),
        ("campaign.json", lambda d: d["capture_input"].update(manifest_sha256="wrong"), "Capture manifest"),
        ("rank-0/result.json", lambda d: d["coverage"].update(cache_before_after_equal=False), "input/cache guards"),
        ("rank-0/result.json", lambda d: d["coverage"].update(primary_blocks=215), "Incomplete measurements"),
        ("rank-0/weights-after.json", lambda d: d.update(fixture_weight_sha256="different"), "Changed native weights"),
        ("rank-0/preflight.json", lambda d: d["versions"].update(sglang="other"), "Native runtime"),
        ("rank-0/preflight.json", lambda d: d.update(machine="x86_64"), "Native runtime"),
        ("rank-0/observer-diagnostics.json", lambda d: d.pop(), "Incomplete retained diagnostic"),
        ("rank-0/subset-measurements.json", lambda d: d.pop(), "Incomplete retained diagnostic"),
        ("rank-0/primary-measurements.json", lambda d: d.pop(), "Missing, duplicate or reordered"),
        ("rank-0/primary-measurements.json", lambda d: d.__setitem__(1, copy.deepcopy(d[0])), "Missing, duplicate"),
        ("rank-0/primary-measurements.json", lambda d: d.reverse(), "Missing, duplicate or reordered"),
        ("rank-0/primary-measurements.json", lambda d: d[0].update(rank=1), "Primary rank"),
        ("rank-0/primary-measurements.json", lambda d: d[0]["raw"].update(latency_ms=-1), "Invalid measured latency"),
        ("rank-0/primary-measurements.json", lambda d: d[0]["raw"].update(latency_ms=True), "Invalid measured latency"),
        ("rank-0/primary-measurements.json", lambda d: d[0]["raw"].update(latency_ms=float("nan")), "Nonfinite JSON"),
        ("rank-0/primary-measurements.json", lambda d: d[0]["raw"].update(used_cuda_graph=False), "timing method"),
        ("rank-0/primary-measurements.json", lambda d: d[0]["raw"].update(num_runs_executed=9), "timing method"),
        ("rank-0/primary-measurements.json", lambda d: d[0]["raw"].update(throttled=True), "timing method"),
        ("rank-0/primary-measurements.json", lambda d: d[0]["raw"].update(latency_ms=123), "aggregate mismatch"),
    ],
)
def test_semantic_mutations_are_rejected_even_with_resealed_fixture(campaign, name, change, message):
    campaign.mutate_json(name, change)
    with pytest.raises(ValueError, match=message):
        campaign.run()
    assert not campaign.output.exists()


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "duplicate"])
def test_archive_paths_and_duplicates_fail_before_extraction(campaign, kind):
    name = {
        "traversal": "../escape",
        "absolute": "/escape",
        "symlink": campaign.native.name + "/link",
        "duplicate": campaign.native.name + "/campaign.json",
    }[kind]
    info = tarfile.TarInfo(name)
    if kind == "symlink":
        info.type, info.linkname = tarfile.SYMTYPE, "../escape"
    campaign.seal(extra=(info, b""))
    with pytest.raises(ValueError, match="Unsafe native|Duplicate archive"):
        campaign.run()
    assert not campaign.output.exists()
    assert not (campaign.base.parent / "escape").exists()


@pytest.mark.parametrize("field", ["distribution", "case_plan_hash"])
def test_profile_and_case_plan_are_fixed(campaign, field):
    campaign.identity[field] = "wrong"
    campaign.save_identity()
    with pytest.raises(ValueError, match="profile mismatch|Case-plan identity"):
        campaign.run()


@pytest.mark.parametrize("mutation", ["duplicate", "nonfinite", "runtime", "already_published"])
def test_invalid_original_rows_cannot_be_silently_replaced(campaign, mutation):
    rows = copy.deepcopy(campaign.old_rows)
    if mutation == "duplicate":
        rows[1] = rows[0]
    elif mutation == "nonfinite":
        rows[0]["latency"] = float("inf")
    elif mutation == "runtime":
        rows[0]["framework"] = "Other"
    else:
        rows[0]["distribution"] = publisher.PROFILE
    pq.write_table(pa.Table.from_pylist(rows), campaign.table / "moe_perf.parquet")
    campaign.bind_base()
    campaign.save_identity()
    with pytest.raises(ValueError, match="Duplicate original|Original row|Profile already"):
        campaign.run()
    assert not campaign.output.exists()


def test_finalization_failure_does_not_publish_or_mutate_inputs(campaign, monkeypatch):
    before = publisher._inventory(campaign.base)

    def fail(*args, **kwargs):
        raise RuntimeError("injected finalization failure")

    monkeypatch.setattr(helper, "finalize_perf_files", fail)
    with pytest.raises(RuntimeError, match="injected finalization"):
        campaign.run()
    assert not campaign.output.exists()
    assert publisher._inventory(campaign.base) == before


def test_existing_destination_is_never_overwritten(campaign):
    campaign.output.mkdir()
    (campaign.output / "keep").write_text("old output")
    with pytest.raises(ValueError, match="already exists"):
        campaign.run()
    assert (campaign.output / "keep").read_text() == "old output"


def test_destination_created_during_finalization_is_never_overwritten(campaign, monkeypatch):
    original = helper.finalize_perf_files

    def competing_output(*args, **kwargs):
        result = original(*args, **kwargs)
        campaign.output.mkdir()
        (campaign.output / "keep").write_text("competing output")
        return result

    monkeypatch.setattr(helper, "finalize_perf_files", competing_output)
    with pytest.raises(FileExistsError):
        campaign.run()
    assert list(campaign.output.iterdir()) == [campaign.output / "keep"]
    assert (campaign.output / "keep").read_text() == "competing output"


def test_reject_source_symlinks(campaign):
    (campaign.base / "link").symlink_to(campaign.review_path)
    with pytest.raises(ValueError, match="Symlink"):
        campaign.run()


def test_copied_producer_hash_must_match(campaign):
    (campaign.native / "source/producer.py").write_text("changed producer\n")
    campaign.seal()
    with pytest.raises(ValueError, match="Executed producer file"):
        campaign.run()


def test_missing_native_file_manifest_entry_is_rejected(campaign):
    manifest = campaign.native / "file-manifest.json"
    data = json.loads(manifest.read_text())
    data.pop("rank-3/result.json")
    write_json(manifest, data)
    campaign.identity["native_file_manifest"] = record(manifest)
    with tarfile.open(campaign.archive, "w:gz") as handle:
        handle.add(campaign.native, arcname=campaign.native.name)
    with tarfile.open(campaign.archive) as handle:
        campaign.identity["native_expanded_bytes"] = sum(m.size for m in handle.getmembers())
    campaign.identity["native_archive"] = record(campaign.archive)
    campaign.review["native_archive"] = campaign.identity["native_archive"]
    campaign.save_identity()
    with pytest.raises(ValueError, match="Native evidence hash mismatch"):
        campaign.run()
