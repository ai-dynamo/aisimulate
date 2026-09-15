# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_pages_site.py"
SPEC = importlib.util.spec_from_file_location("build_pages_site", SCRIPT_PATH)
assert SPEC and SPEC.loader
PAGES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAGES)
FPE_SPEC = importlib.util.spec_from_file_location("prepare_fpe_pages", ROOT / "scripts" / "prepare_fpe_pages.py")
assert FPE_SPEC and FPE_SPEC.loader
FPE = importlib.util.module_from_spec(FPE_SPEC)
FPE_SPEC.loader.exec_module(FPE)

NEW_SHA = "a" * 40
OLD_SHA = "b" * 40
REPOSITORY = "ai-dynamo/aisimulate"


def qualified_archive(
    sha=NEW_SHA, *, report_updates=None, row_updates=None, missing=None, index_files=None, overrides=None
):
    report = {
        "schema_version": 1,
        "qualification": FPE.QUALIFICATION,
        "source_sha": sha,
        "wheel_sha256": "c" * 64,
        "shard_count": 1,
        "required_probe_count": 1,
        "status_counts": {"PASS": 4},
    }
    report.update(report_updates or {})
    row = {
        "HuggingFaceID": "example/model",
        "Architecture": "ExampleForCausalLM",
        "System": "b200_sxm",
        "Backend": "vllm",
        "Version": "0.24.0",
        "Status": "PASS",
        "SourceSHA": sha,
    }
    row.update(row_updates or {})
    csv_output = io.StringIO()
    writer = csv.DictWriter(csv_output, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    files = {
        "fpe-qualification.json": json.dumps(report),
        FPE.DATA_PREFIX + "index.json": json.dumps({"files": index_files or ["b200_sxm.csv"]}),
        FPE.DATA_PREFIX + "b200_sxm.csv": csv_output.getvalue(),
        "python/aisimulate/docs/fpe-support-matrix/index.html": "untrusted artifact HTML",
        "../../escape.py": "untrusted artifact code",
    }
    files.update(overrides or {})
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        for name, value in files.items():
            if name != missing:
                bundle.writestr(name, value)
    return output.getvalue()


def artifact(identifier, sha=NEW_SHA, **updates):
    value = {
        "id": identifier,
        "name": FPE.ARTIFACT_NAME,
        "expired": False,
        "created_at": "2026-09-14T22:00:00Z",
        "workflow_run": {"id": identifier, "head_branch": "main", "head_sha": sha},
    }
    value.update(updates)
    return value


def run(sha=NEW_SHA, **updates):
    value = {
        "path": ".github/workflows/fpe-support-matrix.yml",
        "head_sha": sha,
        "head_branch": "main",
        "head_repository": {"full_name": REPOSITORY},
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
    }
    value.update(updates)
    return value


class FpePagesTest(unittest.TestCase):
    def prepare(self, artifacts, runs, archives, *, output=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = output or Path(temporary.name) / "data"

        def api(repository, endpoint):
            self.assertEqual(repository, REPOSITORY)
            if endpoint.startswith("actions/artifacts?"):
                batch = artifacts[int(endpoint.rsplit("=", 1)[1])] if isinstance(artifacts, dict) else artifacts
                return json.dumps({"artifacts": batch}).encode()
            if endpoint.startswith("actions/runs/"):
                return json.dumps(runs[int(endpoint.split("/")[-1])]).encode()
            return archives[int(endpoint.split("/")[-2])]

        with patch.object(FPE.subprocess, "check_output", return_value=f"{NEW_SHA}\n{OLD_SHA}\n"):
            snapshot = FPE.prepare(REPOSITORY, ROOT, output, api=api)
        return snapshot, output

    def test_qualified_data_is_published_without_artifact_code(self):
        snapshot, output = self.prepare([artifact(1)], {1: run()}, {1: qualified_archive()})
        self.assertEqual(snapshot["source_sha"], NEW_SHA)
        self.assertEqual({p.name for p in output.iterdir()}, {"index.json", "b200_sxm.csv"})
        self.assertEqual(json.loads((output / "index.json").read_text())["snapshot"], snapshot)
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary) / "site"
            PAGES.build_site(ROOT, site, fpe_data_dir=output)
            self.assertEqual(
                (site / "data/fpe-support-matrix/b200_sxm.csv").read_bytes(), (output / "b200_sxm.csv").read_bytes()
            )
            self.assertEqual(
                (site / "data/support-matrix/b200_sxm.csv").read_bytes(),
                (ROOT / PAGES.SYSTEMS_ROOT / "support_matrix/b200_sxm.csv").read_bytes(),
            )
            self.assertNotIn("untrusted artifact", (site / "fpe-support-matrix/index.html").read_text())

    def test_later_rerun_of_old_commit_cannot_displace_newer_source(self):
        snapshot, _ = self.prepare(
            [artifact(20, OLD_SHA), artifact(10)],
            {10: run(), 20: run(OLD_SHA)},
            {10: qualified_archive(), 20: qualified_archive(OLD_SHA)},
        )
        self.assertEqual(snapshot["artifact_id"], 10)

    def test_main_dispatch_publishes_the_qualified_target_commit(self):
        snapshot, output = self.prepare([artifact(1)], {1: run()}, {1: qualified_archive(OLD_SHA)})
        self.assertEqual(snapshot["source_sha"], OLD_SHA)
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary) / "site"
            PAGES.build_site(ROOT, site, fpe_data_dir=output)
            index = json.loads((site / "data/fpe-support-matrix/index.json").read_text())
            self.assertEqual(index["snapshot"], snapshot)
            with (site / "data/fpe-support-matrix/b200_sxm.csv").open() as handle:
                self.assertEqual({row["SourceSHA"] for row in csv.DictReader(handle)}, {OLD_SHA})

    def test_later_historical_dispatch_cannot_displace_newer_tested_source(self):
        snapshot, _ = self.prepare(
            [artifact(20), artifact(10)],
            {10: run(), 20: run()},
            {10: qualified_archive(), 20: qualified_archive(OLD_SHA)},
        )
        self.assertEqual(snapshot["artifact_id"], 10)
        self.assertEqual(snapshot["source_sha"], NEW_SHA)

    def test_latest_run_of_same_source_wins(self):
        snapshot, _ = self.prepare([artifact(1), artifact(2)], {2: run()}, {2: qualified_archive()})
        self.assertEqual(snapshot["artifact_id"], 2)

    def test_newest_source_can_be_on_a_later_artifact_page(self):
        older_ids = range(100, 200)
        snapshot, _ = self.prepare(
            {1: [artifact(i, OLD_SHA) for i in older_ids], 2: [artifact(1)]},
            {**dict.fromkeys(older_ids, run(OLD_SHA)), 1: run()},
            {**dict.fromkeys(older_ids, qualified_archive(OLD_SHA)), 1: qualified_archive()},
        )
        self.assertEqual(snapshot["source_sha"], NEW_SHA)

    def test_nightly_parent_run_is_eligible(self):
        snapshot, _ = self.prepare(
            [artifact(1)], {1: run(path=".github/workflows/nightly-ci.yml", event="schedule")}, {1: qualified_archive()}
        )
        self.assertEqual(snapshot["source_sha"], NEW_SHA)

    def test_partial_nightly_retry_cannot_publish_older_data(self):
        for status, conclusion in [("waiting", None), ("in_progress", None), ("completed", "failure")]:
            with self.subTest(status=status, conclusion=conclusion), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "data"
                with self.assertRaisesRegex(ValueError, "retried"):
                    self.prepare(
                        [artifact(10), artifact(20, OLD_SHA)],
                        {
                            10: run(
                                path=".github/workflows/nightly-ci.yml",
                                event="schedule",
                                status=status,
                                conclusion=conclusion,
                                run_attempt=2,
                            ),
                            20: run(OLD_SHA),
                        },
                        {10: qualified_archive(), 20: qualified_archive(OLD_SHA)},
                        output=output,
                    )
                self.assertFalse(output.exists())

    def test_missing_run_attempt_cannot_publish_older_data(self):
        for status, conclusion in [
            ("waiting", None),
            ("in_progress", None),
            ("completed", "failure"),
            ("completed", "success"),
        ]:
            with self.subTest(status=status, conclusion=conclusion), tempfile.TemporaryDirectory() as temporary:
                latest_run = run(status=status, conclusion=conclusion)
                del latest_run["run_attempt"]
                output = Path(temporary) / "data"
                with self.assertRaisesRegex(ValueError, "run_attempt"):
                    self.prepare(
                        [artifact(10), artifact(20, OLD_SHA)],
                        {10: latest_run, 20: run(OLD_SHA)},
                        {10: qualified_archive(), 20: qualified_archive(OLD_SHA)},
                        output=output,
                    )
                self.assertFalse(output.exists())

    def test_invalid_run_attempt_cannot_publish_data(self):
        for run_attempt in (None, True, False, 0, -1, 0.5, 1.0, 2.0, "1", "2"):
            for status, conclusion in [
                ("waiting", None),
                ("in_progress", None),
                ("completed", "failure"),
                ("completed", "success"),
            ]:
                with (
                    self.subTest(run_attempt=run_attempt, status=status, conclusion=conclusion),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    output = Path(temporary) / "data"
                    with self.assertRaisesRegex(ValueError, "run_attempt"):
                        self.prepare(
                            [artifact(10), artifact(20, OLD_SHA)],
                            {
                                10: run(status=status, conclusion=conclusion, run_attempt=run_attempt),
                                20: run(OLD_SHA),
                            },
                            {10: qualified_archive(), 20: qualified_archive(OLD_SHA)},
                            output=output,
                        )
                    self.assertFalse(output.exists())

    def test_successful_retry_remains_eligible(self):
        snapshot, _ = self.prepare([artifact(1)], {1: run(run_attempt=2)}, {1: qualified_archive()})
        self.assertEqual(snapshot["source_sha"], NEW_SHA)

    def test_retry_of_older_tested_source_does_not_block_newer_snapshot(self):
        snapshot, _ = self.prepare(
            [artifact(20), artifact(10)],
            {10: run(), 20: run(status="in_progress", conclusion=None, run_attempt=2)},
            {10: qualified_archive(), 20: qualified_archive(OLD_SHA)},
        )
        self.assertEqual(snapshot["artifact_id"], 10)
        self.assertEqual(snapshot["source_sha"], NEW_SHA)

    def test_ineligible_producers_cannot_publish(self):
        for changes in [
            {"conclusion": "failure"},
            {"status": "in_progress"},
            {"event": "pull_request"},
            {"path": ".github/workflows/other.yml"},
            {"head_branch": "feature"},
            {"head_repository": {"full_name": "someone/fork"}},
            {"head_sha": OLD_SHA},
        ]:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "no retained qualified"):
                self.prepare([artifact(1)], {1: run(**changes)}, {})

    def test_source_outside_main_history_cannot_publish(self):
        with self.assertRaisesRegex(ValueError, "no retained qualified"):
            self.prepare([artifact(1, "d" * 40)], {}, {})

    def test_main_dispatch_target_outside_main_history_cannot_publish(self):
        with self.assertRaisesRegex(ValueError, "no retained qualified"):
            self.prepare([artifact(1)], {1: run()}, {1: qualified_archive("d" * 40)})

    def test_malformed_off_main_qualification_cannot_fall_back_to_older_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "data"
            with self.assertRaisesRegex(ValueError, "invalid FPE qualification"):
                self.prepare(
                    [artifact(20), artifact(10, OLD_SHA)],
                    {20: run(), 10: run(OLD_SHA)},
                    {
                        20: qualified_archive(
                            overrides={"fpe-qualification.json": json.dumps({"source_sha": "d" * 40})}
                        ),
                        10: qualified_archive(OLD_SHA),
                    },
                    output=output,
                )
            self.assertFalse(output.exists())

    def test_mixed_sources_with_off_main_manifest_cannot_fall_back_to_older_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "data"
            with self.assertRaisesRegex(ValueError, "mixed source"):
                self.prepare(
                    [artifact(20), artifact(10, OLD_SHA)],
                    {20: run(), 10: run(OLD_SHA)},
                    {
                        20: qualified_archive("d" * 40, row_updates={"SourceSHA": NEW_SHA}),
                        10: qualified_archive(OLD_SHA),
                    },
                    output=output,
                )
            self.assertFalse(output.exists())

    def test_mixed_sources_cannot_hide_behind_an_older_manifest_source(self):
        with self.assertRaisesRegex(ValueError, "mixed source"):
            self.prepare(
                [artifact(20), artifact(10)],
                {10: run(), 20: run()},
                {
                    20: qualified_archive(OLD_SHA),
                    10: qualified_archive(OLD_SHA, row_updates={"SourceSHA": NEW_SHA}),
                },
            )

    def test_expired_newer_snapshot_does_not_fall_back_to_old_data(self):
        with self.assertRaisesRegex(ValueError, "has expired"):
            self.prepare(
                [artifact(1, expired=True), artifact(2, OLD_SHA)],
                {1: run(), 2: run(OLD_SHA)},
                {2: qualified_archive(OLD_SHA)},
            )

    def test_current_head_snapshot_supersedes_expired_older_artifact(self):
        snapshot, _ = self.prepare(
            [artifact(10, expired=True), artifact(20, OLD_SHA)],
            {10: run(), 20: run(OLD_SHA)},
            {20: qualified_archive()},
        )
        self.assertEqual(snapshot["artifact_id"], 20)
        self.assertEqual(snapshot["source_sha"], NEW_SHA)

    def test_pre_qualification_artifact_is_not_published(self):
        with self.assertRaisesRegex(ValueError, "no retained qualified"):
            self.prepare([artifact(1)], {1: run()}, {1: qualified_archive(missing="fpe-qualification.json")})

    def test_duplicate_entries_without_qualification_are_still_rejected(self):
        archive = io.BytesIO(qualified_archive(missing="fpe-qualification.json"))
        with zipfile.ZipFile(archive, "a") as bundle, self.assertWarns(UserWarning):
            bundle.writestr("../../escape.py", "duplicate untrusted code")
        with self.assertRaisesRegex(ValueError, "duplicate archive entries"):
            self.prepare([artifact(1)], {1: run()}, {1: archive.getvalue()})

    def test_no_artifact_fails_instead_of_using_committed_snapshot(self):
        with self.assertRaisesRegex(ValueError, "no retained qualified"):
            self.prepare([], {}, {})

    def test_invalid_qualification_is_rejected(self):
        for changes in [
            {"qualification": "incomplete"},
            {"source_sha": OLD_SHA},
            {"wheel_sha256": ""},
            {"shard_count": 2},
            {"required_probe_count": 0},
            {"status_counts": {"PASS": 4, "BUILD_FAILED": 1}},
            {"status_counts": {"PERF_DATA_MISSING": 4}},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                FPE.qualified_files(qualified_archive(report_updates=changes), NEW_SHA)

    def test_mixed_row_identity_or_unknown_status_is_rejected(self):
        for changes in [{"SourceSHA": OLD_SHA}, {"System": "h200_sxm"}, {"Status": "HYBRID_PASS"}, {"Backend": ""}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                FPE.qualified_files(qualified_archive(row_updates=changes), NEW_SHA)

    def test_qualification_numeric_fields_reject_boolean_and_coerced_values(self):
        for field in ("schema_version", "shard_count", "required_probe_count"):
            for value in (True, False, 1.0, "1", None):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    FPE.qualified_files(qualified_archive(report_updates={field: value}), NEW_SHA)

    def test_status_counts_reject_malformed_objects_and_values(self):
        for statuses in (
            None,
            [],
            "PASS",
            {"PASS": True},
            {"PASS": -1},
            {"PASS": 4.0},
            {"PASS": "4"},
            {"PASS": 4, "PERF_DATA_MISSING": False},
            {"PASS": 4, "PERF_DATA_MISSING": -1},
        ):
            with self.subTest(statuses=statuses), self.assertRaises(ValueError):
                FPE.qualified_files(qualified_archive(report_updates={"status_counts": statuses}), NEW_SHA)

    def test_zero_nonpassing_status_count_remains_valid(self):
        files = FPE.qualified_files(
            qualified_archive(report_updates={"status_counts": {"PASS": 4, "BUILD_FAILED": 0}}), NEW_SHA
        )
        self.assertIn("b200_sxm.csv", files)

    def test_unknown_probe_status_keys_are_rejected(self):
        for status in ("BUILD_FAILED ", "QUERY_FAILED ", "UNKNOWN", "pass"):
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "unknown status-count keys"):
                FPE.qualified_files(
                    qualified_archive(report_updates={"status_counts": {"PASS": 4, status: 1}}), NEW_SHA
                )

    def test_probe_status_contract_matches_the_qualifier(self):
        path = ROOT / "python/aisimulate/tools/support_matrix/qualify_fpe_support_matrix.py"
        spec = importlib.util.spec_from_file_location("qualify_fpe", path)
        qualifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qualifier)
        self.assertEqual(FPE.PROBE_STATUSES, qualifier.STATUSES)
        counts = dict.fromkeys(qualifier.STATUSES, 0)
        counts["PASS"] = 4
        self.assertIn(
            "b200_sxm.csv", FPE.qualified_files(qualified_archive(report_updates={"status_counts": counts}), NEW_SHA)
        )

    def test_ambiguous_or_malformed_csv_records_are_rejected(self):
        name = FPE.DATA_PREFIX + "b200_sxm.csv"
        with zipfile.ZipFile(io.BytesIO(qualified_archive())) as archive:
            data = archive.read(name).decode()
        header, row = data.splitlines()
        invalid_csvs = (
            header.replace("Status,", "Status,Status,") + "\n" + row.replace(",PASS,", ",FAIL,PASS,") + "\n",
            data + row + "\n",
            header + "\n" + row + ",extra\n",
            header + ",Extra\n" + row + "\n",
            header + ",ErrMsg\n" + row + ',"unterminated',
        )
        for data in invalid_csvs:
            with self.subTest(data=data), self.assertRaises((ValueError, csv.Error)):
                FPE.qualified_files(qualified_archive(overrides={name: data}), NEW_SHA)

    def test_duplicate_json_fields_are_rejected_at_every_depth(self):
        with zipfile.ZipFile(io.BytesIO(qualified_archive())) as archive:
            manifest = archive.read("fpe-qualification.json").decode()
        ambiguous_manifests = (
            manifest.replace('"schema_version": 1', '"schema_version": false, "schema_version": 1'),
            manifest.replace('"status_counts":', '"status_counts": {}, "status_counts":'),
            manifest.replace('"PASS": 4', '"PASS": 0, "PASS": 4'),
        )
        for manifest in ambiguous_manifests:
            with self.subTest(manifest=manifest), self.assertRaisesRegex(ValueError, "duplicate JSON member"):
                FPE.qualified_files(qualified_archive(overrides={"fpe-qualification.json": manifest}), NEW_SHA)
        with self.assertRaisesRegex(ValueError, "duplicate JSON member"):
            FPE.qualified_files(
                qualified_archive(
                    overrides={FPE.DATA_PREFIX + "index.json": '{"files": ["missing.csv"], "files": ["b200_sxm.csv"]}'}
                ),
                NEW_SHA,
            )

    def test_non_json_numeric_constants_are_rejected(self):
        with zipfile.ZipFile(io.BytesIO(qualified_archive())) as archive:
            manifest = archive.read("fpe-qualification.json").decode()
        for constant in ("NaN", "Infinity", "-Infinity"):
            for name, data in (
                ("fpe-qualification.json", manifest[:-1] + ', "extra": ' + constant + "}"),
                (FPE.DATA_PREFIX + "index.json", '{"files": ["b200_sxm.csv"], "extra": ' + constant + "}"),
            ):
                with (
                    self.subTest(name=name, constant=constant),
                    self.assertRaisesRegex(ValueError, "invalid JSON constant"),
                ):
                    FPE.qualified_files(qualified_archive(overrides={name: data}), NEW_SHA)

    def test_overflowing_json_numbers_cannot_publish_or_fall_back(self):
        with zipfile.ZipFile(io.BytesIO(qualified_archive())) as archive:
            documents = {
                name: archive.read(name).decode() for name in ("fpe-qualification.json", FPE.DATA_PREFIX + "index.json")
            }
        for name, data in documents.items():
            for value in ("1e999", "-1e999", '{"nested": [1e999]}', '{"nested": [-1e999]}'):
                with self.subTest(name=name, value=value), tempfile.TemporaryDirectory() as temporary:
                    output = Path(temporary) / "data"
                    with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
                        self.prepare(
                            [artifact(10), artifact(20, OLD_SHA)],
                            {10: run(), 20: run(OLD_SHA)},
                            {
                                10: qualified_archive(overrides={name: data[:-1] + ', "extra": ' + value + "}"}),
                                20: qualified_archive(OLD_SHA),
                            },
                            output=output,
                        )
                    self.assertFalse(output.exists())

    def test_finite_numeric_metadata_remains_valid_in_published_index(self):
        metadata = {"fraction": 0.5, "nested": [-0.5, 1.7976931348623157e308, -1.7976931348623157e308, 5e-324]}
        snapshot, output = self.prepare(
            [artifact(1)],
            {1: run()},
            {
                1: qualified_archive(
                    report_updates={"extra": metadata},
                    overrides={
                        FPE.DATA_PREFIX + "index.json": json.dumps({"files": ["b200_sxm.csv"], "extra": metadata})
                    },
                )
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary) / "site"
            PAGES.build_site(ROOT, site, fpe_data_dir=output)
            index = json.loads((site / "data/fpe-support-matrix/index.json").read_text(), parse_constant=self.fail)
            self.assertEqual(index["extra"], metadata)
            self.assertEqual(index["snapshot"], snapshot)

    def test_missing_or_unsafe_indexed_csv_is_rejected(self):
        with self.assertRaises(KeyError):
            FPE.qualified_files(qualified_archive(missing=FPE.DATA_PREFIX + "b200_sxm.csv"), NEW_SHA)
        for files in [["../../secret.csv"], ["b200_sxm.csv", "b200_sxm.csv"]]:
            with self.subTest(files=files), self.assertRaises(ValueError):
                FPE.qualified_files(qualified_archive(index_files=files), NEW_SHA)


class LegacySnapshotTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repository = self.root / "repo"
        self.repository.mkdir()
        self.git("init", "-b", "main")
        docs = self.repository / PAGES.DOCS_ROOT
        docs.mkdir(parents=True)
        (docs / "index.html").write_text("landing page")
        for page in PAGES.PUBLIC_PAGE_DIRECTORIES:
            (docs / page).mkdir()
            (docs / page / "index.html").write_text(page)
        for name in PAGES.PUBLIC_DATASETS.values():
            dataset = self.repository / PAGES.SYSTEMS_ROOT / name
            dataset.mkdir(parents=True)
            (dataset / "index.json").write_text(json.dumps({"files": ["b200_sxm.csv"]}))
            (dataset / "b200_sxm.csv").write_text("Model,Status\nexample/model,PASS\n")
        self.dataset = self.repository / PAGES.SYSTEMS_ROOT / "support_matrix"
        self.data_sha = self.commit("2026-09-04T11:42:50-07:00", "legacy data")
        (docs / "support-matrix/index.html").write_text("new website, same data")
        self.commit("2026-09-15T12:00:00-07:00", "website only")

    def git(self, *args, **kwargs):
        return subprocess.check_output(
            ["git", *args], cwd=self.repository, text=True, stderr=subprocess.DEVNULL, **kwargs
        ).strip()

    def commit(self, date, message):
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            message,
            env={**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date},
        )
        return self.git("rev-parse", "HEAD")

    def snapshot(self, repository=None):
        with tempfile.TemporaryDirectory(dir=self.root) as output:
            site = Path(output) / "site"
            PAGES.build_site(repository or self.repository, site)
            return json.loads((site / "data/support-matrix/index.json").read_text())["snapshot"]

    def test_website_rebuild_keeps_data_commit_and_date(self):
        self.assertEqual(
            self.snapshot(),
            {
                "kind": "historical",
                "qualification": "not_recorded",
                "data_commit": self.data_sha,
                "data_updated_at": "2026-09-04T18:42:50Z",
            },
        )

    def test_data_change_advances_snapshot(self):
        (self.dataset / "b200_sxm.csv").write_text("Model,Status\nexample/model,FAIL\n")
        changed = self.commit("2026-09-16T09:00:00Z", "changed data")
        self.assertEqual(self.snapshot()["data_commit"], changed)
        self.assertEqual(self.snapshot()["data_updated_at"], "2026-09-16T09:00:00Z")

    def test_unindexed_file_does_not_advance_snapshot(self):
        (self.dataset / "unpublished.csv").write_text("unused data")
        self.commit("2026-09-16T09:00:00Z", "unpublished data")
        self.assertEqual(self.snapshot()["data_commit"], self.data_sha)

    def test_dirty_and_staged_data_have_no_committed_date(self):
        (self.dataset / "b200_sxm.csv").write_text("Model,Status\nexample/model,FAIL\n")
        unknown = {"kind": "historical", "qualification": "not_recorded"}
        self.assertEqual(self.snapshot(), unknown)
        self.git("add", ".")
        self.assertEqual(self.snapshot(), unknown)

    def test_untracked_indexed_file_has_no_committed_date(self):
        (self.dataset / "index.json").write_text(json.dumps({"files": ["new.csv"]}))
        self.commit("2026-09-16T09:00:00Z", "new index")
        (self.dataset / "new.csv").write_text("Model,Status\nexample/model,PASS\n")
        self.assertNotIn("data_commit", self.snapshot())

    def test_source_archive_does_not_preserve_unverified_snapshot(self):
        index = self.dataset / "index.json"
        index.write_text(
            json.dumps(
                {
                    "files": ["b200_sxm.csv"],
                    "snapshot": {"qualification": "qualified", "data_updated_at": "2026-09-15T12:00:00Z"},
                }
            )
        )
        archive = self.root / "archive"
        shutil.copytree(self.repository, archive, ignore=shutil.ignore_patterns(".git"))
        self.assertEqual(self.snapshot(archive), {"kind": "historical", "qualification": "not_recorded"})

    def test_shallow_clone_does_not_claim_tip_date_as_data_date(self):
        shallow = self.root / "shallow"
        self.git("clone", "--depth=1", self.repository.as_uri(), str(shallow))
        self.assertEqual(self.snapshot(shallow), {"kind": "historical", "qualification": "not_recorded"})


class PagesSiteTest(unittest.TestCase):
    def test_public_pages_artifact_is_allowlisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "site"
            files = PAGES.build_site(ROOT, output_dir)

            self.assertIn(Path("index.html"), files)
            self.assertIn(Path("e2e-accuracy/index.html"), files)
            self.assertIn(Path("e2e-accuracy/app.js"), files)
            self.assertIn(Path("e2e-accuracy/styles.css"), files)
            self.assertIn(Path("e2e-accuracy/summary.json"), files)
            self.assertIn(Path("fpe-support-matrix/index.html"), files)
            self.assertIn(Path("fpe-support-matrix/fpe-support-matrix-preview.png"), files)
            self.assertIn(Path("support-matrix/index.html"), files)
            self.assertIn(Path("data/fpe-support-matrix/index.json"), files)
            self.assertIn(Path("data/support-matrix/index.json"), files)
            self.assertTrue(any(path.match("data/fpe-support-matrix/*.csv") for path in files))
            self.assertTrue(any(path.match("data/support-matrix/*.csv") for path in files))

            self.assertFalse(any(path.parts[0] == "universe" for path in files))
            self.assertFalse(any(path.suffix == ".md" for path in files))
            self.assertFalse(any("src" in path.parts for path in files))

            landing_page = (output_dir / "index.html").read_text()
            self.assertIn('href="./e2e-accuracy/"', landing_page)
            self.assertIn('href="./fpe-support-matrix/"', landing_page)
            self.assertIn('href="./support-matrix/"', landing_page)
            self.assertNotIn('href="./universe/"', landing_page)

            for path in files:
                if path.suffix not in {".html", ".js"}:
                    continue
                text = (output_dir / path).read_text()
                self.assertNotIn("raw.githubusercontent.com", text)
                self.assertNotIn("api.github.com/repos/ai-dynamo/aisimulate", text)

    def test_deployed_matrices_use_packaged_public_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "site"
            PAGES.build_site(ROOT, output_dir)

            legacy_page = (output_dir / "support-matrix" / "index.html").read_text()
            self.assertIn("../data/support-matrix", legacy_page)
            self.assertNotIn("raw.githubusercontent.com", legacy_page)

            fpe_page = (output_dir / "fpe-support-matrix" / "index.html").read_text()
            self.assertIn("../data/fpe-support-matrix", fpe_page)
            self.assertIn('href="../"', fpe_page)
            self.assertNotIn("raw.githubusercontent.com", fpe_page)
            self.assertNotIn("api.github.com/repos/ai-dynamo/aisimulate", fpe_page)

    def test_public_artifact_rejects_symlinked_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            private_file = temporary_root / "private.html"
            symlink = temporary_root / "public.html"
            destination = temporary_root / "site" / "public.html"
            private_file.write_text("private")
            symlink.symlink_to(private_file)

            with self.assertRaisesRegex(PAGES.PagesBuildError, "cannot be a symlink"):
                PAGES._copy_file(symlink, destination)


if __name__ == "__main__":
    unittest.main()
