# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY trees; real tar import gate and metadata-only portable verification.

Accepted-stage orchestration is mocked, never represented as production acceptance.
Fixture construction adapted from the separately frozen task-local closure tests.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from tools.glm53flash_hf import closed_history as h
from tools.glm53flash_hf import native_roots
from tools.glm53flash_hf import portable_history as p
from tools.glm53flash_hf import raw_archive as archive
from tools.glm53flash_hf import raw_campaign as campaign

pytestmark = pytest.mark.unit


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="TEST_ONLY_closed_history_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "original"
        self.source.mkdir()
        self.stage = self.root / "stage"
        self.stage.mkdir()
        self.binding = archive.create_storage_binding(self.source, self.source)
        self.plan = {
            "schema": campaign.PLAN,
            "stage_sha256": "a" * 64,
            "jobs": [],
            "storage_root_binding": self.binding,
        }
        self.inputs = {}
        self.ledgers = {}
        self.requests = {}
        self.next_job = 1000
        for backend in ("vllm", "sglang"):
            attempts = []
            choices = {}
            for deployment in sorted(h.DEPLOYMENTS):
                quant, tp = deployment.split("-tp")
                for phase, role, count in [
                    ("prefill", "calibration", 7),
                    ("decode", "calibration", 4),
                    ("prefill", "holdout", 4),
                    ("decode", "holdout", 3),
                ]:
                    job = {
                        "backend": backend,
                        "weight_quantization": quant,
                        "tp": int(tp),
                        "phase": phase,
                        "role": role,
                        "source_root": str(self.source / ("campaign-" + backend)),
                        "uri": "https://example.invalid/TEST_ONLY/" + backend,
                        "accepted_raw_roots": [],
                    }
                    for _ in range(count):
                        self.next_job += 1
                        attempt = self.attempt(backend, deployment, str(self.next_job), "COLLECTION_PASSED")
                        attempts.append(attempt)
                        cid = attempt["cell_id"]
                        choices[cid] = {
                            "job": attempt["job"],
                            "reason": "TEST_ONLY whole-child selection",
                            "strict_reader": {},
                        }
                        raw = (
                            self.source
                            / attempt["original_attempt_directory"]
                            / "artifacts"
                            / ("b" * 16)
                            / "cells"
                            / cid
                            / "raw/node0000"
                        )
                        raw.mkdir(parents=True)
                        (raw / "TEST_ONLY.txt").write_text("Synthetic payload only")
                        job["accepted_raw_roots"].append(str(raw))
                    self.plan["jobs"].append(job)
            self.next_job += 1
            failed = self.attempt(backend, "fp8-tp2", str(self.next_job), "COLLECTION_FAILED_PRESERVED")
            attempts.append(failed)
            request = {
                "campaign_roots": ["campaign-" + backend],
                "external_control_request": {"original_task_root": str(self.source)},
                "history_floor": [{k: failed[k] for k in ("started", "final", "checkpoint")}],
            }
            ledger = {
                "schema": "fpm_complete_child_selection_v1",
                "request_sha256": h.digest(request),
                "archive_campaign_roots": request["campaign_roots"],
                "attempts": attempts,
                "selections": choices,
            }
            self.requests[backend] = request
            self.ledgers[backend] = ledger
            rp = self.root / (backend + "-request.json")
            lp = self.root / (backend + "-ledger.json")
            h.write(rp, request)
            h.write(lp, ledger)
            self.inputs[backend] = {
                "request": h.reference(rp),
                "ledger": h.reference(lp),
            }
        self.addCleanup(patch.stopall)
        patch.object(
            campaign,
            "load_plan",
            side_effect=lambda *_: {campaign.key(j): j for j in self.plan["jobs"]},
        ).start()
        patch.object(campaign, "archive_one", side_effect=self.archive_one).start()
        patch.object(campaign, "bind", side_effect=self.public_bind).start()

    def enable_collection_scope(self):
        for job in self.plan["jobs"]:
            records = []
            for name in job["accepted_raw_roots"]:
                pod = Path(name)
                cid = pod.parts[-3]
                records.append(
                    {
                        "cell_id": cid,
                        "raw_root": str(pod.parent),
                        "attempt_id": "TEST_ONLY_attempt_" + cid.split("_")[-1],
                        native_roots.FIELD: native_roots.SCOPE,
                        "original_pod_root": str(pod),
                    }
                )
            job.update(
                native_root_scope=native_roots.SCOPE,
                accepted_native_roots=records,
                accepted_raw_roots=[r["raw_root"] for r in records],
            )

    def attempt(self, backend, deployment, job, state):
        cid = "TEST_ONLY_cell_" + job
        directory = Path("campaign-" + backend) / deployment
        directory = directory / ("00-" + cid) / job if backend == "vllm" else directory / job
        child = {
            "child_cell_id": cid,
            "deployment": deployment,
            "child_plan_sha256": "b" * 64,
        }
        native_child = {"child": child} if backend == "vllm" else {"selected": {"child_identity": child}}
        attempt = {
            "job": job,
            "cell_id": cid,
            "deployment": deployment,
            "original_attempt_directory": str(directory),
            "terminal_state": state,
        }
        for label, name, value in [
            (
                "started",
                "started.json",
                {"job": job, **native_child, "state": "RUNNING"},
            ),
            ("final", "result.json", {"job": job, **native_child, "state": state}),
            (
                "checkpoint",
                "checkpoint/fpm_forward.json",
                {
                    "cells": {
                        cid: {
                            "status": "passed" if state == "COLLECTION_PASSED" else "failed",
                            "attempt_id": "TEST_ONLY_attempt_" + job,
                        }
                    }
                },
            ),
        ]:
            path = self.source / directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            h.write(path, value)
            ref = h.reference(path)
            ref["path"] = str(directory / name)
            attempt[label] = ref
        return attempt

    def archive_one(self, stage, plan, label, output):
        job = next(j for j in plan["jobs"] if campaign.name(j) == label)
        labels = [{k: j[k] for k in campaign.FIELDS} for j in plan["jobs"] if j["backend"] == job["backend"]]
        return archive.create_archive(
            job["source_root"],
            output,
            job["uri"],
            labels,
            storage_root_binding=plan["storage_root_binding"],
        )

    def make_archive(self, backend="vllm", name=None):
        out = self.root / (name or ("bundle-" + backend))
        label = campaign.name(next(j for j in self.plan["jobs"] if j["backend"] == backend))
        ref = h.archive_closed(
            self.stage,
            self.plan,
            label,
            out,
            self.inputs,
            campaign=campaign,
            archive=archive,
        )
        return out, json.loads(Path(ref["path"]).read_bytes())

    def public_bind(self, stage, plan, bundles, destination):
        # Deliberately mocked outer accepted-stage orchestration. The unchanged
        # raw tar archive and verification are real, but this is not stage acceptance.
        destination = Path(destination)
        destination.mkdir()
        records = []
        for job in plan["jobs"]:
            label = campaign.name(job)
            bundle = Path(bundles[label])
            target = destination / label
            target.mkdir()
            receipt = json.loads((bundle / "receipt.json").read_bytes())
            row = {k: job[k] for k in campaign.FIELDS}
            row.update(
                sha256=receipt["archive"]["sha256"],
                bytes=receipt["archive"]["bytes"],
                stage_sha256=plan["stage_sha256"],
                native_roots=[
                    {
                        "cell_id": Path(p).parts[-3],
                        "raw_root": p,
                        "attempt_id": "TEST_ONLY_attempt_" + Path(p).parts[-3].split("_")[-1],
                    }
                    for p in job["accepted_raw_roots"]
                ],
            )
            if native_roots.scope(job):
                row["native_roots"] = job["accepted_native_roots"]
            for field, name in [
                ("archive_receipt", "receipt.json"),
                ("source_inventory", archive.INVENTORY),
                ("archive_input_manifest", archive.INPUT_MANIFEST),
            ]:
                shutil.copyfile(bundle / name, target / name)
                row[field] = h.reference(target / name)
            records.append(row)
        h.write(destination / "external-raw-evidence.json", records)
        return records

    def both_bundles(self):
        roots = {b: self.make_archive(b)[0] for b in ("vllm", "sglang")}
        return {campaign.name(j): roots[j["backend"]] for j in self.plan["jobs"]}

    def bind(self, bundles, name="bound"):
        destination = self.root / name
        ref = h.bind_closed(
            self.stage,
            self.plan,
            bundles,
            destination,
            self.inputs,
            campaign=campaign,
            archive=archive,
        )
        proof = json.loads(Path(ref["path"]).read_bytes())
        return destination, proof

    def prepare(self):
        self.bundles = self.both_bundles()
        bound, _ = self.bind(self.bundles)
        history = h.reference(bound / h.BOUND_PROOF)
        history["path"] = h.BOUND_PROOF
        self.portable = self.root / "portable"
        self.records = json.loads((bound / "external-raw-evidence.json").read_bytes())
        with patch.object(archive, "verify_bundle", wraps=archive.verify_bundle) as actual:
            self.entry = p.prepare_portable_history(
                bound,
                history,
                self.bundles,
                self.portable,
                expected_stage_sha256=self.plan["stage_sha256"],
                archive=archive,
            )
            self.assertEqual(actual.call_count, 2)
        return self.entry

    def verify(self, root=None, entry=None, stage=None):
        # Any accidental tar or live storage validation is a test failure.
        original = archive.validate_storage_binding

        def storage(binding, *, live=False):
            self.assertFalse(live)
            return original(binding, live=False)

        with (
            patch.object(
                archive,
                "verify_bundle",
                side_effect=AssertionError("NO offline tar verification"),
            ),
            patch.object(
                archive,
                "verify_archive",
                side_effect=AssertionError("NO offline tar stream"),
            ),
            patch.object(archive, "validate_storage_binding", side_effect=storage),
        ):
            return p.verify_portable_history(
                root or self.portable,
                entry or self.entry,
                self.records,
                expected_stage_sha256=stage or self.plan["stage_sha256"],
                archive=archive,
            )

    def proof(self):
        return json.loads((self.portable / self.entry["proof"]["path"]).read_bytes())

    def replace_proof(self, proof):
        path = self.portable / p.PROOF
        path.write_bytes(h.canonical(proof))
        self.entry["proof"] = dict(h.reference(path), path=p.PROOF)

    def replace_metadata(self, proof, key, value):
        data = h.canonical(value)
        rel = "files/" + h.sha(data)
        (self.portable / rel).write_bytes(data)
        proof[key] = {"path": rel, "sha256": h.sha(data), "bytes": len(data)}

    def test_portable_survives_deleted_tar_original_and_bound_tree(self):
        self.prepare()
        shutil.rmtree(self.source)
        shutil.rmtree(self.root / "bound")
        for bundle in set(self.bundles.values()):
            shutil.rmtree(bundle)
        relocated = self.root / "downloaded-snapshot"
        (relocated / "history").mkdir(parents=True)
        shutil.copytree(self.portable, relocated / "history", dirs_exist_ok=True)
        entry = dict(self.entry, proof=dict(self.entry["proof"], path="history/" + p.PROOF))
        result = self.verify(relocated, entry)
        self.assertEqual(result["state"], "PORTABLE_METADATA_HISTORY_PASS_NO_FRESH_TAR_VERIFICATION")
        self.assertEqual(result["labels"], 32)
        actual = {str(f.relative_to(relocated)): h.sha(f.read_bytes()) for f in relocated.rglob("*") if f.is_file()}
        self.assertEqual(result["files"], actual)
        self.assertTrue(all(name.startswith("history/") for name in result["files"]))

    def test_missing_mandatory_entry_rejected(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "required portable"):
            p.verify_portable_history(
                self.portable,
                {},
                self.records,
                expected_stage_sha256="a" * 64,
                archive=archive,
            )

    def test_changed_top_proof_rejected(self):
        self.prepare()
        with (self.portable / p.PROOF).open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            self.verify()

    def test_missing_bound_history_rejected(self):
        self.prepare()
        (self.portable / self.proof()["bound_history"]["path"]).unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.verify()

    def test_wrong_stage_rejected(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "stage/contract"):
            self.verify(stage="f" * 64)

    def test_missing_or_duplicate_label_rejected(self):
        self.prepare()
        for replacement in (self.records[:-1], self.records[:-1] + [self.records[0]]):
            with self.subTest(count=len(replacement)):
                proof = self.proof()
                bound = json.loads((self.portable / proof["bound_history"]["path"]).read_bytes())
                self.replace_metadata(proof, "external_records", replacement)
                bound["external_records"] = proof["external_records"]
                self.replace_metadata(proof, "bound_history", bound)
                self.replace_proof(proof)
                self.records = replacement
                with self.assertRaisesRegex(ValueError, "exact32"):
                    self.verify()

    def test_native_attempt_tampered_with_consistent_outer_hashes_rejected(self):
        self.prepare()
        proof = self.proof()
        bound = json.loads((self.portable / proof["bound_history"]["path"]).read_bytes())
        self.records[0]["native_roots"][0]["attempt_id"] = "TEST_ONLY_other_attempt"
        self.replace_metadata(proof, "external_records", self.records)
        bound["external_records"] = proof["external_records"]
        self.replace_metadata(proof, "bound_history", bound)
        self.replace_proof(proof)
        with self.assertRaisesRegex(ValueError, "native attempt differs"):
            self.verify()

    def test_missing_receipt_or_inventory_or_input_manifest_rejected(self):
        self.prepare()
        proof = self.proof()
        refs = next(iter(proof["bundles"].values()))
        for name, ref in refs.items():
            with self.subTest(name=name):
                path = self.portable / ref["path"]
                raw = path.read_bytes()
                path.unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.verify()
                path.write_bytes(raw)

    def test_tampered_sidecars_rejected(self):
        self.prepare()
        refs = next(iter(self.proof()["bundles"].values()))
        for name, ref in refs.items():
            with self.subTest(name=name):
                path = self.portable / ref["path"]
                raw = path.read_bytes()
                path.write_bytes(raw + b" ")
                with self.assertRaisesRegex(ValueError, "bytes changed"):
                    self.verify()
                path.write_bytes(raw)

    def test_symlink_metadata_and_escaping_proof_rejected(self):
        self.prepare()
        path = self.portable / self.proof()["external_records"]["path"]
        copy = self.root / "outside.json"
        shutil.copyfile(path, copy)
        path.unlink()
        path.symlink_to(copy)
        with self.assertRaises(ValueError):
            self.verify()
        entry = dict(self.entry, proof=dict(self.entry["proof"], path="../portable/" + p.PROOF))
        with self.assertRaises(ValueError):
            self.verify(entry=entry)

    def test_missing_bundle_label_rejected(self):
        self.prepare()
        proof = self.proof()
        proof["bundles"].pop(next(iter(proof["bundles"])))
        self.replace_proof(proof)
        with self.assertRaisesRegex(ValueError, "exact32"):
            self.verify()

    def test_original_history_missing_with_consistent_outer_hashes_rejected(self):
        self.prepare()
        proof = self.proof()
        bound = json.loads((self.portable / proof["bound_history"]["path"]).read_bytes())
        for value in bound["bundle_proofs"].values():
            if value["snapshot"]["backend"] == "vllm":
                value["snapshot"]["originals"].pop(next(iter(value["snapshot"]["originals"])))
        self.replace_metadata(proof, "bound_history", bound)
        self.replace_proof(proof)
        with self.assertRaises((ValueError, KeyError)):
            self.verify()

    def test_partial_failed_preparation_never_offline_accepted(self):
        self.prepare()
        h.write(self.portable / h.FAILURE, {"state": "TEST_ONLY_failed"})
        with self.assertRaisesRegex(ValueError, "preparation failed"):
            self.verify()

    def test_import_rejects_corrupted_tar_before_portable_output(self):
        bundles = self.both_bundles()
        bound, _ = self.bind(bundles)
        ref = dict(h.reference(bound / h.BOUND_PROOF), path=h.BOUND_PROOF)
        bundle = next(iter(bundles.values()))
        (bundle / archive.ARCHIVE).write_bytes(b"TEST_ONLY corrupt archive")
        output = self.root / "must-not-exist"
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            p.prepare_portable_history(
                bound,
                ref,
                bundles,
                output,
                expected_stage_sha256="a" * 64,
                archive=archive,
            )
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
