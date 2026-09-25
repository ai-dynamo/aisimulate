# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY trees; real bf5e tar validation, mocked public accepted-stage transport."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from tools.glm53flash_hf import closed_history as h
from tools.glm53flash_hf import raw_archive as archive
from tools.glm53flash_hf import raw_campaign as campaign

pytestmark = pytest.mark.unit


class ClosureTests(unittest.TestCase):
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

    def test_real_archive_history_keeps_failed_original(self):
        out, proof = self.make_archive()
        actual = h.verify_bundle_history(out, proof, archive)
        self.assertEqual(len(actual["ledger"]["attempts"]), 73)
        self.assertEqual(
            actual["ledger"]["attempts"][-1]["terminal_state"],
            "COLLECTION_FAILED_PRESERVED",
        )

    def test_missing_history_rejected_before_output(self):
        out = self.root / "unmade"
        with self.assertRaisesRegex(ValueError, "history missing"):
            h.archive_closed(
                self.stage,
                self.plan,
                campaign.name(self.plan["jobs"][0]),
                out,
                {},
                campaign=campaign,
                archive=archive,
            )
        self.assertFalse(out.exists())

    def test_late_stable_attempt_after_plan_rejected(self):
        self.attempt("vllm", "fp8-tp2", "999999", "RUNNING")
        with self.assertRaisesRegex(ValueError, "new or omitted"):
            self.make_archive()
        self.assertFalse((self.root / "bundle-vllm").exists())

    def test_new_stable_attempt_during_archive_fails_and_preserves_tar(self):
        def operation(*args):
            result = self.archive_one(*args)
            self.attempt("vllm", "fp8-tp2", "999999", "RUNNING")
            return result

        with (
            patch.object(campaign, "archive_one", side_effect=operation),
            self.assertRaisesRegex(ValueError, "new or omitted"),
        ):
            self.make_archive()
        out = self.root / "bundle-vllm"
        self.assertTrue((out / archive.ARCHIVE).exists())
        self.assertTrue((out / h.FAILURE).exists())
        self.assertFalse((out / h.BUNDLE_PROOF).exists())

    def test_changed_terminal_bytes_during_archive_rejected(self):
        def operation(*args):
            result = self.archive_one(*args)
            p = self.source / self.ledgers["vllm"]["attempts"][0]["final"]["path"]
            p.write_text("{}")
            return result

        with (
            patch.object(campaign, "archive_one", side_effect=operation),
            self.assertRaisesRegex(ValueError, "metadata changed"),
        ):
            self.make_archive()

    def test_consistently_rebound_nonterminal_final_cannot_be_history(self):
        ledger = self.ledgers["vllm"]
        item = ledger["attempts"][-1]
        p = self.source / item["final"]["path"]
        value = json.loads(p.read_bytes())
        value["state"] = "RUNNING"
        p.write_text(json.dumps(value))
        item["terminal_state"] = "RUNNING"
        ref = h.reference(p)
        ref["path"] = item["final"]["path"]
        item["final"] = ref
        req = self.requests["vllm"]
        req["history_floor"] = []
        ledger["request_sha256"] = h.digest(req)
        for key, value in [("request", req), ("ledger", ledger)]:
            p = self.root / ("changed-" + key + ".json")
            h.write(p, value)
            self.inputs["vllm"][key] = h.reference(p)
        with self.assertRaisesRegex(ValueError, "nonterminal"):
            self.make_archive()

    def test_rebound_proof_cannot_drop_failed_archived_attempt(self):
        out, proof = self.make_archive()
        snap = proof["snapshot"]
        failed = snap["ledger"]["attempts"].pop()
        snap["request"]["history_floor"] = []
        snap["ledger"]["request_sha256"] = h.digest(snap["request"])
        campaign_root = Path(snap["request"]["campaign_roots"][0])
        for key in ("started", "final", "checkpoint"):
            del snap["originals"][str(Path(failed[key]["path"]).relative_to(campaign_root))]
        for key in ("request", "ledger"):
            raw = json.dumps(snap[key]).encode()
            snap["input_bytes"][key] = h._record(raw)
            snap["inputs"][key].update(sha256=h.sha(raw), bytes=len(raw))
        with self.assertRaisesRegex(ValueError, "omitted or added"):
            h.verify_bundle_history(out, proof, archive)

    def test_new_tar_with_late_attempt_rejects_even_consistent_sidecar_hashes(self):
        _, proof = self.make_archive()
        self.attempt("vllm", "fp8-tp2", "999999", "RUNNING")
        out = self.root / "later-tar"
        self.archive_one(self.stage, self.plan, campaign.name(self.plan["jobs"][0]), out)
        proof["bundle_files"] = h._bundle_files(out, archive)
        with self.assertRaisesRegex(ValueError, "omitted or added"):
            h.verify_bundle_history(out, proof, archive)

    def test_offline_verification_needs_no_original_live_tree(self):
        out, proof = self.make_archive()
        shutil.rmtree(self.source / "campaign-vllm")
        self.assertEqual(h.verify_bundle_history(out, proof, archive)["backend"], "vllm")

    def test_32_label_bind_and_required_offline_gate(self):
        bundles = self.both_bundles()
        dest, proof = self.bind(bundles)
        ref = h.reference(dest / h.BOUND_PROOF)
        ref["path"] = h.BOUND_PROOF
        manifest = {"external_raw_history": {"contract": h.CONTRACT, "proof": ref}}
        result = h.require_publication_history(
            manifest,
            dest,
            bundles,
            expected_stage_sha256=self.plan["stage_sha256"],
            archive=archive,
        )
        self.assertEqual((result["histories"], result["labels"]), (2, 32))
        self.assertIn("STREAMED_ARCHIVE_GATE", proof["state"])

    def test_sidecar_alone_without_mandatory_manifest_contract_rejected(self):
        with self.assertRaisesRegex(ValueError, "publication history missing"):
            h.require_publication_history({}, self.root, {}, expected_stage_sha256="a" * 64, archive=archive)

    def test_bare_old_archive_without_history_cannot_bind(self):
        bundles = self.both_bundles()
        first = next(iter(bundles.values()))
        (first / h.BUNDLE_PROOF).unlink()
        with self.assertRaisesRegex(ValueError, "history missing"):
            self.bind(bundles)

    def test_late_stable_attempt_before_bind_rejected(self):
        bundles = self.both_bundles()
        self.attempt("sglang", "fp8-tp2", "999999", "RUNNING")
        with self.assertRaisesRegex(ValueError, "new or omitted"):
            self.bind(bundles)
        self.assertFalse((self.root / "bound").exists())

    def test_late_attempt_during_bind_preserves_failed_output(self):
        bundles = self.both_bundles()

        def operation(*args):
            result = self.public_bind(*args)
            self.attempt("vllm", "fp8-tp2", "999999", "RUNNING")
            return result

        with (
            patch.object(campaign, "bind", side_effect=operation),
            self.assertRaisesRegex(ValueError, "new or omitted"),
        ):
            self.bind(bundles)
        self.assertTrue((self.root / "bound" / h.FAILURE).exists())
        self.assertFalse((self.root / "bound" / h.BOUND_PROOF).exists())

    def test_consistent_record_rehash_cannot_substitute_archive_identity(self):
        bundles = self.both_bundles()
        dest, proof = self.bind(bundles)
        p = dest / "external-raw-evidence.json"
        records = json.loads(p.read_bytes())
        records[0]["sha256"] = "0" * 64
        p.write_text(json.dumps(records))
        proof["external_records"] = h.reference(p)
        with self.assertRaisesRegex(ValueError, "accepted archive"):
            h.verify_bound_history(
                dest,
                proof,
                bundles,
                expected_stage_sha256=self.plan["stage_sha256"],
                archive=archive,
            )

    def test_unknown_or_missing_label_cannot_be_accepted(self):
        bundles = self.both_bundles()
        dest, proof = self.bind(bundles)
        proof["bundle_proofs"].pop(next(iter(bundles)))
        with self.assertRaisesRegex(ValueError, "exact32"):
            h.verify_bound_history(
                dest,
                proof,
                bundles,
                expected_stage_sha256=self.plan["stage_sha256"],
                archive=archive,
            )

    def test_job_directory_without_started_is_not_silently_omitted(self):
        folder = self.source / "campaign-sglang/fp8-tp2/999999"
        folder.mkdir(parents=True)
        h.write(folder / "result.json", {"state": "FAILED_PRESERVED"})
        with self.assertRaisesRegex(ValueError, "original attempt directory"):
            self.make_archive("sglang")

    def test_offline_history_native_attempt_id_must_match_original_checkpoint(self):
        bundles = self.both_bundles()
        dest, proof = self.bind(bundles)
        p = dest / "external-raw-evidence.json"
        records = json.loads(p.read_bytes())
        records[0]["native_roots"][0]["attempt_id"] = "TEST_ONLY_spliced_attempt"
        p.write_text(json.dumps(records))
        proof["external_records"] = h.reference(p)
        with self.assertRaisesRegex(ValueError, "original checkpoint"):
            h.verify_bound_history(
                dest,
                proof,
                bundles,
                expected_stage_sha256=self.plan["stage_sha256"],
                archive=archive,
            )

    def test_offline_history_raw_root_cannot_move_between_attempts(self):
        bundles = self.both_bundles()
        dest, proof = self.bind(bundles)
        p = dest / "external-raw-evidence.json"
        records = json.loads(p.read_bytes())
        records[0]["native_roots"][0]["raw_root"] = str(
            self.source / "other_job/cells" / records[0]["native_roots"][0]["cell_id"] / "raw/node0000"
        )
        p.write_text(json.dumps(records))
        proof["external_records"] = h.reference(p)
        with self.assertRaisesRegex(ValueError, "whole-child selection"):
            h.verify_bound_history(
                dest,
                proof,
                bundles,
                expected_stage_sha256=self.plan["stage_sha256"],
                archive=archive,
            )

    def test_required_publication_reference_digest_cannot_be_rebound_silently(self):
        bundles = self.both_bundles()
        dest, proof = self.bind(bundles)
        ref = h.reference(dest / h.BOUND_PROOF)
        ref["path"] = h.BOUND_PROOF
        p = dest / h.BOUND_PROOF
        p.write_bytes(p.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "digest changed"):
            h.require_publication_history(
                {"external_raw_history": {"contract": h.CONTRACT, "proof": ref}},
                dest,
                bundles,
                expected_stage_sha256=self.plan["stage_sha256"],
                archive=archive,
            )


if __name__ == "__main__":
    unittest.main()
