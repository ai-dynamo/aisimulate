# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY synthetic original files. No actual Slurm/native/accuracy work."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pytest
from tools.glm53flash_hf import cleanup_executor as executor
from tools.glm53flash_hf import cleanup_reconciliation as c

pytestmark = pytest.mark.unit


def ref(root, path):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(root)), "sha256": c.digest(raw), "bytes": len(raw)}


class Fixture:
    def __init__(self, root, *, backend="vllm", job="1001", cid="TEST_ONLY_cell", directory=None):
        self.root = root
        self.job, self.cid, self.backend = job, cid, backend
        self.directory = Path(
            directory or (f"campaign/fp8-tp2/00-{cid}/{job}" if backend == "vllm" else f"campaign/fp8-tp2/{job}")
        )
        self.plan_sha = "b" * 64
        self.cell = root / self.directory / "artifacts" / self.plan_sha[:16] / "cells" / cid
        self.cell.mkdir(parents=True, exist_ok=True)
        self.step = "fpm-" + c.digest(str(self.cell.resolve()).encode())[:20]
        self.points = [
            {"batch_size": 1, "total_prefill_tokens": 2, "total_kv_read_tokens": 0},
            {"batch_size": 1, "total_prefill_tokens": 3, "total_kv_read_tokens": 0},
        ]
        self.plan = {
            "sha256": self.plan_sha,
            "backend": backend,
            "cells": [{"cell_id": cid, "workload_kind": "prefill"}],
            "options": {"benchmark_points": {"payload": {"prefill": self.points, "decode": []}}},
        }
        self.child = {
            "child_cell_id": cid,
            "child_plan_sha256": self.plan_sha,
            "phase": "prefill",
            "original_point_ids": [1, 2],
            "native_directory": str(root / "prepared/native" / cid),
        }
        self.commit = next(iter(c.HOST_SOURCES)) if backend == "vllm" else list(c.HOST_SOURCES)[1]
        child_key = {"child": self.child} if backend == "vllm" else {"selected": {"child_identity": self.child}}
        start = {"job": job, "source_commit": self.commit, **child_key}
        final = {
            "job": job,
            "source_commit": self.commit,
            "state": "COLLECTION_FAILED_PRESERVED",
            "errors": [{"cell_id": cid, "classification": "resource_cleanup_failed"}],
            **child_key,
        }
        self.attempt_id = "TEST_ONLY_attempt_" + job
        entry = {
            "status": "cleanup_failed",
            "attempt_id": self.attempt_id,
            "artifact_dir": str(self.cell.resolve()),
            "cleanup_error": "TEST_ONLY squeue timeout",
            "collector_phase_seconds": dict.fromkeys(
                ["render_s", "schedule_s", "stage_s", "execute_wall_s", "collect_s"], 1
            ),
        }
        checkpoint = {"plan_sha256": self.plan_sha, "cells": {cid: entry}}
        owner = {"job_id": job, "step_name": self.step}
        self.paths = {
            "started": root / self.directory / "started.json",
            "final": root / self.directory / "result.json",
            "checkpoint": root / self.directory / "checkpoint/fpm_forward.json",
            "plan": root / "prepared/plans" / (cid + ".json"),
            "owner": self.cell.parent / ".slurm-owners" / (self.step + ".json"),
            "failure_review": root / "reviews" / (job + ".json"),
        }
        for key, value in [
            ("started", start),
            ("final", final),
            ("checkpoint", checkpoint),
            ("plan", self.plan),
            ("owner", owner),
        ]:
            self.put(self.paths[key], value)
        self.log = self.cell / "logs/native.log"
        self.log.parent.mkdir(parents=True)
        self.log.write_text("TEST_ONLY native completed; transport cleanup timeout\n")
        prov = {"attempt_id": self.attempt_id, "cell_id": cid, "plan_sha256": self.plan_sha}
        self.put(self.cell / "raw/node0000/collector-provenance.json", prov)
        review = {
            "contract": c.REVIEW_CONTRACT,
            "job": job,
            "attempt_id": self.attempt_id,
            "outcome": "SOLE_POST_COLLECTION_CLEANUP_FAILURE",
            "native_failures": [],
            "unresolved_findings": [],
            "original_final_sha256": ref(root, self.paths["final"])["sha256"],
            "original_checkpoint_sha256": ref(root, self.paths["checkpoint"])["sha256"],
            "logs": {str(self.log.relative_to(root)): {k: ref(root, self.log)[k] for k in ("sha256", "bytes")}},
            "TEST_ONLY": True,
        }
        self.put(self.paths["failure_review"], review)
        self.rows = [
            {
                **point,
                **{
                    "cell_id": cid,
                    "collector_attempt_id": self.attempt_id,
                    "source_plan_sha256": self.plan_sha,
                    "warmup_repeats": 5,
                    "measurement_repeats": 10,
                    "runtime_run_id": "TEST_ONLY_run",
                    "runtime_grid_digest": "a" * 64,
                },
            }
            for point in self.points
        ]
        self.consumer = {"TEST_ONLY": True, "native": "not_executed"}
        self.events = []

    @staticmethod
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + "\n")

    def original(self):
        request = {
            "task_root": str(self.root),
            "attempt_directory": str(self.directory),
            "references": {k: ref(self.root, p) for k, p in self.paths.items()},
            "host_source": {
                "commit": self.commit,
                "runner_sha256": c.HOST_SOURCES[self.commit],
                "slurm_sha256": c.SLURM_SOURCE,
            },
        }
        return executor.load_original(request)

    def cleanup(self, identity, output):
        self.events.append("cleanup")
        return {
            "canonical_cell_directory": str(self.cell.resolve()),
            "owner_sha256": identity["original_owner_sha256"],
            "job_id": self.job,
            "step_name": self.step,
            "outcome": "OWNED_STEPS_ABSENT",
            "commands": [
                {
                    "argv": ["squeue", "--steps", "--me", "--noheader", "--format=%i|%j"],
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                }
                for _ in range(2)
            ],
        }

    def native(self, identity, output):
        self.events.append("native")
        executor.write(output / "aggregated-rows.json", self.rows)
        return {
            "method": "PUBLIC_COMPLETE_EXACT_ATTEMPT_AGGREGATE_CELL",
            "attempt_id": self.attempt_id,
            "plan_sha256": self.plan_sha,
            "raw_root": identity["raw_root"],
            "rows_sha256": c.digest(c.canonical(self.rows)),
            "point_count": len(self.rows),
            "consumer_identity": self.consumer,
            "consumer_identity_sha256": c.digest(c.canonical(self.consumer)),
        }

    def scan(self, identity):
        self.events.append("inventory")
        return executor.inventory(identity)

    def run(self, output, **callbacks):
        return executor._reconcile(
            self.original(),
            output,
            cleanup=callbacks.get("cleanup", self.cleanup),
            native=callbacks.get("native", self.native),
            scan=callbacks.get("scan", self.scan),
        )


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="TEST_ONLY_cleanup_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.f = Fixture(self.root)
        self.out = self.root / "new-reconciliation"

    def test_original_failed_status_retained_and_cleanup_precedes_every_raw_read(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        proof = self.f.run(self.out)
        self.assertEqual(self.f.events, ["cleanup", "inventory", "native", "inventory"])
        self.assertEqual(c.verify(proof)["entry"]["status"], "cleanup_failed")
        self.assertTrue(all(Path(p).read_bytes() == raw for p, raw in before.items()))
        self.assertFalse((self.out / "failure.json").exists())

    def test_failed_cleanup_never_reads_raw_or_changes_original(self):
        def failed(*_):
            raise TimeoutError("TEST_ONLY squeue unknown")

        with self.assertRaises(TimeoutError):
            self.f.run(self.out, cleanup=failed)
        self.assertEqual(self.f.events, [])
        self.assertTrue((self.out / "failure.json").is_file())
        self.assertFalse((self.out / "receipt.json").exists())

    def test_partial_strict_rows_reject_after_teardown(self):
        self.f.rows.pop()
        with self.assertRaisesRegex(ValueError, "whole child"):
            self.f.run(self.out)
        self.assertFalse((self.out / "receipt.json").exists())

    def test_native_validation_error_preserved_without_success_receipt(self):
        def failed(*_):
            raise ValueError("TEST_ONLY native invalid timing")

        with self.assertRaisesRegex(ValueError, "native invalid"):
            self.f.run(self.out, native=failed)
        self.assertEqual(self.f.events, ["cleanup", "inventory"])
        self.assertTrue((self.out / "failure.json").is_file())

    def test_original_changed_during_strict_read_rejects(self):
        def mutate(identity, output):
            result = self.f.native(identity, output)
            self.f.log.write_text("TEST_ONLY changed")
            return result

        with self.assertRaisesRegex(ValueError, "artifacts changed"):
            self.f.run(self.out, native=mutate)

    def test_external_plan_changed_during_strict_read_rejects(self):
        def mutate(identity, output):
            result = self.f.native(identity, output)
            self.f.paths["plan"].write_text("{}")
            return result

        with self.assertRaisesRegex(ValueError, "request member changed"):
            self.f.run(self.out, native=mutate)

    def test_primary_failure_and_incomplete_phase_reject_before_cleanup(self):
        for mutation in ["error", "phase"]:
            original = self.f.original()
            cp = c.decode(original["documents"]["checkpoint"])
            if mutation == "error":
                cp["cells"][self.f.cid]["error"] = "TEST_ONLY native failure"
            else:
                del cp["cells"][self.f.cid]["collector_phase_seconds"]["collect_s"]
            raw = c.canonical(cp)
            original["documents"]["checkpoint"] = c.embedded(raw)
            original["references"]["checkpoint"].update(sha256=c.digest(raw), bytes=len(raw))
            with self.assertRaises(ValueError):
                executor._reconcile(original, self.out, cleanup=self.f.cleanup, native=self.f.native, scan=self.f.scan)
        self.assertEqual(self.f.events, [])

    def test_624031_watchdog_review_explicitly_rejects(self):
        f = Fixture(self.root / "watchdog", backend="sglang", job="624031")
        review = json.loads(f.paths["failure_review"].read_bytes())
        review["native_failures"] = ["Original native watchdog terminated worker after final batch"]
        f.put(f.paths["failure_review"], review)
        with self.assertRaisesRegex(ValueError, "watchdog"):
            f.run(self.out)
        self.assertEqual(f.events, [])

    def test_unresolved_native_severity_requires_explicit_disposition(self):
        value = json.loads(self.f.paths["failure_review"].read_bytes())
        value["unresolved_findings"] = ["TEST_ONLY monitor_workers ERROR with unknown lifecycle"]
        self.f.put(self.f.paths["failure_review"], value)
        with self.assertRaisesRegex(ValueError, "unknown failure"):
            self.f.run(self.out)

    def test_internal_member_symlink_rejects(self):
        (self.f.cell / "raw/aliased").symlink_to(self.f.log)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.f.run(self.out)

    def test_output_cannot_mutate_original_attempt(self):
        with self.assertRaisesRegex(ValueError, "diagnostics overlap"):
            self.f.run(self.f.cell / "new")

    def test_wrong_owner_other_job_and_wrong_path_reject(self):
        proof = self.f.run(self.out)
        for field, value in [("job_id", "999"), ("step_name", "wrong")]:
            changed = copy.deepcopy(proof)
            changed["cleanup"][field] = value
            with self.assertRaises(ValueError):
                c.verify(changed)
        changed = copy.deepcopy(proof)
        changed["original"]["references"]["owner"]["path"] = "somewhere/owner.json"
        with self.assertRaisesRegex(ValueError, "owner receipt path"):
            c.verify(changed)

    def test_unrelated_cancellation_or_unproven_empty_query_reject(self):
        proof = self.f.run(self.out)
        cases = [
            [{"argv": ["scancel", "999.0"], "returncode": 0, "stdout": "", "stderr": ""}],
            [
                {
                    "argv": ["squeue", "--steps", "--me", "--noheader", "--format=%i|%j"],
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "timeout",
                }
            ],
        ]
        for commands in cases:
            changed = copy.deepcopy(proof)
            changed["cleanup"]["commands"] = commands
            with self.assertRaises(ValueError):
                c.verify(changed)
        changed = copy.deepcopy(proof)
        changed["cleanup"]["commands"][-1]["stdout"] = self.f.job + ".0|" + self.f.step + "\n"
        with self.assertRaisesRegex(ValueError, "steps remain"):
            c.verify(changed)

    def test_owned_observed_step_can_be_cancelled_only_between_queries(self):
        proof = self.f.run(self.out)
        query = proof["cleanup"]["commands"][0]
        query["stdout"] = self.f.job + ".0|" + self.f.step + "\n999.0|" + self.f.step + "\n"
        proof["cleanup"]["commands"].insert(
            1, {"argv": ["scancel", self.f.job + ".0"], "returncode": 0, "stdout": "", "stderr": ""}
        )
        c.verify(proof)
        proof["cleanup"]["commands"][1]["argv"][1] = "999.0"
        with self.assertRaisesRegex(ValueError, "unrelated"):
            c.verify(proof)

    def test_whole_original_inventory_and_source_tamper_reject(self):
        proof = self.f.run(self.out)
        changed = copy.deepcopy(proof)
        changed["artifact_inventory"].pop(str(self.f.log.relative_to(self.root)))
        with self.assertRaisesRegex(ValueError, "logs"):
            c.verify(changed)
        changed = copy.deepcopy(proof)
        changed["source"]["executor_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source closure"):
            c.verify(changed)

    def test_sg_native_source_identity_stays_distinct_from_host(self):
        f = Fixture(self.root / "sg", backend="sglang", job="2002")
        proof = f.run(self.out)
        self.assertEqual(c.verify(proof)["backend"], "sglang")
        changed = copy.deepcopy(proof)
        changed["original"]["host_source"]["commit"] = next(iter(c.HOST_SOURCES))
        with self.assertRaisesRegex(ValueError, "host source"):
            c.verify(changed)

    def test_no_blanket_legacy_status_allowlist(self):
        proof = self.f.run(self.out)
        original = proof["original"]
        attempt = {
            "job": self.f.job,
            "cell_id": self.f.cid,
            "terminal_state": "COLLECTION_FAILED_PRESERVED",
            "original_attempt_directory": str(self.f.directory),
            **{k: original["references"][k] for k in ("started", "final", "checkpoint")},
        }
        choice = {"job": self.f.job, "reconciliation": ref(self.root, self.out / "receipt.json")}
        ledger = {"schema": c.LEGACY_SELECTION_SCHEMA, "selections": {self.f.cid: choice}}
        with self.assertRaisesRegex(ValueError, "legacy"):
            c.selected_proof(ledger, choice, attempt, lambda _: proof)
        ledger["schema"] = c.SELECTION_SCHEMA
        self.assertIs(c.selected_proof(ledger, choice, attempt, lambda _: proof), proof)
        attempt["checkpoint"] = {**attempt["checkpoint"], "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "another original"):
            c.selected_proof(ledger, choice, attempt, lambda _: proof)


class StorageBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="TEST_ONLY_recovery_alias_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.physical = self.root / "physical"
        self.f = Fixture(self.physical)
        (self.root / "middle/users").mkdir(parents=True)
        (self.root / "lustre").symlink_to(self.root / "middle", target_is_directory=True)
        self.alias = self.root / "middle/users/harrli"
        self.alias.symlink_to(self.physical, target_is_directory=True)
        self.lexical = self.root / "lustre/users/harrli"
        self.original = self.f.original()
        self.original["task_root"] = str(self.lexical)

    def run_case(self, native=None):
        return executor._reconcile(
            self.original,
            self.root / "output",
            cleanup=self.f.cleanup,
            native=native or self.f.native,
            scan=self.f.scan,
        )

    def test_real_two_ancestor_aliases_bind_original_lexical_and_samefile_storage(self):
        proof = self.run_case()
        self.assertEqual(proof["storage_binding"]["canonical_task_root"], str(self.physical))
        self.assertEqual(proof["original"]["task_root"], str(self.lexical))
        self.assertEqual(proof["storage_binding"]["recorded_cell"], str(self.f.cell))
        self.assertEqual(c.verify(proof)["entry"]["status"], "cleanup_failed")

    def test_root_retarget_after_strict_read_rejects_even_if_member_bytes_equal(self):
        shadow = self.root / "shadow"
        shutil.copytree(self.physical, shadow)

        def native(identity, output):
            result = self.f.native(identity, output)
            self.alias.unlink()
            self.alias.symlink_to(shadow, target_is_directory=True)
            return result

        with self.assertRaisesRegex(ValueError, "storage differs|retargeted"):
            self.run_case(native)

    def test_original_relative_reference_cannot_escape_root(self):
        self.original["references"]["plan"]["path"] = "../outside/plan.json"
        with self.assertRaisesRegex(ValueError, "unsafe relative"):
            self.run_case()
        self.assertEqual(self.f.events, [])
