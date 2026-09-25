# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY factory histories; real tar/bind/portable paths, no acceptance claim."""

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from tools.glm53flash_hf import closed_history as h
from tools.glm53flash_hf import external_control_sglang_factory as factory
from tools.glm53flash_hf import portable_history as portable
from tools.glm53flash_hf import raw_archive as archive

pytestmark = pytest.mark.unit


def rewrite_inputs(f):
    request, ledger = f.requests["sglang"], f.ledgers["sglang"]
    ledger["request_sha256"] = h.digest(request)
    for key, value in [("request", request), ("ledger", ledger)]:
        path = Path(f.inputs["sglang"][key]["path"])
        path.write_bytes(h.canonical(value))
        f.inputs["sglang"][key] = h.reference(path)


def add_floor(f):
    root = f.source / "pre-native"
    originals = root / "originals"
    originals.mkdir(parents=True)
    jobs = ["9001", "9002", "9003", "9004"]
    files = {}
    for job in jobs:
        for suffix in (".out", ".err"):
            name = "logs/sg2d51-formal-" + job + suffix
            path = originals / name
            path.parent.mkdir(exist_ok=True)
            raw = b"TEST_ONLY failure log\x00\xff\xfe\n"
            path.write_bytes(raw)
            files[name] = {"bytes": len(raw), "sha256": h.sha(raw)}
    accounting = {
        "returncode": 0,
        "stdout": "".join(job + "|TEST_ONLY_cluster|name|user|FAILED|0|0|1:0|\n" for job in jobs),
    }
    inventory = {
        "files": files,
        "accounting": accounting,
        "output_directories": {deployment: [] for deployment in h.DEPLOYMENTS},
    }
    refs = {}
    for name, value in [("accounting", accounting), ("inventory", inventory)]:
        path = root / (name + ".json")
        path.write_bytes(h.canonical(value))
        refs[name] = {"path": str(path.relative_to(f.source)), "sha256": h.sha(path.read_bytes())}
    receipt = {
        "state": "ORIGINAL_PRE_NATIVE_SHARED_STORAGE_GATE_FAILURES_PRESERVED",
        "native_started": False,
        "originals_rewritten": False,
        "output_directories": inventory["output_directories"],
        "inventory_sha256": refs["inventory"]["sha256"],
        "accounting_sha256": refs["accounting"]["sha256"],
        "files": len(files),
        "bytes": sum(v["bytes"] for v in files.values()),
    }
    path = root / "receipt.json"
    path.write_bytes(h.canonical(receipt))
    refs["receipt"] = {"path": str(path.relative_to(f.source)), "sha256": h.sha(path.read_bytes())}
    floor = {
        **refs,
        "jobs": jobs,
        "originals_root": "pre-native/originals",
        "native_started": False,
        "original_started_records": None,
    }
    f.requests["sglang"].update(factory_history_scope=h.FACTORY_HISTORY_SCOPE, pre_native_failure_floor=floor)
    return floor


@pytest.fixture
def factory_tree():
    spec = importlib.util.spec_from_file_location(
        "original_history_fixture", Path(__file__).with_name("test_glm53flash_closed_history.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    f = module.ClosureTests(methodName="runTest")
    f.setUp()
    try:
        for index, attempt in enumerate(f.ledgers["sglang"]["attempts"]):
            old = Path(attempt["original_attempt_directory"])
            new = old.parent / (str(index % 18) + "-" + attempt["cell_id"]) / old.name
            (f.source / new.parent).mkdir(exist_ok=True)
            (f.source / old).rename(f.source / new)
            attempt["original_attempt_directory"] = str(new)
            selected = {
                "kind": "formal",
                "child_cell_id": attempt["cell_id"],
                "child_plan_sha256": "b" * 64,
                "new_identity": {
                    "child_cell_id": attempt["cell_id"],
                    "child_plan_sha256": "b" * 64,
                    "deployment": attempt["deployment"],
                },
                "original_identity": {"child_cell_id": "historical-" + attempt["cell_id"]},
            }
            start = dict.fromkeys(factory.STARTED_FIELDS, "TEST_ONLY")
            start.update(
                schema=factory.STARTED,
                state="RUNNING",
                job=attempt["job"],
                deployment=attempt["deployment"],
                selected=selected,
            )
            final = {**start, "schema": factory.FINAL, "state": attempt["terminal_state"]}
            for label in ("started", "final", "checkpoint"):
                previous = attempt[label]["path"]
                path = f.source / new / Path(previous).relative_to(old)
                if label in ("started", "final"):
                    path.write_bytes(h.canonical(start if label == "started" else final))
                ref = h.reference(path)
                ref["path"] = str(path.relative_to(f.source))
                attempt[label] = ref
            for job in f.plan["jobs"]:
                job["accepted_raw_roots"] = [
                    str(f.source / new / Path(p).relative_to(f.source / old))
                    if Path(p).is_relative_to(f.source / old)
                    else p
                    for p in job["accepted_raw_roots"]
                ]
        f.requests["sglang"]["external_control_request"].update(schema=factory.SCHEMA, adapter=factory.ADAPTER)
        failed = f.ledgers["sglang"]["attempts"][-1]
        f.requests["sglang"]["history_floor"] = [{k: failed[k] for k in ("started", "final", "checkpoint")}]
        add_floor(f)
        rewrite_inputs(f)
        yield f
    finally:
        f.doCleanups()


def test_factory_archive_preserves_binary_startup_logs_outside_native_tar(factory_tree):
    f = factory_tree
    bundle, proof = f.make_archive("sglang")
    original = copy.deepcopy(proof["snapshot"])
    evidence = original["pre_native_failure_evidence"]
    assert len(original["ledger"]["attempts"]) == 73
    assert evidence["storage"] == "ORIGINAL_BYTES_IN_HISTORY_SIDECAR_NOT_NATIVE_TAR"
    assert len(evidence["originals"]) == 11
    assert all(
        not row["path"].startswith("pre-native") for row in archive.inventory_records(bundle / archive.INVENTORY)
    )
    shutil.rmtree(f.source)
    assert h.verify_bundle_history(bundle, proof, archive) == original


@pytest.mark.parametrize(
    "defect",
    [
        "missing_scope",
        "null_scope",
        "unknown_scope",
        "wrong_backend",
        "missing_floor",
        "missing_file",
        "changed_file",
        "symlink",
        "unbounded",
        "native_record",
    ],
)
def test_factory_scope_and_startup_closure_reject_before_archive(factory_tree, defect):
    f = factory_tree
    request = f.requests["sglang"]
    floor = request["pre_native_failure_floor"]
    path = f.source / floor["originals_root"] / "logs/sg2d51-formal-9001.err"
    if defect == "missing_scope":
        request.pop("factory_history_scope")
    elif defect == "null_scope":
        request["factory_history_scope"] = None
    elif defect == "unknown_scope":
        request["factory_history_scope"] = "unknown"
    elif defect == "wrong_backend":
        request["external_control_request"]["adapter"] = "legacy"
    elif defect == "missing_floor":
        request.pop("pre_native_failure_floor")
    elif defect == "missing_file":
        path.unlink()
    elif defect == "changed_file":
        path.write_bytes(b"changed")
    elif defect == "symlink":
        path.unlink()
        path.symlink_to(f.source / floor["originals_root"] / "logs/sg2d51-formal-9001.out")
    elif defect == "native_record":
        floor["original_started_records"] = ["invented"]
    else:
        floor["jobs"] = ["9001"] * 513
    rewrite_inputs(f)
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        f.make_archive("sglang")
    assert not (f.root / "bundle-sglang").exists()


def test_factory_portable_keeps_supplement_without_live_tree_or_tar(factory_tree):
    f = factory_tree
    bundles = f.both_bundles()
    bound, proof = f.bind(bundles)
    ref = h.reference(bound / h.BOUND_PROOF)
    ref["path"] = h.BOUND_PROOF
    out = f.root / "portable"
    entry = portable.prepare_portable_history(
        bound, ref, bundles, out, expected_stage_sha256=f.plan["stage_sha256"], archive=archive
    )
    records = json.loads((bound / "external-raw-evidence.json").read_bytes())
    for path in set(bundles.values()):
        shutil.rmtree(path)
    shutil.rmtree(f.source)
    shutil.rmtree(bound)
    result = portable.verify_portable_history(
        out, entry, records, expected_stage_sha256=f.plan["stage_sha256"], archive=archive
    )
    assert result["state"] == "PORTABLE_METADATA_HISTORY_PASS_NO_FRESH_TAR_VERIFICATION"
    raw = json.loads((out / portable.PROOF).read_bytes())
    history_path = out / raw["bound_history"]["path"]
    history = json.loads(history_path.read_bytes())
    snapshot = next(v["snapshot"] for v in history["bundle_proofs"].values() if v["snapshot"]["backend"] == "sglang")
    assert len(snapshot["pre_native_failure_evidence"]["originals"]) == 11


def test_rebound_extra_or_missing_supplement_fails_offline(factory_tree):
    f = factory_tree
    bundle, proof = f.make_archive("sglang")
    original = copy.deepcopy(proof)
    for mutate in ("missing", "extra", "mismatch"):
        changed = copy.deepcopy(original)
        snap = changed["snapshot"]
        records = snap["pre_native_failure_evidence"]["originals"]
        if mutate == "missing":
            records.pop(next(iter(records)))
        elif mutate == "extra":
            records["orphan"] = h._record(b"extra")
        else:
            snap["factory_history_scope"] = "legacy"
        with pytest.raises((ValueError, KeyError)):
            h.verify_bundle_history(bundle, changed, archive)


def rebind_floor(f, mutate):
    floor = f.requests["sglang"]["pre_native_failure_floor"]
    docs = {
        key: json.loads((f.source / floor[key]["path"]).read_bytes()) for key in ("receipt", "inventory", "accounting")
    }
    mutate(docs)
    docs["inventory"]["accounting"] = docs["accounting"]
    for key in ("accounting", "inventory"):
        path = f.source / floor[key]["path"]
        path.write_bytes(h.canonical(docs[key]))
        floor[key]["sha256"] = h.sha(path.read_bytes())
        docs["receipt"][key + "_sha256"] = floor[key]["sha256"]
    path = f.source / floor["receipt"]["path"]
    path.write_bytes(h.canonical(docs["receipt"]))
    floor["receipt"]["sha256"] = h.sha(path.read_bytes())
    rewrite_inputs(f)


@pytest.mark.parametrize(
    "defect", ["accounting_pass", "accounting_missing", "log_missing", "over_limit", "native_output"]
)
def test_rebound_supplement_semantics_reject(factory_tree, defect):
    f = factory_tree

    def mutate(docs):
        if defect == "accounting_pass":
            docs["accounting"]["stdout"] = docs["accounting"]["stdout"].replace("|FAILED|", "|COMPLETED|")
        elif defect == "accounting_missing":
            docs["accounting"]["stdout"] = "\n".join(docs["accounting"]["stdout"].splitlines()[:-1])
        elif defect == "log_missing":
            docs["inventory"]["files"].pop("logs/sg2d51-formal-9001.err")
            docs["receipt"]["files"] -= 1
            docs["receipt"]["bytes"] = sum(v["bytes"] for v in docs["inventory"]["files"].values())
        elif defect == "over_limit":
            docs["receipt"]["bytes"] = h.PRE_NATIVE_MAX_BYTES + 1
        else:
            docs["receipt"]["output_directories"]["fp8-tp2"] = ["native"]
            docs["inventory"]["output_directories"] = docs["receipt"]["output_directories"]

    rebind_floor(f, mutate)
    with pytest.raises(ValueError):
        f.make_archive("sglang")


def test_changed_pre_native_original_during_tar_is_preserved_failure(factory_tree):
    from unittest.mock import patch

    from tools.glm53flash_hf import raw_campaign as campaign

    f = factory_tree

    def operation(*args):
        value = f.archive_one(*args)
        (f.source / "pre-native/originals/logs/sg2d51-formal-9001.err").write_bytes(b"changed after archive")
        return value

    with (
        patch.object(campaign, "archive_one", side_effect=operation),
        pytest.raises(ValueError, match="original bytes changed"),
    ):
        f.make_archive("sglang")
    out = f.root / "bundle-sglang"
    assert (out / archive.ARCHIVE).exists() and (out / h.FAILURE).exists() and not (out / h.BUNDLE_PROOF).exists()
