# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original attempt history with an explicit cleanup-reconciliation extension.

Candidate based on the committed portable-history closure. Ordinary legacy
choices still require original success. New v2 choices preserve failed original
states and bind separate teardown/native-read proof to the complete inventory.
This coherent source candidate is not identified as unchanged base production.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

if __package__:
    from . import cleanup_reconciliation as reconciliation
    from . import external_control_sglang_factory as factory
    from . import native_roots
else:
    import cleanup_reconciliation as reconciliation
    import external_control_sglang_factory as factory
    import native_roots

CONTRACT = "fpm_closed_attempt_history_v2"
BUNDLE_PROOF = "closed-attempt-history.json"
BOUND_PROOF = "closed-attempt-history-binding.json"
FAILURE = "closed-attempt-history-failure.json"
FACTORY_HISTORY_SCOPE = "sglang_public_factory_depth4_original_history_v1"
PRE_NATIVE_MAX_FILES = 512
PRE_NATIVE_MAX_BYTES = 8 * 1024**2
TERMINAL = {
    "vllm": {"COLLECTION_PASSED", "COLLECTION_FAILED_PRESERVED"},
    "sglang": {"COLLECTION_PASSED", "COLLECTION_FAILED_PRESERVED", "FAILED_PRESERVED"},
}
DEPLOYMENTS = {f"{q}-tp{t}" for q in ("fp8", "nvfp4") for t in (2, 4)}
MAINTENANCE_IDENTITY = {
    "kind": "public_source_review_followup",
    "profile": "fpm_sglang_public_factory_history_v5",
    "base_commit": "0346c808885baa366bfcdcdb96add32dd4a94a8e",
}
MAINTENANCE = {
    "accounting_termination.py": "67cffe873c0f4225fd07f8786970c27e3ced8ac63366b38540443f4aa8efc890",
    "cleanup_executor.py": "0cf0847469319613b6b8ccab53a43905b079fe3a3135466680c2ecbe19f4bd90",
    "cleanup_reconciliation.py": "dda17105f69078f8f7e2d35cb9a28133c3da3226d4d2bba27e93cf0c74e6ebff",
    "external_control.py": "2d3799c0e04030df72728a9ad2333f4220a09a999b16a48ff43730a409874a23",
    "external_control_current.py": "c84aeda99141400a1dfc478c6249badfa25a7bfcdd9b478686697fba92fbc85e",
    "external_control_sglang_factory.py": "36be308ea67d3999d4f17a2dafb9cd05026ed01543c8c3c5e070b7955160c07e",
    "external_control_sglang_mixed.py": "5d78d422876ae086504020aa4b70deaa3d1732de3ba000da5d04595a0fba9971",
    "external_control_vllm.py": "296130a6a8e31412bf1c0244aa20665fa35dc53bbefceab4bb9b6bebbeab40ae",
    "glm53flash.py": "30c784be0c57120910fbeeb4200971c1468400b6296a1cc688aa4545083b8dbc",
    "import_glm53flash.py": "19a10c03e16cfd465f35a6e58346ebdbcd2fc911dbae2eb1c744c5cd500ccc9f",
    "native_roots.py": "6a029c2353ab0d0da556b3bf407ac55831d4fbab69051e8006225dd3edefeb09",
    "portable_history.py": "752a7483c3f92c77613ba370ce8b7a894e836f7d79cb34ff332991ef3887ec57",
    "profile.py": "f806a78c58edc52b7ae62b1f0d49dc49bd4f8a0aa016b04197a12c03d53447b0",
    "raw_archive.py": "80384152174e45c849623b7f299e0ab17c3d18b93624df220afce651d3b63b40",
    "raw_campaign.py": "592a871aaaf6bf692b16d25a30582a3d95ae60df51d82c0b70f798a5bd580e3b",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(canonical(value))


def relative(value):
    path = Path(value)
    require(
        isinstance(value, str) and value and not path.is_absolute() and ".." not in path.parts,
        "unsafe path",
    )
    return path


def checked_file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular original file required")
    return path


def reference(path):
    path = checked_file(path)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": sha(raw), "bytes": len(raw)}


def bound(ref):
    raw = checked_file(ref["path"]).read_bytes()
    require(sha(raw) == ref["sha256"] and len(raw) == ref["bytes"], "history input changed")
    return json.loads(raw)


def write(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def dependencies(archive, campaign=None):
    root = Path(archive.__file__).resolve().parent
    for name, expected in MAINTENANCE.items():
        require(
            sha((root / name).read_bytes()) == expected,
            "archive maintenance source differs",
        )
    if campaign is not None:
        require(
            Path(campaign.__file__).resolve() == root / "raw_campaign.py",
            "mixed archive maintenance",
        )
        require(campaign.archive is archive, "mixed raw archive import")


def _record(raw):
    return {
        "sha256": sha(raw),
        "bytes": len(raw),
        "base64": base64.b64encode(raw).decode(),
    }


def _decoded(record):
    raw = base64.b64decode(record["base64"], validate=True)
    require(
        sha(raw) == record["sha256"] and len(raw) == record["bytes"],
        "original history bytes changed",
    )
    return json.loads(raw)


def history_depth(snapshot):
    """Select a path contract explicitly; never infer it from a directory shape."""
    request = snapshot["request"]
    scope = request.get("factory_history_scope")
    require(snapshot.get("factory_history_scope") == scope, "factory history request/snapshot scope differs")
    if scope is None:
        require(
            "factory_history_scope" not in request
            and "factory_history_scope" not in snapshot
            and "pre_native_failure_floor" not in request
            and "pre_native_failure_evidence" not in snapshot,
            "pre-native evidence requires explicit factory history scope",
        )
        return 4 if snapshot["backend"] == "vllm" else 3
    require(
        scope == FACTORY_HISTORY_SCOPE
        and snapshot["backend"] == "sglang"
        and request["external_control_request"].get("schema") == factory.SCHEMA
        and request["external_control_request"].get("adapter") == factory.ADAPTER,
        "unknown or cross-backend factory history scope",
    )
    require("pre_native_failure_floor" in request, "complete pre-native failure floor required")
    return 4


def _pre_native_relative(value):
    require(
        isinstance(value, str) and value and "\\" not in value and "\x00" not in value,
        "unsafe pre-native original path",
    )
    path = relative(value)
    require(path.as_posix() == value and value != ".", "noncanonical pre-native original path")
    return path


def _pre_native_floor(request):
    floor = request["pre_native_failure_floor"]
    require(
        isinstance(floor, dict)
        and set(floor)
        == {
            "receipt",
            "inventory",
            "accounting",
            "jobs",
            "originals_root",
            "native_started",
            "original_started_records",
        },
        "unknown pre-native failure contract",
    )
    require(
        floor["native_started"] is False and floor["original_started_records"] is None,
        "pre-native failure cannot invent native records",
    )
    jobs = floor["jobs"]
    require(
        isinstance(jobs, list)
        and jobs
        and all(isinstance(j, str) and j.isdecimal() for j in jobs)
        and len(set(jobs)) == len(jobs),
        "invalid pre-native job set",
    )
    for name in ("receipt", "inventory", "accounting"):
        ref = floor[name]
        require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "invalid pre-native original reference")
        path = _pre_native_relative(ref["path"])
        require(
            all(not path.is_relative_to(relative(p)) for p in request["campaign_roots"]),
            "pre-native metadata must be separate from native campaign",
        )
        require(
            isinstance(ref["sha256"], str)
            and len(ref["sha256"]) == 64
            and set(ref["sha256"]) <= set("0123456789abcdef"),
            "invalid pre-native original SHA",
        )
    root = _pre_native_relative(floor["originals_root"])
    require(
        all(not root.is_relative_to(relative(p)) for p in request["campaign_roots"]),
        "pre-native supplement must be separate from native campaign",
    )
    return floor


def _pre_native_members(request, get):
    """Check exact original bytes, whether supplied live or by an offline proof."""
    floor = _pre_native_floor(request)
    originals = {}

    def load(name, expected, size=None):
        # Reuse archive path semantics, including '.', backslash and traversal rejection.
        if __package__:
            from . import raw_archive as archive
        else:
            import raw_archive as archive

        archive.relative_parts(name)
        raw = get(name)
        require(len(raw) <= PRE_NATIVE_MAX_BYTES, "pre-native original exceeds metadata bound")
        require(sha(raw) == expected and (size is None or len(raw) == size), "pre-native original bytes changed")
        require(name not in originals, "pre-native original paths overlap")
        originals[name] = _record(raw)
        return raw

    docs = {
        name: json.loads(load(floor[name]["path"], floor[name]["sha256"]))
        for name in ("receipt", "inventory", "accounting")
    }
    receipt, inventory, accounting = (docs[name] for name in ("receipt", "inventory", "accounting"))
    require(
        receipt["state"] == "ORIGINAL_PRE_NATIVE_SHARED_STORAGE_GATE_FAILURES_PRESERVED"
        and receipt["native_started"] is False
        and receipt["originals_rewritten"] is False
        and set(receipt["output_directories"]) == DEPLOYMENTS
        and all(value == [] for value in receipt["output_directories"].values())
        and inventory["output_directories"] == receipt["output_directories"],
        "pre-native failure closure relabeled or native output present",
    )
    require(
        receipt["inventory_sha256"] == floor["inventory"]["sha256"]
        and receipt["accounting_sha256"] == floor["accounting"]["sha256"]
        and inventory["accounting"] == accounting
        and accounting["returncode"] == 0,
        "pre-native accounting binding differs",
    )
    parents = [line.split("|") for line in accounting["stdout"].splitlines() if line.split("|", 1)[0].isdecimal()]
    require(
        len(parents) == len(floor["jobs"])
        and {row[0] for row in parents} == set(floor["jobs"])
        and all(len(row) >= 8 and row[1] and row[4] == "FAILED" and row[7] != "0:0" for row in parents)
        and len({row[1] for row in parents}) == 1,
        "pre-native failed scheduler parents differ",
    )
    files = inventory["files"]
    require(
        files and len(files) == receipt["files"] <= PRE_NATIVE_MAX_FILES,
        "incomplete or unbounded pre-native original membership",
    )
    require(
        type(receipt["bytes"]) is int and 0 <= receipt["bytes"] <= PRE_NATIVE_MAX_BYTES,
        "pre-native closure exceeds metadata bound",
    )
    require(
        all(type(item.get("bytes")) is int and item["bytes"] >= 0 for item in files.values())
        and sum(item["bytes"] for item in files.values()) == receipt["bytes"],
        "pre-native declared byte count differs",
    )
    require(
        all("logs/sg2d51-formal-" + job + suffix in files for job in floor["jobs"] for suffix in (".out", ".err")),
        "pre-native original stdout/stderr absent",
    )
    for name, ref in files.items():
        require(Path(name).name not in {"started.json", "result.json"}, "pre-native closure contains native records")
        require(type(ref["bytes"]) is int and ref["bytes"] >= 0, "invalid pre-native size")
        load(
            str(_pre_native_relative(floor["originals_root"]) / _pre_native_relative(name)), ref["sha256"], ref["bytes"]
        )
    require(sum(item["bytes"] for item in files.values()) == receipt["bytes"], "pre-native byte count differs")
    return originals


def verify_pre_native_history(snapshot):
    if snapshot["request"].get("factory_history_scope") is None:
        return
    evidence = snapshot["pre_native_failure_evidence"]
    require(
        set(evidence) == {"scope", "storage", "originals"}
        and evidence["scope"] == FACTORY_HISTORY_SCOPE
        and evidence["storage"] == "ORIGINAL_BYTES_IN_HISTORY_SIDECAR_NOT_NATIVE_TAR",
        "unknown pre-native evidence storage contract",
    )
    require(
        isinstance(evidence["originals"], dict)
        and len(evidence["originals"]) <= PRE_NATIVE_MAX_FILES + 3
        and all(type(r.get("bytes")) is int and r["bytes"] >= 0 for r in evidence["originals"].values())
        and sum(r["bytes"] for r in evidence["originals"].values()) <= 2 * PRE_NATIVE_MAX_BYTES,
        "pre-native embedded closure exceeds metadata bound",
    )

    def get(name):
        record = evidence["originals"][name]
        require(
            type(record["bytes"]) is int
            and 0 <= record["bytes"] <= PRE_NATIVE_MAX_BYTES
            and len(record["base64"]) <= 4 * ((PRE_NATIVE_MAX_BYTES + 2) // 3),
            "pre-native embedded bytes exceed metadata bound",
        )
        raw = base64.b64decode(record["base64"], validate=True)
        require(sha(raw) == record["sha256"] and len(raw) == record["bytes"], "pre-native original proof changed")
        return raw

    checked = _pre_native_members(snapshot["request"], get)
    require(checked == evidence["originals"], "missing or extra pre-native original bytes")
    jobs = set(snapshot["request"]["pre_native_failure_floor"]["jobs"])
    require(
        not jobs.intersection(a["job"] for a in snapshot["ledger"]["attempts"]),
        "pre-native job was relabeled as a native attempt",
    )


def _validate_snapshot(snapshot):
    """Pure offline checks against the preserved original small-file bytes."""
    require(snapshot["backend"] in {"vllm", "sglang"}, "unknown history backend")
    request, ledger = snapshot["request"], snapshot["ledger"]
    depth = history_depth(snapshot)
    verify_pre_native_history(snapshot)
    for key in ("request", "ledger"):
        original_input = snapshot["input_bytes"][key]
        require(_decoded(original_input) == snapshot[key], "original input JSON changed")
        require(
            all(original_input[k] == snapshot["inputs"][key][k] for k in ("sha256", "bytes")),
            "original input reference changed",
        )
    reconciliation.selection_schema(ledger)
    require(ledger["request_sha256"] == digest(request), "request/ledger mismatch")
    require(
        ledger["archive_campaign_roots"] == request["campaign_roots"],
        "campaign roots changed",
    )
    require(len(request["campaign_roots"]) == 1, "one whole backend campaign required")
    campaign = relative(request["campaign_roots"][0])
    original = Path(request["external_control_request"]["original_task_root"])
    require(
        original.is_absolute() and str(original / campaign) == snapshot["source_root"],
        "source root changed",
    )
    attempts = ledger["attempts"]
    by_job = {a["job"]: a for a in attempts}
    require(attempts and len(by_job) == len(attempts), "duplicate or empty attempt history")
    expected = {}
    starts = set()
    for attempt in attempts:
        require(
            attempt["terminal_state"] in TERMINAL[snapshot["backend"]] and attempt["final"] is not None,
            "nonterminal attempt",
        )
        job = attempt["job"]
        require(isinstance(job, str) and job.isdecimal(), "invalid original job")
        start_path = relative(attempt["started"]["path"])
        suffix = start_path.relative_to(campaign)
        require(
            len(suffix.parts) == depth
            and suffix.parts[-2:] == (job, "started.json")
            and suffix.parts[0] == attempt["deployment"],
            "original attempt path changed",
        )
        if request.get("factory_history_scope") == FACTORY_HISTORY_SCOPE:
            index, separator, cell = suffix.parts[1].partition("-")
            require(
                index.isdecimal() and 0 <= int(index) < 18 and separator and cell == attempt["cell_id"],
                "factory attempt index/child path differs",
            )
        require(
            str(start_path.parent) == attempt["original_attempt_directory"],
            "attempt directory changed",
        )
        require(str(suffix) not in starts, "duplicate original started file")
        starts.add(str(suffix))
        for label in ("started", "final", "checkpoint"):
            ref = attempt.get(label)
            if ref is None:
                require(label == "checkpoint", "original terminal files missing")
                continue
            name = str(relative(ref["path"]).relative_to(campaign))
            require(
                Path(name).is_relative_to(suffix.parent),
                "attempt metadata crosses jobs",
            )
            require(name not in expected, "duplicate original metadata member")
            expected[name] = {"sha256": ref["sha256"], "bytes": ref["bytes"]}
        start = _decoded(snapshot["originals"][str(suffix)])
        final_name = str(relative(attempt["final"]["path"]).relative_to(campaign))
        final = _decoded(snapshot["originals"][final_name])
        require(start["job"] == final["job"] == job, "original job differs")
        if snapshot["backend"] == "vllm":
            start_child, final_child = start["child"], final["child"]
        elif request["external_control_request"].get("adapter") == factory.ADAPTER:
            start_child = final_child = factory.history_identity(start, final)
        else:
            start_child, final_child = (
                start["selected"]["child_identity"],
                final["selected"]["child_identity"],
            )
        require(start_child == final_child, "original child changed")
        require(start_child["child_cell_id"] == attempt["cell_id"], "original cell differs")
        require(
            final["state"] == attempt["terminal_state"],
            "terminal summary differs from original final",
        )
    require(
        set(snapshot["originals"]) == set(expected),
        "missing or orphan original history bytes",
    )
    for name, ref in expected.items():
        _decoded(snapshot["originals"][name])
        require(
            all(snapshot["originals"][name][k] == v for k, v in ref.items()),
            "metadata reference changed",
        )
    for prior in request.get("history_floor", []):
        matches = [a for a in attempts if a["started"]["path"] == prior["started"]["path"]]
        require(
            len(matches) == 1 and all(matches[0].get(k) == v for k, v in prior.items()),
            "failed history floor omitted",
        )
    choices = ledger["selections"]
    require(len(choices) == 72, "exact72 complete-child choices required")
    deployments = dict.fromkeys(DEPLOYMENTS, 0)
    reconciled = snapshot.get("reconciliations", {})
    expected_reconciled = {cid for cid, choice in choices.items() if "reconciliation" in choice}
    require(set(reconciled) == expected_reconciled, "missing or orphan reconciliation proof")

    def load_reconciliation(cid, ref):
        record = reconciled[cid]
        require(all(record[k] == ref[k] for k in ("sha256", "bytes")), "reconciliation reference differs")
        return _decoded(record)

    for cid, choice in choices.items():
        require(choice["job"] in by_job, "selected job missing from history")
        selected = by_job[choice["job"]]
        require(selected["cell_id"] == cid, "invalid whole-child choice")
        proof = reconciliation.selected_proof(ledger, choice, selected, lambda ref: load_reconciliation(cid, ref))
        if proof is not None:
            identity = reconciliation.verify(proof)
            require(
                identity["task_root"] == str(original) and identity["backend"] == snapshot["backend"],
                "reconciliation campaign differs",
            )
            for path, ref in proof["artifact_inventory"].items():
                name = str(relative(path).relative_to(campaign))
                require(name not in expected or expected[name] == ref, "reconciliation original metadata differs")
                expected[name] = ref
        require(selected["deployment"] in deployments, "unknown deployment")
        deployments[selected["deployment"]] += 1
    require(
        set(deployments.values()) == {18},
        "exact18 selected children per deployment required",
    )
    if "native_roots" in snapshot:
        require(native_roots.scope(snapshot) == native_roots.SCOPE, "history root mapping requires explicit scope")
    if native_roots.scope(snapshot):
        _selected_roots(snapshot, snapshot["native_roots"], check_attempt_id=True)
    return expected, starts


def snapshot(backend, inputs, plan, archive):
    """Recheck original terminal bytes and the complete currently present started set."""
    request, ledger = bound(inputs["request"]), bound(inputs["ledger"])
    root = Path(request["external_control_request"]["original_task_root"])
    campaign = relative(request["campaign_roots"][0])
    original_source = root / campaign
    binding = plan.get("storage_root_binding")
    physical = archive.storage_path(original_source, binding, live=True)
    originals = {}
    for attempt in ledger["attempts"]:
        for key in ("started", "final", "checkpoint"):
            ref = attempt.get(key)
            if ref is None:
                continue
            p = archive.storage_path(root / relative(ref["path"]), binding, live=True)
            require(p.is_relative_to(physical), "history member escapes campaign")
            raw = checked_file(p).read_bytes()
            require(
                sha(raw) == ref["sha256"] and len(raw) == ref["bytes"],
                "original attempt metadata changed",
            )
            originals[str(relative(ref["path"]).relative_to(campaign))] = _record(raw)
    result = {
        "backend": backend,
        "request": request,
        "ledger": ledger,
        "inputs": inputs,
        "source_root": str(original_source),
        "originals": originals,
        "input_bytes": {key: _record(checked_file(inputs[key]["path"]).read_bytes()) for key in ("request", "ledger")},
    }
    if "factory_history_scope" in request:
        result["factory_history_scope"] = request["factory_history_scope"]
        history_depth(result)

        def read_pre_native(name):
            path = archive.storage_path(root / relative(name), binding, live=True)
            require(not path.is_relative_to(physical), "pre-native original aliases the native campaign")
            path = archive.absolute_safe(path)
            require(path.stat().st_size <= PRE_NATIVE_MAX_BYTES, "pre-native original exceeds metadata bound")
            return checked_file(path).read_bytes()

        result["pre_native_failure_evidence"] = {
            "scope": FACTORY_HISTORY_SCOPE,
            "storage": "ORIGINAL_BYTES_IN_HISTORY_SIDECAR_NOT_NATIVE_TAR",
            "originals": _pre_native_members(request, read_pre_native),
        }
    if ledger["schema"] == reconciliation.SELECTION_SCHEMA:
        result["reconciliations"] = {}
        for cid, choice in ledger["selections"].items():
            if "reconciliation" not in choice:
                continue
            ref = reconciliation.file_ref(choice["reconciliation"])
            p = archive.storage_path(root / relative(ref["path"]), binding, live=True)
            require(not p.is_relative_to(physical), "reconciliation diagnostics must be outside original campaign")
            require(not (p.parent / "failure.json").exists(), "reconciliation failure is preserved")
            raw = checked_file(p).read_bytes()
            require(sha(raw) == ref["sha256"] and len(raw) == ref["bytes"], "reconciliation proof changed")
            result["reconciliations"][cid] = _record(raw)
    _, expected_starts = _validate_snapshot(result)
    pattern = "/".join("*" for _ in range(history_depth(result) - 1))
    job_dirs = {str(p.relative_to(physical)) for p in physical.glob(pattern) if p.is_dir()}
    require(
        all(Path(p).name.isdecimal() for p in job_dirs),
        "unknown campaign job directory",
    )
    require(
        job_dirs == {str(Path(p).parent) for p in expected_starts},
        "new or omitted original attempt directory",
    )
    observed_starts = {str(p.relative_to(physical)) for p in physical.glob(pattern + "/started.json")}
    require(observed_starts == expected_starts, "new or omitted original attempt")
    jobs = [j for j in plan["jobs"] if j["backend"] == backend]
    require(
        jobs and all(j["source_root"] == str(original_source) for j in jobs),
        "archive source root differs",
    )
    raw_roots = [p for j in jobs for p in j["accepted_raw_roots"]]
    require(
        len(raw_roots) == len(set(raw_roots)) == 72,
        "exact72 accepted raw roots required",
    )
    if native_roots.uniform(jobs):
        selected = [r for j in jobs for r in j["accepted_native_roots"]]
        require([r["raw_root"] for r in selected] == raw_roots, "archive native root mapping differs")
        result.update(native_root_scope=native_roots.SCOPE, native_roots=selected)
        _selected_roots(result, selected, check_attempt_id=True)
    else:
        _selected_roots(
            result,
            [{"cell_id": Path(p).parts[-3], "raw_root": p} for p in raw_roots],
            check_attempt_id=False,
        )
    archive.validate_storage_binding(binding, live=True)
    return result


def _selected_roots(snapshot_value, roots, *, check_attempt_id):
    ledger = snapshot_value["ledger"]
    root_scope = native_roots
    require(root_scope.uniform(roots) == root_scope.scope(snapshot_value), "native root scope differs from history")
    require(
        len(roots) == 72 and len({r["raw_root"] for r in roots}) == 72,
        "exact72 native roots required",
    )
    indexed = {r["cell_id"]: r for r in roots}
    require(
        len(indexed) == 72 and set(indexed) == set(ledger["selections"]),
        "native whole-child set differs",
    )
    by_job = {a["job"]: a for a in ledger["attempts"]}
    task = Path(snapshot_value["request"]["external_control_request"]["original_task_root"])
    campaign = Path(snapshot_value["request"]["campaign_roots"][0])
    for cid, choice in ledger["selections"].items():
        original = by_job[choice["job"]]
        parent = task / relative(original["original_attempt_directory"])
        start_key = str(relative(original["started"]["path"]).relative_to(campaign))
        start = _decoded(snapshot_value["originals"][start_key])
        if (
            snapshot_value["backend"] == "sglang"
            and snapshot_value["request"]["external_control_request"].get("adapter") == factory.ADAPTER
        ):
            final_key = str(relative(original["final"]["path"]).relative_to(campaign))
            child = factory.history_identity(start, _decoded(snapshot_value["originals"][final_key]))
        else:
            child = start["child"] if snapshot_value["backend"] == "vllm" else start["selected"]["child_identity"]
        plan_sha = child["child_plan_sha256"]
        require(
            isinstance(plan_sha, str) and len(plan_sha) == 64 and set(plan_sha) <= set("0123456789abcdef"),
            "original child plan hash invalid",
        )
        expected_raw = parent / "artifacts" / plan_sha[:16] / "cells" / cid / "raw/node0000"
        if root_scope.scope(snapshot_value):
            collection, pod = root_scope.collection(indexed[cid], task)
            require(
                collection == expected_raw.parent
                and pod == expected_raw
                and indexed[cid].get("original_pod_root") == str(expected_raw),
                "accepted collection/pod differs from whole-child selection",
            )
        else:
            require(
                indexed[cid]["raw_root"] == str(expected_raw),
                "accepted raw root differs from whole-child selection",
            )
        if check_attempt_id:
            require(
                original["checkpoint"] is not None,
                "selected original checkpoint missing",
            )
            key = str(relative(original["checkpoint"]["path"]).relative_to(campaign))
            checkpoint = _decoded(snapshot_value["originals"][key])
            entry = checkpoint["cells"][cid]
            status = "cleanup_failed" if "reconciliation" in choice else "passed"
            require(
                entry["status"] == status and entry["attempt_id"] == indexed[cid]["attempt_id"],
                "accepted native attempt differs from original checkpoint",
            )
            if "reconciliation" in choice:
                proof = _decoded(snapshot_value["reconciliations"][cid])
                identity = reconciliation.verify(proof)
                require(identity["attempt_id"] == indexed[cid]["attempt_id"], "reconciled native attempt differs")


def _bundle_files(bundle, archive):
    return {
        name: reference(Path(bundle) / name)
        for name in (
            "receipt.json",
            archive.ARCHIVE,
            archive.INVENTORY,
            archive.INPUT_MANIFEST,
        )
    }


def verify_bundle_history(bundle, proof, archive):
    """Offline: actual tar verification plus exact history/inventory SHA closure."""
    dependencies(archive)
    bundle = Path(bundle)
    require(not (bundle / FAILURE).exists(), "history operation failed")
    require(
        proof["contract"] == CONTRACT and proof["maintenance"] == MAINTENANCE,
        "history contract/source differs",
    )
    require(
        proof["maintenance_source"] == MAINTENANCE_IDENTITY,
        "maintenance revision differs",
    )
    archive.verify_bundle(bundle)  # Original tar extraction/hash/member checks stay intact.
    files = _bundle_files(bundle, archive)
    for name, actual in files.items():
        require(
            all(proof["bundle_files"][name][k] == actual[k] for k in ("sha256", "bytes")),
            "archive bytes changed",
        )
    expected, starts = _validate_snapshot(proof["snapshot"])
    input_manifest = json.loads((bundle / archive.INPUT_MANIFEST).read_bytes())
    require(
        input_manifest.get("original_source_path", input_manifest["source_path"]) == proof["snapshot"]["source_root"],
        "archive source differs from history",
    )
    records = {r["path"]: r for r in archive.inventory_records(bundle / archive.INVENTORY)}
    depth = history_depth(proof["snapshot"])
    actual_starts = {
        name
        for name, r in records.items()
        if r["kind"] == "file" and len(Path(name).parts) == depth and Path(name).name == "started.json"
    }
    actual_jobs = {
        name for name, r in records.items() if r["kind"] == "directory" and len(Path(name).parts) == depth - 1
    }
    require(
        actual_starts == starts and actual_jobs == {str(Path(p).parent) for p in starts},
        "archive omitted or added original attempts",
    )
    for name, expected_file in expected.items():
        actual = records.get(name)
        require(
            actual is not None
            and actual["kind"] == "file"
            and actual["sha256"] == expected_file["sha256"]
            and actual["stat"]["size"] == expected_file["bytes"],
            "archive history member differs",
        )
    verify_reconciliation_inventory(proof["snapshot"], records)
    verify_native_inventory(proof["snapshot"], records)
    return proof["snapshot"]


def verify_native_inventory(snapshot_value, records):
    """Recheck copied inventory membership, without opening native data or tar."""
    if native_roots.scope(snapshot_value):
        _selected_roots(snapshot_value, snapshot_value["native_roots"], check_attempt_id=True)
        source = Path(snapshot_value["source_root"])
        for entry in snapshot_value["native_roots"]:
            root, _pod = native_roots.collection(entry, source)
            require(root.is_relative_to(source), "native collection escapes campaign")
            native_roots.inventory_root(records, str(root.relative_to(source)))


def verify_reconciliation_inventory(snapshot_value, records):
    for record in snapshot_value.get("reconciliations", {}).values():
        reconciliation.verify_archive_inventory(
            _decoded(record), records, campaign_root=snapshot_value["request"]["campaign_roots"][0]
        )


def _failure(output, error):
    output = Path(output)
    if output.is_dir() and not (output / FAILURE).exists():
        write(
            output / FAILURE,
            {
                "contract": CONTRACT,
                "state": "FAILED_PRESERVED_NOT_PUBLISHABLE",
                "error": repr(error),
            },
        )


def archive_closed(stage_root, plan, label, output, history_inputs, *, campaign, archive):
    dependencies(archive, campaign)
    jobs = campaign.load_plan(stage_root, plan)
    matches = [j for j in jobs.values() if campaign.name(j) == label]
    require(len(matches) == 1, "unknown archive label")
    backend = matches[0]["backend"]
    require(backend in history_inputs, "required history missing")
    before = snapshot(backend, history_inputs[backend], plan, archive)
    try:
        campaign.archive_one(stage_root, plan, label, output)
        after = snapshot(backend, history_inputs[backend], plan, archive)
        require(before == after, "history changed during archive")
        proof = {
            "contract": CONTRACT,
            "state": "ARCHIVE_HISTORY_PROVED_NOT_PUBLICATION_ACCEPTANCE",
            "maintenance_source": MAINTENANCE_IDENTITY,
            "maintenance": MAINTENANCE,
            "plan_sha256": digest(plan),
            "stage_sha256": plan["stage_sha256"],
            "snapshot": before,
            "bundle_files": _bundle_files(output, archive),
        }
        verify_bundle_history(output, proof, archive)
        require(
            snapshot(backend, history_inputs[backend], plan, archive) == before,
            "history changed during archive verification",
        )
        write(Path(output) / BUNDLE_PROOF, proof)
        return reference(Path(output) / BUNDLE_PROOF)
    except BaseException as error:
        _failure(output, error)
        raise


def bind_closed(stage_root, plan, bundles, destination, history_inputs, *, campaign, archive):
    dependencies(archive, campaign)
    jobs = campaign.load_plan(stage_root, plan)
    require(
        set(bundles) == {campaign.name(j) for j in jobs.values()},
        "complete bundle map required",
    )
    backends = {j["backend"] for j in jobs.values()}
    require(
        backends == set(history_inputs) == {"sglang", "vllm"},
        "both backend histories required",
    )
    before = {b: snapshot(b, history_inputs[b], plan, archive) for b in backends}
    proofs = {}
    for job in jobs.values():
        label = campaign.name(job)
        bundle = Path(bundles[label])
        require((bundle / BUNDLE_PROOF).is_file(), "required archived history missing")
        proof = json.loads(checked_file(bundle / BUNDLE_PROOF).read_bytes())
        require(
            proof["plan_sha256"] == digest(plan) and proof["stage_sha256"] == plan["stage_sha256"],
            "archive plan/stage differs",
        )
        key = str(bundle.resolve())
        if key not in proofs:
            require(
                verify_bundle_history(bundle, proof, archive) == before[job["backend"]],
                "archived history differs from current ledger",
            )
            proofs[key] = proof
        require(proof["snapshot"]["backend"] == job["backend"], "cross-backend history")
    try:
        campaign.bind(stage_root, plan, bundles, destination)
        require(
            {b: snapshot(b, history_inputs[b], plan, archive) for b in backends} == before,
            "history changed during bind",
        )
        result = {
            "contract": CONTRACT,
            "state": "BOUND_HISTORY_PROVED_STREAMED_ARCHIVE_GATE",
            "plan_sha256": digest(plan),
            "stage_sha256": plan["stage_sha256"],
            "external_records": reference(Path(destination) / "external-raw-evidence.json"),
            "bundle_proofs": {label: proofs[str(Path(bundle).resolve())] for label, bundle in bundles.items()},
        }
        # Same offline gate is also mandatory after the normal canonical validator.
        verify_bound_history(
            destination,
            result,
            bundles,
            expected_stage_sha256=plan["stage_sha256"],
            archive=archive,
        )
        require(
            {b: snapshot(b, history_inputs[b], plan, archive) for b in backends} == before,
            "history changed during bound verification",
        )
        write(Path(destination) / BOUND_PROOF, result)
        return reference(Path(destination) / BOUND_PROOF)
    except BaseException as error:
        _failure(destination, error)
        raise


def verify_bound_history(destination, proof, bundles, *, expected_stage_sha256, archive):
    """Additional mandatory publisher/offline gate; existing canonical gate still required.

    Caller must hash-bind `proof` through the publication manifest. Supplying an
    optional adjacent JSON file is insufficient and is not publication admission.
    """
    require(
        proof["contract"] == CONTRACT and proof["stage_sha256"] == expected_stage_sha256,
        "bound history/stage differs",
    )
    destination = Path(destination)
    require(not (destination / FAILURE).exists(), "history binding failed")
    data = checked_file(destination / "external-raw-evidence.json").read_bytes()
    require(
        sha(data) == proof["external_records"]["sha256"] and len(data) == proof["external_records"]["bytes"],
        "bound raw evidence changed",
    )
    records = json.loads(data)
    expected_labels = {
        f"{b}-{q}-{t}-{p}-{r}"
        for b in ("vllm", "sglang")
        for q in ("fp8", "nvfp4")
        for t in (2, 4)
        for p in ("prefill", "decode")
        for r in ("calibration", "holdout")
    }
    labels = [f"{r['backend']}-{r['weight_quantization']}-{r['tp']}-{r['phase']}-{r['role']}" for r in records]
    require(
        len(labels) == len(set(labels)) == 32
        and set(labels) == set(bundles) == set(proof["bundle_proofs"]) == expected_labels,
        "exact32 bound history labels required",
    )
    checked = {}
    backend_history = {}
    accepted_native_roots = {"vllm": [], "sglang": []}
    for label, record in zip(labels, records, strict=True):
        archive_proof = proof["bundle_proofs"][label]
        require(
            archive_proof["plan_sha256"] == proof["plan_sha256"]
            and archive_proof["stage_sha256"] == expected_stage_sha256,
            "mixed plan/stage proof",
        )
        key = (str(Path(bundles[label]).resolve()), digest(archive_proof))
        if key not in checked:
            checked[key] = verify_bundle_history(bundles[label], archive_proof, archive)
        snapshot_value = checked[key]
        backend = record["backend"]
        require(
            record["stage_sha256"] == expected_stage_sha256,
            "native records belong to another stage",
        )
        accepted_native_roots[backend].extend(record["native_roots"])
        require(snapshot_value["backend"] == backend, "history backend differs")
        require(
            backend_history.setdefault(backend, snapshot_value) == snapshot_value,
            "mixed backend history epochs",
        )
        files = archive_proof["bundle_files"]
        require(
            record["sha256"] == files[archive.ARCHIVE]["sha256"] and record["bytes"] == files[archive.ARCHIVE]["bytes"],
            "history not bound to accepted archive",
        )
        for field, name in (
            ("archive_receipt", "receipt.json"),
            ("source_inventory", archive.INVENTORY),
            ("archive_input_manifest", archive.INPUT_MANIFEST),
        ):
            require(
                record[field]["sha256"] == files[name]["sha256"] and record[field]["bytes"] == files[name]["bytes"],
                "history not bound to accepted archive sidecar",
            )
    for backend, snap in backend_history.items():
        _selected_roots(snap, accepted_native_roots[backend], check_attempt_id=True)
    return {
        "contract": CONTRACT,
        "histories": len(backend_history),
        "labels": 32,
        "state": "OFFLINE_HISTORY_CLOSURE_PASS_NOT_NATIVE_OR_ACCURACY_ACCEPTANCE",
    }


def require_publication_history(manifest, destination, bundles, *, expected_stage_sha256, archive):
    """Mandatory full-archive import gate; separate from portable metadata validation.

    No optional-sidecar discovery. The new publication contract must require this
    field, and independently run the unchanged canonical acceptance validator.
    """
    entry = manifest.get("external_raw_history")
    require(
        isinstance(entry, dict) and set(entry) == {"contract", "proof"},
        "required publication history missing",
    )
    require(entry["contract"] == CONTRACT, "unknown publication history contract")
    ref = entry["proof"]
    path = archive.absolute_safe(Path(destination) / relative(ref["path"]))
    require(
        path.is_relative_to(Path(destination).resolve()),
        "publication history escapes bound root",
    )
    raw = checked_file(path).read_bytes()
    require(
        sha(raw) == ref["sha256"] and len(raw) == ref["bytes"],
        "publication history digest changed",
    )
    return verify_bound_history(
        destination,
        json.loads(raw),
        bundles,
        expected_stage_sha256=expected_stage_sha256,
        archive=archive,
    )
