# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mandatory portable metadata closure after a separate actual-tar import gate.

The offline verifier never opens an archive or a live producer tree. It checks
copied original receipts/inventories/input manifests and attempt history. Its
result is not a fresh tar verification, native qualification, or accuracy claim.
"""

from __future__ import annotations

import json
from pathlib import Path

if __package__:
    from . import closed_history as h
else:
    import closed_history as h

PORTABLE_CONTRACT = "fpm_portable_attempt_history_metadata_v1"
PROOF = "portable-history.json"
FIELDS = ("backend", "weight_quantization", "tp", "phase", "role")
LABELS = {
    f"{b}-{q}-{t}-{p}-{r}"
    for b in ("vllm", "sglang")
    for q in ("fp8", "nvfp4")
    for t in (2, 4)
    for p in ("prefill", "decode")
    for r in ("calibration", "holdout")
}
require = h.require


def _sha(value):
    require(
        isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef"),
        "invalid SHA256",
    )
    return value


def _read(root, ref, archive, files):
    require(
        isinstance(ref, dict) and set(ref) == {"path", "sha256", "bytes"},
        "invalid portable reference",
    )
    require(type(ref["bytes"]) is int and ref["bytes"] >= 0, "invalid portable byte count")
    _sha(ref["sha256"])
    root = archive.absolute_safe(Path(root))
    path = archive.absolute_safe(root / h.relative(ref["path"]))
    require(path.is_relative_to(root), "portable reference escapes root")
    data = h.checked_file(path).read_bytes()
    require(
        h.sha(data) == ref["sha256"] and len(data) == ref["bytes"],
        "portable metadata bytes changed",
    )
    files[str(path)] = ref["sha256"]
    return data, path


def _label(record):
    require(type(record["tp"]) is int, "noninteger TP identity")
    return "-".join(str(record[key]) for key in FIELDS)


def _same_size_sha(left, right):
    return all(left[k] == right[k] for k in ("sha256", "bytes"))


def _metadata(proof, refs, base, archive, files):
    """Recheck copied originals, not the referenced tar or producer filesystem."""
    require(
        proof["contract"] == h.CONTRACT and proof["maintenance"] == h.MAINTENANCE,
        "history contract/source differs",
    )
    require(
        proof["maintenance_source"] == h.MAINTENANCE_IDENTITY,
        "history source identity differs",
    )
    names = {"receipt.json", archive.INVENTORY, archive.INPUT_MANIFEST}
    require(set(refs) == names, "missing or extra portable bundle sidecar")
    loaded, paths = {}, {}
    for name in sorted(names):
        raw, paths[name] = _read(base, refs[name], archive, files)
        require(
            _same_size_sha(refs[name], proof["bundle_files"][name]),
            "history sidecar digest differs",
        )
        if name != archive.INVENTORY:
            loaded[name] = json.loads(raw)
    receipt, manifest = loaded["receipt.json"], loaded[archive.INPUT_MANIFEST]
    require(
        receipt["schema"] == archive.SCHEMA and receipt["status"] == "PASS",
        "original archive receipt did not pass",
    )
    for field, name in (
        ("archive", archive.ARCHIVE),
        ("source_inventory", archive.INVENTORY),
        ("input_manifest", archive.INPUT_MANIFEST),
    ):
        require(
            receipt[field]["path"] == name and _same_size_sha(receipt[field], proof["bundle_files"][name]),
            "original receipt sidecar/archive identity differs",
        )
    require(
        manifest["source_inventory"] == receipt["source_inventory"]
        and manifest["labels"] == receipt["labels"]
        and manifest["external_uri"] == receipt["external_uri"],
        "input manifest disagrees with original receipt",
    )
    require(
        manifest.get("storage_root_binding") == receipt.get("storage_root_binding"),
        "original storage identity differs",
    )
    if "storage_root_binding" in manifest:
        require(
            archive.storage_path(manifest["original_source_path"], manifest["storage_root_binding"])
            == Path(manifest["source_path"]),
            "original/canonical metadata paths differ",
        )
    archive.validate_identity(receipt["external_uri"], receipt["labels"])
    snap = proof["snapshot"]
    expected, starts = h._validate_snapshot(snap)
    require(
        manifest.get("original_source_path", manifest["source_path"]) == snap["source_root"],
        "archive/history original source differs",
    )
    inventory = {row["path"]: row for row in archive.inventory_records(paths[archive.INVENTORY])}
    depth = 4 if snap["backend"] == "vllm" else 3
    actual_starts = {
        name
        for name, row in inventory.items()
        if row["kind"] == "file" and len(Path(name).parts) == depth and Path(name).name == "started.json"
    }
    actual_jobs = {
        name for name, row in inventory.items() if row["kind"] == "directory" and len(Path(name).parts) == depth - 1
    }
    require(
        actual_starts == starts and actual_jobs == {str(Path(p).parent) for p in starts},
        "portable inventory attempt membership differs",
    )
    for name, expected_ref in expected.items():
        actual = inventory.get(name)
        require(
            actual is not None
            and actual["kind"] == "file"
            and actual["sha256"] == expected_ref["sha256"]
            and actual["stat"]["size"] == expected_ref["bytes"],
            "original history/inventory member differs",
        )
    h.verify_reconciliation_inventory(snap, inventory)
    counts = {
        "files": sum(row["kind"] == "file" for row in inventory.values()),
        "directories": sum(row["kind"] == "directory" for row in inventory.values()),
        "logical_bytes": sum(row["stat"]["size"] for row in inventory.values() if row["kind"] == "file"),
    }
    require(
        all(receipt["verification"][key] == value for key, value in counts.items()),
        "original receipt/inventory counts differ",
    )
    return snap, receipt, inventory


def verify_portable_history(root, entry, records, *, expected_stage_sha256, archive):
    """Verify mandatory metadata closure only; return every local immutable SHA."""
    h.dependencies(archive)
    _sha(expected_stage_sha256)
    require(
        isinstance(entry, dict) and set(entry) == {"contract", "proof"} and entry["contract"] == PORTABLE_CONTRACT,
        "required portable history contract missing",
    )
    files = {}
    raw, proof_path = _read(root, entry["proof"], archive, files)
    proof = json.loads(raw)
    require(
        set(proof)
        == {
            "contract",
            "stage_sha256",
            "import_gate",
            "bound_history",
            "external_records",
            "bundles",
        },
        "unknown portable proof fields",
    )
    require(
        proof["contract"] == PORTABLE_CONTRACT and proof["stage_sha256"] == expected_stage_sha256,
        "portable history stage/contract differs",
    )
    require(
        proof["import_gate"] == "ACTUAL_TAR_AND_CLOSED_HISTORY_VERIFIED_AT_IMPORT",
        "missing actual import gate declaration",
    )
    base = proof_path.parent
    require(not (base / h.FAILURE).exists(), "portable preparation failed")
    raw, _ = _read(base, proof["bound_history"], archive, files)
    bound = json.loads(raw)
    require(
        bound["contract"] == h.CONTRACT and bound["stage_sha256"] == expected_stage_sha256,
        "bound history stage/contract differs",
    )
    raw, _ = _read(base, proof["external_records"], archive, files)
    require(
        _same_size_sha(proof["external_records"], bound["external_records"]) and json.loads(raw) == records,
        "bound external records differ",
    )
    labels = [_label(record) for record in records]
    require(
        len(labels) == len(set(labels)) == 32
        and set(labels) == set(proof["bundles"]) == set(bound["bundle_proofs"]) == LABELS,
        "exact32 portable labels required",
    )
    checked, histories = {}, {}
    native_roots = {"vllm": [], "sglang": []}
    for label, record in zip(labels, records, strict=True):
        bundle_proof = bound["bundle_proofs"][label]
        require(
            bundle_proof["plan_sha256"] == bound["plan_sha256"]
            and bundle_proof["stage_sha256"] == expected_stage_sha256
            and record["stage_sha256"] == expected_stage_sha256,
            "mixed portable plan/stage",
        )
        refs = proof["bundles"][label]
        key = h.digest({"proof": bundle_proof, "refs": refs})
        if key not in checked:
            checked[key] = _metadata(bundle_proof, refs, base, archive, files)
        snap, receipt, inventory = checked[key]
        backend = record["backend"]
        require(
            snap["backend"] == backend and histories.setdefault(backend, snap) == snap,
            "mixed portable backend history",
        )
        require(
            {k: record[k] for k in FIELDS} in receipt["labels"],
            "record label absent from original archive receipt",
        )
        bundle_files = bundle_proof["bundle_files"]
        require(
            _same_size_sha(record, bundle_files[archive.ARCHIVE]),
            "record archive differs from history",
        )
        for field, name in (
            ("archive_receipt", "receipt.json"),
            ("source_inventory", archive.INVENTORY),
            ("archive_input_manifest", archive.INPUT_MANIFEST),
        ):
            require(
                _same_size_sha(record[field], bundle_files[name]),
                "record sidecar differs from history",
            )
        for native in record["native_roots"]:
            relative = str(Path(native["raw_root"]).relative_to(Path(snap["source_root"])))
            require(
                relative in inventory and inventory[relative]["kind"] == "directory",
                "native root absent from original inventory",
            )
        native_roots[backend].extend(record["native_roots"])
    require(set(histories) == {"vllm", "sglang"}, "both portable histories required")
    for backend, snap in histories.items():
        h._selected_roots(snap, native_roots[backend], check_attempt_id=True)
    root = archive.absolute_safe(Path(root))
    return {
        "contract": PORTABLE_CONTRACT,
        "state": "PORTABLE_METADATA_HISTORY_PASS_NO_FRESH_TAR_VERIFICATION",
        "histories": 2,
        "labels": 32,
        "native_or_accuracy_acceptance": "NOT_EVALUATED",
        "files": {str(Path(path).relative_to(root)): sha for path, sha in sorted(files.items())},
    }


def prepare_portable_history(bound_root, history_ref, bundles, destination, *, expected_stage_sha256, archive):
    """Stream actual archives through the strict gate, then copy exact metadata."""
    bound_root, destination = Path(bound_root), Path(destination)
    require(not destination.exists(), "portable destination must be new")
    entry = {"contract": h.CONTRACT, "proof": history_ref}
    # This performs real archive/member hashing; no metadata-only fallback.
    h.require_publication_history(
        {"external_raw_history": entry},
        bound_root,
        bundles,
        expected_stage_sha256=expected_stage_sha256,
        archive=archive,
    )
    raw, _ = _read(bound_root, history_ref, archive, {})
    bound = json.loads(raw)
    destination.mkdir(parents=True)
    (destination / "files").mkdir()

    def copy_original(path, expected):
        data = h.checked_file(archive.absolute_safe(Path(path))).read_bytes()
        require(
            h.sha(data) == expected["sha256"] and len(data) == expected["bytes"],
            "original changed after import verification",
        )
        rel = "files/" + h.sha(data)
        target = destination / rel
        if target.exists():
            require(target.read_bytes() == data, "portable content collision")
        else:
            with target.open("xb") as stream:
                stream.write(data)
        return {"path": rel, "sha256": h.sha(data), "bytes": len(data)}

    try:
        history = copy_original(bound_root / h.relative(history_ref["path"]), history_ref)
        records_ref = copy_original(bound_root / "external-raw-evidence.json", bound["external_records"])
        sidecars = {
            label: {
                name: copy_original(
                    Path(bundle) / name,
                    bound["bundle_proofs"][label]["bundle_files"][name],
                )
                for name in ("receipt.json", archive.INVENTORY, archive.INPUT_MANIFEST)
            }
            for label, bundle in sorted(bundles.items())
        }
        result = {
            "contract": PORTABLE_CONTRACT,
            "stage_sha256": expected_stage_sha256,
            "import_gate": "ACTUAL_TAR_AND_CLOSED_HISTORY_VERIFIED_AT_IMPORT",
            "bound_history": history,
            "external_records": records_ref,
            "bundles": sidecars,
        }
        h.write(destination / PROOF, result)
        ref = h.reference(destination / PROOF)
        ref["path"] = PROOF
        portable = {"contract": PORTABLE_CONTRACT, "proof": ref}
        records = json.loads((destination / records_ref["path"]).read_bytes())
        verify_portable_history(
            destination,
            portable,
            records,
            expected_stage_sha256=expected_stage_sha256,
            archive=archive,
        )
        return portable
    except BaseException as error:
        h._failure(destination, error)
        raise
