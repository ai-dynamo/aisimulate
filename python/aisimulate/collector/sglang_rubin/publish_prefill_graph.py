# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish the approved, exact seven-context Rubin prefill graph profile offline.

The portable evidence bundle contains the unchanged qualified producers, raw
exports and decisions. Frozen producer verifiers run again before any rows are
derived. Native full-forward measurements are not inputs to the timing reducers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

MODULE = "collector.sglang_rubin.publish_prefill_graph"
PROFILE = "sglang_glm52_nvfp4_vr200_tp4_graph_v1"
IDENTITY_PATH = Path(__file__).with_name("prefill_graph_identity.json")
PROTOCOLS = {"attention": "joint_attention_sequence", "communication": "block_5x20"}
ATTENTION_ARMS = ("separate_intervals", "common_interval_separate_graphs", "joint_attention_sequence")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(path.read_text(), parse_constant=reject)


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _record(path):
    _require(path.is_file() and not path.is_symlink(), f"Missing regular file: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "size_bytes": path.stat().st_size}


def _relative(root, name):
    path = Path(name)
    _require(
        not path.is_absolute() and path.parts and ".." not in path.parts and str(path) == name,
        f"Noncanonical bundle-relative path: {name}",
    )
    result = root / path
    _require(result.resolve() == result, f"Symlink or path escape: {name}")
    return result


def _pin(root, value):
    path = _relative(root, value["path"])
    _require(_record(path) == {key: value[key] for key in ("sha256", "size_bytes")}, f"Input changed: {value['path']}")
    return path


def _inventory(root):
    _require(root.is_dir() and root.resolve() == root, "Invalid or symlinked bundle root")
    files = {}
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), f"Symlink in bundle: {path}")
        if not path.is_dir():
            files[str(path.relative_to(root))] = _record(path)
    return files


def _same_pin(left, right):
    return all(left[key] == right[key] for key in ("sha256", "size_bytes"))


def _authority(evidence, identity):
    decisions = {
        name: _read(_pin(evidence, pin)) for name, pin in identity["authority"].items() if name != "composition_source"
    }
    approval, acceptance = decisions["approval"], decisions["accuracy_acceptance"]
    _require(
        approval["status"] == "USER_APPROVED_COORDINATED_PUBLIC_CONTRACT_AND_DRAFT_PR"
        and _same_pin(approval["approved_contract"], identity["authority"]["proposal_freeze"])
        and _same_pin(approval["root_reviewed_contract"], identity["authority"]["proposal_root_acceptance"])
        and _same_pin(approval["existing_accuracy_acceptance"], identity["authority"]["accuracy_acceptance"]),
        "Missing explicit approval of this reviewed public contract",
    )
    _require(
        acceptance["status"] == "ROOT_ACCEPTED_VERIFIED_BOUNDED_FORWARD_TARGET_MET"
        and acceptance["accuracy_acceptance"] is True
        and acceptance["accuracy_gate"]["passing_contexts"] == acceptance["accuracy_gate"]["required_contexts"] == 7
        and acceptance["accuracy_gate"]["failing_contexts"] == 0
        and acceptance["accuracy_gate"]["threshold"] == 0.15
        and acceptance["accuracy_gate"]["all_seven_within_15_percent"] is True
        and acceptance["native_latency_used_as_prediction_input"] is False
        and acceptance["no_fitting_or_sample_selection"] is True
        and _same_pin(acceptance["independent_verdict"], identity["authority"]["accuracy_verdict"]),
        "Missing separate fixed-formula accuracy acceptance",
    )
    for kind, operator in identity["operators"].items():
        _require(
            _same_pin(acceptance["operator_qualifications"][kind], operator["qualification"]),
            "Accuracy acceptance concerns different operator inputs",
        )
    proposal = decisions["proposal"]
    _require(
        proposal["table_contracts"] == identity["table_contracts"]
        and proposal["table_contracts"]["attention"]["key_inventory"] == identity["admitted_contexts"],
        "Rows differ from approved table contract",
    )


def _verify_operator(evidence, identity, kind):
    operator = identity["operators"][kind]
    qualification = _read(_pin(evidence, operator["qualification"]))
    verdict = _read(_pin(evidence, operator["verdict"]))
    candidate_path = _pin(evidence, operator["candidate"])
    candidate = _read(candidate_path)
    export = _read(_pin(evidence, operator["export_manifest"]))
    _require(
        qualification["status"] == "ROOT_ACCEPTED_QUALIFIED_OPERATOR_MEASUREMENTS"
        and qualification["operator"] == kind
        and qualification["numerical_and_dispatch_qualified"] is True
        and qualification["timing_qualified"] is True
        and qualification["accuracy_acceptance"] is False
        and qualification["qualified_for_private_composition"] is True
        and qualification["prediction_protocol"] == PROTOCOLS[kind]
        and _same_pin(qualification["candidate_freeze"], operator["candidate"])
        and _same_pin(qualification["export_manifest"], operator["export_manifest"])
        and _same_pin(qualification["independent_verdict"], operator["verdict"])
        and verdict["findings"] == [],
        f"Incomplete {kind} root qualification",
    )
    if kind == "attention":
        _require(
            verdict["status"] == "CLEAN_ACTUAL_OPERATOR_REVIEW_WITH_DECLARED_PRIVATE_PROXY_LIMITS"
            and verdict["numerical_and_dispatch_conclusion"]["qualified"] is True
            and verdict["timing_conclusion"]["suitable_for_declared_private_proxy"] is True
            and verdict["timing_conclusion"]["recommended_prediction_protocol"] == PROTOCOLS[kind],
            "Incomplete independent attention qualification",
        )
    else:
        _require(
            verdict["status"] == "NUMERICAL_DISPATCH_PASS_BLOCK_TIMING_RECOMMENDED_FOR_DECLARED_PRIVATE_PROXY"
            and verdict["numerical_dispatch_conclusion"]["status"] == "PASS"
            and verdict["timing_conclusion"]["status"]
            == "RECOMMEND_BLOCK_5X20_AS_DECLARED_PRIVATE_APPROXIMATION_PENDING_ROOT_AUTHORIZATION",
            "Incomplete independent communication qualification",
        )
    # Only the already pinned documentary member may live outside the original
    # candidate root. Its original name hash and exact bytes bind the relocation.
    external = set()
    for name, expected in candidate["files"].items():
        if Path(name).is_absolute():
            key = hashlib.sha256(name.encode()).hexdigest()
            pin = operator["external_candidate_members"].get(key)
            _require(pin is not None and _same_pin(pin, expected), "Unknown external candidate member")
            _pin(evidence, pin)
            external.add(key)
        else:
            path = _relative(candidate_path.parent, name)
            _require(_record(path) == expected, f"Candidate member changed: {name}")
    _require(external == set(operator["external_candidate_members"]), "Unused external-member exception")
    run = _relative(evidence, operator["run_root"])
    _require(_inventory(run) == export["files"], "Incomplete or changed raw export")
    source = candidate_path.parent / "implementation"
    inputs = _read(candidate_path.parent / "candidate/input-manifest.json")
    _require(inputs == _read(run / "source/input-manifest.json"), "Runtime input manifest differs")
    for name, expected in inputs["files"].items():
        _require(
            _record(_relative(source, name)) == _record(_relative(run / "source", name)) == expected,
            "Consumed runtime source differs from the frozen candidate",
        )
    verifier = source / "verify_results.py"
    spec = importlib.util.spec_from_file_location(f"qualified_prefill_{kind}_verifier", verifier)
    _require(spec is not None and spec.loader is not None, "Cannot load frozen verifier")
    module = importlib.util.module_from_spec(spec)
    old_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
        verified = module.verify(run / "results", source)
    finally:
        sys.dont_write_bytecode = old_bytecode
    return verified, run, source


def _mean(samples):
    _require(
        len(samples) == 30 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples),
        "Expected all 30 positive finite timing samples",
    )
    # Match the frozen private predictor's correctly rounded arithmetic mean.
    return statistics.mean(samples)


def _attention_rows(verified, identity):
    contract = identity["table_contracts"]["attention"]
    _require(
        verified["status"] == "passed" and verified["primary_arm"] == PROTOCOLS["attention"],
        "Attention verification did not pass",
    )
    rows = []
    for context in verified["contexts"]:
        shape = context["shape"]
        new, past = shape["new_tokens"], shape["past_kv"]
        _require(new and len(new) == len(past) and len(set(new)) == len(set(past)) == 1, "Heterogeneous context")
        _require(set(context["arms"]) == set(ATTENTION_ARMS), "Missing required attention control arm")
        for arm in ATTENTION_ARMS:
            samples = context["arms"][arm]["timing"]["samples"]
            _require(
                [(x["fixture"], x["window"], x["sample"]) for x in samples]
                == [(f, w, s) for f in (0, 1) for w in range(3) for s in range(5)],
                "Incomplete attention fixture/window/sample matrix",
            )
            _mean([x["gpu_ms"] for x in samples])
        rows.append(
            {
                **_row_labels(identity, "attention", contract["kernel_source"]),
                "batch_size": len(new),
                "input_seq_len": new[0],
                "prefix_len": past[0],
                "latency": _mean([x["gpu_ms"] for x in context["arms"][PROTOCOLS["attention"]]["timing"]["samples"]]),
            }
        )
    _require(
        [{key: row[key] for key in ("batch_size", "input_seq_len", "prefix_len")} for row in rows]
        == identity["admitted_contexts"],
        "Attention contexts differ from the seven approved keys",
    )
    return rows


def _row_labels(identity, kind, kernel):
    return {
        **{key: identity["runtime"][key] for key in ("framework", "version", "device")},
        "op_name": identity["table_contracts"][kind]["operation_name"],
        "kernel_source": kernel,
        "measurement_protocol": PROTOCOLS[kind],
    }


def _communication_rows(verified, identity):
    _require(verified["status"] == "diagnostic_complete", "Communication verification did not pass")
    contract = identity["table_contracts"]["communication"]
    boundaries = verified["boundaries"]
    keys = [(b["tokens"], b["role"], b["protocol"]) for b in boundaries]
    expected = {
        (key["num_tokens"], key["boundary_role"], protocol)
        for key in contract["key_inventory"]
        for protocol in ("single_replay", "block_5x20")
    }
    _require(len(keys) == len(set(keys)) == len(expected) and set(keys) == expected, "Incomplete communication matrix")
    values = {}
    for boundary in boundaries:
        tokens, role, protocol = boundary["tokens"], boundary["role"], boundary["protocol"]
        count = 100 if protocol == "block_5x20" else 1
        normalization = {
            "captured_calls": 5 if count == 100 else 1,
            "replays_per_sample": 20 if count == 100 else 1,
            "native_calls_per_sample": count,
        }
        samples = boundary["timing"]["samples"]
        _require(
            boundary["mode"] == "graph"
            and boundary["path"] == contract["kernel_source_by_tokens"][str(tokens)]
            and boundary["timing"]["normalization"] == normalization
            and [(x["window"], x["sample"]) for x in samples] == [(w, s) for w in range(3) for s in range(10)],
            "Communication role, path, timing protocol or sample matrix changed",
        )
        reduced = []
        for sample in samples:
            ranks = sample["rank_cuda_ms"]
            _require(
                len(ranks) == 4 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in ranks),
                "Incomplete or invalid TP4 rank timings",
            )
            maximum = max(ranks)
            _require(
                sample["max_rank_cuda_ms"] == maximum and sample["per_call_max_rank_cuda_ms"] == maximum / count,
                "Incorrect aligned rank maximum or per-call normalization",
            )
            reduced.append(maximum / count)
        mean = _mean(reduced)
        if protocol == PROTOCOLS["communication"]:
            values[tokens, role] = mean
    return [
        {
            **_row_labels(identity, "communication", contract["kernel_source_by_tokens"][str(key["num_tokens"])]),
            **key,
            "latency": values[key["num_tokens"], key["boundary_role"]],
        }
        for key in contract["key_inventory"]
    ]


def qualified_profile(evidence, base, identity):
    """Replay complete qualification and derive the immutable, portable profile."""
    _require(_inventory(evidence) == identity["evidence_files"], "Evidence bundle is incomplete or changed")
    for pin in identity["retained_files"] + identity["base_metadata"]:
        _pin(base, pin)
    _authority(evidence, identity)
    attention, attention_run, attention_source = _verify_operator(evidence, identity, "attention")
    communication, communication_run, communication_source = _verify_operator(evidence, identity, "communication")
    devices = {_read(path)["runtime"]["device"] for path in (attention_run / "results").glob("*/result.json")} | {
        _read(communication_run / f"results/rank-{rank}.json")["device"]["name"] for rank in range(4)
    }
    _require(devices == {identity["runtime"]["device"]}, "Qualified GPU device label changed")
    tables = {}
    for kind, rows in (
        ("attention", _attention_rows(attention, identity)),
        ("communication", _communication_rows(communication, identity)),
    ):
        contract = identity["table_contracts"][kind]
        tables[kind] = {
            "relative_path": f"{contract['relative_directory']}/{contract['canonical_filename']}",
            "rows": rows,
        }
    profile = {
        "schema_version": 1,
        "profile_name": PROFILE,
        "runtime": identity["runtime"],
        "admitted_contexts": identity["admitted_contexts"],
        "tables": tables,
        "retained_files": identity["retained_files"],
        "fixed_composition": identity["fixed_composition"],
        "limitations": identity["limitations"],
        "energy_measured": False,
        "sol_supported": False,
        "full_model_memory_qualified": False,
        "provenance": {
            "publisher_identity": _record(IDENTITY_PATH),
            "authority": identity["authority"],
            "evidence_files": identity["evidence_files"],
            "operators": identity["operators"],
            "original_collection_metadata": identity["base_metadata"],
            "runtime": identity["source_runtime"],
            "native_profile": _read(attention_source / "native-profile.json"),
            "attention_config": _read(attention_source / "pilot_config.json"),
            "attention_provenance": _read(attention_source / "provenance.json"),
            "communication_config": _read(communication_source / "pilot_config.json"),
            "communication_provenance": _read(communication_source / "provenance.json"),
            "representative_norm_tensors": _read(communication_run / "results/rank-0.json")["norms"],
            "all_primary_samples_retained": True,
            "control_samples_used_for_costs": False,
            "native_forward_timings_used_for_costs": False,
        },
    }
    _require(_inventory(evidence) == identity["evidence_files"], "Evidence changed during verification")
    return profile


def _write_table(target, rows, contract):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from collector import helper

    csv = target / contract["staging_filename"]
    labels = ("framework", "version", "device", "op_name", "kernel_source")
    for row in rows:
        helper.log_perf(
            [{key: value for key, value in row.items() if key not in labels}],
            **{key: row[key] for key in labels if key != "device"},
            device_name=row["device"],
            perf_filename=str(csv),
        )
    parquet = target / contract["canonical_filename"]
    _require(helper.finalize_perf_files([csv], merge_existing=False) == [parquet], "Table finalization failed")
    schema = pa.schema(
        [pa.field(c["name"], pa.type_for_alias(c["arrow_type"]), nullable=False) for c in contract["columns"]]
    )
    # CSV inference may round Float64 text. Check all keys, then write the exact
    # recomputed Float64 values and approved non-null Arrow schema.
    inferred = pq.read_table(parquet).to_pylist()
    _require(len(inferred) == len(rows), "Finalizer lost rows")
    for actual, expected in zip(inferred, rows, strict=True):
        _require(
            {k: v for k, v in actual.items() if k != "latency"}
            == {k: v for k, v in expected.items() if k != "latency"},
            "Finalizer changed row identity",
        )
        _require(math.isclose(actual["latency"], expected["latency"], rel_tol=1e-14), "Finalizer changed latency")
    exact = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(exact, parquet)
    _require(pq.read_table(parquet).equals(exact, check_metadata=True), "Exact Arrow readback differs")
    lock = parquet.with_name(parquet.name + ".mergelock")
    if lock.exists():
        _require(_record(lock)["size_bytes"] == 0, "Unexpected private merge lock")
        lock.unlink()


def publish(*, evidence, base_systems, output_systems, validate_only=False):
    """Expose a complete fresh systems tree atomically, or validate an exact repeat."""
    import yaml

    from collector import helper, provenance

    evidence, base, output = (Path(p).absolute() for p in (evidence, base_systems, output_systems))
    _require(
        output.resolve() == output
        and output != base
        and not output.is_relative_to(base)
        and not output.is_relative_to(evidence.resolve()),
        "Unsafe output location",
    )
    identity = _read(IDENTITY_PATH)
    _require(identity["profile_name"] == PROFILE and identity["module"] == MODULE, "Wrong publisher identity")
    before = _inventory(base)
    profile = qualified_profile(evidence, base, identity)
    data = _bytes(profile)
    profile_id = hashlib.sha256(data).hexdigest()
    package = Path(__file__).resolve().parents[2]
    closures = provenance.load_closures(package / "collector/hash_closures.yaml")
    module_hash = provenance.collector_hash(MODULE, package, closures)
    receipt_name = f"prefill_graph_publications/{PROFILE}.json"
    result = {
        "profile_id": profile_id,
        "profile_name": PROFILE,
        "rows": {kind: len(table["rows"]) for kind, table in profile["tables"].items()},
    }
    if validate_only:
        return {"status": "VALIDATED_ONLY", **result}
    runtime = {key: identity["runtime"][key] for key in ("framework", "version", "image_digest")}
    runtime["image"] = "gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo-ci"
    with tempfile.TemporaryDirectory(prefix="aisim-prefill-graph-", dir=output.parent) as temporary:
        staged = Path(temporary) / "systems"
        shutil.copytree(base, staged)
        _require(_inventory(staged) == before, "Base changed while staging")
        changed = set()
        added = {receipt_name}
        for kind, table in profile["tables"].items():
            contract = identity["table_contracts"][kind]
            target = _relative(staged, contract["relative_directory"])
            target.mkdir(parents=True, exist_ok=True)
            parquet = _relative(staged, table["relative_path"])
            profile_file = target / identity["profile_filename"]
            _require(not parquet.exists() and not profile_file.exists(), "Base already contains graph profile data")
            rows = [{**row, "profile_id": profile_id} for row in table["rows"]]
            _write_table(target, rows, contract)
            profile_file.write_bytes(data)
            meta_path = target / "collection_meta.yaml"
            metadata = yaml.safe_load(meta_path.read_text())
            provenance.validate_collection_meta_for_update(metadata)
            original_runtime = metadata["runtime"]
            _require(
                all(original_runtime.get(k) == runtime[k] for k in ("framework", "version"))
                and (
                    (original_runtime.get("image"), original_runtime.get("image_digest"))
                    == (runtime["image"], runtime["image_digest"])
                    or (original_runtime.get("image"), original_runtime.get("image_digest"))
                    == (runtime["image"] + "@" + runtime["image_digest"], None)
                ),
                "Base runtime differs",
            )
            event = {
                "collector_ref": MODULE,
                "collector_hash": module_hash,
                "case_plan_hash": "sha256:" + _record(IDENTITY_PATH)["sha256"],
                "collected_at": identity["collected_at"],
                "rows": len(rows),
                "status": "complete",
                "runtime": runtime,
            }
            table_name = Path(contract["canonical_filename"]).stem
            _require(table_name not in metadata["tables"], "Base metadata already contains graph table")
            provenance.write_collection_meta(
                target,
                metadata["runtime"],
                {**metadata["tables"], table_name: {"rows": len(rows), "status": "complete", "collections": [event]}},
            )
            updated = yaml.safe_load(meta_path.read_text())
            provenance.validate_collection_meta_for_update(updated)
            for old_name, old_entry in metadata["tables"].items():
                expected_entry = (
                    old_entry
                    if "collections" in old_entry
                    else {"rows": old_entry["rows"], "status": old_entry["status"], "collections": [old_entry]}
                )
                _require(updated["tables"][old_name] == expected_entry, "Existing collection history changed")
            changed.add(str(meta_path.relative_to(staged)))
            added.update((str(parquet.relative_to(staged)), str(profile_file.relative_to(staged))))
        after = _inventory(staged)
        _require(set(after) == set(before) | (added - {receipt_name}), "Unexpected staged publication file")
        _require(
            all(after[name] == pin for name, pin in before.items() if name not in changed), "Existing data changed"
        )
        receipt = {
            **result,
            "publisher_hash": module_hash,
            "publisher_identity": _record(IDENTITY_PATH),
            "evidence_files": identity["evidence_files"],
            "original_files": before,
            "published_files": after,
            "approval": identity["authority"]["approval"],
            "accuracy_acceptance": identity["authority"]["accuracy_acceptance"],
        }
        receipt_path = _relative(staged, receipt_name)
        receipt_path.parent.mkdir(parents=True)
        receipt_path.write_bytes(_bytes(receipt))
        _require(_inventory(base) == before, "Original systems changed during publication")
        _require(
            provenance.collector_hash(MODULE, package, closures) == module_hash, "Publisher changed during publication"
        )
        if output.exists():
            _require(
                _inventory(output) == _inventory(staged),
                "Existing publication differs; changed content under one identity is forbidden",
            )
            return {"status": "ALREADY_PUBLISHED", **result}
        helper._rename_noreplace(staged, output)
    return {"status": "PUBLISHED", **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence", "base-systems", "output-systems"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    print(json.dumps(publish(**vars(parser.parse_args())), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
