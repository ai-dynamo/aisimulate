# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind an independent SG graph control to original native request inputs.

Original integration with sgl-project/sglang at
94602c9c2b7cbdb8efd5c52802dac6a1c180089e (Apache-2.0), source paths below.
Uses public Engine.generate sampling dictionaries; no native code copied.
See THIRD_PARTY_NOTICES.md. A finite bias is not a selection guarantee.
"""

from __future__ import annotations

import inspect
import json
import math
import re
from pathlib import Path

from collector.glm53flash_contract import canonical_json, sha256_json
from collector.glm53flash_jsonl import file_sha256, iter_records

SCHEMA = "glm53flash_sglang_control_inputs_v1"
SUBMISSION = "glm53flash_sglang_request_submission_v1"
BIAS = 100.0
REFERENCE = "sglang-control-reference.json"
PRODUCER = "sglang-input-producer.json"
JOURNAL = "sglang-request-inputs.jsonl"
SOURCE_PINS = {
    "srt/entrypoints/engine.py": "7d79f61f6171902d60dcece76ce6ee16e11c4dae6e9e98e620941be92effc6b0",
    "srt/model_executor/model_runner.py": "df38e26ace5deae708f2353b56962afab7f92fac35d1bd4f274b7d6d62356372",
    "srt/layers/sampler.py": "7fbd8ee623b472d0b11a31854e123229fb984fd11ffdd87a7724c44a57178d92",
    "srt/sampling/sampling_params.py": "7295f275a1a286ae5a4414bcedd1d48a3a9824a0c2a5c915e091ed06b6f74730",
    "srt/sampling/sampling_batch_info.py": "67813f6b76a31768c8530904a0b58c987d21532751ba734c53099b0fee920b72",
}

METHOD_SOURCES = {
    "ModelRunner.sample": "srt/model_executor/model_runner.py",
    "ModelRunner._preprocess_logits": "srt/model_executor/model_runner.py",
    "Sampler.forward": "srt/layers/sampler.py",
}
REQUIRED_COLLECTOR_SOURCES = {
    "glm53flash_sglang_control.py",
    "glm53flash_sglang_runtime.py",
    "glm53flash_sglang_graph_ops.py",
    "fpm_forward/sglang_driver.py",
}


def _hashes(value, required=()):
    return (
        isinstance(value, dict)
        and bool(value)
        and set(required).issubset(value)
        and all(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest)
            for name, digest in value.items()
        )
    )


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(canonical_json(value) + "\n")


def sampling_sources():
    import sglang

    root = Path(sglang.__file__).resolve().parent
    actual = {name: file_sha256(root / name) for name in SOURCE_PINS}
    if actual != SOURCE_PINS:
        raise RuntimeError("native input control sampling source changed")
    return root, actual


def producer_identity():
    _, native = sampling_sources()
    root = Path(__file__).resolve().parent
    # Full collector Python closure is independent of installation prefix and
    # records actual bytes. The external payload separately binds commit/wheel;
    # never pretend a source hash proves a Git revision or wheel origin.
    return {
        "schema": SUBMISSION,
        "collector_sources": {str(path.relative_to(root)): file_sha256(path) for path in sorted(root.rglob("*.py"))},
        "native_sampling_sources": native,
    }


def worker_identity(runner):
    root, pins = sampling_sources()
    methods = {}
    for name, method, source in (
        ("ModelRunner.sample", runner.sample, "srt/model_executor/model_runner.py"),
        ("ModelRunner._preprocess_logits", runner._preprocess_logits, "srt/model_executor/model_runner.py"),
        ("Sampler.forward", runner.sampler.forward, "srt/layers/sampler.py"),
    ):
        function = getattr(method, "__func__", method)
        if (
            getattr(function, "__wrapped__", None) is not None
            or not inspect.isfunction(function)
            or Path(function.__code__.co_filename).resolve() != root / source
            or function.__qualname__ != name
        ):
            raise RuntimeError("native input control has another loaded sampling callable")
        methods[name] = {
            "source": source,
            "source_sha256": pins[source],
            "signature": str(inspect.signature(method)),
            "qualname": function.__qualname__,
            "wrapped": False,
        }
    return {"schema": SUBMISSION, "native_sampling_sources": pins, "methods": methods}


def validate_scope(purpose, phase, dataset_role, path, digest):
    if bool(path) != bool(digest):
        raise ValueError("native input control requires both original reference path and SHA")
    if path and (purpose, phase, dataset_role) != ("ops_graph_holdout", "decode", "calibration"):
        raise ValueError("native input control is restricted to independent SG graph calibration control")


def _token_list(value):
    return isinstance(value, list) and value and all(type(item) is int and item >= 0 for item in value)


def load_reference(path, digest, producer):
    path = Path(path)
    if file_sha256(path) != digest:
        raise ValueError("native control reference original bytes changed")
    value = json.loads(path.read_bytes())
    if (
        value.get("schema") != SCHEMA
        or type(value.get("logit_bias")) not in (int, float)
        or not math.isfinite(value["logit_bias"])
        or value["logit_bias"] != BIAS
        or canonical_json(value.get("producer")) != canonical_json(producer)
        or not _hashes(value.get("source_files"))
        or type(value.get("tp_size")) is not int
        or value["tp_size"] not in (2, 4)
        or not isinstance(value.get("source_run_id"), str)
        or not value["source_run_id"]
        or not isinstance(value.get("targets"), list)
        or not value["targets"]
    ):
        raise ValueError("native control reference has unqualified source or experimental bias")
    seen, requests = set(), set()
    for target in value["targets"]:
        key = target.get("benchmark_id"), target.get("repetition")
        items = target.get("requests")
        if (
            any(type(item) is not int for item in key)
            or key in seen
            or key[0] < 1
            or not 0 <= key[1] < 15
            or not isinstance(items, list)
            or not 1 <= len(items) <= 32
        ):
            raise ValueError("native control reference repeats or omits a point/repetition")
        seen.add(key)
        forwards = target.get("source_forwards")
        if (
            not isinstance(forwards, list)
            or len(forwards) != value["tp_size"]
            or any(
                type(row.get("rank")) is not int
                or row["rank"] != rank
                or type(row.get("invocation")) is not int
                or row["invocation"] < 0
                or not isinstance(row.get("forward_id"), str)
                or not row["forward_id"]
                for rank, row in enumerate(forwards)
            )
        ):
            raise ValueError("native control reference omits exact source TP forwards")
        for position, request in enumerate(items):
            rid = request.get("source_request_id")
            prompt, query = request.get("prompt_token_ids"), request.get("native_query_token_ids")
            if (
                type(request.get("submitted_position")) is not int
                or request["submitted_position"] != position
                or not isinstance(rid, str)
                or not rid
                or rid in requests
                or not _token_list(prompt)
                or not _token_list(query)
                or len(query) != 1
                or request.get("prompt_sha256") != sha256_json(prompt)
                or request.get("input_tokens_sha256") != sha256_json(prompt + query)
                or type(request.get("computed_tokens_before")) is not int
                or request["computed_tokens_before"] != len(prompt)
                or type(request.get("computed_tokens_after")) is not int
                or request["computed_tokens_after"] != len(prompt) + 1
            ):
                raise ValueError("native control reference lacks exact original prompt/query/history")
            requests.add(rid)
    if any({rep for bid, rep in seen if bid == point} != set(range(15)) for point, _ in seen):
        raise ValueError("native control reference requires all original 5+10 repetitions")
    return value


def sampling_parameters(reference, benchmark_id, repetition, inputs):
    base = {"temperature": 0, "max_new_tokens": 2, "ignore_eos": True}
    if reference is None:
        return base
    rows = [
        row for row in reference["targets"] if (row["benchmark_id"], row["repetition"]) == (benchmark_id, repetition)
    ]
    if len(rows) != 1 or inputs != [row["prompt_token_ids"] for row in rows[0]["requests"]]:
        raise ValueError("native control public request changed original submitted prompt/order")
    return [{**base, "logit_bias": {str(row["native_query_token_ids"][0]): BIAS}} for row in rows[0]["requests"]]


def append_submission(root, benchmark_id, repetition, request_ids, inputs, params, reference_sha256):
    if len(request_ids) != len(inputs) or len(set(request_ids)) != len(request_ids) or not inputs:
        raise ValueError("native public request submission is incomplete")
    write = {
        "schema": SUBMISSION,
        "benchmark_id": benchmark_id,
        "repetition": repetition,
        "request_ids": request_ids,
        "input_ids": inputs,
        "sampling_params": params,
        "control_reference_sha256": reference_sha256,
    }
    with (Path(root) / JOURNAL).open("a") as stream:
        stream.write(canonical_json(write) + "\n")


def read_submission(root, run, proof, files):
    """Read optional new producer evidence without relabeling old artifacts."""
    from collector.glm53flash_graph_export import _local

    root = Path(root)
    provenance = json.loads(_local(root, "sglang-provenance.json").read_bytes())
    declared = provenance.get("native_request_submission")
    any_new = any((root / name).exists() for name in (PRODUCER, JOURNAL, REFERENCE)) or any(
        root.glob("sampling-source-rank-*.json")
    )
    if not any_new and declared is None:
        return None
    if declared != SUBMISSION:
        raise ValueError("native request evidence was not declared by its original producer")
    files.update((PRODUCER, JOURNAL))
    producer = json.loads(_local(root, PRODUCER).read_bytes())
    if (
        producer.get("schema") != SUBMISSION
        or producer.get("native_sampling_sources") != SOURCE_PINS
        or not _hashes(producer.get("collector_sources"), REQUIRED_COLLECTOR_SOURCES)
    ):
        raise ValueError("native request source identity is incomplete")
    reference = None
    reference_digest = provenance.get("native_control_reference_sha256")
    if "native_control_reference_sha256" not in provenance or (
        (root / REFERENCE).exists() != (reference_digest is not None)
    ):
        raise ValueError("native control reference differs from original sampling declaration")
    if (root / REFERENCE).exists():
        if run["role"] != "control":
            raise ValueError("calibration and holdout cannot apply native control sampling")
        files.add(REFERENCE)
        reference = load_reference(_local(root, REFERENCE), reference_digest, producer)
    workers = []
    for rank in range(run["key"][2]):
        name = f"sampling-source-rank-{rank}.json"
        files.add(name)
        value = json.loads(_local(root, name).read_bytes())
        methods = value.get("methods", {})
        if (
            value.get("schema") != SUBMISSION
            or value.get("native_sampling_sources") != SOURCE_PINS
            or set(methods) != set(METHOD_SOURCES)
            or any(
                item.get("source") != METHOD_SOURCES[name]
                or item.get("source_sha256") != SOURCE_PINS[METHOD_SOURCES[name]]
                or item.get("qualname") != name
                or item.get("wrapped") is not False
                or not isinstance(item.get("signature"), str)
                or not item["signature"]
                for name, item in methods.items()
            )
        ):
            raise ValueError("native control lacks loaded worker sampling method evidence")
        workers.append(value)
    if any(value != workers[0] for value in workers):
        raise ValueError("native control workers loaded different sampling callables")
    targets = {}
    # The common reader owns complete state chains. This pass additionally
    # joins the real public submission to the actual per-rank model inputs.
    actual = {}
    for rank in range(run["key"][2]):
        seen = set()
        for row in iter_records(_local(root, f"forward-rank-{rank}.jsonl")):
            if row.get("stage") != "measure":
                continue
            if (
                row.get("native_request_submission") != SUBMISSION
                or "native_control_reference_sha256" not in row
                or row["native_control_reference_sha256"] != reference_digest
            ):
                raise ValueError("old native forwards cannot acquire a new public submission witness")
            key = row["benchmark_id"], row["repetition"]
            if key in seen or key not in proof["forwards"]:
                raise ValueError("native public submission has duplicate or unexpected actual targets")
            seen.add(key)
            items = {r["request_id"]: r for r in row["requests"]}
            if len(items) != len(row["requests"]):
                raise ValueError("native request identity reused in a forward")
            signature = {
                rid: {
                    k: r[k]
                    for k in (
                        "prompt_token_ids",
                        "native_query_token_ids",
                        "input_tokens_sha256",
                        "computed_tokens_before",
                        "computed_tokens_after",
                    )
                }
                for rid, r in items.items()
            }
            if key in actual and actual[key] != signature:
                raise ValueError("native submission differs across actual TP ranks")
            actual[key] = signature
        if seen != proof["forwards"].keys():
            raise ValueError("native public submission omits a TP target")
    for row in iter_records(_local(root, JOURNAL)):
        key = row.get("benchmark_id"), row.get("repetition")
        ids, inputs = row.get("request_ids"), row.get("input_ids")
        if (
            row.get("schema") != SUBMISSION
            or key in targets
            or key not in proof["forwards"]
            or not isinstance(ids, list)
            or len(set(ids)) != len(ids)
            or not isinstance(inputs, list)
            or len(ids) != len(inputs)
            or set(ids) != set(actual.get(key, {}))
        ):
            raise ValueError("native public submission has another or duplicate cohort")
        params = sampling_parameters(reference, *key, inputs)
        digest = file_sha256(root / REFERENCE) if reference is not None else None
        if (
            canonical_json(row.get("sampling_params")) != canonical_json(params)
            or row.get("control_reference_sha256") != digest
        ):
            raise ValueError("native public sampling arguments differ from fixed source-bound control")
        items = []
        for position, (rid, prompt) in enumerate(zip(ids, inputs, strict=True)):
            witness = actual[key][rid]
            if prompt != witness["prompt_token_ids"]:
                raise ValueError("native model prompt differs from actual public input")
            items.append(
                {
                    "submitted_position": position,
                    "source_request_id": rid,
                    **witness,
                    "prompt_sha256": sha256_json(prompt),
                }
            )
        if reference is not None:
            expected = next(r["requests"] for r in reference["targets"] if (r["benchmark_id"], r["repetition"]) == key)
            for observed, source in zip(items, expected, strict=True):
                if (
                    any(observed[k] != source[k] for k in observed if k != "source_request_id")
                    or observed["source_request_id"] == source["source_request_id"]
                ):
                    raise ValueError("native finite bias did not reproduce the exact independent model input")
        targets[key] = {
            "benchmark_id": key[0],
            "repetition": key[1],
            "requests": items,
            "source_forwards": [
                {"rank": rank, "forward_id": row["forward_id"], "invocation": row["invocation"]}
                for rank, row in sorted(proof["forwards"][key].items())
            ],
        }
    if targets.keys() != proof["forwards"].keys():
        raise ValueError("native public request journal omits frozen forwards")
    return {
        "producer": producer,
        "worker": workers[0],
        "targets": [targets[key] for key in sorted(targets)],
        "reference": reference,
    }


def reference_from_proof(root, proof):
    submission = proof.get("request_submission")
    if (
        submission is None
        or submission["reference"] is not None
        or any(
            row["dataset_role"] != "calibration" or row["binding"] is None
            for ranks in proof["forwards"].values()
            for row in ranks.values()
        )
    ):
        raise ValueError("control reference requires a fresh complete original calibration")
    return {
        "schema": SCHEMA,
        "logit_bias": BIAS,
        "producer": submission["producer"],
        "worker": submission["worker"],
        "tp_size": len(next(iter(proof["forwards"].values()))),
        "source_run_id": next(iter(proof["forwards"].values()))[0]["request_set"],
        "source_plan_sha256": proof["source_plan_sha256"],
        "source_files": {name: file_sha256(Path(root) / name) for name in sorted(proof["files"])},
        "policy": {
            "snapshot": proof["native_snapshot"],
            "provenance": proof["provenance"],
            "execution_policy": proof["execution_policy"],
        },
        "corpus_sha256": proof["corpus_sha256"],
        "targets": submission["targets"],
    }


def freeze_reference(root, run, destination):
    from collector.glm53flash_graph_export import read_graph_run
    from collector.glm53flash_validation import _load_native

    if run["role"] != "calibration":
        raise ValueError("only original calibration can define a native input control")
    _load_native(run, Path(root), calibration_evidence=False)
    proof = read_graph_run(Path(root), run)
    value = reference_from_proof(root, proof)
    write_new(destination, value)
    return file_sha256(destination)


def verify_pair(root, proof, control):
    observed = control.get("request_submission")
    calibration = proof.get("request_submission")
    if observed is None and calibration is None:
        return
    if (
        observed is None
        or calibration is None
        or observed["worker"] != calibration["worker"]
        or observed["producer"] != calibration["producer"]
    ):
        raise ValueError("independent native control changes actual producer/sampling identity")
    if observed["reference"] is None:
        return
    expected = reference_from_proof(root, proof)
    if (
        canonical_json(observed["reference"]) != canonical_json(expected)
        or observed["worker"] != expected["worker"]
        or observed["producer"] != expected["producer"]
    ):
        raise ValueError("independent native control reference belongs to another source calibration")
