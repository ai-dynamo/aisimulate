# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify actual worker completion and compare ordinary Engine token outputs."""

import argparse
import hashlib
import json
import re
from pathlib import Path

if __package__:
    from .worker_probe import digest, token_digest
else:
    from worker_probe import digest, token_digest


def records(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def bind_native_request_ids(outputs, prompts):
    """Validate the pinned native external/internal bijection without rewriting raw.

    vLLM ced6857: v1/engine/input_processor.py:262-279 appends '-' and the
    first eight lowercase hexadecimal characters of utils.random_uuid();
    v1/engine/output_processor.py:384 returns the original external ID.
    Prompt bytes and their digest bind each pair. validate_native subsequently
    checks every completed worker token/sample chain against that same output.
    """
    external = {item["request_id"]: item for item in outputs}
    if len(external) != len(outputs) or any(type(key) is not str or not key for key in external):
        raise ValueError("external request IDs must be unique nonempty strings")
    native = {}
    mapping = {}
    for prompt in prompts:
        rid = prompt["request_id"]
        if type(rid) is not str or rid in native:
            raise ValueError("native request IDs must be unique strings")
        matches = [key for key in external if re.fullmatch(re.escape(key) + r"-[0-9a-f]{8}", rid)]
        if len(matches) != 1 or matches[0] in mapping:
            raise ValueError("native request ID does not form the pinned unique external/internal mapping")
        key = matches[0]
        out = external[key]
        if (
            prompt["prompt_token_ids"] != out["prompt_token_ids"]
            or token_digest(prompt["prompt_token_ids"]) != prompt["prompt_sha256"]
        ):
            raise ValueError("worker prompt bytes differ from actual request API")
        native[rid] = out
        mapping[key] = rid
    if set(mapping) != set(external):
        raise ValueError("worker prompt identity coverage differs")
    return native, mapping


def validate_cohort_admission(root, preflight, outputs, mapping, tp):
    protocol = preflight.get("cohort_admission_protocol")
    if protocol is None:
        return "legacy_generate_sequential_admission"
    sources = json.loads(Path(__file__).with_name("cohort-source.json").read_text())["sources"]
    if protocol != "native_scheduling_pause_enqueue_v1" or preflight.get("cohort_admission_source_sha256") != {
        row["path"]: row["sha256"] for row in sources
    }:
        raise ValueError("native cohort admission protocol/source differs")
    groups = {}
    for row in outputs:
        groups.setdefault((row["case"], row["repetition"]), []).append(row)
    events = list(records(root / "cohort-admission.jsonl"))
    if len(events) != len(groups) * 4:
        raise ValueError("native cohort admission receipt is incomplete")
    for index, (key, group) in enumerate(groups.items()):
        selected = events[index * 4 : index * 4 + 4]
        group = sorted(group, key=lambda row: row["item"])
        native_ids = [mapping[row["request_id"]] for row in group]
        if (
            [(row["case"], row["repetition"]) for row in selected] != [key] * 4
            or [row["event"] for row in selected]
            != ["scheduling_paused", "cohort_enqueued", "scheduling_resumed", "cohort_completed"]
            or selected[0].get("level") != 0
            or selected[0].get("mode") != "keep"
            or selected[1].get("native_request_ids") != native_ids
            or selected[1].get("prompt_sha256") != [token_digest(row["prompt_token_ids"]) for row in group]
            or selected[2].get("tags") != ["scheduling"]
            or selected[3].get("external_request_ids") != [row["request_id"] for row in group]
        ):
            raise ValueError("native cohort admission differs from actual completed requests")
        for rank in range(tp):
            initial = [
                {req["request_id"] for req in row["requests"] if req["prefix"] == 0}
                for row in records(root / f"forward-rank-{rank}.jsonl")
                if any(req["request_id"] in native_ids and req["prefix"] == 0 for req in row["requests"])
            ]
            if initial != [set(native_ids)]:
                raise ValueError("native initial cohort was not one completed worker batch")
    return protocol


def validate_checkpoint_identity(preflight, expected_runtime, checkpoint=None):
    """Bind the observed config and both loader revisions to the model label.

    The native probe hashes the actual mounted config before constructing the
    Engine. Recheck those original bytes' identity when reading evidence: three
    matching receipt labels alone cannot establish a checkpoint's precision.
    """
    matches = [
        name
        for name, pin in expected_runtime["checkpoints"].items()
        if pin["config_sha256"] == preflight.get("checkpoint_config_sha256")
    ]
    if len(matches) != 1 or (checkpoint is not None and matches[0] != checkpoint):
        raise ValueError("native checkpoint config does not match the qualification label")
    name = matches[0]
    pin = expected_runtime["checkpoints"][name]
    args = preflight.get("public_engine_args")
    if not isinstance(args, dict) or any(
        args.get(field) != pin["revision"] for field in ("revision", "tokenizer_revision")
    ):
        raise ValueError("native checkpoint/tokenizer revision differs from its immutable pin")
    return {"checkpoint": name, **pin}


def validate_native(root, tp, mode, policy, runtime_kind=None, *, checkpoint=None):
    if __package__:
        from .probe import CASES
    else:
        from probe import CASES

    outputs = list(records(root / "outputs.jsonl"))
    by_id = {x["request_id"]: x for x in outputs}
    expected = {(case, rep, i) for case, lengths in CASES.items() for rep in range(2) for i in range(len(lengths))}
    keys = [(x["case"], x["repetition"], x["item"]) for x in outputs]
    if set(keys) != expected or len(keys) != len(expected) or len(by_id) != len(keys):
        raise ValueError("native final requests do not exactly cover frozen cases and repetitions")
    expected_runtime = json.loads(Path(__file__).with_name("expected-runtime.json").read_text())
    preflight = json.loads((root / "preflight.json").read_text())
    checkpoint_identity = validate_checkpoint_identity(preflight, expected_runtime, checkpoint)
    if runtime_kind is None:
        runtime_kind = next(
            (
                key
                for key, version in expected_runtime["versions"].items()
                if version == preflight["runtime"]["version"]
            ),
            None,
        )
    if (
        runtime_kind not in expected_runtime["versions"]
        or preflight["runtime"]["version"] != expected_runtime["versions"][runtime_kind]
    ):
        raise ValueError("requested runtime identity differs from actual preflight")
    installed = json.loads((root / "worker-installation.json").read_text())
    completed = json.loads((root / "worker-completion.json").read_text())
    if (
        {x["tp_rank"] for x in installed} != set(range(tp))
        or len(installed) != tp
        or {x["tp_rank"] for x in completed} != set(range(tp))
        or len(completed) != tp
    ):
        raise ValueError("native installation/completion does not cover every TP worker exactly")
    modes = set()
    reference = None
    all_splits = {}
    uuids = set()
    request_id_mapping = None
    for rank in range(tp):
        worker = json.loads((root / f"worker-rank-{rank}.json").read_text())
        if (
            worker["tp_rank"] != rank
            or worker["hardware"]["compute_capability"] != [10, 3]
            or "GB300" not in worker["hardware"]["name"].upper()
        ):
            raise ValueError("native rank/GPU proof differs")
        actual = worker["runtime"]
        versions = expected_runtime["versions"]
        matching = [key for key, value in versions.items() if value == actual["version"]]
        if len(matching) != 1:
            raise ValueError("native runtime version is not the frozen stock/candidate")
        runtime = matching[0]
        if runtime != runtime_kind or actual != preflight["runtime"]:
            raise ValueError("native worker runtime differs from parent preflight/runtime label")
        uuid = worker["hardware"].get("uuid")
        if uuid and uuid in uuids:
            raise ValueError("TP workers share a physical GPU UUID")
        if uuid:
            uuids.add(uuid)
        wanted_files = {
            **expected_runtime["source_pins"],
            **expected_runtime["native_binaries"],
            expected_runtime["helper_path"]: expected_runtime["helper_sha256"][runtime],
        }
        if actual["loaded_files"] != wanted_files or actual.get("candidate_wheel_sha256") != (
            expected_runtime["candidate_wheel_sha256"] if runtime == "candidate" else None
        ):
            raise ValueError("native package/helper/binary source closure differs")
        settings = {
            "tp_size": tp,
            "ep_enabled": False,
            "async_scheduling": False,
            "prefix_caching": False,
            "mamba_cache_mode": "none",
            "kv_dtype": "fp8_e4m3",
            "max_model_len": 131079,
            "long_prefill_token_threshold": 4097 if mode == "split" else 0,
            "max_num_batched_tokens": 16398,
            "speculative": False,
            "enforce_eager": policy == "eager",
        }
        if worker.get("settings") != settings:
            raise ValueError("actual native runtime settings differ from frozen public arguments")
        prompts = list(records(root / f"prompts-rank-{rank}.jsonl"))
        by_id, mapping = bind_native_request_ids(outputs, prompts)
        if request_id_mapping is not None and mapping != request_id_mapping:
            raise ValueError("TP workers disagree on external/internal request identity")
        request_id_mapping = mapping
        histories = {}
        sampled = {}
        slots = {}
        splits = {rid: [] for rid in by_id}
        canonical = []
        batch_targets = {}
        for invocation, row in enumerate(records(root / f"forward-rank-{rank}.jsonl"), start=1):
            if (
                row["invocation"] != invocation
                or row["tp_rank"] != rank
                or row.get("gpu_completed") is not True
                or row.get("model_execute_returned") is not True
            ):
                raise ValueError("native forward lacks ordered rank/GPU completion")
            requests = row["requests"]
            if len({x["request_id"] for x in requests}) != len(requests) or len(
                {x["native_slot"] for x in requests}
            ) != len(requests):
                raise ValueError("native request or state slot aliases within batch")
            modes.add(row["native_mode"])
            normalized = []
            for request in requests:
                rid = request["request_id"]
                out = by_id[rid]
                prompt = out["prompt_token_ids"]
                p = request["prefix"]
                q = request["query"]
                tokens = request["query_token_ids"]
                previous = histories.get(rid, [])
                if (
                    p != len(previous)
                    or len(tokens) != q
                    or q < 1
                    or request["prompt_length"] != len(prompt)
                    or request["prompt_sha256"] != token_digest(prompt)
                ):
                    raise ValueError("same-request native prefix/query chain is incomplete")
                if rid in slots and slots[rid] != request["native_slot"]:
                    raise ValueError("native request changed live state slot")
                slots[rid] = request["native_slot"]
                if p < len(prompt):
                    if request["is_prefilling"] is not True or tokens != prompt[p : p + q]:
                        raise ValueError("prefill used tokens outside original request")
                    splits[rid].append([p, q])
                    if p == 4097:
                        batch_targets.setdefault((out["case"], out["repetition"], row["invocation"]), []).append(
                            (out["item"], q)
                        )
                else:
                    if (
                        request["is_prefilling"] is not False
                        or q != 1
                        or not sampled.get(rid)
                        or tokens != sampled[rid][-1:]
                    ):
                        raise ValueError("decode did not consume preceding native sampled token")
                histories[rid] = previous + tokens
                value = request["sampled_token_id"]
                if type(value) is not int or value < 0:
                    raise ValueError("missing native sampler result")
                if p + q >= len(prompt):
                    sampled.setdefault(rid, []).append(value)
                normalized.append({k: v for k, v in request.items() if k != "native_slot"})
            canonical.append(
                {
                    "requests": normalized,
                    "native_mode": row["native_mode"],
                    "actual_padded_tokens": row["actual_padded_tokens"],
                }
            )
        completion = next(x for x in completed if x["tp_rank"] == rank)
        if completion["completed_native_forwards"] != len(canonical) or completion["requests"] != len(by_id):
            raise ValueError("native completion RPC differs from actual trace coverage")
        if set(histories) != set(by_id):
            raise ValueError("native workers missed actual completed requests")
        for rid, out in by_id.items():
            n = len(out["prompt_token_ids"])
            if sampled.get(rid) != out["output_token_ids"] or len(sampled[rid]) != 32 or len(histories[rid]) != n + 31:
                raise ValueError("native sampled tokens/computed history disagree with final Engine output")
            wanted = (
                [[0, n]]
                if mode == "reference"
                else ([[0, 4097], [4097, n - 4097]] if n < 8194 else [[0, 4097], [4097, 4097], [8194, n - 8194]])
            )
            if splits[rid] != wanted:
                raise ValueError(f"actual native prefill partition differs: {rid}: {splits[rid]} != {wanted}")
        if mode == "split":
            for case in ("heterogeneous_b2", "four_tails_b4"):
                for rep in range(2):
                    groups = [sorted(rows) for (c, r, _), rows in batch_targets.items() if (c, r) == (case, rep)]
                    want = [(i, n - 4097) for i, n in enumerate(CASES[case])]
                    if groups != [want]:
                        raise ValueError("required distinct-request/tail target was not one actual native batch")
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        signature = hashlib.sha256(encoded).hexdigest()
        if reference is not None and signature != reference:
            raise ValueError("native TP workers disagree on actual token/phase/sample traces")
        reference = signature
        all_splits[str(rank)] = splits
    identity_protocol = preflight.get("request_identity_protocol")
    if identity_protocol is not None:
        sources = json.loads(Path(__file__).with_name("request-id-source.json").read_text())["sources"]
        if identity_protocol != "native_assign_request_id_v1" or preflight.get("request_identity_source_sha256") != {
            item["path"]: item["sha256"] for item in sources
        }:
            raise ValueError("native request identity protocol/source closure differs")
        assigned = list(records(root / "request-id-map.jsonl"))
        if len(assigned) != len(outputs):
            raise ValueError("native request assignment witness is incomplete")
        output_ids = {item["request_id"]: item for item in outputs}
        witnessed = {}
        for row in assigned:
            external = row["external_request_id"]
            if (
                external in witnessed
                or row.get("original_assignment_returned") is not True
                or external not in output_ids
                or row["native_request_id"] != request_id_mapping.get(external)
                or row["prompt_sha256"] != token_digest(output_ids[external]["prompt_token_ids"])
            ):
                raise ValueError("actual native request assignment disagrees with completed worker chain")
            witnessed[external] = row["native_request_id"]
        if witnessed != request_id_mapping:
            raise ValueError("actual native request assignment does not cover the worker bijection")
    if policy == "eager" and modes != {"NONE"}:
        raise ValueError("requested eager policy encountered graph dispatch")
    if policy == "production" and "FULL" not in modes:
        raise ValueError("production correctness probe did not execute native full decode graph")
    for case, lengths in CASES.items():
        for i in range(len(lengths)):
            rows = [x for x in outputs if x["case"] == case and x["item"] == i]
            if (
                rows[0]["prompt_token_ids"] != rows[1]["prompt_token_ids"]
                or rows[0]["output_token_ids"] != rows[1]["output_token_ids"]
            ):
                raise ValueError("greedy outputs changed across repeated identical real requests")
    cohort_protocol = validate_cohort_admission(root, preflight, outputs, request_id_mapping, tp)
    return {
        "requests": len(outputs),
        "checkpoint_identity": checkpoint_identity,
        "native_modes": sorted(modes),
        "all_tp_trace_digest": reference,
        "external_to_native_request_ids": request_id_mapping,
        "request_identity_protocol": identity_protocol or "pinned_source_offline_bijection",
        "cohort_admission_protocol": cohort_protocol,
        "actual_prefill_splits": all_splits,
        "files": [{"path": p.name, "sha256": digest(p)} for p in sorted(root.iterdir()) if p.is_file()],
    }


def compare(stock, candidate, split, out):
    roots = [Path(stock), Path(candidate), Path(split)]
    receipts = [json.loads((p / "native-receipt.json").read_text()) for p in roots]
    scope = {(r["checkpoint"], r["tp"], r["policy"]) for r in receipts}
    if len(scope) != 1 or [(r["runtime_kind"], r["mode"]) for r in receipts] != [
        ("stock", "reference"),
        ("candidate", "reference"),
        ("candidate", "split"),
    ]:
        raise ValueError("comparison runtime/mode/cell identities differ")
    checkpoint, tp, policy = scope.pop()
    evidence = []
    for root, r in zip(roots, receipts, strict=True):
        evidence.append(validate_native(root, tp, r["mode"], policy, r["runtime_kind"], checkpoint=checkpoint))
    if len({item["cohort_admission_protocol"] for item in evidence}) != 1:
        raise ValueError("comparison changes the native cohort admission policy")
    maps = [{(x["case"], x["repetition"], x["item"]): x for x in records(p / "outputs.jsonl")} for p in roots]
    differences = []
    for key in maps[0]:
        a, b, c = [m[key] for m in maps]
        if not (a["prompt_token_ids"] == b["prompt_token_ids"] == c["prompt_token_ids"]):
            raise ValueError("comparison changes original input token bytes")
        for name, left, right in [("stock_vs_candidate_oneshot", a, b), ("candidate_oneshot_vs_split", b, c)]:
            if left["output_token_ids"] != right["output_token_ids"]:
                first = next(
                    i
                    for i, (x, y) in enumerate(zip(left["output_token_ids"], right["output_token_ids"], strict=True))
                    if x != y
                )
                differences.append(
                    {
                        "case": key,
                        "comparison": name,
                        "first_differing_token": first,
                        "left": left["output_token_ids"][first],
                        "right": right["output_token_ids"][first],
                    }
                )
    result = {
        "status": "passed" if not differences else "failed",
        "scope": "native_Engine_functional_correctness_for_frozen_geometry_suite_only",
        "checkpoint": checkpoint,
        "tp": tp,
        "policy": policy,
        "requests_per_mode": len(maps[0]),
        "greedy_output_tokens_per_request": 32,
        "comparisons": 2 * len(maps[0]),
        "differences": differences,
        "native_receipts": evidence,
        "accuracy_acceptance": "NOT_EVALUATED",
        "formal_8_cell_coverage": "NOT_EVALUATED",
    }
    with Path(out).open("x") as f:
        json.dump(result, f, indent=2)
    if differences:
        raise ValueError("full native Engine exact-token comparison failed; differences preserved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stock-reference", required=True)
    p.add_argument("--candidate-reference", required=True)
    p.add_argument("--candidate-split", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    compare(a.stock_reference, a.candidate_reference, a.candidate_split, a.output)
