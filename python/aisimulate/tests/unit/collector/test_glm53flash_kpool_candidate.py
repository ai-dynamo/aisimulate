# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU integrity tests only; these fixtures are not native qualification data."""

import base64
import hashlib
import json
from pathlib import Path

import pytest

from collector.fpm_forward.runtime.glm53flash_vllm_kpool_candidate.qualification import probe
from collector.fpm_forward.runtime.glm53flash_vllm_kpool_candidate.qualification.probe import CASES
from collector.fpm_forward.runtime.glm53flash_vllm_kpool_candidate.qualification.validate import (
    bind_native_request_ids,
    validate_native,
)
from collector.fpm_forward.runtime.glm53flash_vllm_kpool_candidate.qualification.worker_probe import token_digest

pytestmark = pytest.mark.unit


def write(path, value):
    path.write_text(json.dumps(value))


def lines(path, value):
    path.write_text("".join(json.dumps(x) + "\n" for x in value))


@pytest.fixture
def evidence(tmp_path):
    expected = json.loads(Path(probe.__file__).with_name("expected-runtime.json").read_text())
    actual = {
        "version": expected["versions"]["candidate"],
        "loaded_files": {
            **expected["source_pins"],
            **expected["native_binaries"],
            expected["helper_path"]: expected["helper_sha256"]["candidate"],
        },
        "candidate_wheel_sha256": expected["candidate_wheel_sha256"],
    }
    write(tmp_path / "preflight.json", {"runtime": actual})
    outputs = []
    prompts = []
    rows = []
    for case, lengths in CASES.items():
        for rep in range(2):
            group = []
            for i, n in enumerate(lengths):
                rid = f"{case}-{rep}-{i}"
                tokens = [(j + i) % 1000 for j in range(n)]
                answer = list(range(1001, 1033))
                outputs.append(
                    {
                        "case": case,
                        "repetition": rep,
                        "item": i,
                        "request_id": rid,
                        "prompt_token_ids": tokens,
                        "output_token_ids": answer,
                    }
                )
                rid += "-012abcde"
                prompts.append({"request_id": rid, "prompt_token_ids": tokens, "prompt_sha256": token_digest(tokens)})
                group.append((rid, i, tokens, answer))
            chunks = [[0, 4097], [4097, min(n - 4097, 4097)]]
            if lengths[0] >= 8194:
                chunks.append([8194, 0])
            for step, (prefix, _) in enumerate(chunks):
                reqs = []
                for rid, i, tokens, answer in group:
                    q = min(4097, len(tokens) - prefix)
                    reqs.append(
                        {
                            "request_id": rid,
                            "prefix": prefix,
                            "query": q,
                            "native_slot": i,
                            "is_prefilling": True,
                            "prompt_length": len(tokens),
                            "prompt_sha256": token_digest(tokens),
                            "query_token_ids": tokens[prefix : prefix + q],
                            "sampled_token_id": answer[0] if prefix + q == len(tokens) else 999,
                        }
                    )
                rows.append(
                    {"requests": reqs, "native_mode": "NONE", "actual_padded_tokens": sum(x["query"] for x in reqs)}
                )
            for step in range(31):
                reqs = []
                for rid, i, tokens, answer in group:
                    reqs.append(
                        {
                            "request_id": rid,
                            "prefix": len(tokens) + step,
                            "query": 1,
                            "native_slot": i,
                            "is_prefilling": False,
                            "prompt_length": len(tokens),
                            "prompt_sha256": token_digest(tokens),
                            "query_token_ids": [answer[step]],
                            "sampled_token_id": answer[step + 1],
                        }
                    )
                rows.append({"requests": reqs, "native_mode": "FULL", "actual_padded_tokens": len(reqs)})
    lines(tmp_path / "outputs.jsonl", outputs)
    installed = []
    completed = []
    for rank in range(2):
        hardware = {"name": "NVIDIA GB300", "compute_capability": [10, 3], "uuid": f"gpu-{rank}"}
        settings = {
            "tp_size": 2,
            "ep_enabled": False,
            "async_scheduling": False,
            "prefix_caching": False,
            "mamba_cache_mode": "none",
            "kv_dtype": "fp8_e4m3",
            "max_model_len": 131079,
            "long_prefill_token_threshold": 4097,
            "max_num_batched_tokens": 16398,
            "speculative": False,
            "enforce_eager": False,
        }
        write(
            tmp_path / f"worker-rank-{rank}.json",
            {"tp_rank": rank, "hardware": hardware, "runtime": actual, "settings": settings},
        )
        lines(tmp_path / f"prompts-rank-{rank}.jsonl", prompts)
        lines(
            tmp_path / f"forward-rank-{rank}.jsonl",
            [
                {**row, "tp_rank": rank, "invocation": i, "gpu_completed": True, "model_execute_returned": True}
                for i, row in enumerate(rows, 1)
            ],
        )
        installed.append({"tp_rank": rank, "status": "installed_read_only_native_wrappers"})
        completed.append({"tp_rank": rank, "completed_native_forwards": len(rows), "requests": len(outputs)})
    write(tmp_path / "worker-installation.json", installed)
    write(tmp_path / "worker-completion.json", completed)
    return tmp_path


def test_complete_chain(evidence):
    assert validate_native(evidence, 2, "split", "production", "candidate")["requests"] == 20


@pytest.mark.parametrize(
    "failure", ["completion", "runtime", "helper", "sample_chain", "prefix", "native_slot", "graph", "gpu_uuid"]
)
def test_reject_corrupt_evidence(evidence, failure):
    if failure in ("runtime", "helper", "gpu_uuid"):
        path = evidence / "worker-rank-1.json"
        row = json.loads(path.read_text())
        if failure == "runtime":
            row["runtime"]["version"] = "0.30.0"
        if failure == "helper":
            row["runtime"]["loaded_files"]["vllm/model_executor/layers/sparse_attn_indexer_kpool.py"] = "0" * 64
        if failure == "gpu_uuid":
            row["hardware"]["uuid"] = "gpu-0"
        write(path, row)
    else:
        path = evidence / "forward-rank-0.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        if failure == "completion":
            rows[0]["gpu_completed"] = False
        if failure == "sample_chain":
            rows[2]["requests"][0]["query_token_ids"] = [999]
        if failure == "prefix":
            rows[1]["requests"][0]["prefix"] += 1
        if failure == "native_slot":
            rows[1]["requests"][0]["native_slot"] += 1
        if failure == "graph":
            for row in rows:
                row["native_mode"] = "NONE"
        lines(path, rows)
    with pytest.raises(ValueError):
        validate_native(evidence, 2, "split", "production", "candidate")


def test_reject_runtime_relabel(evidence):
    with pytest.raises(ValueError, match="runtime identity"):
        validate_native(evidence, 2, "split", "production", "stock")


def test_encoded_repair_keeps_original_candidate_identity():
    root = Path(probe.__file__).parents[1]
    original = base64.b64decode((root / "retained-tail-prefill.patch.b64").read_bytes().strip(), validate=True)
    identity = json.loads((root / "patch-identity.json").read_text())
    assert hashlib.sha256(original).hexdigest() == identity["patch_sha256"]
    assert original.startswith(b"--- a/vllm/model_executor/layers/sparse_attn_indexer_kpool.py\n")
    assert (root / "retained-tail-prefill.review.diff").read_text() == "\n".join(
        line.rstrip() for line in original.decode().splitlines()
    ) + "\n"


@pytest.mark.parametrize("suffix", ["", "-012ABCde", "-012abcd", "-012abcdee", "_012abcde", "-012abcdeg"])
def test_native_id_rejects_non_native_suffix(suffix):
    prompt = [1, 2, 3]
    with pytest.raises(ValueError, match="mapping"):
        bind_native_request_ids(
            [{"request_id": "a-b", "prompt_token_ids": prompt}],
            [{"request_id": "a-b" + suffix, "prompt_token_ids": prompt, "prompt_sha256": token_digest(prompt)}],
        )


def test_native_id_preserves_entire_external_id_and_duplicate_prompt_bytes():
    tokens = [7]
    outputs = [{"request_id": rid, "prompt_token_ids": tokens} for rid in ["a-b", "a-b-012abcde"]]
    prompts = [
        {
            "request_id": row["request_id"] + "-012abcde",
            "prompt_token_ids": tokens,
            "prompt_sha256": token_digest(tokens),
        }
        for row in outputs
    ]
    native, mapping = bind_native_request_ids(outputs, prompts)
    assert native["a-b-012abcde-012abcde"] is outputs[1]
    assert mapping == {"a-b": "a-b-012abcde", "a-b-012abcde": "a-b-012abcde-012abcde"}


@pytest.mark.parametrize("defect", ["extra", "duplicate_external", "duplicate_native", "wrong_tokens", "wrong_digest"])
def test_native_id_bijection_and_prompt_binding(defect):
    outputs = [{"request_id": "request", "prompt_token_ids": [7]}]
    prompts = [{"request_id": "request-012abcde", "prompt_token_ids": [7], "prompt_sha256": token_digest([7])}]
    if defect == "extra":
        prompts.append({**prompts[0], "request_id": "extra-012abcde"})
    elif defect == "duplicate_external":
        prompts.append({**prompts[0], "request_id": "request-123abcde"})
    elif defect == "duplicate_native":
        prompts.append(dict(prompts[0]))
    elif defect == "wrong_tokens":
        prompts[0]["prompt_token_ids"] = [8]
    else:
        prompts[0]["prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        bind_native_request_ids(outputs, prompts)


@pytest.mark.parametrize("defect", [None, "missing", "native_id", "returned", "digest", "source"])
def test_actual_native_assignment_must_bind_the_completed_worker_chain(evidence, defect):
    path = evidence / "preflight.json"
    preflight = json.loads(path.read_text())
    sources = json.loads(Path(probe.__file__).with_name("request-id-source.json").read_text())["sources"]
    preflight["request_identity_protocol"] = "native_assign_request_id_v1"
    preflight["request_identity_source_sha256"] = {item["path"]: item["sha256"] for item in sources}
    if defect == "source":
        preflight["request_identity_source_sha256"][sources[0]["path"]] = "0" * 64
    write(path, preflight)
    outputs = [json.loads(row) for row in (evidence / "outputs.jsonl").read_text().splitlines()]
    assigned = [
        {
            "external_request_id": row["request_id"],
            "native_request_id": row["request_id"] + "-012abcde",
            "prompt_sha256": token_digest(row["prompt_token_ids"]),
            "original_assignment_returned": True,
        }
        for row in outputs
    ]
    if defect == "native_id":
        assigned[0]["native_request_id"] += "bad"
    elif defect == "returned":
        del assigned[0]["original_assignment_returned"]
    elif defect == "digest":
        assigned[0]["prompt_sha256"] = "0" * 64
    if defect != "missing":
        lines(evidence / "request-id-map.jsonl", assigned)
    if defect:
        with pytest.raises((ValueError, FileNotFoundError)):
            validate_native(evidence, 2, "split", "production", "candidate")
    else:
        assert (
            validate_native(evidence, 2, "split", "production", "candidate")["request_identity_protocol"]
            == "native_assign_request_id_v1"
        )


def test_assignment_witness_calls_original_and_preserves_its_result(tmp_path):
    from types import SimpleNamespace

    from collector.fpm_forward.runtime.glm53flash_vllm_kpool_candidate.qualification.worker_probe import (
        install_request_id_witness,
    )

    def original(request):
        request.external_req_id = request.request_id
        request.request_id += "-012abcde"
        return "native-return-value"

    processor = SimpleNamespace(assign_request_id=original)
    request = SimpleNamespace(request_id="complete-original-id", prompt_token_ids=[1, 3])
    install_request_id_witness(processor, tmp_path)
    assert processor.assign_request_id(request) == "native-return-value"
    row = json.loads((tmp_path / "request-id-map.jsonl").read_text())
    assert row == {
        "external_request_id": "complete-original-id",
        "native_request_id": "complete-original-id-012abcde",
        "prompt_sha256": token_digest([1, 3]),
        "original_assignment_returned": True,
    }
