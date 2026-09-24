# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU integrity tests only; these fixtures are not native qualification data."""

import json
from pathlib import Path

import pytest
from collector.fpm_forward.runtime.glm53flash_vllm_kpool_candidate.qualification import probe
from collector.fpm_forward.runtime.glm53flash_vllm_kpool_candidate.qualification.probe import CASES
from collector.fpm_forward.runtime.glm53flash_vllm_kpool_candidate.qualification.validate import validate_native
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
