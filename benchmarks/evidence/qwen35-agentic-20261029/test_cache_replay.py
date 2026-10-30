# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the observed-geometry adapter and the native cache gate."""

import copy
import json

import pytest

from cache_probe import derive_manual_geometry, make_receipt
from replay import (
    CorePredictionConfig,
    EngineReplayRunnerFactory,
    ReplayOutputRequirements,
    build_config,
    compare_cache,
    materialize,
    prediction_to_replay_spec,
    replay_case,
)


def observed_worker():
    groups = []
    for index in range(4):
        spec = {"block_size": 4}
        if index:
            spec.update(
                mamba_cache_mode="align",
                num_speculative_blocks=0,
                shapes=[[3, 8], [2, 4, 4]],
                dtypes=["torch.bfloat16", "torch.float32"],
            )
        groups.append(
            {
                "layer_names": [f"group{index}/layer{layer}" for layer in range(15)],
                "spec_type": "MambaSpec" if index else "FullAttentionSpec",
                "page_size_bytes": 16,
                "spec": spec,
            }
        )
    return {
        "vllm_version": "0.31.0",
        "rank": 0,
        "parallelism": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "enable_expert_parallel": False,
        },
        "scheduler": {
            "async_scheduling": False,
            "enable_chunked_prefill": True,
            "disable_hybrid_kv_cache_manager": False,
            "max_num_batched_tokens": 64,
            "max_num_seqs": 16,
        },
        "speculative_config": None,
        "kv_transfer_config": None,
        "cache_config": {
            "block_size": 4,
            "mamba_cache_mode": "align",
            "enable_prefix_caching": True,
            "prefix_match_unit": None,
            "use_replayssm": False,
        },
        "prefix_cache_retention_interval": 0,
        "num_blocks": 128,
        "groups": groups,
        "tensors": [
            {
                "size": 128 * 240,
                "offset": 0,
                "layers": group["layer_names"],
                "layer_stride": 128 * 16,
                "block_stride": 16,
            }
            for group in groups
        ],
        "model": {"max_model_len": 4096},
    }


def requests():
    base = {
        "cell_id": "smoke",
        "lane_id": 0,
        "think_time_ms": 0,
        "input_token_ids": [101, 102, 101, 103, 104, 105, 106, 107],
        "output_token_ids": [205],
        "status": "completed",
    }
    return [
        dict(
            base,
            request_id="a-primer",
            phase="warmup",
            turn_index=-1,
            parent_request_id=None,
            cached_tokens=0,
        ),
        dict(
            base,
            request_id="b-measured",
            phase="measured",
            turn_index=0,
            parent_request_id="a-primer",
            cached_tokens=4,
        ),
    ]


def test_geometry_uses_padded_shared_pool_and_reserves_null_block():
    mapping = derive_manual_geometry(observed_worker())
    assert mapping["capacity"] == {"type": "fixed", "blocks": 127}
    assert mapping["bytes_per_token"] == 60
    assert mapping["state_cache"]["bytes_per_request"] == 720
    assert mapping["prefix_match_unit"] == 4


@pytest.mark.parametrize(
    "change", ["page", "layers", "scratch", "capacity", "retention", "async"]
)
def test_geometry_rejects_unrepresentable_observations(change):
    worker = observed_worker()
    if change == "page":
        worker["groups"][1]["page_size_bytes"] += 1
    elif change == "layers":
        worker["groups"][1]["layer_names"].pop()
    elif change == "scratch":
        worker["groups"][1]["spec"]["num_speculative_blocks"] = 1
    elif change == "capacity":
        worker["tensors"][0]["size"] += 1
    elif change == "retention":
        worker["prefix_cache_retention_interval"] = None
    else:
        worker["scheduler"]["async_scheduling"] = True
    with pytest.raises(ValueError):
        derive_manual_geometry(worker)


def test_probe_requires_every_rank_and_replay_checks_receipt_integrity(tmp_path):
    worker = observed_worker()
    worker["parallelism"]["tensor_parallel_size"] = 2
    with pytest.raises(ValueError, match="every tensor-parallel rank"):
        make_receipt([worker])
    receipt = make_receipt([observed_worker()])
    receipt["manual_kv_cache"]["capacity"]["blocks"] += 1
    with pytest.raises(ValueError, match="does not match"):
        build_config(receipt, tmp_path / "trace.jsonl", None, True)


def test_token_mapping_matches_sorted_native_interning_and_warmup_barrier():
    rows = requests()
    second = copy.deepcopy(rows[0])
    second.update(request_id="z-primer", lane_id=1)
    rows.append(second)
    rows[0]["output_token_ids"] = [107, 205]
    rows[1]["input_token_ids"] += [107, 205]
    trace, evidence = materialize(list(reversed(rows)), "model")
    native_map = {}
    for row in trace[1:]:
        for token in row["hash_ids"]:
            native_map.setdefault(token, len(native_map))
    assert trace[1]["output_token_ids"] == [native_map[107], native_map[205]]
    assert {edge["request_id"] for edge in trace[2]["dependencies"]} == {
        "a-primer",
        "z-primer",
    }
    assert evidence["requests"]["z-primer"]["lane_id"] == 1


def test_gate_does_not_pass_missing_silicon_evidence():
    rows = requests()
    rows[0]["cached_tokens"] = None
    native = {
        "per_request": [
            {
                "request_id": row["request_id"],
                "terminal_status": "completed",
                "output_length": 1,
                "first_admission_reused_input_tokens": row["cached_tokens"],
            }
            for row in rows
        ]
    }
    assert not compare_cache(rows, native)["passed"]


def test_native_engine_preserves_warm_cache_across_dependency(tmp_path):
    source = tmp_path / "requests.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in requests()))
    result = replay_case(
        source, make_receipt([observed_worker()]), None, tmp_path / "replay", True
    )
    assert result["passed"], result
    assert [row["predicted_cached_tokens"] for row in result["comparisons"]] == [0, 4]
    report = json.loads((tmp_path / "replay/native-report.json").read_text())
    by_id = {row["request_id"]: row for row in report["per_request"]}
    assert (
        by_id["b-measured"]["dispatched_at_ms"] >= by_id["a-primer"]["dispatched_at_ms"]
    )


def test_native_generated_tokens_match_later_prompt_under_bijection(tmp_path):
    rows = requests()
    rows[0]["input_token_ids"] = [104, 105, 104, 106]
    rows[0]["output_token_ids"] = [205, 206, 207, 208, 209]
    rows[1]["input_token_ids"] = (
        rows[0]["input_token_ids"] + rows[0]["output_token_ids"] + [301, 302]
    )
    trace, _ = materialize(rows, "model")
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in trace))
    config = build_config(make_receipt([observed_worker()]), trace_path, None, True)
    # Linear KV makes the generated suffix reusable: this isolates token identity
    # from Qwen's semantic state checkpoint-retention policy.
    cache = config["engine"]["workers"]["aggregated"]["kv_cache"]
    del cache["state_cache"]
    del cache["prefix_match_unit"]
    runner = EngineReplayRunnerFactory().create(worker_id=0)
    try:
        report = runner.run(
            prediction_to_replay_spec(CorePredictionConfig.model_validate(config)),
            output_requirements=ReplayOutputRequirements(
                include_raw_report=True, capture_per_request=True
            ),
        )
    finally:
        runner.close()
    native = report.metadata["native_report"]["per_request"]
    by_id = {row["request_id"]: row for row in native}
    assert by_id["b-measured"]["first_admission_reused_input_tokens"] == 8
