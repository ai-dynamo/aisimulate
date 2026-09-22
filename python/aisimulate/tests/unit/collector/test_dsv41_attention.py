# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from collector.sglang import dsv41_attention_runner as producer
from collector.sglang.dsv41_contract import build_manifest, canonical_json, write_parquet
from collector.sglang.dsv41_workloads import freeze_workloads

pytestmark = pytest.mark.unit


def inputs(tp=2, bounded=False):
    workloads = freeze_workloads(
        {
            "schema_version": 3,
            "prefill": [{"batch_size": 2, "total_prefill_tokens": 258, "total_kv_read_tokens": 1024}],
            "decode": [{"batch_size": 2, "total_kv_read_tokens": 1024}],
        }
    )
    manifest = build_manifest(tp, bounded)
    plan = dict(
        schema="dsv41.attention-collection.v1",
        purpose="calibration",
        tp_size=tp,
        execution_profile="decoder_bounded" if bounded else "full",
        weight_initializer=producer.WEIGHT_INITIALIZER,
        input_method=producer.INPUT_METHOD,
        warmup=2,
        iterations=5,
        seed=20260921,
        context_length=8192,
        max_total_tokens=8192,
        max_requests=4,
        expected_gpu="H100",
        expected_sm=90,
        moe_runner_backend="flashinfer_mxfp4",
        framework_commit=producer.FRAMEWORK_COMMIT,
        collector_revision="a" * 40,
        source_pins=dict.fromkeys(producer.ATTENTION_SOURCES, "b" * 64),
        metadata_pins=dict.fromkeys(("config.json", "tokenizer.json", "tokenizer_config.json"), "c" * 64),
        image_sha256="d" * 64,
        runtime_digest="sha256:" + "e" * 64,
        workloads_sha256="f" * 64,
        prompt_sha256="0" * 64,
    )
    return plan, manifest, workloads


@pytest.mark.parametrize(
    "change",
    [
        {"tp_size": 4},
        {"expected_sm": 100},
        {"max_total_tokens": 4},
        {"purpose": "promoted_smoke"},
        {"source_pins": {}},
        {"weight_initializer": {"name": "zero"}},
        {"input_method": "real_model_hidden"},
        {"iterations": True},
        {"collector_revision": "dirty"},
    ],
)
def test_unqualified_plan_fails_before_native_import(change):
    plan, manifest, workloads = inputs()
    with pytest.raises(ValueError):
        producer.validate_plan(plan | change, manifest, workloads)


def observations():
    return [dict(layer=i, rows=2, finite=True, nonzero=True) for i in range(40)]


def test_output_validation_requires_actual_complete_nondegenerate_attention():
    producer.validate_attention_outputs(observations())
    for bad in (
        [],
        observations()[:-1],
        [dict(finite_logits=True)],
        observations()[:4] + [dict(layer=4, rows=2, finite=True, nonzero=False)] + observations()[5:],
    ):
        with pytest.raises(RuntimeError):
            producer.validate_attention_outputs(bad)


def fake_layers(manifest, log):
    result = []
    for entry in manifest["phases"]["context"]:
        if entry["component"] != "attention":
            continue
        geometry = json.loads(entry["geometry"])
        indexer = None
        if geometry["role"] in ("full", "reindex"):
            indexer = SimpleNamespace(
                n_heads=geometry["index_n_heads"],
                n_local_heads=geometry["index_n_heads"],
                index_head_dim=geometry["index_head_dim"],
                index_topk=geometry["index_topk"],
                wq_b=SimpleNamespace(weight=SimpleNamespace(shape=(4096, 1280))),
                weights_proj=SimpleNamespace(weight=SimpleNamespace(shape=(32, 5120))),
            )
        attn = SimpleNamespace(
            indexer=indexer,
            attn_tp_size=manifest["tp_size"],
            wo_b=SimpleNamespace(reduce_results=True),
            n_local_heads=geometry["num_heads"],
            n_local_groups=geometry["o_groups"],
            head_dim=geometry["head_dim"],
            q_lora_rank=geometry["q_lora_rank"],
            o_lora_rank=geometry["o_lora_rank"],
            compress_ratio=geometry["compress_ratio"],
        )

        def forward(index=entry["layer"]):
            log.append(("native", index))
            return "native_attention_output"

        attn.forward = forward
        result.append(SimpleNamespace(self_attn=attn))
    return result


@pytest.mark.parametrize("tp", [2, 4])
def test_native_reduction_is_after_cuda_end_and_only_finish_synchronizes(tp):
    _, manifest, _ = inputs(tp)
    log = []

    class Event:
        def __init__(self, **unused):
            pass

        def record(self):
            log.append("event")

        def elapsed_time(self, end):
            return 0.125

    torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event, synchronize=lambda: log.append("sync")))
    layers = fake_layers(manifest, log)
    originals = [layer.self_attn.forward for layer in layers]

    def reduce(value):
        assert value == "native_attention_output"
        log.append("all_reduce")
        return value

    recorder = producer.AttentionRecorder(layers, manifest, torch_module=torch, all_reduce=reduce)
    recorder.active = True
    layers[0].self_attn.forward()
    assert log == ["event", ("native", 0), "event", "all_reduce"]
    rows = recorder.finish(2, 129, 512, True)
    assert log[-1] == "sync" and rows[0]["latency"] == 0.125
    assert (rows[0]["batch_size"], rows[0]["prefix"], rows[0]["x"]) == (2, 512, 129)
    recorder.restore()
    assert all(
        layer.self_attn.forward is original and layer.self_attn.wo_b.reduce_results
        for layer, original in zip(layers, originals, strict=True)
    )


def test_bad_tp_owner_fails_before_any_native_method_mutation():
    _, manifest, _ = inputs()
    layers = fake_layers(manifest, [])
    layers[-1].self_attn.attn_tp_size = 1
    before = [layer.self_attn.forward for layer in layers]
    with pytest.raises(RuntimeError, match="pure-TP"):
        producer.AttentionRecorder(layers, manifest, torch_module=object(), all_reduce=object())
    assert all(
        layer.self_attn.forward is original and layer.self_attn.wo_b.reduce_results
        for layer, original in zip(layers, before, strict=True)
    )


@pytest.mark.parametrize("bounded", [False, True])
def test_real_prefix_batch_and_decode_lifecycle_has_separate_output_qualification(bounded):
    plan, _, workloads = inputs(bounded=bounded)
    calls = []
    stack = SimpleNamespace(qualifying=False)
    recorder = SimpleNamespace(active=False, phase="context", finish=lambda *args: [])

    def extend(reqs):
        calls.append(("extend", stack.qualifying, recorder.active, len(reqs), reqs[0]["length"]))
        # Deliberately invalid logits prove the adapter never uses the stub as
        # an attention validator; actual output proof is a different channel.
        return [7] * len(reqs), float("nan"), object()

    def decode(ids, batch):
        calls.append(("decode", stack.qualifying, recorder.active, len(ids)))
        return ids, float("nan")

    runner = SimpleNamespace(
        clear=lambda: None,
        cleanup=lambda batch: None,
        extend=extend,
        decode=decode,
        torch_runner=SimpleNamespace(model=stack),
    )

    def prepare(batch, length, ids):
        assert len(ids) == batch and all(len(row) == length for row in ids)
        return [dict(length=length) for _ in range(batch)]

    def suffix(args, ids, reqs, actual):
        assert actual is runner.torch_runner and args.cut_len == 512
        for req, row in zip(reqs, ids, strict=True):
            req["length"] = len(row) - args.cut_len
        calls.append(("real_prefix", args.cut_len, len(reqs)))

    bench = SimpleNamespace(
        prepare_synthetic_inputs_for_latency_test=prepare, prepare_extend_inputs_for_correctness_test=suffix
    )
    stack.finish_qualification = lambda: observations() + observations()
    result = producer.collect_workloads(
        runner, recorder, bench, list(range(1024)), plan, workloads, io.StringIO(), 0, {}
    )
    assert len(result) == 2
    assert sum(call[0] == "real_prefix" for call in calls) == 8
    assert sum(call[0] == "decode" for call in calls) == 8
    assert all(not call[2] for call in calls if call[0] in ("extend", "decode") and call[1])
    assert sum(call[0] == "decode" and call[2] for call in calls) == 5


def test_bounded_aliases_are_preserved_with_every_owner_for_publication_gate():
    _, manifest, _ = inputs(bounded=True)
    workloads = freeze_workloads(
        {
            "schema_version": 3,
            "prefill": [
                {"batch_size": 2, "total_prefill_tokens": 1024, "total_kv_read_tokens": 0},
                {"batch_size": 2, "total_prefill_tokens": 512, "total_kv_read_tokens": 512},
            ],
            "decode": [],
        }
    )
    collisions = producer.collision_owners(manifest, workloads)
    assert len(workloads["cases"]) == 2 and collisions
    assert all(item["case_ids"] == ["prefill-0000", "prefill-0001"] for item in collisions)


def write_fixture(tmp_path):
    plan, manifest, workloads = inputs()
    paths = [tmp_path / name for name in ("plan.json", "manifest.json", "workloads.json")]
    paths[2].write_text(json.dumps(workloads))
    plan["workloads_sha256"] = producer.sha(paths[2])
    paths[0].write_text(json.dumps(plan))
    paths[1].write_text(json.dumps(manifest))
    for rank in range(2):
        receipt = dict(
            state="complete_pending_admission",
            tp_rank=rank,
            plan_sha256=producer.sha(paths[0]),
            manifest_sha256=producer.sha(paths[1]),
            workloads_sha256=producer.sha(paths[2]),
            full_model=False,
            checkpoint_weights_loaded=False,
            input_method=producer.INPUT_METHOD,
            purpose="calibration",
            source_hashes=plan["source_pins"],
            qualifications=[
                dict(case_id=case["case_id"], observations=observations() + observations())
                for case in workloads["cases"]
            ],
        )
        (tmp_path / f"attention-rank-{rank}.json").write_text(json.dumps(receipt))
        rows = []
        for index, case in enumerate(workloads["cases"]):
            for sample in range(2, 7):
                for key in sorted(producer.attention_keys(manifest, case)):
                    row = dict(zip(producer.PHYSICAL_KEY, key, strict=True))
                    row.update(
                        latency=0.1 + rank * 0.1,
                        sample_count=1,
                        measurement_scope="local_compute",
                        used_cuda_graph=False,
                        kernel_source="test.native_attention",
                        kv_seed_regime="real_kv",
                        source_sha256=hashlib.sha256(canonical_json(plan["source_pins"]).encode()).hexdigest(),
                        config_sha256=manifest["config_sha256"],
                        runtime_digest=plan["runtime_digest"],
                        execution_profile="full",
                        case_plan_sha256=producer.sha(paths[0]),
                        collection_purpose="calibration",
                        producer_kind="native_attention_isolated",
                        input_method=producer.INPUT_METHOD,
                        tp_rank=rank,
                        invocation=index,
                        sample=sample,
                        case_id=case["case_id"],
                    )
                    rows.append(row)
        (tmp_path / f"rank-{rank}.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return paths


def test_admission_complete_real_kv_rows_use_existing_parquet_contract(tmp_path):
    paths = write_fixture(tmp_path)
    rows = producer.aggregate_attention_records(tmp_path, *paths)
    assert len(rows) == 12 and all(row["latency"] == 0.2 and row["sample_count"] == 5 for row in rows)
    write_parquet(rows, tmp_path / "attention.parquet")
    import pyarrow.parquet as pq

    assert pq.read_table(tmp_path / "attention.parquet").num_rows == 12


@pytest.mark.parametrize("change", ["missing", "duplicate", "whole_model", "source", "output"])
def test_admission_rejects_missing_samples_mixed_producers_and_fake_output(tmp_path, change):
    paths = write_fixture(tmp_path)
    raw = tmp_path / "rank-1.jsonl"
    lines = raw.read_text().splitlines()
    if change == "missing":
        raw.write_text("\n".join(lines[:-1]))
    elif change == "duplicate":
        raw.write_text("\n".join(lines + [lines[0]]))
    elif change in ("whole_model", "source"):
        row = json.loads(lines[0])
        row["producer_kind" if change == "whole_model" else "source_sha256"] = (
            "whole_model" if change == "whole_model" else "0" * 64
        )
        raw.write_text("\n".join([json.dumps(row), *lines[1:]]))
    else:
        path = tmp_path / "attention-rank-1.json"
        receipt = json.loads(path.read_text())
        receipt["qualifications"][0]["observations"] = [dict(finite_logits=True)]
        path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, RuntimeError)):
        producer.aggregate_attention_records(tmp_path, *paths)
