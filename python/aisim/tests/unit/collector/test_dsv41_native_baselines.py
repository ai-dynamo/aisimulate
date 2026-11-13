# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from collector.sglang.collect_dsv41_module import aggregate_baseline_records
from collector.sglang.dsv41_native_runner import collect_native_baselines

pytestmark = pytest.mark.unit


@pytest.fixture
def native_baseline(monkeypatch, tmp_path):
    """Exercise the producer's actual row emission without CUDA or SGLang."""
    dist = Mock()
    dist.get_backend.return_value = "nccl"
    torch = Mock(distributed=dist)
    torch.cuda.Event.return_value.elapsed_time.return_value = 1.25
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe.topk", SimpleNamespace(StandardTopKOutput=Mock()))

    def make(tp_size, rank=0):
        dist.get_world_size.return_value = tp_size
        dist.get_rank.return_value = rank
        # Two token counts, six selections per token; every rank agrees.
        torch.bincount.return_value.cpu.return_value.tolist.side_effect = [[6] + [0] * 383, [12] + [0] * 383]
        quant = type("Mxfp4FlashinferTrtllmMoEMethod", (), {"flashinfer_mxfp4_moe_precision": "default"})()
        experts = Mock(
            quant_method=quant,
            reduce_results=False,
            moe_tp_size=tp_size,
            moe_ep_size=1,
            w2_weight=SimpleNamespace(shape=(384, 5120, 2304 // tp_size // 2)),
        )
        gate = Mock(weight=SimpleNamespace(shape=(384, 5120)))
        lm_head = Mock(weight=SimpleNamespace(shape=(129280 // tp_size, 5120)))
        layer = type("DeepseekV4DecoderLayer", (), {})()
        layer.mlp = SimpleNamespace(experts=experts, gate=gate)
        model = SimpleNamespace(
            tp_size=tp_size,
            config=SimpleNamespace(vocab_size=129280),
            modules=lambda: [layer] * 40,
            lm_head=lm_head,
        )
        options = SimpleNamespace(output=tmp_path, workload_plan=None, batches=[], lengths=[], iterations=1)
        provenance = {
            "source_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "runtime_digest": "sha256:" + "c" * 64,
            "execution_profile": "full",
        }
        return SimpleNamespace(model=model), options, provenance, experts, gate

    return make, torch, dist


@pytest.mark.parametrize(("tp_size", "local_vocab"), [(2, 64640), (4, 32320)])
def test_baselines_emit_loaded_tp_keys_and_preserve_measurements(native_baseline, tmp_path, tp_size, local_vocab):
    make, torch, dist = native_baseline
    for rank in range(tp_size):
        runner, options, provenance, experts, gate = make(tp_size, rank)
        collect_native_baselines(runner, options, rank, provenance)
        rows = [json.loads(line) for line in (tmp_path / f"baseline-rank-{rank}.jsonl").read_text().splitlines()]
        assert [row["kind"] for row in rows] == ["gemm", "gemm", "moe", "nccl", "nccl"] * 2
        assert all(row["latency"] == 1.25 and row["sample"] == 2 and row["tp_rank"] == rank for row in rows)
        assert [(r["m"], r["n"], r["k"]) for r in rows if r["kind"] == "gemm"] == [
            (1, 384, 5120),
            (1, local_vocab, 5120),
            (2, 384, 5120),
            (2, local_vocab, 5120),
        ]
        assert [(r["num_tokens"], r["moe_tp_size"], r["moe_ep_size"]) for r in rows if r["kind"] == "moe"] == [
            (1, tp_size, 1),
            (2, tp_size, 1),
        ]
        assert [(r["num_gpus"], r["message_size"]) for r in rows if r["kind"] == "nccl"] == [
            (tp_size, 10240),
            (tp_size, 12288),
            (tp_size, 20480),
            (tp_size, 24576),
        ]
        assert gate.call_count == experts.call_count == runner.model.lm_head.quant_method.apply.call_count == 6
    assert dist.all_reduce.call_count == tp_size * 12
    assert torch.cuda.Event.call_count == tp_size * 60
    tables = aggregate_baseline_records(sorted(tmp_path.glob("baseline-rank-*.jsonl")), tp_size)
    assert {kind: len(rows) for kind, rows in tables.items()} == {"gemm": 4, "moe": 2, "nccl": 4}
    assert {r["message_size"] for r in tables["nccl"]} == {5120, 6144, 10240, 12288}
    assert all(r["latency"] == 1.25 for rows in tables.values() for r in rows)


@pytest.mark.parametrize("mismatch", ["world_size", "rank", "backend", "moe_tp", "moe_ep", "lm_head", "gate", "vocab"])
@pytest.mark.parametrize("tp_size", [2, 4])
def test_baselines_reject_mislabeled_native_layout_before_measurement(native_baseline, tmp_path, mismatch, tp_size):
    make, torch, dist = native_baseline
    runner, options, provenance, experts, gate = make(tp_size)
    other_tp = 4 if tp_size == 2 else 2
    if mismatch == "world_size":
        dist.get_world_size.return_value = other_tp
    elif mismatch == "rank":
        dist.get_rank.return_value = 1
    elif mismatch == "backend":
        dist.get_backend.return_value = "gloo"
    elif mismatch == "moe_tp":
        experts.moe_tp_size = other_tp
    elif mismatch == "moe_ep":
        experts.moe_ep_size = 2
    elif mismatch == "lm_head":
        runner.model.lm_head.weight.shape = (129280 // other_tp, 5120)
    elif mismatch == "gate":
        gate.weight.shape = (192, 5120)
    else:
        runner.model.config.vocab_size = 129281
    with pytest.raises(RuntimeError, match="native baseline"):
        collect_native_baselines(runner, options, 0, provenance)
    torch.randn.assert_not_called()
    dist.all_reduce.assert_not_called()
    assert not list(tmp_path.glob("baseline-rank-*.jsonl"))


def test_baselines_preserve_hopper_native_w4a16_dispatch(native_baseline, tmp_path):
    make, _torch, _dist = native_baseline
    for rank in range(2):
        runner, options, provenance, experts, _gate = make(2, rank)
        experts.quant_method = type(
            "Mxfp4FlashinferCutlassMoEMethod",
            (),
            {"_use_mxfp8_act_scaling": False, "_use_sm90_humming": False},
        )()
        experts._dsv4_mxfp4_backend = "flashinfer_cutlass_sm90"
        collect_native_baselines(runner, options, rank, provenance)
        rows = [json.loads(line) for line in (tmp_path / f"baseline-rank-{rank}.jsonl").read_text().splitlines()]
        moe_rows = [row for row in rows if row["kind"] == "moe"]
        assert len(moe_rows) == 2
        assert all(row["moe_dtype"] == "w4a16_mxfp4_cutlass" for row in moe_rows)
        assert all(row["kernel_source"] == "sglang_flashinfer_cutlass_moe" for row in moe_rows)
        assert all(row["inter_size"] == 2304 and row["moe_tp_size"] == 2 for row in moe_rows)
        assert experts.call_count == 6
    tables = aggregate_baseline_records(sorted(tmp_path.glob("baseline-rank-*.jsonl")), 2)
    assert all(row["moe_dtype"] == "w4a16_mxfp4_cutlass" for row in tables["moe"])


@pytest.mark.parametrize("mismatch", ["sm120", "humming", "missing_precision", "unloaded", "unknown", "trtllm_bf16"])
def test_baselines_reject_unqualified_moe_precision_before_measurement(native_baseline, tmp_path, mismatch):
    make, torch, dist = native_baseline
    runner, options, provenance, experts, _gate = make(2)
    if mismatch == "trtllm_bf16":
        experts.quant_method.flashinfer_mxfp4_moe_precision = "bf16"
    else:
        attributes = {"_use_mxfp8_act_scaling": False, "_use_sm90_humming": False}
        if mismatch == "sm120":
            attributes["_use_mxfp8_act_scaling"] = True
        elif mismatch == "humming":
            attributes["_use_sm90_humming"] = True
        elif mismatch == "missing_precision":
            del attributes["_use_sm90_humming"]
        name = "UnknownMoEMethod" if mismatch == "unknown" else "Mxfp4FlashinferCutlassMoEMethod"
        experts.quant_method = type(name, (), attributes)()
        experts._dsv4_mxfp4_backend = None if mismatch == "unloaded" else "flashinfer_cutlass_sm90"
    with pytest.raises(RuntimeError, match="native baseline.*MoE"):
        collect_native_baselines(runner, options, 0, provenance)
    torch.randn.assert_not_called()
    dist.all_reduce.assert_not_called()
    assert not list(tmp_path.glob("baseline-rank-*.jsonl"))


def test_humming_rows_require_matching_native_operand_qualification(native_baseline, tmp_path, monkeypatch):
    from collector.sglang import dsv41_native_runner as producer

    make, _torch, _dist = native_baseline
    monkeypatch.setattr(producer, "loaded_humming_geometry", lambda experts: {"w2": {"shape_k": 640}})
    monkeypatch.setattr(
        producer,
        "qualify_native_humming",
        lambda *args: {
            "state": "actual_bf16_native_humming_calls_verified",
            "calls": [
                {
                    "sublayer": name,
                    "input_dtype": "torch.bfloat16",
                    "has_input_scale": False,
                    "compute_config": {"use_f16_accum": False, "gemm_type": "indexed"},
                }
                for name in ("w13", "w2")
            ],
            "configs": [{"core_id": 123}],
        },
    )
    for rank in range(4):
        runner, options, provenance, experts, _gate = make(4, rank)
        experts.quant_method = type("Mxfp4HummingMoEMethod", (), {})()
        collect_native_baselines(runner, options, rank, provenance)
        rows = [json.loads(line) for line in (tmp_path / f"baseline-rank-{rank}.jsonl").read_text().splitlines()]
        assert all(row["physical_local_intermediate"] == 640 for row in rows)
    paths = sorted(tmp_path.glob("baseline-rank-*.jsonl"))
    tables = aggregate_baseline_records(paths, 4)
    assert all(row["moe_dtype"] == "w4a16_mxfp4_humming" for row in tables["moe"])
    assert all(row["kernel_source"] == "sglang_mxfp4_humming_moe" for row in tables["moe"])
    receipt = tmp_path / "humming-qualification-rank-3.json"
    bad = json.loads(receipt.read_text())
    bad["calls"][1]["input_dtype"] = "torch.float8_e4m3fn"
    receipt.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="quantized activations"):
        aggregate_baseline_records(paths, 4)
    receipt.unlink()
    with pytest.raises(ValueError, match="missing actual native Humming"):
        aggregate_baseline_records(paths, 4)
