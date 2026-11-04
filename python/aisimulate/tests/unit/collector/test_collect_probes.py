# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""collect.py --ops probe.<name> runs a probe module as an ordinary pipeline shard; the
trace_case probe runs ONE planned case (a batch row cut to that point) under the
profiler, the collector side of an opharness path gate."""

import json
import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
from collector import probes
from collector.probes import trace_case, vllm_decode_step

pytestmark = pytest.mark.unit

# aisim-auto-collector's shard-plan token charset for an op name (security._TOKEN_RE)
_PIPELINE_OP_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def test_probe_ops_are_pipeline_op_tokens():
    names = probes.probe_op_names()
    assert {"probe.trace_case", "probe.vllm_decode_step"} <= set(names)
    assert all(_PIPELINE_OP_TOKEN.fullmatch(n) for n in names)


def test_probe_args_are_key_value_pairs():
    assert probes.parse_args(["model=org/m,layers=4+8", "points=1x1024"]) == {
        "model": "org/m",
        "layers": "4+8",
        "points": "1x1024",
    }
    with pytest.raises(ValueError, match="not key=value"):
        probes.parse_args(["model"])


def _fake_probe(monkeypatch, run):
    mod = types.ModuleType("collector.probes.fake_probe")
    mod.run = run
    monkeypatch.setitem(sys.modules, "collector.probes.fake_probe", mod)


def test_run_probe_writes_one_json_and_sets_the_exit_code(monkeypatch, tmp_path):
    _fake_probe(monkeypatch, lambda args, out_dir: {"echo": args["x"]})
    assert probes.run_probe("fake_probe", ["x=1"], tmp_path) == 0
    assert json.loads((tmp_path / "probe_fake_probe.json").read_text())["result"] == {"echo": "1"}

    _fake_probe(monkeypatch, lambda args, out_dir: {"failed": True, "why": "no engine"})
    assert probes.run_probe("fake_probe", [], tmp_path) == 1
    assert json.loads((tmp_path / "probe_fake_probe.json").read_text())["result"]["why"] == "no engine"

    def boom(args, out_dir):
        raise RuntimeError("no GPU")

    _fake_probe(monkeypatch, boom)
    assert probes.run_probe("fake_probe", [], tmp_path) == 1
    assert "no GPU" in json.loads((tmp_path / "probe_fake_probe.json").read_text())["error"]


def test_decode_step_derives_the_per_layer_cost_from_two_layer_counts(monkeypatch, tmp_path):
    def engine(spec):
        step = {"graph": 1.0, "eager": 10.0}[spec["mode"]] + spec["layers"] * {"graph": 0.1, "eager": 3.0}[spec["mode"]]
        return {**spec, "points": [{"point": list(p), "step_ms": step} for p in spec["points"]]}

    monkeypatch.setattr(vllm_decode_step, "_engine", engine)
    out = vllm_decode_step.run({"model": "org/m", "layers": "4+8", "points": "1x1024"}, tmp_path)
    got = {(r["mode"], r["batch"]): round(r["ms"], 6) for r in out["per_layer_ms"]}
    assert got == {("graph", 1): 0.1, ("eager", 1): 3.0} and out["failed"] is False

    monkeypatch.setattr(vllm_decode_step, "_engine", lambda spec: {**spec, "points": [], "error": "child exit 1"})
    assert vllm_decode_step.run({"model": "org/m"}, tmp_path)["failed"] is True


def test_decode_step_cuts_layers_and_empties_engram_tables(tmp_path):
    model_dir, has_vision = vllm_decode_step._model_dir("deepseek-ai/DeepSeek-V4.1-Flash", 4, str(tmp_path))
    text = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
    assert text["num_hidden_layers"] == 4 and text["engram_layer_ids"] == [] and text["engram_num_embeddings"] == []
    assert has_vision is True


def test_planned_row_restricts_a_batch_row_to_the_selected_point():
    select = dict(model_path="deepseek-ai/DeepSeek-V3.2", kv_cache_dtype="fp8", compute_dtype="bfloat16")
    select.update(gemm_type="fp8_block", num_heads=128, batch_size=1, seq_len=4096, prefix_len=0)
    hopper = trace_case.planned_row("sglang", "dsa_context_module", 90, **select)
    blackwell = trace_case.planned_row("sglang", "dsa_context_module", 100, **select)
    assert hopper[-1] == blackwell[-1] == [[1, 4096, 0]]
    # the FP8-KV backend the plan pins per SM reaches the traced row
    assert (hopper[10], blackwell[10]) == ("flashmla_kv", "trtllm")
    with pytest.raises(ValueError, match="matches 5 planned cases"):
        trace_case.planned_row(
            "sglang", "dsa_context_module", 90, **{k: v for k, v in select.items() if k != "num_heads"}
        )


class _Event:
    def __init__(self, key, device_time):
        self.key, self.self_device_time_total = key, device_time


def _fake_torch(monkeypatch, events):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    profiler = types.ModuleType("torch.profiler")
    profiler.ProfilerActivity = types.SimpleNamespace(CPU="cpu", CUDA="cuda")

    @contextmanager
    def profile(activities):
        yield types.SimpleNamespace(key_averages=lambda: events)

    profiler.profile = profile
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.profiler", profiler)


def test_trace_case_runs_the_planned_row_with_its_perf_file_discarded(monkeypatch, tmp_path):
    _fake_torch(monkeypatch, [_Event("gemm_kernel", 5.0), _Event("aten::mm", 5.0), _Event("cpu_only", 0)])
    calls = []

    def run_func(a, b, *, perf_filename, device):
        calls.append((a, b, perf_filename))
        Path(perf_filename).write_text("row\n")

    entry = types.SimpleNamespace(perf_filename="gemm_perf.txt", extra_perf_filenames=(), run_kwargs=None)
    monkeypatch.setattr(trace_case, "_resolve", lambda backend, op: (entry, run_func))
    monkeypatch.setattr(trace_case, "planned_row", lambda backend, op, sm, **sel: (int(sel["m"]), 7))
    monkeypatch.setattr("collector.helper.get_sm_version", lambda: 100)
    out = trace_case.run({"backend": "vllm", "op": "gemm", "m": "4096"}, tmp_path)
    assert out["kernels"] == ["gemm_kernel"] and out["failed"] is False and out["select"] == {"m": "4096"}
    assert calls[0][:2] == (4096, 7) and not Path(calls[0][2]).exists()  # the scratch perf file is gone

    def broken(*row, perf_filename, device):
        raise RuntimeError("bad shape")

    monkeypatch.setattr(trace_case, "_resolve", lambda backend, op: (entry, broken))
    out = trace_case.run({"backend": "vllm", "op": "gemm", "m": "1"}, tmp_path)
    assert out["failed"] is True and out["error"] == "RuntimeError: bad shape"


def test_run_probe_passes_collects_backend(monkeypatch, tmp_path):
    _fake_probe(monkeypatch, lambda args, out_dir: dict(args))
    assert probes.run_probe("fake_probe", ["op=gemm"], tmp_path, backend="trtllm") == 0
    assert json.loads((tmp_path / "probe_fake_probe.json").read_text())["result"] == {"backend": "trtllm", "op": "gemm"}
