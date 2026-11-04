# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""collect.py's two validation entries: --ops probe.<name> runs a probe module as an
ordinary pipeline shard, and AIS_TRACE_KERNELS=1 records each case's kernels from an
extra run whose rows are discarded, so table rows never depend on the switch."""

import functools
import json
import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
from collector import kernel_trace, probes
from collector.probes import vllm_decode_step

pytestmark = pytest.mark.unit

# aisim-auto-collector's shard-plan token charset for an op name (security._TOKEN_RE)
_PIPELINE_OP_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def test_probe_ops_are_pipeline_op_tokens():
    names = probes.probe_op_names()
    assert "probe.vllm_decode_step" in names
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


def _perf_writer(calls):
    def run_case(a, b, *, perf_filename, extra_perf_filenames=(), device):
        calls.append(perf_filename)
        Path(perf_filename).write_text(f"{a},{b}\n")
        for extra in extra_perf_filenames:
            Path(extra).write_text("x\n")

    return run_case


def test_scratch_bound_redirects_every_perf_file(tmp_path):
    func = functools.partial(
        _perf_writer([]), perf_filename=str(tmp_path / "gemm_perf.txt"), extra_perf_filenames=(str(tmp_path / "s.txt"),)
    )
    scratch = tmp_path / "scratch"
    bound = kernel_trace.scratch_bound(func, scratch)
    assert bound.keywords["perf_filename"] == str(scratch / "gemm_perf.txt")
    assert bound.keywords["extra_perf_filenames"] == (str(scratch / "s.txt"),)


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


def test_trace_case_records_kernels_and_discards_the_traced_rows(monkeypatch, tmp_path):
    _fake_torch(monkeypatch, [_Event("gemm_kernel", 5.0), _Event("aten::mm", 5.0), _Event("cpu_only", 0)])
    calls = []
    perf = tmp_path / "gemm_perf.txt"
    func = functools.partial(_perf_writer(calls), perf_filename=str(perf))
    kernel_trace.trace_case(func, (1, 2), "cuda:0", "t1", restart_code=10)
    assert not perf.exists()  # the traced run wrote into a discarded scratch dir
    assert calls and Path(calls[0]).name == "gemm_perf.txt" and Path(calls[0]).parent != tmp_path
    line = json.loads((tmp_path / "gemm_perf.txt.kernels.jsonl").read_text())
    assert line == {"task_id": "t1", "case": "(1, 2)", "kernels": ["gemm_kernel"], "error": None}


def test_trace_case_records_a_failure_and_ignores_a_worker_recycle(monkeypatch, tmp_path):
    _fake_torch(monkeypatch, [])
    perf = tmp_path / "moe_perf.txt"

    def recycle(*task, perf_filename, device):
        raise SystemExit(10)

    def broken(*task, perf_filename, device):
        raise RuntimeError("bad shape")

    kernel_trace.trace_case(functools.partial(recycle, perf_filename=str(perf)), (), "cuda:0", "a", restart_code=10)
    kernel_trace.trace_case(functools.partial(broken, perf_filename=str(perf)), (), "cuda:0", "b", restart_code=10)
    lines = [json.loads(x) for x in (tmp_path / "moe_perf.txt.kernels.jsonl").read_text().splitlines()]
    assert [(x["task_id"], x["error"]) for x in lines] == [("a", None), ("b", "RuntimeError: bad shape")]
