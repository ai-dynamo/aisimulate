# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""collect.py --ops probe.<name> runs a probe module as an ordinary pipeline shard; the
trace_case probe runs ONE planned case (a batch row cut to that point) under the
profiler, the collector side of an opharness path gate."""

import json
import os
import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
from collector import probes
from collector.probes import trace_case

pytestmark = pytest.mark.unit

# aisim-auto-collector's shard-plan token charset for an op name (security._TOKEN_RE)
_PIPELINE_OP_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def test_probe_ops_are_pipeline_op_tokens():
    names = probes.probe_op_names()
    assert {"probe.trace_case", "probe.serving"} <= set(names)
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
    def __init__(self, name, device_type="cuda", start=1.0, end=None, annotation=False):
        self.name, self.device_type, self.is_user_annotation = name, device_type, annotation
        self.time_range = types.SimpleNamespace(start=start, end=start + 1 if end is None else end)


def _fake_torch(monkeypatch, events):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.autograd = types.ModuleType("torch.autograd")
    torch.autograd.DeviceType = types.SimpleNamespace(CUDA="cuda", CPU="cpu")
    profiler = types.ModuleType("torch.profiler")
    profiler.ProfilerActivity = types.SimpleNamespace(CPU="cpu", CUDA="cuda")

    @contextmanager
    def profile(activities):
        yield types.SimpleNamespace(events=lambda: events)

    profiler.profile = profile
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.profiler", profiler)
    monkeypatch.setitem(sys.modules, "torch.autograd", torch.autograd)


def _pinned(monkeypatch, version="pinned"):
    """The container holds the release the op pins (the check itself has its own test)."""
    monkeypatch.setattr(trace_case, "installed_version", lambda backend: version)
    monkeypatch.setattr("collector.framework_manifest.require_collector_runtime", lambda *a, **k: None)


def test_trace_case_runs_the_planned_row_with_its_perf_file_discarded(monkeypatch, tmp_path):
    # a custom op launching its kernel directly carries self device time on the CPU side: it
    # is the op, not a kernel
    events = [_Event("gemm_kernel"), _Event("vllm::fp8_gemm_nt_op", "cpu"), _Event("Memcpy HtoD (Pageable -> Device)")]
    _fake_torch(monkeypatch, events)
    _pinned(monkeypatch)
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
    assert out["timed_only"] is False  # the row timed outside benchmark_with_power: the whole run
    assert out["framework_version"] == "pinned"
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


def test_trace_child_argv_wraps_a_measurement_child_only_while_tracing(monkeypatch, tmp_path):
    from collector import helper

    argv = ["/py", "-c", "print(1)"]
    monkeypatch.delenv(helper.TRACE_CHILD_DIR_ENV, raising=False)
    assert helper.trace_child_argv(argv) == argv
    monkeypatch.setenv(helper.TRACE_CHILD_DIR_ENV, str(tmp_path))
    wrapped = helper.trace_child_argv(argv)
    assert (
        wrapped[0] == "/py"
        and wrapped[1].endswith("probes/_traced_child.py")
        and wrapped[2:] == [str(tmp_path), *argv[1:]]
    )
    assert helper.trace_child_argv(["/py", "-m", "collector.x", "--a"])[3:] == ["-m", "collector.x", "--a"]
    assert helper.trace_child_argv(["/py", "script.py"]) == ["/py", "script.py"]


def test_traced_child_runs_the_code_and_records_its_kernels(monkeypatch, tmp_path):
    from collector.probes import _traced_child

    _fake_torch(monkeypatch, [_Event("sparse_mla_kernel")])
    marker = tmp_path / "ran.txt"
    monkeypatch.setattr(sys, "argv", ["w", str(tmp_path / "kernels"), "-c", f"open({str(marker)!r}, 'w').write('ok')"])
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(SystemExit) as done:
        _traced_child.main()
    assert done.value.code == 0 and marker.read_text() == "ok"
    (record,) = [json.loads(p.read_text()) for p in (tmp_path / "kernels").glob("child_*.json")]
    assert record["kernels"] == ["sparse_mla_kernel"] and record["timed"] is False and record["status"] == 0
    assert record["measured"] is None


def test_only_kernels_inside_the_timed_range_count(monkeypatch, tmp_path):
    """The DSA decode case fills its indexer cache with indexer_k_quant_and_cache before timing;
    serving fuses that write, so a whole-run trace graded the case as diverged."""
    from collector import helper
    from collector.probes import _traced_child

    assert helper.TRACE_TIMED_RANGE == _traced_child.TIMED_RANGE
    events = [
        _Event("indexer_k_quant_and_cache_kernel", start=2.0),
        _Event(helper.TRACE_TIMED_RANGE, "cpu", start=10.0, end=20.0),
        _Event(helper.TRACE_TIMED_RANGE, start=10.5, end=19.0, annotation=True),
        _Event("fmhaSm100fKernel", start=12.0),
        _Event("sparse_attn_fwd_kernel", start=30.0),  # after the timed range: teardown
    ]
    _fake_torch(monkeypatch, events)
    prof = types.SimpleNamespace(events=lambda: events)
    assert _traced_child.kernel_names(prof) == (["fmhaSm100fKernel"], True, None)


def test_the_measured_range_keeps_only_the_timed_runs(monkeypatch, tmp_path):
    """The warmups' one-time work (an autotuner's random inputs) is timed-range work the row never
    pays per call; the measured range inside it keeps the replayed kernels alone."""
    from collector import helper
    from collector.probes import _traced_child

    assert helper.TRACE_MEASURED_RANGE == _traced_child.MEASURED_RANGE
    events = [
        _Event(helper.TRACE_TIMED_RANGE, "cpu", start=10.0, end=20.0),
        _Event("distribution_elementwise_grid_stride_kernel", start=11.0),  # warmup: autotune inputs
        _Event("fmhaSm100fKernel", start=12.0),
        _Event(helper.TRACE_MEASURED_RANGE, "cpu", start=15.0, end=19.0),
        _Event(helper.TRACE_MEASURED_RANGE, start=15.5, end=18.0, annotation=True),
        _Event("fmhaSm100fKernel", start=16.0),
    ]
    _fake_torch(monkeypatch, events)
    prof = types.SimpleNamespace(events=lambda: events)
    assert _traced_child.kernel_names(prof) == (
        ["distribution_elementwise_grid_stride_kernel", "fmhaSm100fKernel"],
        True,
        ["fmhaSm100fKernel"],
    )


@pytest.mark.parametrize("child_timed, expect", [(True, ["child_attn"]), (False, ["child_attn", "parent_gemm"])])
def test_trace_case_includes_the_kernels_of_measurement_children(monkeypatch, tmp_path, child_timed, expect):
    """A parent that only spawned a timed child built the module: its kernels are setup."""
    from collector import helper

    _fake_torch(monkeypatch, [_Event("parent_gemm")])
    _pinned(monkeypatch)

    def spawning_run(*row, perf_filename, device):
        child_dir = Path(os.environ[helper.TRACE_CHILD_DIR_ENV])
        child_dir.mkdir(parents=True, exist_ok=True)
        record = {"kernels": ["child_attn"], "timed": child_timed, "status": 0}
        record["measured"] = ["child_attn"] if child_timed else None
        (child_dir / "child_1.json").write_text(json.dumps(record))

    entry = types.SimpleNamespace(perf_filename="mla_perf.txt", extra_perf_filenames=(), run_kwargs=None)
    monkeypatch.setattr(trace_case, "_resolve", lambda backend, op: (entry, spawning_run))
    monkeypatch.setattr(trace_case, "planned_row", lambda backend, op, sm, **sel: (1,))
    monkeypatch.setattr("collector.helper.get_sm_version", lambda: 100)
    monkeypatch.delenv(helper.TRACE_CHILD_DIR_ENV, raising=False)
    out = trace_case.run({"backend": "sglang", "op": "mla_context_module"}, tmp_path)
    assert out["kernels"] == expect and out["timed_only"] is child_timed and out["measurement_children"] == 1
    assert out["measured_kernels"] == (["child_attn"] if child_timed else None)
    assert helper.TRACE_CHILD_DIR_ENV not in os.environ  # restored after the probe


def test_planned_row_narrows_a_sweep_field_to_a_subset_of_its_planned_values():
    moe = dict(moe_type="fp8_block", model_name="deepseek-ai/DeepSeek-V3", hidden_size=7168, inter_size=2048)
    moe.update(topk=8, num_experts=256, tp=1, ep=1, distribution="balanced", power_law_alpha=0.0)
    row = trace_case.planned_row("vllm", "moe", 100, num_tokens="1+4096", **moe)
    assert list(row[1]) == [1, 4096]  # the case's own sweep, cut to the two selected token counts
    assert list(trace_case.planned_row("vllm", "moe", 100, num_tokens=[4096], **moe)[1]) == [4096]
    with pytest.raises(ValueError, match="matches 0 planned cases"):
        trace_case.planned_row("vllm", "moe", 100, num_tokens="3", **moe)  # 3 tokens is not in the planned sweep
    gdn = dict(phase="context", model_name="Qwen/Qwen3.5-0.8B", d_model=1024, d_conv=4, num_k_heads=16)
    gdn.update(head_k_dim=128, num_v_heads=16, head_v_dim=128, batch_sizes="1", seq_lens="4096")
    case_row = trace_case.planned_row("vllm", "gdn", 100, **gdn)
    assert [1] in [list(x) for x in case_row if isinstance(x, (list, tuple))]
    assert [4096] in [list(x) for x in case_row if isinstance(x, (list, tuple))]


def test_planned_row_picks_one_row_of_a_case_planned_as_several():
    moe = dict(moe_type="nvfp4", model_name="nvidia/DeepSeek-V3.1-NVFP4", hidden_size=7168, inter_size=2048)
    moe.update(topk=8, num_experts=256, tp=1, ep=1, distribution="balanced", num_tokens="1+4096")
    with pytest.raises(ValueError, match="runs as 2 rows; name one with row"):
        trace_case.planned_row("trtllm", "moe", 100, **moe)  # min_latency_mode False and True
    default, min_latency = (trace_case.planned_row("trtllm", "moe", 100, row=i, **moe) for i in ("0", 1))
    assert False in default and True in min_latency and list(default[1]) == [1, 4096]
    with pytest.raises(ValueError, match="no row 2"):
        trace_case.planned_row("trtllm", "moe", 100, row=2, **moe)


def test_trace_case_sets_env_arguments_for_the_run_only(monkeypatch, tmp_path):
    _fake_torch(monkeypatch, [_Event("fused_kda")])
    _pinned(monkeypatch)
    seen = {}

    def run_func(*row, perf_filename, device):
        seen["paths"] = os.environ.get("AIS_KDA_DECODE_PATHS")

    entry = types.SimpleNamespace(perf_filename="kda_perf.txt", extra_perf_filenames=(), run_kwargs=None)
    monkeypatch.setattr(trace_case, "_resolve", lambda backend, op: (entry, run_func))
    monkeypatch.setattr(trace_case, "planned_row", lambda backend, op, sm, **sel: (1,))
    monkeypatch.setattr("collector.helper.get_sm_version", lambda: 90)
    monkeypatch.delenv("AIS_KDA_DECODE_PATHS", raising=False)
    out = trace_case.run({"backend": "vllm", "op": "kda", "env.AIS_KDA_DECODE_PATHS": "fused"}, tmp_path)
    assert seen["paths"] == "fused" and out["env"] == {"AIS_KDA_DECODE_PATHS": "fused"} and out["select"] == {}
    assert "AIS_KDA_DECODE_PATHS" not in os.environ


def test_trace_case_refuses_a_release_the_op_does_not_pin(monkeypatch, tmp_path):
    """vLLM gemm has a family pin next to the 0.30.0 default: traced in the default image, the
    gate compared another release's kernels than the table's."""
    _fake_torch(monkeypatch, [_Event("gemm_kernel")])
    monkeypatch.setattr(trace_case, "installed_version", lambda backend: "0.30.0")
    with pytest.raises(RuntimeError, match=r"requires exactly (?!0\.30\.0)\d+\.\d+\.\d+, found 0\.30\.0"):
        trace_case.run({"backend": "vllm", "op": "gemm", "m": "1"}, tmp_path)


def test_measured_timing_gives_one_kernel_table_per_timed_point(monkeypatch):
    """A timed gate compares the per-launch device time of each benchmark_with_power call (one
    per swept point) with serving's at the matching shape."""
    from collector import helper
    from collector.probes import _traced_child

    events = [
        _Event(helper.TRACE_MEASURED_RANGE, "cpu", start=10.0, end=20.0),
        _Event("fc_kernel", start=11.0, end=13.0),
        _Event("fc_kernel", start=14.0, end=16.5),
        _Event("Memcpy DtoD (Device -> Device)", start=17.0),
        _Event("fc_kernel", start=25.0, end=26.0),  # between the windows: the next warm-up
        _Event(helper.TRACE_MEASURED_RANGE, "cpu", start=30.0, end=40.0),
        _Event(helper.TRACE_MEASURED_RANGE, start=30.5, end=39.0, annotation=True),
        _Event("fc_kernel", start=31.0, end=41.0),
    ]
    _fake_torch(monkeypatch, events)
    prof = types.SimpleNamespace(events=lambda: events)
    assert _traced_child.measured_timing(prof) == [
        {"fc_kernel": {"us": 4.5, "launches": 2}},
        {"fc_kernel": {"us": 10.0, "launches": 1}},
    ]
    assert _traced_child.measured_timing(types.SimpleNamespace(events=lambda: events[1:5])) is None
