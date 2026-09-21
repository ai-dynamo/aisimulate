# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pyarrow.parquet as pq
import pytest
import yaml
from collector.sglang_rubin import collect_all_reduce as comm
from collector.sglang_rubin.runtime import EXPECTED_BUILD_ENV, REQUIRED_SERVING_ENV

pytestmark = pytest.mark.unit


def _args(tmp_path, *extra):
    return comm._parse_args(
        [
            "--checkpoint-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--launcher-image",
            comm.IMAGE_REF,
            *extra,
        ]
    )


@pytest.fixture
def inventory():
    return {
        "observed": {
            "platform": {"system": "Linux", "machine": "aarch64"},
            "package_versions": {
                "sglang": {"version": comm.SGLANG_DISTRIBUTION_VERSION},
                "torch": {"version": "2.14.0a0+4fdf77b940.nvinternal.rubin.0.8full.66388760"},
            },
            "reported_build_environment": dict(EXPECTED_BUILD_ENV),
            "serving_environment": dict(REQUIRED_SERVING_ENV),
            "cuda": {
                "available": True,
                "torch_version": "2.14.0a0+4fdf77b940.nvinternal.rubin.0.8full.66388760",
                "torch_cuda_version": "13.5",
                "devices": [
                    {
                        "index": rank,
                        "uuid": f"GPU-{rank}",
                        "name": "NVIDIA Graphics Device",
                        "capability": [10, 7],
                        "total_memory_bytes": 299083431936,
                    }
                    for rank in range(4)
                ],
            },
            "checkpoint": {
                "files": {name: {"sha256": digest} for name, digest in comm.CHECKPOINT_METADATA_SHA256.items()}
            },
        }
    }


def test_help_has_no_framework_or_third_party_imports():
    root = Path(comm.__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-S", "-m", comm.MODULE, "--help"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--tokens" in result.stdout
    assert "message_size is the number of elements" in result.stdout


@pytest.mark.parametrize(
    "extra",
    [
        ["--tokens", "0"],
        ["--tokens", "1", "1"],
        ["--modes", "graph", "graph"],
        ["--modes", "custom"],
        ["--iterations", "0"],
        ["--warmup", "-1"],
        ["--launcher-image", "unpinned-image:latest"],
    ],
)
def test_bad_arguments_fail_before_runtime_import(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        _args(tmp_path, *extra)
    assert error.value.code == 2


def test_default_plan_and_sampling_are_bounded(tmp_path):
    args = _args(tmp_path)
    assert args.tokens == [1, 8, 32, 128, 1024, 8192, 16384]
    assert args.modes == ["graph", "eager"]
    assert args.output_dir.is_absolute()
    assert _args(tmp_path, "--tokens", "8", "1024", "--modes", "graph").tokens == [8, 1024]


def _launch_env(**overrides):
    return {
        "RANK": "2",
        "LOCAL_RANK": "2",
        "WORLD_SIZE": "4",
        "LOCAL_WORLD_SIZE": "4",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",
        "GROUP_RANK": "0",
        **overrides,
    }


def test_four_rank_single_node_mapping():
    assert comm._rank_from_environment(_launch_env()) == 2


@pytest.mark.parametrize(
    "env",
    [
        {},
        _launch_env(WORLD_SIZE="8"),
        _launch_env(LOCAL_WORLD_SIZE="2"),
        _launch_env(LOCAL_RANK="0"),
        _launch_env(RANK="4", LOCAL_RANK="4"),
        _launch_env(GROUP_RANK="1"),
        _launch_env(MASTER_PORT="65536"),
    ],
)
def test_invalid_rank_mapping_rejected(env):
    with pytest.raises(ValueError):
        comm._rank_from_environment(env)


def test_preflight_requires_exact_distribution_and_distinct_devices(inventory):
    assert comm._preflight_errors(inventory) == []
    inventory["observed"]["package_versions"]["sglang"]["version"] = "0.5.18"
    inventory["observed"]["cuda"]["devices"][3]["uuid"] = "GPU-0"
    errors = comm._preflight_errors(inventory)
    assert any("Expected SGLang" in error for error in errors)
    assert any("distinct" in error for error in errors)


def test_preflight_rejects_three_gpus_wrong_sm_and_checkpoint(inventory):
    inventory["observed"]["cuda"]["devices"].pop()
    inventory["observed"]["cuda"]["devices"][1]["capability"] = [10, 0]
    inventory["observed"]["checkpoint"]["files"]["config.json"]["sha256"] = "wrong"
    errors = comm._preflight_errors(inventory)
    assert any("Exactly four" in error for error in errors)
    assert any("requires SM107" in error for error in errors)
    assert any("config.json" in error for error in errors)


def _reports(inventory):
    return [
        {"rank": rank, "hostname": "hecate", "device": device, "communication": {"mode": "native"}}
        for rank, device in enumerate(inventory["observed"]["cuda"]["devices"])
    ]


@pytest.mark.parametrize("fault", ["host", "device", "order", "dispatch"])
def test_cross_rank_checks_detect_mismatched_topology_and_dispatch(inventory, fault):
    reports = _reports(inventory)
    comm._validate_rank_reports(reports)
    if fault == "host":
        reports[1]["hostname"] = "another-host"
    elif fault == "device":
        reports[1]["device"]["uuid"] = reports[0]["device"]["uuid"]
    elif fault == "order":
        reports.reverse()
    else:
        reports[1]["communication"] = {"mode": "different"}
    with pytest.raises(RuntimeError):
        comm._validate_rank_reports(reports)


def test_remote_failure_reaches_each_rank():
    def gather(reports, value, *, group):
        reports[:] = [None, None, "RuntimeError: bad sum", None]

    dist = SimpleNamespace(all_gather_object=gather)
    with pytest.raises(RuntimeError, match="sum validation failed: rank 2: RuntimeError: bad sum"):
        comm._stage(dist, object(), "sum validation", lambda: "locally successful")


def test_local_publication_error_is_not_swallowed():
    observed = []

    def gather(reports, value, *, group):
        observed.append(value)
        reports[:] = [value, None, None, None]

    def fail():
        raise OSError("disk full")

    with pytest.raises(RuntimeError, match="publication failed: rank 0: OSError: disk full"):
        comm._stage(SimpleNamespace(all_gather_object=gather), object(), "publication", fail)
    assert observed == ["OSError: disk full"]


def _group(**overrides):
    return SimpleNamespace(
        **{
            "ca_comm": None,
            "qr_comm": None,
            "pymscclpp_comm": None,
            "torch_symm_mem_comm": None,
            "pynccl_comm": None,
            "_flashinfer_allreduce": lambda value: value,
            "_fi_workspace_hint": None,
            "device": "cuda:0",
            **overrides,
        }
    )


def test_dispatch_trace_records_invoked_fallback_and_restores_methods():
    class Custom:
        def custom_all_reduce(self, value):
            return value

    group = _group(ca_comm=Custom())
    original = lambda value: None
    dist = SimpleNamespace(all_reduce=original)
    with comm._trace_dispatch(group, dist, "mnnvl") as calls:
        dist.all_reduce("input")
    assert calls == ["torch.distributed.all_reduce[nccl]"]
    assert dist.all_reduce is original
    with comm._trace_dispatch(group, dist, "mnnvl") as calls:
        group._flashinfer_allreduce("input")
    assert calls == ["sglang.srt.layers.flashinfer_comm_fusion.flashinfer_allreduce[mnnvl]"]


class _Tensor:
    def __init__(self, shape, value, dtype, device, events):
        self.shape, self.value, self.dtype, self.device, self.events = shape, value, dtype, device, events

    def fill_(self, value):
        self.value = value

    def zero_(self):
        self.events.append("zero")
        self.value = 0


def _fake_torch(events):
    class Event:
        def __init__(self, *, enable_timing):
            pass

        def record(self):
            events.append("event")

        def elapsed_time(self, other):
            return 8.0

    def full(shape, value, *, dtype, device):
        events.append("allocate")
        return _Tensor(shape, value, dtype, device, events)

    def equal(left, right):
        events.append("validate")
        return left.value == right.value

    return SimpleNamespace(
        full=full,
        full_like=lambda item, value: full(item.shape, value, dtype=item.dtype, device=item.device),
        bfloat16="bfloat16",
        equal=equal,
        cuda=SimpleNamespace(synchronize=lambda: events.append("synchronize"), Event=Event),
    )


def _install_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)


def test_eager_timing_excludes_resets_and_validates_sum_before_timing(monkeypatch, tmp_path):
    events = []
    torch = _fake_torch(events)

    def all_reduce(value):
        events.append(f"reduce:{value.value}")
        value.value = 10 if value.value else 0

    def gather(reports, value, *, group):
        reports[:] = [value * (rank + 1) if isinstance(value, float) else value for rank in range(4)]

    dist = SimpleNamespace(
        all_reduce=all_reduce, all_gather_object=gather, monitored_barrier=lambda **kwargs: events.append("barrier")
    )

    def native_all_reduce(value):
        dist.all_reduce(value)
        return value

    _install_module(
        monkeypatch, "sglang.srt.distributed.communication_op", tensor_model_parallel_all_reduce=native_all_reduce
    )
    _install_module(monkeypatch, "sglang.srt.distributed.parallel_state", graph_capture=None)
    args = _args(tmp_path, "--iterations", "2", "--warmup", "1")
    result = comm._benchmark(torch, dist, _group(), object(), {}, args, 0, 8, "eager")
    start, end = [index for index, event in enumerate(events) if event == "event"]
    assert events[start + 1 : end] == ["reduce:0"] * 10
    assert events.index("validate") < events.index("zero") < events.index("barrier") < start
    assert result["latency"] == pytest.approx(3.2)  # Slowest rank, per all-reduce.
    assert result["message_size"] == 8 * 6144  # Elements, not bytes.
    assert result["backend"] == "sglang_eager"
    assert result["kernel_source"] == "torch.distributed.all_reduce[nccl]_eager"


def test_wrong_sum_fails_before_any_timing_event():
    events = []
    torch = _fake_torch(events)
    inputs = [torch.full((1, 6144), 1, dtype="bfloat16", device="cuda:0") for _ in range(comm.REPEAT)]
    with pytest.raises(RuntimeError, match="sum validation failed"):
        comm._validate_outputs(torch, inputs, inputs)
    assert "event" not in events


def test_serving_workspace_capacity_is_not_expanded_to_sweep_size(monkeypatch):
    observed = []
    managers = {
        True: SimpleNamespace(initialized=True, backend="mnnvl", max_token_num=2048, hidden_dim=6144),
        False: SimpleNamespace(initialized=True, backend="mnnvl", max_token_num=2048, hidden_dim=6144),
    }
    _install_module(monkeypatch, "sglang.srt.layers.communicator", FUSE_ALLREDUCE_MAX_BATCH_SIZE=2048)
    _install_module(
        monkeypatch,
        "sglang.srt.layers.flashinfer_comm_fusion",
        _get_workspace_manager=lambda attention: managers[attention],
        pre_initialize_workspaces=lambda **kwargs: observed.append(kwargs),
    )
    result = comm._initialize_workspaces(
        SimpleNamespace(flashinfer_allreduce_fusion_backend="auto"), SimpleNamespace(bfloat16="bfloat16")
    )
    assert observed == [{"max_token_num": 2048, "hidden_dim": 6144, "dtype": "bfloat16"}]
    assert result["attn_tp"]["backend"] == "mnnvl"


def test_publication_uses_existing_perf_contract_and_truthful_metadata(tmp_path, inventory, monkeypatch):
    # Other collector tests import both historical spellings during discovery.
    # A real pilot worker binds them before executor imports; mirror that
    # process bootstrap while testing publication in the shared pytest process.
    import collector.helper as helper

    from aisimulate_core import resolve_op_sources_report_json

    monkeypatch.setitem(sys.modules, "helper", helper)
    monkeypatch.setitem(sys.modules, "collector.helper", helper)

    args = _args(tmp_path)
    args.output_dir.mkdir()
    rows = [
        {
            "allreduce_dtype": "bfloat16",
            "num_gpus": 4,
            "message_size": 6144,
            "latency": 0.005,
            "backend": f"sglang_{mode}",
            "kernel_source": f"observed_implementation_{mode}",
            "rank_latencies_ms": [0.004, 0.005, 0.004, 0.004],
        }
        for mode in ("graph", "eager")
    ]
    comm._publish(args, inventory, _reports(copy.deepcopy(inventory)), rows)
    # Admit the exact published parquet and sidecar through the native engine's
    # strict resolver, using the family layout consumed by CPU replay. Diagnostic
    # JSON stays outside the performance-data directory.
    systems_root = tmp_path / "systems"
    system_data_root = systems_root / "data" / "vr200_hecate"
    version_dir = system_data_root / "comm" / "sglang" / comm.SGLANG_DISTRIBUTION_VERSION
    version_dir.mkdir(parents=True)
    for filename in ("custom_allreduce_perf.parquet", "collection_meta.yaml"):
        shutil.copy2(args.output_dir / filename, version_dir / filename)
    report = json.loads(
        resolve_op_sources_report_json(
            str(systems_root),
            str(system_data_root),
            "sglang",
            comm.SGLANG_DISTRIBUTION_VERSION,
            "custom_allreduce_perf.parquet",
            strict=True,
        )
    )
    assert report["records"] == [
        {
            "version": comm.SGLANG_DISTRIBUTION_VERSION,
            "path": str(version_dir / "custom_allreduce_perf.parquet"),
            "channel": "primary",
            "exists": True,
            "ks_filter": None,
        }
    ]
    assert report["warnings"] == []
    table = pq.read_table(args.output_dir / "custom_allreduce_perf.parquet")
    assert table.column_names == [
        "framework",
        "version",
        "device",
        "op_name",
        "kernel_source",
        "allreduce_dtype",
        "num_gpus",
        "message_size",
        "latency",
        "backend",
    ]
    assert table["message_size"].to_pylist() == [6144, 6144]
    assert table["backend"].to_pylist() == ["sglang_graph", "sglang_eager"]
    assert table["allreduce_dtype"].to_pylist() == ["bfloat16", "bfloat16"]
    assert not (args.output_dir / "custom_allreduce_perf.txt").exists()
    meta = yaml.safe_load((args.output_dir / "collection_meta.yaml").read_text())
    assert meta["tables"]["custom_allreduce_perf"]["rows"] == 2
    assert meta["tables"]["custom_allreduce_perf"]["collector_ref"] == comm.MODULE
    assert meta["runtime"]["version"] == comm.SGLANG_DISTRIBUTION_VERSION
    assert meta["runtime"]["backend_capability"]["fused_residual_rmsnorm_measured"] is False
    observed = json.loads((args.output_dir / "all_reduce_observed.json").read_text())
    assert observed["image_identity_verified"] is False
    assert observed["measurements"][0]["rank_latencies_ms"] == rows[0]["rank_latencies_ms"]
