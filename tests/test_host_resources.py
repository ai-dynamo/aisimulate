# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

import aisimulate.main as cli
import aisimulate.resources as resources
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.config.common import ResourceConfig, split_config_sections
from aisimulate.resources import (
    GB,
    GuardedRunnerFactory,
    HostResources,
    ResourceEstimate,
    ResourceLimitError,
    build_plan,
    constrain_to_cgroups,
    estimate_workload,
    guard_replay,
    require_plan,
    resolve_budget,
    workload_bounds,
)


@pytest.fixture
def host():
    return HostResources(32 * GB, 16 * GB, 8)


def _supervised(monkeypatch, memory_limit_bytes=8 * GB):
    # Unit admission tests must not depend on the pytest process or its children.
    monkeypatch.setattr(resources.psutil.Process, "memory_info", lambda self: SimpleNamespace(rss=64 * resources.MIB))
    monkeypatch.setattr(resources, "_child_memory_bytes", lambda: 0)
    monkeypatch.setenv(
        "_AISIMULATE_SUPERVISED_BUDGET",
        json.dumps(
            {
                "supervisor_pid": os.getpid(),
                "memory_limit_bytes": memory_limit_bytes,
                "cpu_limit": 4,
                "reserved_host_memory_bytes": GB,
            }
        ),
    )


def _config():
    return {
        "engine": {
            "model": "openai/gpt-oss-120b",
            "hardware": "gb300",
            "backend": "vllm",
            "mode": "aggregated",
            "context_length": 16384,
            "workers": {"aggregated": {}},
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 10240, "output_tokens": 1024},
            "load": {"type": "concurrency", "concurrency": {"choices": [8, 64512]}},
            "stop": {"requests_per_load_unit": 100},
        },
        "optimization": {"target": "pareto", "constraints": {"min_candidate_gpus": 72, "max_candidate_gpus": 72}},
        "optimizer": {"parallelism": 8, "max_trials": 256},
    }


def test_reported_dynamo_allocation_is_rejected_without_materialization(host):
    raw = _config()
    original = deepcopy(raw)
    bounds = workload_bounds(CoreRecommendationConfig.model_validate(raw))
    plan = build_plan(bounds, stack="dynamo", host=host, requested_parallelism=8)
    assert plan["estimate"]["request_count"] == 6_451_200
    assert plan["estimate"]["input_token_bytes"] == 264_241_152_000
    assert plan["estimate"]["lower_bound_bytes"] / GB == 264.241152
    assert plan["effective_parallelism"] == 0
    with pytest.raises(ResourceLimitError, match="264.24 GB"):
        require_plan(plan)
    assert raw == original


def test_budget_reserves_headroom_and_leaves_cpu(host):
    budget = resolve_budget(ResourceConfig(), host)
    assert budget["memory_limit_bytes"] == 14_400_000_000
    assert budget["reserved_host_memory_bytes"] == 1_000_000_000
    assert budget["cpu_limit"] == 7
    assert resolve_budget(ResourceConfig(), HostResources(32 * GB, 16 * GB, 0.5))["cpu_limit"] == 1


@pytest.mark.parametrize(
    ("available", "expected_budget"),
    [
        (900_000_000, 0),
        (1_000_000_000, 0),
        (1_000_000_001, 1),
        (4_000_000_000, 3_000_000_000),
        (16_000_000_000, 14_400_000_000),
    ],
)
def test_default_reserve_is_one_decimal_gb_on_large_hosts(available, expected_budget):
    budget = resolve_budget(ResourceConfig(), HostResources(512 * GB, available, 8))
    assert budget["reserved_host_memory_bytes"] == 1_000_000_000
    assert budget["memory_limit_bytes"] == expected_budget


@pytest.mark.parametrize(
    ("settings", "expected_reserve", "expected_budget"),
    [
        ({"reserve_memory_gb": 1.5}, 1_500_000_000, 2_500_000_000),
        ({"reserve_memory_fraction": 0.1}, 3_200_000_000, 800_000_000),
    ],
)
def test_explicit_reserve_overrides(settings, expected_reserve, expected_budget):
    budget = resolve_budget(ResourceConfig(**settings), HostResources(32 * GB, 4 * GB, 8))
    assert budget["reserved_host_memory_bytes"] == expected_reserve
    assert budget["memory_limit_bytes"] == expected_budget


def test_low_memory_never_falls_back_to_one_worker():
    host = HostResources(8 * GB, GB, 4)
    plan = build_plan({"isl": 8, "osl": 2, "request_count": 1}, stack="engine", host=host)
    assert plan["effective_parallelism"] == 0
    with pytest.raises(ResourceLimitError):
        require_plan(plan)


def test_explicit_budgets_are_validated_against_live_limits(host):
    with pytest.raises(ResourceLimitError, match="20.00 GB exceeds available headroom 15.00 GB"):
        resolve_budget(ResourceConfig(memory_limit_gb=20.0), host)
    with pytest.raises(ResourceLimitError, match="CPU allowance"):
        resolve_budget(ResourceConfig(cpu_limit=16), host)
    assert resolve_budget(ResourceConfig(memory_limit_gb=4.0, cpu_limit=2), host)["memory_limit_bytes"] == 4_000_000_000


@pytest.mark.parametrize(
    "field,value",
    [
        ("memory_limit_gb", True),
        ("memory_limit_gb", 0),
        ("memory_limit_gb", float("inf")),
        ("cpu_limit", True),
        ("reserve_memory_fraction", 1.0),
    ],
)
def test_invalid_resource_policy(field, value):
    with pytest.raises(ValidationError):
        ResourceConfig.model_validate({field: value})


def test_cgroup_limits_use_ancestors_and_current_usage(tmp_path, host):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/parent/child\n")
    (proc / "self/mountinfo").write_text("42 30 0:27 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n")
    base = tmp_path / "sys/fs/cgroup"
    child = base / "parent/child"
    child.mkdir(parents=True)
    for directory, limit, usage, cpu in [
        (base, 32 * GB, 4 * GB, "max 100000"),
        (child.parent, 8 * GB, 7 * GB, "150000 100000"),
        (child, 6 * GB, GB, "max 100000"),
    ]:
        (directory / "memory.max").write_text(str(limit))
        (directory / "memory.current").write_text(str(usage))
        (directory / "cpu.max").write_text(cpu)
    actual = constrain_to_cgroups(host, proc=proc, root=tmp_path)
    assert actual.total_memory_bytes == 6 * GB
    assert actual.available_memory_bytes == GB
    assert actual.cpu_count == 1.5
    (child / "memory.current").unlink()
    assert constrain_to_cgroups(host, proc=proc, root=tmp_path).available_memory_bytes == 0


def test_unqualified_estimate_does_not_claim_admission(host):
    plan = build_plan({"trace_path": "unknown.parquet"}, stack="dynamo", host=host)
    assert plan["status"] == "resource_limited"
    assert plan["estimate"]["estimated_peak_bytes"] is None


def test_native_estimate_is_distinct_from_eager_dynamo():
    workload = {"request_count": 1000, "concurrency": 10, "isl": 100, "osl": 5}
    engine = estimate_workload(workload, stack="engine")
    dynamo = estimate_workload(workload, stack="dynamo")
    assert engine.allocation_model != dynamo.allocation_model
    assert engine.input_token_bytes == 10 * 100 * 4
    assert dynamo.input_token_bytes == 1000 * 100 * 4


def test_plugin_estimate_is_versioned(host):
    class BadFactory:
        def estimate_host_resources(self, workload, *, concurrency):
            return ResourceEstimate("bad", 1, 0, 0, 1, api_version=99)

    with pytest.raises(ResourceLimitError, match="incompatible"):
        build_plan({}, stack="custom", factory=BadFactory(), host=host)


def test_execution_section_is_core_and_roundtrips():
    raw = _config()
    raw["execution"] = {"resources": {"memory_limit_gb": 4.0}}
    core, adapters = split_config_sections(raw, command="recommend")
    assert not adapters
    config = CoreRecommendationConfig.model_validate(core)
    roundtrip = CoreRecommendationConfig.model_validate(config.model_dump(mode="json"))
    assert roundtrip.execution.resources.memory_limit_gb == 4.0


@pytest.mark.parametrize("command", ["predict"])
def test_cli_blocks_before_runner_creation(tmp_path, monkeypatch, host, command):
    class NeverExecute:
        def create(self, worker_id):
            pytest.fail("oversized workload must not create a runner")

        def capabilities(self):
            pytest.fail("oversized workload must stop before runtime preparation")

    monkeypatch.setattr(cli, "resolve_runner_factory", lambda _: NeverExecute())
    monkeypatch.setattr(resources, "discover_host", lambda: host)
    raw = _config()
    if command == "predict":
        raw.pop("optimizer")
        raw.pop("optimization")
        raw["traffic"]["load"]["concurrency"] = 64512
        CorePredictionConfig.model_validate(raw)
    config = tmp_path / "case.yaml"
    config.write_text(yaml.safe_dump(raw))
    out = tmp_path / "result"
    args = [command, "--stack", "dynamo", "--config", str(config), "--output-dir", str(out)]
    assert cli.main(args) == 3
    report = json.loads((out / "resource-plan.json").read_text())
    assert report["status"] == "resource_limited"
    assert report["estimate"]["input_token_bytes"] == 264_241_152_000


def test_recommendation_applies_slots_without_changing_suggestion_batches(monkeypatch):
    import aisimulate.recommend as recommendation
    import aisimulate.sweeper.search as search
    from aisimulate.output_adapter import OUTPUT_ADAPTER_API_VERSION, RecommendationOutputCallbacks

    raw = _config()
    raw["traffic"]["load"]["concurrency"] = 8
    config = CoreRecommendationConfig.model_validate(raw)
    monkeypatch.setattr(recommendation, "resolve_budget", lambda *a, **kw: {"cpu_limit": 2})
    seen = []

    class SubscribedAdapter:
        name = "dgd"
        api_version = OUTPUT_ADAPTER_API_VERSION

        def subscribe(self, config):
            assert config == {"socket": "/tmp/events.sock"}
            return RecommendationOutputCallbacks(on_candidate=lambda candidate: None)

        def write(self, config, *, result, output_dir):
            del config, result, output_dir
            return []

    class CaptureSweeper:
        def __init__(self, **kwargs):
            pass

        def run(self, smart, *, top_n, on_candidate=None, on_round=None):
            del top_n
            seen.append((smart.sweep, on_candidate, on_round))
            return "result"

    monkeypatch.setattr(search, "Sweeper", CaptureSweeper)
    assert (
        recommendation._run_recommendation(
            config,
            stack="engine",
            runner_factory=object(),
            output_configs={"dgd": {"socket": "/tmp/events.sock"}},
            output_adapters={"dgd": SubscribedAdapter()},
        )
        == "result"
    )
    sweep, on_candidate, on_round = seen[0]
    assert sweep.parallel_evals == 2
    assert sweep.candidates_per_round == 8
    assert sweep.max_trials == 256
    assert on_candidate is not None
    assert on_round is None
    assert config.optimizer.parallelism == 8


def test_resource_refusal_is_not_a_failed_replay_observation():
    from aisimulate.sweeper.search import _run_replay_detailed

    class Refuse:
        def run(self, spec):
            raise ResourceLimitError("live pressure")

    with pytest.raises(ResourceLimitError, match="live pressure"):
        _run_replay_detailed(None, Refuse())


def test_cgroup_namespace_root_and_v1_quota(tmp_path, host):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("1:memory:/\n2:cpu,cpuacct:/\n")
    (proc / "self/mountinfo").write_text(
        "42 30 0:27 /container /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n"
        "43 30 0:28 /container /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n"
    )
    memory = tmp_path / "sys/fs/cgroup/memory"
    cpu = tmp_path / "sys/fs/cgroup/cpu"
    memory.mkdir(parents=True)
    cpu.mkdir(parents=True)
    (memory / "memory.limit_in_bytes").write_text(str(4 * GB))
    (memory / "memory.usage_in_bytes").write_text(str(GB))
    (cpu / "cpu.cfs_quota_us").write_text("50000")
    (cpu / "cpu.cfs_period_us").write_text("100000")
    actual = constrain_to_cgroups(host, proc=proc, root=tmp_path)
    assert actual.total_memory_bytes == 4 * GB
    assert actual.available_memory_bytes == 3 * GB
    assert actual.cpu_count == 0.5


def test_error_diagnostic_preserves_existing_output(tmp_path, monkeypatch, host):
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda _: object())
    monkeypatch.setattr(resources, "discover_host", lambda: host)
    config = tmp_path / "case.yaml"
    raw = _config()
    raw.pop("optimizer")
    raw.pop("optimization")
    raw["traffic"]["load"]["concurrency"] = 64512
    config.write_text(yaml.safe_dump(raw))
    output = tmp_path / "output"
    output.mkdir()
    evidence = output / "resource-plan.json"
    evidence.write_text("existing evidence")
    assert cli.main(["predict", "--stack", "dynamo", "--config", str(config), "--output-dir", str(output)]) == 3
    assert evidence.read_text() == "existing evidence"


def test_probe_failure_does_not_assume_unlimited_memory(monkeypatch):
    def fail():
        raise OSError("unavailable")

    monkeypatch.setattr(resources.psutil, "virtual_memory", fail)
    with pytest.raises(ResourceLimitError, match="cannot discover"):
        resources.discover_host()


def test_resource_refusal_survives_process_transport():
    import pickle

    error = ResourceLimitError("pressure", plan={"status": "resource_limited", "reason": "pressure", "bytes": 123})
    transported = pickle.loads(pickle.dumps(error))
    assert transported.plan == error.plan


def test_namespace_relative_descendant_keeps_ancestor_limits(tmp_path, host):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/child\n")
    (proc / "self/mountinfo").write_text("42 30 0:27 /container /sys/fs/cgroup rw - cgroup2 cgroup rw\n")
    root = tmp_path / "sys/fs/cgroup"
    (root / "child").mkdir(parents=True)
    (root / "memory.max").write_text(str(4 * GB))
    (root / "memory.current").write_text(str(GB))
    actual = constrain_to_cgroups(host, proc=proc, root=tmp_path)
    assert actual.total_memory_bytes == 4 * GB
    assert actual.available_memory_bytes == 3 * GB


def test_missing_container_mount_probe_fails_closed(tmp_path, host):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/\n")
    with pytest.raises(ResourceLimitError, match="membership and mounts"):
        constrain_to_cgroups(host, proc=proc, root=tmp_path)


@pytest.mark.parametrize("trace_format", ["mooncake", "agentic_mooncake"])
@pytest.mark.parametrize("length_field", ["input_length", "output_length", "input_tokens", "output_tokens"])
@pytest.mark.parametrize("stack", ["engine", "dynamo"])
def test_trace_scalar_lengths_are_counted_before_token_expansion(
    tmp_path, host, monkeypatch, trace_format, length_field, stack
):
    trace = tmp_path / "large.jsonl"
    row = {"input_length": 1, "output_length": 1, "hash_ids": [1]}
    row.pop(length_field.replace("_tokens", "_length"))
    row[length_field] = 10**12
    trace.write_text(json.dumps(row) + "\n")
    _supervised(monkeypatch)
    plan = build_plan({"trace_path": str(trace), "trace_format": trace_format}, stack=stack, host=host)
    assert plan["status"] == "resource_limited"
    if trace_format == "agentic_mooncake":
        assert plan["estimate"]["estimated_peak_bytes"] is None
        assert plan["estimate"]["lower_bound_bytes"] >= 4 * 10**12
        assert "lower bound plus worker baseline exceeds available host memory" in plan["reason"]
        assert "requires supervised" not in plan["reason"]
    else:
        assert plan["estimate"]["estimated_peak_bytes"] > 32 * 10**12


def test_trace_inspection_streams_large_documents_and_records(tmp_path, host, monkeypatch):
    trace = tmp_path / "large.jsonl"
    # Each valid record exceeds the old 256 KiB limit, total exceeds 16 MiB.
    record = '{"input_length": 2, "output_length": 1, "ignored": [' + "0," * 150_000 + "0]}\n"
    with trace.open("w") as stream:
        for _ in range(60):
            stream.write(record)
    monkeypatch.setattr(resources.json, "loads", lambda _: pytest.fail("must not materialize entire JSON records"))
    plan = build_plan({"trace_path": str(trace), "trace_format": "mooncake"}, stack="engine", host=host)
    assert plan["status"] == "admitted"
    assert plan["estimate"]["estimated_peak_bytes"] > trace.stat().st_size


def test_delta_trace_accounts_for_cumulative_prompts(tmp_path):
    trace = tmp_path / "delta.jsonl"
    trace.write_text((json.dumps({"input_length": 100, "output_length": 10, "hash_ids": [1]}) + "\n") * 5)
    base = {"trace_path": str(trace), "trace_block_size": 4}
    ordinary = estimate_workload({**base, "trace_format": "mooncake"}, stack="engine")
    delta = estimate_workload({**base, "trace_format": "mooncake-delta"}, stack="engine")
    assert delta.estimated_peak_bytes > ordinary.estimated_peak_bytes


def test_fixed_capacity_kv_domain_has_a_conservative_count_bound(host):
    raw = _config()
    raw["traffic"]["load"] = {"type": "kv_capacity_fraction", "fraction": {"choices": [0.5, 1.5]}}
    raw["engine"]["workers"]["aggregated"]["kv_cache"] = {
        "block_size": 64,
        "capacity": {"type": "fixed", "blocks": 256},
    }
    bounds = workload_bounds(CoreRecommendationConfig.model_validate(raw))
    assert bounds["concurrency"] == 256 * 64 * 72 * 1.5
    assert bounds["request_count"] == bounds["concurrency"] * 100


def test_cgroup_parent_components_never_escape_a_root_mount(tmp_path, host):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/../outside\n")
    (proc / "self/mountinfo").write_text("42 30 0:27 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n")
    with pytest.raises(ResourceLimitError, match="outside its visible namespace"):
        constrain_to_cgroups(host, proc=proc, root=tmp_path)


def test_unqualified_estimate_requires_supervision_and_serial_admission(monkeypatch, host):
    workload = {"source_type": "synthetic-session", "request_count": 4, "turns_per_session": 2}
    assert build_plan(workload, stack="dynamo", host=host)["status"] == "resource_limited"
    _supervised(monkeypatch)
    plan = build_plan(workload, stack="dynamo", host=host, requested_parallelism=4)
    assert plan["status"] == "admitted"
    assert plan["effective_parallelism"] == 1
    assert plan["estimate"]["estimated_peak_bytes"] is None


@pytest.mark.parametrize("trace_format,profile", [("mooncake", None), ("agentic_mooncake", {})])
def test_trace_storage_is_rejected_before_parser_allocates_scalars(tmp_path, host, monkeypatch, trace_format, profile):
    trace = tmp_path / "oversized.jsonl"
    with trace.open("wb") as stream:
        stream.truncate(128 * resources.MIB)  # Sparse sentinel; no large allocation.
    monkeypatch.setattr(resources.ijson, "parse", lambda *a, **kw: pytest.fail("must refuse before parsing"))
    workload = {"trace_path": str(trace), "trace_format": trace_format}
    if profile is not None:
        workload["agentic_profile"] = profile
    plan = build_plan(workload, stack="engine", host=host)
    assert plan["status"] == "resource_limited"
    assert plan["estimate"]["estimated_peak_bytes"] > plan["budget"]["memory_limit_bytes"]
    assert "before metadata parsing" in plan["estimate"]["reason"]


@pytest.mark.parametrize("stack", ["engine", "dynamo"])
def test_finite_agentic_lanes_share_compact_trace_and_require_supervision(tmp_path, host, monkeypatch, stack):
    trace = tmp_path / "agentic.jsonl"
    header = {
        "schema": "dynamo.agentic_mooncake",
        "version": 2,
        "block_size": 1048576,
        "hash_id_scope": "local",
        "source": {"format": "test", "digest": "test"},
    }
    rows = [
        {
            "request_id": str(i),
            "play_id": str(i),
            "session_id": str(i),
            "model": "test",
            "input_length": 1048576,
            "output_length": 1,
            "hash_ids": [i],
            "not_before_ms": 0,
            "dependencies": [],
        }
        for i in range(4)
    ]
    trace.write_text("\n".join(json.dumps(row) for row in [header, *rows]) + "\n")
    base = {"trace_path": str(trace), "trace_format": "agentic_mooncake", "trace_block_size": 1048576}
    monkeypatch.delenv("_AISIMULATE_SUPERVISED_BUDGET", raising=False)
    for lanes in (1, 32, 128, 512):
        plan = build_plan({**base, "agentic_lanes": lanes}, stack=stack, host=host)
        assert plan["estimate"]["estimated_peak_bytes"] is None
        assert plan["estimate"]["lower_bound_bytes"] == 4 * resources.MIB  # One expanded prompt, not the corpus.
        assert plan["status"] == "resource_limited"  # An unknown peak needs the live supervisor.
        assert plan["reason"] == "unknown peak requires supervised serial execution"

    _supervised(monkeypatch, memory_limit_bytes=2 * GB)
    for lanes in (1, 32, 128, 512):
        workload = {**base, "agentic_lanes": lanes}
        original = deepcopy(workload)
        plan = build_plan(workload, stack=stack, host=host, requested_parallelism=4)
        assert plan["status"] == "admitted"
        assert plan["effective_parallelism"] == 1  # One simulation; all requested lanes are retained.
        assert plan["estimate"]["lower_bound_bytes"] == 4 * resources.MIB
        assert workload == original

    # Whitespace adds file bytes without changing stored arrays. The former
    # 128-times-file-size guard would reject this padded trace under 2 GB.
    estimate = plan["estimate"]
    with trace.open("a") as stream:
        for _ in range(256):
            stream.write(" " * (64 * 1024) + "\n")
    padded = build_plan(workload, stack=stack, host=host, requested_parallelism=4)
    assert padded["status"] == "admitted"
    assert padded["estimate"] == estimate

    monkeypatch.setattr(resources, "discover_host", lambda: host)
    spec = SimpleNamespace(workload={**base, "agentic_lanes": 512}, concurrency=None)
    factory = GuardedRunnerFactory(object(), stack, ResourceConfig())
    assert factory.admit_wave([spec])["status"] == "admitted"
    with pytest.raises(ResourceLimitError, match="wave"):
        factory.admit_wave([spec, spec])


@pytest.mark.parametrize("output_length,authored,expected", [(0, None, 16), (5, None, 28), (2, [7, 8], 24)])
def test_finite_agentic_arrays_ignore_unloaded_fields(tmp_path, output_length, authored, expected):
    trace = tmp_path / "agentic.jsonl"
    row = {
        "request_id": "r",
        "input_tokens": 2,
        "output_tokens": output_length,
        "hash_ids": [1, 2],
        "output_token_ids": authored,
    }
    workload = {"trace_path": str(trace), "trace_format": "agentic_mooncake"}
    trace.write_text(json.dumps(row) + "\n")
    original = estimate_workload(workload, stack="engine")
    # Two source u64 hashes occupy 16 bytes; five planned u32 outputs plus
    # compact hashes occupy 28; two authored u32 outputs beside source hashes occupy 24.
    assert original.lower_bound_bytes == expected
    row["input_sequence_hashes"] = [1] * 100
    row["provenance"] = {
        "input_length": "ignored",
        "output_tokens": 10**12,
        "hash_ids": [1] * 100,
        "output_token_ids": [1] * 100,
    }
    trace.write_text(json.dumps(row) + "\n")
    assert estimate_workload(workload, stack="engine") == original


@pytest.mark.parametrize(
    "prefix,lower", [("input_tokens", 4), ("output_tokens", 4), ("hash_ids.item", 8), ("output_token_ids.item", 4)]
)
def test_finite_agentic_inspection_stops_when_storage_cannot_fit(tmp_path, monkeypatch, prefix, lower):
    trace = tmp_path / "agentic.jsonl"
    trace.write_text("{}\n")

    def events(*args, **kwargs):
        yield "", "start_map", None
        yield prefix, "number", 1
        pytest.fail("must stop before reading more metadata after the lower bound cannot fit")

    monkeypatch.setattr(resources.ijson, "parse", events)
    estimate = estimate_workload(
        {"trace_path": str(trace), "trace_format": "agentic_mooncake"},
        stack="engine",
        inspection_budget_bytes=resources.WORKER_BASELINE_BYTES + lower - 1,
    )
    assert estimate.estimated_peak_bytes is None
    assert estimate.lower_bound_bytes == lower


@pytest.mark.parametrize("peak", [None, GB])
def test_only_unknown_peaks_add_baseline_to_array_bound(monkeypatch, host, peak):
    # Fixed supervisor RSS plus coordinator reserve leaves exactly 1 GB free.
    _supervised(monkeypatch, memory_limit_bytes=GB + 320 * resources.MIB)
    factory = SimpleNamespace(
        estimate_host_resources=lambda *args, **kwargs: ResourceEstimate("test-v1", 1, 0, 900_000_000, peak)
    )
    plan = build_plan({}, stack="engine", host=host, factory=factory)
    if peak is None:
        assert plan["status"] == "resource_limited"  # 900 MB alone fits; adding the baseline does not.
        assert "lower bound plus worker baseline" in plan["reason"]
        _supervised(monkeypatch, memory_limit_bytes=900_000_000 + resources.WORKER_BASELINE_BYTES + 320 * resources.MIB)
        assert build_plan({}, stack="engine", host=host, factory=factory)["status"] == "admitted"
    else:
        assert plan["status"] == "admitted"  # A qualified peak already includes its baseline.


def test_unknown_peak_wave_reserves_arrays_after_child_rss(monkeypatch, host):
    _supervised(monkeypatch)
    monkeypatch.setattr(resources, "discover_host", lambda: host)
    factory = GuardedRunnerFactory(
        SimpleNamespace(
            estimate_host_resources=lambda *args, **kwargs: ResourceEstimate("test-v1", 1, 0, 6 * GB, None)
        ),
        "engine",
        ResourceConfig(),
    )
    spec = SimpleNamespace(workload={}, concurrency=None)
    assert factory.admit_wave([spec])["required_bytes"] == 6 * GB + resources.WORKER_BASELINE_BYTES
    monkeypatch.setattr(resources, "_child_memory_bytes", lambda: 5 * GB)
    with pytest.raises(ResourceLimitError) as caught:
        factory.admit_wave([spec])
    assert caught.value.plan["required_bytes"] == 6 * GB + resources.WORKER_BASELINE_BYTES
    assert caught.value.plan["available_bytes"] < 3 * GB


@pytest.mark.parametrize("warmup", [False, True])
def test_snapshot_and_warmup_keep_previous_accounting(tmp_path, warmup):
    from aisimulate.recommend import recommendation_to_sweeper

    trace = tmp_path / "agentic.jsonl"
    trace.write_text(json.dumps({"input_length": 4, "output_length": 1, "hash_ids": [1]}) + "\n")
    raw = _config()
    raw["traffic"] = {
        "source": {"type": "trace", "format": "agentic_mooncake", "paths": [str(trace)], "block_size": 4},
        "load": {
            "type": "trace_timestamps",
            "agentic_lanes": 32,
            "agentic_snapshot": {"seed": 42},
            "agentic_warmup": warmup,
        },
    }
    config = CoreRecommendationConfig.model_validate(raw)
    bounds = workload_bounds(config)
    assert bounds["agentic_snapshot"] == {"seed": 42}
    assert bounds["agentic_warmup"] is warmup
    concrete = recommendation_to_sweeper(config).workload.model_dump(mode="python", exclude_none=True)
    for stack in ("engine", "dynamo"):
        estimate = estimate_workload(bounds, stack=stack)
        assert estimate.allocation_model == "trace-json-metadata-v1"
        assert estimate.estimated_peak_bytes is not None
        assert estimate_workload(concrete, stack=stack) == estimate
    ordinary = {key: value for key, value in bounds.items() if key not in {"agentic_snapshot", "agentic_warmup"}}
    assert estimate_workload(ordinary, stack="engine").estimated_peak_bytes is None


@pytest.mark.parametrize("profile", [{}, {"duration_seconds": 86400}])
def test_profile_resource_admission_requires_supervision_and_one_worker(tmp_path, host, monkeypatch, profile):
    from aisimulate.recommend import recommendation_to_sweeper

    trace = tmp_path / "play.json"
    trace.write_text(
        json.dumps(
            {
                "id": "play",
                "block_size": 64,
                "hash_id_scope": "local",
                "models": ["model"],
                "requests": [{"t": 0, "type": "s", "model": "model", "in": 8, "out": 1}],
            }
        )
    )
    raw = _config()
    raw["traffic"] = {
        "source": {"type": "trace", "format": "weka", "paths": [str(trace)]},
        "load": {
            "type": "trace_timestamps",
            "agentic_lanes": 1,
            "agentic_snapshot": {"seed": 42},
            "agentic_profile": profile,
        },
    }
    config = CoreRecommendationConfig.model_validate(raw)
    bounds = workload_bounds(config)
    assert bounds["agentic_profile"] == config.traffic.load.agentic_profile.model_dump(mode="python")
    assert bounds["agentic_profile"]["duration_seconds"] == profile.get("duration_seconds", 3600)
    finite = {key: value for key, value in bounds.items() if key != "agentic_profile"}
    assert build_plan(finite, stack="engine", host=host, requested_parallelism=4)["effective_parallelism"] == 4
    monkeypatch.delenv("_AISIMULATE_SUPERVISED_BUDGET", raising=False)
    unmonitored = build_plan(bounds, stack="engine", host=host, requested_parallelism=4)
    assert unmonitored["status"] == "resource_limited"
    assert unmonitored["estimate"]["estimated_peak_bytes"] is None
    assert unmonitored["estimate"]["allocation_model"] == "agentic-profile-unqualified-v1"

    # The recommendation compiler's concrete workload reaches both the per-run
    # guard and whole-wave admission; profile estimates must not admit a pool.
    smart = recommendation_to_sweeper(config)
    spec = SimpleNamespace(workload=smart.workload.model_dump(mode="python", exclude_none=True), concurrency=None)
    monkeypatch.setattr(resources, "discover_host", lambda: host)
    with pytest.raises(ResourceLimitError, match="profile"):
        guard_replay(spec, stack="engine")
    _supervised(monkeypatch)
    plan = build_plan(bounds, stack="engine", host=host, requested_parallelism=4)
    assert plan["status"] == "admitted"
    assert plan["effective_parallelism"] == 1
    assert guard_replay(spec, stack="engine")["estimate"]["estimated_peak_bytes"] is None
    factory = GuardedRunnerFactory(object(), "engine", config.execution.resources)
    assert factory.admit_wave([spec])["status"] == "admitted"
    with pytest.raises(ResourceLimitError, match="wave"):
        factory.admit_wave([spec, spec])


@pytest.mark.parametrize("oversized", ["storage", "tokens"])
def test_profile_keeps_initial_trace_materialization_guards(tmp_path, host, monkeypatch, oversized):
    trace = tmp_path / "oversized.jsonl"
    if oversized == "storage":
        with trace.open("wb") as stream:
            stream.truncate(4 * 1024**3)  # Sparse scalar sentinel; refused without reading.
        monkeypatch.setattr(resources.ijson, "parse", lambda *a, **kw: pytest.fail("must refuse before parsing"))
    else:
        trace.write_text(
            json.dumps(
                {
                    "id": "play",
                    "block_size": 64,
                    "hash_id_scope": "local",
                    "models": ["model"],
                    "requests": [{"t": 0, "type": "s", "model": "model", "in": 10**12, "out": 1}],
                }
            )
            + "\n"
        )
    plan = build_plan(
        {"trace_path": str(trace), "trace_format": "weka", "agentic_profile": {}}, stack="engine", host=host
    )
    assert plan["status"] == "resource_limited"
    assert plan["estimate"]["estimated_peak_bytes"] > plan["budget"]["memory_limit_bytes"]
