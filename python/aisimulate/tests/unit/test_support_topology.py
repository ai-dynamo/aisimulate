# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Original synthetic geometries exercise onboarding's conservative byte precheck."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from aisimulate.support import config_profile, topology
from aisimulate.support.config_profile import derive_profile, load_model_config, packaged_hardware_path
from aisimulate.support.schema import SupportRequest
from aisimulate.support.topology import suggest_topologies

pytestmark = pytest.mark.unit


def _config(tmp_path, **updates):
    raw = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 64,
        "max_position_embeddings": 4096,
        "torch_dtype": "bfloat16",
    }
    raw.update(updates)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_model_config(path)


def _large_config(tmp_path, **updates):
    return _config(
        tmp_path,
        **{
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 32768,
            **updates,
        },
    )


def _request(kind="dense", gpu="h200_sxm", concurrency=2, interconnect="NVLink", **search):
    return SupportRequest.model_validate(
        {
            "identity": {
                "model": "example/synthetic",
                "model_revision": "checkpoint-123",
                "model_kind": kind,
                "framework_version": "0.25.1",
                "gpu": gpu,
                "interconnect": interconnect,
                "tokenizer_revision": "tokenizer-456",
                "chat_template_revision": "template-789",
            },
            "workload": {"concurrency": concurrency, "request_count": max(4, concurrency)},
            "search": {"context_length": 4096, **search},
        }
    )


def _runtime(**updates):
    return {"kv_cache_dtype": "bfloat16", "fmha_quant_mode": "bfloat16", "comm_quant_mode": "half", **updates}


def _hardware(tmp_path, monkeypatch, **updates):
    spec = {
        "gpu": {"mem_capacity": 16 * 1024**3},
        "node": {"num_gpus_per_node": 8, "intra_node_bw": 1000, "inter_node_bw": 100},
        "misc": {"other_mem": 0, "nccl_mem": {1: 0, 2: 0, 4: 0, 8: 0}},
        **updates,
    }
    path = tmp_path / "hardware.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    monkeypatch.setattr(topology, "packaged_hardware_path", lambda system: path)
    monkeypatch.setattr(config_profile, "packaged_hardware_path", lambda system: path)
    return spec, path


def test_model_size_and_hardware_capacity_change_starting_widths(tmp_path):
    small = suggest_topologies(_config(tmp_path), _request(), _runtime())
    assert [(item.family, item.required_gpus) for item in small.candidates] == [("tp", 1), ("tp", 2)]
    large_config = _large_config(tmp_path)
    large = suggest_topologies(large_config, _request(), _runtime())
    assert [item.required_gpus for item in large.candidates] == [2, 4]
    assert large.default.required_gpus == 2
    assert large.rejected_candidates[0].required_gpus == 1
    assert "lower bound" in large.rejected_candidates[0].reasons[0]
    bigger_gpu = suggest_topologies(large_config, _request(gpu="gb200"), _runtime())
    assert [item.required_gpus for item in bigger_gpu.candidates] == [1, 2]
    assert bigger_gpu.default.required_gpus == 1
    assert bigger_gpu.hardware.per_gpu_bytes > large.hardware.per_gpu_bytes
    assert all(item.status == "estimated_fit" for item in large.candidates + bigger_gpu.candidates)


def test_moe_families_have_exact_tuples_and_deduplicate_width_one(tmp_path):
    config = _config(tmp_path, architectures=["MixtralForCausalLM"], model_type="mixtral", num_local_experts=4)
    report = suggest_topologies(config, _request("moe"), _runtime())
    assert [(item.family, item.sizes) for item in report.candidates] == [
        ("pure_tp", (1, 1, 1, 1)),
        ("pure_tp", (2, 1, 2, 1)),
        ("dep", (1, 2, 1, 2)),
        ("dep", (1, 4, 1, 4)),
        ("tep", (2, 1, 1, 2)),
        ("tep", (4, 1, 1, 4)),
    ]
    assert len({item.sizes for item in report.candidates}) == 6
    assert report.default.family == "pure_tp"
    for item in report.candidates:
        assert item.apply(_request("moe")).worker_gpus == item.required_gpus
        if item.family in {"dep", "tep"}:
            assert item.status == "needs_inputs"
            assert set(item.missing) == {"comm_overhead_bytes"}
            assert item.estimated_required_bytes is None


def test_moe_expert_and_attention_divisibility_rejects_only_incompatible_tuples(tmp_path):
    config = _config(tmp_path, architectures=["MixtralForCausalLM"], model_type="mixtral", num_local_experts=6)
    report = suggest_topologies(config, _request("moe"), _runtime())
    assert [(item.family, item.required_gpus) for item in report.candidates if item.family != "pure_tp"] == [
        ("dep", 2),
        ("tep", 2),
    ]
    rejected = {(item.family, item.required_gpus): item for item in report.rejected_candidates}
    assert "num_experts" in rejected["dep", 4].reasons[0]
    assert "num_experts" in rejected["tep", 4].reasons[0]
    assert "num_attention_heads" in rejected["pure_tp", 8].reasons[0]


@pytest.mark.parametrize("gpu", ["gb200", "gb300"])
def test_declared_high_speed_rack_domain_is_not_limited_to_physical_node(tmp_path, gpu):
    config = _large_config(
        tmp_path,
        hidden_size=32768,
        intermediate_size=131072,
        num_hidden_layers=256,
        num_attention_heads=256,
        num_key_value_heads=32,
    )
    report = suggest_topologies(config, _request(gpu=gpu), _runtime())
    assert report.hardware.gpus_per_node == 4
    assert report.hardware.gpus_per_rack == report.hardware.fast_domain_gpus == 72
    assert report.hardware.fast_domain == "rack"
    assert "inter_node_bw >= intra_node_bw" in report.hardware.assumption
    assert "not available GPU capacity" in report.hardware.assumption
    assert report.candidates[-1].required_gpus == 64
    assert all(item.status == "needs_inputs" for item in report.candidates)
    assert report.default is None
    assert "activations_bytes" in report.candidates[-1].missing
    assert "comm_overhead_bytes" in report.candidates[-1].missing


def test_slower_rack_connectivity_keeps_node_domain(tmp_path, monkeypatch):
    _hardware(
        tmp_path,
        monkeypatch,
        node={"num_gpus_per_node": 4, "num_gpus_per_rack": 72, "intra_node_bw": 1000, "inter_node_bw": 100},
    )
    report = suggest_topologies(_config(tmp_path), _request(), _runtime())
    assert report.hardware.fast_domain == "node"
    assert report.hardware.fast_domain_gpus == 4
    assert report.hardware.gpus_per_rack == 72
    assert all(item.required_gpus <= 4 for item in report.candidates + report.rejected_candidates)


@pytest.mark.parametrize("interconnect", ["PCIe", "IB", "EFA", "custom-fabric"])
def test_declared_fabric_does_not_silently_assume_nvlink_rack(tmp_path, interconnect):
    report = suggest_topologies(_config(tmp_path), _request(gpu="gb200", interconnect=interconnect), _runtime())
    assert report.hardware.declared_interconnect == interconnect
    assert report.hardware.fast_domain == "node"
    assert report.hardware.fast_domain_gpus == 4
    assert "fabric compatibility is unverified" in report.hardware.assumption


def test_no_declared_interconnect_limits_automatic_choices_to_one_gpu(tmp_path):
    report = suggest_topologies(_config(tmp_path), _request(gpu="gb200", interconnect="none"), _runtime())
    assert report.hardware.fast_domain == "single_gpu"
    assert report.hardware.fast_domain_gpus == 1
    assert len(report.candidates) == 1
    assert report.default.required_gpus == 1
    large = suggest_topologies(_large_config(tmp_path), _request(interconnect="none"), _runtime())
    assert not large.candidates
    assert large.default is None
    assert [item.required_gpus for item in large.rejected_candidates] == [1]


@pytest.mark.parametrize(
    "config_updates,overrides,missing",
    [
        ({}, {}, {"fmha_quant_mode", "comm_quant_mode", "kv_cache_dtype", "activations_bytes", "kv_bytes_per_token"}),
        ({"torch_dtype": None}, _runtime(), {"gemm_quant_mode", "moe_quant_mode", "weights_bytes"}),
        ({"quantization_config": {"quant_method": "fp8"}}, _runtime(), {"weights_bytes"}),
        ({"num_key_value_heads": None}, _runtime(), {"kv_bytes_per_token", "weights_bytes"}),
    ],
)
def test_missing_precision_weight_or_cache_inputs_never_fabricate_fit(tmp_path, config_updates, overrides, missing):
    report = suggest_topologies(_config(tmp_path, **config_updates), _request(), overrides)
    assert report.default is None
    assert report.candidates
    for candidate in report.candidates:
        assert candidate.status == "needs_inputs"
        assert missing <= candidate.missing.keys()
        assert candidate.estimated_required_bytes is None
        assert candidate.known_required_bytes > 0


def test_unknown_layout_preserves_unchecked_status(tmp_path):
    config = _config(tmp_path, architectures=["CustomDecoderForCausalLM"], model_type="custom", num_experts=0)
    report = suggest_topologies(config, _request(), _runtime())
    assert report.default is None
    assert {"weights_bytes", "activations_bytes", "kv_bytes_per_token", "cache_layout"} <= report.candidates[
        0
    ].missing.keys()
    assert any("Unknown architecture constraints" in reason for reason in report.assumptions)


def test_missing_communication_precision_prevents_an_automatic_default_even_with_complete_bytes(tmp_path):
    report = suggest_topologies(
        _config(tmp_path), _request(), {"kv_cache_dtype": "bfloat16", "fmha_quant_mode": "bfloat16"}
    )
    assert report.default is None
    assert report.candidates[0].status == "needs_inputs"
    assert report.candidates[0].estimated_required_bytes is not None
    assert set(report.candidates[0].missing) == {"comm_quant_mode"}


def test_known_weight_lower_bound_can_reject_without_precision_cache_or_overheads(tmp_path, monkeypatch):
    _hardware(tmp_path, monkeypatch, gpu={"mem_capacity": 1024}, misc={})
    report = suggest_topologies(_config(tmp_path), _request())
    assert report.default is None
    assert not report.candidates
    rejected = report.rejected_candidates[0]
    assert rejected.status == "rejected"
    assert rejected.known_required_bytes == 13472
    assert rejected.estimated_required_bytes is None
    assert rejected.known_required_bytes > rejected.memory_budget_bytes
    assert {
        "kv_bytes_per_token",
        "activations_bytes",
        "runtime_overhead_bytes",
        "comm_overhead_bytes",
    } <= rejected.missing.keys()


def test_cache_precheck_requires_one_full_context_independent_of_scheduler_batch_limit(tmp_path):
    config = _config(tmp_path, architectures=["MixtralForCausalLM"], model_type="mixtral", num_local_experts=4)
    request = _request("moe", concurrency=8)
    report = suggest_topologies(config, request, _runtime(max_batch_size=3, max_num_tokens=64))
    dep4 = next(item for item in report.candidates if item.family == "dep" and item.required_gpus == 4)
    known_non_kv = sum(
        dep4.draft.resolved[field]
        for field in ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes")
        if field in dep4.draft.resolved
    )
    assert dep4.resident_sequences_per_rank == 1
    assert dep4.known_required_bytes - known_non_kv == 64 * (4096 + 1)
    assert dep4.draft.resolved["max_num_tokens"] == 64
    assert report.context_length == 4096
    assert report.collection["max_sequences"] == 3
    assert report.collection["max_batched_tokens"] == 64
    assert "workload" not in report.to_dict()
    assert any("not guarantee" in item for item in report.assumptions)


def test_validation_concurrency_does_not_change_topology_fit(tmp_path, monkeypatch):
    _hardware(tmp_path, monkeypatch, gpu={"mem_capacity": 85_000_000})
    config = _config(tmp_path)
    light = suggest_topologies(config, _request(concurrency=1), _runtime())
    assert light.default.required_gpus == 1
    heavy = suggest_topologies(config, _request(concurrency=16), _runtime())
    assert heavy.to_dict() == light.to_dict()


@pytest.mark.parametrize("extra_token", [0, 1])
def test_full_context_boundary_matches_existing_plan_admission(tmp_path, monkeypatch, extra_token):
    from aisimulate.support.plan import create_plan
    from aisimulate_core.sdk import perf_database

    # Hand-counted TP1 tensors plus the existing 70 MiB minimum activation estimate.
    total = 13472 + 70 * 1024**2 + 64 * (4096 + extra_token)
    capacity = (total * 10 + 8) // 9
    spec, _ = _hardware(tmp_path, monkeypatch, gpu={"mem_capacity": capacity})
    monkeypatch.setattr(perf_database, "load_system_spec", lambda *args, **kwargs: spec)
    config, request = _config(tmp_path), _request(concurrency=1)
    report = suggest_topologies(config, request, _runtime())
    payload = request.model_dump()
    payload["fpm_profile"] = derive_profile(config, request, _runtime()).profile
    exact = SupportRequest.model_validate(payload)
    if extra_token:
        assert report.default.required_gpus == 1
        plan = create_plan(exact, tmp_path / "plan")
        assert plan["resources"]["total_kv_size_tokens"] == 4097
    else:
        assert report.default.required_gpus == 2
        assert report.rejected_candidates[0].required_gpus == 1
        with pytest.raises(ValueError, match="insufficient rank-local KV capacity"):
            create_plan(exact, tmp_path / "plan")


@pytest.mark.parametrize(
    "field",
    ["weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes", "kv_bytes_per_token"],
)
def test_flat_per_rank_byte_overrides_require_explicit_topology(tmp_path, field):
    with pytest.raises(
        ValueError, match="cannot transfer rank-local byte overrides.*" + field + ".*select an exact topology"
    ):
        suggest_topologies(_config(tmp_path), _request(), _runtime(**{field: 64}))


def test_attached_exact_profile_is_not_silently_replaced(tmp_path):
    config, request = _config(tmp_path), _request()
    payload = request.model_dump()
    payload["fpm_profile"] = derive_profile(config, request, _runtime()).profile
    request = SupportRequest.model_validate(payload)
    with pytest.raises(ValueError, match="attached FPM profile.*exact topology"):
        suggest_topologies(config, request, _runtime())


@pytest.mark.parametrize(
    "overrides", [{"max_batch_size": True}, {"max_num_tokens": "128"}, {"fmha_quant_mode": "auto"}]
)
def test_invalid_shared_overrides_fail_explicitly(tmp_path, overrides):
    with pytest.raises(ValueError):
        suggest_topologies(_config(tmp_path), _request(), overrides)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"gpu": {}}, "gpu.mem_capacity"),
        ({"gpu": {"mem_capacity": True}}, "gpu.mem_capacity"),
        ({"gpu": {"mem_capacity": "1024"}}, "gpu.mem_capacity"),
        ({"node": None}, "node must be an object"),
        ({"node": {}}, "num_gpus_per_node"),
        ({"node": {"num_gpus_per_node": 4}}, "intra_node_bw"),
        ({"node": {"num_gpus_per_node": 4, "intra_node_bw": float("nan"), "inter_node_bw": 1}}, "intra_node_bw"),
        ({"node": {"num_gpus_per_node": 4, "intra_node_bw": 1, "inter_node_bw": True}}, "inter_node_bw"),
        (
            {"node": {"num_gpus_per_node": 4, "intra_node_bw": 1, "inter_node_bw": 1, "num_gpus_per_rack": 6}},
            "num_gpus_per_rack",
        ),
        ({"misc": None}, "misc must be an object"),
        ({"misc": {"other_mem": -1}}, "other_mem"),
        ({"misc": {"nccl_mem": []}}, "nccl_mem"),
        ({"misc": {"nccl_mem": {"1": 0}}}, "TP key"),
        ({"misc": {"nccl_mem": {128: True}}}, "nccl_mem"),
    ],
)
def test_partial_or_malformed_packaged_hardware_fails_explicitly(tmp_path, monkeypatch, updates, match):
    _hardware(tmp_path, monkeypatch, **updates)
    with pytest.raises(ValueError, match=match):
        suggest_topologies(_config(tmp_path), _request(), _runtime())


def test_unknown_packaged_hardware_fails_without_fallback(tmp_path):
    with pytest.raises(ValueError, match="no packaged hardware metadata for nonexistent_gpu"):
        suggest_topologies(_config(tmp_path), _request(gpu="nonexistent_gpu"), _runtime())


@pytest.mark.parametrize("system", ["../h200_sxm", "/tmp/h200_sxm", "h200_sxm.yaml", "", None])
def test_hardware_path_helper_rejects_paths(system):
    with pytest.raises(ValueError, match="packaged system name"):
        packaged_hardware_path(system)


@pytest.mark.parametrize(
    "config_updates,kind,search,match",
    [
        ({}, "moe", {}, "model_kind"),
        ({}, "dense", {"context_length": 8192}, "context_length"),
        ({"hidden_size": 17, "head_dim": None}, "dense", {}, "hidden_size"),
    ],
)
def test_global_model_request_conflicts_are_not_hidden_as_candidate_rejections(
    tmp_path, config_updates, kind, search, match
):
    with pytest.raises(ValueError, match=match):
        suggest_topologies(_config(tmp_path, **config_updates), _request(kind, **search), _runtime())


def test_input_provenance_is_preserved_and_serialization_excludes_profile_objects(tmp_path):
    config = _config(tmp_path, _name_or_path="source/different-id")
    request = _request(tensor_parallel=4, objective="ttft", seed=17)
    overrides = _runtime(max_batch_size=1, max_num_tokens=1024, provenance="confirmed runtime settings")
    before = (copy.deepcopy(config), request.model_dump(), dict(overrides))
    report = suggest_topologies(config, request, overrides)
    selected = report.default.apply(request)
    serialized = json.loads(json.dumps(report.to_dict()))
    assert config == before[0]
    assert request.model_dump() == before[1]
    assert overrides == before[2]
    assert selected.identity == request.identity
    assert selected.workload == request.workload
    assert selected.search.seed == 17
    assert selected.search.objective == "ttft"
    assert selected.search.tensor_parallel == 1
    assert selected.fpm_profile is None
    assert serialized["identity"]["model"] == "example/synthetic"
    assert serialized["identity"]["model_revision"] == "checkpoint-123"
    assert serialized["overrides"] == overrides
    assert serialized["config_sha256"] == hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    hardware_sha = hashlib.sha256(packaged_hardware_path("h200_sxm").read_bytes()).hexdigest()
    assert serialized["hardware"]["sha256"] == hardware_sha
    assert serialized["default"]["status"] == "estimated_fit"
    assert serialized["default"]["cli_flags"] == [
        "--tensor-parallel",
        "1",
        "--attention-data-parallel",
        "1",
        "--moe-tensor-parallel",
        "1",
        "--moe-expert-parallel",
        "1",
    ]
    assert "draft" not in serialized["candidates"][0]
    assert "profile" not in serialized["candidates"][0]
    provenance = json.loads(report.default.draft.profile.provenance)
    assert provenance["config_sha256"] == config.sha256
    assert provenance["deployment_identity"] == request.identity.model_dump()
    assert hardware_sha in provenance["fields"]["runtime_overhead_bytes"]["source"]
    assert "user override" in serialized["default"]["field_sources"]["provenance"]


def test_cold_process_suggestions_do_not_import_native_runtime_models_or_timing_readers(tmp_path):
    _config(tmp_path)
    code = """
import builtins, importlib.abc, json, sys
forbidden = ('aiconfigurator', 'aiconfigurator_core', 'aisimulate.sdk', 'aisimulate_core.sdk',
             'aisimulate_core._native', 'aisimulate._native', 'aisimulate._runtime',
             'aisimulate.runner', 'aisimulate.engine', 'transformers', 'huggingface_hub',
             'collector', 'numpy', 'pandas', 'pyarrow', 'torch')
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, NoRuntime())
from aisimulate.support.config_profile import load_model_config
from aisimulate.support.schema import SupportRequest
from aisimulate.support.topology import suggest_topologies
request = SupportRequest.model_validate(json.loads(sys.argv[2]))
report = suggest_topologies(load_model_config(sys.argv[1]), request, json.loads(sys.argv[3]))
assert report.default is not None
assert report.default.draft.resolved['runtime_overhead_bytes'] == 3758096384
assert report.default.draft.resolved['comm_overhead_bytes'] == 0
assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules for prefix in forbidden)
core = {name for name in sys.modules if name == 'aisimulate_core' or name.startswith('aisimulate_core.')}
assert core <= {'aisimulate_core', 'aisimulate_core.fpm_profile', 'aisimulate_core.quantization'}, core
print(json.dumps(report.to_dict()))
"""
    source = Path(__file__).parents[2] / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(tmp_path / "config.json"),
            _request().model_dump_json(),
            json.dumps(_runtime()),
        ],
        env={**os.environ, "PYTHONPATH": str(source)},
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["default"]["status"] == "estimated_fit"
