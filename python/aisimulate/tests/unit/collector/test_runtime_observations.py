# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from collector.fpm_forward import runtime_memory
from collector.fpm_forward.runtime import fpm_memory_observer as legacy
from collector.fpm_forward.runtime_instrumentation import load_instrumentation
from collector.fpm_forward.runtime_observations import validate_observations

from aisimulate_core.fpm_profile import FpmResourceProfile

from .test_fpm_runtime_memory import _vllm_config
from .test_runtime_instrumentation import _manifest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("observed", [None, {}, {"hf_quant_config.json": "b" * 64}, {"hf_quant_config.json": "a" * 64}])
def test_import_independently_verifies_model_config_source_hashes(tmp_path, observed):
    path, launches = observation_fixture(tmp_path)
    launches["tp2"]["model_config"]["source_files"] = {"hf_quant_config.json": "a" * 64}
    _replace_launch(path, launches)
    if observed is not None:
        _mutate(path, lambda record: record.update(model_config_source_files=observed))
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == ("complete" if observed == {"hf_quant_config.json": "a" * 64} else "incomplete")
    if result["status"] == "incomplete":
        assert "source" in str(result["diagnostics"])


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def observation_fixture(root: Path, *, packed=False, planar=False, dense=False, tp=2):
    """Synthetic source, rank and runtime records; no GPU or runtime qualification."""
    root.mkdir(parents=True, exist_ok=True)
    path, manifest = _manifest(root)
    manifest["runtime"]["source_files"] = {"vllm/v1/core/block_pool.py": "b" * 64}
    path.write_text(json.dumps(manifest))
    bundle = load_instrumentation(path)
    model = root / "model.json"
    model.write_text('{"model_type":"example"}')
    launch = {
        "identity": {
            "model": "example/model",
            "model_revision": "immutable",
            "model_kind": "dense" if dense else "moe",
            "framework": "vllm",
            "framework_version": "0.28.0",
            "gpu": "gb300",
            "interconnect": "nvlink",
            "sm": 103,
        },
        "topology": {
            "tp": tp,
            "pp": 1,
            "dp": 1 if dense else 2,
            "moe_tp": 1,
            "moe_ep": 1 if dense else tp * 2,
            "cp": 1,
        },
        "precision": {
            "gemm_quant_mode": "nvfp4",
            "moe_quant_mode": "nvfp4",
            "fmha_quant_mode": "bfloat16",
            "kvcache_quant_mode": "bfloat16",
            "comm_quant_mode": "half",
            "moe_backend": "auto",
            "attention_backend": "auto",
            "enable_wideep": False,
            "enable_eplb": False,
        },
        "collection": {
            "max_model_len": 4096,
            "max_num_batched_tokens": 1024,
            "max_num_seqs": 64,
            "gpu_memory_utilization": 0.9,
            "prefill_cudagraph_policy": "runtime",
            "max_prefill_cudagraph_size": None,
            "async_scheduling": False,
        },
        "model_config": {"path": str(model), "sha256": hashlib.sha256(model.read_bytes()).hexdigest()},
        "deployment": {"executor": "slurm", "image": "example/runtime@sha256:" + "c" * 64},
    }
    attempt = {
        "attempt_id": "attempt-001",
        "bundle": {"manifest": "manifest.yaml", "sha256": bundle.sha256},
        "phases": {},
    }
    index = {
        "schema_version": "aisimulate-runtime-observations/v1",
        "configurations": {"tp2": {"launch": launch, "active_attempt_id": "attempt-001", "attempts": [attempt]}},
    }
    for phase in ("prefill", "decode"):
        expected = {
            "workers": [
                {"dp_rank": dp, "tp_rank": tp, "pp_rank": 0}
                for dp in range(launch["topology"]["dp"])
                for tp in range(launch["topology"]["tp"])
            ],
            "schedulers": [{"dp_rank": dp} for dp in range(launch["topology"]["dp"])],
        }
        context = {
            "schema_version": "aisimulate-runtime-probe-launch/v1",
            "configuration": "tp2",
            "attempt_id": "attempt-001",
            "phase": phase,
            "bundle_sha256": bundle.sha256,
            "launch": launch,
            "expected_ranks": expected,
        }
        ref = _write(root / phase / "launch.json", context)
        ref["path"] = str(Path(ref["path"]).relative_to(root))
        result = {"launch_manifest": ref, "artifacts": []}
        attempt["phases"][phase] = result
        for rank in expected["workers"] + [{**r, "tp_rank": None, "pp_rank": None} for r in expected["schedulers"]]:
            worker = rank["tp_rank"] is not None
            dp = rank["dp_rank"]
            count = 100 - dp * 10 - (5 if phase == "decode" else 0)
            groups = [
                {
                    "layer_names": [name],
                    "spec_type": "vllm.v1.kv_cache_interface.FullAttentionSpec",
                    "block_size_tokens": 16,
                    "spec_page_size_bytes": 64,
                    "sliding_window": None,
                    "extra_retained_tokens": 0,
                    "attention_chunk_size": None,
                    "dtype": "torch.bfloat16",
                    "pool_id": "pool",
                }
                for name in ("layer0", "layer1")
            ]
            # Different cache groups reuse one shared pool storage; a second
            # slot represents padding or a simultaneously needed group layer.
            if packed:
                groups[0]["layer_names"].append("layer2")
            cache = {"num_blocks": count, "groups": copy.deepcopy(groups), "pool_id": "pool"}
            if worker:
                cache.update(
                    {
                        "available_cache_bytes": count * 128 + 256,
                        "allocated_cache_bytes": count * 128,
                        "semantics": {
                            "allocation": "shared_block_pool",
                            "storage": "hbm",
                            "retention": "full_or_window",
                            "prefix_reuse": False,
                            "offload": False,
                            "speculative": False,
                        },
                        "storages": [
                            {"storage_id": "s0", "device": "cuda:0", "pointer": 1024, "size_bytes": count * 128}
                        ],
                        "tensor_allocations": [
                            {
                                "storage_id": "s0",
                                "size": count * 128,
                                "shared_by": ["layer0", "layer1"],
                                "block_stride": 128 if packed else 0,
                                "offset": 0,
                            }
                        ],
                        "layer_tensors": {},
                    }
                )
                if packed:
                    cache["tensor_allocations"].append(
                        {
                            "storage_id": "s0",
                            "size": count * 128,
                            "shared_by": ["layer2"],
                            "block_stride": 128,
                            "offset": 64,
                        }
                    )
                for group in cache["groups"]:
                    group["kind"] = "attention"
                    group["layer_classes"] = dict.fromkeys(
                        group["layer_names"], "vllm.model_executor.layers.attention.Attention"
                    )
                    for name in group["layer_names"]:
                        cache["layer_tensors"][name] = {
                            "storage_id": "s0",
                            "storage_offset_bytes": 64 if name == "layer2" else 0,
                            "shape": [2, count, 16] if planar else [count, 32],
                            "stride_bytes": [count * 32, 32, 2] if planar else [128, 2],
                            "element_size_bytes": 2,
                            "block_axis": 1 if planar else 0,
                            "kernel_block_size_tokens": 16,
                        }
            else:
                cache.update(
                    {
                        "initial_free_blocks": count - 2,
                        "reserved_blocks": 2,
                        "null_block_id": 0,
                        "watermark_blocks": 0,
                        "pool_count": 1,
                        "permanent_reserved_block_ids": [0, 1],
                        "group_pool_ids": ["pool"] * len(groups),
                    }
                )
            record = {
                "schema_version": "aisimulate-runtime-observation/v1",
                **{k: context[k] for k in ("configuration", "attempt_id", "phase", "bundle_sha256")},
                "kind": "worker" if worker else "scheduler",
                **rank,
                "identity": launch["identity"],
                "model_config_sha256": launch["model_config"]["sha256"],
                "runtime": {**manifest["runtime"], "image": launch["deployment"]["image"]},
                "launch": launch,
                "hardware": {
                    "gpu": "gb300",
                    "sm": 103,
                    "device_name": "NVIDIA GB300",
                    "total_memory_bytes": 1000000000,
                },
                "resolved_config": legacy.resolved_config(_vllm_config(dp=launch["topology"]["dp"])),
                "cache": cache,
                "lifecycle": {
                    "measurement": "after_warmup",
                    "cache_initialized": True,
                    "warmup_completed": True,
                    "capture_completed": True,
                }
                if worker
                else {"measurement": "scheduler_initialized", "cache_initialized": True},
                "effective_offloader": "vllm.model_executor.offloader.base.NoopOffloader",
                "unresolved_fields": [],
            }
            record["resolved_config"]["parallel_config"]["enable_expert_parallel"] = not dense
            record["resolved_config"]["parallel_config"]["tensor_parallel_size"] = tp
            ref = _write(
                root / phase / (f"worker-{dp}-{rank['tp_rank']}.json" if worker else f"scheduler-{dp}.json"), record
            )
            ref["path"] = str(Path(ref["path"]).relative_to(root))
            result["artifacts"].append({**ref, "kind": "observation"})
    _write(root / "observations.json", index)
    return root / "observations.json", {"tp2": launch}


def _mutate(index_path, edit):
    index = json.loads(index_path.read_text())
    attempt = index["configurations"]["tp2"]["attempts"][0]
    for phase in attempt["phases"].values():
        for artifact in phase["artifacts"]:
            path = index_path.parent / artifact["path"]
            record = json.loads(path.read_text())
            edit(record)
            path.write_text(json.dumps(record))
            artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index))


def _replace_launch(index_path, launches):
    launch = launches["tp2"]
    index = json.loads(index_path.read_text())
    index["configurations"]["tp2"]["launch"] = launch
    for phase in index["configurations"]["tp2"]["attempts"][0]["phases"].values():
        path = index_path.parent / phase["launch_manifest"]["path"]
        context = json.loads(path.read_text())
        context["launch"] = launch
        path.write_text(json.dumps(context))
        phase["launch_manifest"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index))
    _mutate(index_path, lambda r: r.update(launch=launch, identity=launch["identity"]))


@pytest.mark.parametrize("packed,planar", [(False, False), (True, False), (False, True)])
def test_import_recomputes_storage_and_pool_capacity_and_keeps_every_rank(tmp_path, packed, planar):
    path, launch = observation_fixture(tmp_path, packed=packed, planar=planar)
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    resources = FpmResourceProfile.model_validate(result["resources"])
    assert resources.runtime_memory.kv_cache_bytes == 83 * 128
    assert [group.page_size_bytes for group in resources.cache_groups] == [128, 128]
    assert len(result["provenance"]["rank_capacities"]) == 8
    assert len(result["provenance"]["artifacts"]) == 12


@pytest.mark.parametrize("kernel_subdivision", [False, True])
def test_aliased_layers_cannot_assign_same_bytes_to_different_pool_blocks(tmp_path, kernel_subdivision):
    path, launch = observation_fixture(tmp_path)

    def collide(record):
        if record["kind"] == "worker":
            view = record["cache"]["layer_tensors"]["layer1"]
            if kernel_subdivision:
                view.update(
                    shape=[record["cache"]["num_blocks"] * 2, 16],
                    stride_bytes=[32, 2],
                    kernel_block_size_tokens=8,
                )
            else:
                view["stride_bytes"] = [64, 2]

    _mutate(path, collide)
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "incomplete"
    assert result["resources"] is None
    assert "pool block" in " ".join(result["diagnostics"])


@pytest.mark.parametrize("planar", [False, True])
def test_aliases_preserve_pool_ownership_after_kernel_subdivision_and_permutation(tmp_path, planar):
    path, launch = observation_fixture(tmp_path, planar=planar)

    def equivalent_ownership(record):
        if record["kind"] == "worker":
            count = record["cache"]["num_blocks"]
            record["cache"]["layer_tensors"]["layer1"].update(
                shape=[8, 2, count * 2] if planar else [count * 2, 16],
                stride_bytes=[2, count * 32, 16] if planar else [64, 2],
                block_axis=2 if planar else 0,
                kernel_block_size_tokens=8,
            )

    _mutate(path, equivalent_ownership)
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    assert result["resources"]["runtime_memory"]["kv_cache_bytes"] == 83 * 128


@pytest.mark.parametrize("maximum", [16, 64])
def test_explicit_graph_launch_must_match_initialized_capture_sizes(tmp_path, maximum):
    path, launch = observation_fixture(tmp_path)
    launch["tp2"]["collection"].update(prefill_cudagraph_policy="explicit", max_prefill_cudagraph_size=maximum)
    _replace_launch(path, launch)
    # The synthetic runtime captured powers of two through 64. An explicit
    # limit of 16 is ignored; explicit 64 also requires the intervening sizes
    # emitted by the collector's existing prefill sampling policy.
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "incomplete"
    assert result["resources"] is None
    assert "explicit prefill graph" in " ".join(result["diagnostics"])


@pytest.mark.parametrize(
    "maximum,tokens,sizes",
    [
        (16, 1024, [1, 2, 4, 8, 16]),
        (64, 1024, [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]),
        (64, 32, [1, 2, 4, 8, 16, 24, 32]),
    ],
)
@pytest.mark.parametrize("fallback", [None, "PIECEWISE", "NONE"])
def test_explicit_graph_capture_contract_allows_verified_worker_mode_downgrades(
    tmp_path, maximum, tokens, sizes, fallback
):
    path, launch = observation_fixture(tmp_path)
    sequences = min(tokens, 64)
    launch["tp2"]["collection"].update(
        prefill_cudagraph_policy="explicit",
        max_prefill_cudagraph_size=maximum,
        max_num_batched_tokens=tokens,
        max_num_seqs=sequences,
    )
    _replace_launch(path, launch)

    def captured(record):
        config = record["resolved_config"]
        config["scheduler_config"].update(max_num_batched_tokens=tokens, max_num_seqs=sequences)
        config["compilation_config"].update(cudagraph_capture_sizes=sizes, max_cudagraph_capture_size=sizes[-1])
        if fallback and record["kind"] == "worker":
            record["initial_compilation_config"] = copy.deepcopy(config["compilation_config"])
            config["compilation_config"]["cudagraph_mode"] = fallback

    _mutate(path, captured)
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    assert result["provenance"]["runtime_settings"]["compilation_config"]["cudagraph_mode"] == (
        fallback or "FULL_AND_PIECEWISE"
    )


def test_explicit_prefill_does_not_impose_capture_override_on_decode(tmp_path):
    path, launch = observation_fixture(tmp_path)
    launch["tp2"]["collection"].update(prefill_cudagraph_policy="explicit", max_prefill_cudagraph_size=16)
    _replace_launch(path, launch)
    _mutate(
        path,
        lambda r: r["resolved_config"]["compilation_config"].update(
            cudagraph_capture_sizes=[1, 2, 4, 8, 16], max_cudagraph_capture_size=16
        )
        if r["phase"] == "prefill"
        else None,
    )
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "incomplete"
    # Decode legitimately keeps runtime defaults. Its different graph memory
    # still makes the two phases incompatible for a shared resource profile.
    assert "across phases/ranks" in " ".join(result["diagnostics"])


@pytest.mark.parametrize("declared_eager", [False, True])
def test_explicit_graph_override_is_omitted_only_for_declared_eager_execution(tmp_path, declared_eager):
    path, launch = observation_fixture(tmp_path)
    launch["tp2"]["collection"].update(prefill_cudagraph_policy="explicit", max_prefill_cudagraph_size=16)
    if declared_eager:
        launch["tp2"]["collection"]["enforce_eager"] = True
    _replace_launch(path, launch)

    def eager(record):
        record["resolved_config"]["model_config"]["enforce_eager"] = True
        record["resolved_config"]["compilation_config"].update(
            mode="NONE", cudagraph_mode="NONE", cudagraph_capture_sizes=[], max_cudagraph_capture_size=0
        )

    _mutate(path, eager)
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == ("complete" if declared_eager else "incomplete"), result["diagnostics"]


def test_deployed_checkpoint_path_keeps_public_identity_and_loaded_config_binding(tmp_path):
    path, launch = observation_fixture(tmp_path)
    launch["tp2"]["deployment"].update(executor="kubernetes", model_cache="models:/cache:checkpoint")
    _replace_launch(path, launch)
    _mutate(path, lambda r: r["resolved_config"]["model_config"].update(model="/cache/checkpoint"))
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    assert result["provenance"]["launch"]["identity"]["model"] == "example/model"
    assert result["provenance"]["runtime_settings"]["model_config"]["model"] == "/cache/checkpoint"


@pytest.mark.parametrize("corruption", ["path", "hash", "revision", "missing_mount", "parent_subpath", "slurm"])
def test_deployed_checkpoint_mapping_does_not_weaken_model_identity(tmp_path, corruption):
    path, launch = observation_fixture(tmp_path)
    launch["tp2"]["deployment"].update(executor="kubernetes", model_cache="models:/cache:checkpoint")
    if corruption == "missing_mount":
        launch["tp2"]["deployment"]["model_cache"] = "models::checkpoint"
    elif corruption == "parent_subpath":
        launch["tp2"]["deployment"]["model_cache"] = "models:/cache:../checkpoint"
    elif corruption == "slurm":
        launch["tp2"]["deployment"]["executor"] = "slurm"
    _replace_launch(path, launch)

    def mismatch(record):
        model = record["resolved_config"]["model_config"]
        model["model"] = "/cache/another" if corruption == "path" else "/cache/checkpoint"
        if corruption == "hash":
            record["model_config_sha256"] = "f" * 64
        elif corruption == "revision":
            model["revision"] = "another"

    _mutate(path, mismatch)
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "incomplete"
    assert result["resources"] is None


def test_cache_only_pvc_preserves_public_model_path(tmp_path):
    path, launch = observation_fixture(tmp_path)
    launch["tp2"]["deployment"].update(executor="kubernetes", model_cache="models")
    _replace_launch(path, launch)
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]


def test_observed_expert_parallelism_cannot_certify_partial_tensor_expert_split(tmp_path):
    path, launch = observation_fixture(tmp_path)
    launch["tp2"]["topology"].update(moe_tp=2, moe_ep=2)
    _replace_launch(path, launch)
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "incomplete"
    assert result["resources"] is None
    assert "expert topology" in " ".join(result["diagnostics"])


@pytest.mark.parametrize("enabled,moe_tp,moe_ep", [(True, 1, 4), (False, 4, 1)])
def test_expert_axes_follow_resolved_ep_setting_and_attention_worker_width(tmp_path, enabled, moe_tp, moe_ep):
    path, launch = observation_fixture(tmp_path)
    launch["tp2"]["topology"].update(moe_tp=moe_tp, moe_ep=moe_ep)
    _replace_launch(path, launch)
    _mutate(path, lambda r: r["resolved_config"]["parallel_config"].update(enable_expert_parallel=enabled))
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    assert len(result["provenance"]["rank_capacities"]) == 8


@pytest.mark.parametrize(
    "mutation,diagnostic",
    [
        (lambda r: r.update(attempt_id="another"), "attempt"),
        (lambda r: r["runtime"].update(source_revision="e" * 40), "build"),
        (lambda r: r["runtime"]["source_files"].update({"vllm/v1/core/block_pool.py": "e" * 64}), "build"),
        (lambda r: r["resolved_config"]["model_config"].update(quantization=None), "precision"),
        (lambda r: r["lifecycle"].update(warmup_completed=False) if r["kind"] == "worker" else None, "warmup"),
        (lambda r: r["cache"]["groups"][0].update(extra_retained_tokens=3), "retention"),
        (lambda r: r["cache"].update(group_pool_ids=["other", "pool"]) if r["kind"] == "scheduler" else None, "pool"),
        (lambda r: r["cache"]["storages"][0].update(size_bytes=1) if r["kind"] == "worker" else None, "storage"),
        (
            lambda r: r["cache"]["layer_tensors"]["layer0"].update(stride_bytes=[2, 2])
            if r["kind"] == "worker"
            else None,
            "overlap",
        ),
        (
            lambda r: r["cache"]["tensor_allocations"][0].update(shared_by=["layer0"])
            if r["kind"] == "worker"
            else None,
            "layer",
        ),
        (
            lambda r: r["cache"].update(permanent_reserved_block_ids=[0]) if r["kind"] == "scheduler" else None,
            "reservation",
        ),
        (lambda r: r["cache"]["semantics"].update(offload=True) if r["kind"] == "worker" else None, "semantics"),
    ],
)
def test_import_independently_rejects_corrupt_or_unsupported_evidence(tmp_path, mutation, diagnostic):
    path, launch = observation_fixture(tmp_path)
    _mutate(path, mutation)
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "incomplete"
    assert result["resources"] is None
    assert diagnostic in " ".join(result["diagnostics"])


@pytest.mark.parametrize("mode", ["missing", "duplicate", "hash", "phase", "graph"])
def test_rank_phase_hash_and_graph_coverage_is_required(tmp_path, mode):
    path, launch = observation_fixture(tmp_path)
    index = json.loads(path.read_text())
    phases = index["configurations"]["tp2"]["attempts"][0]["phases"]
    if mode == "missing":
        phases["prefill"]["artifacts"].pop()
    elif mode == "duplicate":
        phases["prefill"]["artifacts"].append(phases["prefill"]["artifacts"][0])
    elif mode == "hash":
        phases["prefill"]["artifacts"][0]["sha256"] = "f" * 64
    elif mode == "phase":
        phases.pop("decode")
    path.write_text(json.dumps(index))
    if mode == "graph":
        _mutate(
            path,
            lambda r: r["resolved_config"]["compilation_config"].update(cudagraph_mode="PIECEWISE")
            if r["phase"] == "decode"
            else None,
        )
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"]


def test_other_configurations_keep_successful_result_and_legacy_guard_survives(tmp_path):
    path, launch = observation_fixture(tmp_path)
    result = validate_observations(path, {**launch, "missing": launch["tp2"]})
    assert result["tp2"]["status"] == "complete", result
    assert result["missing"]["status"] == "incomplete"
    with pytest.raises(ValueError, match="currently supports vLLM 0.27.0"):
        runtime_memory.resolve_runtime_resources(
            None,
            tmp_path,
            expected_plan_sha256="p",
            expected_attempt_id="a",
            expected_backend_version="0.28.0",
            expected_context_length=4096,
            expected_max_num_tokens=1024,
            expected_max_batch_size=64,
            expected_gpu_memory_utilization=0.9,
        )


def test_dense_tp_configuration_does_not_require_moe_axes_to_equal_worker_width(tmp_path):
    path, launch = observation_fixture(tmp_path, dense=True)
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    assert len(result["provenance"]["rank_capacities"]) == 4


def test_packaged_hardware_is_checked_when_request_omits_optional_sm(tmp_path):
    path, launch = observation_fixture(tmp_path)
    launch["tp2"]["identity"].pop("sm")
    index = json.loads(path.read_text())
    index["configurations"]["tp2"]["launch"] = launch["tp2"]
    for phase in index["configurations"]["tp2"]["attempts"][0]["phases"].values():
        context_path = path.parent / phase["launch_manifest"]["path"]
        context = json.loads(context_path.read_text())
        context["launch"] = launch["tp2"]
        context_path.write_text(json.dumps(context))
        phase["launch_manifest"]["sha256"] = hashlib.sha256(context_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(index))

    def wrong_hardware(record):
        record["identity"] = launch["tp2"]["identity"]
        record["launch"] = launch["tp2"]
        record["hardware"].update(sm=90, device_name="NVIDIA H100 80GB HBM3")

    _mutate(path, wrong_hardware)
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "incomplete"
    assert "hardware" in " ".join(result["diagnostics"])


@pytest.mark.parametrize("change", ["missing", "different"])
def test_loaded_model_config_hash_is_independently_required(tmp_path, change):
    path, launch = observation_fixture(tmp_path)
    _mutate(
        path, lambda r: r.pop("model_config_sha256") if change == "missing" else r.update(model_config_sha256="f" * 64)
    )
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "incomplete"
    assert "model configuration" in " ".join(result["diagnostics"])


def test_changed_bundle_code_invalidates_saved_observations_without_execution(tmp_path):
    path, launch = observation_fixture(tmp_path)
    (tmp_path / "observer.py").write_text("raise RuntimeError('must never execute')\n")
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "incomplete"
    assert "bundle hash" in " ".join(result["diagnostics"])


def test_invented_observer_verdict_does_not_authorize_invalid_physical_evidence(tmp_path):
    path, launch = observation_fixture(tmp_path, packed=True)

    def corrupt(record):
        record["passed"] = True
        record["status"] = "validated"
        if record["kind"] == "worker":
            record["cache"]["tensor_allocations"][1]["offset"] = 0
            record["cache"]["layer_tensors"]["layer2"]["storage_offset_bytes"] = 0

    _mutate(path, corrupt)
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "incomplete"
    assert "overlap" in " ".join(result["diagnostics"])
