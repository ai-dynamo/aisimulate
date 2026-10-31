# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
import math
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


def test_cpu_sidecars_are_archived_without_changing_memory_import(tmp_path):
    from collector.fpm_forward.runtime_probe import observation_artifacts

    path, launches = observation_fixture(tmp_path)
    baseline = validate_observations(path, launches)["tp2"]
    cpu = tmp_path / "prefill/fpm-cpu-scheduler-dp0.json"
    cpu.write_text(
        json.dumps({"schema_name": "aisimulate_fpm_cpu_affinity", "schema_version": 1, "status": "unavailable"})
    )
    references = observation_artifacts(tmp_path, tmp_path / "prefill")
    reference = next(item for item in references if item["path"] == "prefill/fpm-cpu-scheduler-dp0.json")
    assert reference["kind"] == "runtime-artifact"
    index = json.loads(path.read_text())
    index["configurations"]["tp2"]["attempts"][0]["phases"]["prefill"]["artifacts"].append(reference)
    path.write_text(json.dumps(index))
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete"
    assert result["resources"]["cache_groups"] == baseline["resources"]["cache_groups"]
    assert (
        result["resources"]["runtime_memory"]["kv_cache_bytes"]
        == baseline["resources"]["runtime_memory"]["kv_cache_bytes"]
    )
    assert reference in result["provenance"]["runtime_artifacts"]


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


_V031_SPEC = "vllm.v1.kv_cache_interface."


def _v031_records(layout, *, prefix_caching, dp, max_model_len=4096):
    """The worker cache, scheduler cache and resolved config of one vLLM 0.31.0 rank.

    ``glm_tp4`` follows the live GLM-5.3-Flash TP4 fake of ``test_runtime_adapter._v031_runtime``
    (ruling R-S): G0 lists each indexer before its MLA latent layer, the scheduler copy of G0 is
    the first indexer spec, MLA and indexer views are 3-D, KDA layer i of every Mamba group
    overlays the region of MLA layer i and cuts its two states from that page, and tail layer i
    overlays the region of indexer layer i. ``mamba_hybrid`` is vLLM's generic layer-outer
    grouping of two attention and four Mamba layers: every group's tensor starts at byte 0
    (kv_cache_utils.py:1775-1819). Without prefix caching a Mamba block holds max_model_len
    tokens (models/config.py:674-675), and the scheduler block size is the LCM of the group
    blocks (kv_cache_utils.py:770, :779-780).
    """
    mode = "align" if prefix_caching else "none"
    allocations, views = [], {}

    def spec(name, page, dtype, tokens, role=None, storage=None):
        return {
            "spec_type": _V031_SPEC + name,
            "page_size_bytes": page,
            "dtype": dtype,
            "tokens_per_state": tokens,
            "cache_role": role,
            "storage_block_size": storage,
        }

    def group(names, name, block, page, specs, kind, dtype, classes, state=None):
        record = {
            "layer_names": names,
            "spec_type": _V031_SPEC + name,
            "block_size_tokens": block,
            "spec_page_size_bytes": page,
            "sliding_window": None,
            "dtype": dtype,
            "kind": kind,
            "layer_specs": specs,
            "layer_classes": dict.fromkeys(names, classes) if isinstance(classes, str) else classes,
            "extra_retained_tokens": 0,
            "attention_chunk_size": None,
            "pool_id": "pool",
        }
        if state is not None:
            record["state"] = state
        return record

    def mamba_state(shapes, dtypes, real_page):
        return {
            "mamba_cache_mode": mode,
            "num_speculative_blocks": 0,
            "num_prefill_checkpoint_blocks": 0,
            "shapes": shapes,
            "dtypes": dtypes,
            "real_page_size_bytes": real_page,
            "blocks_per_request": 1,
        }

    def place(names, offset, layer_stride, block_stride):
        allocations.append(
            {
                "storage_id": "s0",
                "shared_by": names,
                "layers": names,
                "layer_stride": layer_stride,
                "block_stride": block_stride,
                "offset": offset,
            }
        )

    def view(offset, shape, strides, element, kernel):
        return {
            "storage_id": "s0",
            "storage_offset_bytes": offset,
            "shape": shape,
            "stride_bytes": strides,
            "element_size_bytes": element,
            "block_axis": 0,
            "kernel_block_size_tokens": kernel,
        }

    def scheduled(record, **first_spec):
        return {key: value for key, value in record.items() if key not in ("kind", "layer_classes")} | first_spec

    if layout == "glm_tp4":
        blocks, free, page, block, hash_block = 8, 7, 13_045_120, 2176, 2176
        mamba_block = block if prefix_caching else max_model_len
        mla = list(range(3, 45, 4))
        kda = [index for index in range(45) if index % 4 != 3]
        # Round robin into four Mamba groups of 9, 9, 8 and 8 layers (kv_cache_utils.py:1276-1286).
        members = [[f"layers.{index}.kda" for index in kda[start::4]] for start in range(4)]
        latent = spec("MLAAttentionSpec", 1_114_112, "torch.uint8", 1, "SPARSE")
        indexer = spec("MLAAttentionSpec", 71_808, "torch.uint8", 4, "INDEXER", 128)
        tail = spec("KpoolTailSpec", 71_808, "torch.bfloat16", 1)
        attention = [name for index in mla for name in (f"layers.{index}.indexer", f"layers.{index}.mla")]
        tails = [f"layers.{index}.tail" for index in mla]
        mla_class = "vllm.model_executor.layers.attention.mla_attention.MLAAttention"
        indexer_class = "vllm.models.glm5next.common.attention.Glm5NextIndexerCache"
        groups = [
            group(
                attention,
                "UniformTypeKVCacheSpecs",
                block,
                13_045_120,
                {name: latent if name.endswith(".mla") else indexer for name in attention},
                "attention",
                "torch.uint8",
                {name: mla_class if name.endswith(".mla") else indexer_class for name in attention},
            ),
            group(
                tails,
                "UniformTypeKVCacheSpecs",
                4,
                789_888,
                dict.fromkeys(tails, tail),
                "state",
                "torch.bfloat16",
                "vllm.models.glm5next.common.attention.Glm5NextTailCache",
                {"blocks_per_request": 1},
            ),
            *(
                group(
                    names,
                    "MambaSpec",
                    mamba_block,
                    1_114_112,
                    dict.fromkeys(names, spec("MambaSpec", 1_114_112, None, -1)),
                    "state",
                    None,
                    "vllm.models.glm5next.common.kda.Glm5NextLinearAttention",
                    mamba_state([[6144, 3], [16, 128, 128]], ["torch.bfloat16", "torch.float32"], 1_085_440),
                )
                for names in members
            ),
        ]
        # The scheduler copy of a uniform group keeps its first inner spec (kv_cache_utils.py:2434-2440).
        scheduler_groups = [
            scheduled(
                groups[0],
                spec_type=indexer["spec_type"],
                spec_page_size_bytes=71_808,
                dtype="torch.uint8",
                layer_specs=dict.fromkeys(attention, indexer),
            ),
            scheduled(
                groups[1],
                spec_type=tail["spec_type"],
                spec_page_size_bytes=71_808,
                dtype="torch.bfloat16",
                layer_specs=dict.fromkeys(tails, tail),
            ),
            *(scheduled(item) for item in groups[2:]),
        ]
        for position, index in enumerate(mla):
            offset = position * 8_912_896
            place([f"layers.{index}.mla"], offset, 8_912_896, 1_114_112)
            views[f"layers.{index}.mla"] = [view(offset, [272, 64, 512], [32_768, 512, 1], 1, 64)]
            for names in members:
                if position < len(names):
                    place([names[position]], offset, 8_912_896, 1_114_112)
                    views[names[position]] = [
                        view(offset, [8, 6144, 3], [1_114_112, 6, 2], 2, mamba_block),
                        view(offset + 36_864, [8, 16, 128, 128], [1_114_112, 65_536, 512, 4], 4, mamba_block),
                    ]
        for position, index in enumerate(mla):
            offset = 98_041_856 + position * 574_464
            place([f"layers.{index}.indexer"], offset, 574_464, 71_808)
            views[f"layers.{index}.indexer"] = [view(offset, [136, 32, 132], [4224, 132, 1], 1, 128)]
            place([f"layers.{index}.tail"], offset, 574_464, 71_808)
            views[f"layers.{index}.tail"] = [view(offset, [8, 2, 4, 128], [71_808, 1024, 256, 2], 2, 4)]
        kernels = [64, 4, *[mamba_block] * 4]
    elif layout == "block_outer_short":
        # Generic block-outer layout: one tensor per (group, spec), every group from byte 0, the
        # layer stride is the spec page and the block stride the largest group's bytes per block
        # (kv_cache_utils.py:1606-1632, :1775-1819). G0 packs two specs; G2 is the short last group.
        # A block-outer view cannot split kernel blocks (kv_cache_interface.py:373-387).
        blocks, free, page, block, hash_block = 10, 9, 10_240, 64, 64
        mamba_block = block if prefix_caching else max_model_len
        wide = spec("FullAttentionSpec", 4096, "torch.bfloat16", 1)
        narrow = spec("FullAttentionSpec", 1024, "torch.bfloat16", 1)
        attention = ["attn.0", "small.0", "attn.1", "small.1"]
        mamba = [["mamba.0", "mamba.2"], ["mamba.1"]]
        groups = [
            group(
                attention,
                "UniformTypeKVCacheSpecs",
                block,
                10_240,
                {name: wide if name.startswith("attn.") else narrow for name in attention},
                "attention",
                "torch.bfloat16",
                "vllm.model_executor.layers.attention.Attention",
            ),
            *(
                group(
                    names,
                    "MambaSpec",
                    mamba_block,
                    5120,
                    dict.fromkeys(names, spec("MambaSpec", 5120, None, -1)),
                    "state",
                    None,
                    "vllm.model_executor.layers.mamba.mamba_mixer2.MambaMixer2",
                    mamba_state([[4, 100], [2, 16, 32]], ["torch.bfloat16", "torch.float32"], 4896),
                )
                for names in mamba
            ),
        ]
        scheduler_groups = [
            scheduled(
                groups[0],
                spec_type=wide["spec_type"],
                spec_page_size_bytes=4096,
                dtype="torch.bfloat16",
                layer_specs=dict.fromkeys(attention, wide),
            ),
            *(scheduled(item) for item in groups[1:]),
        ]
        for names, offset, layer_stride, shape, strides in (
            (["attn.0", "attn.1"], 0, 4096, [10, 1, 64, 32], [10_240, 4096, 64, 2]),
            (["small.0", "small.1"], 8192, 1024, [10, 1, 64, 8], [10_240, 1024, 16, 2]),
            *((names, 0, 5120, None, None) for names in mamba),
        ):
            place(names, offset, layer_stride, 10_240)
            for position, name in enumerate(names):
                region = offset + position * layer_stride
                views[name] = (
                    [view(region, shape, strides, 2, block)]
                    if shape is not None
                    else [
                        view(region, [10, 4, 100], [10_240, 200, 2], 2, mamba_block),
                        view(region + 800, [10, 2, 16, 32], [10_240, 2048, 128, 4], 4, mamba_block),
                    ]
                )
        kernels = [block, mamba_block, mamba_block]
    else:
        blocks, free, page, block, hash_block = 10, 9, 16_384, 64, 64
        mamba_block = block if prefix_caching else max_model_len
        # layer_outer_short ends with a short last group (kv_cache_utils.py:1546-1592).
        short = layout == "layer_outer_short"
        layers = [["attn.0", "attn.1"], ["mamba.0", "mamba.2"], ["mamba.1"] if short else ["mamba.1", "mamba.3"]]
        state_spec = spec("MambaSpec", 8192, None, -1)
        groups = [
            group(
                layers[0],
                "FullAttentionSpec",
                block,
                8192,
                dict.fromkeys(layers[0], spec("FullAttentionSpec", 8192, "torch.bfloat16", 1)),
                "attention",
                "torch.bfloat16",
                "vllm.model_executor.layers.attention.Attention",
            ),
            *(
                group(
                    names,
                    "MambaSpec",
                    mamba_block,
                    8192,
                    dict.fromkeys(names, state_spec),
                    "state",
                    None,
                    "vllm.model_executor.layers.mamba.mamba_mixer2.MambaMixer2",
                    mamba_state([[4, 100], [2, 30, 30]], ["torch.bfloat16", "torch.float32"], 8000),
                )
                for names in layers[1:]
            ),
        ]
        scheduler_groups = [scheduled(item) for item in groups]
        for names in layers:
            # Layer-outer: layer l of every group's tensor holds its pages of every block.
            place(names, 0, 81_920, 8192)
            for position, name in enumerate(names):
                region = position * 81_920
                views[name] = (
                    [view(region, [40, 2, 16, 32], [2048, 1024, 64, 2], 2, 16)]
                    if name.startswith("attn.")
                    else [
                        view(region, [10, 4, 100], [8192, 200, 2], 2, mamba_block),
                        view(region + 800, [10, 2, 30, 30], [8192, 3600, 120, 4], 4, mamba_block),
                    ]
                )
        kernels = [16, mamba_block, mamba_block]
    for allocation in allocations:
        allocation["size"] = blocks * page
    scheduler_block = math.lcm(*(item["block_size_tokens"] for item in groups))
    hash_block = hash_block if prefix_caching else scheduler_block
    worker = {
        "num_blocks": blocks,
        "available_cache_bytes": blocks * page + 256,
        "allocated_cache_bytes": blocks * page,
        "storages": [{"storage_id": "s0", "device": "cuda:0", "pointer": 1024, "size_bytes": blocks * page}],
        "tensor_allocations": allocations,
        "groups": groups,
        "pool_id": "pool",
        "prefix_caching": {"enable_prefix_caching": prefix_caching, "mamba_cache_mode": mode},
        "semantics": {
            "allocation": "shared_block_pool",
            "storage": "hbm",
            "retention": "full_or_window",
            "prefix_reuse": False,
            "offload": False,
            "speculative": False,
        },
        "model_runner": {"class": "vllm.v1.worker.gpu.model_runner.GPUModelRunner", "kernel_block_sizes": kernels},
        "layer_tensors": views,
    }
    scheduler = {
        "num_blocks": blocks,
        "initial_free_blocks": free,
        "reserved_blocks": blocks - free,
        "null_block_id": 0,
        "watermark_blocks": 0,
        "pool_count": 1,
        "groups": scheduler_groups,
        "pool_id": "pool",
        "permanent_reserved_block_ids": [0],
        "group_pool_ids": ["pool"] * len(groups),
        "block_size": scheduler_block,
        "hash_block_size": hash_block,
        "mamba_cache_mode": mode,
        # Engine-core startup capacity: it follows num_blocks and the Mamba mode, and nothing compares it.
        "kv_cache_size_tokens": free * scheduler_block,
        "kv_cache_max_concurrency": free * scheduler_block / max_model_len,
        "prefix_caching": {
            "enable_prefix_caching": prefix_caching,
            "mamba_cache_mode": mode,
            "block_size": scheduler_block,
            "hash_block_size": hash_block,
        },
    }
    config = _vllm_config(dp=dp)
    vars(config.cache_config).update(
        cache_dtype="fp8_e4m3" if layout == "glm_tp4" else "bfloat16",
        enable_prefix_caching=prefix_caching,
        block_size=block,
        mamba_cache_mode=mode,
        mamba_block_size=mamba_block,
        prefix_match_unit=None,
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="auto",
    )
    resolved = {
        **legacy.resolved_config(config, hybrid_evidence=True),
        "speculative_config": None,
        "kv_transfer_config": None,
    }
    return worker, scheduler, resolved


def observation_fixture(
    root: Path,
    *,
    packed=False,
    planar=False,
    dense=False,
    tp=2,
    version=None,
    layout="attention",
    decode_prefix_caching=True,
):
    """Synthetic source, rank and runtime records; no GPU or runtime qualification.

    ``layout`` "attention" writes vLLM 0.28.0 records by default; "glm_tp4", "mamba_hybrid",
    "layer_outer_short" and "block_outer_short" write vLLM 0.31.0 records (see
    ``_v031_records``), and ``decode_prefix_caching=False`` runs their decode phase without
    prefix caching.
    """
    version = version or ("0.28.0" if layout == "attention" else "0.31.0")
    root.mkdir(parents=True, exist_ok=True)
    path, manifest = _manifest(root)
    manifest["runtime"]["version"] = version
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
            "framework_version": version,
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
            # GLM-5.3-Flash runs an fp8_e4m3 cache (audit Q1).
            "kvcache_quant_mode": "fp8" if layout == "glm_tp4" else "bfloat16",
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
        v031 = (
            None
            if layout == "attention"
            else _v031_records(
                layout, prefix_caching=phase == "prefill" or decode_prefix_caching, dp=launch["topology"]["dp"]
            )
        )
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
            resolved = legacy.resolved_config(_vllm_config(dp=launch["topology"]["dp"]))
            if v031 is not None:
                # vLLM 0.31.0 records replace the attention cache and add the hybrid cache settings.
                worker_cache, scheduler_cache, resolved = copy.deepcopy(v031)
                cache = worker_cache if worker else scheduler_cache
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
                "resolved_config": resolved,
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


@pytest.mark.parametrize("worker_type,mode", [("prefill", "PIECEWISE"), ("decode", "FULL_DECODE_ONLY")])
def test_role_observations_keep_their_own_graph_settings_and_capacity(tmp_path, worker_type, mode):
    path, launches = observation_fixture(tmp_path)
    launch = launches["tp2"]
    launch["worker_type"] = worker_type
    launch["collection"]["cudagraph_mode"] = mode
    _replace_launch(path, launches)
    index = json.loads(path.read_text())
    phases = index["configurations"]["tp2"]["attempts"][0]["phases"]
    phases.pop("decode" if worker_type == "prefill" else "prefill")
    path.write_text(json.dumps(index))
    _mutate(path, lambda record: record["resolved_config"]["compilation_config"].update(cudagraph_mode=mode))
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result
    assert result["resources"]["runtime_memory"]["kv_cache_bytes"] == (88 if worker_type == "prefill" else 83) * 128
    assert {value["phase"] for value in result["provenance"]["rank_capacities"]} == {worker_type}
    # A correctly hashed worker record still cannot contradict its role's
    # accepted engine settings. Other roles are not substitutes for evidence.
    _mutate(path, lambda record: record["resolved_config"]["compilation_config"].update(cudagraph_mode="FULL"))
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert "reviewed serving role" in str(result["diagnostics"])


def test_aggregated_generic_graph_settings_are_checked_in_both_phases(tmp_path):
    path, launches = observation_fixture(tmp_path)
    launch = launches["tp2"]
    launch["worker_type"] = "aggregated"
    launch["collection"].update(cudagraph_mode="FULL_AND_PIECEWISE", cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64])
    _replace_launch(path, launches)
    _mutate(
        path,
        lambda record: record["resolved_config"]["compilation_config"].update(
            cudagraph_mode="FULL_AND_PIECEWISE",
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64],
            max_cudagraph_capture_size=64,
        ),
    )
    assert validate_observations(path, launches)["tp2"]["status"] == "complete"
    _mutate(
        path,
        lambda record: record["resolved_config"]["compilation_config"].update(
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32], max_cudagraph_capture_size=32
        )
        if record["phase"] == "decode"
        else None,
    )
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert "reviewed serving role" in str(result["diagnostics"])


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


@pytest.mark.parametrize(
    "executor,model_cache,cache_root",
    [
        ("slurm", None, "/root/.cache/huggingface/hub"),
        ("slurm", None, "/custom/cache"),
        ("kubernetes", "models", "/cache"),
    ],
)
def test_cached_hf_snapshot_keeps_public_identity_and_loaded_config_binding(
    tmp_path, executor, model_cache, cache_root
):
    path, launches = observation_fixture(tmp_path)
    launch = launches["tp2"]
    revision = "a" * 40
    launch["identity"]["model_revision"] = revision
    launch["deployment"].update(executor=executor, model_cache=model_cache)
    _replace_launch(path, launches)
    snapshot = f"{cache_root}/models--example--model/snapshots/{revision}"
    _mutate(
        path,
        lambda record: record["resolved_config"]["model_config"].update(
            model=snapshot, revision=revision, loaded_config_commit_hash=revision
        ),
    )
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    assert result["provenance"]["launch"]["identity"]["model"] == "example/model"
    assert result["provenance"]["runtime_settings"]["model_config"]["model"] == snapshot


@pytest.mark.parametrize(
    "corruption",
    [
        "repo",
        "snapshot_revision",
        "symbolic_revision",
        "local_path",
        "relative_path",
        "nested_path",
        "parent_path",
        "blob_path",
        "explicit_checkpoint",
        "hash",
        "sidecar",
        "revision",
        "loaded_revision",
        "identity",
    ],
)
def test_cached_hf_snapshot_does_not_weaken_model_identity(tmp_path, corruption):
    path, launches = observation_fixture(tmp_path)
    launch = launches["tp2"]
    revision = "main" if corruption == "symbolic_revision" else "a" * 40
    launch["identity"]["model_revision"] = revision
    launch["model_config"]["source_files"] = {"hf_quant_config.json": "b" * 64}
    if corruption == "explicit_checkpoint":
        launch["deployment"].update(executor="kubernetes", model_cache="models:/cache:checkpoint")
    _replace_launch(path, launches)

    def mismatch(record):
        model = record["resolved_config"]["model_config"]
        snapshot = f"/cache/models--example--model/snapshots/{revision}"
        if corruption == "repo":
            snapshot = snapshot.replace("models--example--model", "models--another--model")
        elif corruption == "snapshot_revision":
            snapshot = snapshot.replace(revision, "c" * 40)
        elif corruption == "local_path":
            snapshot = "/unexpected/checkpoint"
        elif corruption == "relative_path":
            snapshot = snapshot.lstrip("/")
        elif corruption == "nested_path":
            snapshot += "/checkpoint"
        elif corruption == "parent_path":
            snapshot = "/cache/../" + snapshot.lstrip("/")
        elif corruption == "blob_path":
            snapshot = snapshot.replace("/snapshots/", "/blobs/")
        model.update(model=snapshot, revision=revision, loaded_config_commit_hash=revision)
        record["model_config_source_files"] = {"hf_quant_config.json": "b" * 64}
        if corruption == "hash":
            record["model_config_sha256"] = "f" * 64
        elif corruption == "sidecar":
            record["model_config_source_files"]["hf_quant_config.json"] = "f" * 64
        elif corruption == "revision":
            model["revision"] = "f" * 40
        elif corruption == "loaded_revision":
            model["loaded_config_commit_hash"] = "f" * 40
        elif corruption == "identity":
            record["identity"]["model"] = "another/model"

    _mutate(path, mismatch)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["resources"] is None


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


def _phase_records(path, phase, kind):
    index = json.loads(path.read_text())
    artifacts = index["configurations"]["tp2"]["attempts"][0]["phases"][phase]["artifacts"]
    records = [json.loads((path.parent / artifact["path"]).read_text()) for artifact in artifacts]
    return [record for record in records if record["kind"] == kind]


def _evidence(result):
    memory = result["resources"]["runtime_memory"]
    return tuple(memory.get(name) for name in ("enable_prefix_caching", "mamba_cache_mode", "hash_block_size_tokens"))


def glm_tp4_cache_groups():
    """The imported cache groups of GLM-5.3-Flash at TP 4: every page is one 13,045,120-byte pool block."""
    page = 13_045_120
    return [
        {
            "name": "runtime_group_0",
            "kind": "attention",
            "num_layers": 22,
            "block_size_tokens": 2176,
            "page_size_bytes": page,
            "sliding_window": None,
        },
        {
            "name": "runtime_group_1",
            "kind": "state",
            "num_layers": 11,
            "block_size_tokens": 4,
            "page_size_bytes": page,
            "sliding_window": None,
            "blocks_per_request": 1,
        },
        *(
            {
                "name": f"runtime_group_{index}",
                "kind": "state",
                "num_layers": layers,
                "block_size_tokens": 2176,
                "page_size_bytes": page,
                "sliding_window": None,
                "blocks_per_request": 1,
            }
            for index, layers in zip(range(2, 6), (9, 9, 8, 8), strict=True)
        ),
    ]


def test_glm_tp4_observations_import_state_groups(tmp_path):
    path, launches = observation_fixture(tmp_path, layout="glm_tp4")
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    resources = result["resources"]
    assert resources["cache_groups"] == glm_tp4_cache_groups()
    assert resources["runtime_memory"]["kv_cache_bytes"] == 7 * 13_045_120 == 91_315_840
    assert _evidence(result) == (True, "align", 2176)
    FpmResourceProfile.model_validate(resources)


def test_mamba_hybrid_observations_import_state_groups(tmp_path):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid")
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    assert result["resources"]["cache_groups"] == [
        {
            "name": "runtime_group_0",
            "kind": "attention",
            "num_layers": 2,
            "block_size_tokens": 64,
            "page_size_bytes": 16_384,
            "sliding_window": None,
        },
        *(
            {
                "name": f"runtime_group_{index}",
                "kind": "state",
                "num_layers": 2,
                "block_size_tokens": 64,
                "page_size_bytes": 16_384,
                "sliding_window": None,
                "blocks_per_request": 1,
            }
            for index in (1, 2)
        ),
    ]
    assert result["resources"]["runtime_memory"]["kv_cache_bytes"] == 9 * 16_384


def test_aggregated_phases_with_different_prefix_caching_import_the_prefill_values(tmp_path):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid", decode_prefix_caching=False)
    # The engine core sets a different startup capacity in each phase; nothing compares it (rulings R-D, R-O).
    capacity = {
        phase: _phase_records(path, phase, "scheduler")[0]["cache"]["kv_cache_size_tokens"]
        for phase in ("prefill", "decode")
    }
    assert capacity["prefill"] != capacity["decode"]
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    assert _evidence(result) == (True, "align", 64)
    provenance = result["provenance"]
    assert provenance["prefix_caching"] == {
        "prefill": {"enable_prefix_caching": True, "mamba_cache_mode": "align", "hash_block_size_tokens": 64},
        "decode": {"enable_prefix_caching": False, "mamba_cache_mode": "none", "hash_block_size_tokens": 4096},
    }
    assert result["resources"]["cache_groups"][1]["block_size_tokens"] == 64
    assert provenance["phase_geometry_differences"] == [
        "decode: runtime_group_1 block_size_tokens 4096 != 64",
        "decode: runtime_group_2 block_size_tokens 4096 != 64",
    ]
    # Decision 4 and rulings R-D, R-D+: the mode-dependent settings leave the compared runtime settings.
    cache = provenance["runtime_settings"]["cache_config"]
    assert not {
        "enable_prefix_caching",
        "mamba_cache_mode",
        "mamba_block_size",
        "prefix_match_unit",
        "block_size",
    } & set(cache)
    assert (cache["mamba_cache_dtype"], cache["mamba_ssm_cache_dtype"]) == ("auto", "auto")


def test_decode_role_record_holds_the_decode_values(tmp_path):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid", decode_prefix_caching=False)
    launches["tp2"]["worker_type"] = "decode"
    _replace_launch(path, launches)
    index = json.loads(path.read_text())
    index["configurations"]["tp2"]["attempts"][0]["phases"].pop("prefill")
    path.write_text(json.dumps(index))
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    assert _evidence(result) == (False, "none", 4096)
    assert result["provenance"]["phase_geometry_differences"] == []


def test_mamba_cache_mode_all_is_incomplete(tmp_path):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid")
    _mutate(path, lambda record: record["cache"]["prefix_caching"].update(mamba_cache_mode="all"))
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert "mamba_cache_mode all is not modeled; run the probe with prefix caching off (none) or on (align)" in (
        " ".join(result["diagnostics"])
    )


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ("align_without_prefix_caching", "mamba_cache_mode align requires enable_prefix_caching=true"),
        ("one_worker_disagrees", "prefill prefix-caching settings across ranks"),
        ("state_mode_disagrees", "prefill Mamba cache mode"),
    ],
)
def test_prefix_caching_evidence_must_be_modeled_and_agree_across_ranks(tmp_path, change, diagnostic):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid")

    def edit(record):
        if change == "align_without_prefix_caching":
            record["cache"]["prefix_caching"]["enable_prefix_caching"] = False
        elif change == "one_worker_disagrees" and (record["kind"], record["dp_rank"], record["tp_rank"]) == (
            "worker",
            1,
            1,
        ):
            record["cache"]["prefix_caching"]["mamba_cache_mode"] = "none"
        elif change == "state_mode_disagrees":
            # Every MambaSpec records the mode of its own cache config (mamba/abstract.py:76).
            record["cache"]["groups"][1]["state"]["mamba_cache_mode"] = "none"

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert diagnostic in " ".join(result["diagnostics"])


@pytest.mark.parametrize("change", ["speculative", "checkpoint"])
def test_speculative_or_checkpoint_mamba_groups_are_incomplete(tmp_path, change):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid")

    def edit(record):
        if record["phase"] != "prefill":
            return
        for group in record["cache"]["groups"]:
            state = group.get("state", {})
            if "num_speculative_blocks" not in state:
                continue
            if change == "speculative":
                # What a speculative run records (ruling R-F): the marker and one more block per request.
                state.update(num_speculative_blocks=1, blocks_per_request=2)
            else:
                state["num_prefill_checkpoint_blocks"] = 1
        if change == "speculative" and record["kind"] == "worker":
            record["cache"]["semantics"]["speculative"] = True

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    diagnostics = " ".join(result["diagnostics"])
    if change == "speculative":
        assert "supported allocation semantics" in diagnostics
    else:
        assert "mamba prefill checkpoint blocks are not modeled: ['mamba.0', 'mamba.2']" in diagnostics


@pytest.mark.parametrize(
    "layout,names",
    [
        # A MambaSpec without speculative blocks holds one block per request (kv_cache_interface.py:1048).
        ("mamba_hybrid", ["mamba.0", "mamba.2"]),
        # A KpoolTailSpec holds one block per request (kv_cache_interface.py:1019-1025).
        ("glm_tp4", [f"layers.{index}.tail" for index in range(3, 45, 4)]),
    ],
    ids=["mamba", "tail"],
)
def test_state_group_blocks_per_request_must_follow_its_spec(tmp_path, layout, names):
    path, launches = observation_fixture(tmp_path, layout=layout)

    def edit(record):
        # Every record of the group gets the same count, so only the spec rule (decision 6) can refuse it.
        state = record["cache"]["groups"][1]["state"]
        assert (state["blocks_per_request"], state.get("num_speculative_blocks", 0)) == (1, 0)
        state["blocks_per_request"] = 2

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [f"tp2: runtime state cache group blocks_per_request contradicts its spec: {names}"]


@pytest.mark.parametrize("change", [None, "latent_page", "block_size"])
def test_uniform_type_groups_compare_their_first_spec_with_the_scheduler(tmp_path, change):
    path, launches = observation_fixture(tmp_path, layout="glm_tp4")
    # The live scheduler copy of G0 is its first inner spec, the indexer (ruling R-S).
    scheduled = _phase_records(path, "prefill", "scheduler")[0]["cache"]["groups"][0]
    assert scheduled["spec_page_size_bytes"] == 71_808
    assert scheduled["layer_specs"]["layers.3.indexer"]["cache_role"] == "INDEXER"

    def edit(record):
        if record["kind"] != "scheduler":
            return
        if change == "latent_page":
            record["cache"]["groups"][0]["spec_page_size_bytes"] = 1_114_112
        elif change == "block_size":
            record["cache"]["groups"][2]["block_size_tokens"] = 4096

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    if change is None:
        assert result["status"] == "complete", result["diagnostics"]
        return
    assert result["status"] == "incomplete"
    assert "geometry disagrees" in " ".join(result["diagnostics"])


def test_unknown_spec_class_is_incomplete_with_its_name(tmp_path):
    path, launches = observation_fixture(tmp_path, layout="glm_tp4")

    def edit(record):
        group = record["cache"]["groups"][2]
        group["spec_type"] = "vllm.v1.kv_cache_interface.HiSparseHotSpec"
        if "kind" in group:  # Only the worker records the kind (ruling R-B).
            group["kind"] = "hisparse"

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    diagnostics = " ".join(result["diagnostics"])
    assert "runtime cache group kind is unsupported" in diagnostics
    assert "vllm.v1.kv_cache_interface.HiSparseHotSpec" in diagnostics


@pytest.mark.parametrize(
    "layout,groups",
    [
        ("layer_outer_short", [("attention", 2, 16_384), ("state", 2, 16_384), ("state", 1, 16_384)]),
        ("block_outer_short", [("attention", 4, 10_240), ("state", 2, 10_240), ("state", 1, 10_240)]),
    ],
)
def test_v031_generic_layouts_with_a_short_last_group_import_complete(tmp_path, layout, groups):
    # Overlapping tensors of different groups differ in layer count, and block-outer ones in offset (ruling R-U).
    path, launches = observation_fixture(tmp_path, layout=layout)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    resources = result["resources"]
    assert [(item["kind"], item["num_layers"], item["page_size_bytes"]) for item in resources["cache_groups"]] == groups
    assert resources["runtime_memory"]["kv_cache_bytes"] == 9 * groups[0][2]


def _move(cache, layers, shift):
    """Move an allocation and its layer views by ``shift`` bytes."""
    allocation = next(item for item in cache["tensor_allocations"] if item["layers"] == layers)
    allocation["offset"] += shift
    for name in layers:
        for view in cache["layer_tensors"][name]:
            view["storage_offset_bytes"] += shift


@pytest.mark.parametrize(
    "case,layout,diagnostic",
    [
        ("region", "mamba_hybrid", "allocation layers exceed their block stride or the backing storage"),
        (
            "same_group",
            "glm_tp4",
            "cache pages of one block overlap for layers of one group: layers.3.mla, layers.7.mla",
        ),
        ("cross_block", "layer_outer_short", "cache pages of different block IDs overlap: attn.0, mamba.1"),
        ("view", "glm_tp4", "layer view lies outside the page of its layer: layers.0.kda"),
        ("block_outer_view", "block_outer_short", "layer view lies outside the page of its layer: attn.0"),
    ],
)
def test_v031_layout_rules_reject_region_overflow_and_same_group_aliasing(tmp_path, case, layout, diagnostic):
    path, launches = observation_fixture(tmp_path, layout=layout)

    def edit(record):
        if record["kind"] != "worker":
            return
        cache = record["cache"]
        if case == "region":
            # Layer 1 of each two-layer tensor then ends past the 163,840-byte storage.
            for allocation in cache["tensor_allocations"]:
                allocation["layer_stride"] = 90_112
        elif case == "same_group":
            # Two G0 layers on one region would hold the same pool block for one request.
            _move(cache, ["layers.7.mla"], -8_912_896)
        elif case == "cross_block":
            # Half a page off, block b of mamba.1 straddles blocks b and b + 1 of attn.0.
            _move(cache, ["mamba.1"], 4096)
        elif case == "view":
            cache["layer_tensors"]["layers.0.kda"][0]["storage_offset_bytes"] += 1_114_112
        else:
            # The block-outer block holds attn.1 right after attn.0, so this view enters the page of attn.1.
            cache["layer_tensors"]["attn.0"][0]["storage_offset_bytes"] += 1024

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["resources"] is None
    assert result["diagnostics"] == [f"tp2: {diagnostic}"]


_MAMBA_ALL = "mamba_cache_mode all is not modeled; run the probe with prefix caching off (none) or on (align)"


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        # Case A of the task review: no evidence key, Mamba groups in mode all.
        ("no_evidence_mode_all", _MAMBA_ALL),
        ("no_evidence", "runtime observation requires complete prefix-caching evidence"),
        ("no_scheduler_evidence", "runtime observation requires complete prefix-caching evidence"),
        ("mode_all_with_evidence", _MAMBA_ALL),
    ],
)
def test_v031_records_require_prefix_caching_evidence_and_refuse_mamba_mode_all(tmp_path, case, diagnostic):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid")

    def edit(record):
        if case.startswith("no_evidence") or (case == "no_scheduler_evidence" and record["kind"] == "scheduler"):
            record["cache"].pop("prefix_caching")
        if case.endswith("mode_all") or case == "mode_all_with_evidence":
            for group in record["cache"]["groups"]:
                if "mamba_cache_mode" in group.get("state", {}):
                    group["state"]["mamba_cache_mode"] = "all"

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [f"tp2: {diagnostic}"]


@pytest.mark.parametrize(
    "shape", ["scheduler_without_hash_block_size", "enable_prefix_caching_string", "unknown_mamba_cache_mode"]
)
def test_v031_malformed_prefix_caching_evidence_is_incomplete(tmp_path, shape):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid")

    def edit(record):
        # The evidence key is present, so only the malformed-evidence rule can refuse it.
        evidence = record["cache"]["prefix_caching"]
        if shape == "scheduler_without_hash_block_size":
            if record["kind"] == "scheduler":
                evidence.pop("hash_block_size")
        elif shape == "enable_prefix_caching_string":
            evidence["enable_prefix_caching"] = "true"
        else:
            evidence["mamba_cache_mode"] = "auto"

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == ["tp2: runtime observation requires complete prefix-caching evidence"]


@pytest.mark.parametrize("spec", ["MLAAttentionSpec", "MambaSpec"])
def test_legacy_records_refuse_classes_outside_the_legacy_allow_list(tmp_path, spec):
    path, launches = observation_fixture(tmp_path)

    def edit(record):
        group = record["cache"]["groups"][0]
        group["spec_type"] = f"vllm.v1.kv_cache_interface.{spec}"
        if spec == "MambaSpec":
            group["state"] = {"blocks_per_request": 1}
            if record["kind"] == "worker":
                group["kind"] = "state"

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [f"tp2: unsupported runtime cache spec: 'vllm.v1.kv_cache_interface.{spec}'"]


def _restride(cache, layers, field, value):
    """Change an allocation's layer or block stride and move its layer views to match."""
    allocation = next(item for item in cache["tensor_allocations"] if item["layers"] == layers)
    shift = value - allocation[field]
    allocation[field] = value
    for position, name in enumerate(layers):
        for view in cache["layer_tensors"][name]:
            if field == "layer_stride":
                view["storage_offset_bytes"] += position * shift
            else:
                view["stride_bytes"][0] = value


@pytest.mark.parametrize(
    "case,layout,diagnostic",
    [
        ("block_stride", "layer_outer_short", "aliased cache allocations must share one block stride"),
        ("views_overlap", "mamba_hybrid", "views of one layer overlap inside the block: mamba.0"),
        ("views_missing", "mamba_hybrid", "physical layer views are missing: attn.0"),
        ("two_storages", "mamba_hybrid", "layer-strided cache allocations require one backing storage"),
        (
            "kind_contradicts",
            "mamba_hybrid",
            "runtime cache group kind contradicts its spec classes: "
            "'attention' ['vllm.v1.kv_cache_interface.MambaSpec']",
        ),
        ("layer_specs", "mamba_hybrid", "runtime memory cache lacks complete layer spec evidence"),
        ("state_window", "mamba_hybrid", "state cache groups cannot have a sliding_window"),
        # Only the pairing of a tensor with itself sees two of its own layers overlap.
        ("self_pair", "block_outer_short", "cache pages of one block overlap for layers of one group: attn.0, attn.1"),
        ("phases_mismatch", "attention", "runtime observation prefix-caching evidence across phases mismatch"),
    ],
)
def test_v031_importer_diagnostics_are_reachable(tmp_path, case, layout, diagnostic):
    path, launches = observation_fixture(tmp_path, layout=layout)

    def edit(record):
        cache, worker = record["cache"], record["kind"] == "worker"
        if case == "block_stride" and worker:
            _restride(cache, ["mamba.1"], "block_stride", 16_384)
        elif case == "views_overlap" and worker:
            cache["layer_tensors"]["mamba.0"][1]["storage_offset_bytes"] = 400
        elif case == "views_missing" and worker:
            cache["layer_tensors"]["attn.0"] = []
        elif case == "two_storages" and worker:
            cache["storages"].append(
                {"storage_id": "s1", "device": "cuda:0", "pointer": 999_999, "size_bytes": 163_840}
            )
            cache["allocated_cache_bytes"] += 163_840
            cache["available_cache_bytes"] += 163_840
        elif case == "kind_contradicts" and worker:
            cache["groups"][1]["kind"] = "attention"
        elif case == "layer_specs":
            cache["groups"][1].pop("layer_specs")
        elif case == "state_window":
            cache["groups"][1]["sliding_window"] = 4
        elif case == "self_pair" and worker:
            _restride(cache, ["attn.0", "attn.1"], "layer_stride", 2048)
        elif case == "phases_mismatch" and record["phase"] == "decode":
            # Legacy-shaped records with evidence in one phase only.
            cache["prefix_caching"] = {"enable_prefix_caching": True, "mamba_cache_mode": "none"}
            if not worker:
                cache["prefix_caching"].update(block_size=16, hash_block_size=16)

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [f"tp2: {diagnostic}"]


def test_v031_state_group_layer_specs_stay_compared_across_phases(tmp_path):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid", decode_prefix_caching=False)

    def edit(record):
        # A layer spec field that followed the Mamba block size would differ between the phases (ruling R-W).
        for group in record["cache"]["groups"]:
            if "mamba_cache_mode" in group.get("state", {}):
                for spec in group["layer_specs"].values():
                    spec["tokens_per_state"] = group["block_size_tokens"]

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == ["tp2: runtime observation cache geometry across phases/ranks mismatch"]


@pytest.mark.parametrize("packed", [False, True])
def test_legacy_bundle_imports_are_unchanged_by_the_state_mapping(tmp_path, packed):
    path, launch = observation_fixture(tmp_path, packed=packed)
    result = validate_observations(path, launch)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    resources = result["resources"]
    assert resources["cache_groups"] == [
        {
            "name": f"runtime_group_{index}",
            "kind": "attention",
            "num_layers": 2 if packed and index == 0 else 1,
            "block_size_tokens": 16,
            "page_size_bytes": 128,
            "sliding_window": None,
        }
        for index in range(2)
    ]
    assert resources["runtime_memory"]["kv_cache_bytes"] == 83 * 128
    assert not {"enable_prefix_caching", "mamba_cache_mode", "hash_block_size_tokens"} & set(
        resources["runtime_memory"]
    )
    provenance = result["provenance"]
    assert len(provenance["rank_capacities"]) == 8
    assert len(provenance["artifacts"]) == 12
    # Ruling R-M: the provenance keeps the keys that 5bf93cb0 writes, because finalized campaigns compare the
    # runtime memory provenance byte for byte.
    assert not {"prefix_caching", "phase_geometry_differences"} & set(provenance)
    assert (
        set(provenance)
        == set(json.loads(resources["runtime_memory"]["provenance"]))
        == {
            "source",
            "observation_schema",
            "configuration",
            "attempt_id",
            "bundle_sha256",
            "runtime",
            "launch",
            "runtime_settings",
            "rank_capacities",
            "launch_artifacts",
            "artifacts",
            "runtime_artifacts",
            "instrumentation",
            "scope",
            "observations_index",
        }
    )
    # Only enable_prefix_caching leaves the legacy settings, as before.
    assert set(provenance["runtime_settings"]["cache_config"]) == {
        "cache_dtype",
        "gpu_memory_utilization",
        "kv_cache_memory_bytes",
    }


@pytest.mark.parametrize(
    "layout,field,value,complete",
    [
        ("mamba_hybrid", "block_size", 16, True),
        ("mamba_hybrid", "mamba_ssm_cache_dtype", "float32", False),
        ("attention", "enable_prefix_caching", False, False),
    ],
)
def test_v031_worker_block_size_is_not_compared_with_the_scheduler(tmp_path, layout, field, value, complete):
    path, launches = observation_fixture(tmp_path, layout=layout)
    # vLLM 0.31.0 resolves block_size in the engine-core process only (engine/core.py:341-350, ruling R-D+).
    _mutate(
        path,
        lambda record: record["resolved_config"]["cache_config"].update({field: value})
        if record["kind"] == "worker"
        else None,
    )
    result = validate_observations(path, launches)["tp2"]
    if complete:
        assert result["status"] == "complete", result["diagnostics"]
        return
    assert result["status"] == "incomplete"
    assert "configurations differ between scheduler and worker" in " ".join(result["diagnostics"])
