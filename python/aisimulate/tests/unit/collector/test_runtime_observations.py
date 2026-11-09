# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
from collector.fpm_forward import runtime_memory, runtime_observations
from collector.fpm_forward.runtime import fpm_memory_observer as legacy
from collector.fpm_forward.runtime_instrumentation import load_instrumentation
from collector.fpm_forward.runtime_observations import validate_observations

from aisimulate_core.fpm_profile import FpmResourceProfile

from .test_fpm_forward import _MODEL_CONFIGS
from .test_fpm_profile_collection import no_models_or_timing_data  # noqa: F401
from .test_fpm_runtime_memory import _probe_backed_glm_plan, _vllm_config
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

    ``glm_tp4`` follows the live GLM-5.3-Flash TP4 fake of ``test_runtime_adapter._v031_runtime``:
    G0 lists each indexer before its MLA latent layer, the scheduler copy of G0 is
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

    def reserved(item):
        # The startup count of one request at max_model_len in a group (kv_cache_interface.py:578-583, :1083-1094,
        # :1019-1022): the context in attention blocks, 2 Mamba blocks in align mode and 1 in mode none, 1 tail block.
        if item["kind"] == "attention":
            return -(-max_model_len // item["block_size_tokens"])
        return (2 if mode == "align" else 1) if "mamba_cache_mode" in item["state"] else 1

    # The engine-core startup capacity (kv_cache_utils.py:1105-1116, :2444-2452). Import checks the concurrency.
    concurrency = blocks / sum(reserved(item) for item in groups)
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
        "kv_cache_size_tokens": int(concurrency * max_model_len),
        "kv_cache_max_concurrency": concurrency,
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
    # The GLM layout runs an fp8 KV cache, so collection labels FMHA with the checkpoint-native mode, fp8 for this
    # NVFP4 checkpoint, and the import reads that label from the configuration that the probe stages.
    quantization = {"quant_method": "modelopt", "quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8"}
    model.write_text(
        json.dumps({"model_type": "example", "quantization_config": quantization})
        if layout == "glm_tp4"
        else '{"model_type":"example"}'
    )
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
            "fmha_quant_mode": "fp8" if layout == "glm_tp4" else "bfloat16",
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
    # The import reads the staged configuration only for a KV cache other than bfloat16. The other layouts keep the
    # evidence that they had before the probe staged copies were read.
    if launch["precision"]["kvcache_quant_mode"] != "bfloat16":
        stage_model_config(root / "observations.json")
    return root / "observations.json", {"tp2": launch}


def stage_model_config(index_path):
    """Copy the launch's model configuration and its declared sidecars into every phase, as the probe stages them.

    runtime_probe.stage_runtime_instrumentation writes model-config.json and each declared sidecar into the phase
    directory, whose artifact list then holds them. Earlier copies are replaced.
    """
    index = json.loads(index_path.read_text())
    entry = index["configurations"]["tp2"]
    config = entry["launch"]["model_config"]
    source = Path(config["path"])
    files = {"model-config.json": source, **{name: source.parent / name for name in config.get("source_files", {})}}
    for phase, result in entry["attempts"][0]["phases"].items():
        result["artifacts"] = [item for item in result["artifacts"] if Path(item["path"]).name not in files]
        for name, origin in files.items():
            target = index_path.parent / phase / name
            target.write_bytes(origin.read_bytes())
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            result["artifacts"].append({"path": f"{phase}/{name}", "sha256": digest, "kind": "runtime-artifact"})
    index_path.write_text(json.dumps(index))


def _mutate(index_path, edit):
    index = json.loads(index_path.read_text())
    attempt = index["configurations"]["tp2"]["attempts"][0]
    for phase in attempt["phases"].values():
        for artifact in phase["artifacts"]:
            if artifact.get("kind") != "observation":
                continue
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
    records = [
        json.loads((path.parent / artifact["path"]).read_text())
        for artifact in artifacts
        if artifact.get("kind") == "observation"
    ]
    return [record for record in records if record["kind"] == kind]


def _evidence(result):
    memory = result["resources"]["runtime_memory"]
    return tuple(memory.get(name) for name in ("enable_prefix_caching", "mamba_cache_mode", "hash_block_size_tokens"))


def glm_tp4_cache_groups():
    """The imported cache groups of GLM-5.3-Flash at TP 4: every page is one 13,045,120-byte pool block.

    A block transfer moves the viewed bytes of each layer, without padding: the MLA latent and indexer pages of the 11
    attention layers, 2,048 bytes for each tail ring layer, whose recorded page is padded to the 71,808-byte indexer
    page, and the 1,085,440-byte real page of each KDA layer. Only the tail ring takes no part in prefix caching.
    """
    page = 13_045_120
    return [
        {
            "name": "runtime_group_0",
            "kind": "attention",
            "num_layers": 22,
            "block_size_tokens": 2176,
            "page_size_bytes": page,
            "sliding_window": None,
            "prefix_cacheable": True,
            "transfer_bytes_per_block": 11 * (1_114_112 + 71_808),
        },
        {
            "name": "runtime_group_1",
            "kind": "state",
            "num_layers": 11,
            "block_size_tokens": 4,
            "page_size_bytes": page,
            "sliding_window": None,
            "blocks_per_request": 1,
            "prefix_cacheable": False,
            "transfer_bytes_per_block": 11 * 2_048,
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
                "prefix_cacheable": True,
                "transfer_bytes_per_block": layers * 1_085_440,
                "prefill_checkpoint_blocks": 0,
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
    # The transfer bytes of one block of each group, without padding.
    assert [group["transfer_bytes_per_block"] for group in resources["cache_groups"]] == [
        13_045_120,
        22_528,
        9_768_960,
        9_768_960,
        8_683_520,
        8_683_520,
    ]
    assert resources["runtime_memory"]["kv_cache_bytes"] == 7 * 13_045_120 == 91_315_840
    assert _evidence(result) == (True, "align", 2176)
    FpmResourceProfile.model_validate(resources)


def _glm_tp4_at_probe_scale(tmp_path, concurrency):
    """The glm_tp4 records at the size of the GB300 probes: 13,816 pool blocks and a 1,048,576-token context.

    Every scheduler records ``concurrency`` as its kv_cache_max_concurrency.
    """
    path, launches = observation_fixture(tmp_path, layout="glm_tp4")
    launches["tp2"]["collection"]["max_model_len"] = 1_048_576
    _replace_launch(path, launches)
    factor = 13_816 // 8

    def edit(record):
        record["resolved_config"]["model_config"]["max_model_len"] = 1_048_576
        cache = record["cache"]
        cache["num_blocks"] *= factor
        if record["kind"] == "scheduler":
            cache.update(initial_free_blocks=cache["num_blocks"] - 1, kv_cache_max_concurrency=concurrency)
            return
        # The packaged GB300 memory holds the larger pool. Each layer region and each view grow with the pool.
        record["hardware"]["total_memory_bytes"] = 298_013_687_808
        for name in ("available_cache_bytes", "allocated_cache_bytes"):
            cache[name] *= factor
        cache["storages"][0]["size_bytes"] *= factor
        for allocation in cache["tensor_allocations"]:
            for position, name in enumerate(allocation["layers"]):
                region = allocation["offset"] + position * allocation["layer_stride"]
                for view in cache["layer_tensors"][name]:
                    view["storage_offset_bytes"] += region * (factor - 1)
            for name in ("size", "offset", "layer_stride"):
                allocation[name] *= factor
        for views in cache["layer_tensors"].values():
            for view in views:
                view["shape"][0] *= factor

    _mutate(path, edit)
    return path, launches


_GLM_SCHEDULER = "prefill scheduler of DP rank 0"


@pytest.mark.parametrize(
    "concurrency,diagnostic",
    [
        # The value of the GB300 probes: 13,816 / 491. For one request at 1,048,576 tokens, vLLM counts 482 attention
        # blocks, 2 blocks of each of the 4 KDA groups in align mode and 1 tail block (kv_cache_utils.py:1105-1116).
        (28.138492871690428, None),
        # One block per KDA group gives 487 blocks per request.
        (
            13_816 / 487,
            f"{_GLM_SCHEDULER} records kv_cache_max_concurrency 28.369609856262834, which gives 487.000 blocks per "
            "request, but its cache groups reserve 491",
        ),
        (None, f"{_GLM_SCHEDULER} lacks a positive kv_cache_max_concurrency"),
    ],
    ids=["vllm", "one_block_per_kda_group", "missing"],
)
def test_v031_glm_scheduler_concurrency_must_follow_its_cache_groups(tmp_path, concurrency, diagnostic):
    path, launches = _glm_tp4_at_probe_scale(tmp_path, concurrency)
    result = validate_observations(path, launches)["tp2"]
    if diagnostic is None:
        assert result["status"] == "complete", result["diagnostics"]
        # The kv_cache_bytes that the import of the GB300 probe gave, and the steady count of one block per state group.
        assert result["resources"]["runtime_memory"]["kv_cache_bytes"] == 13_815 * 13_045_120 == 180_218_332_800
        assert result["resources"]["cache_groups"] == glm_tp4_cache_groups()
        return
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [f"tp2: {diagnostic}"]


def _startup_record(mode, specs, concurrency):
    """A vLLM 0.31.0 scheduler record of 13,816 blocks at 1,048,576 tokens, one group per (class, block)."""
    groups = [
        {"spec_type": _V031_SPEC + spec, "block_size_tokens": block}
        | ({"state": {"num_speculative_blocks": 0, "num_prefill_checkpoint_blocks": 0}} if spec == "MambaSpec" else {})
        for spec, block in specs
    ]
    return {
        "phase": "decode",
        "dp_rank": 1,
        "cache": {
            "num_blocks": 13_816,
            "mamba_cache_mode": mode,
            "groups": groups,
            "kv_cache_max_concurrency": concurrency,
        },
        "resolved_config": {"model_config": {"max_model_len": 1_048_576}},
    }


_GLM_SPECS = [("MLAAttentionSpec", 2176), ("KpoolTailSpec", 4), *[("MambaSpec", 2176)] * 4]


@pytest.mark.parametrize(
    "mode,specs,concurrency,blocks",
    [
        # In mode none vLLM counts 1 + k blocks for a Mamba group (kv_cache_interface.py:1093-1094): 482 + 1 + 4 x 1.
        ("none", _GLM_SPECS, 13_816 / 487, None),
        ("none", _GLM_SPECS, 13_816 / 491, 487),
        ("align", _GLM_SPECS, 13_816 / 491, None),
        # vLLM counts one block for a ring (:904-907) and cdiv(1,048,576, 64) for a full-attention group (:578-583).
        ("none", [("FullAttentionSpec", 64), ("CircularBufferSpec", 128)], 13_816 / 16_385, None),
        # vLLM's count for a windowed group also covers the tokens in flight (:840-877), which the check does not model.
        ("align", [("SlidingWindowSpec", 64), ("MambaSpec", 64)], 1.0, None),
    ],
    ids=["none", "none_with_the_align_count", "align", "ring", "windowed"],
)
def test_v031_startup_concurrency_follows_the_mamba_cache_mode(mode, specs, concurrency, blocks):
    record = _startup_record(mode, specs, concurrency)
    if blocks is None:
        runtime_observations._validate_startup_concurrency(record)
        return
    with pytest.raises(ValueError, match=f"but its cache groups reserve {blocks}$"):
        runtime_observations._validate_startup_concurrency(record)


@pytest.mark.parametrize("layout", ["mamba_hybrid", "glm_tp4"])
def test_import_reads_the_staged_model_configuration_not_the_launch_path(tmp_path, layout):
    # Finalization and the collection verifier recheck saved evidence when the launch path can be gone. The fp8 KV
    # cache of the GLM layout takes its FMHA label from the configuration that the probe staged. The bfloat16 KV cache
    # of the hybrid layout needs no staged configuration.
    path, launches = observation_fixture(tmp_path, layout=layout)
    phases = json.loads(path.read_text())["configurations"]["tp2"]["attempts"][0]["phases"]
    staged = [Path(item["path"]).name for phase in phases.values() for item in phase["artifacts"]]
    assert staged.count("model-config.json") == (2 if layout == "glm_tp4" else 0)
    Path(launches["tp2"]["model_config"]["path"]).unlink()
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]


def _sidecar_fixture(tmp_path):
    """GLM evidence of a checkpoint that names its NVFP4 quantization only in its declared hf_quant_config.json."""
    path, launches = observation_fixture(tmp_path, layout="glm_tp4")
    config = launches["tp2"]["model_config"]
    model = Path(config["path"])
    model.write_text('{"model_type":"example"}')
    sidecar = model.parent / "hf_quant_config.json"
    sidecar.write_text(json.dumps({"quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8"}}))
    config.update(
        sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        source_files={sidecar.name: hashlib.sha256(sidecar.read_bytes()).hexdigest()},
    )
    _replace_launch(path, launches)
    _mutate(
        path,
        lambda record: record.update(
            model_config_sha256=config["sha256"], model_config_source_files=config["source_files"]
        ),
    )
    stage_model_config(path)
    return path, launches


def test_fp8_kv_import_reads_the_staged_sidecar_beside_the_staged_configuration(tmp_path):
    path, launches = _sidecar_fixture(tmp_path)
    # Without its sidecar this checkpoint names no quantization, and its FMHA label would be bfloat16. The loader
    # reads the staged copy and the staged sidecar beside it, so the launch files can be gone.
    model = Path(launches["tp2"]["model_config"]["path"])
    model.unlink()
    (model.parent / "hf_quant_config.json").unlink()
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("missing", "runtime phase must stage exactly one model-config.json, found 0"),
        ("second_copy", "runtime phase must stage exactly one model-config.json, found 2"),
        ("other_digest", "staged model-config.json differs from the launch model configuration SHA-256"),
        ("undeclared_sidecar", "staged hf_quant_config.json is not declared in model_config.source_files"),
        ("unlisted_sidecar", "staged hf_quant_config.json is not declared in model_config.source_files"),
        ("unstaged_sidecar", "declared hf_quant_config.json is not staged beside model-config.json"),
        ("other_sidecar_digest", "staged hf_quant_config.json differs from its declared SHA-256"),
    ],
)
def test_fp8_kv_import_refuses_a_missing_or_mismatched_staged_file(tmp_path, case, diagnostic):
    declared = case in {"unstaged_sidecar", "other_sidecar_digest"}
    path, launches = _sidecar_fixture(tmp_path) if declared else observation_fixture(tmp_path, layout="glm_tp4")
    assert validate_observations(path, launches)["tp2"]["status"] == "complete"
    index = json.loads(path.read_text())
    artifacts = index["configurations"]["tp2"]["attempts"][0]["phases"]["prefill"]["artifacts"]
    staged = {Path(item["path"]).name: item for item in artifacts if item["kind"] == "runtime-artifact"}

    def write(relative, payload, *, listed=True):
        target = path.parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        if listed:
            digest = hashlib.sha256(payload).hexdigest()
            artifacts.append({"path": relative, "sha256": digest, "kind": "runtime-artifact"})

    if case == "missing":
        artifacts.remove(staged["model-config.json"])
    elif case == "second_copy":
        write("prefill/copy/model-config.json", (path.parent / "prefill/model-config.json").read_bytes())
    elif case == "other_digest":
        artifacts.remove(staged["model-config.json"])
        write("prefill/model-config.json", b'{"model_type":"other"}')
    elif case in {"undeclared_sidecar", "unlisted_sidecar"}:
        # The loader would merge a sidecar beside the staged copy, so an unlisted one is refused too.
        write("prefill/hf_quant_config.json", b"{}", listed=case == "undeclared_sidecar")
    elif case == "unstaged_sidecar":
        artifacts.remove(staged["hf_quant_config.json"])
    else:
        artifacts.remove(staged["hf_quant_config.json"])
        write("prefill/hf_quant_config.json", b"{}")
    path.write_text(json.dumps(index))
    assert validate_observations(path, launches)["tp2"]["diagnostics"] == [f"tp2: {diagnostic}"]


# The labels of an FP8 block checkpoint such as GLM-5.3-Flash with an fp8 KV cache. Every published GLM-5.3-Flash and
# DeepSeek V4.1 profile with an fp8 KV cache has this FMHA label.
_FP8_BLOCK_PRECISION = {
    "gemm_quant_mode": "fp8_block",
    "moe_quant_mode": "fp8_block",
    "fmha_quant_mode": "fp8",
    "kvcache_quant_mode": "fp8",
}


def _fp8_block_evidence(root, precision):
    """GLM evidence of the packaged GLM-5.3-Flash checkpoint with ``precision``, staged as the probe stages it."""
    path, launches = observation_fixture(root, layout="glm_tp4")
    launch = launches["tp2"]
    model = Path(launch["model_config"]["path"])
    model.write_bytes((_MODEL_CONFIGS / "zai-org--GLM-5.3-Flash_config.json").read_bytes())
    launch["model_config"]["sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
    launch["precision"].update(precision)
    _replace_launch(path, launches)

    def fp8_block(record):
        record["model_config_sha256"] = launch["model_config"]["sha256"]
        # The vLLM quantization of this checkpoint: FP8 weights in 128 x 128 blocks with dynamic activation scales.
        record["resolved_config"]["model_config"]["quantization"] = "fp8"
        record["resolved_config"]["quantization_config"] = {
            "type": "vllm.model_executor.layers.quantization.fp8.Fp8Config",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }

    _mutate(path, fp8_block)
    stage_model_config(path)
    return path, launches


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_one_fp8_block_precision_passes_the_probe_preview_the_import_and_collection_planning(tmp_path):
    from collector.fpm_forward.runtime_probe import build_runtime_probe_plan

    precision = _FP8_BLOCK_PRECISION
    for name in ("plan", "plan-bfloat16"):
        (tmp_path / name).mkdir()
    # Collection planning: the profile deployment has the same labels, and the planner rechecks the probe launch.
    plan, _ = _probe_backed_glm_plan(tmp_path / "plan", precision=precision)
    assert {(cell.fmha_quant_mode, cell.kv_cache_dtype) for cell in plan.cells} == {("fp8", "fp8")}
    # The probe preview of that launch, which runs before any GPU work.
    launch, bundle = plan.runtime_launch, plan.runtime_instrumentation
    assert {name: launch["precision"][name] for name in precision} == precision
    build_runtime_probe_plan("gb300-tp4", launch, bundle)
    # The import of staged evidence with the same labels.
    path, launches = _fp8_block_evidence(tmp_path / "import", precision)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]

    # The FMHA label of the compute dtype fails every step, and each text names the checkpoint-native fp8.
    wrong = {**precision, "fmha_quant_mode": "bfloat16"}
    with pytest.raises(ValueError) as error:
        build_runtime_probe_plan("gb300-tp4", {**launch, "precision": {**launch["precision"], **wrong}}, bundle)
    assert str(error.value) == "precision.fmha_quant_mode='bfloat16' does not match checkpoint-native 'fp8'"
    path, launches = _fp8_block_evidence(tmp_path / "import-bfloat16", wrong)
    assert validate_observations(path, launches)["tp2"]["diagnostics"] == [
        "tp2: FMHA precision 'bfloat16' differs from the checkpoint-native 'fp8' for KV-cache dtype 'fp8'"
    ]
    with pytest.raises(ValueError) as error:
        _probe_backed_glm_plan(tmp_path / "plan-bfloat16", precision=wrong)
    assert str(error.value).startswith(
        "FPM collection profile identity mismatch: fmha_quant_mode: resolved='fp8', profile='bfloat16';"
    )


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
            "prefix_cacheable": True,
            "transfer_bytes_per_block": 2 * 8192,
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
                "prefix_cacheable": True,
                # Each Mamba layer moves its 8,000-byte real page, not its 8,192-byte padded page.
                "transfer_bytes_per_block": 2 * 8000,
                "prefill_checkpoint_blocks": 0,
            }
            for index in (1, 2)
        ),
    ]
    assert result["resources"]["runtime_memory"]["kv_cache_bytes"] == 9 * 16_384


def test_aggregated_phases_with_different_prefix_caching_import_the_prefill_values(tmp_path):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid", decode_prefix_caching=False)
    # The engine core sets a different startup capacity in each phase. Import checks each one against the Mamba cache
    # mode of its own phase, and it does not compare the phases.
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
    # The mode-dependent settings and the engine-core block size leave the compared runtime settings.
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
                # What a speculative run records: the marker and one more block per request.
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
        # Every record of the group gets the same count, so only the check against the spec's count can refuse it.
        state = record["cache"]["groups"][1]["state"]
        assert (state["blocks_per_request"], state.get("num_speculative_blocks", 0)) == (1, 0)
        state["blocks_per_request"] = 2

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [f"tp2: runtime state cache group blocks_per_request contradicts its spec: {names}"]


@pytest.mark.parametrize("uniform", [False, True], ids=["mamba", "uniform"])
@pytest.mark.parametrize("field", [None, "num_prefill_checkpoint_blocks", "num_speculative_blocks"])
def test_v031_mamba_state_groups_require_their_block_counts(tmp_path, field, uniform):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid")

    def edit(record):
        group = record["cache"]["groups"][1]
        if uniform and record["kind"] == "worker":
            # The worker groups the two Mamba layers as a UniformTypeKVCacheSpecs, whose page holds both layer
            # pages. The scheduler copy keeps the first inner spec (kv_cache_utils.py:2434-2440).
            group.update(spec_type=_V031_SPEC + "UniformTypeKVCacheSpecs", spec_page_size_bytes=16_384)
        if field is not None:
            # Every worker and scheduler Mamba state lacks the count, so the records still agree with each other.
            for item in record["cache"]["groups"]:
                if "mamba_cache_mode" in item.get("state", {}):
                    item["state"].pop(field)

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    if field is None:
        assert result["status"] == "complete", result["diagnostics"]
        return
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [
        "tp2: runtime state cache group lacks blocks_per_request evidence: ['mamba.0', 'mamba.2']"
    ]


def test_v031_mamba_speculative_blocks_are_refused_without_the_speculative_marker(tmp_path):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid")

    def edit(record):
        # The count matches 1 + num_speculative_blocks, while speculative_config stays None and the worker's
        # speculative marker stays false.
        for group in record["cache"]["groups"]:
            if "mamba_cache_mode" in group.get("state", {}):
                group["state"].update(num_speculative_blocks=1, blocks_per_request=2)

    _mutate(path, edit)
    records = _phase_records(path, "prefill", "worker")
    assert {record["cache"]["semantics"]["speculative"] for record in records} == {False}
    assert {record["resolved_config"]["speculative_config"] for record in records} == {None}
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == ["tp2: mamba speculative blocks are not modeled: ['mamba.0', 'mamba.2']"]


@pytest.mark.parametrize("change", [None, "latent_page", "block_size"])
def test_uniform_type_groups_compare_their_first_spec_with_the_scheduler(tmp_path, change):
    path, launches = observation_fixture(tmp_path, layout="glm_tp4")
    # The live scheduler copy of G0 is its first inner spec, the indexer.
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
        if "kind" in group:  # Only the worker records the kind.
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
    # Overlapping tensors of different groups differ in layer count, and block-outer ones in offset, but
    # their pages share bytes only for the same block ID in different groups.
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
    "shape",
    [
        "scheduler_without_hash_block_size",
        "enable_prefix_caching_string",
        "unknown_mamba_cache_mode",
        "evidence_not_a_mapping",
    ],
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
        elif shape == "evidence_not_a_mapping":
            # A list of the record's own key names passes the key-set comparison, so only the mapping check refuses
            # it. A string such as "align" would also fail the key-set comparison.
            record["cache"]["prefix_caching"] = sorted(evidence)
        else:
            evidence["mamba_cache_mode"] = "auto"

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == ["tp2: runtime observation requires complete prefix-caching evidence"]


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("worker", "enable_prefix_caching", False),
        ("worker", "mamba_cache_mode", "none"),
        ("scheduler", "enable_prefix_caching", False),
    ],
)
def test_v031_prefix_caching_evidence_must_match_the_resolved_cache_settings(tmp_path, kind, field, value):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid")

    def edit(record):
        # One process resolved a setting that its own evidence contradicts. The evidence of every rank still agrees,
        # and the launch-setting comparisons leave the prefix-caching settings out.
        rank = (record["phase"], record["kind"], record["dp_rank"], record["tp_rank"])
        if rank == ("prefill", kind, 0, 0 if kind == "worker" else None):
            record["resolved_config"]["cache_config"][field] = value

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == ["tp2: runtime prefix-caching evidence contradicts the resolved cache settings"]


@pytest.mark.parametrize("uniform", [False, True], ids=["mamba", "uniform"])
@pytest.mark.parametrize("records", ["worker_and_scheduler", "scheduler"])
def test_v031_mamba_state_groups_require_their_cache_mode(tmp_path, records, uniform):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid")

    def edit(record):
        group = record["cache"]["groups"][1]
        if uniform and record["kind"] == "worker":
            # The worker groups the two Mamba layers as a UniformTypeKVCacheSpecs, whose page holds both layer
            # pages. The scheduler copy keeps the first inner spec (kv_cache_utils.py:2434-2440).
            group.update(spec_type=_V031_SPEC + "UniformTypeKVCacheSpecs", spec_page_size_bytes=16_384)
        if record["phase"] == "decode" and (records == "worker_and_scheduler" or record["kind"] == "scheduler"):
            group["state"].pop("mamba_cache_mode")

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    if records == "scheduler":
        # The scheduler copy of every group is compared with the worker group field by field.
        assert result["diagnostics"] == ["tp2: worker and scheduler cache group geometry disagrees"]
        return
    # Every copy of the group lacks the mode, so the copies still agree with each other.
    assert result["diagnostics"] == [
        "tp2: runtime state cache group lacks Mamba cache mode evidence: ['mamba.0', 'mamba.2']"
    ]


@pytest.mark.parametrize("field,value", [("hash_block_size", 128), ("block_size", 128), ("block_size", 64.0)])
def test_v031_prefix_caching_evidence_must_match_the_scheduler_block_sizes(tmp_path, field, value):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid")

    def edit(record):
        # One scheduler reports its own block size apart from the accepted copy in its prefix-caching evidence.
        if (record["phase"], record["kind"], record["dp_rank"]) == ("prefill", "scheduler", 0):
            assert record["cache"][field] == record["cache"]["prefix_caching"][field] == 64
            record["cache"][field] = value

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == ["tp2: runtime prefix-caching evidence contradicts the scheduler block sizes"]


def test_v031_compressed_storage_outside_the_indexer_is_incomplete(tmp_path):
    path, launches = observation_fixture(tmp_path, layout="glm_tp4")

    def edit(record):
        # Every worker records a storage block on a sparse MLA latent spec. The scheduler copy of G0 holds the first
        # indexer spec only (kv_cache_utils.py:2434-2440), so the copies still agree.
        if record["kind"] == "worker":
            record["cache"]["groups"][0]["layer_specs"]["layers.3.mla"]["storage_block_size"] = 128

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    names = [name for index in range(3, 45, 4) for name in (f"layers.{index}.indexer", f"layers.{index}.mla")]
    assert result["diagnostics"] == [f"tp2: compressed cache storage is unsupported: {names}"]


@pytest.mark.parametrize(
    "case,layout,names",
    [
        (
            "uniform_group_page",
            "glm_tp4",
            [f"layers.{i}.{kind}" for i in range(3, 45, 4) for kind in ("indexer", "mla")],
        ),
        ("state_layer_page", "mamba_hybrid", ["mamba.0", "mamba.2"]),
        ("attention_group_page", "mamba_hybrid", ["attn.0", "attn.1"]),
    ],
    ids=["uniform_group_page", "state_layer_page", "attention_group_page"],
)
def test_v031_cache_group_page_must_follow_its_layer_specs(tmp_path, case, layout, names):
    path, launches = observation_fixture(tmp_path, layout=layout)

    def edit(record):
        groups = record["cache"]["groups"]
        if case == "uniform_group_page" and record["kind"] == "worker":
            # The page of a uniform group is the sum of its layer pages (kv_cache_interface.py:1261-1263). The
            # scheduler copy of G0 takes the page of its first inner spec (kv_cache_utils.py:2434-2440), so it still
            # matches.
            groups[0]["spec_page_size_bytes"] = 1
        elif case == "state_layer_page":
            # Every layer of a MambaSpec group has the group spec. This 8,000-byte page still holds the 8,000 bytes
            # of states that the views of mamba.2 cut from it.
            groups[1]["layer_specs"]["mamba.2"]["page_size_bytes"] = 8000
        elif case == "attention_group_page":
            groups[0]["spec_page_size_bytes"] = 4096

    # A change is the same in every record that it touches, so the worker and scheduler copies and the phases agree.
    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [f"tp2: cache group page contradicts its layer specs: {names}"]


@pytest.mark.parametrize(
    "layout,name,kernel,shape,strides",
    [
        # The indexer view with the group's kernel block of 64 tokens instead of its 128-token storage block:
        # 34 kernel blocks of 16 x 132 bytes per 71,808-byte page (worker/utils.py:449-450).
        ("glm_tp4", "layers.3.indexer", 64, [272, 16, 132], [2112, 132, 1]),
        # An attention view with 32-token kernel blocks while the runner gives the group 16-token ones.
        ("mamba_hybrid", "attn.0", 32, [20, 2, 32, 32], [4096, 2048, 64, 2]),
    ],
    ids=["indexer", "attention"],
)
def test_v031_layer_view_kernel_block_must_follow_its_cache_spec(tmp_path, layout, name, kernel, shape, strides):
    path, launches = observation_fixture(tmp_path, layout=layout)

    def edit(record):
        # The view keeps its bytes and pool block geometry; only its kernel block contradicts the spec.
        if record["kind"] == "worker":
            (view,) = record["cache"]["layer_tensors"][name]
            view.update(kernel_block_size_tokens=kernel, shape=shape, stride_bytes=strides)

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [f"tp2: layer view kernel block contradicts its cache spec: {name}"]


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ("missing", "initialized kernel block sizes do not cover every cache group"),
        ("short", "initialized kernel block sizes do not cover every cache group"),
        ("zero", "kernel block size must be a positive integer no greater than 2**53"),
    ],
    ids=["missing", "short", "zero"],
)
def test_v031_runner_kernel_blocks_must_cover_every_cache_group(tmp_path, change, diagnostic):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid")

    def edit(record):
        if record["kind"] != "worker":
            return
        if change == "missing":
            record["cache"].pop("model_runner")
            return
        kernels = record["cache"]["model_runner"]["kernel_block_sizes"]
        # The runner gives the two Mamba groups their 64-token group block; no view takes its kernel block from them.
        assert kernels == [16, 64, 64]
        if change == "short":
            kernels.pop()
        else:
            kernels[2] = 0

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [f"tp2: {diagnostic}"]


def _uniform(record, index, name, spec_type):
    """Make group ``index`` of a worker record a uniform group and give layer ``name`` another spec class."""
    group = record["cache"]["groups"][index]
    # The page of a uniform group holds both layer pages. The scheduler copy keeps the first inner spec
    # (kv_cache_utils.py:2434-2440), so it still matches.
    group.update(spec_type=_V031_SPEC + "UniformTypeKVCacheSpecs", spec_page_size_bytes=16_384)
    group["layer_specs"][name] = {**group["layer_specs"][name], "spec_type": _V031_SPEC + spec_type}


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("config_format_hf", None),
        ("allocation_shared_by", "allocation plan has missing, duplicate or unknown layers"),
        ("scheduler_mamba_mode", "runtime prefix-caching evidence contradicts the resolved cache settings"),
        ("extra_reservation", "additional initialized reservations lack an audited permanent-reservation mapping"),
        ("config_format", "loaded config format requires another source mapping: mistral"),
        (
            "runner_class",
            "model runner has no audited cache mapping: vllm.v1.worker.cpu_model_runner.CPUModelRunner",
        ),
        ("window_mix", "cache group layers disagree on sliding window"),
        ("state_mix", "cache group layers disagree on state evidence"),
    ],
    ids=[
        "config_format_hf",
        "allocation_shared_by",
        "scheduler_mamba_mode",
        "extra_reservation",
        "config_format",
        "runner_class",
        "window_mix",
        "state_mix",
    ],
)
def test_v031_records_must_hold_what_the_observer_guarantees(tmp_path, case, diagnostic):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid")

    def edit(record):
        cache, worker = record["cache"], record["kind"] == "worker"
        if case.startswith("config_format"):
            # Every process records the format. The observer hashes config.json only for auto and hf.
            record["resolved_config"]["model_config"]["config_format"] = "hf" if case.endswith("_hf") else "mistral"
        elif case == "allocation_shared_by" and worker:
            # The observer writes shared_by and layers from one list.
            allocation = next(item for item in cache["tensor_allocations"] if item["layers"] == ["attn.0", "attn.1"])
            allocation["shared_by"] = ["attn.0"]
        elif case == "scheduler_mamba_mode" and (record["phase"], record["kind"], record["dp_rank"]) == (
            "prefill",
            "scheduler",
            0,
        ):
            # The scheduler records the mode of its cache config next to its evidence.
            cache["mamba_cache_mode"] = "none"
        elif case == "extra_reservation" and not worker:
            # A second permanent reservation, counted consistently, which the observer never writes.
            cache.update(permanent_reserved_block_ids=[0, 1], reserved_blocks=2, initial_free_blocks=8)
        elif case == "runner_class" and worker:
            cache["model_runner"]["class"] = "vllm.v1.worker.cpu_model_runner.CPUModelRunner"
        elif case == "window_mix" and worker:
            # A full-attention and a sliding-window layer in one uniform group without a window.
            _uniform(record, 0, "attn.1", "SlidingWindowSpec")
        elif case == "state_mix" and worker:
            # A Mamba layer and a circular-buffer layer in one uniform state group with Mamba state evidence.
            _uniform(record, 1, "mamba.2", "CircularBufferSpec")

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    if diagnostic is None:
        assert result["status"] == "complete", result["diagnostics"]
        return
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [f"tp2: {diagnostic}"]


_BLOCK_SIZES = "scheduler block sizes contradict its cache groups"
_STATE_VIEWS = "layer views do not match the recorded state tensors: mamba.0"


@pytest.mark.parametrize(
    "case,layout,diagnostic",
    [
        ("scheduler_block_size", "mamba_hybrid", _BLOCK_SIZES),
        ("hash_block_size", "glm_tp4", _BLOCK_SIZES),
        ("prefix_match_unit", "mamba_hybrid", _BLOCK_SIZES),
        ("prefix_match_unit_control", "mamba_hybrid", None),
        ("mamba_state_order", "mamba_hybrid", _STATE_VIEWS),
        ("mamba_state_strides", "mamba_hybrid", _STATE_VIEWS),
        ("no_prefix_caching_hash_block_size", "mamba_hybrid", _BLOCK_SIZES),
        ("no_prefix_caching_control", "glm_tp4", None),
        (
            "mamba_block_size",
            "mamba_hybrid",
            "cache group block size contradicts the resolved Mamba block size: ['mamba.0', 'mamba.2']",
        ),
        ("mamba_runner_kernel", "mamba_hybrid", "layer view kernel block contradicts its cache spec: mamba.0"),
    ],
    ids=[
        "scheduler_block_size",
        "hash_block_size",
        "prefix_match_unit",
        "prefix_match_unit_control",
        "mamba_state_order",
        "mamba_state_strides",
        "no_prefix_caching_hash_block_size",
        "no_prefix_caching_control",
        "mamba_block_size",
        "mamba_runner_kernel",
    ],
)
def test_v031_records_must_hold_what_vllm_derives_from_them(tmp_path, case, layout, diagnostic):
    # The no_prefix_caching cases run the decode phase without prefix caching.
    path, launches = observation_fixture(
        tmp_path, layout=layout, decode_prefix_caching=not case.startswith("no_prefix_caching")
    )

    def edit(record):
        cache, worker = record["cache"], record["kind"] == "worker"
        if case == "no_prefix_caching_hash_block_size" and not worker and record["phase"] == "decode":
            # Both copies of every decode scheduler hash in 64 tokens, while the scheduler block is the 4,096-token
            # LCM of the group blocks.
            cache["hash_block_size"] = cache["prefix_caching"]["hash_block_size"] = 64
        if case == "mamba_state_order" and worker:
            # The SSM state first and the convolution state after it. The views still hold the 8,000 bytes of states
            # in the 8,192-byte page of mamba.0, without overlap.
            convolution, ssm = cache["layer_tensors"]["mamba.0"]
            convolution["storage_offset_bytes"], ssm["storage_offset_bytes"] = 7200, 0
        elif case == "mamba_state_strides" and worker:
            # The [4, 100] convolution state viewed column by column holds the same 800 bytes.
            cache["layer_tensors"]["mamba.0"][0]["stride_bytes"] = [8192, 2, 8]
        elif case == "mamba_block_size":
            # The two Mamba groups of every record hold 128-token blocks, with view and runner kernels of 128, the
            # scheduler block 128 (the LCM with the 64-token attention block) and the hash block 64 (the GCD). The
            # resolved Mamba block size and block size stay 64.
            for group in cache["groups"][1:]:
                group["block_size_tokens"] = 128
                for name in group["layer_names"] if worker else []:
                    for view in cache["layer_tensors"][name]:
                        view["kernel_block_size_tokens"] = 128
            if worker:
                cache["model_runner"]["kernel_block_sizes"][1:] = [128, 128]
            else:
                cache["block_size"] = cache["prefix_caching"]["block_size"] = 128
        elif case == "mamba_runner_kernel" and worker:
            # The runner gives the Mamba group G1 a 32-token kernel block, while its views keep its 64-token block.
            cache["model_runner"]["kernel_block_sizes"][1] = 32
        if record["phase"] != "prefill":
            return
        if case.startswith("prefix_match_unit"):
            # Every prefill process resolves a 32-token prefix-match unit, which sets the hash block size.
            record["resolved_config"]["cache_config"]["prefix_match_unit"] = 32
        if worker:
            return
        # Both copies of the block sizes of every prefill scheduler change, so they still agree with each other.
        evidence = cache["prefix_caching"]
        if case == "scheduler_block_size":
            # The LCM of the 64-token group blocks is 64.
            cache["block_size"] = evidence["block_size"] = 128
        elif case == "hash_block_size":
            # The GCD with the 4-token tail group, which takes no part in prefix caching, instead of the 2,176 tokens
            # of the other groups.
            cache["hash_block_size"] = evidence["hash_block_size"] = 4
        elif case == "prefix_match_unit_control":
            cache["hash_block_size"] = evidence["hash_block_size"] = 32

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    if diagnostic is None:
        assert result["status"] == "complete", result["diagnostics"]
        if case == "prefix_match_unit_control":
            assert _evidence(result) == (True, "align", 32)
        else:
            # Without prefix caching, the scheduler hashes in its block size: the LCM of the 2,176-token attention,
            # 4-token tail and 4,096-token Mamba group blocks.
            assert result["provenance"]["prefix_caching"]["decode"] == {
                "enable_prefix_caching": False,
                "mamba_cache_mode": "none",
                "hash_block_size_tokens": 69_632,
            }
        return
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [f"tp2: {diagnostic}"]


@pytest.mark.parametrize(
    "layout,unit,diagnostic",
    [
        # 48 tokens divide none of the 64-token blocks of the groups in prefix caching (kv_cache_utils.py:799-805).
        ("mamba_hybrid", 48, _BLOCK_SIZES),
        # 2 tokens divide the 2,176-token attention and Mamba blocks. The Mamba blocks are larger than the hash block,
        # so the cache-hit alignment is the 2-token hash block, which is not a multiple of the 4 tokens per state of
        # the indexer spec that the scheduler keeps for the attention group (kv_cache_utils.py:806-832, :2434-2440).
        ("glm_tp4", 2, _BLOCK_SIZES),
        # 8 tokens divide the 2,176-token attention and Mamba blocks and are a multiple of the 4 tokens per state.
        ("glm_tp4", 8, None),
    ],
    ids=["divisibility", "state_alignment", "state_alignment_control"],
)
def test_v031_prefix_match_unit_must_fit_the_scheduler_cache_groups(tmp_path, layout, unit, diagnostic):
    path, launches = observation_fixture(tmp_path, layout=layout)

    def edit(record):
        if record["phase"] != "prefill":
            return
        # Every prefill process resolves the unit, and both copies of the hash block size of every prefill scheduler
        # take it (kv_cache_utils.py:797-798).
        record["resolved_config"]["cache_config"]["prefix_match_unit"] = unit
        if record["kind"] == "scheduler":
            record["cache"]["hash_block_size"] = record["cache"]["prefix_caching"]["hash_block_size"] = unit

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    if diagnostic is None:
        assert result["status"] == "complete", result["diagnostics"]
        assert _evidence(result) == (True, "align", unit)
        return
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [f"tp2: {diagnostic}"]


@pytest.mark.parametrize(
    "specs,prefix_caching,unit,expected",
    [
        (["FullAttentionSpec", "FullAttentionSpec"], True, None, [32]),
        # A SlidingWindowMLASpec group takes part in prefix caching unless it replays a bounded window
        # (kv_cache_interface.py:949-951), which no record holds, so the hash block size is 32 or 64 tokens.
        (["SlidingWindowMLASpec", "FullAttentionSpec"], True, None, [32, 64]),
        # Each of two such groups may take part: 64 tokens with neither, 32 with the 96-token group alone, and 16 with
        # the 80-token group, alone or with the other.
        (["SlidingWindowMLASpec", "FullAttentionSpec", "SlidingWindowMLASpec"], True, None, [16, 32, 64]),
        # A KpoolTailSpec group takes no part (kv_cache_interface.py:1032-1034). Without the 96-token group no group
        # takes part, and vLLM hashes in the GCD of every block, 32 tokens (kv_cache_utils.py:792-798). With that group
        # alone, it hashes in 96 tokens.
        (["SlidingWindowMLASpec", "KpoolTailSpec"], True, None, [32, 96]),
        # One cache group hashes in its own block size, whatever the prefix-match unit (kv_cache_utils.py:761-765).
        (["FullAttentionSpec"], True, 32, [96]),
        # Several groups hash in the scheduler block size, the LCM of their block sizes, without prefix caching or
        # with a Mamba group outside align mode, whatever the prefix-match unit (kv_cache_utils.py:767-790).
        (["FullAttentionSpec", "FullAttentionSpec"], False, 32, [192]),
        (["FullAttentionSpec", "MambaSpec"], True, 32, [192]),
    ],
    ids=[
        "two_groups",
        "bounded_replay_unknown",
        "two_bounded_replay_unknown",
        "no_group_in_prefix_caching",
        "one_group",
        "prefix_caching_off",
        "mamba_none_mode",
    ],
)
def test_v031_scheduler_hash_block_size_follows_its_cache_groups(specs, prefix_caching, unit, expected):
    groups = [
        {"spec_type": _V031_SPEC + spec, "block_size_tokens": block}
        | ({"state": {"mamba_cache_mode": "none"}} if spec == "MambaSpec" else {})
        for spec, block in zip(specs, (96, 64, 80), strict=False)
    ]
    record = {
        "cache": {"groups": groups},
        "resolved_config": {"cache_config": {"enable_prefix_caching": prefix_caching, "prefix_match_unit": unit}},
    }
    assert runtime_observations._scheduler_hash_block_sizes(record) == expected


_SLIDING_AND_FULL = [("SlidingWindowMLASpec", 96, 1), ("FullAttentionSpec", 64, 1)]
_SLIDING_WITH_MAMBA = [("FullAttentionSpec", 64, 1), ("MambaSpec", 64, -1), ("SlidingWindowMLASpec", 96, 3)]


@pytest.mark.parametrize(
    "specs,unit,hash_block,refused",
    [
        # vLLM hashes these groups in 32 tokens when the 96-token SlidingWindowMLASpec group takes part in prefix
        # caching, and in 64 tokens when it replays a bounded window (kv_cache_utils.py:792-798,
        # kv_cache_interface.py:949-951). No hash block size of 7 tokens can follow.
        (_SLIDING_AND_FULL, None, 7, True),
        (_SLIDING_AND_FULL, None, 32, False),
        (_SLIDING_AND_FULL, None, 64, False),
        # 32 tokens need the 96-token group in prefix caching. The 64-token Mamba block is then larger than the hash
        # block, so the cache-hit alignment is 32 tokens, which is not a multiple of that group's 3 tokens per state
        # (kv_cache_utils.py:806-832). Without that group, vLLM hashes in 64 tokens.
        (_SLIDING_WITH_MAMBA, None, 32, True),
        (_SLIDING_WITH_MAMBA, None, 64, False),
        # With a 32-token unit, the 96-token group meets the divisibility rule, but its 5 tokens per state do not divide
        # the 96-token scheduler block. When it takes no part, no group does, and the unit must divide every block,
        # also the 4-token tail block (kv_cache_utils.py:792-805). So each rule holds only under the other choice.
        ([("SlidingWindowMLASpec", 96, 5), ("KpoolTailSpec", 4, 1)], 32, 32, True),
        # With 3 tokens per state, both rules hold when the 96-token group takes part.
        ([("SlidingWindowMLASpec", 96, 3), ("KpoolTailSpec", 4, 1)], 32, 32, False),
        # Without a unit, no group takes part when the 96-token group replays a bounded window, and vLLM hashes in the
        # GCD of every block, 4 tokens (kv_cache_utils.py:792-798). With that group alone, it hashes in 96 tokens, and
        # the 96-token scheduler block is a multiple of the group's 3 tokens per state (kv_cache_utils.py:806-832).
        ([("SlidingWindowMLASpec", 96, 3), ("KpoolTailSpec", 4, 1)], None, 4, False),
        ([("SlidingWindowMLASpec", 96, 3), ("KpoolTailSpec", 4, 1)], None, 96, False),
        # The 192-token scheduler block is not a multiple of the 5 tokens per state, so vLLM starts only when the
        # 96-token group takes no part. Then no group does, and the 32-token unit divides every block
        # (kv_cache_utils.py:792-832).
        ([("SlidingWindowMLASpec", 96, 5), ("KpoolTailSpec", 64, 1)], 32, 32, False),
    ],
    ids=[
        "7",
        "32",
        "64",
        "alignment_only_without_the_hash",
        "alignment_control",
        "unit_rules_under_different_choices",
        "unit_rules_under_one_choice",
        "every_block_without_a_unit",
        "sliding_group_alone_without_a_unit",
        "every_block_with_a_unit",
    ],
)
def test_v031_scheduler_hash_block_size_must_be_one_that_vllm_can_derive(specs, unit, hash_block, refused):
    # One worker and one scheduler of a prefix-caching phase, with one layer for each (class, block, tokens per state)
    # and the LCM of the blocks as the scheduler block.
    groups = [
        {
            "layer_names": [f"layer.{index}"],
            "spec_type": _V031_SPEC + spec,
            "block_size_tokens": block,
            "layer_specs": {
                f"layer.{index}": {"spec_type": _V031_SPEC + spec, "page_size_bytes": 4096, "tokens_per_state": tokens}
            },
        }
        | ({"kind": "state", "state": {"mamba_cache_mode": "align"}} if spec == "MambaSpec" else {})
        for index, (spec, block, tokens) in enumerate(specs)
    ]
    mamba = [block for spec, block, _ in specs if spec == "MambaSpec"]
    settings = {"enable_prefix_caching": True, "mamba_cache_mode": "align" if mamba else "none"}
    resolved = {
        "cache_config": {**settings, "prefix_match_unit": unit, "mamba_block_size": mamba[0] if mamba else None}
    }
    sizes = {"block_size": math.lcm(*(block for _, block, _ in specs)), "hash_block_size": hash_block}
    workers = {
        (0, 0): {
            "cache": {"tensor_allocations": [{"layer_stride": 4096}], "groups": groups, "prefix_caching": settings},
            "resolved_config": resolved,
        }
    }
    schedulers = {
        0: {
            "cache": {
                "groups": groups,
                "mamba_cache_mode": settings["mamba_cache_mode"],
                "prefix_caching": {**settings, **sizes},
                **sizes,
            },
            "resolved_config": resolved,
        }
    }
    if refused:
        with pytest.raises(ValueError, match=f"^{_BLOCK_SIZES}$"):
            runtime_observations._phase_prefix_caching("prefill", workers, schedulers)
        return
    assert runtime_observations._phase_prefix_caching("prefill", workers, schedulers) == {
        **settings,
        "hash_block_size_tokens": hash_block,
    }


@pytest.mark.parametrize(
    "specs,unit,expected",
    [
        # A SlidingWindowMLASpec group takes part in prefix caching unless it replays a bounded window
        # (kv_cache_interface.py:949-951), which no record holds. A 64-token unit divides the 64-token full-attention
        # block, so vLLM starts unless the 96-token group takes part (kv_cache_utils.py:792-805).
        ([("SlidingWindowMLASpec", 96, 1), ("FullAttentionSpec", 64, 1)], 64, True),
        # A 48-token unit does not divide the 64-token block, whichever groups take part.
        ([("SlidingWindowMLASpec", 96, 1), ("FullAttentionSpec", 64, 1)], 48, False),
        # Without the 96-token group no group takes part, and the unit must divide every block, so a 64-token unit
        # fails either way.
        ([("SlidingWindowMLASpec", 96, 1), ("KpoolTailSpec", 4, 1)], 64, False),
        # 3 tokens per state do not divide the 64-token scheduler block (kv_cache_utils.py:806-832). vLLM starts unless
        # the SlidingWindowMLASpec group that holds them takes part.
        ([("SlidingWindowMLASpec", 64, 3), ("FullAttentionSpec", 64, 1)], None, True),
        # The full-attention group that holds them always takes part.
        ([("FullAttentionSpec", 64, 3), ("SlidingWindowMLASpec", 64, 1)], None, False),
    ],
    ids=[
        "bounded_replay_unit",
        "unit_fails_either_way",
        "unit_without_other_groups",
        "bounded_replay_alignment",
        "alignment_fails_either_way",
    ],
)
def test_v031_scheduler_hash_block_size_fits_its_cache_groups(specs, unit, expected):
    groups = [
        {
            "spec_type": _V031_SPEC + spec,
            "block_size_tokens": block,
            "layer_specs": {f"layer.{index}": {"spec_type": _V031_SPEC + spec, "tokens_per_state": tokens}},
        }
        for index, (spec, block, tokens) in enumerate(specs)
    ]
    blocks = [block for _, block, _ in specs]
    record = {
        "cache": {"groups": groups, "block_size": math.lcm(*blocks), "hash_block_size": unit or math.gcd(*blocks)},
        "resolved_config": {"cache_config": {"enable_prefix_caching": True, "prefix_match_unit": unit}},
    }
    assert runtime_observations._scheduler_hash_block_size_fits(record) is expected


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

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [f"tp2: {diagnostic}"]


@pytest.mark.parametrize("change", ["missing_view", "shape", "element_size", "real_page"])
def test_v031_mamba_layer_views_must_match_the_recorded_state_tensors(tmp_path, change):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid")

    def edit(record):
        if change == "real_page":
            # Every record of the group keeps one state, now with a real page that the views do not fill.
            record["cache"]["groups"][1]["state"]["real_page_size_bytes"] = 8064
            return
        if (record["phase"], record["kind"], record["dp_rank"], record["tp_rank"]) != ("prefill", "worker", 0, 0):
            return
        # A bfloat16 [10, 4, 100] convolution view and a float32 [10, 2, 30, 30] SSM view cut the recorded
        # states [4, 100] and [2, 30, 30] from the 8,192-byte page of mamba.0 (mamba/abstract.py:28-42).
        views = record["cache"]["layer_tensors"]["mamba.0"]
        if change == "missing_view":
            # The convolution view alone is nonempty, fits the page and aliases nothing.
            views.pop(1)
        elif change == "shape":
            views[1]["shape"][1] = 1
        else:
            views[1]["element_size_bytes"] = 2

    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == ["tp2: layer views do not match the recorded state tensors: mamba.0"]


def test_v031_state_group_layer_specs_stay_compared_across_phases(tmp_path):
    path, launches = observation_fixture(tmp_path, layout="mamba_hybrid", decode_prefix_caching=False)

    def edit(record):
        # A layer spec field that followed the Mamba block size would differ between the phases.
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
    # The provenance keeps the keys of imports before vLLM 0.31.0 support, because finalized campaigns compare the
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


@pytest.mark.parametrize("version", ["0.27.0", "0.28.0"])
@pytest.mark.parametrize(
    "field,value",
    [("block_size", 16), ("mamba_cache_mode", "none"), ("mamba_block_size", 16), ("prefix_match_unit", 16)],
)
def test_legacy_imports_keep_extra_cache_settings_in_the_runtime_settings(tmp_path, version, field, value):
    path, launches = observation_fixture(tmp_path, version=version)
    # A campaign-local adapter can record more cache settings than the bundled adapters. Finalized campaigns compare
    # the runtime settings of the provenance byte for byte, so only a vLLM 0.31.0 record drops these settings.
    _mutate(path, lambda record: record["resolved_config"]["cache_config"].update({field: value}))
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    expected = {"cache_dtype": "bfloat16", "gpu_memory_utilization": 0.9, "kv_cache_memory_bytes": None, field: value}
    assert result["provenance"]["runtime_settings"]["cache_config"] == expected
    recorded = json.loads(result["resources"]["runtime_memory"]["provenance"])
    assert recorded["runtime_settings"]["cache_config"] == expected


@pytest.mark.parametrize(
    "field,values",
    [
        ("block_size", (16, 32)),
        ("mamba_cache_mode", ("none", "align")),
        ("mamba_block_size", (16, 64)),
        ("prefix_match_unit", (16, 32)),
    ],
)
def test_legacy_imports_compare_extra_cache_settings_across_phases(tmp_path, field, values):
    path, launches = observation_fixture(tmp_path)
    # The extra settings stay in the compared runtime settings, so phases that disagree on them are refused.
    for phase, value in zip(("prefill", "decode"), values, strict=True):
        _mutate(
            path,
            lambda record, phase=phase, value=value: (
                record["resolved_config"]["cache_config"].update({field: value}) if record["phase"] == phase else None
            ),
        )
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == [
        "tp2: runtime observation effective runtime/graph settings across phases/ranks mismatch"
    ]


@pytest.mark.parametrize("version", ["0.27.0", "0.28.0"])
@pytest.mark.parametrize("evidence", ["complete", "workers_only", "decode_only", "malformed"])
def test_legacy_imports_do_not_read_a_prefix_caching_key_of_an_adapter(tmp_path, version, evidence):
    path, launches = observation_fixture(tmp_path, version=version)

    def edit(record):
        worker = record["kind"] == "worker"
        value = {"enable_prefix_caching": True, "mamba_cache_mode": "none"}
        if not worker:
            value.update(block_size=16, hash_block_size=16)
        if (evidence == "workers_only" and not worker) or (evidence == "decode_only" and record["phase"] != "decode"):
            return
        record["cache"]["prefix_caching"] = "yes" if evidence == "malformed" else value

    # Only a vLLM 0.31.0 record holds this evidence. A campaign-local adapter can add a key of that name to its
    # records, which import as they did before: finalized campaigns compare the provenance byte for byte.
    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    assert not {"prefix_caching", "phase_geometry_differences"} & set(result["provenance"])
    assert set(result["resources"]["runtime_memory"]) == {
        "kv_cache_bytes",
        "gpu_memory_utilization",
        "max_model_len",
        "provenance",
    }


def _make_layer_strided(record):
    """Rewrite a vLLM 0.28.0 attention record as the vLLM 0.31.0 observer records it.

    The worker allocation places its layers with a layer stride, the views of a layer form a list, and the
    kernel blocks of the runner are recorded. A scheduler reserves only the null block. Every process records the
    prefix-caching evidence of its resolved cache config, and the scheduler adds its block sizes and its startup
    concurrency.
    """
    cache, resolved = record["cache"], record["resolved_config"]["cache_config"]
    resolved["mamba_cache_mode"] = "none"
    evidence = {"enable_prefix_caching": True, "mamba_cache_mode": "none"}
    if record["kind"] == "worker":
        # The fixture holds 16-token blocks and views that stride 128 bytes per block. Its two layers belong to two
        # groups, so they can overlay each other.
        (allocation,) = cache["tensor_allocations"]
        allocation.update(layers=allocation["shared_by"], layer_stride=0, block_stride=128)
        cache["layer_tensors"] = {name: [view] for name, view in cache["layer_tensors"].items()}
        cache["model_runner"] = {
            "class": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
            "kernel_block_sizes": [16] * len(cache["groups"]),
        }
    else:
        sizes = {"block_size": 16, "hash_block_size": 16}
        cache.update(
            reserved_blocks=1,
            initial_free_blocks=cache["num_blocks"] - 1,
            permanent_reserved_block_ids=[0],
            mamba_cache_mode="none",
            # For one request at 4,096 tokens, vLLM counts 256 blocks of each of the two 16-token groups.
            kv_cache_max_concurrency=cache["num_blocks"] / 512,
            **sizes,
        )
        evidence.update(sizes)
    cache["prefix_caching"] = evidence


@pytest.mark.parametrize(
    "strided,complete",
    [(("prefill",), False), (("decode",), False), (("prefill", "decode"), True)],
    ids=["prefill_strided", "decode_strided", "both_strided_control"],
)
def test_v031_role_phases_cannot_mix_layer_strided_and_older_records(tmp_path, strided, complete):
    path, launches = observation_fixture(tmp_path)

    def edit(record):
        # Every phase holds the layer specs of a vLLM 0.31.0 group, so that the cache geometry of the phases agrees
        # and only the prefix-caching evidence differs.
        for group in record["cache"]["groups"]:
            group["layer_specs"] = {
                name: {
                    "spec_type": group["spec_type"],
                    "page_size_bytes": group["spec_page_size_bytes"],
                    "dtype": group["dtype"],
                    "tokens_per_state": None,
                    "cache_role": None,
                    "storage_block_size": None,
                }
                for name in group["layer_names"]
            }
        if record["phase"] in strided:
            _make_layer_strided(record)

    # Only a layer-strided phase holds prefix-caching evidence, and only its imported groups carry the prefix-caching
    # marker and the transfer bytes. So the groups of the two phases differ, and import refuses the role there. The
    # role with both phases layer-strided shows that the edits build a valid layer-strided phase.
    _mutate(path, edit)
    result = validate_observations(path, launches)["tp2"]
    if complete:
        assert result["status"] == "complete", result["diagnostics"]
        evidence = {"enable_prefix_caching": True, "mamba_cache_mode": "none", "hash_block_size_tokens": 16}
        assert result["provenance"]["prefix_caching"] == {"prefill": evidence, "decode": evidence}
        return
    assert result["status"] == "incomplete"
    assert result["diagnostics"] == ["tp2: runtime observation physical page cost across phases/ranks mismatch"]


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
    # vLLM 0.31.0 resolves block_size in the engine-core process only (engine/core.py:341-350).
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
