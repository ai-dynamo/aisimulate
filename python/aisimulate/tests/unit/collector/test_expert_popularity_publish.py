# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from aiconfigurator_core.sdk.expert_popularity import load_expert_popularity
from collector.expert_popularity.publish import build_bundle

pytestmark = pytest.mark.unit


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_bundle_from_passing_collection(tmp_path: Path):
    collection = tmp_path / "collection"
    output = tmp_path / "output"
    collection.mkdir()
    _write_json(
        collection / "collection_result.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "model": {
                "id": "example/Tiny-MoE",
                "revision": "a" * 40,
                "tokenizer_revision": "a" * 40,
                "architecture": "TinyMoeForCausalLM",
            },
            "routing": {
                "num_layers": 2,
                "moe_layer_ids": [1],
                "num_routed_experts": 3,
                "top_k": 2,
                "recorder_count_divisor": 1,
            },
            "workload": {
                "phase": "prefill",
                "token_distribution": "uniform_valid_non_special_token_ids",
                "isl_distribution": "discrete_uniform",
                "isl_min": 128,
                "isl_max": 4096,
                "repeat_count": 2,
                "max_new_tokens": 1,
                "temperature": 0,
                "sequential": True,
                "shard_count": 4,
                "seeds": [2026082501, 2026082502, 2026082503, 2026082504],
                "tokens_per_shard_minimum": 65536,
                "total_prompt_tokens": 262144,
                "tokenizer_vocab_size": 100,
                "tokenizer_effective_vocab_size": 103,
                "excluded_special_token_ids": [100, 101, 102],
            },
            "validation_gates": {
                "repeat_validation_mode": "exact",
                "repeat_counts_exact": True,
                "minimum_mean_layer_pearson": 0.95,
                "maximum_mean_layer_jensen_shannon_divergence_bits": 0.01,
                "layer_assignment_conservation": True,
            },
            "stability": {
                "minimum_mean_layer_pearson": 0.99,
                "maximum_mean_layer_jensen_shannon_divergence_bits": 0.001,
            },
            "repeat_stability": {"all_counts_exact": True},
            "collection": {
                "framework": "sglang",
                "framework_version": "0.5.14",
                "framework_components": {
                    "flashinfer-python": "0.6.13",
                    "flashinfer-cubin": "0.6.13",
                    "flashinfer-jit-cache": "0.6.13+cu130",
                },
                "gpu_info": ["0, Example GPU, GPU-private-uuid"],
                "server_args": ["--dist-init-addr", "private-node:1234"],
                "runtime_environment": {"CACHE": "/home/private/cache"},
                "slurm_job_id": "123456",
                "slurm_node": "private-node",
                "collection_checkpoint": {
                    "id": "repack/Tiny-MoE-FP8",
                    "revision": "b" * 40,
                    "quantization": "fp8_block",
                },
                "recorder_count_divisor": 1,
            },
        },
    )
    _write_json(
        collection / "normalized_counts.json",
        {
            "schema_version": 1,
            "model_id": "example/Tiny-MoE",
            "model_revision": "a" * 40,
            "num_layers": 2,
            "num_experts": 3,
            "top_k": 2,
            "moe_layer_ids": [1],
            "routed_token_count": 262144,
            "counts": {"1": [262144, 196608, 65536]},
        },
    )

    bundle = build_bundle(collection, output)

    table = load_expert_popularity("example/Tiny-MoE", output)
    assert bundle.name == "example--Tiny-MoE"
    assert table["popularity_rank"].tolist() == [1, 2, 3]
    assert table["token_hit_rate"].tolist() == [1.0, 0.75, 0.25]
    metadata_text = (bundle / "metadata.yaml").read_text(encoding="utf-8")
    metadata = yaml.safe_load(metadata_text)
    assert metadata_text.startswith("# SPDX-FileCopyrightText:")
    assert metadata["provenance"]["collection_checkpoint"] == {
        "id": "repack/Tiny-MoE-FP8",
        "revision": "b" * 40,
        "quantization": "fp8_block",
    }
    assert metadata["provenance"]["recorder_count_divisor"] == 1
    assert metadata["provenance"]["framework_components"] == {
        "flashinfer-python": "0.6.13",
        "flashinfer-cubin": "0.6.13",
        "flashinfer-jit-cache": "0.6.13+cu130",
    }
    assert metadata["measurement"]["workload"]["tokenizer_vocab_size"] == 100
    assert metadata["measurement"]["workload"]["tokenizer_effective_vocab_size"] == 103
    assert "private-node" not in metadata_text
    assert "private-uuid" not in metadata_text
    assert "/home/private" not in metadata_text
    assert "slurm_job_id" not in metadata_text


def test_build_bundle_rejects_failed_collection(tmp_path: Path):
    collection = tmp_path / "collection"
    collection.mkdir()
    _write_json(collection / "collection_result.json", {"schema_version": 1, "status": "FAIL"})
    _write_json(collection / "normalized_counts.json", {"schema_version": 1})

    with pytest.raises(ValueError, match="only passing"):
        build_bundle(collection, tmp_path / "output")


def test_build_bundle_rejects_passing_smoke_collection(tmp_path: Path):
    collection = tmp_path / "collection"
    collection.mkdir()
    _write_json(
        collection / "collection_result.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "model": {"id": "example/Tiny-MoE", "revision": "a" * 40},
            "workload": {"phase": "prefill", "tokens_per_shard_minimum": 64},
        },
    )
    _write_json(
        collection / "normalized_counts.json",
        {
            "schema_version": 1,
            "model_id": "example/Tiny-MoE",
            "model_revision": "a" * 40,
        },
    )

    with pytest.raises(ValueError, match="production workload or stability evidence"):
        build_bundle(collection, tmp_path / "output")
