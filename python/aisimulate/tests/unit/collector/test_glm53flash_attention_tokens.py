# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seeded request-token generator of the GLM attention collectors."""

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml
from collector.glm53flash_attention_contract import build_plan
from collector.glm53flash_attention_launch import SMOKE_SWEEP, prepare
from collector.glm53flash_attention_tokens import (
    SEED,
    TOKENIZER_JSON_SHA256,
    VOCAB_HIGH,
    VOCAB_LOW,
    generate,
    manifest_tokens,
    ordinary_vocab_range,
    required_token_count,
    spec,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SWEEP = ROOT / "collector/cases/base_ops/glm53flash_attention.yaml"
CONFIG = ROOT / "src/aisimulate_core/model_configs/zai-org--GLM-5.3-Flash_config.json"
GOLDEN_PREFIX = [11648, 121572, 142980, 148051, 2300, 54808, 50838, 78024]
GOLDEN_SHA256 = "72df69eb7b1d7a8aafebf1957158b54abe599bb1157e5ac311f44ebad57a71c1"

ADDED_IDS = tuple(range(VOCAB_HIGH, VOCAB_HIGH + 36))


class FakeTokenizer:
    def __init__(self, vocab_size=VOCAB_HIGH, added=ADDED_IDS):
        self.vocab_size = vocab_size
        self.added_tokens_decoder = {i: f"<added-{i}>" for i in added}
        self.all_special_ids = [VOCAB_HIGH]


def test_generator_is_deterministic_and_inside_the_ordinary_vocabulary():
    first = generate(SEED, 50_000, VOCAB_LOW, VOCAB_HIGH)
    assert first == generate(SEED, 50_000, VOCAB_LOW, VOCAB_HIGH)
    assert len(first) == 50_000
    assert all(VOCAB_LOW <= t < VOCAB_HIGH for t in first)
    assert len(set(first)) > 40_000
    # A prefix of a longer draw is the shorter draw; another seed differs.
    assert generate(SEED, 1000, VOCAB_LOW, VOCAB_HIGH) == first[:1000]
    assert generate(SEED + 1, 1000, VOCAB_LOW, VOCAB_HIGH) != first[:1000]
    small = generate(SEED, 5000, 10, 13)
    assert set(small) == {10, 11, 12}


def test_generator_stream_is_pinned():
    # Golden values: any change to the algorithm must bump GENERATOR_VERSION.
    tokens = generate(SEED, 4096, VOCAB_LOW, VOCAB_HIGH)
    assert tokens[:8] == GOLDEN_PREFIX
    assert hashlib.sha256(json.dumps(tokens).encode()).hexdigest() == GOLDEN_SHA256


def test_generator_rejects_invalid_ranges():
    for low, high in ((5, 5), (-1, 3), (0, 2**63 + 1)):
        with pytest.raises(ValueError):
            generate(SEED, 1, low, high)


def test_count_covers_the_longest_request_and_window_offsets():
    full = build_plan(yaml.safe_load(SWEEP.read_text())["common_case_values"]["glm53flash_attention"])
    assert required_token_count(full) == 131072 + 32 * 4099
    smoke = build_plan(SMOKE_SWEEP)
    assert required_token_count(smoke) == 32768 + 2048 + 32 * 4099


def test_vocab_range_excludes_added_and_special_tokens():
    assert ordinary_vocab_range(FakeTokenizer()) == (VOCAB_LOW, VOCAB_HIGH)
    with pytest.raises(RuntimeError, match="inside its base vocabulary"):
        ordinary_vocab_range(FakeTokenizer(added=[7]))


def test_manifest_tokens_check_the_pinned_tokenizer(tmp_path, monkeypatch):
    from collector import glm53flash_attention_tokens as tokens_module

    plan = build_plan(SMOKE_SWEEP)
    manifest = {"plan": plan, "input_tokens": spec(plan)}
    (tmp_path / "tokenizer.json").write_text("{}")
    with pytest.raises(RuntimeError, match="not the pinned tokenizer"):
        manifest_tokens(manifest, FakeTokenizer(), tmp_path)
    monkeypatch.setattr(tokens_module, "TOKENIZER_JSON_SHA256", hashlib.sha256(b"{}").hexdigest())
    manifest = {"plan": plan, "input_tokens": tokens_module.spec(plan)}
    tokens, provenance = manifest_tokens(manifest, FakeTokenizer(), tmp_path)
    assert len(tokens) == provenance["token_count"] == required_token_count(plan)
    assert provenance["input_tokens"]["seed"] == SEED
    assert provenance["input_tokens"]["vocab_high_exclusive"] == VOCAB_HIGH
    with pytest.raises(RuntimeError, match="ordinary vocabulary"):
        manifest_tokens(manifest, FakeTokenizer(vocab_size=VOCAB_HIGH - 1), tmp_path)
    with pytest.raises(ValueError, match="input_tokens differ"):
        manifest_tokens({"plan": plan, "input_tokens": {**manifest["input_tokens"], "seed": 1}}, None, tmp_path)


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_launcher_freezes_the_generator_and_needs_no_corpus(tmp_path, backend):
    args = Namespace(
        attempt=tmp_path / "attempt",
        config=CONFIG,
        smoke=True,
        layer_id=None,
        sweep=None,
        backend=backend,
        checkpoint="fp8",
        tp=2,
        source_commit="c" * 40,
        allocator_max_split_mb=None,
        only_sets=None,
        skip_sets=None,
        sglang_mem_fraction=None,
        vllm_gpu_memory_utilization=None,
        prefill_graph=False,
        remote_attempt="/remote/attempt",
        remote_source="/remote/src",
        remote_model="/remote/model",
        remote_tail="/remote/tail",
        image="image.sqsh",
        account="acct",
        partition="batch",
        time="01:00:00",
    )
    attempt = prepare(args)
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["input_tokens"] == spec(manifest["plan"])
    assert manifest["input_tokens"]["tokenizer_json_sha256"] == TOKENIZER_JSON_SHA256
    for script in ("run.sbatch", "dryrun.sbatch"):
        assert "corpus" not in (attempt / script).read_text()
