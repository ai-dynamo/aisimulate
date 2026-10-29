# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seeded random request tokens for the GLM attention collectors.

The collectors need real token ids to build KV, IndexPool and retained-tail
state; their values are not part of the published key. Instead of an external
text corpus, ids are drawn deterministically from a fixed seed with a
standard-library SHA-256 counter stream, so the same seed yields the same ids
on every Python, NumPy or framework version.

Algorithm ``sha256_counter_rejection`` version 1: block ``b`` (0, 1, ...) is
``sha256(DOMAIN || seed_u64_be || b_u64_be)``, read as four big-endian uint64
values ``v``. A value is accepted iff ``v < 2**64 - 2**64 % span`` (rejection
removes modulo bias) and maps to ``vocab_low + v % span``, where ``span =
vocab_high - vocab_low``. Ids are emitted in block/value order until ``count``
are accepted.

The id range is the pinned tokenizer's ordinary BPE vocabulary. Both pinned
checkpoints ship the same ``tokenizer.json`` (sha256 below): ids
``0..154819`` are the BPE vocabulary and all 36 added/special tokens are ids
``154820..154855``, so drawing from ``[0, 154820)`` never emits a special token.
The runners re-derive the range from the loaded tokenizer and hash its
``tokenizer.json`` before generating anything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

GENERATOR = "sha256_counter_rejection"
GENERATOR_VERSION = 1
SEED = 53
DOMAIN = b"aisimulate/glm53flash_attention/input_tokens\x00"
# zai-org/GLM-5.3-Flash@eb9eb208 and nvidia/GLM-5.3-Flash-NVFP4@09b04e5e.
TOKENIZER_JSON_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
VOCAB_LOW = 0
VOCAB_HIGH = 154820
# Same per-request window stride and slack the runners always used: request r
# starts at offset r * 4099, for up to 32 requests per set.
REQUEST_STRIDE = 4099
MAX_REQUESTS = 32


def required_token_count(plan: dict) -> int:
    """Longest planned sequence plus 32 distinct per-request window offsets."""
    longest = max(s["targets"][-1] + (s["query"] if s["phase"] == "context" else 0) for s in plan["sets"])
    return longest + MAX_REQUESTS * REQUEST_STRIDE


def spec(plan: dict) -> dict:
    """Generator parameters frozen into the attempt manifest."""
    return {
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "count": required_token_count(plan),
        "vocab_low": VOCAB_LOW,
        "vocab_high_exclusive": VOCAB_HIGH,
        "vocab_source": "pinned tokenizer.json ordinary BPE vocabulary; added/special tokens excluded",
        "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
    }


def generate(seed: int, count: int, low: int, high: int) -> list[int]:
    if count < 0 or not 0 <= low < high or high - low > 2**63:
        raise ValueError(f"invalid token generator request: count={count} range=[{low}, {high})")
    span = high - low
    limit = 2**64 - 2**64 % span
    prefix = DOMAIN + int(seed).to_bytes(8, "big")
    tokens: list[int] = []
    block = 0
    while len(tokens) < count:
        digest = hashlib.sha256(prefix + block.to_bytes(8, "big")).digest()
        for offset in range(0, 32, 8):
            value = int.from_bytes(digest[offset : offset + 8], "big")
            if value < limit:
                tokens.append(low + value % span)
        block += 1
    return tokens[:count]


def ordinary_vocab_range(tokenizer) -> tuple[int, int]:
    """``[0, vocab_size)`` of the loaded tokenizer, proven free of special ids."""
    high = int(tokenizer.vocab_size)
    reserved = {int(i) for i in getattr(tokenizer, "added_tokens_decoder", {}) or {}}
    reserved |= {int(i) for i in getattr(tokenizer, "all_special_ids", ()) or ()}
    inside = sorted(i for i in reserved if 0 <= i < high)
    if inside:
        raise RuntimeError(f"tokenizer has added/special ids inside its base vocabulary: {inside[:8]}")
    return 0, high


def manifest_tokens(manifest: dict, tokenizer, model_path: Path) -> tuple[list[int], dict]:
    """Generate the manifest's frozen token ids after checking the tokenizer."""
    frozen = manifest["input_tokens"]
    if frozen != spec(manifest["plan"]):
        raise ValueError("manifest input_tokens differ from this collector's generator")
    tokenizer_sha = hashlib.sha256((Path(model_path) / "tokenizer.json").read_bytes()).hexdigest()
    if tokenizer_sha != frozen["tokenizer_json_sha256"]:
        raise RuntimeError(f"tokenizer.json {tokenizer_sha} is not the pinned tokenizer")
    derived = ordinary_vocab_range(tokenizer)
    if derived != (frozen["vocab_low"], frozen["vocab_high_exclusive"]):
        raise RuntimeError(f"tokenizer ordinary vocabulary {derived} differs from the manifest")
    tokens = generate(frozen["seed"], frozen["count"], *derived)
    return tokens, {
        "input_tokens": frozen,
        "token_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "token_count": len(tokens),
        "unique_token_count": len(set(tokens)),
    }
