// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// SPDX-License-Identifier: Apache-2.0
// Cache-salt behavior adapted from SGLang v0.5.22; see THIRD_PARTY_NOTICES.md.
// Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d

//! Engine-local KV block hashing.
//!
//! These helpers preserve the mocker's existing XXH3 wire values without
//! depending on Dynamo's router or token crates.

use xxhash_rust::xxh3::xxh3_64_with_seed;

pub(crate) type BlockHash = u64;
pub(crate) type SequenceHash = u64;
pub(crate) type Token = u32;

pub(crate) const XXH3_SEED: u64 = 1337;

/// AISim's XXH3 cache namespace. Empty salt preserves all legacy hash values.
/// This models SGLang's namespace isolation, not its SHA256 event encoding.
pub(crate) fn cache_salt_seed(cache_salt: Option<&str>) -> u64 {
    match cache_salt.filter(|salt| !salt.is_empty()) {
        Some(salt) => xxh3_64_with_seed(
            salt.as_bytes(),
            xxh3_64_with_seed(b"aisim-cache-salt-v1\0", XXH3_SEED),
        ),
        None => XXH3_SEED,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub(crate) struct LocalBlockHash(pub(crate) u64);

#[inline]
pub(crate) fn compute_block_hash_for_tokens(tokens: &[Token], seed: u64) -> BlockHash {
    #[cfg(target_endian = "little")]
    let bytes = unsafe {
        // SAFETY: `u32` is plain data and its little-endian in-memory layout
        // matches the canonical token encoding used by the original mocker.
        std::slice::from_raw_parts(tokens.as_ptr().cast::<u8>(), std::mem::size_of_val(tokens))
    };

    #[cfg(target_endian = "big")]
    let encoded = tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect::<Vec<_>>();
    #[cfg(target_endian = "big")]
    let bytes = encoded.as_slice();

    xxh3_64_with_seed(bytes, seed)
}

#[inline]
pub(crate) fn compute_next_sequence_hash(
    parent_sequence_hash: SequenceHash,
    child_block_hash: BlockHash,
) -> SequenceHash {
    let values = [parent_sequence_hash, child_block_hash];
    #[cfg(target_endian = "little")]
    let bytes = unsafe {
        // SAFETY: `[u64; 2]` is plain data and the target representation is
        // the legacy little-endian chain encoding.
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(&values))
    };
    #[cfg(target_endian = "big")]
    let encoded = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    #[cfg(target_endian = "big")]
    let bytes = encoded.as_slice();
    xxh3_64_with_seed(bytes, XXH3_SEED)
}

/// Hash each complete `block_size` block of `tokens` with [`XXH3_SEED`].
///
/// Trace-synthesized prompts repeat one token across a trace block, so a block
/// equal to its predecessor reuses the predecessor's hash instead of hashing
/// the same bytes again.
pub(crate) fn block_hashes(
    tokens: &[Token],
    block_size: usize,
) -> impl Iterator<Item = BlockHash> + '_ {
    block_hashes_with_salt(tokens, block_size, None)
}

pub(crate) fn block_hashes_with_salt<'a>(
    tokens: &'a [Token],
    block_size: usize,
    cache_salt: Option<&str>,
) -> impl Iterator<Item = BlockHash> + 'a {
    let seed = cache_salt_seed(cache_salt);
    let mut previous: Option<(&[Token], BlockHash)> = None;
    tokens.chunks_exact(block_size).map(move |block| {
        let hash = match previous {
            Some((prior, hash)) if prior[0] == block[0] && prior == block => hash,
            _ => compute_block_hash_for_tokens(block, seed),
        };
        previous = Some((block, hash));
        hash
    })
}

#[cfg(test)]
pub(crate) fn compute_block_hash_for_seq(
    tokens: &[Token],
    block_size: usize,
) -> Vec<LocalBlockHash> {
    if block_size == 0 {
        return Vec::new();
    }
    block_hashes(tokens, block_size)
        .map(LocalBlockHash)
        .collect()
}

pub(crate) fn compute_next_seq_hash(parent: SequenceHash, current: LocalBlockHash) -> SequenceHash {
    compute_next_sequence_hash(parent, current.0)
}

#[cfg(test)]
pub(crate) fn compute_seq_hash_for_block(block_hashes: &[LocalBlockHash]) -> Vec<SequenceHash> {
    let mut sequence_hashes = Vec::with_capacity(block_hashes.len());
    for block_hash in block_hashes {
        let next = sequence_hashes
            .last()
            .copied()
            .map_or(block_hash.0, |parent| {
                compute_next_seq_hash(parent, *block_hash)
            });
        sequence_hashes.push(next);
    }
    sequence_hashes
}
