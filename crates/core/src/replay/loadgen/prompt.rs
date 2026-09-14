// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Graph-owned source-unit identities shared by full prompts and prefix views.

use anyhow::{Context, Result, ensure};
use rustc_hash::{FxHashMap, FxHashSet};

use super::trace::synthesize_trace_tokens;
use super::{AgenticNode, ReplayRequestHashes};

/// Immutable source-hash to logical-token dictionary for one complete graph.
///
/// Sorted source identities make allocation independent of request traversal.
/// Prefix consumers must retain this context: rebuilding a dictionary from a
/// subset or a different corpus can assign different token IDs. This is not a
/// per-play cache namespace; play lifecycle and cache busting belong to the
/// caller that authors source identities.
#[derive(Debug, PartialEq, Eq)]
pub struct AgenticPromptMaterializer {
    block_size: usize,
    token_by_hash: FxHashMap<u64, u32>,
}

impl AgenticPromptMaterializer {
    pub(super) fn new(block_size: usize, nodes: &[AgenticNode]) -> Result<Self> {
        ensure!(block_size > 0, "source block_size must be greater than 0");
        // Deduplicate before sorting so repeated long prefixes do not add a
        // corpus-sized temporary copy of every source hash.
        let unique = nodes
            .iter()
            .flat_map(|node| node.hash_ids.iter().copied())
            .collect::<FxHashSet<_>>();
        let mut hashes = unique.into_iter().collect::<Vec<_>>();
        hashes.sort_unstable();
        let token_by_hash = hashes
            .into_iter()
            .enumerate()
            .map(|(index, hash)| Ok((hash, token_id(index)?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            block_size,
            token_by_hash,
        })
    }

    /// Recorded source hash unit, independent of the physical engine block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Materialize an exact prefix of an original node in this context.
    ///
    /// The original source identities (including private missing units and
    /// partial tails) are retained when cutting inside a source unit. Do not
    /// normalize a shortened Weka request to construct a primer. Unknown source
    /// hashes and lengths beyond the original prompt are errors; this context
    /// never extends or reallocates its dictionary.
    pub fn materialize_prefix(&self, node: &AgenticNode, prefix_length: usize) -> Result<Vec<u32>> {
        ensure!(
            prefix_length <= node.input_length,
            "request {} prefix length {} exceeds input_length {}",
            node.request_id,
            prefix_length,
            node.input_length
        );
        let hash_ids = self.interned_hash_ids(node)?;
        // Expand only the requested prefix, using the full node's identities.
        synthesize_trace_tokens(prefix_length, &hash_ids, self.block_size)
    }

    /// Router/engine identity inputs for the complete engine blocks in a prefix.
    ///
    /// Together with the node's request/play IDs and input length, these hashes
    /// describe ideal prefix overlap. Actual reuse must be read from the engine's
    /// first admission record, not inferred from this static identity view.
    pub fn replay_hashes(
        &self,
        node: &AgenticNode,
        prefix_length: usize,
        engine_block_size: usize,
    ) -> Result<ReplayRequestHashes> {
        ensure!(
            engine_block_size > 0,
            "engine_block_size must be greater than 0"
        );
        let engine_block_size =
            u32::try_from(engine_block_size).context("engine_block_size does not fit in u32")?;
        let tokens = self.materialize_prefix(node, prefix_length)?;
        Ok(ReplayRequestHashes::from_tokens(&tokens, engine_block_size))
    }

    pub(super) fn interned_hash_ids(&self, node: &AgenticNode) -> Result<Vec<u32>> {
        ensure!(
            node.hash_ids.len() == node.input_length.div_ceil(self.block_size),
            "request {} source hash count does not match input_length at source block_size {}",
            node.request_id,
            self.block_size
        );
        node.hash_ids
            .iter()
            .map(|hash| {
                self.token_by_hash.get(hash).copied().with_context(|| {
                    format!(
                        "request {} has unknown source hash {} in prompt materializer",
                        node.request_id, hash
                    )
                })
            })
            .collect()
    }
}

fn token_id(index: usize) -> Result<u32> {
    u32::try_from(index).context("trace contains more unique hash IDs than u32 can represent")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(input_length: usize, hash_ids: Vec<u64>) -> AgenticNode {
        AgenticNode {
            request_id: "original".into(),
            input_length,
            hash_ids,
            ..Default::default()
        }
    }

    #[test]
    fn source_ids_are_losslessly_interned_in_sorted_order() {
        let original = node(130, vec![u64::MAX, 1, 1_u64 << 32 | 1]);
        let materializer =
            AgenticPromptMaterializer::new(64, std::slice::from_ref(&original)).unwrap();
        assert_eq!(
            materializer.interned_hash_ids(&original).unwrap(),
            [2, 0, 1]
        );
        assert!(
            materializer
                .materialize_prefix(&original, 0)
                .unwrap()
                .is_empty()
        );
        assert_eq!(materializer.materialize_prefix(&original, 65).unwrap(), {
            let mut expected = vec![2; 64];
            expected.push(0);
            expected
        });
    }

    #[test]
    fn invalid_prefix_views_fail_without_extending_the_dictionary() {
        let original = node(130, vec![7, 8, 9]);
        let materializer =
            AgenticPromptMaterializer::new(64, std::slice::from_ref(&original)).unwrap();
        assert!(materializer.materialize_prefix(&original, 131).is_err());
        assert!(materializer.replay_hashes(&original, 64, 0).is_err());
        // Validate the whole original identity even when the requested prefix
        // does not reach its unknown source unit.
        let foreign = node(130, vec![7, 8, 10]);
        assert!(
            materializer
                .materialize_prefix(&foreign, 64)
                .unwrap_err()
                .to_string()
                .contains("unknown source hash")
        );
        assert!(
            materializer
                .materialize_prefix(&node(130, vec![7]), 64)
                .is_err()
        );
        assert_eq!(
            materializer.interned_hash_ids(&original).unwrap(),
            [0, 1, 2]
        );
    }

    #[test]
    fn source_block_size_and_token_capacity_are_checked() {
        assert!(AgenticPromptMaterializer::new(0, &[]).is_err());
        assert_eq!(token_id(u32::MAX as usize).unwrap(), u32::MAX);
        if let Some(overflow) = (u32::MAX as usize).checked_add(1) {
            assert!(token_id(overflow).is_err());
            let original = node(1, vec![7]);
            let materializer =
                AgenticPromptMaterializer::new(64, std::slice::from_ref(&original)).unwrap();
            assert!(materializer.replay_hashes(&original, 1, overflow).is_err());
        }
    }
}
