// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Derive conversation ancestry once, before snapshot filtering. General request
//! DAGs remain valid even when they do not describe a conversation tree; their
//! ambiguous lineage is unavailable rather than guessed from arrival order.

use std::collections::{BTreeMap, BTreeSet};

use super::{AgenticDependencyRelation, AgenticNode};
use crate::replay::{AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1, AgenticConversationLineage};

pub(super) fn conversation_lineage(
    nodes: &[AgenticNode],
) -> Vec<Option<AgenticConversationLineage>> {
    let by_request = nodes
        .iter()
        .map(|node| (node.request_id.as_str(), node))
        .collect::<BTreeMap<_, _>>();
    // Session names can be reused in different authored plays.
    let mut parents = BTreeMap::<(&str, &str), BTreeSet<&str>>::new();
    for node in nodes {
        let candidates = parents
            .entry((&node.play_id, &node.session_id))
            .or_default();
        for edge in &node.dependencies {
            if edge.relation == AgenticDependencyRelation::Spawn {
                candidates.insert(by_request[edge.request_id.as_str()].session_id.as_str());
            }
        }
    }
    let roots = ancestry_roots(&parents, || {});
    nodes
        .iter()
        .map(|node| {
            let key = (node.play_id.as_str(), node.session_id.as_str());
            roots[&key].map(|root| AgenticConversationLineage {
                schema: AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1.into(),
                root_conversation_id: root.into(),
                parent_conversation_id: parents[&key].first().map(|parent| (*parent).into()),
            })
        })
        .collect()
}

fn ancestry_roots<'a>(
    parents: &BTreeMap<(&'a str, &'a str), BTreeSet<&'a str>>,
    mut on_walk: impl FnMut(),
) -> BTreeMap<(&'a str, &'a str), Option<&'a str>> {
    let mut roots = BTreeMap::new();
    for &(play, conversation) in parents.keys() {
        let mut ancestry = BTreeSet::new();
        let mut cursor = conversation;
        let root = loop {
            if let Some(root) = roots.get(&(play, cursor)) {
                break *root;
            }
            if !ancestry.insert(cursor) {
                break None;
            }
            on_walk();
            let candidates = &parents[&(play, cursor)];
            match candidates.len() {
                0 => break Some(cursor),
                1 => cursor = candidates.first().copied().expect("one parent"),
                _ => break None,
            }
        };
        // Cache every visited conversation, including ambiguous or cyclic ancestry.
        for conversation in ancestry {
            roots.insert((play, conversation), root);
        }
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_chains_visit_each_conversation_once_including_unresolved_ancestry() {
        let ids = (0..50_000).map(|i| format!("{i:05}")).collect::<Vec<_>>();
        let mut parents = ids
            .windows(2)
            .map(|pair| {
                (
                    ("play", pair[0].as_str()),
                    BTreeSet::from([pair[1].as_str()]),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let last = ids.last().unwrap().as_str();
        for root_parents in [
            BTreeSet::new(),
            BTreeSet::from([ids[0].as_str()]),
            BTreeSet::from([ids[0].as_str(), ids[1].as_str()]),
        ] {
            let expected = root_parents.is_empty().then_some(last);
            parents.insert(("play", last), root_parents);
            let mut walks = 0;
            let roots = ancestry_roots(&parents, || walks += 1);
            assert_eq!(walks, ids.len());
            assert_eq!(roots.len(), ids.len());
            assert!(roots.values().all(|root| *root == expected));
        }
    }
}
