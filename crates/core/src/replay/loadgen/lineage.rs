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
    let mut resolved = BTreeMap::new();
    for &(play, conversation) in parents.keys() {
        let mut ancestry = BTreeSet::new();
        let mut cursor = conversation;
        let mut root = None;
        while ancestry.insert(cursor) {
            let candidates = &parents[&(play, cursor)];
            match candidates.len() {
                0 => {
                    root = Some(cursor);
                    break;
                }
                1 => cursor = candidates.first().copied().expect("one parent"),
                _ => break,
            }
        }
        resolved.insert(
            (play, conversation),
            root.map(|root| AgenticConversationLineage {
                schema: AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1.into(),
                root_conversation_id: root.into(),
                parent_conversation_id: parents[&(play, conversation)]
                    .first()
                    .map(|parent| (*parent).into()),
            }),
        );
    }
    nodes
        .iter()
        .map(|node| resolved[&(node.play_id.as_str(), node.session_id.as_str())].clone())
        .collect()
}
