// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn frozen_anchor_prefix_preserves_tree_order_missing_routes_and_resume() {
    let mut config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager);
    config.mutable_tree_tail = Some(1);
    let mut frozen = split(0, true);
    frozen.value[1] = 1.0;
    frozen.value[2] = 2.0;
    let (mut learner, mut model) = fixture(
        config,
        &[&[f64::NAN], &[1.0]],
        &[4.0, 6.0],
        1e16,
        vec![leaf(-1e16), frozen, split(0, true)],
        settings(1.0, 0.0),
    );
    // Sequential baseline + first tree cancels exactly, then the second tree
    // yields 1 (missing routes left) or 2. Summing trees first loses the 1.
    assert_eq!(learner.stores[&KIND].frozen_anchor_prefix, vec![1.0, 2.0]);
    let pointer = learner.stores[&KIND].frozen_anchor_prefix.as_ptr();
    learner
        .observe(KIND, vec![f64::NAN], 10.0, &mut model)
        .unwrap();
    // Left mean target=(4+10)/2=7; right target=6. Prefixes are 1 and 2.
    assert_eq!(model.store(KIND).unwrap().trees[2].value[1..], [6.0, 4.0]);
    assert_eq!(fixture_prediction(&model, &[f64::NAN]), 7.0);
    assert_eq!(fixture_prediction(&model, &[1.0]), 6.0);
    assert_eq!(learner.stores[&KIND].frozen_anchor_prefix.as_ptr(), pointer);

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cached-prefix");
    learner.save(&path, &model, "{}").unwrap();
    let mut restored_model = checkpoint_model(&path).unwrap();
    let mut restored =
        OnlineLearner::load(&path, &mut restored_model, &learner.config, &canonical()).unwrap();
    assert_eq!(restored.stores[&KIND].frozen_anchor_prefix, vec![1.0, 2.0]);
    learner.observe(KIND, vec![1.0], 12.0, &mut model).unwrap();
    restored
        .observe(KIND, vec![1.0], 12.0, &mut restored_model)
        .unwrap();
    assert_eq!(model.artifact(), restored_model.artifact());
    assert_eq!(
        state_bytes(&learner.stores[&KIND]),
        state_bytes(&restored.stores[&KIND])
    );
}

#[test]
fn frozen_anchor_prefix_is_derived_and_absent_without_frozen_trees() {
    for tail in [None, Some(3), Some(9)] {
        let mut config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager);
        config.mutable_tree_tail = tail;
        let (learner, _) = fixture(
            config,
            &[&[0.0]],
            &[4.0],
            1.0,
            vec![leaf(1.0), leaf(1.0), leaf(1.0)],
            settings(1.0, 0.0),
        );
        assert_eq!(learner.stores[&KIND].frozen_anchor_prefix.capacity(), 0);
    }
    let (mut learner, _) = structural_tail_fixture(
        controls(
            LearnedUpdateMode::Structural,
            LearnedDerivativeRefresh::Eager,
        ),
        1.0,
    );
    let state = learner.stores.get_mut(&KIND).unwrap();
    let serialized = state_bytes(state);
    let bytes = state.estimated_bytes();
    let cache_bytes = state.frozen_anchor_prefix.capacity() * std::mem::size_of::<f64>();
    assert!(cache_bytes > 0);
    state.frozen_anchor_prefix = Vec::new();
    assert_eq!(
        state_bytes(state),
        serialized,
        "derived cache must not change the runtime format"
    );
    assert_eq!(state.estimated_bytes() + cache_bytes, bytes);
}
