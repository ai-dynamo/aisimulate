// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rule and coordinator cases of the grouped prefix cache, with the vLLM
//! 0.31.0 and 0.32.0 literals re-derived from the pinned trees. Some cases
//! are ported from the vLLM tests at `db9527a46873454610df6dbedf79a36d6bf1a7f6`
//! and `5df63a7203cdb9328d824a27b682e77c7b9928be`. Copyright contributors to
//! the vLLM project. See THIRD_PARTY_NOTICES.md. Apache-2.0.
//!
//! - https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py
//! - https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py
//! - https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/tests/v1/core/test_mamba_align_chunk_split.py

use super::*;
use crate::engine::cache::vllm_block_pool::CacheKey;
use crate::engine::common::hashing::SequenceHash;
use crate::engine::kv_manager::grouped::{GroupLease, GroupedBlockPool, GroupedKvPool};
use crate::engine::kv_manager::vllm_backend::BlockRequestLease;
use crate::perfmodel::{FpmCacheGroup, FpmCacheKind};
use uuid::Uuid;

const BUDGET: usize = 8192;
const GLM_BLOCK: u32 = 2176;

fn group(
    name: &str,
    kind: FpmCacheKind,
    block: u32,
    window: Option<u32>,
    state: Option<(bool, u32)>,
) -> FpmCacheGroup {
    FpmCacheGroup {
        name: name.into(),
        kind,
        num_layers: 1,
        block_size_tokens: block,
        page_size_bytes: 1,
        sliding_window: window,
        blocks_per_request: state.map(|_| 1),
        prefix_cacheable: Some(state.is_none_or(|(cacheable, _)| cacheable)),
        transfer_bytes_per_block: None,
        prefill_checkpoint_blocks: state.and_then(|(cacheable, ckpt)| cacheable.then_some(ckpt)),
        prefill_checkpoint_alignment: state
            .and_then(|(cacheable, ckpt)| (cacheable && ckpt == 1).then_some(16)),
    }
}

fn attention(name: &str, block: u32, window: Option<u32>) -> FpmCacheGroup {
    group(name, FpmCacheKind::Attention, block, window, None)
}

fn state(name: &str, block: u32, cacheable: bool, checkpoint: bool) -> FpmCacheGroup {
    group(
        name,
        FpmCacheKind::State,
        block,
        None,
        Some((cacheable, u32::from(checkpoint))),
    )
}

/// The six GLM-5.3-Flash TP 4 groups that the runtime probe recorded on
/// vLLM 0.31.0. With `checkpoint`, G2 to G5 carry
/// `prefill_checkpoint_blocks: Some(1)` and alignment 16.
fn groups_glm(checkpoint: bool) -> Vec<FpmCacheGroup> {
    let mut groups = vec![
        attention("mla", GLM_BLOCK, None),
        state("kpool_tail", 4, false, false),
    ];
    for i in 0..4 {
        groups.push(state(&format!("kda_{i}"), GLM_BLOCK, true, checkpoint));
    }
    groups
}

fn rules_glm_hash(
    version: VllmRuleVersion,
    checkpoint: bool,
    hash_block: usize,
) -> GroupedPrefixRules {
    GroupedPrefixRules::new(&groups_glm(checkpoint), hash_block, version).unwrap()
}

fn rules_glm(version: VllmRuleVersion, checkpoint: bool) -> GroupedPrefixRules {
    rules_glm_hash(version, checkpoint, GLM_BLOCK as usize)
}

fn rules(groups: &[FpmCacheGroup], hash_block: usize) -> GroupedPrefixRules {
    GroupedPrefixRules::new(groups, hash_block, VllmRuleVersion::V0_31_0).unwrap()
}

/// A block pool with `blocks` pool blocks for `groups`.
fn kv_pool(groups: &[FpmCacheGroup], blocks: u64) -> GroupedKvPool {
    GroupedKvPool::with_prefix_cache(groups.to_vec(), blocks, 1)
}

/// A lease with chained hashes of the tokens `0..tokens` at `hash_block`.
fn lease(tokens: usize, hash_block: usize) -> BlockRequestLease {
    let tokens: Vec<u32> = (0..tokens as u32).collect();
    let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
    lease.configure_prefix_hashes(&tokens, hash_block);
    lease
}

fn hash_at(hash_block: usize, tokens: usize) -> SequenceHash {
    lease(tokens, hash_block).prefix_hash_at(tokens).unwrap()
}

fn key(group: u16, hash_block: usize, tokens: usize) -> CacheKey {
    CacheKey::Grouped {
        group,
        hash: hash_at(hash_block, tokens),
    }
}

/// Publish one block under `key` and leave it in the inactive LRU, as a
/// finished request leaves its cached blocks.
fn seed(pool: &mut GroupedBlockPool, key: CacheKey) {
    let mut reservation = pool.reserve(&[], 1).unwrap();
    let id = pool.take_private(&mut reservation);
    assert!(pool.cache(id, key));
    pool.release(id);
}

fn seed_many(pool: &mut GroupedBlockPool, group: u16, hash_block: usize, ends: &[usize]) {
    for &end in ends {
        seed(pool, key(group, hash_block, end));
    }
}

fn seed_states(pool: &mut GroupedBlockPool, groups: &[u16], hash_block: usize, end: usize) {
    for &group in groups {
        seed(pool, key(group, hash_block, end));
    }
}

fn hit_exists(pool: &GroupedBlockPool, group: u16, hash_block: usize, tokens: usize) -> bool {
    pool.key_hit(group, hash_at(hash_block, tokens)).is_some()
}

fn state_lease(lease: &GroupedBlockLease, group: usize) -> &GroupStateLease {
    match &lease.groups[group] {
        GroupLease::State(state) => state,
        other => panic!("group {group} is not a state lease: {other:?}"),
    }
}

fn attention_copies(lease: &GroupedBlockLease, group: usize) -> usize {
    match &lease.groups[group] {
        GroupLease::Attention { copies, .. } => copies.iter().flatten().count(),
        other => panic!("group {group} is not an attention lease: {other:?}"),
    }
}

fn split(
    rules: &GroupedPrefixRules,
    start: usize,
    prompt: usize,
    known: usize,
    desired: usize,
    junction: usize,
) -> usize {
    rules.align_split(start, prompt, known, desired, junction, BUDGET)
}

/// Run a whole prefill of `prompt` tokens from `hit`: the chunks of
/// `align_split` at `budget`, one step each. Returns the chunk ends.
fn prefill(
    coordinator: &mut GroupedPrefixCoordinator,
    pool: &mut GroupedBlockPool,
    lease: &mut GroupedBlockLease,
    request: &BlockRequestLease,
    prompt: usize,
    hit: &GroupedHit,
    budget: usize,
) -> Vec<usize> {
    let mut ends = Vec::new();
    let mut computed = hit.tokens;
    let mut step = 1;
    while computed < prompt {
        coordinator.begin_step();
        let chunk = coordinator.rules().align_split(
            computed,
            prompt,
            prompt,
            budget,
            hit.tokens + hit.junction_tokens,
            budget,
        );
        assert!(chunk > 0, "the split must make progress at {computed}");
        let target = computed + chunk;
        assert!(coordinator.allocate(pool, lease, prompt, computed, target, step));
        coordinator.finalize(pool, lease, request, prompt, target, hit, step);
        computed = target;
        ends.push(target);
        step += 1;
    }
    coordinator.begin_step();
    ends
}

const GLM_STATES: [u16; 4] = [2, 3, 4, 5];

/// One batch of `BUDGET` tokens in flight and no model-length cap.
const LIMITS: AdmissionLimits = AdmissionLimits {
    max_in_flight_tokens: BUDGET,
    max_model_len: usize::MAX,
};

fn with_limits(rules: GroupedPrefixRules) -> GroupedPrefixCoordinator {
    GroupedPrefixCoordinator::new(rules, LIMITS)
}

#[test]
fn version_rows_differ_in_the_last_cacheable_stop_and_the_partial_tail() {
    // v0.31 sched:444 `num_tokens - num_tokens % block_size`; v0.32
    // sched:476-480 with iface:1110-1122 `(n - 1) // B * B`.
    let v31 = rules_glm(VllmRuleVersion::V0_31_0, false);
    let v32 = rules_glm(VllmRuleVersion::V0_32_0, false);
    assert_eq!(v31.last_cacheable_stop(9000), 8704);
    assert_eq!(v32.last_cacheable_stop(9000), 8704);
    assert_eq!(v31.last_cacheable_stop(8704), 8704);
    assert_eq!(v32.last_cacheable_stop(8704), 6528);
    assert_eq!(v32.last_cacheable_stop(0), 0);
    // v0.31 sched:482-487 `prompt // hash * hash`; v0.32 sched:517-525
    // `(prompt - 1) // hash * hash`.
    let v31 = rules_glm_hash(VllmRuleVersion::V0_31_0, false, 4);
    let v32 = rules_glm_hash(VllmRuleVersion::V0_32_0, false, 4);
    assert_eq!(v31.partial_tail(9000), 9000);
    assert_eq!(v32.partial_tail(9000), 8996);
    assert_eq!(v31.partial_tail(4356), 4356);
    assert_eq!(v32.partial_tail(4356), 4352);
    assert_eq!(
        VllmRuleVersion::parse("0.31.0"),
        Some(VllmRuleVersion::V0_31_0)
    );
    assert_eq!(
        VllmRuleVersion::parse("0.32.0"),
        Some(VllmRuleVersion::V0_32_0)
    );
    assert_eq!(VllmRuleVersion::parse("0.30.0"), None);
}

#[test]
fn rules_refuse_unsupported_checkpoint_fields() {
    // v0.32 sched:390-394 runs checkpoint mode only when every Mamba
    // group has a positive count.
    let err = GroupedPrefixRules::new(&groups_glm(true), 2176, VllmRuleVersion::V0_31_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with vllm_version=0.31.0 do not model prefill checkpoint blocks"
    );
    let mut groups = groups_glm(false);
    groups[2].prefill_checkpoint_blocks = None;
    let err = GroupedPrefixRules::new(&groups, 2176, VllmRuleVersion::V0_32_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with vllm_version=0.32.0 require prefill_checkpoint_blocks on every prefix-cacheable state group; re-import the runtime probe"
    );
    let mut groups = groups_glm(true);
    groups[3].prefill_checkpoint_alignment = Some(32);
    let err = GroupedPrefixRules::new(&groups, 2176, VllmRuleVersion::V0_32_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with vllm_version=0.32.0 require one prefill_checkpoint_alignment across state groups"
    );
    let zero = rules_glm(VllmRuleVersion::V0_32_0, false);
    assert!(zero.checkpoint.is_none());
    let one = rules_glm(VllmRuleVersion::V0_32_0, true);
    assert_eq!(one.checkpoint, Some(CheckpointRule { alignment: 16 }));
    // Block mode holds one running block per state group, so another count
    // is refused.
    let mut groups = groups_glm(false);
    groups[4].blocks_per_request = Some(2);
    let err = GroupedPrefixRules::new(&groups, 2176, VllmRuleVersion::V0_31_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with enable_prefix_caching=true model one state block per request; blocks_per_request must be 1"
    );
    // utils:799-805 and coord:87-92.
    assert!(GroupedPrefixRules::new(&groups_glm(false), 3, VllmRuleVersion::V0_31_0).is_err());
}

#[test]
fn rules_refuse_convolution_groups() {
    // A convolution group keeps a window, and the table has no rule for it.
    let mut convolution = attention("convolution", 4, Some(4));
    convolution.kind = FpmCacheKind::Convolution;
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        for groups in [
            vec![convolution.clone()],
            vec![attention("full", 4, None), convolution.clone()],
        ] {
            let err = GroupedPrefixRules::new(&groups, 4, version)
                .unwrap_err()
                .to_string();
            assert_eq!(
                err,
                "grouped prefix caching does not support convolution cache groups; turn prefix caching off to replay them cold",
                "{version:?}"
            );
        }
    }
}

#[test]
fn alignment_is_the_hash_block_only_with_fine_grained_hits() {
    // coord:687-735. G2 to G5 are state rules with block 2176 and G0 is
    // full attention. At hash 2176 no state block exceeds the hash block.
    let glm = rules_glm(VllmRuleVersion::V0_31_0, false);
    assert_eq!(glm.alignment, 2176);
    assert_eq!(glm.cacheable_block, 2176);
    let fine = rules_glm_hash(VllmRuleVersion::V0_31_0, false, 4);
    assert_eq!(fine.alignment, 4);
    // Fine-grained hits need a Mamba group in align mode whose block exceeds
    // the hash block. Two attention groups have none, so they give the
    // scheduler block, with or without a window (coord:691-704).
    let windowed = [attention("a", 64, Some(100)), attention("b", 16, None)];
    assert_eq!(rules(&windowed, 16).alignment, 64);
    let plain = [attention("a", 64, None), attention("b", 16, None)];
    assert_eq!(rules(&plain, 16).alignment, 64);
    // A state group above the hash block turns fine-grained hits on when
    // every other cacheable rule is full attention or at the hash block.
    let with_state = [
        attention("a", 64, None),
        attention("b", 16, None),
        state("s", 64, true, false),
    ];
    assert_eq!(rules(&with_state, 16).alignment, 16);
    // A window at another block than the hash block turns them off again
    // (single:960 has no sub-block lookup).
    let with_window = [
        attention("a", 64, Some(100)),
        attention("b", 16, None),
        state("s", 64, true, false),
    ];
    assert_eq!(rules(&with_window, 16).alignment, 64);
}

#[test]
fn full_attention_hit_is_capped_below_the_prompt() {
    // mgr:289-295 caps the lookup at `num_tokens - 1`.
    let groups = [attention("full", 16, None)];
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 16, version).unwrap());
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 16, &[16, 32, 48, 64]);
        let hit = coordinator.find_longest_hit(pool, &lease(64, 16), 64);
        assert_eq!(hit.tokens, 48);
        assert_eq!(hit.junction_tokens, 0);
        assert_eq!(hit.per_group.len(), 1);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 3));
        let hit = coordinator.find_longest_hit(pool, &lease(65, 16), 65);
        assert_eq!(hit.tokens, 64);
    }
}

#[test]
fn sliding_window_hits_need_a_contiguous_run_before_an_aligned_end() {
    // single:960-1069: `cdiv(32, 16) == 2` contiguous blocks before an
    // aligned end, or a run from block 0.
    let groups = [
        attention("full", 16, None),
        attention("window", 16, Some(33)),
    ];
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let r = GroupedPrefixRules::new(&groups, 16, version).unwrap();
        assert_eq!(r.alignment, 16);
        let coordinator = with_limits(r);
        let mut kv = kv_pool(&groups, 32);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 16, &[16, 32, 48, 64, 80, 96]);
        seed_many(pool, 1, 16, &[32, 48, 64, 96]);
        let hit = coordinator.find_longest_hit(pool, &lease(96, 16), 96);
        // The end 64 has blocks 2 and 3 before it. The end 80 lacks block 4.
        assert_eq!(hit.tokens, 64);
        match &hit.per_group[1] {
            GroupHitBlocks::Attention(keys) => {
                assert_eq!(keys.len(), 4);
                assert!(keys[0].is_none() && keys[1].is_none());
                assert_eq!(keys[2], Some(key(1, 16, 48)));
                assert_eq!(keys[3], Some(key(1, 16, 64)));
            }
            other => panic!("{other:?}"),
        }
        // Junction: the full-attention hit before the loop (80) minus 64.
        assert_eq!(hit.junction_tokens, 16);

        let mut kv = kv_pool(&groups, 32);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 16, &[16, 32, 48, 64, 80, 96]);
        seed_many(pool, 1, 16, &[16]);
        let hit = coordinator.find_longest_hit(pool, &lease(96, 16), 96);
        assert_eq!(hit.tokens, 16);
    }
}

#[test]
fn one_token_window_hits_only_at_a_cached_block() {
    // A window of one token needs `cdiv(0, 4) == 0` contiguous blocks
    // (v0.31 single:961-971, v0.32 :943-953). vLLM counts a block only
    // after it finds the block cached, so a hit still needs the cached
    // block at its end (v0.31 single:1021-1044, v0.32 :1003-1026).
    let groups = [attention("window", 4, Some(1))];
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 32);
        let pool = kv.block_pool_mut().unwrap();
        let hit = coordinator.find_longest_hit(pool, &lease(100, 4), 100);
        assert_eq!((hit.tokens, hit.junction_tokens), (0, 0), "{version:?}");
        seed_many(pool, 0, 4, &[48]);
        let hit = coordinator.find_longest_hit(pool, &lease(100, 4), 100);
        assert_eq!((hit.tokens, hit.junction_tokens), (48, 0), "{version:?}");
        match &hit.per_group[0] {
            GroupHitBlocks::Attention(keys) => {
                assert_eq!(keys.len(), 12, "{version:?}");
                assert!(keys[..11].iter().all(Option::is_none), "{version:?}");
                assert_eq!(keys[11], Some(key(0, 4, 48)), "{version:?}");
            }
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn sliding_window_admission_reserves_the_window_plus_the_in_flight_tokens() {
    // iface:859-866: `cdiv(min(window - 1 + max_in_flight_tokens,
    // max_model_len), block) + 1`, with `max_in_flight_tokens =
    // max_concurrent_batches * max_num_batched_tokens` (vllm.py:605-613).
    let groups = [
        attention("full", 16, None),
        attention("window", 16, Some(33)),
    ];
    let prompt: usize = 100_000;
    let hit = GroupedHit {
        tokens: 0,
        junction_tokens: 0,
        per_group: vec![
            GroupHitBlocks::Attention(Vec::new()),
            GroupHitBlocks::Attention(Vec::new()),
        ],
    };
    let full_blocks = prompt.div_ceil(16);
    let kv = kv_pool(&groups, 16);
    let pool = kv.block_pool().unwrap();
    // One batch of 8,192: (32 + 8192) / 16 + 1 = 515.
    let one_batch = with_limits(rules(&groups, 16));
    assert_eq!(
        one_batch.admission_blocks(pool, &hit, prompt),
        full_blocks + 515
    );
    // Two batches of 8,192: (32 + 16384) / 16 + 1 = 1027.
    let two_batches = GroupedPrefixCoordinator::new(
        rules(&groups, 16),
        AdmissionLimits {
            max_in_flight_tokens: 2 * BUDGET,
            max_model_len: usize::MAX,
        },
    );
    assert_eq!(
        two_batches.admission_blocks(pool, &hit, prompt),
        full_blocks + 1027
    );
    // The model length caps it: cdiv(4096, 16) + 1 = 257.
    let capped = GroupedPrefixCoordinator::new(
        rules(&groups, 16),
        AdmissionLimits {
            max_in_flight_tokens: 2 * BUDGET,
            max_model_len: 4096,
        },
    );
    assert_eq!(
        capped.admission_blocks(pool, &hit, prompt),
        full_blocks + 257
    );
    // A short prompt needs fewer blocks than the cap.
    assert_eq!(one_batch.admission_blocks(pool, &hit, 100), 7 + 7);
}

#[test]
fn state_group_hits_only_at_a_kept_boundary() {
    // single:1487-1564 scans aligned ends right to left for one cached
    // state block. pool:198-224: every group of the spec group must hold
    // the key.
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704]);
        seed_states(pool, &GLM_STATES, hash, 8704);
        let request = lease(9000, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 9000);
        assert_eq!(hit.tokens, 8704);
        assert_eq!(hit.junction_tokens, 0);
        assert_eq!(
            hit.per_group[2],
            GroupHitBlocks::State(Some(key(2, hash, 8704)))
        );
        assert_eq!(hit.per_group[1], GroupHitBlocks::None);

        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704]);
        seed_states(pool, &[2, 4, 5], hash, 8704);
        let hit = coordinator.find_longest_hit(pool, &request, 9000);
        assert_eq!(hit.tokens, 0);
        assert_eq!(hit.per_group[0], GroupHitBlocks::Attention(Vec::new()));
        assert_eq!(hit.per_group[3], GroupHitBlocks::State(None));

        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704]);
        seed_states(pool, &GLM_STATES, hash, 6528);
        let hit = coordinator.find_longest_hit(pool, &request, 9000);
        assert_eq!(hit.tokens, 6528);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 3));
    }
}

#[test]
fn ring_group_never_hits_and_holds_one_block() {
    // single:1214-1230 claims one block per request; single:498-499
    // caches nothing for a group that is not prefix cacheable.
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let groups = groups_glm(false);
        let mut coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(2500, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 2500);
        assert_eq!(hit.tokens, 0);
        let mut block_lease = coordinator.new_lease();
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, 2500, 0, 2176, 1));
        assert_eq!(
            block_lease.groups[1],
            GroupLease::Ring {
                copy: Some(ring_copy(&block_lease))
            }
        );
        coordinator.finalize(pool, &mut block_lease, &request, 2500, 2176, &hit, 1);
        assert_eq!(pool.inactive_cached_blocks(), 0);
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, 2500, 2176, 2500, 2));
        coordinator.finalize(pool, &mut block_lease, &request, 2500, 2500, &hit, 2);
        coordinator.release_request(pool, &mut block_lease);
        coordinator.begin_step();

        let hit = coordinator.find_longest_hit(pool, &lease(2500, hash), 2500);
        assert_eq!(hit.tokens, 2176);
        assert_eq!(hit.per_group[1], GroupHitBlocks::None);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 1));
        assert_eq!(
            hit.per_group[2],
            GroupHitBlocks::State(Some(key(2, hash, 2176)))
        );
    }
}

fn ring_copy(lease: &GroupedBlockLease) -> crate::engine::cache::vllm_block_pool::BlockCopyId {
    match &lease.groups[1] {
        GroupLease::Ring { copy: Some(copy) } => *copy,
        other => panic!("{other:?}"),
    }
}

#[test]
fn hybrid_hit_is_the_longest_prefix_every_cacheable_group_serves() {
    // coord:842-977: the fixed point shortens to 4,352; coord:970-977
    // reports the uncached common prefix.
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704, 10880]);
        seed_states(pool, &GLM_STATES, hash, 4352);
        let hit = coordinator.find_longest_hit(pool, &lease(11000, hash), 11000);
        assert_eq!(hit.tokens, 4352);
        assert_eq!(hit.junction_tokens, 10880 - 4352);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 2));
    }
}

#[test]
fn hybrid_partial_hash_truncates_full_attention_hit_length() {
    // Ported from v0.31 partial_test:1596. Values hand-derived: the
    // fine-grained probe of single:806-825 extends the two full blocks to
    // the partial entry at 4,356, and the state hit there keeps it.
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm_hash(version, false, 4));
        let hash = 4;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 4356]);
        seed_states(pool, &GLM_STATES, hash, 4356);
        let request = lease(4500, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 4500);
        assert_eq!(hit.tokens, 4356);
        assert_eq!(hit.junction_tokens, 0);
        match &hit.per_group[0] {
            GroupHitBlocks::Attention(keys) => {
                assert_eq!(keys.len(), 3);
                assert_eq!(keys[2], Some(key(0, hash, 4356)));
            }
            other => panic!("{other:?}"),
        }

        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 4356]);
        seed_states(pool, &GLM_STATES, hash, 4352);
        let hit = coordinator.find_longest_hit(pool, &request, 4500);
        assert_eq!(hit.tokens, 4352);
        // coord:970-977: the full-attention hit of 4,356 minus 4,352.
        assert_eq!(hit.junction_tokens, 4);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 2));
    }
}

/// Two full-attention groups of one spec and a state group, all at block 64.
fn two_full_groups_and_a_state() -> [FpmCacheGroup; 3] {
    [
        attention("full_a", 64, None),
        attention("full_b", 64, None),
        state("state", 64, true, false),
    ]
}

#[test]
fn groups_of_one_spec_hit_jointly_and_set_the_junction_together() {
    // coord:737-779 (v0.32 coord:708-750) looks up the groups of one spec
    // together, and pool:198-224 finds a block only when every group of the
    // spec holds it. The longest hit is the largest result of one spec's
    // lookup (coord:947, :970-977). Group a holds 192 tokens, group b 64 and
    // the state group the state at 64, so vLLM hits 64 with no junction.
    let groups = two_full_groups_and_a_state();
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 64, version).unwrap());
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 64, &[64, 128, 192]);
        seed_many(pool, 1, 64, &[64]);
        seed_many(pool, 2, 64, &[64]);
        let hit = coordinator.find_longest_hit(pool, &lease(401, 64), 401);
        assert_eq!((hit.tokens, hit.junction_tokens), (64, 0), "{version:?}");
        assert_eq!(
            hit.per_group[0],
            GroupHitBlocks::Attention(vec![Some(key(0, 64, 64))])
        );
        assert_eq!(
            hit.per_group[1],
            GroupHitBlocks::Attention(vec![Some(key(1, 64, 64))])
        );
        assert_eq!(
            hit.per_group[2],
            GroupHitBlocks::State(Some(key(2, 64, 64)))
        );
        // So the first chunk runs to the last cacheable stop 384 (v0.31
        // sched:444, v0.32 sched:476-480). A junction at 192 would end it
        // there.
        let junction = hit.tokens + hit.junction_tokens;
        assert_eq!(
            split(coordinator.rules(), 64, 401, 401, BUDGET, junction),
            320,
            "{version:?}"
        );
    }
}

#[test]
fn groups_of_one_spec_hit_inside_a_block_only_jointly() {
    // With fine-grained hits, single:794-828 probes the hash boundaries of the
    // first partial block for every group of the spec at once. Group a holds
    // two full blocks, group b one full block and the partial entry at 96.
    // No boundary inside the second block is cached in both, so vLLM hits 64.
    let groups = two_full_groups_and_a_state();
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let rules = GroupedPrefixRules::new(&groups, 16, version).unwrap();
        assert_eq!(rules.alignment, 16);
        let coordinator = with_limits(rules);
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 16, &[64, 128]);
        seed_many(pool, 1, 16, &[64, 96]);
        seed_many(pool, 2, 16, &[64, 96]);
        let hit = coordinator.find_longest_hit(pool, &lease(200, 16), 200);
        assert_eq!((hit.tokens, hit.junction_tokens), (64, 0), "{version:?}");
        assert_eq!(
            hit.per_group[0],
            GroupHitBlocks::Attention(vec![Some(key(0, 16, 64))])
        );
        assert_eq!(
            hit.per_group[1],
            GroupHitBlocks::Attention(vec![Some(key(1, 16, 64))])
        );
        assert_eq!(
            hit.per_group[2],
            GroupHitBlocks::State(Some(key(2, 16, 64)))
        );
    }
}

#[test]
fn state_groups_of_one_spec_hit_jointly() {
    // Without a full-attention group, the first spec looked up is the first
    // one in group order (coord:775-779), here the two state groups. They
    // hold the states at 64 and 192, and at 64. The joint lookup
    // (single:1540-1564) hits 64 and leaves no junction.
    let groups = [
        state("state_a", 64, true, false),
        state("state_b", 64, true, false),
        attention("window", 64, Some(65)),
    ];
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 64, version).unwrap());
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 64, &[64, 192]);
        seed_many(pool, 1, 64, &[64]);
        seed_many(pool, 2, 64, &[64]);
        let hit = coordinator.find_longest_hit(pool, &lease(401, 64), 401);
        assert_eq!((hit.tokens, hit.junction_tokens), (64, 0), "{version:?}");
        assert_eq!(
            hit.per_group[0],
            GroupHitBlocks::State(Some(key(0, 64, 64)))
        );
        assert_eq!(
            hit.per_group[1],
            GroupHitBlocks::State(Some(key(1, 64, 64)))
        );
    }
    // On GLM-5.3-Flash the full-attention group G0 comes first, so its hit
    // sets the junction, as in vLLM. The extra state of G2 at 8,704 changes
    // nothing, and the four KDA groups hit 4,352 together.
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704]);
        seed_states(pool, &GLM_STATES, hash, 4352);
        seed_states(pool, &[2], hash, 8704);
        let hit = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
        assert_eq!(
            (hit.tokens, hit.junction_tokens),
            (4352, 8704 - 4352),
            "{version:?}"
        );
        for group in GLM_STATES {
            assert_eq!(
                hit.per_group[usize::from(group)],
                GroupHitBlocks::State(Some(key(group, hash, 4352)))
            );
        }
    }
}

#[test]
fn admission_counts_inactive_hits_and_state_copies() {
    // single:216-268 counts inactive hit blocks again. The keys are seeded
    // directly, because a finished request of 4,352 tokens keeps its state
    // key only at its replay boundary, 2,176.
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352]);
        seed_states(pool, &GLM_STATES, hash, 4352);
        let hit = coordinator.find_longest_hit(pool, &lease(4400, hash), 4400);
        assert_eq!(hit.tokens, 4352);
        // 2 inactive G0 blocks, 1 new G0 block, 1 ring, 4 x (running +
        // inactive cached state block).
        assert_eq!(coordinator.admission_blocks(pool, &hit, 4400), 12);
        // Another running request holds the two G0 blocks active.
        let other = [key(0, hash, 2176), key(0, hash, 4352)];
        let mut reservation = pool.reserve(&other, 0).unwrap();
        let held = pool.activate(&mut reservation);
        assert_eq!(held.len(), 2);
        assert_eq!(coordinator.admission_blocks(pool, &hit, 4400), 10);
        pool.cancel(reservation);
    }
}

#[test]
fn resumed_state_holds_cached_and_running_blocks_during_the_first_step() {
    // single:1858-1868: a resumed request holds the hit block and the
    // running block. single:1658-1675 frees the hit block once
    // `cdiv(processed, block) - 1 > hit_index`: `cdiv(6000, 2176) - 1 ==
    // 2 > 1`.
    let groups = groups_glm(false);
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    let hash = 2176;
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[2176, 4352]);
    seed_states(pool, &GLM_STATES, hash, 4352);
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    assert_eq!(hit.tokens, 4352);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert_eq!(state.hit.map(|(_, index)| index), Some(1));
        assert!(state.running.is_none());
    }
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 4352, 6000, 1));
    let state_blocks: usize = GLM_STATES
        .iter()
        .map(|&group| {
            let state = state_lease(&block_lease, usize::from(group));
            assert!(state.hit.is_some() && state.running.is_some());
            assert!(state.previous.is_none() && state.copy.is_none());
            2
        })
        .sum();
    assert_eq!(state_blocks, 8);
    // G0: 2 hits + 1 new, ring 1, states 8.
    assert_eq!(pool.active_blocks(), 12);
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 6000, &hit, 1);
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.hit.is_none() && state.running.is_some());
    }
    assert_eq!(pool.active_blocks(), 8);
    assert_eq!(pool.inactive_cached_blocks(), 4);
}

#[test]
fn boundary_crossing_keeps_the_previous_running_block_for_one_step() {
    // single:1658-1675 and mamba_utils:576-584: the block of the earlier
    // boundary goes at the next allocation. One pass, as on the linear path,
    // where vLLM's async scheduling holds it one more.
    let groups = groups_glm(false);
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    let hash = 2176;
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 0, 2176, 1));
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 2176, &hit, 1);
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 2176, 4352, 2));
    let state_and_ring = |lease: &GroupedBlockLease| -> usize {
        lease.groups[1..]
            .iter()
            .map(|g| match g {
                GroupLease::Attention { .. } => 0,
                GroupLease::Ring { copy } => usize::from(copy.is_some()),
                GroupLease::State(s) => [
                    s.running,
                    s.hit.map(|h| h.0),
                    s.previous.map(|p| p.0),
                    s.copy,
                    s.checkpoint.map(|c| c.0),
                ]
                .iter()
                .flatten()
                .count(),
            })
            .sum()
    };
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.running.is_some());
        assert_eq!(state.previous.map(|(_, step)| step), Some(2));
    }
    assert_eq!(state_and_ring(&block_lease), 9);
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 4352, &hit, 2);
    assert_eq!(state_and_ring(&block_lease), 9);
    coordinator.begin_step();
    let before = pool.active_blocks();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 4352, 6528, 3));
    // Four previous blocks go, four new running blocks and one G0 block come.
    assert_eq!(pool.active_blocks(), before + 1);
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert_eq!(state.previous.map(|(_, step)| step), Some(3));
    }
    assert_eq!(state_and_ring(&block_lease), 9);
}

#[test]
fn partial_hit_takes_a_copy_block_for_the_shared_block() {
    // single:387-409 and single:1770-1777: a hit that ends inside a block
    // takes one copy block. single:800-805: the full-attention probe needs
    // the full blocks below the partial entry, so blocks 0 and 1 are
    // cached too.
    let groups = [attention("full", 16, None), state("s", 32, true, false)];
    let r = rules(&groups, 4);
    assert_eq!(r.alignment, 4);
    let mut coordinator = with_limits(r);
    let hash = 4;
    let mut kv = kv_pool(&groups, 32);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[16, 32, 36]);
    seed_many(pool, 1, hash, &[36]);
    let request = lease(50, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 50);
    assert_eq!(hit.tokens, 36);
    // G0: 4 live - 3 hits + 3 inactive + 1 copy = 5. State
    // (single:1757-1777): `cdiv(50, 32) - cdiv(36, 32) == 0` new blocks
    // plus the copy block, plus the inactive hit = 2.
    assert_eq!(coordinator.admission_blocks(pool, &hit, 50), 7);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert_eq!(pool.active_blocks(), 4);
    assert!(coordinator.allocate(pool, &mut block_lease, 50, 36, 50, 1));
    // G0 holds blocks 0 to 3, with block 2 as a private copy. The shared
    // block 2 is inactive again. The target 50 stays in the hit's block,
    // so the state's copy-on-write block is its running block
    // (single:1891-1895, `_apply_cow` at single:454-474); the hit block
    // stays until the copy has run (sched:1434-1442).
    assert_eq!(attention_copies(&block_lease, 0), 4);
    let state = state_lease(&block_lease, 1);
    assert!(state.hit.is_some() && state.running.is_some() && state.copy.is_none());
    assert_eq!(pool.active_blocks(), 4 + 2);
    assert_eq!(pool.inactive_cached_blocks(), 1);
    coordinator.finalize(pool, &mut block_lease, &request, 50, 50, &hit, 1);
    let state = state_lease(&block_lease, 1);
    assert!(state.hit.is_none() && state.running.is_some());
    assert_eq!(pool.active_blocks(), 4 + 1);
    assert_eq!(pool.inactive_cached_blocks(), 2);

    // A target beyond the hit's block: the copy sits at the hit's index
    // and a running block follows. `remove_skipped_blocks` frees the
    // copy at the next allocation (single:1658-1675, index set at :1868).
    let mut kv = kv_pool(&groups, 32);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[16, 32, 36]);
    seed_many(pool, 1, hash, &[36]);
    let request = lease(70, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 70);
    assert_eq!(hit.tokens, 36);
    // G0: 5 live - 3 hits + 3 inactive + 1 copy = 6. State: `cdiv(70, 32)
    // - cdiv(36, 32) == 1` new block plus the copy block plus the
    // inactive hit = 3.
    assert_eq!(coordinator.admission_blocks(pool, &hit, 70), 9);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 70, 36, 70, 1));
    let state = state_lease(&block_lease, 1);
    assert!(state.hit.is_some() && state.running.is_some() && state.copy.is_some());
    assert_eq!(pool.active_blocks(), 5 + 3);
    coordinator.finalize(pool, &mut block_lease, &request, 70, 70, &hit, 1);
    let state = state_lease(&block_lease, 1);
    assert!(state.hit.is_none() && state.copy.is_some());
    assert_eq!(pool.active_blocks(), 5 + 2);
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 70, 70, 71, 2));
    let state = state_lease(&block_lease, 1);
    assert!(state.copy.is_none() && state.running.is_some());
    assert_eq!(pool.active_blocks(), 5 + 1);
}

#[test]
fn finalize_publishes_states_at_the_replay_boundary_and_the_junction() {
    // single:1614-1622 with retention 0 (cache:160-166): the replay
    // boundary `(9000 - 1) // 2176 * 2176 == 8704` and the junction.
    let groups = groups_glm(false);
    let hash = 2176;
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    let mut block_lease = coordinator.new_lease();
    let ends = prefill(
        &mut coordinator,
        pool,
        &mut block_lease,
        &request,
        9000,
        &hit,
        2176,
    );
    assert_eq!(ends, vec![2176, 4352, 6528, 8704, 9000]);
    coordinator.release_request(pool, &mut block_lease);
    for group in GLM_STATES {
        for end in [2176, 4352, 6528] {
            assert!(
                !hit_exists(pool, group, hash, end),
                "group {group} at {end}"
            );
        }
        assert!(hit_exists(pool, group, hash, 8704));
    }
    // G0 blocks 0 to 3 and four state blocks stay cached.
    assert_eq!(pool.inactive_cached_blocks(), 8);
    assert_eq!(pool.active_blocks(), 0);

    // A hit of 4,352 with a junction of 2,176.
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[2176, 4352, 6528]);
    seed_states(pool, &GLM_STATES, hash, 4352);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    assert_eq!((hit.tokens, hit.junction_tokens), (4352, 2176));
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    let ends = prefill(
        &mut coordinator,
        pool,
        &mut block_lease,
        &request,
        9000,
        &hit,
        2176,
    );
    assert_eq!(ends, vec![6528, 8704, 9000]);
    coordinator.release_request(pool, &mut block_lease);
    for group in GLM_STATES {
        assert!(hit_exists(pool, group, hash, 6528), "group {group}");
        assert!(hit_exists(pool, group, hash, 8704), "group {group}");
        assert!(!hit_exists(pool, group, hash, 2176), "group {group}");
    }
}

#[test]
fn align_split_follows_vllm_0310() {
    // sched:414-527 at v0.31.
    let r = rules_glm(VllmRuleVersion::V0_31_0, false);
    // The floor of 8,192 to 2,176 is 6,528 (sched:460-468).
    assert_eq!(split(&r, 0, 9000, 9000, 8192, 0), 6528);
    // The last cacheable stop `9000 - 9000 % 2176 == 8704` (sched:444).
    assert_eq!(split(&r, 6528, 9000, 9000, 8192, 0), 2176);
    assert_eq!(split(&r, 8704, 9000, 9000, 8192, 0), 296);
    // A prompt of exactly 8,704 tokens has no stop at 6,528, because its
    // last cacheable stop `8704 - 8704 % 2176` is the prompt itself
    // (sched:444).
    assert_eq!(split(&r, 6528, 8704, 8704, 8192, 0), 2176);
    // With hash 4 the tail 9,000 equals the prompt and adds no stop
    // (sched:482-487, :512-514). With prompt 9,002 the tail is 9,000.
    let fine = rules_glm_hash(VllmRuleVersion::V0_31_0, false, 4);
    assert_eq!(split(&fine, 8704, 9000, 9000, 8192, 0), 296);
    assert_eq!(split(&fine, 8704, 9002, 9002, 8192, 0), 296);
    // The junction stop (sched:488-519).
    assert_eq!(split(&r, 4352, 9000, 9000, 8192, 6528), 2176);
    // Past the prefill the split returns the request (sched:434-436).
    assert_eq!(split(&r, 9000, 9000, 9001, 1, 0), 1);
}

#[test]
fn align_split_follows_vllm_0320() {
    // v0.32 sched:443-556.
    let r = rules_glm(VllmRuleVersion::V0_32_0, false);
    assert_eq!(split(&r, 0, 9000, 9000, 8192, 0), 6528);
    assert_eq!(split(&r, 6528, 9000, 9000, 2472, 0), 2176);
    // The last stop of 8,704 tokens is `(8704 - 1) // 2176 * 2176 == 6528`
    // (v0.32 sched:476-480).
    assert_eq!(split(&r, 4352, 8704, 8704, 4352, 0), 2176);
    let v31 = rules_glm(VllmRuleVersion::V0_31_0, false);
    assert_eq!(split(&v31, 4352, 8704, 8704, 4352, 0), 4352);
    // Checkpoint mode (v0.32 sched:482-501, iface:1125-1146). For
    // (6528, 9000, pos 8704): 6528 % 2176 == 0; cdiv(9000, 2176) - 2 == 3
    // > 6527 / 2176 == 2; 6528 + 2176 <= 8704; 6528 < 8704 < 9000;
    // 2176 % 16 == 0.
    let ckpt = rules_glm(VllmRuleVersion::V0_32_0, true);
    assert_eq!(ckpt.checkpoint_position(9000), 8704);
    assert!(ckpt.checkpoint_valid(6528, 9000, 8704));
    assert_eq!(split(&ckpt, 6528, 9000, 9000, 2472, 0), 2472);
    // For (4352, 8704, pos 6528): 4352 % 2176 == 0; cdiv(8704, 2176) - 2
    // == 2 > 4351 / 2176 == 1; 4352 + 2176 <= 6528; 4352 < 6528 < 8704;
    // 2176 % 16 == 0.
    assert_eq!(ckpt.checkpoint_position(8704), 6528);
    assert!(ckpt.checkpoint_valid(4352, 8704, 6528));
    assert_eq!(split(&ckpt, 4352, 8704, 8704, 4352, 0), 4352);
    // Without a checkpoint rule the check is false.
    assert!(!r.checkpoint_valid(6528, 9000, 8704));
}

#[test]
fn align_split_with_a_checkpoint_block_ports_vllm_0320() {
    // Ported from v0.32 split_test:127-180, which stubs `block_size` to
    // 1,600 and `hash_block_size` to 16 (split_test:139-153). The engine
    // shape with the same numbers: one full-attention rule and one state
    // rule at block 1,600, hash 16, checkpoint 1 at alignment 16. The
    // state block is above the hash block and the other rule is full
    // attention, so fine-grained hits are on and A is 16.
    let groups = [attention("full", 1600, None), state("s", 1600, true, true)];
    let r = GroupedPrefixRules::new(&groups, 16, VllmRuleVersion::V0_32_0).unwrap();
    assert_eq!(r.alignment, 16);
    assert_eq!(r.cacheable_block, 1600);
    // The engine's checkpoint position is `(2002 - 1) / 16 * 16 == 2000`.
    // The vLLM test stubs partial hits off and gets 1,600. Both positions
    // pass the five conditions (0 % 16 == 0; cdiv(2002, 1600) - 2 == 0 >
    // -1; 0 + 16 <= pos; 0 < pos < 2002; pos % 16 == 0), so both give
    // 2,002.
    assert_eq!(r.checkpoint_position(2002), 2000);
    assert!(r.checkpoint_valid(0, 2002, 2000));
    assert!(r.checkpoint_valid(0, 2002, 1600));
    assert_eq!(r.align_split(0, 2002, 2002, 2002, 0, 16384), 2002);
    // The end 2,000 is below the prompt, so the floor to 1,600 applies
    // and no checkpoint is taken.
    assert_eq!(r.align_split(0, 3602, 3602, 2000, 0, 16384), 1600);
    let groups = [attention("full", 1600, None), state("s", 1600, true, false)];
    let v31 = GroupedPrefixRules::new(&groups, 16, VllmRuleVersion::V0_31_0).unwrap();
    assert_eq!(v31.align_split(0, 2002, 2002, 2002, 0, 16384), 1600);
    assert_eq!(v31.align_split(0, 3602, 3602, 2000, 0, 16384), 1600);
}

#[test]
fn align_split_needs_a_mamba_state_group() {
    // vLLM splits only with Mamba layers in align mode (v0.31 `sched:354-356`,
    // `:695-697`, v0.32 `sched:377-379`, `:724-726`). Without a state rule
    // every chunk is the desired count, on both rows.
    let full = [attention("full", 64, None)];
    let windowed = [attention("full", 64, None), attention("swa", 64, Some(128))];
    // A tail ring, such as the GLM `KpoolTailSpec`, is not a MambaSpec.
    let ring = [attention("full", 64, None), state("ring", 4, false, false)];
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        for groups in [&full[..], &windowed[..], &ring[..]] {
            let r = GroupedPrefixRules::new(groups, 64, version).unwrap();
            // The last block boundary of the prompt is no stop.
            assert_eq!(split(&r, 0, 300, 300, 300, 0), 300, "{version:?}");
            // A chunk that ends below the prompt keeps its length.
            assert_eq!(split(&r, 0, 1000, 1000, 500, 0), 500, "{version:?}");
            // A start inside a block runs past the next boundary.
            assert_eq!(split(&r, 100, 1000, 1000, 500, 0), 500, "{version:?}");
            // The junction is no stop.
            assert_eq!(split(&r, 64, 1000, 1000, 800, 640), 800, "{version:?}");
        }
        // A state group keeps the split: the floor of 8,192 to 2,176.
        assert_eq!(
            split(&rules_glm(version, false), 0, 9000, 9000, 8192, 0),
            6528,
            "{version:?}"
        );
    }
}

#[test]
fn partial_tail_moves_one_unit_down_on_vllm_0320() {
    // At hash 4 the 0.32.0 tail, `(prompt - 1) / hash * hash`, is 8,996,
    // between the last stop 8,704 and the prompt (v0.32 sched:517-525,
    // single:860-868, single:2069-2073). The 0.31.0 tail is 9,000.
    let groups = groups_glm(false);
    let hash = 4;
    for (version, expected_ends, expected_hit) in [
        (VllmRuleVersion::V0_32_0, vec![6528, 8704, 8996, 9000], 8996),
        (VllmRuleVersion::V0_31_0, vec![6528, 8704, 9000], 8704),
    ] {
        let mut coordinator = with_limits(rules_glm_hash(version, false, hash));
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(9000, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 9000);
        let mut block_lease = coordinator.new_lease();
        let ends = prefill(
            &mut coordinator,
            pool,
            &mut block_lease,
            &request,
            9000,
            &hit,
            BUDGET,
        );
        assert_eq!(ends, expected_ends, "{version:?}");
        if version == VllmRuleVersion::V0_32_0 {
            for group in GLM_STATES {
                assert!(hit_exists(pool, group, hash, 8704));
                assert!(hit_exists(pool, group, hash, 8996));
            }
        }
        let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
        assert_eq!(resend.tokens, expected_hit, "{version:?}");
        coordinator.release_request(pool, &mut block_lease);
        let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
        assert_eq!(resend.tokens, expected_hit, "{version:?}");
    }
}

#[test]
fn checkpoint_block_is_taken_in_the_final_chunk_and_published_at_the_prompt_checkpoint() {
    // v0.32 single:1643-1674, :1745-1773 (one more block in the chunk
    // that reaches the prompt), :2017-2066 (published at the prompt
    // checkpoint under retention 0).
    let groups = groups_glm(true);
    let hash = 2176;
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_32_0, true));
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    assert_eq!(hit.tokens, 0);
    // 5 for G0, 1 for the ring, 2 for each KDA group.
    assert_eq!(coordinator.admission_blocks(pool, &hit, 9000), 14);
    let v31 = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    assert_eq!(v31.admission_blocks(pool, &hit, 9000), 10);

    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert_eq!(split(coordinator.rules(), 0, 9000, 9000, BUDGET, 0), 6528);
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 0, 6528, 1));
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.running.is_some());
        assert!(state.previous.is_none() && state.checkpoint.is_none() && state.hit.is_none());
    }
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 6528, &hit, 1);
    coordinator.begin_step();
    assert_eq!(
        split(coordinator.rules(), 6528, 9000, 9000, BUDGET, 0),
        2472
    );
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 6528, 9000, 2));
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.running.is_some() && state.previous.is_some());
        assert_eq!(state.checkpoint.map(|(_, position)| position), Some(8704));
    }
    // G0 5, ring 1, 4 x (running + previous + checkpoint).
    assert_eq!(pool.active_blocks(), 5 + 1 + 12);
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 9000, &hit, 2);
    for group in GLM_STATES {
        assert!(hit_exists(pool, group, hash, 8704), "group {group}");
        let state = state_lease(&block_lease, usize::from(group));
        let (checkpoint, _) = state.checkpoint.unwrap();
        assert!(!pool.is_private(checkpoint));
        assert!(pool.is_private(state.running.unwrap()));
    }
    coordinator.begin_step();
    let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
    assert_eq!(resend.tokens, 8704);
    coordinator.release_request(pool, &mut block_lease);
    // Four G0 blocks and the four checkpoint copies stay in the LRU.
    assert_eq!(pool.inactive_cached_blocks(), 8);
    assert_eq!(pool.active_blocks(), 0);
    let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
    assert_eq!(resend.tokens, 8704);
}

#[test]
fn checkpoint_block_goes_at_the_next_step_and_stays_cached_until_eviction() {
    // A Mamba manager skips every token before the last computed one (v0.32
    // single:1954-1959). So the request's next allocation first frees every
    // block below its last block, the checkpoint block included (v0.32
    // kv_cache_manager.py:546-559, single:672-710, :1592-1641). A freed
    // block keeps its key in the LRU (v0.32 block_pool.py:774-803).
    let groups = groups_glm(true);
    let hash = 2176;
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_32_0, true));
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    let mut block_lease = coordinator.new_lease();
    let ends = prefill(
        &mut coordinator,
        pool,
        &mut block_lease,
        &request,
        9000,
        &hit,
        BUDGET,
    );
    assert_eq!(ends, [6528, 9000]);
    // G0 5, ring 1, 4 x (running + previous + checkpoint).
    assert_eq!(pool.active_blocks(), 5 + 1 + 12);
    // The prefill ran steps 1 and 2. Each decode step holds G0 5, the ring
    // and the four running blocks only, and a resend still hits the
    // checkpoint keys.
    for (step, target) in [(3, 9001), (4, 9002)] {
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, 9000, target - 1, target, step));
        assert_eq!(pool.active_blocks(), 5 + 1 + 4, "step {step}");
        for group in GLM_STATES {
            let state = state_lease(&block_lease, usize::from(group));
            assert!(state.checkpoint.is_none() && state.previous.is_none());
            let cached = pool.key_hit(group, hash_at(hash, 8704));
            assert!(cached.is_some_and(|hit| !hit.is_active), "group {group}");
        }
        coordinator.finalize(pool, &mut block_lease, &request, 9000, target, &hit, step);
        let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
        assert_eq!(resend.tokens, 8704, "step {step}");
    }
    // The checkpoint keys go only when a new block evicts them, oldest first
    // in the order the groups freed them (v0.32 coord:447-450).
    assert_eq!(pool.inactive_cached_blocks(), 4);
    let free = pool.capacity_blocks() - pool.active_blocks() - 4;
    let mut reservation = pool.reserve(&[], free + 1).unwrap();
    for _ in 0..=free {
        pool.take_private(&mut reservation);
    }
    pool.cancel(reservation);
    assert!(!hit_exists(pool, 2, hash, 8704));
    for group in [3, 4, 5] {
        assert!(hit_exists(pool, group, hash, 8704), "group {group}");
    }
}

#[test]
fn checkpoint_block_publishes_the_prompt_checkpoint_under_retention_zero() {
    // Ported from v0.32 partial_test:775-807: the prompt-end checkpoint
    // `(prompt - 1) / 32 * 32` stays published under retention 0.
    let groups = [attention("full", 32, None), state("s", 32, true, true)];
    let hash = 32;
    for (prompt, expected) in [(120, 96), (104, 96), (96, 64)] {
        let mut coordinator =
            with_limits(GroupedPrefixRules::new(&groups, hash, VllmRuleVersion::V0_32_0).unwrap());
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(prompt, hash);
        let hit = coordinator.find_longest_hit(pool, &request, prompt);
        let mut block_lease = coordinator.new_lease();
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, prompt, 0, prompt, 1));
        assert_eq!(
            state_lease(&block_lease, 1).checkpoint.map(|(_, p)| p),
            Some(expected)
        );
        coordinator.finalize(pool, &mut block_lease, &request, prompt, prompt, &hit, 1);
        assert!(hit_exists(pool, 1, hash, expected), "prompt {prompt}");
        coordinator.release_request(pool, &mut block_lease);
        coordinator.begin_step();
        let resend = coordinator.find_longest_hit(pool, &lease(prompt, hash), prompt);
        assert_eq!(resend.tokens, expected, "prompt {prompt}");
    }
}

#[test]
fn state_key_published_in_a_step_is_not_a_hit_in_that_step() {
    // single:1726-1734: a state block cached in this step is not a hit
    // until the next step. The block pool stores no step, so the coordinator
    // keeps the set.
    let groups = groups_glm(false);
    let hash = 2176;
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(2500, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 2500);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 2500, 0, 2176, 1));
    coordinator.finalize(pool, &mut block_lease, &request, 2500, 2176, &hit, 1);
    for group in GLM_STATES {
        assert!(hit_exists(pool, group, hash, 2176));
    }
    let same_step = coordinator.find_longest_hit(pool, &lease(2500, hash), 2500);
    assert_eq!(same_step.tokens, 0);
    coordinator.begin_step();
    let next_step = coordinator.find_longest_hit(pool, &lease(2500, hash), 2500);
    assert_eq!(next_step.tokens, 2176);
}

#[test]
fn allocate_fails_atomically() {
    let groups = groups_glm(false);
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    // Room for the first chunk (G0 1, ring 1, 4 running) but not the
    // second (G0 1 more, 4 new running blocks while 4 previous stay).
    let mut kv = kv_pool(&groups, 9);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, 2176);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 0, 2176, 1));
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 2176, &hit, 1);
    assert_eq!(pool.active_blocks(), 6);
    coordinator.begin_step();
    let before = format!("{block_lease:?}");
    assert!(!coordinator.allocate(pool, &mut block_lease, 9000, 2176, 4352, 2));
    assert_eq!(pool.active_blocks(), 6);
    assert_eq!(format!("{block_lease:?}"), before);
    coordinator.release_request(pool, &mut block_lease);
    assert_eq!(pool.active_blocks(), 0);
}

#[test]
fn release_frees_groups_in_profile_order_each_tail_first() {
    // coord:387-395 frees the managers in group order, and each manager frees
    // its blocks tail first (single:581-589). Cached blocks join the LRU tail
    // in that order (pool:777-808). So the cached block of a state group that
    // the profile lists first is evicted before the attention blocks.
    let groups = [state("s", 64, true, false), attention("full", 64, None)];
    let hash = 64;
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 6);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(200, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 200);
        let mut block_lease = coordinator.new_lease();
        let ends = prefill(
            &mut coordinator,
            pool,
            &mut block_lease,
            &request,
            200,
            &hit,
            256,
        );
        assert_eq!(ends, vec![192, 200], "{version:?}");
        coordinator.release_request(pool, &mut block_lease);
        // The state at the replay boundary 192 and the attention blocks that
        // end at 64, 128 and 192 stay cached. Two blocks are free.
        assert_eq!(pool.inactive_cached_blocks(), 4, "{version:?}");
        assert_eq!(pool.active_blocks(), 0, "{version:?}");
        // Three fresh blocks evict one cached block: the state block.
        let reservation = pool.reserve(&[], 3).unwrap();
        pool.cancel(reservation);
        assert!(!hit_exists(pool, 0, hash, 192), "{version:?}");
        for end in [64, 128, 192] {
            assert!(hit_exists(pool, 1, hash, end), "{version:?} at {end}");
        }
        // Four fresh blocks evict one more: the attention tail block.
        let reservation = pool.reserve(&[], 4).unwrap();
        pool.cancel(reservation);
        assert!(!hit_exists(pool, 1, hash, 192), "{version:?}");
        for end in [64, 128] {
            assert!(hit_exists(pool, 1, hash, end), "{version:?} at {end}");
        }
    }
}

#[test]
fn checkpoint_rule_needs_state_groups_at_the_smallest_cacheable_block() {
    // vLLM 0.32.0 tests a checkpoint against `cache_config.block_size`, the
    // smallest prefix-cacheable block, in the scheduler (v0.32 sched:469,
    // :496) and against the Mamba block in the manager (v0.32 single:1652,
    // :1662). The rule table holds one block for both.
    let small_attention = [attention("full", 16, None), state("s", 64, true, true)];
    let err = GroupedPrefixRules::new(&small_attention, 16, VllmRuleVersion::V0_32_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with vllm_version=0.32.0 require every state group with prefill checkpoint blocks to have the smallest block_size_tokens of the prefix_cacheable groups"
    );
    // Without checkpoint blocks the same shape is modeled on both rows.
    let no_checkpoint = [attention("full", 16, None), state("s", 64, true, false)];
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        assert!(GroupedPrefixRules::new(&no_checkpoint, 16, version).is_ok());
    }
    // A larger attention block keeps the state block the smallest. The
    // checkpoint position then moves in units of the scheduler block, the LCM
    // 128, in the manager (v0.32 coord:699-706) and in the scheduler (v0.32
    // sched:473-475, :209): `(300 - 1) / 128 * 128 == 256`.
    let large_attention = [attention("full", 128, None), state("s", 64, true, true)];
    let r = GroupedPrefixRules::new(&large_attention, 64, VllmRuleVersion::V0_32_0).unwrap();
    assert_eq!((r.alignment, r.cacheable_block), (128, 64));
    assert_eq!(r.checkpoint_position(300), 256);
}

#[test]
fn retired_window_blocks_are_released_tail_first() {
    // v0.31 single:660-685 and v0.32 single:645-670 free a retired window
    // range backward, and free_blocks appends its cached blocks to the free
    // queue in that order (v0.31 pool:777-806, v0.32 pool:774-803). So the
    // tail block of the range is evicted first.
    let groups = [attention("window", 4, Some(8))];
    let hash = 4;
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 8);
        let pool = kv.block_pool_mut().unwrap();
        // The run that a finished request of 100 tokens keeps before its
        // replay boundary 96: the blocks that end at 92 and 96.
        seed_many(pool, 0, hash, &[92, 96]);
        let request = lease(108, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 108);
        assert_eq!(hit.tokens, 96, "{version:?}");
        let mut block_lease = coordinator.new_lease();
        coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
        assert!(coordinator.allocate(pool, &mut block_lease, 108, 96, 104, 1));
        coordinator.finalize(pool, &mut block_lease, &request, 108, 104, &hit, 1);
        // The next allocation retires both hit blocks: (104 - 7) / 4 = 24.
        assert!(coordinator.allocate(pool, &mut block_lease, 108, 104, 108, 2));
        assert_eq!(pool.inactive_cached_blocks(), 2, "{version:?}");
        assert_eq!(pool.active_blocks(), 3, "{version:?}");
        // Three blocks are free, so four fresh blocks evict one cached block:
        // the tail block of the retired range, which ends at 96.
        let reservation = pool.reserve(&[], 4).unwrap();
        pool.cancel(reservation);
        assert!(!hit_exists(pool, 0, hash, 96), "{version:?}");
        assert!(hit_exists(pool, 0, hash, 92), "{version:?}");
    }
}

/// The sliding-window search before the single traversal: every candidate
/// end scans its own run again.
fn rescanning_window_run(
    max: usize,
    block: usize,
    step: usize,
    needed: usize,
    mut resident: impl FnMut(usize) -> bool,
) -> (usize, usize) {
    let mut end = max / step * step;
    while end > 0 {
        let blocks = end / block;
        let mut run = 0;
        while run < blocks && run < needed && resident(blocks - run - 1) {
            run += 1;
        }
        if run >= needed || run == blocks {
            return (end, run);
        }
        end -= step;
    }
    (0, 0)
}

#[test]
fn window_search_looks_up_each_block_at_most_once() {
    // single:1018-1044 finds a sliding-window hit in one backward traversal.
    // 4,096 candidate blocks of which the last 2,048 are resident, and a
    // window that needs 4,096 contiguous blocks: no hit.
    let (block, needed) = (16, 4096);
    let resident = |i: usize| i >= 2048;
    let mut lookups = 0;
    let found = window_run(4096 * block, block, block, needed, |i| {
        lookups += 1;
        resident(i)
    });
    let mut rescans = 0;
    let rescanned = rescanning_window_run(4096 * block, block, block, needed, |i| {
        rescans += 1;
        resident(i)
    });
    assert_eq!(found, (0, 0));
    assert_eq!(rescanned, found);
    assert_eq!(rescans, 2_102_272);
    assert_eq!(lookups, 4096);
}

#[test]
fn window_search_finds_the_hits_of_the_rescanning_search() {
    // A fixed xorshift sequence: random residency, blocks, alignments,
    // windows and lengths. Each block is looked up at most once.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    for case in 0..20_000 {
        let block = 1 + next(8) as usize;
        let step = block * (1 + next(4) as usize);
        let needed = next(7) as usize;
        let max = next(400) as usize;
        let density = next(101);
        let blocks = max / block;
        let resident: Vec<bool> = (0..blocks).map(|_| next(100) < density).collect();
        let mut lookups = vec![0; blocks];
        let found = window_run(max, block, step, needed, |i| {
            lookups[i] += 1;
            resident[i]
        });
        let expected = rescanning_window_run(max, block, step, needed, |i| resident[i]);
        assert_eq!(
            found, expected,
            "case {case}: max {max} block {block} step {step} needed {needed} {resident:?}"
        );
        assert!(lookups.iter().all(|&count| count <= 1), "case {case}");
    }
}

/// Whether a scan of every completed attention block, as finalization did
/// before it kept a cursor, would still find a block to publish.
fn rescan_finds_a_block(
    coordinator: &GroupedPrefixCoordinator,
    pool: &GroupedBlockPool,
    lease: &GroupedBlockLease,
    request: &BlockRequestLease,
    prompt: usize,
    computed_after: usize,
    hit: &GroupedHit,
) -> bool {
    use crate::engine::cache::vllm_block_pool::BlockCopyId;
    let rules = coordinator.rules();
    let junction = (hit.junction_tokens > 0).then_some(hit.tokens + hit.junction_tokens);
    let publishable = |g: usize, copy: Option<&Option<BlockCopyId>>, end: usize| {
        copy.is_some_and(|copy| copy.is_some_and(|id| pool.is_private(id)))
            && grouped_key(g, request.prefix_hash_at(end)).is_some()
    };
    rules.groups.iter().enumerate().any(|(g, rule)| {
        let GroupLease::Attention { copies, .. } = &lease.groups[g] else {
            return false;
        };
        let block = rule.block;
        match rule.kind {
            GroupRuleKind::FullAttention => {
                (0..computed_after / block).any(|i| publishable(g, copies.get(i), (i + 1) * block))
            }
            GroupRuleKind::SlidingWindow { window } => [
                Some(rules.replay_boundary(prompt, block)),
                junction.map(|junction| rules.block_floor(junction, block)),
            ]
            .into_iter()
            .flatten()
            .filter(|&pos| pos > 0 && pos <= computed_after)
            .any(|pos| {
                let top = pos / block;
                (top.saturating_sub((window - 1).div_ceil(block))..top)
                    .any(|i| publishable(g, copies.get(i), (i + 1) * block))
            }),
            _ => false,
        }
    })
}

#[test]
fn finalize_publishes_what_a_full_rescan_would() {
    // Two requests share a prefix on a full-attention and a sliding-window
    // group, with chunked prefills and decode steps. After every step a scan
    // of all completed blocks finds nothing more to publish, so the cursor
    // skips only blocks that are cached or have no key.
    let mut state: u64 = 0xD1B5_4A32_D192_ED03;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % bound) as usize
    };
    let mut hits = 0;
    for case in 0..400 {
        let hash = [4, 8][next(2)];
        let groups = [
            attention("full", (hash * (1 + next(2))) as u32, None),
            attention(
                "window",
                (hash * (1 + next(2))) as u32,
                Some(2 + next(40) as u32),
            ),
        ];
        let version = [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0][next(2)];
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 4096);
        let pool = kv.block_pool_mut().unwrap();
        let first = 1 + next(300);
        // Half of the cases resend the whole first prompt, which reaches its
        // replay boundary.
        let shared = if next(2) == 0 {
            first
        } else {
            next(first as u64 + 1)
        };
        let second: Vec<u32> = (0..shared as u32)
            .chain(10_000..10_000 + (1 + next(300)) as u32)
            .collect();
        let first: Vec<u32> = (0..first as u32).collect();
        let mut step = 1;
        for tokens in [first, second] {
            let mut request = BlockRequestLease::new(Uuid::nil(), Vec::new());
            request.configure_prefix_hashes(&tokens, hash);
            let prompt = tokens.len();
            let hit = coordinator.find_longest_hit(pool, &request, prompt);
            hits += usize::from(hit.tokens > 0);
            let mut block_lease = coordinator.new_lease();
            coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
            // A chunked prefill, then up to eight decode steps.
            let (budget, end) = (1 + next(64), prompt + 1 + next(8));
            let mut computed = hit.tokens;
            while computed < end {
                coordinator.begin_step();
                let target = (computed + budget).min(prompt).max(computed + 1);
                assert!(coordinator.allocate(
                    pool,
                    &mut block_lease,
                    prompt,
                    computed,
                    target,
                    step
                ));
                coordinator.finalize(pool, &mut block_lease, &request, prompt, target, &hit, step);
                assert!(
                    !rescan_finds_a_block(
                        &coordinator,
                        pool,
                        &block_lease,
                        &request,
                        prompt,
                        target,
                        &hit
                    ),
                    "case {case} {groups:?} prompt {prompt} hit {} at {target}",
                    hit.tokens
                );
                computed = target;
                step += 1;
            }
            coordinator.release_request(pool, &mut block_lease);
        }
    }
    assert!(hits > 150, "{hits} hits");
}

#[test]
fn first_allocation_test_counts_the_blocks_that_the_first_allocation_takes() {
    // The test before a first allocation (mgr:558-576) counts the inactive
    // hit blocks and the fresh blocks of the chunk. pin_hit and the first
    // allocate hold exactly these at their peak, so the test never refuses a
    // chunk that fits and never admits one that does not.
    let layouts = [
        (
            vec![
                attention("full", 16, None),
                attention("window", 16, Some(40)),
            ],
            16,
            VllmRuleVersion::V0_31_0,
        ),
        (
            vec![attention("full", 64, None), state("s", 64, true, false)],
            16,
            VllmRuleVersion::V0_31_0,
        ),
        (
            vec![attention("full", 64, None), state("s", 64, true, false)],
            16,
            VllmRuleVersion::V0_32_0,
        ),
        (
            vec![
                attention("full", 64, None),
                state("ring", 4, false, false),
                state("s", 64, true, true),
            ],
            64,
            VllmRuleVersion::V0_32_0,
        ),
    ];
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % bound) as usize
    };
    let mut hits = 0;
    for (groups, hash, version) in &layouts {
        for case in 0..150 {
            // A small step budget, so that the admission cap of the window
            // binds. The allocation test leaves it out.
            let mut coordinator = GroupedPrefixCoordinator::new(
                GroupedPrefixRules::new(groups, *hash, *version).unwrap(),
                AdmissionLimits {
                    max_in_flight_tokens: 16,
                    max_model_len: usize::MAX,
                },
            );
            let mut kv = kv_pool(groups, 512);
            let pool = kv.block_pool_mut().unwrap();
            let first = 1 + next(400);
            let request = lease(first, *hash);
            let hit = coordinator.find_longest_hit(pool, &request, first);
            let mut block_lease = coordinator.new_lease();
            let budget = 64 * (1 + next(4));
            prefill(
                &mut coordinator,
                pool,
                &mut block_lease,
                &request,
                first,
                &hit,
                budget,
            );
            coordinator.release_request(pool, &mut block_lease);
            // A second request shares a prefix of the first one.
            let shared = if next(2) == 0 {
                first
            } else {
                next(first as u64 + 1)
            };
            let tokens: Vec<u32> = (0..shared as u32)
                .chain(10_000..10_000 + (1 + next(300)) as u32)
                .collect();
            let mut request = BlockRequestLease::new(Uuid::nil(), Vec::new());
            request.configure_prefix_hashes(&tokens, *hash);
            let prompt = tokens.len();
            coordinator.begin_step();
            let hit = coordinator.find_longest_hit(pool, &request, prompt);
            hits += usize::from(hit.tokens > 0);
            let target = if next(2) == 0 {
                prompt
            } else {
                hit.tokens + 1 + next((prompt - hit.tokens) as u64)
            };
            let needed = coordinator.first_allocation_blocks(pool, &hit, prompt, target);
            let before = pool.active_blocks();
            pool.reset_peak_active_blocks();
            let mut block_lease = coordinator.new_lease();
            coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
            assert!(coordinator.allocate(pool, &mut block_lease, prompt, hit.tokens, target, 100));
            // The copy of a shared block is taken while the hit block is
            // still held, so the peak counts both.
            assert_eq!(
                pool.peak_active_blocks() - before,
                needed,
                "{version:?} case {case}: prompt {prompt} hit {} target {target}",
                hit.tokens
            );
            coordinator.release_request(pool, &mut block_lease);
        }
    }
    assert!(hits > 200, "{hits} hits");
}
