// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hybrid (attention + recurrent state) prefix-cache rules in align mode: vLLM
//! v0.29.0's shared-prefix junction and same-pass wait, and vLLM 43b4aaea's
//! prefill checkpoint and drop switch. `B` is both the block size and the hash
//! unit. See THIRD_PARTY_NOTICES.md for the vLLM sources.

use uuid::Uuid;

use crate::engine::StateCacheConfig;
use crate::engine::common::protocols::{DirectRequest, MockEngineArgs};
use crate::engine::scheduler::EnginePassResult;

use super::core::VllmCore;

const B: usize = 4;

/// One 64-byte block per state at 16 bytes per token.
fn args(prefill_checkpoint: bool, junction_checkpoints: bool) -> MockEngineArgs {
    MockEngineArgs::builder()
        .num_gpu_blocks(256)
        .block_size(B)
        .max_num_batched_tokens(Some(1024))
        .max_num_seqs(Some(16))
        .enable_prefix_caching(true)
        .speedup_ratio(0.0)
        .kv_cache_bytes_per_token(Some(16))
        .state_cache(Some(StateCacheConfig {
            bytes_per_request: 64,
            num_speculative_states: None,
            prefill_checkpoint,
            junction_checkpoints,
        }))
        .prefix_match_unit(Some(B))
        .build()
        .unwrap()
}

fn submit(core: &mut VllmCore, id: u128, tokens: Vec<u32>, output: usize) -> Uuid {
    core.receive(DirectRequest {
        uuid: Some(Uuid::from_u128(id)),
        tokens,
        max_output_tokens: output,
        ..Default::default()
    })
}

fn step(core: &mut VllmCore) -> EnginePassResult {
    core.try_execute_hidden_pass(0.0).unwrap()
}

fn run_to_completion(core: &mut VllmCore) {
    for _ in 0..256 {
        if core.state.requests.is_empty() {
            return;
        }
        step(core);
    }
    panic!("requests did not finish");
}

fn reused(pass: &EnginePassResult, id: Uuid) -> usize {
    pass.admissions
        .iter()
        .find(|admission| admission.uuid == id)
        .unwrap_or_else(|| panic!("{id} was not admitted"))
        .reused_input_tokens
}

fn tokens(prefix: std::ops::Range<u32>, suffix: std::ops::Range<u32>) -> Vec<u32> {
    prefix.chain(suffix).collect()
}

#[test]
fn prefill_checkpoint_finishes_the_prompt_in_one_pass() {
    // The final chunk runs to the prompt end and exports the state at
    // (P - 1) // B * B = 12 from inside its forward, in one extra state
    // reserved for that pass. Without it the chunk stops at 12.
    for (prefill_checkpoint, first_pass) in [(true, 14), (false, 12)] {
        let mut core = VllmCore::new(args(prefill_checkpoint, false));
        let id = submit(&mut core, 1, (0..14).collect(), 4);
        step(&mut core);
        assert_eq!(core.state.requests[&id].num_computed_tokens, first_pass);
        if prefill_checkpoint {
            // Four token blocks, the working state and the checkpoint.
            assert_eq!(core.kv_manager.num_active_blocks(), 6);
        }
        // Either way the state at 12 is hittable from the next pass.
        let reader = submit(&mut core, 2, (0..20).collect(), 1);
        assert_eq!(reused(&step(&mut core), reader), 12);
        run_to_completion(&mut core);
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
    }
}

#[test]
fn junction_checkpoint_keeps_the_shared_prefix_state_for_siblings() {
    // The parent caches the shared 12 tokens as attention only; its state sits
    // at its own last full block, 28. A child's attention hit runs deeper than
    // its state hit, so with junctions its prefill stops at 12 and keeps the
    // state there for the next child.
    for (junction_checkpoints, first_pass, sibling_reuse) in [(true, 12, 12), (false, 28, 0)] {
        let mut core = VllmCore::new(args(true, junction_checkpoints));
        submit(&mut core, 1, tokens(0..12, 1000..1020), 2);
        run_to_completion(&mut core);
        let child = submit(&mut core, 2, tokens(0..12, 2000..2016), 2);
        assert_eq!(reused(&step(&mut core), child), 0);
        assert_eq!(core.state.requests[&child].num_computed_tokens, first_pass);
        let sibling = submit(&mut core, 3, tokens(0..12, 3000..3016), 2);
        assert_eq!(reused(&step(&mut core), sibling), sibling_reuse);
        run_to_completion(&mut core);
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
    }
}

#[test]
fn lookup_waits_one_pass_for_a_state_published_in_the_same_pass() {
    // vLLM's lookup sees a state cached by this pass, but cannot allocate a
    // request that hits it. Two identical prompts arrive together; the first
    // exports its checkpoint at 12, and with junction_checkpoints the second
    // waits one pass and resumes there instead of recomputing the prompt.
    let mut core = VllmCore::new(args(true, true));
    let first = submit(&mut core, 1, (0..14).collect(), 2);
    let second = submit(&mut core, 2, (0..14).collect(), 2);
    let pass = step(&mut core);
    assert_eq!(reused(&pass, first), 0);
    assert!(
        pass.admissions
            .iter()
            .all(|admission| admission.uuid != second)
    );
    assert_eq!(reused(&step(&mut core), second), 12);
    run_to_completion(&mut core);
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn disable_eagle_block_drop_keeps_the_last_matched_block_under_mtp() {
    // The previous prompt ends mid-block (14 = 3B + 2). Without the drop its
    // follow-up resumes at that prompt's last full block, 12; with the drop,
    // one block earlier.
    for (disable_eagle_block_drop, expected) in [(true, 12), (false, 8)] {
        let mut config = args(false, false);
        config.aic_nextn = Some(5);
        config.aic_nextn_accept_rates = Some("1,1,1,1,1".to_string());
        config.disable_eagle_block_drop = disable_eagle_block_drop;
        config.state_cache.as_mut().unwrap().num_speculative_states = Some(5);
        let mut core = VllmCore::new(config);
        submit(&mut core, 1, (0..14).collect(), 3);
        run_to_completion(&mut core);
        let follow_up = submit(&mut core, 2, (0..22).collect(), 1);
        assert_eq!(reused(&step(&mut core), follow_up), expected);
    }
}
