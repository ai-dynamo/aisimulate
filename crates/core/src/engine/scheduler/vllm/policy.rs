// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-specific policy for the shared vLLM/TRT-LLM scheduler core.
//!
//! The core stays backend-neutral and delegates these policy seams here:
//! request-length normalization, ordinary waiting admission, P/D destination
//! admission/reservation, preemption, and first-token timing. Prefix accounting
//! and no-evict headroom are implementation details of those hooks, not separate schedulers.

use crate::engine::common::protocols::{PrefillCost, SchedulingPolicy};
use crate::engine::kv_manager::{AllocationRequirement, DestinationReservationMode, G1Manager};
#[cfg(test)]
use crate::engine::kv_manager::{apply_mtp_prefix_recompute, apply_prefix_recompute};
use crate::engine::scheduler::vllm::request::RequestKvState;

pub(super) trait PolicySequence {
    fn len(&self) -> usize;
    fn max_output_tokens(&self) -> usize;
    fn generated_tokens(&self) -> usize;
    fn num_input_tokens(&self) -> usize;
    fn num_allocated_tokens(&self) -> usize;
    fn current_known_blocks(&self) -> usize;
    fn to_completion_blocks(&self) -> usize;
    fn prefill_cost(&self, kv_manager: &G1Manager) -> PrefillCost;
    fn admission_requirement(
        &self,
        kv_manager: &G1Manager,
        cost: &PrefillCost,
        reserved_blocks: usize,
    ) -> AllocationRequirement;
}

impl PolicySequence for RequestKvState {
    fn len(&self) -> usize {
        self.len()
    }

    fn max_output_tokens(&self) -> usize {
        self.max_output_tokens()
    }

    fn generated_tokens(&self) -> usize {
        self.generated_tokens()
    }

    fn num_input_tokens(&self) -> usize {
        self.num_input_tokens()
    }

    fn num_allocated_tokens(&self) -> usize {
        self.num_allocated_tokens()
    }

    fn current_known_blocks(&self) -> usize {
        self.current_known_blocks()
    }

    fn to_completion_blocks(&self) -> usize {
        self.to_completion_blocks()
    }

    fn prefill_cost(&self, kv_manager: &G1Manager) -> PrefillCost {
        kv_manager.get_native_prefill_cost(&self.sequence, &self.lease)
    }

    fn admission_requirement(
        &self,
        kv_manager: &G1Manager,
        cost: &PrefillCost,
        reserved_blocks: usize,
    ) -> AllocationRequirement {
        kv_manager.admission_requirement(&self.lease, self.len(), cost, reserved_blocks)
    }
}

#[derive(Debug)]
pub(super) enum AdmissionDecision {
    Admit { prefill_cost: PrefillCost },
    Wait,
    Reject,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct WaitingAdmissionConfig {
    pub(super) policy: SchedulingPolicy,
    pub(super) num_gpu_blocks: usize,
    pub(super) block_size: usize,
    pub(super) mtp_enabled: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DestinationAdmissionConfig {
    pub(super) policy: SchedulingPolicy,
    pub(super) num_gpu_blocks: usize,
    pub(super) block_size: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(super) enum DestinationAdmissionDecision {
    Reserve { mode: DestinationReservationMode },
    Wait,
}

pub(super) fn destination_capacity_error<S: PolicySequence>(
    policy: SchedulingPolicy,
    sequence: &S,
    num_gpu_blocks: usize,
) -> Option<&'static str> {
    let (exceeds, message) = match policy {
        SchedulingPolicy::Vllm => (
            sequence.current_known_blocks() > num_gpu_blocks,
            "destination prompt exceeds the KV pool capacity",
        ),
        SchedulingPolicy::TrtllmGuaranteedNoEvict => (
            sequence.to_completion_blocks() > num_gpu_blocks,
            "TRT-LLM destination request exceeds the to-completion KV pool capacity",
        ),
    };
    exceeds.then_some(message)
}

/// TRT-LLM reserves through completion, so admission uses the realizable output
/// budget. vLLM keeps its requested budget and enforces the limit during decode.
pub(super) fn cap_output_for_model_len(
    policy: SchedulingPolicy,
    prompt_len: usize,
    max_output_tokens: usize,
    max_model_len: Option<usize>,
) -> usize {
    match (policy, max_model_len) {
        (SchedulingPolicy::TrtllmGuaranteedNoEvict, Some(limit)) => {
            max_output_tokens.min(limit.saturating_sub(prompt_len))
        }
        _ => max_output_tokens,
    }
}

pub(super) fn should_reject_for_model_len<S: PolicySequence>(
    sequence: &S,
    max_model_len: Option<usize>,
) -> bool {
    max_model_len.is_some_and(|limit| sequence.num_input_tokens() >= limit)
}

/// Number of additional tokens the request may generate before reaching
/// either its requested output length or the model sequence-length limit.
pub(super) fn remaining_generation_tokens<S: PolicySequence>(
    sequence: &S,
    max_model_len: Option<usize>,
) -> usize {
    let requested_remaining = sequence
        .max_output_tokens()
        .saturating_sub(sequence.generated_tokens());
    let context_remaining = max_model_len
        .map(|limit| limit.saturating_sub(sequence.len()))
        .unwrap_or(usize::MAX);
    requested_remaining.min(context_remaining)
}

pub(super) fn generation_complete<S: PolicySequence>(
    sequence: &S,
    max_model_len: Option<usize>,
) -> bool {
    remaining_generation_tokens(sequence, max_model_len) == 0
}

pub(super) fn samples_first_token_from_prefill(policy: SchedulingPolicy) -> bool {
    policy == SchedulingPolicy::Vllm
}

/// Decide whether the FIFO head can enter the shared scheduler core.
///
/// vLLM reserves only the current known sequence. TRT-LLM
/// `GUARANTEED_NO_EVICT` reserves the request through its maximum completion
/// and accounts for the completion reservations of running requests and
/// decode destinations whose transferred prompt KV is not running yet.
#[cfg(test)]
pub(super) fn decide_waiting_admission<'a, S: PolicySequence + 'a>(
    config: WaitingAdmissionConfig,
    sequence: &S,
    is_fresh: bool,
    running: impl Iterator<Item = &'a S>,
    kv_manager: &G1Manager,
) -> AdmissionDecision {
    let raw_prefill_cost = sequence.prefill_cost(kv_manager);
    decide_waiting_admission_with_cost(
        config,
        sequence,
        is_fresh,
        running,
        kv_manager,
        0,
        0,
        0,
        0,
        raw_prefill_cost,
    )
}

/// Apply the scheduler capacity rule to a prefix lookup that has already run.
///
/// A framework-native host tier probes after the authoritative G1 lookup but
/// before G1 allocation. Passing that observed cost through this boundary
/// avoids a second lookup changing recency or hiding a lifecycle divergence.
/// Held and activated-waiting block counts describe completion headroom
/// already promised to decode destinations. Policy decides whether it counts.
#[allow(clippy::too_many_arguments)]
pub(super) fn decide_waiting_admission_with_cost<'a, S: PolicySequence + 'a>(
    config: WaitingAdmissionConfig,
    sequence: &S,
    is_fresh: bool,
    running: impl Iterator<Item = &'a S>,
    kv_manager: &G1Manager,
    reserved_request_blocks: usize,
    inflight_prefill_reserved_blocks: usize,
    held_completion_blocks: usize,
    activated_waiting_completion_blocks: usize,
    raw_prefill_cost: PrefillCost,
) -> AdmissionDecision {
    let WaitingAdmissionConfig {
        policy,
        num_gpu_blocks,
        block_size,
        mtp_enabled,
    } = config;

    if is_fresh {
        match policy {
            SchedulingPolicy::Vllm => {
                // Total worker KV remains a fallback one-time admission cap
                // when max_model_len is unset or larger than the KV pool.
                if sequence.current_known_blocks() > num_gpu_blocks {
                    return AdmissionDecision::Reject;
                }
            }
            SchedulingPolicy::TrtllmGuaranteedNoEvict => {
                if sequence.to_completion_blocks() > num_gpu_blocks {
                    return AdmissionDecision::Reject;
                }
            }
        }
    }

    let prefill_cost = kv_manager.resolve_prefill_cost(
        policy,
        sequence.len(),
        mtp_enabled,
        !generation_complete(sequence, None),
        raw_prefill_cost,
    );
    let handoff_reserved_blocks = handoff_completion_headroom(
        policy,
        held_completion_blocks,
        activated_waiting_completion_blocks,
    );
    let available = match policy {
        SchedulingPolicy::Vllm => num_gpu_blocks
            .saturating_sub(kv_manager.num_active_blocks())
            .saturating_sub(inflight_prefill_reserved_blocks),
        SchedulingPolicy::TrtllmGuaranteedNoEvict => {
            available_blocks(running, num_gpu_blocks, block_size, kv_manager)
                .saturating_sub(handoff_reserved_blocks)
        }
    };
    let needed = match policy {
        SchedulingPolicy::Vllm => {
            match sequence.admission_requirement(kv_manager, &prefill_cost, reserved_request_blocks)
            {
                AllocationRequirement::Blocks(count) => count,
                AllocationRequirement::Impossible => return AdmissionDecision::Reject,
            }
        }
        SchedulingPolicy::TrtllmGuaranteedNoEvict => {
            blocks_needed_to_finish(sequence, block_size, kv_manager, Some(&prefill_cost))
        }
    };

    if needed > available {
        AdmissionDecision::Wait
    } else {
        AdmissionDecision::Admit { prefill_cost }
    }
}

/// Select destination admission and physical prompt-reservation behavior at
/// one scheduler-policy boundary.
///
/// vLLM keeps its optimistic/preemptible destination path and may reuse a
/// resident decode prefix. TRT-LLM `GUARANTEED_NO_EVICT` generation-init
/// requests receive no prefix credit: the candidate needs its full
/// to-completion footprint, while prompt blocks and completion tails already
/// promised to other handoffs remain unavailable.
pub(super) fn decide_destination_admission<'a, S: PolicySequence + 'a>(
    config: DestinationAdmissionConfig,
    sequence: &S,
    running: impl Iterator<Item = &'a S>,
    kv_manager: &G1Manager,
    held_completion_blocks: usize,
    activated_waiting_completion_blocks: usize,
) -> DestinationAdmissionDecision {
    match config.policy {
        SchedulingPolicy::Vllm => DestinationAdmissionDecision::Reserve {
            mode: DestinationReservationMode::ReuseResidentPrefix,
        },
        SchedulingPolicy::TrtllmGuaranteedNoEvict => {
            let handoff_reserved_blocks = handoff_completion_headroom(
                config.policy,
                held_completion_blocks,
                activated_waiting_completion_blocks,
            );
            let available = available_blocks(
                running,
                config.num_gpu_blocks,
                config.block_size,
                kv_manager,
            )
            .saturating_sub(handoff_reserved_blocks);
            if sequence.to_completion_blocks() > available {
                DestinationAdmissionDecision::Wait
            } else {
                DestinationAdmissionDecision::Reserve {
                    mode: DestinationReservationMode::FreshOnly,
                }
            }
        }
    }
}

/// Completion tail already promised to non-running destination handoffs.
/// vLLM remains optimistic/preemptible; TRT-LLM no-evict admission must keep
/// both held and activated-waiting tails unavailable.
fn handoff_completion_headroom(
    policy: SchedulingPolicy,
    held_completion_blocks: usize,
    activated_waiting_completion_blocks: usize,
) -> usize {
    match policy {
        SchedulingPolicy::Vllm => 0,
        SchedulingPolicy::TrtllmGuaranteedNoEvict => {
            held_completion_blocks.saturating_add(activated_waiting_completion_blocks)
        }
    }
}

/// Blocks a request still needs to reserve to run to completion under the
/// TRT-LLM `GUARANTEED_NO_EVICT` policy.
///
/// ```text
/// needed = ceil((prompt_len + max_output_tokens) / block_size)
///          - blocks_already_held
///          - active_cached_prefix_blocks   (waiting candidates only)
/// ```
///
/// For a running request, the blocks it already holds are physical (counted in
/// the KV manager's active blocks), so only the remaining footprint is reserved.
/// For a waiting candidate, only the active cached prefix is discounted
/// (`active_cached_tokens`).
pub(super) fn blocks_needed_to_finish<S: PolicySequence>(
    sequence: &S,
    block_size: usize,
    kv_manager: &G1Manager,
    prefill_cost: Option<&PrefillCost>,
) -> usize {
    let full_blocks = sequence.to_completion_blocks();
    if sequence.num_allocated_tokens() == 0 {
        let reusable_blocks = prefill_cost
            .map(|cost| cost.active_cached_tokens)
            .unwrap_or_else(|| sequence.prefill_cost(kv_manager).active_cached_tokens)
            / block_size;
        full_blocks.saturating_sub(reusable_blocks)
    } else {
        let allocated_blocks = sequence.num_allocated_tokens().div_ceil(block_size);
        full_blocks.saturating_sub(allocated_blocks)
    }
}

/// Free blocks remaining after reserving every running request's to-completion
/// footprint. A waiting candidate may be admitted iff its
/// [`blocks_needed_to_finish`] is `<=` this value.
///
/// `running` yields the active sequence of each currently-running request;
/// `num_gpu_blocks` is the KV pool size and `kv_manager` supplies the count of
/// physically allocated blocks.
fn available_blocks<'a, S: PolicySequence + 'a>(
    running: impl Iterator<Item = &'a S>,
    num_gpu_blocks: usize,
    block_size: usize,
    kv_manager: &G1Manager,
) -> usize {
    let reserved: usize = running
        .map(|sequence| blocks_needed_to_finish(sequence, block_size, kv_manager, None))
        .sum();
    let free = num_gpu_blocks.saturating_sub(kv_manager.num_active_blocks());
    free.saturating_sub(reserved)
}

pub(super) fn allows_preemption(policy: SchedulingPolicy) -> bool {
    policy == SchedulingPolicy::Vllm
}

/// TRT-LLM enqueue normalization: a no-evict request's `prompt + output` can
/// reserve at most the whole KV pool. Returns `max_output_tokens` clamped to the
/// room left after the prompt, or `None` if the prompt alone leaves no decode
/// room (the request can never run and should be rejected).
pub(super) fn normalize_max_output_tokens(
    policy: SchedulingPolicy,
    prompt_len: usize,
    max_output_tokens: usize,
    num_gpu_blocks: usize,
    block_size: usize,
) -> Option<usize> {
    if policy == SchedulingPolicy::Vllm {
        return Some(max_output_tokens);
    }
    let capacity_tokens = num_gpu_blocks.saturating_mul(block_size);
    if prompt_len >= capacity_tokens {
        return None;
    }
    Some(max_output_tokens.min(capacity_tokens - prompt_len))
}

/// Fail loudly when the no-evict invariant is violated.
///
/// Under `GUARANTEED_NO_EVICT` the capacity gate reserves blocks for every
/// admitted request up front, so a preemption should never be required.
/// Reaching the preemption path means the reservation under-counted physical
/// KV demand (e.g. a reusable prefix block was evicted before the request
/// claimed it). A silent preempt would still produce output but no longer
/// represent TRT-LLM, degrading timing fidelity undetectably — so debug builds
/// assert and release builds log and decline to preempt.
pub(super) fn report_no_preemption_violation() {
    debug_assert!(
        false,
        "no-evict invariant violated: trtllm GUARANTEED_NO_EVICT required preemption"
    );
    tracing::error!(
        "trtllm GUARANTEED_NO_EVICT required preemption; reservation under-counted physical KV demand"
    );
}

#[cfg(test)]
mod tests;
