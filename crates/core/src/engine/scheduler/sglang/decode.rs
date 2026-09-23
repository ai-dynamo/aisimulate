// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use uuid::Uuid;

use crate::engine::common::protocols::OutputSignal;
use crate::engine::common::speculative::SpeculativeDecodeSampler;
use crate::engine::common::utils::compute_prefill_handoff_delay_ms;
use crate::engine::kv_manager::SglangKvManager;
use crate::engine::kv_manager::sglang_backend::DecodeTokenReservation;
use crate::engine::{
    DecodeAcceptance, PressureEvent, PressureKind, PressureState, modeled_duration_ms,
};

use super::config::{SglangConfig, floor_to_block};
use super::request::SglangRequest;

/// SGLang `SGLANG_RETRACT_DECODE_STEPS`: decode steps worth of KV reserved per surviving request
/// when `retract_decode` re-estimates `new_token_ratio`.
pub(super) const RETRACT_DECODE_STEPS: usize = 20;

#[derive(Default)]
pub(super) struct DecodeResult {
    pub(super) requests: Vec<SglangRequest>,
    pub(super) completed_requests: Vec<SglangRequest>,
    pub(super) output_signals: Vec<OutputSignal>,
    pub(super) pressure_events: Vec<PressureEvent>,
    /// SGLang `ScheduleBatch.retract_decode` re-estimate of `new_token_ratio`, computed from the
    /// surviving requests at the retraction point (before this pass's forward), when a retraction
    /// happened.
    pub(super) new_token_ratio_estimate: Option<f64>,
    pub(super) end_ms: f64,
    pub(super) decode_acceptance: DecodeAcceptance,
    /// Prefix-cache commits the host loop applies when it observes this forward:
    /// `(request, sequence length materialized)`. Empty without a host loop, where
    /// the commits were applied directly.
    pub(super) cache_commits: Vec<(Uuid, usize)>,
}

/// What a scheduler pass asks the step simulation to do with `running`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum StepKind {
    /// A decode forward: memory check with retraction, speculative bursts, modeled duration.
    Decode,
    /// Bookkeeping for requests that were just prefilled in this pass: the prefill forward already
    /// produced exactly one token per request, so no speculative burst and no extra time.
    PrefillFirstToken,
}

/// `min(1, (decoded + RETRACT_DECODE_STEPS * n) / (max_new + 1))` over the surviving batch, as in
/// SGLang `ScheduleBatch.retract_decode`.
fn retraction_ratio_estimate(running: &[SglangRequest]) -> f64 {
    let total_decoded: usize = running.iter().map(|req| req.output_len()).sum();
    let total_max_new: usize = running.iter().map(|req| req.max_output_tokens).sum();
    let estimate = (total_decoded as f64 + RETRACT_DECODE_STEPS as f64 * running.len() as f64)
        / (total_max_new as f64 + 1.0);
    estimate.min(1.0)
}

fn decode_page_growth_needed(
    running: &[SglangRequest],
    block_size: usize,
    max_burst: usize,
) -> usize {
    running
        .iter()
        .map(|req| {
            let target = if req.pending_terminal {
                // `prepare_for_decode` still allocates the finished row's slot: the
                // scheduler has not observed the finish, so `filter_batch` kept it.
                super::config::ceil_to_block(req.current_sequence_len(), block_size)
            } else {
                let burst = max_burst.min(req.remaining_output_tokens());
                super::config::ceil_to_block(req.current_sequence_len() + burst, block_size)
            };
            target.saturating_sub(req.allocated_tokens)
        })
        .sum()
}

/// Allocate the KV slot a ghost decode row receives for the token that finished
/// it. `alloc_for_decode` commits the slot like any other row's, so the request
/// caches it with the rest of its KV when the scheduler observes the finish.
fn allocate_ghost_slot(
    req: &mut SglangRequest,
    kv_manager: &mut SglangKvManager,
    reservation: &mut DecodeTokenReservation,
    block_size: usize,
) {
    if req.materialized_tokens >= req.current_sequence_len() {
        return;
    }
    // The finishing token's KV is computed by this row, so its page can be cached.
    req.kv_lease
        .ensure_page_hashes(&req.sequence_tokens, block_size);
    let crossing_page_boundary = req.materialized_tokens + 1 > req.allocated_tokens;
    kv_manager.extend_decode(&mut req.kv_lease, reservation);
    if crossing_page_boundary {
        req.allocated_tokens += block_size;
    }
    req.materialized_tokens += 1;
    req.debug_assert_invariants(block_size);
}

/// Make the prefix a forward materialized visible to admission. Under the host
/// loop the scheduler learns it only when it observes that forward's result
/// (`maybe_cache_unfinished_req`), so the commit is recorded, not applied.
fn commit_materialized_prefix(
    req: &mut SglangRequest,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    deferred: &mut Vec<(Uuid, usize)>,
) {
    if config.host_loop {
        deferred.push((req.uuid, req.materialized_tokens));
    } else {
        cache_materialized_prefix(req, kv_manager, config);
    }
}

fn decode_capacity_state(
    running: &[SglangRequest],
    kv_manager: &SglangKvManager,
    config: &SglangConfig,
    max_burst: usize,
) -> (usize, usize, usize) {
    let actual_available =
        kv_manager.cache().available_tokens() + kv_manager.cache().evictable_size;
    // Full partial pages are already owned by PagePool and excluded from
    // `actual_available`; subtracting their slack again would double-charge it.
    let logical_available = actual_available;
    let page_growth_needed = decode_page_growth_needed(running, config.block_size, max_burst);

    (actual_available, logical_available, page_growth_needed)
}

pub(super) fn cache_materialized_prefix(
    req: &mut SglangRequest,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
) {
    cache_prefix_through(req, kv_manager, config, req.materialized_tokens);
}

/// Cache the page-aligned prefix through `tokens`, bounded by what the request
/// still materializes: a retracted request has nothing to publish.
pub(super) fn cache_prefix_through(
    req: &mut SglangRequest,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    tokens: usize,
) {
    let aligned_tokens = floor_to_block(tokens.min(req.materialized_tokens), config.block_size);
    if aligned_tokens == 0 || aligned_tokens <= req.cached_tokens() {
        return;
    }

    if !req.kv_lease.is_active() {
        panic!(
            "cache_materialized_prefix: request {} has aligned_tokens={aligned_tokens} but no active KV lease",
            req.uuid
        );
    }

    let sequence = &req.sequence_tokens[..aligned_tokens];
    kv_manager.extend_cached_prefix(sequence, &mut req.kv_lease);
    req.debug_assert_invariants(config.block_size);
}

#[cfg(test)]
pub(super) fn check_decode_mem(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
) -> Vec<SglangRequest> {
    check_decode_mem_with_pressure_events(running, kv_manager, config, 1, 0.0).0
}

#[cfg(test)]
pub(super) fn check_decode_mem_with_pressure_events(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    max_burst: usize,
    at_ms: f64,
) -> (Vec<SglangRequest>, Vec<PressureEvent>) {
    let mut pressure_events = Vec::new();
    let requests = check_decode_mem_for_burst(
        running,
        kv_manager,
        config,
        max_burst,
        at_ms,
        &mut pressure_events,
    );
    (requests, pressure_events)
}

fn check_decode_mem_for_burst(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    max_burst: usize,
    at_ms: f64,
    pressure_events: &mut Vec<PressureEvent>,
) -> Vec<SglangRequest> {
    let mut retracted = Vec::new();

    loop {
        let (_actual_available, logical_available, page_growth_needed) =
            decode_capacity_state(running, kv_manager, config, max_burst);
        if logical_available >= page_growth_needed {
            break;
        }
        if running.len() <= 1 {
            break;
        }

        let Some((idx, _)) = running
            .iter()
            .enumerate()
            .filter(|(_, req)| !req.pending_terminal)
            .min_by_key(|(_, req)| req.output_len())
        else {
            break;
        };

        let request = &running[idx];
        let request_id = request.uuid;
        let state_before = pressure_state(running.len(), kv_manager, config.block_size);
        let request_active_blocks_before = request.allocated_tokens.div_ceil(config.block_size);
        let logical_available_blocks_before = logical_available / config.block_size;
        let required_blocks_before = page_growth_needed.div_ceil(config.block_size);
        let mut req = running.remove(idx);
        kv_manager.retract_in_place(&mut req.kv_lease);
        req.reset_for_retract();
        req.debug_assert_invariants(config.block_size);
        pressure_events.push(PressureEvent {
            at_ms,
            kind: PressureKind::SglangRetraction,
            request_id,
            state_before,
            state_after: pressure_state(running.len(), kv_manager, config.block_size),
            request_active_blocks_before,
            logical_available_blocks_before: Some(logical_available_blocks_before),
            required_blocks_before: Some(required_blocks_before),
        });
        retracted.push(req);
    }

    let available = kv_manager.cache().available_tokens();
    let page_growth_needed = decode_page_growth_needed(running, config.block_size, max_burst);
    if available < page_growth_needed {
        kv_manager.evict(page_growth_needed - available);
    }

    if !retracted.is_empty() {
        tracing::warn!(
            num_retracted = retracted.len(),
            remaining = running.len(),
            "SGLang decode retract requests because KV pool is full"
        );
    }

    retracted
}

fn pressure_state(
    running_requests: usize,
    kv_manager: &SglangKvManager,
    block_size: usize,
) -> PressureState {
    let active_tokens = kv_manager.cache().total_tokens() - kv_manager.cache().available_tokens();
    PressureState {
        running_requests,
        waiting_requests: None,
        active_blocks: active_tokens.div_ceil(block_size),
    }
}

#[cfg(test)]
pub(super) fn simulate_decode_step(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    current_time_ms: f64,
    apply_speedup: bool,
) -> DecodeResult {
    let mut result = simulate_decode_step_with_sampler(
        running,
        kv_manager,
        config,
        None,
        current_time_ms,
        apply_speedup,
    )
    .expect("SGLang decode simulation failed");
    for mut request in result.completed_requests.drain(..) {
        cleanup_completed_request(&mut request, kv_manager, config.block_size);
    }
    result
}

/// Bookkeeping for the first token that a prefill forward produced for each freshly prefilled
/// request. SGLang finishes a request whose first token is also its last right after the prefill
/// batch (`check_finished` in `process_batch_result_prefill`) without ever giving that token a KV
/// slot; the others get their slot at the next decode step (`prepare_for_decode`), where the
/// decode memory check and retraction live. This is a pure prefill forward, so it never retracts
/// and takes no decode time. When the first-output slots do not fit even after evicting cached
/// pages, the continuing requests keep their first token until the next decode step (which is
/// where SGLang would have to retract for them).
fn prefill_first_tokens(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    current_time_ms: f64,
    mut completed_requests: Vec<SglangRequest>,
    mut output_signals: Vec<OutputSignal>,
) -> DecodeResult {
    let mut completed_indices = Vec::new();
    let mut cache_commits = Vec::new();

    // Requests that finish with their first token need no KV slot.
    for (idx, req) in running.iter_mut().enumerate() {
        if req.remaining_output_tokens() != 1 {
            continue;
        }
        let token_id = req.next_output_token();
        req.append_final_output_token(token_id);
        req.debug_assert_invariants(config.block_size);
        output_signals.push(OutputSignal {
            uuid: req.uuid,
            token_id: Some(token_id),
            completed: true,
            rejected: false,
            cached_tokens: None,
            handoff_delay_ms: compute_prefill_handoff_delay_ms(
                config.worker_type,
                true,
                req.prompt_len(),
                config.kv_transfer_bandwidth,
                config.kv_transfer_bytes_per_token,
            ),
        });
        if config.host_loop {
            // The scheduler observes this completion one iteration later; until then
            // the request stays a member of the next batch.
            req.pending_terminal = true;
        } else {
            completed_indices.push(idx);
        }
    }
    let mut newly_completed = Vec::with_capacity(completed_indices.len());
    for &idx in completed_indices.iter().rev() {
        newly_completed.push(running.remove(idx));
    }
    newly_completed.reverse();
    completed_requests.extend(newly_completed);

    // The rest need a slot for the first output token. Make room by evicting cached pages only.
    let needed = decode_page_growth_needed(running, config.block_size, 1);
    let available = kv_manager.cache().available_tokens();
    if available < needed {
        kv_manager.evict(needed - available);
    }
    let reserved_pages = needed / config.block_size;
    let Some(mut reservation) = kv_manager.reserve_decode_pages(reserved_pages) else {
        return DecodeResult {
            completed_requests,
            output_signals,
            end_ms: current_time_ms,
            ..DecodeResult::default()
        };
    };
    for req in running.iter_mut() {
        if req.pending_terminal {
            continue;
        }
        let crossing_page_boundary = req.current_sequence_len() + 1 > req.allocated_tokens;
        kv_manager.extend_decode(&mut req.kv_lease, &mut reservation);
        if crossing_page_boundary {
            req.allocated_tokens += config.block_size;
        }
        let token_id = req.next_output_token();
        req.append_output_token(token_id, config.block_size);
        commit_materialized_prefix(req, kv_manager, config, &mut cache_commits);
        req.debug_assert_invariants(config.block_size);
        output_signals.push(OutputSignal {
            uuid: req.uuid,
            token_id: Some(token_id),
            completed: false,
            rejected: false,
            cached_tokens: None,
            handoff_delay_ms: compute_prefill_handoff_delay_ms(
                config.worker_type,
                false,
                req.prompt_len(),
                config.kv_transfer_bandwidth,
                config.kv_transfer_bytes_per_token,
            ),
        });
    }
    debug_assert!(reservation.len() <= reserved_pages);
    kv_manager.release_decode_reservation(reservation);

    DecodeResult {
        completed_requests,
        output_signals,
        end_ms: current_time_ms,
        cache_commits,
        ..DecodeResult::default()
    }
}

pub(super) fn cleanup_completed_request(
    request: &mut SglangRequest,
    kv_manager: &mut SglangKvManager,
    block_size: usize,
) {
    // `release_kv_cache` caches the committed KV, page aligned: everything the request
    // materialized, a ghost decode row's slot included. Only the unaligned tail is freed.
    let tokens_to_cache = floor_to_block(request.materialized_tokens, block_size);
    if !request.kv_lease.is_active() {
        return;
    }
    let lease = std::mem::take(&mut request.kv_lease);
    kv_manager.finish(request.sequence_prefix(tokens_to_cache), lease);
}

pub(super) fn simulate_decode_step_with_sampler(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    sampler: Option<&mut SpeculativeDecodeSampler>,
    current_time_ms: f64,
    apply_speedup: bool,
) -> anyhow::Result<DecodeResult> {
    simulate_step(
        running,
        kv_manager,
        config,
        sampler,
        current_time_ms,
        apply_speedup,
        StepKind::Decode,
    )
}

/// Record the first token that the prefill forward produced for each freshly prefilled request.
/// Shares the KV bookkeeping with the decode step but takes no speculative burst and no time.
pub(super) fn simulate_prefill_first_tokens(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    current_time_ms: f64,
) -> anyhow::Result<DecodeResult> {
    simulate_step(
        running,
        kv_manager,
        config,
        None,
        current_time_ms,
        false,
        StepKind::PrefillFirstToken,
    )
}

fn simulate_step(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    mut sampler: Option<&mut SpeculativeDecodeSampler>,
    current_time_ms: f64,
    apply_speedup: bool,
    kind: StepKind,
) -> anyhow::Result<DecodeResult> {
    if running.is_empty() {
        return Ok(DecodeResult {
            end_ms: current_time_ms,
            ..DecodeResult::default()
        });
    }

    // Terminal requests have no decode work and otherwise remain in `running` forever.
    // Under the host loop a finished request is a ghost batch member until the next
    // iteration observes its result; it was already signaled when it finished.
    let already_completed_indices = running
        .iter()
        .enumerate()
        .filter_map(|(idx, req)| {
            (req.remaining_output_tokens() == 0 && !req.pending_terminal).then_some(idx)
        })
        .collect::<Vec<_>>();
    let mut output_signals = already_completed_indices
        .iter()
        .map(|&idx| {
            let req = &running[idx];
            OutputSignal {
                uuid: req.uuid,
                token_id: None,
                completed: true,
                rejected: false,
                cached_tokens: None,
                handoff_delay_ms: compute_prefill_handoff_delay_ms(
                    config.worker_type,
                    true,
                    req.prompt_len(),
                    config.kv_transfer_bandwidth,
                    config.kv_transfer_bytes_per_token,
                ),
            }
        })
        .collect::<Vec<_>>();
    let mut completed_requests = Vec::new();
    if config.host_loop {
        // The scheduler observes these completions with this forward's result; until
        // then the rows stay batch members like any other finished request.
        for &idx in &already_completed_indices {
            running[idx].pending_terminal = true;
        }
    } else {
        completed_requests.extend(
            already_completed_indices
                .iter()
                .rev()
                .map(|&idx| running.remove(idx)),
        );
        completed_requests.reverse();
    }

    if running.is_empty() {
        return Ok(DecodeResult {
            completed_requests,
            output_signals,
            end_ms: current_time_ms,
            ..DecodeResult::default()
        });
    }

    if kind == StepKind::PrefillFirstToken {
        return Ok(prefill_first_tokens(
            running,
            kv_manager,
            config,
            current_time_ms,
            completed_requests,
            output_signals,
        ));
    }

    let max_burst = if config.worker_type == crate::engine::common::protocols::WorkerType::Prefill {
        1
    } else {
        config.speculative_max_tokens.unwrap_or(1)
    };
    let mut pressure_events = Vec::new();
    let retracted = check_decode_mem_for_burst(
        running,
        kv_manager,
        config,
        max_burst,
        current_time_ms,
        &mut pressure_events,
    );
    // SGLang re-estimates the ratio in `retract_decode`, i.e. from the survivors before the
    // forward that follows, not from their state after this pass's tokens were appended.
    let new_token_ratio_estimate =
        (!retracted.is_empty()).then(|| retraction_ratio_estimate(running));
    if running.is_empty() {
        return Ok(DecodeResult {
            completed_requests,
            output_signals,
            requests: retracted,
            pressure_events,
            new_token_ratio_estimate,
            end_ms: current_time_ms,
            ..DecodeResult::default()
        });
    }

    let total_context: usize = running
        .iter()
        .map(SglangRequest::current_sequence_len)
        .sum();
    let avg_context = total_context / running.len();
    let active_kv_tokens = total_context;
    let decode_time = config.perf_model.predict_decode_time(
        running.len(),
        active_kv_tokens,
        avg_context,
        config.total_kv_tokens,
    )?;
    let effective_ratio = config.speedup_ratio * config.decode_speedup_ratio;
    let speedup_ratio = if apply_speedup { effective_ratio } else { 0.0 };
    let modeled_ms = modeled_duration_ms(decode_time, speedup_ratio)?;
    let total_time = Duration::from_secs_f64(modeled_ms / 1_000.0);

    let reserved_page_tokens = decode_page_growth_needed(running, config.block_size, max_burst);
    let reserved_pages = reserved_page_tokens / config.block_size;
    let Some(mut reservation) = kv_manager.reserve_decode_pages(reserved_pages) else {
        tracing::warn!(
            reserved_pages,
            "Failed to reserve speculative decode pages after capacity preflight"
        );
        return Ok(DecodeResult {
            completed_requests,
            output_signals,
            requests: retracted,
            pressure_events,
            new_token_ratio_estimate,
            end_ms: current_time_ms,
            ..DecodeResult::default()
        });
    };

    output_signals.reserve(running.len());
    let mut completed_indices = Vec::new();
    let mut cache_commits = Vec::new();
    let mut decode_acceptance = DecodeAcceptance::default();

    for (idx, req) in running.iter_mut().enumerate() {
        if req.pending_terminal {
            allocate_ghost_slot(req, kv_manager, &mut reservation, config.block_size);
            continue;
        }
        let remaining = req.remaining_output_tokens();
        let accepted =
            if config.worker_type == crate::engine::common::protocols::WorkerType::Prefill {
                remaining.min(1)
            } else if remaining == 0 {
                0
            } else if let Some(sampler) = sampler.as_deref_mut() {
                sampler.sample_accepted_tokens()
            } else {
                remaining.min(1)
            };
        if accepted > 0
            && config.worker_type != crate::engine::common::protocols::WorkerType::Prefill
        {
            decode_acceptance.accepted_tokens += accepted;
            decode_acceptance.forwards += 1;
        }
        let burst = accepted.min(remaining);
        for _ in 0..burst {
            let crossing_page_boundary = req.current_sequence_len() + 1 > req.allocated_tokens;
            kv_manager.extend_decode(&mut req.kv_lease, &mut reservation);
            if crossing_page_boundary {
                req.allocated_tokens += config.block_size;
            }
            let token_id = req.next_output_token();
            req.append_output_token(token_id, config.block_size);
            req.debug_assert_invariants(config.block_size);

            let is_complete = req.output_len() >= req.max_output_tokens;
            output_signals.push(OutputSignal {
                uuid: req.uuid,
                token_id: Some(token_id),
                completed: is_complete,
                rejected: false,
                cached_tokens: None,
                handoff_delay_ms: compute_prefill_handoff_delay_ms(
                    config.worker_type,
                    is_complete,
                    req.prompt_len(),
                    config.kv_transfer_bandwidth,
                    config.kv_transfer_bytes_per_token,
                ),
            });

            if is_complete {
                if config.host_loop {
                    req.pending_terminal = true;
                } else {
                    completed_indices.push(idx);
                }
                break;
            }

            commit_materialized_prefix(req, kv_manager, config, &mut cache_commits);
            req.debug_assert_invariants(config.block_size);
        }
    }

    debug_assert!(reservation.len() <= reserved_pages);
    kv_manager.release_decode_reservation(reservation);

    let mut newly_completed_requests = Vec::with_capacity(completed_indices.len());
    for &idx in completed_indices.iter().rev() {
        newly_completed_requests.push(running.remove(idx));
    }
    newly_completed_requests.reverse();
    completed_requests.extend(newly_completed_requests);

    Ok(DecodeResult {
        requests: retracted,
        completed_requests,
        output_signals,
        pressure_events,
        new_token_ratio_estimate,
        end_ms: current_time_ms + total_time.as_secs_f64() * 1000.0,
        decode_acceptance,
        cache_commits,
    })
}
