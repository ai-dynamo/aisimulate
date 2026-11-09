// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
// SPDX-License-Identifier: Apache-2.0
// Modified behavioral adaptation of SGLang v0.5.22; see THIRD_PARTY_NOTICES.md.
// Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d

use std::cmp::Reverse;
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
/// when `retract_decode` re-estimates `new_token_ratio`, and the per-request token reserve that
/// `release_req` evicts cached pages toward after each retraction.
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
    /// When the step's unpriced work ends: the step's start time. The core
    /// adds the decode forward's duration from [`Self::decode`].
    pub(super) end_ms: f64,
    /// The decode batch the step's forward computes, when it has one.
    pub(super) decode: Option<DecodeShape>,
    pub(super) decode_acceptance: DecodeAcceptance,
    /// Prefix-cache commits the host loop applies when it observes this forward:
    /// `(request, sequence length materialized)`. Empty without a host loop, where
    /// the commits were applied directly.
    pub(super) cache_commits: Vec<(Uuid, usize)>,
}

/// The decode batch of a step, as the timing provider prices it.
#[derive(Debug, Default)]
pub(super) struct DecodeShape {
    /// Batch rows, including finished rows still in the batch.
    pub(super) batch_size: usize,
    /// Sum of the rows' sequence lengths, each including its current token.
    pub(super) active_kv_tokens: usize,
    pub(super) mean_context_length: usize,
    /// Each row's sequence length.
    pub(super) lengths: Vec<usize>,
    /// Whether the batch ran. A step whose page reservation failed after the
    /// capacity preflight asks the provider but charges no time; a provider
    /// that prices only whole iterations is not asked.
    pub(super) charged: bool,
}

/// Decode time of `shape` priced by the scheduler's provider, with the decode
/// speedup applied when `apply_speedup` is set.
pub(super) fn price_decode_shape(
    shape: &DecodeShape,
    config: &SglangConfig,
    apply_speedup: bool,
) -> anyhow::Result<Duration> {
    if !shape.charged && !config.perf_model.prices_phases() {
        // The step computed no forward, and the provider has no decode phase.
        return Ok(Duration::ZERO);
    }
    let decode_time = config.perf_model.predict_decode_time(
        shape.batch_size,
        shape.active_kv_tokens,
        shape.mean_context_length,
        config.total_kv_tokens,
    )?;
    let effective_ratio = config.speedup_ratio * config.decode_speedup_ratio;
    let speedup_ratio = if apply_speedup { effective_ratio } else { 0.0 };
    let modeled_ms = modeled_duration_ms(decode_time, speedup_ratio)?;
    Ok(if shape.charged {
        Duration::from_secs_f64(modeled_ms / 1_000.0)
    } else {
        Duration::ZERO
    })
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

fn decode_page_growth_needed<'a>(
    running: impl Iterator<Item = &'a SglangRequest>,
    config: &SglangConfig,
    max_burst: usize,
) -> usize {
    running
        .map(|req| {
            let committed = if req.pending_terminal && config.speculative_max_tokens.is_some() {
                req.materialized_tokens
            } else {
                // A sampled bonus has no KV. A length stop inside accepted drafts does.
                req.current_sequence_len()
                    + max_burst
                        .saturating_sub(1)
                        .min(req.remaining_output_tokens())
            };
            let target = super::config::ceil_to_block(committed, config.block_size);
            target.saturating_sub(req.allocated_tokens)
        })
        .sum()
}

/// Materialize only tokens consumed by a forward, independently of sampled output.
fn materialize_through(
    req: &mut SglangRequest,
    kv_manager: &mut SglangKvManager,
    reservation: &mut DecodeTokenReservation,
    block_size: usize,
    tokens: usize,
) {
    debug_assert!(tokens <= req.current_sequence_len());
    req.kv_lease
        .ensure_page_hashes(&req.sequence_tokens, block_size);
    while req.materialized_tokens < tokens {
        let crossing_page_boundary = req.materialized_tokens + 1 > req.allocated_tokens;
        kv_manager.extend_decode(&mut req.kv_lease, reservation);
        if crossing_page_boundary {
            req.allocated_tokens += block_size;
        }
        req.materialized_tokens += 1;
    }
    req.debug_assert_invariants(block_size);
}

/// Make the prefix a forward materialized visible to admission. Under the host
/// loop the scheduler learns it only when it observes that forward's result
/// (`maybe_cache_unfinished_req`), so the commit is recorded, not applied. A
/// request's cached prefix only grows until then, so a commit that adds no page
/// now adds none when observed and is not recorded.
fn commit_materialized_prefix(
    req: &mut SglangRequest,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    deferred: &mut Vec<(Uuid, usize)>,
) {
    if config.host_loop {
        if uncached_aligned_prefix(req, config, req.materialized_tokens).is_some()
            || kv_manager.needs_host_insert(&req.kv_lease, req.materialized_tokens)
        {
            deferred.push((req.uuid, req.materialized_tokens));
        }
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
    let page_growth_needed = decode_page_growth_needed(running.iter(), config, max_burst);

    (actual_available, logical_available, page_growth_needed)
}

pub(super) fn decode_memory_fits(
    running: &[SglangRequest],
    kv_manager: &SglangKvManager,
    config: &SglangConfig,
) -> bool {
    let (_, available, needed) = decode_capacity_state(running, kv_manager, config, 1);
    available >= needed
}

pub(super) fn cache_materialized_prefix(
    req: &mut SglangRequest,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
) {
    cache_prefix_through_ready(
        req,
        kv_manager,
        config,
        req.materialized_tokens,
        !config.host_loop,
    );
}

/// Cache the page-aligned prefix through `tokens`, bounded by what the request
/// still materializes: a retracted request has nothing to publish.
pub(super) fn cache_prefix_through(
    req: &mut SglangRequest,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    tokens: usize,
) {
    cache_prefix_through_ready(req, kv_manager, config, tokens, true);
}

fn cache_prefix_through_ready(
    req: &mut SglangRequest,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    tokens: usize,
    backup_ready: bool,
) {
    let Some(aligned_tokens) = uncached_aligned_prefix(req, config, tokens) else {
        if backup_ready {
            kv_manager.backup_cached_prefix(&mut req.kv_lease, tokens);
        }
        return;
    };

    if !req.kv_lease.is_active() {
        panic!(
            "cache_materialized_prefix: request {} has aligned_tokens={aligned_tokens} but no active KV lease",
            req.uuid
        );
    }

    let sequence = &req.sequence_tokens[..aligned_tokens];
    kv_manager.extend_cached_prefix(sequence, &mut req.kv_lease);
    if backup_ready {
        kv_manager.backup_cached_prefix(&mut req.kv_lease, aligned_tokens);
    }
    req.debug_assert_invariants(config.block_size);
}

/// The page-aligned length through `tokens` to cache, when it extends the
/// request's cached prefix.
fn uncached_aligned_prefix(
    req: &SglangRequest,
    config: &SglangConfig,
    tokens: usize,
) -> Option<usize> {
    if !config.enable_prefix_caching {
        return None;
    }
    let aligned_tokens = floor_to_block(tokens.min(req.materialized_tokens), config.block_size);
    (aligned_tokens > req.cached_tokens()).then_some(aligned_tokens)
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

        // `retract_decode` stably sorts the batch once, longest output first and then shortest
        // prompt first, retracts from the end, and keeps the survivors in that order. Ghost rows
        // are sorted with the batch but never retracted.
        if retracted.is_empty() && running.iter().any(|req| !req.pending_terminal) {
            running.sort_by_key(|req| (Reverse(req.output_len()), req.prompt_len()));
        }
        let Some(idx) = running.iter().rposition(|req| !req.pending_terminal) else {
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
        // `release_req` evicts cached pages toward a reserve for the remaining requests. Only the
        // next step's demand decides whether to retract another request.
        let reserve = running.len() * RETRACT_DECODE_STEPS;
        let available = kv_manager.cache().available_tokens();
        if available < reserve {
            kv_manager.evict(reserve - available);
        }
        retracted.push(req);
    }

    let available = kv_manager.cache().available_tokens();
    let page_growth_needed = decode_page_growth_needed(running.iter(), config, max_burst);
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
    let mut result =
        simulate_decode_step_with_sampler(running, kv_manager, config, None, current_time_ms)
            .expect("SGLang decode simulation failed");
    if let Some(shape) = &result.decode {
        let total_time =
            price_decode_shape(shape, config, apply_speedup).expect("SGLang decode pricing failed");
        result.end_ms += total_time.as_secs_f64() * 1000.0;
    }
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
/// and takes no decode time or output-token KV slot.
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

    // Prefill computes the prompt's KV and samples one output without computing its KV.
    for (idx, req) in running.iter_mut().enumerate() {
        commit_materialized_prefix(req, kv_manager, config, &mut cache_commits);
        if req.pending_terminal {
            continue;
        }
        let token_id = req.next_output_token();
        req.append_output_token(token_id, config.block_size);
        req.debug_assert_invariants(config.block_size);
        let completed = req.remaining_output_tokens() == 0;
        output_signals.push(OutputSignal {
            uuid: req.uuid,
            token_id: Some(token_id),
            completed,
            rejected: false,
            cached_tokens: None,
            handoff_delay_ms: compute_prefill_handoff_delay_ms(
                config.worker_type,
                completed,
                req.prompt_len(),
                config.kv_transfer_bandwidth,
                config.kv_transfer_bytes_per_token,
            ),
        });
        if completed {
            if config.host_loop {
                req.pending_terminal = true;
            } else {
                completed_indices.push(idx);
            }
        }
    }
    let mut newly_completed = Vec::with_capacity(completed_indices.len());
    for &idx in completed_indices.iter().rev() {
        newly_completed.push(running.remove(idx));
    }
    newly_completed.reverse();

    completed_requests.extend(newly_completed);

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
    let mut lease = std::mem::take(&mut request.kv_lease);
    lease.set_prompt_tokens(request.prompt_len());
    kv_manager.finish(request.sequence_prefix(tokens_to_cache), lease);
}

pub(super) fn simulate_decode_step_with_sampler(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    sampler: Option<&mut SpeculativeDecodeSampler>,
    current_time_ms: f64,
) -> anyhow::Result<DecodeResult> {
    simulate_step(
        running,
        kv_manager,
        config,
        sampler,
        current_time_ms,
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
        StepKind::PrefillFirstToken,
    )
}

fn simulate_step(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    mut sampler: Option<&mut SpeculativeDecodeSampler>,
    current_time_ms: f64,
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

    let lengths: Vec<usize> = running
        .iter()
        .map(SglangRequest::current_sequence_len)
        .collect();
    let total_context: usize = lengths.iter().sum();
    let mut decode = DecodeShape {
        batch_size: running.len(),
        active_kv_tokens: total_context,
        mean_context_length: total_context / running.len(),
        lengths,
        charged: false,
    };

    let reserved_page_tokens = decode_page_growth_needed(running.iter(), config, max_burst);
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
            decode: Some(decode),
            ..DecodeResult::default()
        });
    };
    decode.charged = true;

    output_signals.reserve(running.len());
    let mut completed_indices = Vec::new();
    let cache_commits = Vec::new();
    let mut decode_acceptance = DecodeAcceptance::default();

    for (idx, req) in running.iter_mut().enumerate() {
        if req.pending_terminal {
            if config.speculative_max_tokens.is_none() {
                materialize_through(
                    req,
                    kv_manager,
                    &mut reservation,
                    config.block_size,
                    req.current_sequence_len(),
                );
            }
            continue;
        }
        let input_len = req.current_sequence_len();
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
            let token_id = req.next_output_token();
            req.append_output_token(token_id, config.block_size);

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
        }
        let committed = input_len + accepted.saturating_sub(1).min(burst);
        materialize_through(
            req,
            kv_manager,
            &mut reservation,
            config.block_size,
            committed,
        );
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
        end_ms: current_time_ms,
        decode: Some(decode),
        decode_acceptance,
        cache_commits,
    })
}
