// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared speculative role rules for canonical launch and native replay.

use anyhow::{Result, ensure};

use super::{Backend, EngineConfig};

pub(crate) fn speculative_capacity_mismatch(
    kind: &str,
    capacity_is_explicit: bool,
    num_gpu_blocks: usize,
) -> Option<&'static str> {
    (!matches!(kind, "none" | "ngram") && !(capacity_is_explicit && num_gpu_blocks > 0))
        .then_some("generic speculation requires explicitly configured positive num_gpu_blocks")
}

/// Return the same role constraints for saved-timing checks and replay launches.
/// Depth must already be resolved from the scheme registry; an absent scheduler
/// depth is derived at the input boundary before calling this function.
pub(crate) fn speculative_role_mismatches(
    rank: &EngineConfig,
    kind: &str,
    timing_backend: &str,
    depth: u32,
    capacity_is_explicit: bool,
) -> Vec<String> {
    let mut mismatches = Vec::new();
    if depth > 0 {
        let backend = match rank.backend {
            Backend::Vllm => "vllm",
            Backend::Sglang => "sglang",
            Backend::Trtllm => "trtllm",
        };
        if timing_backend != backend {
            mismatches.push("AIS backend must match engine backend".into());
        }
        if let Err(error) = ngram_async_scheduling(rank, kind) {
            mismatches.push(error.to_string());
        }
        if !matches!(timing_backend, "vllm" | "sglang") {
            mismatches.push("generic speculation requires backend=vllm or backend=sglang".into());
        }
        if kind != "ngram"
            && !rank
                .aic_nextn_accept_rates
                .as_deref()
                .is_some_and(|rates| !rates.trim().is_empty())
        {
            mismatches
                .push("generic speculative replay requires explicit aic_nextn_accept_rates".into());
        }
        if let Some(message) =
            speculative_capacity_mismatch(kind, capacity_is_explicit, rank.num_gpu_blocks)
        {
            mismatches.push(message.into());
        }
    }
    if depth as usize != rank.aic_nextn.unwrap_or(0) {
        mismatches.push("canonical speculative depth conflicts with scheduler aic_nextn".into());
    }
    mismatches
}

/// vLLM prompt lookup defaults to synchronous scheduling and rejects explicit async.
fn ngram_async_scheduling(rank: &EngineConfig, kind: &str) -> Result<Option<bool>> {
    if kind == "ngram" && rank.backend == Backend::Vllm {
        ensure!(
            rank.async_scheduling != Some(true),
            "vLLM does not support async scheduling with ngram speculative decoding"
        );
        Ok(Some(false))
    } else {
        Ok(rank.async_scheduling)
    }
}

pub(crate) fn resolve_speculative_role(
    rank: &mut EngineConfig,
    kind: &str,
    timing_backend: &str,
    depth: u32,
    capacity_is_explicit: bool,
) -> Result<()> {
    if rank.aic_nextn.is_none() {
        rank.aic_nextn = (depth > 0).then_some(depth as usize);
    }
    if let Some(message) =
        speculative_role_mismatches(rank, kind, timing_backend, depth, capacity_is_explicit)
            .into_iter()
            .next()
    {
        anyhow::bail!(message);
    }
    rank.async_scheduling = ngram_async_scheduling(rank, kind)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_defaults_and_explicit_async_follow_the_same_role_rules() {
        let mut rank = EngineConfig::default();
        resolve_speculative_role(&mut rank, "ngram", "vllm", 3, false).unwrap();
        assert_eq!(rank.aic_nextn, Some(3));
        assert_eq!(rank.async_scheduling, Some(false));
        assert_eq!(rank.aic_nextn_accept_rates, None);
        rank.async_scheduling = Some(true);
        assert!(
            resolve_speculative_role(&mut rank, "ngram", "vllm", 3, false)
                .unwrap_err()
                .to_string()
                .contains("async scheduling")
        );
    }

    #[test]
    fn ngram_scheduling_policy_leaves_other_backends_and_schemes_unchanged() {
        let mut rank = EngineConfig::for_backend(Backend::Sglang);
        assert_eq!(ngram_async_scheduling(&rank, "ngram").unwrap(), None);
        rank.async_scheduling = Some(true);
        assert_eq!(ngram_async_scheduling(&rank, "ngram").unwrap(), Some(true));
        rank.backend = Backend::Vllm;
        for kind in ["none", "mtp", "eagle3"] {
            assert_eq!(ngram_async_scheduling(&rank, kind).unwrap(), Some(true));
        }
    }

    #[test]
    fn generic_role_requires_acceptance_and_explicit_positive_capacity() {
        let mut rank = EngineConfig::default();
        rank.aic_nextn = Some(3);
        assert_eq!(
            speculative_role_mismatches(&rank, "mtp", "vllm", 3, false),
            [
                "generic speculative replay requires explicit aic_nextn_accept_rates",
                "generic speculation requires explicitly configured positive num_gpu_blocks",
            ]
        );
        rank.aic_nextn_accept_rates = Some("1,1,0.4".into());
        resolve_speculative_role(&mut rank, "mtp", "vllm", 3, true).unwrap();
        rank.num_gpu_blocks = 0;
        assert!(resolve_speculative_role(&mut rank, "mtp", "vllm", 3, true).is_err());
    }
}
