// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-neutral identities and timing inputs for prefill-to-decode handoff.
//!
//! These value types are owned by the `aisimulate-core::engine` module because native
//! schedulers must retain them with their KV ownership state. Replay owns the
//! handoff coordinator, ordering, and virtual transfer event.
//!
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::engine::config::WorkerType;

/// Stable identifier for one replay-local prefill-to-decode handoff attempt.
///
/// The caller allocates the UUID identity. Engines only retain and echo it
/// in lifecycle effects, so creating an engine does not require randomness or
/// a Dynamo runtime.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct HandoffId(Uuid);

impl HandoffId {
    /// Construct an identity from the caller-owned replay coordinator.
    pub const fn new(value: Uuid) -> Self {
        Self(value)
    }

    /// Return the caller-owned UUID identity.
    pub const fn get(self) -> Uuid {
        self.0
    }
}

impl From<Uuid> for HandoffId {
    fn from(value: Uuid) -> Self {
        Self(value)
    }
}

impl From<HandoffId> for Uuid {
    fn from(value: HandoffId) -> Self {
        value.0
    }
}

/// Prompt footprint used when the Replayer models KV transfer time.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferTimingMode {
    /// Transfer time is based on every prompt token.
    #[default]
    FullPrompt,
    /// Transfer time is based only on prompt tokens missing at the destination.
    DestinationMissing,
}

/// Source-provided inputs for calculating KV transfer delay.
///
/// Schedulers publish this value when source KV ownership becomes held.
/// The Replayer combines it with the destination's missing-token observation
/// and schedules the virtual transfer completion.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HandoffTransferTiming {
    /// Which prompt footprint contributes to transfer time.
    pub mode: TransferTimingMode,
    /// Full source prompt length.
    pub full_prompt_tokens: usize,
    /// Modeled KV bytes occupied by one prompt token.
    pub kv_bytes_per_token: Option<usize>,
    /// Fixed request-local payload, charged once even when no tokens are missing.
    #[serde(default)]
    pub kv_bytes_per_request: usize,
    /// Modeled transfer bandwidth in decimal gigabytes per second.
    pub bandwidth_gb_s: Option<f64>,
}

impl HandoffTransferTiming {
    /// Calculate transfer delay in milliseconds.
    ///
    /// Returns `None` when the source did not provide a complete timing model
    /// or when bandwidth is non-positive. In that case the Replayer may apply
    /// its configured fallback delay.
    pub fn delay_ms(self, destination_missing_tokens: usize) -> Option<f64> {
        let tokens = match self.mode {
            TransferTimingMode::FullPrompt => self.full_prompt_tokens,
            TransferTimingMode::DestinationMissing => destination_missing_tokens,
        };
        let bandwidth_gb_s = self.bandwidth_gb_s?;
        if self.kv_bytes_per_token.is_none() && self.kv_bytes_per_request == 0 {
            return None;
        }
        let bytes_per_token = self.kv_bytes_per_token.unwrap_or(0);
        if !bandwidth_gb_s.is_finite() || bandwidth_gb_s <= 0.0 {
            return None;
        }

        Some(
            (tokens as f64 * bytes_per_token as f64 + self.kv_bytes_per_request as f64)
                / (bandwidth_gb_s * 1e9)
                * 1000.0,
        )
    }

    /// Calculate delay using the full prompt irrespective of `mode`.
    pub fn full_prompt_delay_ms(self) -> Option<f64> {
        Self {
            mode: TransferTimingMode::FullPrompt,
            ..self
        }
        .delay_ms(0)
    }
}

/// Compute the client-visible prefill-to-decode handoff delay.
///
/// A delay is exposed only for a completed prefill-worker request. Aggregated
/// and decode workers do not cross a prefill/decode transfer boundary.
pub fn prefill_handoff_delay_ms(
    worker_type: WorkerType,
    completed: bool,
    num_input_tokens: usize,
    bandwidth_gb_s: Option<f64>,
    kv_bytes_per_token: Option<usize>,
) -> Option<f64> {
    prefill_handoff_delay_with_request_bytes_ms(
        worker_type,
        completed,
        num_input_tokens,
        bandwidth_gb_s,
        kv_bytes_per_token,
        0,
    )
}

/// Affine counterpart of [`prefill_handoff_delay_ms`].
pub fn prefill_handoff_delay_with_request_bytes_ms(
    worker_type: WorkerType,
    completed: bool,
    num_input_tokens: usize,
    bandwidth_gb_s: Option<f64>,
    kv_bytes_per_token: Option<usize>,
    kv_bytes_per_request: usize,
) -> Option<f64> {
    if worker_type != WorkerType::Prefill || !completed {
        return None;
    }
    HandoffTransferTiming {
        mode: TransferTimingMode::FullPrompt,
        full_prompt_tokens: num_input_tokens,
        kv_bytes_per_token,
        kv_bytes_per_request,
        bandwidth_gb_s,
    }
    .full_prompt_delay_ms()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_transfer_charges_fixed_payload_once_and_scales_with_bandwidth() {
        let timing = HandoffTransferTiming {
            mode: TransferTimingMode::DestinationMissing,
            full_prompt_tokens: 4096,
            kv_bytes_per_token: Some(4444),
            kv_bytes_per_request: 21_479_424,
            bandwidth_gb_s: Some(100.0),
        };
        for tokens in [0, 64, 512, 4096] {
            let expected = (tokens * 4444 + 21_479_424) as f64 / 100_000_000.0;
            assert!((timing.delay_ms(tokens).unwrap() - expected).abs() < 1e-12);
            let slower = HandoffTransferTiming {
                bandwidth_gb_s: Some(1.0),
                ..timing
            };
            assert!((slower.delay_ms(tokens).unwrap() - expected * 100.0).abs() < 1e-10);
        }
        let fixed_only = HandoffTransferTiming {
            kv_bytes_per_token: None,
            ..timing
        };
        assert_eq!(fixed_only.delay_ms(0), fixed_only.delay_ms(4096));
        assert!(fixed_only.delay_ms(0).unwrap() > 0.0);
    }

    #[test]
    fn old_handoff_json_defaults_to_zero_fixed_payload() {
        let timing: HandoffTransferTiming = serde_json::from_value(serde_json::json!({
            "mode": "full_prompt", "full_prompt_tokens": 128,
            "kv_bytes_per_token": 1000, "bandwidth_gb_s": 1.0
        }))
        .unwrap();
        assert_eq!(timing.kv_bytes_per_request, 0);
        assert_eq!(timing.full_prompt_delay_ms(), Some(0.128));
    }

    #[test]
    fn handoff_id_round_trips_caller_owned_value() {
        let value = Uuid::from_u128(17);
        let handoff_id = HandoffId::new(value);
        assert_eq!(handoff_id.get(), value);
    }

    #[test]
    fn transfer_delay_uses_selected_prompt_footprint() {
        let timing = HandoffTransferTiming {
            mode: TransferTimingMode::DestinationMissing,
            full_prompt_tokens: 100,
            kv_bytes_per_request: 0,
            kv_bytes_per_token: Some(1_000),
            bandwidth_gb_s: Some(1.0),
        };

        assert_eq!(timing.delay_ms(20), Some(0.02));
        assert_eq!(timing.full_prompt_delay_ms(), Some(0.1));
    }

    #[test]
    fn incomplete_or_non_positive_timing_model_has_no_delay() {
        let timing = HandoffTransferTiming {
            mode: TransferTimingMode::FullPrompt,
            full_prompt_tokens: 100,
            kv_bytes_per_request: 0,
            kv_bytes_per_token: None,
            bandwidth_gb_s: Some(1.0),
        };
        assert_eq!(timing.delay_ms(0), None);

        let timing = HandoffTransferTiming {
            kv_bytes_per_token: Some(1_000),
            bandwidth_gb_s: Some(0.0),
            ..timing
        };
        assert_eq!(timing.delay_ms(0), None);
    }

    #[test]
    fn prefill_handoff_delay_requires_completed_prefill_work() {
        let args = (128, Some(1.0), Some(1_000_000));
        assert_eq!(
            prefill_handoff_delay_ms(WorkerType::Prefill, true, args.0, args.1, args.2,),
            Some(128.0)
        );
        assert_eq!(
            prefill_handoff_delay_ms(WorkerType::Prefill, false, args.0, args.1, args.2,),
            None
        );
        assert_eq!(
            prefill_handoff_delay_ms(WorkerType::Decode, true, args.0, args.1, args.2,),
            None
        );
    }
}
