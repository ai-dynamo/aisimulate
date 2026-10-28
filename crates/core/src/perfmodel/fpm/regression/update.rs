// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publish complete coefficient/normalization snapshots only when due.
use super::super::estimator::RegressionUpdatePolicy;
use super::spline::ErrorMonitor;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub(super) struct LinearUpdateState {
    /// Configuration-only and internally tagged; snapshots rebuild it from
    /// the canonical configuration.
    #[serde(skip)]
    policy: RegressionUpdatePolicy,
    monitor: ErrorMonitor,
    accepted: usize,
    age: usize,
}

impl LinearUpdateState {
    pub(super) fn new(policy: RegressionUpdatePolicy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }

    pub(super) fn is_always(&self) -> bool {
        self.policy.is_always()
    }

    /// The prediction precedes admission. Rebuilds bypass the error gate.
    pub(super) fn accepted(
        &mut self,
        prior: Option<f64>,
        target: f64,
        ready: bool,
        rebuild: bool,
    ) -> bool {
        let RegressionUpdatePolicy::ErrorThreshold {
            relative_tolerance,
            absolute_tolerance_ms,
            window,
            trigger,
            cooldown,
            startup_observations,
        } = self.policy
        else {
            return true;
        };
        self.accepted = self.accepted.saturating_add(1);
        self.age = self.age.saturating_add(1);
        let prior = prior.filter(|prediction| prediction.is_finite());
        if let Some(prediction) = prior {
            self.monitor.observe(
                (prediction - target).abs()
                    > absolute_tolerance_ms.max(relative_tolerance * target),
                window,
            );
        }
        rebuild
            || !ready
            || prior.is_none()
            || self.accepted <= startup_observations
            || (self.monitor.bad >= trigger && self.age >= cooldown)
    }

    pub(super) fn fitted(&mut self, success: bool) {
        if success && !self.is_always() {
            self.age = 0;
            self.monitor.clear();
        }
    }
}

impl LinearUpdateState {
    /// Adopt the configured policy and validate the saved clocks against it.
    pub(super) fn restore_snapshot(&mut self, fresh: &Self) -> Result<(), String> {
        self.policy = fresh.policy.clone();
        match self.policy {
            RegressionUpdatePolicy::Always => {
                if !self.monitor.is_empty() || self.accepted != 0 || self.age != 0 {
                    return Err("eager regression updates carry lazy-update state".into());
                }
            }
            RegressionUpdatePolicy::ErrorThreshold { window, .. } => {
                self.monitor.validate_restored(window)?;
                if self.age > self.accepted {
                    return Err("lazy regression update clocks are inconsistent".into());
                }
            }
        }
        Ok(())
    }
}
