// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Paired pre-update APE evidence; no training rows or query-time mutations.
//! Twelve fixed keys bound storage independently of trace length. Both experts
//! keep learning when unselected. Ties favor online; readiness alone never does.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::estimator::MixedQualityGateConfig;
use super::model::{ForwardPassRegressionWorkloadKind as Store, WorkloadKind as Query};
use super::selector::{SelectedExpert, index, query_name};
use crate::AicError;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Evidence {
    comparisons: u64,
    interpolation_ewma: f64,
    gbdt_ewma: f64,
    online: bool,
    switches: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct QualityGate {
    config: MixedQualityGateConfig,
    states: [Evidence; 12],
}

pub(super) struct QualityUpdate {
    index: usize,
    evidence: Evidence,
}

impl QualityGate {
    pub(super) fn new(config: &MixedQualityGateConfig) -> Self {
        Self {
            config: config.clone(),
            states: [Evidence::default(); 12],
        }
    }

    pub(super) fn selected(&self, store: Store, query: Query) -> SelectedExpert {
        if self.states[index(store, query)].online {
            SelectedExpert::Online
        } else {
            SelectedExpert::Stable
        }
    }

    pub(super) fn stage(
        &self,
        store: Store,
        query: Query,
        interpolation: f64,
        gbdt: f64,
        wall: f64,
    ) -> Result<QualityUpdate, AicError> {
        if [interpolation, gbdt, wall]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(invalid(
                "predictions and measured latency must be finite and positive",
            ));
        }
        let losses = [
            (interpolation / wall - 1.0).abs(),
            (gbdt / wall - 1.0).abs(),
        ];
        if losses.iter().any(|v| !v.is_finite()) {
            return Err(invalid("APE overflow"));
        }
        let index = index(store, query);
        let mut evidence = self.states[index];
        evidence.comparisons = evidence
            .comparisons
            .checked_add(1)
            .ok_or_else(|| invalid("counter overflow"))?;
        if evidence.comparisons == 1 {
            evidence.interpolation_ewma = losses[0];
            evidence.gbdt_ewma = losses[1];
        } else {
            let alpha = self.config.alpha;
            evidence.interpolation_ewma =
                (1.0 - alpha) * evidence.interpolation_ewma + alpha * losses[0];
            evidence.gbdt_ewma = (1.0 - alpha) * evidence.gbdt_ewma + alpha * losses[1];
        }
        if !evidence.interpolation_ewma.is_finite() || !evidence.gbdt_ewma.is_finite() {
            return Err(invalid("EWMA overflow"));
        }
        let online = evidence.comparisons >= self.config.min_comparisons as u64
            && evidence.gbdt_ewma <= evidence.interpolation_ewma;
        if online != evidence.online {
            evidence.switches = evidence
                .switches
                .checked_add(1)
                .ok_or_else(|| invalid("switch counter overflow"))?;
        }
        evidence.online = online;
        Ok(QualityUpdate { index, evidence })
    }

    pub(super) fn commit(&mut self, update: QualityUpdate) {
        self.states[update.index] = update.evidence;
    }

    pub(super) fn query_diagnostics(&self, store: Store, query: Query) -> serde_json::Value {
        let e = self.states[index(store, query)];
        serde_json::json!({
            "query_type": query_name(query), "paired_comparisons": e.comparisons,
            "observations": e.comparisons, "warmup": self.config.min_comparisons,
            "warmup_complete": e.comparisons >= self.config.min_comparisons as u64,
            "selected": self.selected(store, query).name(), "switches": e.switches,
            "stable_ewma": (e.comparisons > 0).then_some(e.interpolation_ewma),
            "online_ewma": (e.comparisons > 0).then_some(e.gbdt_ewma),
            "quality_gate": self.config, "streak": 0,
        })
    }

    /// Validate before restoring any calibration or evidence. The source binding
    /// separately pins the policy, original A, interpolation table and geometry.
    pub(super) fn validate_snapshot(
        &self,
        expected: &MixedQualityGateConfig,
        admissions: &BTreeMap<Store, u64>,
    ) -> Result<(), AicError> {
        if &self.config != expected {
            return Err(invalid("saved policy differs"));
        }
        expected.validate()?;
        for store in [
            Store::PureDecode,
            Store::ContainsLocallyMixed,
            Store::CrossRankAggregated,
            Store::PurePrefill,
        ] {
            let mut total = 0u64;
            for query in [Query::Prefill, Query::Decode, Query::Mixed] {
                let e = self.states[index(store, query)];
                total = total
                    .checked_add(e.comparisons)
                    .ok_or_else(|| invalid("counter overflow"))?;
                let ready = e.comparisons >= expected.min_comparisons as u64;
                if !e.interpolation_ewma.is_finite()
                    || !e.gbdt_ewma.is_finite()
                    || e.interpolation_ewma < 0.0
                    || e.gbdt_ewma < 0.0
                    || e.switches > e.comparisons
                    || e.online != (e.switches % 2 == 1)
                    || (!ready && (e.online || e.switches > 0))
                    || e.online != (ready && e.gbdt_ewma <= e.interpolation_ewma)
                    || (e.comparisons == 0 && e != Evidence::default())
                {
                    return Err(invalid("invalid saved evidence"));
                }
            }
            if total > admissions.get(&store).copied().unwrap_or(0) {
                return Err(invalid("comparisons exceed admitted labels"));
            }
        }
        Ok(())
    }
}

fn invalid(message: &str) -> AicError {
    AicError::InvalidForwardPassMetrics(format!("mixed fallback quality gate: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_losses_warmup_ties_reentry_and_isolation() {
        let mut gate = QualityGate::new(&MixedQualityGateConfig {
            alpha: 1.0,
            min_comparisons: 2,
        });
        let store = Store::ContainsLocallyMixed;
        let query = Query::Mixed;
        // 14/10 => 40% fallback APE; 10/10 => zero GBDT APE.
        let update = gate.stage(store, query, 14.0, 10.0, 10.0).unwrap();
        assert_eq!(gate.selected(store, query), SelectedExpert::Stable);
        gate.commit(update);
        assert_eq!(gate.selected(store, query), SelectedExpert::Stable);
        // Equal predictions have exactly equal losses: equality qualifies.
        let update = gate.stage(store, query, 10.0, 10.0, 10.0).unwrap();
        gate.commit(update);
        assert_eq!(gate.selected(store, query), SelectedExpert::Online);
        let update = gate.stage(store, query, 10.0, 14.0, 10.0).unwrap();
        gate.commit(update);
        assert_eq!(gate.selected(store, query), SelectedExpert::Stable);
        assert_eq!(
            gate.selected(Store::CrossRankAggregated, query),
            SelectedExpert::Stable
        );
        assert_eq!(gate.selected(store, Query::Prefill), SelectedExpert::Stable);
        assert!(gate.stage(store, query, f64::NAN, 10.0, 10.0).is_err());
        gate.validate_snapshot(&gate.config, &BTreeMap::from([(store, 3)]))
            .unwrap();
        assert!(
            gate.validate_snapshot(&gate.config, &BTreeMap::from([(store, 2)]))
                .is_err()
        );
    }

    #[test]
    fn ewma_uses_preceding_evidence_without_zero_initialization() {
        let mut gate = QualityGate::new(&MixedQualityGateConfig {
            alpha: 0.5,
            min_comparisons: 1,
        });
        let s = Store::CrossRankAggregated;
        let q = Query::Decode;
        // First losses are 1 and 0, then 0 and 0.5; EWMAs are 0.5 and 0.25.
        for (a, b) in [(20.0, 10.0), (10.0, 15.0)] {
            let u = gate.stage(s, q, a, b, 10.0).unwrap();
            gate.commit(u);
        }
        let d = gate.query_diagnostics(s, q);
        assert_eq!(d["stable_ewma"], 0.5);
        assert_eq!(d["online_ewma"], 0.25);
        assert_eq!(d["selected"], "online");
    }
}
