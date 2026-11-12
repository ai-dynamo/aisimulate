// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Causal expert selection with bounded, independent store/query evidence.
//!
//! The first admitted loss initializes each EWMA, and a strict relative
//! improvement must persist before switching. Selection reads never admit
//! labels. The caller computes both expert predictions before feedback, then
//! stages and commits the corresponding evidence only when the observation is
//! admitted. See `docs/perf-model/fpm_ml/online-gbdt-and-causal-selection.md`
//! for the complete recurrence.

use super::estimator::SelectorConfig;
use super::model::{ForwardPassRegressionWorkloadKind as Store, WorkloadKind as Query};
use crate::AicError;

const STORES: [Store; 4] = [
    Store::PureDecode,
    Store::ContainsLocallyMixed,
    Store::CrossRankAggregated,
    Store::PurePrefill,
];
const QUERIES: [Query; 3] = [Query::Prefill, Query::Decode, Query::Mixed];
pub(crate) const KEY_COUNT: usize = STORES.len() * QUERIES.len();

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum SelectedExpert {
    #[default]
    Stable,
    Online,
}

impl SelectedExpert {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Online => "online",
        }
    }

    fn other(self) -> Self {
        match self {
            Self::Stable => Self::Online,
            Self::Online => Self::Stable,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Evidence {
    observations: u64,
    stable_ewma: f64,
    online_ewma: f64,
    active: SelectedExpert,
    streak: u64,
    switches: u64,
}

/// A validated next value for one key. Keep staging and committing within one
/// synchronous admission: no other update to this selector may intervene.
#[derive(Debug)]
pub(crate) struct SelectorUpdate {
    index: usize,
    next: Evidence,
}

#[derive(Clone, Debug)]
pub(crate) struct CausalSelector {
    config: SelectorConfig,
    states: [Evidence; KEY_COUNT],
}

impl CausalSelector {
    pub(crate) fn new(config: &SelectorConfig) -> Result<Self, AicError> {
        config.validate()?;
        Ok(Self {
            config: config.clone(),
            states: [Evidence::default(); KEY_COUNT],
        })
    }

    /// Read the choice made from earlier admitted labels. No state is created
    /// or changed on queries, including queries for a key with no evidence.
    pub(crate) fn selected(&self, store: Store, query: Query) -> SelectedExpert {
        self.states[index(store, query)].active
    }

    /// Validate and prepare one observation without mutating the selector.
    /// Losses are unbounded fractional APE, never percentages or clipped values.
    pub(crate) fn stage(
        &self,
        store: Store,
        query: Query,
        stable_ms: f64,
        online_ms: f64,
        wall_ms: f64,
    ) -> Result<SelectorUpdate, AicError> {
        if [stable_ms, online_ms, wall_ms]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(invalid(
                "expert predictions and measured latency must be finite and positive",
            ));
        }
        let losses = [
            (stable_ms / wall_ms - 1.0).abs(),
            (online_ms / wall_ms - 1.0).abs(),
        ];
        if losses.iter().any(|loss| !loss.is_finite()) {
            return Err(invalid("fractional APE overflow"));
        }
        let index = index(store, query);
        let mut next = self.states[index];
        next.observations = next
            .observations
            .checked_add(1)
            .ok_or_else(|| invalid("observation counter overflow"))?;
        if next.observations == 1 {
            next.stable_ewma = losses[0];
            next.online_ewma = losses[1];
        } else {
            // Preserve the research recurrence and operation order; initializing
            // at zero or using an FMA would change the finite-history evidence.
            next.stable_ewma =
                (1.0 - self.config.alpha) * next.stable_ewma + self.config.alpha * losses[0];
            next.online_ewma =
                (1.0 - self.config.alpha) * next.online_ewma + self.config.alpha * losses[1];
        }
        if !next.stable_ewma.is_finite() || !next.online_ewma.is_finite() {
            return Err(invalid("EWMA overflow"));
        }
        let (current, alternative) = match next.active {
            SelectedExpert::Stable => (next.stable_ewma, next.online_ewma),
            SelectedExpert::Online => (next.online_ewma, next.stable_ewma),
        };
        let preferred = next.observations >= self.config.warmup as u64
            && alternative < (1.0 - self.config.margin) * current;
        next.streak = if preferred {
            next.streak
                .checked_add(1)
                .ok_or_else(|| invalid("persistence counter overflow"))?
        } else {
            0
        };
        if next.streak >= self.config.persistence as u64 {
            next.switches = next
                .switches
                .checked_add(1)
                .ok_or_else(|| invalid("switch counter overflow"))?;
            next.active = next.active.other();
            next.streak = 0;
        }
        Ok(SelectorUpdate { index, next })
    }

    /// Apply a staged, accepted observation. The caller holds exclusive model
    /// access throughout the admission and must not commit stale staged updates.
    pub(crate) fn commit(&mut self, update: SelectorUpdate) {
        self.states[update.index] = update.next;
    }

    /// Convenience for callers that already know the observation is admitted.
    pub(crate) fn observe(
        &mut self,
        store: Store,
        query: Query,
        stable_ms: f64,
        online_ms: f64,
        wall_ms: f64,
    ) -> Result<(), AicError> {
        let update = self.stage(store, query, stable_ms, online_ms, wall_ms)?;
        self.commit(update);
        Ok(())
    }

    /// Every per-(store, query) evidence key, in the fixed key order.
    pub(crate) fn states(&self) -> &[Evidence; KEY_COUNT] {
        &self.states
    }

    /// Validate saved evidence against this selector's configuration and the
    /// forest stores that can receive feedback.
    pub(crate) fn validate_states(
        &self,
        states: &[Evidence; KEY_COUNT],
        stores: &[Store],
    ) -> Result<(), String> {
        for store in STORES {
            for query in QUERIES {
                let evidence = states[index(store, query)];
                if evidence.observations == 0 || !stores.contains(&store) {
                    if evidence != Evidence::default() {
                        return Err(
                            "selector evidence exists for a key without admitted feedback".into(),
                        );
                    }
                    continue;
                }
                let switched = evidence.switches % 2 == 1;
                if !evidence.stable_ewma.is_finite()
                    || !evidence.online_ewma.is_finite()
                    || evidence.stable_ewma < 0.0
                    || evidence.online_ewma < 0.0
                    || evidence.streak >= self.config.persistence as u64
                    || evidence.switches > evidence.observations
                    || (evidence.active == SelectedExpert::Online) != switched
                {
                    return Err("selector evidence is inconsistent".into());
                }
            }
        }
        Ok(())
    }

    pub(crate) fn set_states(&mut self, states: [Evidence; KEY_COUNT]) {
        self.states = states;
    }

    /// Bounded evidence for the one queried key; avoids allocating all twelve
    /// diagnostic states when a caller asks for prediction route metadata.
    pub(crate) fn query_diagnostics(&self, store: Store, query: Query) -> serde_json::Value {
        let evidence = &self.states[index(store, query)];
        serde_json::json!({"query_type":query_name(query),"paired_comparisons":evidence.observations,
            "warmup":self.config.warmup,"warmup_complete":evidence.observations >= self.config.warmup as u64,
            "switches":evidence.switches,"streak":evidence.streak})
    }

    /// Diagnostics allocate only on request, never while selecting or updating.
    pub(crate) fn diagnostics(&self) -> serde_json::Value {
        let states: Vec<_> = STORES
            .iter()
            .flat_map(|&store| {
                QUERIES.iter().map(move |&query| {
                    let evidence = &self.states[index(store, query)];
                    serde_json::json!({
                        "store": store,
                        "query_type": query_name(query),
                        "observations": evidence.observations,
                        "selected": evidence.active.name(),
                        "stable_ewma": (evidence.observations > 0).then_some(evidence.stable_ewma),
                        "online_ewma": (evidence.observations > 0).then_some(evidence.online_ewma),
                        "warmup_complete": evidence.observations >= self.config.warmup as u64,
                        "streak": evidence.streak,
                        "switches": evidence.switches,
                    })
                })
            })
            .collect();
        serde_json::json!({
            "config": self.config,
            "loss": "unbounded_fractional_ape",
            "scope": "forest_store_and_query_type",
            "states": states,
        })
    }
}

fn invalid(message: &str) -> AicError {
    AicError::InvalidForwardPassMetrics(format!("causal selector: {message}"))
}

pub(super) fn index(store: Store, query: Query) -> usize {
    let store = match store {
        Store::PureDecode => 0,
        Store::ContainsLocallyMixed => 1,
        Store::CrossRankAggregated => 2,
        Store::PurePrefill => 3,
    };
    let query = match query {
        Query::Prefill => 0,
        Query::Decode => 1,
        Query::Mixed => 2,
    };
    store * QUERIES.len() + query
}

pub(super) fn query_name(query: Query) -> &'static str {
    match query {
        Query::Prefill => "prefill",
        Query::Decode => "decode",
        Query::Mixed => "mixed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STORE: Store = Store::PureDecode;
    const QUERY: Query = Query::Decode;

    fn selector(alpha: f64, warmup: usize, margin: f64, persistence: usize) -> CausalSelector {
        CausalSelector::new(&SelectorConfig {
            alpha,
            warmup,
            margin,
            persistence,
        })
        .unwrap()
    }

    fn evidence(selector: &CausalSelector) -> Evidence {
        selector.states[index(STORE, QUERY)]
    }

    #[test]
    fn first_label_initializes_loss_and_prediction_reads_are_pure() {
        let mut gate = selector(0.25, 100, 0.0, 1);
        let before = gate.states;
        for _ in 0..20 {
            assert_eq!(gate.selected(STORE, QUERY), SelectedExpert::Stable);
        }
        assert_eq!(gate.states, before);
        // Absolute percentage errors are 2 and 1, not alpha-scaled zero starts.
        gate.observe(STORE, QUERY, 3.0, 2.0, 1.0).unwrap();
        assert_eq!(evidence(&gate).stable_ewma, 2.0);
        assert_eq!(evidence(&gate).online_ewma, 1.0);
        // .75*2+.25*0=1.5; .75*1+.25*4=1.75.
        gate.observe(STORE, QUERY, 1.0, 5.0, 1.0).unwrap();
        assert_eq!(evidence(&gate).stable_ewma, 1.5);
        assert_eq!(evidence(&gate).online_ewma, 1.75);
        assert_eq!(evidence(&gate).observations, 2);
    }

    #[test]
    fn warmup_label_still_uses_previous_choice_and_uncommitted_feedback_is_ignored() {
        let mut gate = selector(0.5, 3, 0.0, 1);
        for _ in 0..2 {
            assert_eq!(gate.selected(STORE, QUERY), SelectedExpert::Stable);
            gate.observe(STORE, QUERY, 1.25, 1.0, 1.0).unwrap();
        }
        let before = gate.states;
        {
            let _discarded = gate.stage(STORE, QUERY, 1.25, 1.0, 1.0).unwrap();
            assert_eq!(gate.selected(STORE, QUERY), SelectedExpert::Stable);
            assert_eq!(gate.states, before);
        }
        assert_eq!(gate.states, before);
        let accepted = gate.stage(STORE, QUERY, 1.25, 1.0, 1.0).unwrap();
        gate.commit(accepted);
        assert_eq!(gate.selected(STORE, QUERY), SelectedExpert::Online);
        assert_eq!(evidence(&gate).observations, 3);
        assert_eq!(evidence(&gate).switches, 1);
    }

    #[test]
    fn strict_relative_margin_and_ties_hold_in_both_directions() {
        let mut gate = selector(1.0, 1, 0.5, 1);
        // Challenger .5 equals (1-.5)*active 1: equality must not switch.
        gate.observe(STORE, QUERY, 2.0, 1.5, 1.0).unwrap();
        assert_eq!(gate.selected(STORE, QUERY), SelectedExpert::Stable);
        gate.observe(STORE, QUERY, 2.0, 1.25, 1.0).unwrap();
        assert_eq!(gate.selected(STORE, QUERY), SelectedExpert::Online);
        // Reverse challenger .25 equals (1-.5)*active .5.
        gate.observe(STORE, QUERY, 1.25, 1.5, 1.0).unwrap();
        assert_eq!(gate.selected(STORE, QUERY), SelectedExpert::Online);
        gate.observe(STORE, QUERY, 1.125, 1.5, 1.0).unwrap();
        assert_eq!(gate.selected(STORE, QUERY), SelectedExpert::Stable);
        assert_eq!(evidence(&gate).switches, 2);
        gate.observe(STORE, QUERY, 1.0, 1.0, 1.0).unwrap();
        assert_eq!(gate.selected(STORE, QUERY), SelectedExpert::Stable);
        assert_eq!(evidence(&gate).switches, 2);
    }

    #[test]
    fn persistence_resets_on_interruption_and_switches_after_new_consecutive_evidence() {
        let mut gate = selector(0.5, 3, 0.0, 2);
        let losses = [
            (1.0, 0.5),
            (1.0, 0.5),
            (0.0, 0.0),
            (0.0, 2.0),
            (4.0, 0.0),
            (4.0, 0.0),
            (0.0, 4.0),
            (0.0, 4.0),
        ];
        let expected_before = [
            SelectedExpert::Stable,
            SelectedExpert::Stable,
            SelectedExpert::Stable,
            SelectedExpert::Stable,
            SelectedExpert::Stable,
            SelectedExpert::Stable,
            SelectedExpert::Online,
            SelectedExpert::Online,
        ];
        let expected_streaks = [0, 0, 1, 0, 1, 0, 1, 0];
        for (i, ((stable, online), expected)) in losses.into_iter().zip(expected_before).enumerate()
        {
            assert_eq!(gate.selected(STORE, QUERY), expected);
            gate.observe(STORE, QUERY, 1.0 + stable, 1.0 + online, 1.0)
                .unwrap();
            assert_eq!(evidence(&gate).streak, expected_streaks[i]);
        }
        // Successive halving gives exact binary fractions after the eighth row.
        assert_eq!(evidence(&gate).stable_ewma, 0.765625);
        assert_eq!(evidence(&gate).online_ewma, 3.0703125);
        assert_eq!(gate.selected(STORE, QUERY), SelectedExpert::Stable);
        assert_eq!(evidence(&gate).switches, 2);
    }

    #[test]
    fn all_twelve_store_query_keys_have_independent_evidence() {
        let mut gate = selector(1.0, 1, 0.0, 1);
        let mut visited = [false; KEY_COUNT];
        for store in STORES {
            for query in QUERIES {
                let key = index(store, query);
                assert!(!visited[key]);
                let before = gate.states;
                gate.observe(store, query, 2.0, 1.0, 1.0).unwrap();
                visited[key] = true;
                for (other, (actual, previous)) in gate.states.iter().zip(before).enumerate() {
                    if other == key {
                        assert_eq!(actual.active, SelectedExpert::Online);
                        assert_eq!(actual.observations, 1);
                    } else {
                        assert_eq!(*actual, previous);
                    }
                }
            }
        }
        assert!(visited.into_iter().all(|value| value));
        assert_eq!(gate.states.len(), 12);
    }

    #[test]
    fn invalid_values_and_loss_overflow_leave_every_key_unchanged() {
        let mut gate = selector(0.5, 1, 0.0, 2);
        gate.observe(STORE, QUERY, 2.0, 1.0, 1.0).unwrap();
        let before = gate.states;
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for tuple in [(bad, 1.0, 1.0), (1.0, bad, 1.0), (1.0, 1.0, bad)] {
                assert!(
                    gate.observe(STORE, QUERY, tuple.0, tuple.1, tuple.2)
                        .is_err()
                );
                assert_eq!(gate.states, before);
            }
        }
        assert!(
            gate.observe(STORE, QUERY, f64::MAX, 1.0, f64::MIN_POSITIVE)
                .is_err()
        );
        assert_eq!(gate.states, before);
    }

    #[test]
    fn counter_overflow_is_rejected_before_publishing_any_state() {
        let mut gate = selector(1.0, 1, 0.0, 1);
        gate.states[index(STORE, QUERY)].observations = u64::MAX;
        let before = gate.states;
        assert!(gate.observe(STORE, QUERY, 2.0, 1.0, 1.0).is_err());
        assert_eq!(gate.states, before);
        gate.states[index(STORE, QUERY)].observations = 1;
        gate.states[index(STORE, QUERY)].switches = u64::MAX;
        let before = gate.states;
        assert!(gate.observe(STORE, QUERY, 2.0, 1.0, 1.0).is_err());
        assert_eq!(gate.states, before);
    }

    #[test]
    fn constructor_uses_rust_config_validation_and_diagnostics_separate_unseen_keys() {
        let invalid = SelectorConfig {
            alpha: f64::NAN,
            ..SelectorConfig::default()
        };
        assert!(CausalSelector::new(&invalid).is_err());
        let mut gate = selector(0.25, 2, 0.0, 1);
        gate.observe(STORE, QUERY, 3.0, 2.0, 1.0).unwrap();
        let diagnostics = gate.diagnostics();
        let states = diagnostics["states"].as_array().unwrap();
        assert_eq!(states.len(), 12);
        let observed = &states[index(STORE, QUERY)];
        assert_eq!(observed["observations"], 1);
        assert_eq!(observed["stable_ewma"], 2.0);
        assert_eq!(observed["online_ewma"], 1.0);
        assert_eq!(observed["warmup_complete"], false);
        let unseen = &states[index(Store::PurePrefill, Query::Prefill)];
        assert_eq!(unseen["observations"], 0);
        assert!(unseen["stable_ewma"].is_null());
        assert!(unseen["online_ewma"].is_null());
        assert_eq!(unseen["selected"], "stable");
    }
}
