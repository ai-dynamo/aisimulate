// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independent frozen/calibrated and online experts with causal selection.
//! Feedback uses predictions from before that admission. The caller must supply
//! observations chronologically; delayed labels must not masquerade as the
//! query-time errors of a different model version.

use super::super::selector::{CausalSelector, Evidence, KEY_COUNT, SelectedExpert};
use super::*;

#[derive(Clone, Debug)]
pub(super) struct CalibratedOnlineModel {
    pub(super) stable: Arc<LearnedForwardPassModel>,
    pub(super) learned: LearnedForwardPassModel,
    pub(super) online: OnlineLearner,
    pub(super) identity_verified: bool,
    // At most four stores. Dedicated roles allocate only their trained store.
    corrections: Vec<(
        ForwardPassRegressionWorkloadKind,
        WorkloadStores<CorrectionBuckets>,
    )>,
    selector: CausalSelector,
}

impl CalibratedOnlineModel {
    pub(super) fn new(
        stable: LearnedForwardPassModel,
        learned: LearnedForwardPassModel,
        online: OnlineLearner,
        identity_verified: bool,
        options: &ForwardPassPerfOptions,
        config: &super::super::SelectorConfig,
    ) -> Result<Self, AicError> {
        debug_assert_eq!(stable.feature_names(), learned.feature_names());
        debug_assert_eq!(stable.store_kinds(), learned.store_kinds());
        let corrections = stable
            .store_kinds()
            .into_iter()
            .map(|kind| (kind, WorkloadStores::with_options(options)))
            .collect();
        Ok(Self {
            stable: Arc::new(stable),
            learned,
            online,
            identity_verified,
            corrections,
            selector: CausalSelector::new(config)?,
        })
    }

    fn corrections(
        &self,
        kind: ForwardPassRegressionWorkloadKind,
    ) -> &WorkloadStores<CorrectionBuckets> {
        &self
            .corrections
            .iter()
            .find(|(store, _)| *store == kind)
            .expect("trained forest has a calibration store")
            .1
    }

    fn corrections_mut(
        &mut self,
        kind: ForwardPassRegressionWorkloadKind,
    ) -> &mut WorkloadStores<CorrectionBuckets> {
        &mut self
            .corrections
            .iter_mut()
            .find(|(store, _)| *store == kind)
            .expect("trained forest has a calibration store")
            .1
    }

    fn stable_prediction(
        &self,
        kind: ForwardPassRegressionWorkloadKind,
        feature: &IterationFeatures,
        base: f64,
    ) -> Result<f64, AicError> {
        let prediction = base
            * self
                .corrections(kind)
                .store(feature.workload_kind)
                .correction_factor_for(&feature.x);
        if !prediction.is_finite() || prediction <= 0.0 {
            return Err(AicError::InvalidForwardPassMetrics(
                "calibrated frozen prediction must be finite and positive".into(),
            ));
        }
        Ok(prediction)
    }

    pub(super) fn predict(&self, metrics: &[ForwardPassMetrics]) -> Result<Option<f64>, AicError> {
        let Some((kind, vector)) = prepare_learned_iteration(&self.stable, metrics)? else {
            return Ok(Some(0.0));
        };
        let Some(vector) = vector else {
            return Ok(None);
        };
        let feature =
            IterationFeatures::from_validated_metrics(metrics).expect("nonempty learned query");
        // Only the chosen expert is evaluated during ordinary inference.
        match self.selector.selected(kind, feature.workload_kind) {
            SelectedExpert::Stable => self
                .stable
                .predict_ms(kind, &vector)?
                .map(|base| self.stable_prediction(kind, &feature, base))
                .transpose(),
            SelectedExpert::Online => self.learned.predict_ms(kind, &vector),
        }
    }

    pub(super) fn observe(&mut self, metrics: &[ForwardPassMetrics]) -> Result<(), AicError> {
        let (kind, vector) = match prepare_learned_iteration(&self.stable, metrics) {
            Ok(Some((kind, Some(vector)))) => (kind, vector),
            Ok(None | Some((_, None))) => return Ok(()),
            Err(error) => {
                self.online.record_rejected_observation();
                return Err(error);
            }
        };
        let Some(wall_ms) = max_positive_wall_time_ms(metrics) else {
            self.online.record_rejected_observation();
            return Ok(());
        };
        let feature =
            IterationFeatures::from_validated_metrics(metrics).expect("nonempty learned feedback");
        // The retained row owns this one projection. Both pre-update forest
        // predictions reuse it; the raw frozen prediction also feeds calibration.
        let x = self.learned.selected_features(&vector);
        let staged = (|| {
            let base = self
                .stable
                .predict_selected_ms(kind, &x)?
                .expect("trained frozen store");
            let stable_ms = self.stable_prediction(kind, &feature, base)?;
            let online_ms = self
                .learned
                .predict_selected_ms(kind, &x)?
                .expect("matching online store");
            let update =
                self.selector
                    .stage(kind, feature.workload_kind, stable_ms, online_ms, wall_ms)?;
            Ok::<_, AicError>((base, update))
        })();
        let (base, update) = match staged {
            Ok(value) => value,
            Err(error) => {
                self.online.record_rejected_observation();
                return Err(error);
            }
        };
        let accepted_before = self.online.accepted_observations_for(kind);
        let result = self.online.observe(kind, x, wall_ms, &mut self.learned);
        let admitted = self.online.accepted_observations_for(kind) != accepted_before;
        if admitted {
            // Fitting can fail after admission, leaving a bounded pending batch.
            // That label still belongs to both expert histories exactly once.
            self.corrections_mut(kind)
                .store_mut(feature.workload_kind)
                .add_observation(feature.x, wall_ms, base);
            self.selector.commit(update);
        }
        match result {
            Err(error) if admitted => Err(AicError::InvalidEngineConfig(format!(
                "combined feedback was admitted; online batch is pending, retry flush_updates: {error}"
            ))),
            Err(error) => Err(error),
            Ok(_) => Ok(()),
        }
    }

    pub(super) fn correction_stores(
        &self,
    ) -> &[(
        ForwardPassRegressionWorkloadKind,
        WorkloadStores<CorrectionBuckets>,
    )] {
        &self.corrections
    }

    pub(super) fn selector_states(&self) -> [Evidence; KEY_COUNT] {
        *self.selector.states()
    }

    /// Validate saved calibration and selector evidence against this model's
    /// freshly configured stores.
    pub(super) fn validate_snapshot(
        &self,
        corrections: &[(
            ForwardPassRegressionWorkloadKind,
            WorkloadStores<CorrectionBuckets>,
        )],
        states: &[Evidence; KEY_COUNT],
    ) -> Result<(), String> {
        if corrections.len() != self.corrections.len()
            || corrections
                .iter()
                .zip(&self.corrections)
                .any(|((saved, _), (fresh, _))| saved != fresh)
        {
            return Err("calibration stores differ from the frozen forest stores".into());
        }
        for ((kind, saved), (_, fresh)) in corrections.iter().zip(&self.corrections) {
            saved
                .validate_restored(fresh)
                .map_err(|error| format!("{kind:?} calibration: {error}"))?;
        }
        self.selector
            .validate_states(states, &self.stable.store_kinds())
    }

    pub(super) fn install_snapshot(
        &mut self,
        corrections: Vec<(
            ForwardPassRegressionWorkloadKind,
            WorkloadStores<CorrectionBuckets>,
        )>,
        states: [Evidence; KEY_COUNT],
    ) {
        self.corrections = corrections;
        self.selector.set_states(states);
    }

    pub(super) fn set_identity_verified(&mut self, verified: bool) {
        self.identity_verified = verified;
    }

    pub(super) fn correction_count(&self) -> usize {
        self.corrections
            .iter()
            .map(|(_, store)| store.observation_count())
            .sum()
    }

    pub(super) fn store_correction_count(&self, kind: ForwardPassRegressionWorkloadKind) -> usize {
        self.corrections
            .iter()
            .find(|(store, _)| *store == kind)
            .map_or(0, |(_, store)| store.observation_count())
    }

    pub(super) fn ready_bucket_count(&self) -> usize {
        self.corrections
            .iter()
            .map(|(_, store)| store.ready_bucket_count())
            .sum()
    }

    pub(super) fn correction_factors(&self) -> Vec<f64> {
        self.corrections
            .iter()
            .flat_map(|(_, store)| store.correction_factors())
            .collect()
    }

    pub(super) fn diagnostics(&self) -> serde_json::Value {
        let mut value = self.selector.diagnostics();
        value["worker_type"] =
            serde_json::to_value(self.stable.worker_type()).expect("worker role serializes");
        value["identity_verified"] = self.identity_verified.into();
        value["calibration"] = self.corrections.iter().flat_map(|(kind, stores)| {
            [(WorkloadKind::Prefill, "prefill"), (WorkloadKind::Decode, "decode"), (WorkloadKind::Mixed, "mixed")]
                .into_iter().map(move |(query, name)| {
                    let store = stores.store(query);
                    serde_json::json!({"store": kind, "query_type": name,
                        "ready": store.is_ready(), "retained_observations": store.observation_count(),
                        "ready_buckets": store.ready_bucket_count()})
                })
        }).collect::<Vec<_>>().into();
        value
    }
}

#[cfg(test)]
#[path = "combined_tests.rs"]
mod tests;
