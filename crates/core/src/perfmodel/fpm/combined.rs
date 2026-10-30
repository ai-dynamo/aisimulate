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
    appended_mixed_sources:
        std::collections::BTreeMap<ForwardPassRegressionWorkloadKind, serde_json::Value>,
}

pub(super) struct MixedSelectorPrediction {
    pub(super) estimate: crate::ForwardPassEstimate,
    pub(super) selected: SelectedExpert,
    pub(super) preferred: SelectedExpert,
    pub(super) stable_available: Option<bool>,
    pub(super) online_available: Option<bool>,
    pub(super) query: WorkloadKind,
    pub(super) kind: ForwardPassRegressionWorkloadKind,
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
            appended_mixed_sources: Default::default(),
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

    pub(super) fn append_mixed_checkpoint(
        &mut self,
        path: &std::path::Path,
        eligible: &[ForwardPassRegressionWorkloadKind],
    ) -> Result<usize, AicError> {
        let (kind, source) =
            self.online
                .append_mixed_training_checkpoint(path, &mut self.learned, eligible)?;
        self.appended_mixed_sources.insert(kind, source);
        Ok(1)
    }

    pub(super) fn predict_mixed(
        &self,
        metrics: &[ForwardPassMetrics],
        mixed: &MixedComposition,
        kind: ForwardPassRegressionWorkloadKind,
        detailed: bool,
    ) -> Result<MixedSelectorPrediction, AicError> {
        let feature =
            IterationFeatures::from_validated_metrics(metrics).expect("nonempty mixed query");
        let preferred = self.selector.selected(kind, feature.workload_kind);
        let online_ready = self.learned.has_store(kind);
        if preferred == SelectedExpert::Online && online_ready {
            let online = learned_base_ms(&self.learned, metrics)?;
            return Ok(MixedSelectorPrediction {
                estimate: ForwardPassPerfModel::learned_estimate_details(online),
                selected: SelectedExpert::Online,
                preferred,
                stable_available: None,
                online_available: Some(online.is_some()),
                query: feature.workload_kind,
                kind,
            });
        }
        let prior_feature = mixed.owned_feature(metrics)?.expect("eligible mixed query");
        let prior = mixed.predict(metrics, &prior_feature, detailed)?;
        if prior.latency_ms.is_some() || !online_ready {
            return Ok(MixedSelectorPrediction {
                stable_available: Some(prior.latency_ms.is_some()),
                online_available: (!online_ready).then_some(false),
                estimate: prior,
                selected: SelectedExpert::Stable,
                preferred,
                query: feature.workload_kind,
                kind,
            });
        }
        let online = learned_base_ms(&self.learned, metrics)?;
        Ok(MixedSelectorPrediction {
            estimate: ForwardPassPerfModel::learned_estimate_details(online),
            selected: SelectedExpert::Online,
            preferred,
            stable_available: Some(false),
            online_available: Some(online.is_some()),
            query: feature.workload_kind,
            kind,
        })
    }

    pub(super) fn mixed_route_metadata(
        &self,
        metrics: &[ForwardPassMetrics],
        mixed: &MixedComposition,
        prediction: &MixedSelectorPrediction,
    ) -> Result<serde_json::Value, AicError> {
        let mut route = mixed.route(metrics, prediction.selected == SelectedExpert::Online)?;
        let object = route.as_object_mut().expect("route object");
        object.insert("selected_expert".into(), prediction.selected.name().into());
        object.insert(
            "preferred_expert".into(),
            prediction.preferred.name().into(),
        );
        object.insert("stable_source".into(), "interpolation".into());
        object.insert(
            "online_ready".into(),
            self.learned.has_store(prediction.kind).into(),
        );
        object.insert(
            "stable_available".into(),
            prediction.stable_available.into(),
        );
        object.insert(
            "online_available".into(),
            prediction.online_available.into(),
        );
        object.extend(
            self.selector
                .query_diagnostics(prediction.kind, prediction.query)
                .as_object()
                .expect("query evidence object")
                .clone(),
        );
        Ok(route)
    }

    /// Observe the interpolation stable expert and C-trained online expert from
    /// one pre-update state. Missing/unsupported prior skips paired evidence,
    /// while hard prior errors remain visible after valid online admission.
    pub(super) fn observe_mixed(
        &mut self,
        metrics: &[ForwardPassMetrics],
        mixed: &mut MixedComposition,
        feature: &RegressionIterationFeatures,
    ) -> Result<(), AicError> {
        let kind = feature.workload_kind;
        let pending = mixed.prepare_observation(metrics, feature);
        let prepared = match prepare_learned_iteration(&self.learned, metrics) {
            Ok(value) => value,
            Err(error) => {
                self.online.record_rejected_observation();
                return Err(error);
            }
        };
        let Some((_, Some(vector))) = prepared else {
            if let Some(observation) = pending? {
                mixed.commit_observation(observation);
            }
            return Ok(());
        };
        let Some(wall_ms) = max_positive_wall_time_ms(metrics) else {
            self.online.record_rejected_observation();
            pending?;
            return Ok(());
        };
        let query = IterationFeatures::from_validated_metrics(metrics)
            .expect("mixed feedback")
            .workload_kind;
        let x = self.learned.selected_features(&vector);
        let online_ms = self
            .learned
            .predict_selected_ms(kind, &x)?
            .expect("ready online store");
        let comparison = match &pending {
            Ok(Some(observation)) => mixed
                .staged_prediction(observation)
                .and_then(|stable_ms| {
                    self.selector
                        .stage(kind, query, stable_ms, online_ms, wall_ms)
                })
                .map(Some),
            _ => Ok(None),
        };
        let before = self.online.accepted_observations_for(kind);
        let result = self.online.observe(kind, x, wall_ms, &mut self.learned);
        let admitted = self.online.accepted_observations_for(kind) != before;
        let comparison_error = match comparison {
            Ok(Some(update)) if admitted => {
                self.selector.commit(update);
                None
            }
            Err(error) => Some(error),
            _ => None,
        };
        if admitted {
            match pending {
                Ok(Some(observation)) => mixed.commit_observation(observation),
                Err(prior_error) => {
                    result?;
                    return Err(prior_error);
                }
                _ => {}
            }
        } else {
            result?;
            pending?;
            if let Some(error) = comparison_error {
                return Err(error);
            }
            return Ok(());
        }
        result?;
        // An invalid comparison does not erase valid online/calibration history.
        if let Some(error) = comparison_error {
            return Err(error);
        }
        Ok(())
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
        if !self.appended_mixed_sources.is_empty() {
            value["appended_mixed_online_sources"] =
                serde_json::to_value(&self.appended_mixed_sources)
                    .expect("append sources serialize");
        }
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
pub(super) mod tests;
