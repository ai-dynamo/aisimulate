// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whole-iteration timing of a regression or learned forward-pass estimator.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};

use super::timing::{IterationMs, RankWork, TimingModel};
use crate::{
    EstimationMode, ForwardPassPerfModel, ForwardPassRegressionWorkloadKind, ForwardPassWorkerType,
    UNKNOWN_SIZE,
};

/// One rank's canonical query: `[num_prefill_requests, sum_prefill_tokens,
/// sum_prefill_kv_tokens, num_decode_requests, sum_decode_kv_tokens]`.
pub(crate) type RankQuery = [u32; 5];

/// The work kind of one rank's FPM record in a group iteration. A rank that
/// computes prefill and decode is reported with decode, as the decode share
/// of a mixed pass was.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum IterationKind {
    Prefill,
    Decode,
    Mixed,
}

impl IterationKind {
    /// The kind of `query`'s work, or `None` for an idle rank.
    pub(crate) fn of(query: &RankQuery) -> Option<Self> {
        match (query[1] > 0, query[3] > 0) {
            (false, false) => None,
            (true, false) => Some(Self::Prefill),
            (false, true) => Some(Self::Decode),
            (true, true) => Some(Self::Mixed),
        }
    }
}

/// The timing of a replay role whose estimator is a regression or learned
/// model (`fpm_regression`, `fpm_learned`): every pass of an attention-DP
/// group, mixed passes included, is priced with one canonical
/// `ForwardPassMetrics` query, one entry per rank, with the per-request detail
/// the estimator reads. Such an estimator was fitted to whole iterations, so
/// it never prices a separate phase ([`TimingModel::prices_phases`] is
/// `false`) and records no evidence.
pub struct FittedIterationTiming {
    model: Arc<ForwardPassPerfModel>,
    worker_type: ForwardPassWorkerType,
    /// The identity's attention DP, or `UNKNOWN_SIZE`.
    attention_dp: u32,
    /// Tokens each decode request verifies: one plus the fitted `nextn`.
    verify_width: usize,
    /// Whether the estimator reads per-request detail, which the cache key
    /// does not hold.
    detail: bool,
    /// Priced iterations keyed by the complete ordered rank list; absent when
    /// the estimator reads per-request detail.
    cache: Option<quick_cache::sync::Cache<Vec<RankQuery>, f64>>,
}

impl FittedIterationTiming {
    /// The timing of the replay role `model` was built for, wrapping the
    /// model [`ForwardPassPerfModel::best_available`] returned. The selected
    /// estimator must be the requested regression or learned model, able to
    /// price the role's execution, with every workload store the role needs
    /// ready. The role is the resolved identity's `worker_type`, and its
    /// stores are checked at the identity's attention DP;
    /// [`Self::require_replay_role`] checks them again at the engine's width.
    pub fn from_model(model: Arc<ForwardPassPerfModel>) -> Result<Self> {
        let provenance = model
            .provenance()
            .context("canonical estimator omitted construction provenance")?;
        let mode = provenance.selected_estimation_mode;
        ensure!(
            matches!(
                mode,
                EstimationMode::FpmRegression | EstimationMode::FpmLearned
            ),
            "canonical {mode:?} estimator has no native engine"
        );
        if provenance.requested_estimation_mode != mode {
            let failures = provenance.selection_failures.join("; ");
            return Err(anyhow!(
                "no native estimator is available for replay ({failures}); replay does not fall back to an \
                 untrained fpm_regression model"
            ));
        }
        let resolved = &provenance.config;
        require_fitted_execution(
            mode,
            resolved.backend.as_str(),
            resolved.nextn > 0,
            resolved
                .speculation
                .as_ref()
                .map(|speculation| speculation.kind.as_str()),
            resolved.decoder_replay,
        )?;
        let worker_type = resolved.worker_type;
        let attention_dp = resolved.attention_dp;
        let verify_width = resolved.nextn as usize + 1;
        require_iteration_stores(&model, worker_type, attention_dp)?;
        let detail = model.reads_request_detail();
        Ok(Self {
            model,
            worker_type,
            attention_dp,
            verify_width,
            detail,
            cache: (!detail).then(|| quick_cache::sync::Cache::new(256)),
        })
    }

    /// Check the replay role this timing serves: the engine's attention-DP
    /// width, which must match the identity's unless that is unknown, the
    /// stores at that width, and a unit `decode_speedup_ratio`.
    pub fn require_replay_role(&self, attention_dp: u32, decode_speedup_ratio: f64) -> Result<()> {
        ensure!(
            self.attention_dp == UNKNOWN_SIZE || self.attention_dp == attention_dp,
            "the fitted model's attention_dp={} does not match the engine's attention DP {attention_dp}",
            self.attention_dp
        );
        require_iteration_stores(&self.model, self.worker_type, attention_dp)?;
        require_unit_decode_speedup(&self.model, decode_speedup_ratio)
    }

    /// Whether priced iterations are cached: only when the estimator reads no
    /// per-request detail.
    #[cfg(all(test, feature = "python"))]
    pub(crate) fn caches_iterations(&self) -> bool {
        self.cache.is_some()
    }
}

impl TimingModel for FittedIterationTiming {
    fn predict_prefill_ms(
        &self,
        _batch_size: usize,
        _mean_isl: usize,
        _mean_prefix: usize,
    ) -> Result<f64> {
        Err(phase_query())
    }

    fn predict_decode_ms(
        &self,
        _batch_size: usize,
        _active_kv_tokens: usize,
        _mean_context_length: usize,
        _total_kv_tokens: usize,
    ) -> Result<f64> {
        Err(phase_query())
    }

    /// Price an attention-DP group pass with one canonical
    /// `ForwardPassMetrics` query, one entry per rank. A decode pass's
    /// `verify_width` must be the fitted `nextn + 1`.
    fn predict_iteration_ms(&self, ranks: &[RankWork]) -> Result<Option<IterationMs>> {
        for rank in ranks {
            if let Some(decode) = &rank.decode
                && decode.batch_size > 0
            {
                ensure!(
                    decode.verify_width == self.verify_width,
                    "rank {} verifies {} tokens per decode request; the model was fitted at nextn {}",
                    rank.dp_rank,
                    decode.verify_width,
                    self.verify_width - 1
                );
            }
        }
        let queries = ranks
            .iter()
            .map(fitted_rank_query)
            .collect::<Result<Vec<_>>>()?;
        if queries
            .iter()
            .all(|query| IterationKind::of(query).is_none())
        {
            // An iteration without model work computes nothing.
            return Ok(Some(IterationMs::new(0.0)));
        }
        if let Some(total_ms) = self.cache.as_ref().and_then(|cache| cache.get(&queries)) {
            return Ok(Some(IterationMs::new(total_ms)));
        }
        let total_ms = price_fitted_iteration(&self.model, &queries, ranks, self.detail)?;
        if let Some(cache) = &self.cache {
            cache.insert(queries, total_ms);
        }
        Ok(Some(IterationMs::new(total_ms)))
    }

    fn prices_phases(&self) -> bool {
        false
    }
}

fn phase_query() -> anyhow::Error {
    anyhow!(
        "a regression or learned estimator prices only whole iterations; replay asked it for a separate phase"
    )
}

/// Price one iteration with a regression or learned estimator: one plain
/// estimate of every rank. The role's stores were checked at
/// construction, so a missing estimate is an error, never another
/// estimator. Latency-only: no evidence is recorded.
fn price_fitted_iteration(
    model: &ForwardPassPerfModel,
    queries: &[RankQuery],
    ranks: &[RankWork],
    detail: bool,
) -> Result<f64> {
    let metrics = fitted_metrics(queries, ranks, detail)?;
    model
        .estimate_forward_pass_time_ms(&metrics)
        .context("AIC iteration prediction failed")?
        .with_context(|| {
            format!("the regression or learned model has no iteration estimate for {metrics:?}")
        })
}

/// One rank's canonical query as `ForwardPassMetrics` totals.
pub(crate) fn rank_metrics(query: &RankQuery) -> crate::ForwardPassMetrics {
    let [
        num_prefill_requests,
        sum_prefill_tokens,
        sum_prefill_kv_tokens,
        num_decode_requests,
        sum_decode_kv_tokens,
    ] = *query;
    crate::ForwardPassMetrics {
        scheduled_requests: crate::ScheduledRequestMetrics {
            num_prefill_requests,
            sum_prefill_tokens,
            sum_prefill_kv_tokens,
            num_decode_requests,
            sum_decode_kv_tokens,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// The group's metrics for a regression or learned estimator, one per
/// rank in DP-rank order, idle ranks included. Each carries its rank's
/// position as `dp_rank`, as an attention-DP group's telemetry does; a
/// learned model's mixed composition validates complete rank groups.
pub(crate) fn fitted_metrics(
    queries: &[RankQuery],
    ranks: &[RankWork],
    detail: bool,
) -> Result<Vec<crate::ForwardPassMetrics>> {
    queries
        .iter()
        .zip(ranks)
        .enumerate()
        .map(|(index, (query, rank))| {
            let mut metrics = if detail {
                detailed_rank_metrics(query, rank)?
            } else {
                rank_metrics(query)
            };
            metrics.dp_rank = u32::try_from(index).context("attention-DP rank overflows u32")?;
            Ok(metrics)
        })
        .collect()
}

/// One rank's query for a regression or learned estimator: its scheduled
/// totals as the `ForwardPassMetrics` telemetry such models are fitted to
/// reports them. Dynamo's vLLM telemetry sums each decode request's
/// computed tokens, so decode KV is the requests' past KV, their context
/// lengths minus one current token each, as whole-model FPM reads it but
/// without the scheduler's KV clamp. A decode request counts once,
/// however many tokens it verifies. No other query prices the pass, so
/// totals beyond the 32-bit counters are an error.
pub(crate) fn fitted_rank_query(rank: &RankWork) -> Result<RankQuery> {
    let mut query = [0; 5];
    if let Some(prefill) = &rank.prefill
        && prefill.totals.batch_size > 0
        && prefill.totals.new_tokens > 0
    {
        let totals = prefill.totals;
        query[0] = checked_u32(totals.batch_size, "prefill batch size")?;
        query[1] = checked_u32(totals.new_tokens, "total prefill tokens")?;
        query[2] = checked_u32(totals.prefix_tokens, "total prefill KV tokens")?;
    }
    if let Some(decode) = &rank.decode
        && decode.batch_size > 0
    {
        let past_kv_tokens = decode
            .active_kv_tokens
            .checked_sub(decode.batch_size)
            .context("active decode tokens must include one current token per request")?;
        query[3] = checked_u32(decode.batch_size, "decode batch size")?;
        query[4] = checked_u32(past_kv_tokens, "total decode past KV tokens")?;
    }
    Ok(query)
}

/// `query`'s metrics with the per-request detail of `rank`'s work in it:
/// each prefill request's new tokens and prefix, and each decode
/// request's verification width (one, or one plus its drafts) and past KV
/// (its context excluding the current token), as `extend_lengths` /
/// `past_kv_lengths`; and the population variances of the prefill
/// requests' full prompt lengths, as Dynamo's vLLM telemetry and the
/// canonical field report them whatever chunk of the prompt a pass
/// computes, and of the decode requests' past KV, the totals' decode KV
/// convention.
pub(crate) fn detailed_rank_metrics(
    query: &RankQuery,
    rank: &RankWork,
) -> Result<crate::ForwardPassMetrics> {
    fn variance(values: impl Iterator<Item = usize> + Clone) -> f64 {
        let count = values.clone().count() as f64;
        let mean = values.clone().map(|value| value as f64).sum::<f64>() / count;
        values
            .map(|value| (value as f64 - mean).powi(2))
            .sum::<f64>()
            / count
    }
    let mut metrics = rank_metrics(query);
    let scheduled = &mut metrics.scheduled_requests;
    let (mut extend, mut past) = (Vec::new(), Vec::new());
    if query[1] > 0 {
        let prefill = rank
            .prefill
            .as_ref()
            .context("prefill query without prefill work")?;
        ensure!(
            prefill.requests.len() == prefill.totals.batch_size,
            "rank {} reported {} prefill requests for a batch of {}",
            rank.dp_rank,
            prefill.requests.len(),
            prefill.totals.batch_size
        );
        for &(new_tokens, prefix_tokens, _) in &prefill.requests {
            extend.push(new_tokens as u64);
            past.push(prefix_tokens as u64);
        }
        scheduled.var_prefill_length = variance(
            prefill
                .requests
                .iter()
                .map(|&(_, _, prompt_len)| prompt_len),
        );
    }
    if query[3] > 0 {
        let decode = rank
            .decode
            .as_ref()
            .context("decode query without decode work")?;
        ensure!(
            decode.lengths.len() == decode.batch_size
                && decode.widths.len() == decode.batch_size
                && decode.lengths.iter().all(|length| *length > 0),
            "rank {} reported {} decode context lengths and {} widths for a batch of {}, each length including its current token",
            rank.dp_rank,
            decode.lengths.len(),
            decode.widths.len(),
            decode.batch_size
        );
        for (&length, &width) in decode.lengths.iter().zip(&decode.widths) {
            extend.push(width as u64);
            past.push(length as u64 - 1);
        }
        scheduled.var_decode_kv_tokens = variance(decode.lengths.iter().map(|length| length - 1));
    }
    if !extend.is_empty() {
        scheduled.extend_lengths = Some(extend);
        scheduled.past_kv_lengths = Some(past);
    }
    Ok(metrics)
}

/// Require execution a regression or learned `mode` estimator can price
/// through its iteration query: no speculation scheme, `nextn > 0` only for a
/// vLLM regression (which pins the `nextn` it was fitted at), and no bounded
/// decoder replay.
pub(crate) fn require_fitted_execution(
    mode: EstimationMode,
    backend: &str,
    nextn_positive: bool,
    speculation_kind: Option<&str>,
    decoder_replay: bool,
) -> Result<()> {
    let nextn_priced =
        !nextn_positive || (mode == EstimationMode::FpmRegression && backend == "vllm");
    let mode = wire_name(mode);
    ensure!(
        nextn_priced && speculation_kind.is_none_or(|kind| kind == "none"),
        "{mode} replay does not support speculative decoding here: speculation schemes are \
         rejected, and nextn > 0 is priced only by a vLLM fpm_regression snapshot, at the \
         nextn it was fitted at"
    );
    ensure!(
        !decoder_replay,
        "{mode} replay does not support bounded decoder replay"
    );
    Ok(())
}

/// Require the workload stores a regression or learned estimator needs to
/// price every iteration replay schedules for a `worker_type` role with
/// `attention_dp` ranks: pure prefill for a prefill worker, pure decode for a
/// decode worker, and both for an aggregated worker. An aggregated rank also
/// computes prefill and decode together when its backend's scheduler does
/// (see [`schedules_locally_mixed`]), and its ranks compute different kinds of
/// work when attention DP is known to be above one. Unknown attention DP adds
/// no store here; the role checks again with its engine's width.
pub(crate) fn require_iteration_stores(
    model: &ForwardPassPerfModel,
    worker_type: ForwardPassWorkerType,
    attention_dp: u32,
) -> Result<()> {
    use ForwardPassRegressionWorkloadKind::*;
    let provenance = model
        .provenance()
        .context("canonical estimator omitted construction provenance")?;
    let mut required = match worker_type {
        ForwardPassWorkerType::Prefill => vec![PurePrefill],
        ForwardPassWorkerType::Decode => vec![PureDecode],
        ForwardPassWorkerType::Aggregated => vec![PurePrefill, PureDecode],
    };
    if worker_type == ForwardPassWorkerType::Aggregated {
        if schedules_locally_mixed(&provenance.config.backend) {
            required.push(ContainsLocallyMixed);
        }
        if attention_dp > 1 && attention_dp != UNKNOWN_SIZE {
            required.push(CrossRankAggregated);
        }
    }
    let report = model.readiness_report();
    let missing = report
        .stores
        .iter()
        .filter(|store| required.contains(&store.workload_kind) && !store.ready)
        .map(|store| match store.not_ready_reason {
            Some(reason) => format!("{} ({})", wire_name(store.workload_kind), wire_name(reason)),
            None => wire_name(store.workload_kind),
        })
        .collect::<Vec<_>>();
    ensure!(
        missing.is_empty(),
        "the {} estimator cannot price every iteration of a {} worker: workload store(s) {} not ready; \
         tune it with observed forward passes of that work, or restore a snapshot calibrated on it, before replay",
        wire_name(provenance.selected_estimation_mode),
        wire_name(worker_type),
        missing.join(", ")
    );
    Ok(())
}

/// Whether `backend`'s replay scheduler computes prefill and decode in one
/// rank pass. vLLM, and TensorRT-LLM on the shared vLLM core, schedule chunked
/// prefill beside running decodes. SGLang runs a formed prefill batch alone
/// (prefill and decode share a forward only with `--enable-mixed-chunk`,
/// which is not modeled), so its ranks never compute locally mixed work.
fn schedules_locally_mixed(backend: &crate::BackendKind) -> bool {
    !matches!(backend, crate::BackendKind::Sglang)
}

/// Require the unit `decode_speedup_ratio` a regression or learned role needs.
/// Replay prices a group pass with one iteration query only at a unit decode
/// speedup; at any other, each rank prices its own phases, which such an
/// estimator never does, so the role would fail at its first pass.
fn require_unit_decode_speedup(
    model: &ForwardPassPerfModel,
    decode_speedup_ratio: f64,
) -> Result<()> {
    ensure!(
        decode_speedup_ratio == 1.0,
        "{} replay requires decode_speedup_ratio=1.0, got {decode_speedup_ratio}: it prices only whole \
         iterations, and replay prices a pass as one iteration only at a unit decode speedup",
        model.provenance().map_or_else(
            || "regression or learned".to_owned(),
            |provenance| wire_name(provenance.selected_estimation_mode)
        )
    );
    Ok(())
}

/// The snake-case wire spelling of a unit enum value.
pub(crate) fn wire_name(value: impl serde::Serialize) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(name)) => name,
        other => format!("{other:?}"),
    }
}

pub(crate) fn checked_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{name} {value} exceeds AIC's u32 limit"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{DecodeWork, PrefillTotals, PrefillWork};

    /// A learned model, fitted at `attention_dp`: pure prefill 10 ms, pure
    /// decode 5 ms, a rank pass computing both 7 ms, and a group whose ranks
    /// compute different work 8 ms.
    fn forest_timing(attention_dp: u32) -> FittedIterationTiming {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("forest.json");
        let artifact = serde_json::json!({
            "schema": "aic_fpm_learned_forward_perf", "schema_version": 1,
            "worker_type": "aggregated", "target": "ms",
            "features": ["num_decode_requests"], "metadata": {},
            "stores": {
                "pure_prefill": {"baseline": 10.0, "trees": []},
                "pure_decode": {"baseline": 5.0, "trees": []},
                "contains_locally_mixed": {"baseline": 7.0, "trees": []},
                "cross_rank_aggregated": {"baseline": 8.0, "trees": []},
            },
        });
        std::fs::write(&path, artifact.to_string()).unwrap();
        let mut config = crate::ForwardPassPerfModelConfig::new(
            "test/model",
            "test-gpu",
            crate::BackendKind::Vllm,
            ForwardPassWorkerType::Aggregated,
        );
        config.backend_version = Some("test-version".into());
        config.attention_dp = attention_dp;
        config.estimation_mode = EstimationMode::FpmLearned;
        config.estimator_config.fpm_learned.artifact_path = Some(path);
        config.estimator_config.fpm_learned.adaptation = crate::LearnedAdaptation::Frozen;
        config.estimator_config.correction.enabled = false;
        let model = ForwardPassPerfModel::best_available(config).unwrap();
        FittedIterationTiming::from_model(Arc::new(model)).unwrap()
    }

    /// One rank computing a prefill and two decodes that each verify
    /// `verify_width` tokens.
    fn mixed_rank(verify_width: usize) -> RankWork {
        RankWork {
            dp_rank: 0,
            prefill: Some(PrefillWork {
                totals: PrefillTotals {
                    batch_size: 1,
                    new_tokens: 96,
                    prefix_tokens: 16,
                },
                requests: vec![(96, 16, 112)],
            }),
            decode: Some(DecodeWork {
                batch_size: 2,
                active_kv_tokens: 800,
                mean_context_length: 400,
                kv_bound_tokens: 800,
                lengths: vec![300, 500],
                widths: vec![verify_width; 2],
                verify_width,
            }),
        }
    }

    #[test]
    fn a_mixed_pass_is_one_query_of_the_locally_mixed_store() {
        let timing = forest_timing(1);
        // One query of the mixed store, not a prefill pass plus a decode pass.
        let iteration = timing
            .predict_iteration_ms(&[mixed_rank(1)])
            .unwrap()
            .unwrap();
        assert_eq!(iteration.total_ms, 7.0);
        assert!(timing.predict_prefill_ms(1, 112, 16).is_err());
    }

    #[test]
    fn a_replay_role_must_match_a_known_fitted_attention_dp() {
        forest_timing(2).require_replay_role(2, 1.0).unwrap();
        let error = forest_timing(2)
            .require_replay_role(1, 1.0)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("attention_dp=2 does not match the engine's attention DP 1"),
            "{error}"
        );
        forest_timing(UNKNOWN_SIZE)
            .require_replay_role(1, 1.0)
            .unwrap();
    }

    #[test]
    fn a_decode_pass_must_verify_the_fitted_nextn_plus_one_tokens() {
        // Fitted at nextn 0, each decode request verifies one token.
        let error = forest_timing(1)
            .predict_iteration_ms(&[mixed_rank(2)])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("verifies 2 tokens per decode request; the model was fitted at nextn 0"),
            "{error}"
        );
    }
}
