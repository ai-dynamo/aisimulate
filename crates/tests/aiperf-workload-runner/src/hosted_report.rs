// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Focused normal native artifact finalizer for the AISim hosted runner.
//!
//! The finalizer consumes the exact typed results the shared hosted run
//! retained ([`HostedTypedReport`]) and produces the normal AIPerf artifacts:
//!
//! - the authoritative native-v2 report, stamped once with the caller's
//!   coordinator metadata and pair facts and written through
//!   [`aiperf_report_io::report::write_finalized_report_json`];
//! - the GenAI-Perf v1.7 summary, written through its owning
//!   [`GenaiPerfV1Exporter`] with the normal [`export_for_run`] policy.
//!
//! Coverage is evaluated from a live [`MetricsAccumulator`] rebuilt from each
//! phase's own captured `RecordIngest` rows, never from a lossy summary or
//! hand-built report JSON. The caller retains the typed [`HostedTypedReport`]
//! and supplies the exact executable metadata and pair facts; this module
//! fabricates neither.

use std::path::Path;

use aiperf_config::model::BenchmarkRun;
use aiperf_engine_input::execution_input::{
    attach_submission_inputs, export_for_run, scenario_locks_satisfied,
};
use aiperf_engine_kernel::profile_metric_coverage::profile_metric_coverage_failures;
use aiperf_export::genai_perf::GenaiPerfV1Exporter;
use aiperf_metrics::MetricsAccumulator;
use aiperf_metrics_model::ProfileMetricDurationCoverage;
use aiperf_metrics_model::{NativeReport, RunOutcome};
use aiperf_report_io::report::write_finalized_report_json;
use aiperf_report_model::exporter::Exporter;
use aiperf_report_model::report_output::{
    ReportError, ReportPairRunFacts, ReportRunInfo, ReportRunMetadata, ReportSummary,
};
use aiperf_timing::PhaseKind;
use anyhow::{Context, Result, anyhow, ensure};

use crate::participant_adapter::{HostedPhaseRecords, HostedTypedReport};

fn coverage_accumulator(
    config: aiperf_metrics_model::MetricsConfig,
    phases: &[HostedPhaseRecords],
) -> MetricsAccumulator {
    let mut accumulator = MetricsAccumulator::with_config(config);
    for phase in phases {
        for (_, record) in &phase.records {
            // Captures retain phase-local request slots. Coverage merges their
            // observations, not their positional storage: each phase may own
            // slot zero. Preserve the original record identity in the evidence.
            accumulator.process_record_unindexed(record);
        }
    }
    accumulator
}

/// Finalized normal output plus the coverage evidence that produced it.
pub struct FinalizedHostedReport {
    /// Authoritative native-v2 report after caller-supplied finalization.
    pub native_report: NativeReport,
    /// Per-profiling-phase coverage evaluated from the live accumulator.
    pub coverage: Vec<ProfileMetricDurationCoverage>,
    /// Coverage-plan validation errors; also appended to the report's errors.
    /// They never prevent the normal artifacts from being written.
    pub coverage_errors: Vec<ReportError>,
    /// Per-phase failures below the scenario's required ratio, or `None` when
    /// every phase passed (including when no scenario check applied).
    pub coverage_failures: Option<Vec<String>>,
}

/// Build the normal native-v2 report, evaluate scenario coverage, and write
/// both normal artifacts under `artifact_dir`.
///
/// `metadata` and `facts` are the caller's exact coordinator/pair facts: this
/// function stamps them once and never derives or fabricates parity evidence.
/// The native report is written at the run's own artifact name, so an authored
/// export stem keeps its normal stem-relative identity.
pub fn finalize_hosted_run_report(
    run: &BenchmarkRun,
    typed: &HostedTypedReport,
    metadata: ReportRunMetadata,
    facts: ReportPairRunFacts,
    artifact_dir: &Path,
) -> Result<FinalizedHostedReport> {
    // Phased reports are strict per-phase, never cumulative. Select the unique
    // profiling phase explicitly; multiple profiling entries are refused
    // rather than silently aggregated by this finalizer.
    let mut profiling = typed
        .phased
        .reports
        .iter()
        .filter(|phase| phase.kind == PhaseKind::Profiling);
    let finalized = profiling
        .next()
        .ok_or_else(|| anyhow!("hosted phased report has no profiling phase report"))?;
    ensure!(
        profiling.next().is_none(),
        "hosted phased report has multiple profiling phase reports; aggregate with the shared owner instead of assuming one"
    );

    let warmup = typed
        .phased
        .reports
        .iter()
        .find(|phase| phase.kind == PhaseKind::Warmup)
        .map(|phase| phase.report.native_metrics.clone());

    let was_cancelled = typed.phased.phases.iter().any(|phase| phase.was_cancelled);
    let phase_starts: Vec<Option<i64>> = typed
        .phased
        .phases
        .iter()
        .map(|phase| phase.start_ns)
        .collect();

    // Rebuild the live accumulator from the exact typed records captured by
    // each phase's own NativeMetricsObserver. The capture sink already stamped
    // the engine-recorded phase origin; nothing is re-derived here.
    let accumulator = coverage_accumulator(typed.metrics_config.clone(), &typed.records);

    // The coverage owner reads the scenario's required ratio; no ratio is
    // hardcoded here.
    let (coverage, coverage_errors) = typed.coverage_plan.evaluate(
        &accumulator,
        run.cfg.scenario.as_deref(),
        &phase_starts,
        was_cancelled,
    );
    let coverage_failures = profile_metric_coverage_failures(&coverage);

    let mut native_report = NativeReport::from_outcome(
        &finalized.report.native_metrics,
        &RunOutcome {
            run: ReportRunInfo {
                mode: Some("aisimulate:hosted".into()),
                model: run
                    .cfg
                    .models
                    .as_ref()
                    .and_then(|models| models.items.first())
                    .map(|model| model.name.clone()),
                ..Default::default()
            },
            summary: ReportSummary {
                was_cancelled,
                ..ReportSummary::default()
            },
            warmup,
            ..RunOutcome::default()
        },
    );
    native_report.profile_metric_coverage = coverage.clone();
    native_report.errors.extend(coverage_errors.iter().cloned());

    // The normal export policy is derived from the typed run exactly once; it
    // already carries the static scenario verdict, projected input config and
    // public dataset metadata, so no second config is constructed here.
    let mut export = export_for_run(run).context("deriving normal export policy")?;
    // Bypass-free verdict: the normal submission is only comparable when the
    // locks hold without `unsafe_override`. Invalidate the owner-carried static
    // verdict rather than inventing a replacement submission block.
    attach_submission_inputs(
        &mut export,
        run.cfg.agentx_fast,
        scenario_locks_satisfied(&run.cfg),
        &coverage,
        coverage_failures.is_some(),
    )?;
    if !coverage_errors.is_empty() {
        export
            .genai_perf
            .envelope
            .runtime_submission_invalid_reasons
            .push("profile_metric_coverage_validation_failed".into());
    }

    let report = native_report
        .finalize_run(metadata, facts)
        .context("finalizing native report run metadata")?;

    let report_path = artifact_dir.join(aiperf_config::model::artifacts::report_artifact_name(
        run.cfg.artifacts.as_ref(),
    ));
    write_finalized_report_json(&report, &report_path)
        .with_context(|| format!("writing native report {}", report_path.display()))?;
    GenaiPerfV1Exporter
        .export(&report, artifact_dir, &export)
        .context("writing GenAI-Perf v1 summary")?;

    Ok(FinalizedHostedReport {
        native_report: report,
        coverage,
        coverage_errors,
        coverage_failures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiperf_metrics_model::{Phase, RecordIngest};

    #[test]
    fn coverage_accepts_reused_phase_local_request_slots() {
        let phases = [
            (Phase::Warmup, "warmup", None, 0, 1_000_000_000),
            (
                Phase::Profiling,
                "profiling",
                Some(0),
                2_000_000_000,
                902_000_000_000,
            ),
        ]
        .into_iter()
        .enumerate()
        .map(|(phase_index, (phase, name, authored, start, end))| {
            let mut record = RecordIngest::minimal(start, end, phase);
            record.request_index = Some(0);
            record.phase_index = authored;
            record.first_token_ns = Some(end);
            record.first_output_token_ns = Some(end);
            record.tokens.input = Some(8);
            record.tokens.output = Some(1);
            HostedPhaseRecords {
                phase,
                phase_index,
                phase_name: name.into(),
                phase_kind: name,
                records: vec![(uuid::Uuid::from_u128(phase_index as u128 + 1), record)],
            }
        })
        .collect::<Vec<_>>();
        let accumulator = coverage_accumulator(Default::default(), &phases);
        let ctx = aiperf_metrics_model::window::ExportContext {
            start_ns: Some(2_000_000_000),
            end_ns: None,
            phase: Some(Phase::Profiling),
            phase_index: Some(0),
        };
        let coverage = accumulator
            .profile_metric_duration_coverage(&ctx, 900.0, 0.95)
            .expect("profiling coverage");
        assert!(coverage.passed());
        assert_eq!(phases[0].records[0].1.request_index, Some(0));
        assert_eq!(phases[1].records[0].1.request_index, Some(0));
    }
}
