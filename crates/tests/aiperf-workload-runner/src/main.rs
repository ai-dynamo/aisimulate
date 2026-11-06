// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independently compiled simulator-owned stdio workload proof.

mod dispatch;
#[allow(dead_code)]
mod model;
mod workload;
mod rendezvous;

use std::{io::{BufReader,Write},rc::Rc,sync::Arc};
use anyhow::{Context,Result};
use aiperf_clock_runtime::{cancellation::RunCancellation,clock::SimClock};
use aiperf_engine_kernel::{control_reader::{self,ControlAuthority,LiveGpuTelemetry},event_writer::{self,Sequencer,EventWriterHandle}};
use aiperf_protocol::runner_v2::*;

struct Progress { events:EventWriterHandle, seq:Arc<Sequencer>, id:String }
impl aiperf_timing::PhaseObserver for Progress {
    fn on_phase_start(&self,_:&aiperf_timing::PhaseConfig,stats:aiperf_timing::PhaseStats) { self.on_progress(stats); }
    fn on_progress(&self,stats:aiperf_timing::PhaseStats) {
        let kind = if stats.kind == aiperf_timing::PhaseKind::Warmup { PhaseKindV2::Warmup } else { PhaseKindV2::Profiling };
        self.events.publish_coalesced_with(&self.seq,|sequence| PhaseProgressV2 { protocol_version:2,event:RunEventKindV2::PhaseProgress,sequence,benchmark_id:self.id.clone(),phase_index:stats.phase_index as u32,phase_name:stats.phase_id.clone(),phase_kind:kind,revision:sequence,observed_at_ns:stats.requests_end_ns.or(stats.start_ns).unwrap_or(0),stats:PhaseProgressStatsV2 { requests_sent:stats.requests_sent,requests_completed:stats.requests_completed,..Default::default() } });
    }
    fn on_sending_complete(&self,stats:aiperf_timing::PhaseStats) { self.on_progress(stats); }
    fn on_phase_complete(&self,stats:aiperf_timing::PhaseStats,_:Option<aiperf_timing::PhaseBranchStats>) { self.on_progress(stats); }
}

fn main() {
    if let Err(error) = run() { eprintln!("{error:#}"); std::process::exit(1); }
}

fn run() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("capabilities") {
        println!("{}",serde_json::json!({"protocol_version":2,"runner":"aisim-aiperf-library-proof","status":"experimental-unverified","workload":"native_agentic_replay","sources":["inline_weka"],"tokenizer":"explicit_builtin","model":"aisim_core_fixed_1ms","control":["cancel"],"artifacts":"proof_report_not_product_aiperf_v2","per_token_ipc":false,"proof_instrumentation":"--proof-wait-for-cancel: bounded pause before first committed model deadline; never default"}));
        return Ok(());
    }
    let mut input = BufReader::new(std::io::stdin());
    let frame = framing::read_bounded_frame(&mut input,16*1024*1024)?.context("missing run request")?;
    let request = decode_run_request_v2(&frame)?;
    let run = request.run;
    let (events,writer) = event_writer::spawn(std::io::stdout(),SimClock::new(),event_writer::EVENT_WRITE_DEADLINE);
    let seq = Arc::new(Sequencer::new());
    events.enqueue_reliable_with(&seq,|sequence|RunEventV2::RunStarted(RunStartedV2 { protocol_version:2,event:RunEventKindV2::RunStarted,sequence,benchmark_id:run.benchmark_id.clone(),started_at_ns:0 })).map_err(|e|anyhow::anyhow!("writer: {e:?}"))?;
    let cancel = RunCancellation::new();
    let authority = ControlAuthority::new(cancel.clone(),LiveGpuTelemetry::unavailable());
    let control_cancel = cancel.clone();
    let control_authority = authority.clone();
    let control_events = events.clone();
    let control_seq = seq.clone();
    let id = run.benchmark_id.clone();
    // stdin is process-owned: a blocked reader ends with this executable, never
    // detaches a child process or retains model resources. Native loop owns framing.
    // One mutex serializes control refusal/cancellation with terminal cutoff.
    let control_failure = Arc::new(std::sync::Mutex::new((false,false)));
    let control_state = control_failure.clone();
    std::thread::spawn(move || {
        let result = control_reader::run_control_loop(input,control_authority,control_events,control_seq,id);
        let mut state = control_state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.1 {
            state.0 = !matches!(result,control_reader::ControlReaderOutcome::Eof);
            // Match native post-request EOF semantics: frontend disappearance
            // gracefully cancels an active run, but never relabels a terminal.
            control_cancel.cancel();
        }
    });
    let observer = Rc::new(Progress { events:events.clone(),seq:seq.clone(),id:run.benchmark_id.clone() });
    let proof_wait = std::env::args().any(|argument|argument == "--proof-wait-for-cancel");
    let mut result = workload::execute(&run,cancel.clone(),observer,proof_wait.then(||(events.clone(),seq.clone())));
    let mut metadata = std::collections::BTreeMap::from([("runner".into(),"aisim-owned-library-proof".into()),("report_contract".into(),"proof-only-not-aiperf-v2".into()),("aisim_source".into(),env!("AISIM_SOURCE_REVISION").into()),("aiperf_source".into(),env!("AIPERF_SOURCE_REVISION").into())]);
    metadata.insert("proof_wait_for_cancel".into(),proof_wait.to_string());
    let artifact = if let Ok(report) = &result {
        (|| -> Result<()> {
            let directory = aiperf_fsio::artifact_file::open_run_directory(&run.artifact_dir)?;
            aiperf_fsio::artifact_file::write_create_new_at(&directory,"aisim-library-proof.json",&serde_json::to_vec_pretty(report)?)?;
            Ok(())
        })()
    } else { Ok(()) };
    let artifact_failed = artifact.is_err();
    if let Err(error) = artifact { result = Err(error.context("committing proof artifact")); }
    else if result.is_ok() { metadata.insert("proof_artifact".into(),"aisim-library-proof.json".into()); }
    authority.begin_terminal();
    let malformed = {
        let mut state = control_failure.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.1 = true;
        state.0
    };
    let success = result.is_ok() && !malformed && !cancel.is_cancelled();
    let mut diagnostics = Vec::new();
    if result.is_err() {
        diagnostics.push(DiagnosticV2 { code:"proof_execution_failed".into(),message:"runner capability or execution rejected; inspect local stderr".into(),path:None });
        if let Err(error) = &result { eprintln!("{error:#}"); }
    }
    let terminal = RunTerminalV2 { protocol_version:2,event:RunEventKindV2::RunTerminal,sequence:0,benchmark_id:Some(run.benchmark_id),outcome:if malformed || artifact_failed {RunOutcomeV2::Failed} else if cancel.is_cancelled() {RunOutcomeV2::Cancelled} else if success {RunOutcomeV2::Succeeded} else {RunOutcomeV2::Failed},success,report_path:None,summary_observation:None,stage:if success {None} else if malformed {Some(FailureStageV2::Protocol)} else if artifact_failed {Some(FailureStageV2::Reporting)} else {Some(FailureStageV2::Execution)},errors:std::mem::take(&mut diagnostics),diagnostic_artifacts:Vec::new(),run_metadata:metadata };
    let outcome = events.finish(&seq,terminal);
    writer.join().map_err(|_|anyhow::anyhow!("event writer panicked"))?;
    anyhow::ensure!(matches!(outcome,event_writer::WriterOutcome::Completed),"event writer failed: {outcome:?}");
    std::io::stdout().flush()?;
    Ok(())
}
