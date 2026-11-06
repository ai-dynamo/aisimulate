// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Simulator-owned in-process composition of AIPerf's reusable workload libraries.
//!
//! No executable, subprocess, transport, or per-token IPC is needed by this API.
//! Arrival, AgentX reconstruction, branching, stopping and reporting remain
//! AIPerf-owned; this package supplies the actual AISim Engine binding.

mod dispatch;
#[allow(dead_code)]
mod model;
mod rendezvous;
mod workload;

/// Execute an accepted native run directly inside the simulator process.
///
/// Returns the native phase report and exact request/body/model evidence in
/// this proof package's explicitly non-product artifact envelope. The caller
/// owns artifact publication. No stdout, stdin, or child process is opened.
pub fn execute(
    run: &aiperf_config::model::BenchmarkRun,
    cancellation: aiperf_clock_runtime::cancellation::RunCancellation,
    observer: std::rc::Rc<dyn aiperf_timing::PhaseObserver>,
) -> anyhow::Result<serde_json::Value> {
    workload::execute(run,cancellation,observer,None)
}

/// Opt-in executable-boundary instrumentation, separate from ordinary execution.
///
/// The stdio proof uses this only when explicitly asked to suspend at a
/// committed model deadline until a framed cancellation arrives.
#[doc(hidden)]
pub fn execute_with_control_rendezvous(
    run: &aiperf_config::model::BenchmarkRun,
    cancellation: aiperf_clock_runtime::cancellation::RunCancellation,
    observer: std::rc::Rc<dyn aiperf_timing::PhaseObserver>,
    events: aiperf_engine_kernel::event_writer::EventWriterHandle,
    sequencer: std::sync::Arc<aiperf_engine_kernel::event_writer::Sequencer>,
) -> anyhow::Result<serde_json::Value> {
    workload::execute(run,cancellation,observer,Some((events,sequencer)))
}
