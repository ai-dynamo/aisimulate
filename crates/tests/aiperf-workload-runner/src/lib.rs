// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Simulator-owned in-process composition of AIPerf's reusable workload libraries.
//!
//! No executable, subprocess, transport, or per-token IPC is needed by this API.
//! Arrival, AgentX reconstruction, branching, stopping and reporting remain
//! AIPerf-owned; this package supplies the actual AISim Engine binding.

mod dispatch;
mod graph;
/// Actual simulator-owned Engine binding, exposed for cross-owner semantic proofs.
pub mod model;
mod rendezvous;
mod reporting;
mod scheduled;
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
    workload::execute(
        run, cancellation, observer, None, None,
        std::rc::Rc::new(aiperf_rng::identity::OsIdentitySource),
    )
}

/// Execute the shared native workload against a caller-prepared AISim factory.
///
/// This function never substitutes fixed timing. GPU predictor callers must
/// construct and validate calibrated timing through the canonical AISim API,
/// then pass the corresponding native factory. Admission capacity must match
/// that factory's scheduler capacity. Artifact publication remains caller-owned.
pub fn execute_with_engine_factory(
    run: &aiperf_config::model::BenchmarkRun,
    cancellation: aiperf_clock_runtime::cancellation::RunCancellation,
    observer: std::rc::Rc<dyn aiperf_timing::PhaseObserver>,
    factory: &aisimulate_core::engine::EngineFactory,
    admission_capacity: usize,
) -> anyhow::Result<serde_json::Value> {
    workload::execute(
        run,
        cancellation,
        observer,
        None,
        Some((factory, admission_capacity)),
        std::rc::Rc::new(aiperf_rng::identity::OsIdentitySource),
    )
}

/// Execute with reproducible run-local identities for a controlled comparison.
///
/// The old product must receive the same identity seed before materialization.
/// This seed is separate from dataset sampling and does not change normal APIs'
/// OS-random defaults. An optional model factory preserves the caller's timing
/// and admission selection exactly as in [`execute_with_engine_factory`].
pub fn execute_with_identity_seed(
    run: &aiperf_config::model::BenchmarkRun,
    cancellation: aiperf_clock_runtime::cancellation::RunCancellation,
    observer: std::rc::Rc<dyn aiperf_timing::PhaseObserver>,
    identity_seed: [u8; 32],
    model_factory: Option<(&aisimulate_core::engine::EngineFactory, usize)>,
) -> anyhow::Result<serde_json::Value> {
    workload::execute(
        run, cancellation, observer, None, model_factory,
        std::rc::Rc::new(aiperf_rng::identity::DeterministicIdentitySource::new(identity_seed)),
    )
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
    workload::execute(
        run, cancellation, observer, Some((events, sequencer)), None,
        std::rc::Rc::new(aiperf_rng::identity::OsIdentitySource),
    )
}
