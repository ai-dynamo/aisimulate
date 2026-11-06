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
pub mod participant_adapter;
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
        run,
        cancellation,
        observer,
        None,
        std::rc::Rc::new(aiperf_rng::identity::OsIdentitySource),
        None,
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
        Some((factory, admission_capacity)),
        std::rc::Rc::new(aiperf_rng::identity::OsIdentitySource),
        None,
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
        run,
        cancellation,
        observer,
        model_factory,
        std::rc::Rc::new(aiperf_rng::identity::DeterministicIdentitySource::new(
            identity_seed,
        )),
        None,
    )
}

/// Execute native workload semantics against a simulator-owned deployment.
///
/// The deployment retains physical placement, prefill/decode handoff, KV transfer
/// and its original collector. Fine-grained calls remain inside this process.
pub fn execute_with_deployment(
    run: &aiperf_config::model::BenchmarkRun,
    cancellation: aiperf_clock_runtime::cancellation::RunCancellation,
    observer: std::rc::Rc<dyn aiperf_timing::PhaseObserver>,
    deployment: Box<dyn aisimulate_core::replay::loadgen::SteppableReplay>,
    admission_capacity: usize,
    identity_seed: [u8; 32],
) -> anyhow::Result<serde_json::Value> {
    workload::execute(
        run,
        cancellation,
        observer,
        None,
        std::rc::Rc::new(aiperf_rng::identity::DeterministicIdentitySource::new(
            identity_seed,
        )),
        Some((deployment, admission_capacity)),
    )
}
