// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit proof-only external-control rendezvous; never enabled by default.

use crate::model::ModelExecution;
use aiperf_clock_runtime::{
    cancellation::RunCancellation,
    clock::{
        SimClock,
        driver::{SimDriveError, SimEventSource, SimStep},
    },
};
use aiperf_engine_kernel::event_writer::{EventWriterHandle, Sequencer};
use aiperf_protocol::runner_v2::{
    MetricsSnapshotV2, RunEventKindV2, RunEventV2, SnapshotCoverageV2, SnapshotScopeV2,
};
use std::{
    cell::Cell,
    rc::Rc,
    sync::{Arc, mpsc},
    time::Duration,
};

pub struct Rendezvous {
    pub model: Rc<ModelExecution>,
    pub clock: Rc<SimClock>,
    pub cancellation: RunCancellation,
    pub events: EventWriterHandle,
    pub sequencer: Arc<Sequencer>,
    pub benchmark_id: String,
    pub entered: Cell<bool>,
    pub resume_application: Cell<bool>,
}

impl SimEventSource for Rendezvous {
    fn next_event_ns(&self) -> Result<Option<i64>, SimDriveError> {
        let next = self.model.next_event_ns()?;
        if let Some(deadline) = next
            && deadline > self.clock.now_ns()
            && !self.entered.replace(true)
        {
            let now = self.clock.now_ns();
            self.events
                .enqueue_reliable_with(&self.sequencer, |sequence| {
                    RunEventV2::MetricsSnapshot(MetricsSnapshotV2 {
                        protocol_version: 2,
                        event: RunEventKindV2::MetricsSnapshot,
                        sequence,
                        benchmark_id: self.benchmark_id.clone(),
                        scope: SnapshotScopeV2::Run,
                        revision: sequence,
                        observed_at_ns: now,
                        coverage: SnapshotCoverageV2 {
                            worker_groups_observed: 1,
                            worker_groups_expected: 1,
                            stale_worker_groups: Vec::new(),
                        },
                        metrics: std::collections::BTreeMap::from([
                            ("proof_rendezvous_waiting".into(), Some(1.0)),
                            (
                                "proof_pending_model_deadline_ns".into(),
                                Some(deadline as f64),
                            ),
                        ]),
                    })
                })
                .map_err(|error| {
                    SimDriveError::EventSource(format!("rendezvous event: {error:?}"))
                })?;
            // The virtual driver is deliberately paused BEFORE advancing a
            // committed pass. Only the proof flag admits this real IPC wait.
            // A separate reactor waits for native ControlAuthority's token;
            // it owns no model and sends no token-level IPC.
            let cancellation = self.cancellation.clone();
            let (sender, receiver) = mpsc::sync_channel(1);
            let waiter = std::thread::spawn(move || {
                if let Ok(runtime) = tokio::runtime::Builder::new_current_thread().build() {
                    runtime.block_on(cancellation.cancelled());
                    let _ = sender.send(());
                }
            });
            let result = receiver.recv_timeout(Duration::from_secs(10));
            if result.is_err() {
                self.cancellation.cancel();
            }
            let _ = waiter.join();
            result.map_err(|_| {
                SimDriveError::EventSource(
                    "proof rendezvous timed out without framed cancellation".into(),
                )
            })?;
            // Return at the CURRENT instant, then make one no-op source
            // step so the driver polls the cancellation-woken application
            // before it completes the suspended model pass.
            self.resume_application.set(true);
            return Ok(Some(now));
        }
        Ok(next)
    }
    fn set_time_ns(&self, time: i64) -> Result<(), SimDriveError> {
        self.model.set_time_ns(time)
    }
    fn step(&self, time: i64) -> Result<SimStep, SimDriveError> {
        if self.resume_application.replace(false) {
            return Ok(SimStep {
                end_ns: time,
                made_progress: true,
            });
        }
        self.model.step(time)
    }
    fn is_idle(&self) -> bool {
        self.model.is_idle()
    }
}
