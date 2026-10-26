// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The response path from a worker to the client.
//!
//! A serving engine returns a pass's outputs through its server processes,
//! which detokenize each request's new tokens, format streamed chunks and write
//! them to the connections. Every worker that emits tokens has `workers` such
//! detokenizer servers, shared by its attention-DP ranks, and each request
//! belongs to one of them. A server takes the outputs of one pass that belong
//! to it as one job, after the jobs it has taken before: the job takes
//! `base_ms + per_output_ms * outputs`, and the client receives every output of
//! the job when it ends.
//!
//! The client records a finished request and acts on it when it receives the
//! end of its response. That is when a closed-loop slot, a session's next turn
//! or a dependent agentic request is released.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use anyhow::{Result, ensure};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use crate::replay::ReplayTerminalStatus;

/// Detokenization cost and server count of every worker that emits tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetokenizationLatency {
    workers: usize,
    base_ms: f64,
    per_output_ms: f64,
}

impl DetokenizationLatency {
    /// Returns `None` when both costs are zero, which keeps outputs at the pass end.
    pub(crate) fn new(workers: usize, base_ms: f64, per_output_ms: f64) -> Option<Self> {
        (base_ms > 0.0 || per_output_ms > 0.0).then_some(Self {
            workers,
            base_ms,
            per_output_ms,
        })
    }

    fn service_ms(self, outputs: usize) -> f64 {
        self.base_ms + self.per_output_ms * outputs as f64
    }

    /// The detokenizer server that `request_id` belongs to. A stable hash of
    /// the request id stands in for how connections spread over the serving
    /// engine's server processes.
    fn server(self, request_id: Uuid) -> usize {
        if self.workers == 1 {
            return 0;
        }
        let id = request_id.as_u128();
        // SplitMix64 finalizer over the folded id.
        let mut hash = (id as u64) ^ ((id >> 64) as u64);
        hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        hash ^= hash >> 31;
        (hash % self.workers as u64) as usize
    }
}

/// When the client receives each output of one pass.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PassReceipt {
    /// Every output of the pass arrives at once.
    Together(f64),
    /// Each request's output arrives when its server finishes the pass's job.
    PerServer {
        latency: DetokenizationLatency,
        received_ms: Vec<f64>,
    },
}

impl PassReceipt {
    /// The time the client receives `request_id`'s output of the pass.
    pub(crate) fn received_ms(&self, request_id: Uuid) -> f64 {
        match self {
            Self::Together(received_ms) => *received_ms,
            Self::PerServer {
                latency,
                received_ms,
            } => received_ms[latency.server(request_id)],
        }
    }
}

/// Outputs on their way to the client, and finished requests the client has
/// not received yet.
pub(crate) struct ClientPath {
    detokenization: DetokenizationLatency,
    /// Time each `(worker, server)` finishes the jobs it has taken. Only
    /// servers that have taken a job are tracked.
    server_free_at_ms: FxHashMap<(usize, usize), f64>,
    terminals: BinaryHeap<Reverse<PendingTerminal>>,
    next_entry: u64,
}

impl ClientPath {
    pub(crate) fn new(detokenization: DetokenizationLatency) -> Self {
        Self {
            detokenization,
            server_free_at_ms: FxHashMap::default(),
            terminals: BinaryHeap::new(),
            next_entry: 0,
        }
    }

    /// Hand the outputs of a pass that ended at `pass_end_ms` on `worker` to
    /// its detokenizer servers, one output per request in `outputs`, and return
    /// when the client receives them. Call once per pass, in pass order. A time
    /// that is not finite is rejected before any state changes.
    pub(crate) fn receive_pass(
        &mut self,
        worker: usize,
        pass_end_ms: f64,
        outputs: impl Iterator<Item = Uuid>,
    ) -> Result<PassReceipt> {
        let latency = self.detokenization;
        if latency.workers == 1 {
            let outputs = outputs.count();
            if outputs == 0 {
                return Ok(PassReceipt::Together(pass_end_ms));
            }
            let finished_ms = self.job_end_ms(worker, 0, pass_end_ms, outputs)?;
            self.server_free_at_ms.insert((worker, 0), finished_ms);
            return Ok(PassReceipt::Together(finished_ms));
        }
        let mut counts = vec![0usize; latency.workers];
        for request_id in outputs {
            counts[latency.server(request_id)] += 1;
        }
        let mut received_ms = vec![pass_end_ms; latency.workers];
        for (server, &outputs) in counts.iter().enumerate() {
            if outputs > 0 {
                received_ms[server] = self.job_end_ms(worker, server, pass_end_ms, outputs)?;
            }
        }
        for (server, &outputs) in counts.iter().enumerate() {
            if outputs > 0 {
                self.server_free_at_ms
                    .insert((worker, server), received_ms[server]);
            }
        }
        Ok(PassReceipt::PerServer {
            latency,
            received_ms,
        })
    }

    fn job_end_ms(
        &self,
        worker: usize,
        server: usize,
        pass_end_ms: f64,
        outputs: usize,
    ) -> Result<f64> {
        let start_ms = self
            .server_free_at_ms
            .get(&(worker, server))
            .map_or(pass_end_ms, |free_at_ms| free_at_ms.max(pass_end_ms));
        let finished_ms = start_ms + self.detokenization.service_ms(outputs);
        ensure!(
            finished_ms.is_finite(),
            "detokenizing {outputs} outputs of a pass that ends at {pass_end_ms} ms does not finish in finite time"
        );
        Ok(finished_ms)
    }

    /// Hold a finished request until the client receives the end of its
    /// response at `received_ms`.
    pub(crate) fn defer_terminal(
        &mut self,
        received_ms: f64,
        request_id: Uuid,
        status: ReplayTerminalStatus,
    ) {
        self.terminals.push(Reverse(PendingTerminal {
            at_ms: received_ms,
            entry: self.next_entry,
            request_id,
            status,
        }));
        self.next_entry += 1;
    }

    /// Finished requests the client has not received yet.
    pub(crate) fn len(&self) -> usize {
        self.terminals.len()
    }

    pub(crate) fn next_release_ms(&self) -> Option<f64> {
        self.terminals.peek().map(|Reverse(pending)| pending.at_ms)
    }

    /// Remove every finished request the client has received by `now_ms`, in
    /// receive time and entry order.
    pub(crate) fn release_due(&mut self, now_ms: f64) -> Vec<(f64, Uuid, ReplayTerminalStatus)> {
        let mut released = Vec::new();
        while self
            .terminals
            .peek()
            .is_some_and(|Reverse(pending)| pending.at_ms <= now_ms)
        {
            let Reverse(pending) = self.terminals.pop().expect("peeked client terminal");
            released.push((pending.at_ms, pending.request_id, pending.status));
        }
        released
    }
}

#[derive(Debug, Clone, Copy)]
struct PendingTerminal {
    at_ms: f64,
    entry: u64,
    request_id: Uuid,
    status: ReplayTerminalStatus,
}

impl PartialEq for PendingTerminal {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for PendingTerminal {}

impl PartialOrd for PendingTerminal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingTerminal {
    fn cmp(&self, other: &Self) -> Ordering {
        self.at_ms
            .total_cmp(&other.at_ms)
            .then_with(|| self.entry.cmp(&other.entry))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(workers: usize, base_ms: f64, per_output_ms: f64) -> ClientPath {
        ClientPath::new(DetokenizationLatency::new(workers, base_ms, per_output_ms).unwrap())
    }

    fn ids(range: std::ops::Range<u128>) -> impl Iterator<Item = Uuid> {
        range.map(Uuid::from_u128)
    }

    #[test]
    fn a_pass_reaches_the_client_when_its_job_ends() {
        let mut path = path(1, 1.0, 0.5);
        assert_eq!(
            path.receive_pass(0, 10.0, ids(0..3)).unwrap(),
            PassReceipt::Together(12.5)
        );
    }

    #[test]
    fn a_pass_without_outputs_takes_no_server_time() {
        let mut path = path(1, 1.0, 0.5);
        assert_eq!(
            path.receive_pass(0, 10.0, ids(0..0)).unwrap(),
            PassReceipt::Together(10.0)
        );
        assert!(path.server_free_at_ms.is_empty());
    }

    #[test]
    fn base_time_occupies_the_server_and_workers_have_their_own() {
        let mut path = path(1, 4.0, 0.0);
        assert_eq!(
            path.receive_pass(0, 0.0, ids(0..1))
                .unwrap()
                .received_ms(Uuid::nil()),
            4.0
        );
        // The next pass ends while the server is still busy and queues behind it.
        assert_eq!(
            path.receive_pass(0, 1.0, ids(0..1))
                .unwrap()
                .received_ms(Uuid::nil()),
            8.0
        );
        assert_eq!(
            path.receive_pass(1, 1.0, ids(0..1))
                .unwrap()
                .received_ms(Uuid::nil()),
            5.0
        );
        // An idle server starts at the pass end.
        assert_eq!(
            path.receive_pass(0, 20.0, ids(0..1))
                .unwrap()
                .received_ms(Uuid::nil()),
            24.0
        );
    }

    #[test]
    fn a_request_always_belongs_to_the_same_server() {
        let latency = DetokenizationLatency::new(4, 1.0, 0.0).unwrap();
        let request = Uuid::from_u128(12_345);
        let server = latency.server(request);
        assert!((0..100).all(|_| latency.server(request) == server));
        // Requests spread over every server.
        let used: std::collections::HashSet<_> = ids(0..64).map(|id| latency.server(id)).collect();
        assert_eq!(used.len(), 4);
    }

    #[test]
    fn servers_split_a_pass_and_queue_independently() {
        let mut path = path(2, 1.0, 1.0);
        let latency = path.detokenization;
        let (on_0, on_1): (Vec<_>, Vec<_>) = ids(0..16).partition(|id| latency.server(*id) == 0);
        let a0 = on_0[0];
        let a1 = on_1[0];
        // Three outputs on server 0 and one on server 1.
        let first = path
            .receive_pass(0, 0.0, [on_0[0], on_0[1], on_0[2], on_1[0]].into_iter())
            .unwrap();
        assert_eq!(first.received_ms(a0), 4.0);
        assert_eq!(first.received_ms(a1), 2.0);
        // Server 1 is free again at 2 ms, server 0 only at 4 ms.
        let second = path.receive_pass(0, 3.0, [a0, a1].into_iter()).unwrap();
        assert_eq!(second.received_ms(a0), 6.0);
        assert_eq!(second.received_ms(a1), 5.0);
    }

    #[test]
    fn the_client_acts_on_a_request_when_it_receives_it() {
        let mut path = path(1, 1.0, 0.0);
        let (first, second, third) = (Uuid::from_u128(1), Uuid::from_u128(2), Uuid::from_u128(3));
        path.defer_terminal(7.0, first, ReplayTerminalStatus::Completed);
        path.defer_terminal(5.0, second, ReplayTerminalStatus::Rejected);
        path.defer_terminal(7.0, third, ReplayTerminalStatus::Completed);
        assert_eq!(path.len(), 3);
        assert_eq!(path.next_release_ms(), Some(5.0));
        assert_eq!(path.release_due(4.0), vec![]);
        assert_eq!(
            path.release_due(5.0),
            vec![(5.0, second, ReplayTerminalStatus::Rejected)]
        );
        // Requests received together are released in entry order.
        assert_eq!(
            path.release_due(7.0),
            vec![
                (7.0, first, ReplayTerminalStatus::Completed),
                (7.0, third, ReplayTerminalStatus::Completed),
            ]
        );
        assert_eq!(path.len(), 0);
        assert_eq!(path.next_release_ms(), None);
    }

    #[test]
    fn a_non_finite_receive_time_is_rejected_without_state_changes() {
        let mut path = path(2, 0.0, f64::MAX);
        let latency = path.detokenization;
        let (on_0, on_1): (Vec<_>, Vec<_>) = ids(0..16).partition(|id| latency.server(*id) == 0);
        // Server 1 can take one output, but two on server 0 overflow: nothing changes.
        assert!(
            path.receive_pass(0, 0.0, [on_1[0], on_0[0], on_0[1]].into_iter())
                .is_err()
        );
        assert!(path.server_free_at_ms.is_empty());
    }

    #[test]
    fn zero_costs_disable_detokenization() {
        assert_eq!(DetokenizationLatency::new(1, 0.0, 0.0), None);
    }
}
