// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Offline secondary storage. Each Replay deployment owns its G3 tier.
//! No global namespace, wall clock, payload allocation, or hardware connection.

use crate::engine::config::{G3OffloadConfig, G3Scope};
use crate::engine::host_offload::HostBlockKey;
use anyhow::{Result, ensure};
use rustc_hash::FxHashMap;
use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct G3IoStats {
    pub submitted_jobs: u64,
    pub completed_jobs: u64,
    pub cancelled_jobs: u64,
    pub completed_bytes: u64,
    pub transfer_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct G3Stats {
    /// Every actual block probe, including retries; pending is not a hit.
    pub lookup_probes: u64,
    pub lookup_hits: u64,
    pub lookup_pending: u64,
    pub read: G3IoStats,
    pub write: G3IoStats,
    pub evictions: u64,
    pub cross_worker_read_blocks: u64,
    pub resident_blocks: usize,
    pub pending_blocks: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Probe {
    Miss,
    Pending,
    Resident,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Direction {
    Read,
    Write,
}

#[derive(Clone, Copy)]
struct Entry {
    pending: bool,
    pins: usize,
    writer: usize,
    touched: u64,
}

struct Job {
    id: u64,
    worker: usize,
    direction: Direction,
    keys: Vec<HostBlockKey>,
    submitted: f64,
    ready: f64,
    remaining: f64,
}

// Ephemeral counts for one queue/time snapshot; never retained across events.
#[derive(Default)]
struct ReadyCounts {
    total: [usize; 2],
    workers: FxHashMap<usize, [usize; 2]>,
}

impl ReadyCounts {
    fn new<'a>(jobs: impl Iterator<Item = &'a Job>, now: f64) -> Self {
        let mut counts = Self::default();
        for job in jobs {
            if job.ready <= now {
                let direction = job.direction as usize;
                counts.total[direction] += 1;
                counts.workers.entry(job.worker).or_default()[direction] += 1;
            }
        }
        counts
    }

    // Equal bandwidth shares among data-moving jobs, capped independently by
    // worker bandwidth and, only for shared storage, shared backend bandwidth.
    // Reads and writes use separate full-duplex
    // budgets. Jobs waiting for first-byte latency do not consume bandwidth.
    fn rate(&self, config: &G3OffloadConfig, job: &Job) -> f64 {
        let direction = job.direction as usize;
        let total = self.total[direction];
        let local = self
            .workers
            .get(&job.worker)
            .map_or(0, |counts| counts[direction]);
        let (worker, shared) = match job.direction {
            Direction::Read => (
                config.read_bandwidth_gbps,
                config.shared_read_bandwidth_gbps,
            ),
            Direction::Write => (
                config.write_bandwidth_gbps,
                config.shared_write_bandwidth_gbps,
            ),
        };
        let rate = |bw: f64, n: usize| {
            if bw == 0.0 {
                f64::INFINITY
            } else {
                bw * 1e6 / n.max(1) as f64
            }
        };
        let worker_rate = rate(worker, local);
        match config.scope {
            G3Scope::WorkerLocal => worker_rate,
            G3Scope::ClusterShared => worker_rate.min(rate(shared, total)),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Completion {
    pub id: u64,
    pub direction: Direction,
    pub keys: Vec<HostBlockKey>,
    pub at_ms: f64,
}

pub(crate) type SharedG3Tier = Arc<Mutex<G3Tier>>;

pub(crate) struct G3Tier {
    config: G3OffloadConfig,
    next_worker_id: usize,
    block_bytes: usize,
    entries: BTreeMap<(usize, HostBlockKey), Entry>,
    jobs: VecDeque<Job>,
    completed: BTreeMap<usize, Vec<Completion>>,
    now: f64,
    ordinal: u64,
    pub(crate) completion_epoch: u64,
    pub stats: G3Stats,
}

impl G3Tier {
    pub(crate) fn new(
        config: G3OffloadConfig,
        workers: usize,
        block_bytes: usize,
    ) -> Result<SharedG3Tier> {
        config.validate()?;
        ensure!(block_bytes > 0, "G3 requires positive block bytes");
        let bytes = config
            .num_g3_blocks
            .checked_mul(block_bytes)
            .ok_or_else(|| anyhow::anyhow!("G3 capacity bytes overflow"))?;
        for bandwidth in [config.read_bandwidth_gbps, config.write_bandwidth_gbps]
            .into_iter()
            .chain(
                (config.scope == G3Scope::ClusterShared)
                    .then_some([
                        config.shared_read_bandwidth_gbps,
                        config.shared_write_bandwidth_gbps,
                    ])
                    .into_iter()
                    .flatten(),
            )
        {
            ensure!(
                bandwidth == 0.0 || (bytes as f64 / (bandwidth * 1e6)).is_finite(),
                "G3 transfer duration overflow"
            );
        }
        Ok(Arc::new(Mutex::new(Self {
            config,
            next_worker_id: workers,
            block_bytes,
            entries: BTreeMap::new(),
            jobs: VecDeque::new(),
            completed: (0..workers).map(|worker| (worker, Vec::new())).collect(),
            now: 0.0,
            ordinal: 0,
            completion_epoch: 0,
            stats: G3Stats::default(),
        })))
    }

    pub(crate) fn current_time_ms(&self) -> f64 {
        self.now
    }

    pub(crate) fn register_worker(&mut self, worker: usize) -> Result<()> {
        ensure!(
            worker >= self.next_worker_id,
            "G3 worker IDs cannot be reused"
        );
        self.next_worker_id = worker
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("G3 worker ID overflow"))?;
        self.completed.insert(worker, Vec::new());
        Ok(())
    }

    pub(crate) fn has_work(&self, worker: usize) -> bool {
        !self.completed[&worker].is_empty() || self.jobs.iter().any(|job| job.worker == worker)
    }

    pub(crate) fn unregister_worker(&mut self, worker: usize) -> Result<()> {
        ensure!(
            self.completed.contains_key(&worker),
            "G3 worker is not registered"
        );
        ensure!(
            !self.has_work(worker),
            "cannot unregister G3 worker with pending I/O"
        );
        self.completed.remove(&worker);
        if self.config.scope == G3Scope::WorkerLocal {
            self.entries.retain(|(pool, _), _| *pool != worker);
        }
        Ok(())
    }

    fn pool(&self, worker: usize) -> usize {
        assert!(
            self.completed.contains_key(&worker),
            "G3 worker is not registered"
        );
        match self.config.scope {
            G3Scope::WorkerLocal => worker,
            G3Scope::ClusterShared => 0,
        }
    }

    /// Read-only availability for a capacity check; does not touch LRU or stats.
    pub(crate) fn contains(&self, worker: usize, key: HostBlockKey) -> bool {
        self.entries.contains_key(&(self.pool(worker), key))
    }

    pub(crate) fn probe(&mut self, worker: usize, key: HostBlockKey) -> Probe {
        let identity = (self.pool(worker), key);
        self.stats.lookup_probes += 1;
        self.ordinal += 1;
        match self.entries.get_mut(&identity) {
            None => Probe::Miss,
            Some(entry) if entry.pending => {
                self.stats.lookup_pending += 1;
                Probe::Pending
            }
            Some(entry) => {
                entry.touched = self.ordinal;
                self.stats.lookup_hits += 1;
                Probe::Resident
            }
        }
    }

    pub(crate) fn submit(
        &mut self,
        worker: usize,
        direction: Direction,
        keys: &[HostBlockKey],
        now: f64,
    ) -> Option<u64> {
        // Commands from a lagging rank can never schedule work in the past.
        self.advance(now.max(self.now));
        let pool = self.pool(worker);
        let keys = match direction {
            Direction::Write => keys
                .iter()
                .copied()
                .filter(|key| !self.entries.contains_key(&(pool, *key)))
                .collect::<Vec<_>>(),
            Direction::Read => keys.to_vec(),
        };
        if keys.is_empty() {
            return None;
        }
        if direction == Direction::Write {
            let used = self.entries.keys().filter(|(p, _)| *p == pool).count();
            let needed = (used + keys.len()).saturating_sub(self.config.num_g3_blocks);
            let mut victims = self
                .entries
                .iter()
                .filter(|((p, _), e)| *p == pool && !e.pending && e.pins == 0)
                .map(|(key, entry)| (entry.touched, *key))
                .collect::<Vec<_>>();
            victims.sort_unstable();
            if victims.len() < needed {
                return None;
            }
            for (_, key) in victims.into_iter().take(needed) {
                self.entries.remove(&key);
                self.stats.evictions += 1;
            }
            for key in &keys {
                self.entries.insert(
                    (pool, *key),
                    Entry {
                        pending: true,
                        pins: 0,
                        writer: worker,
                        touched: 0,
                    },
                );
            }
        } else {
            if keys
                .iter()
                .any(|key| self.entries.get(&(pool, *key)).is_none_or(|e| e.pending))
            {
                return None;
            }
            for key in &keys {
                self.entries.get_mut(&(pool, *key)).unwrap().pins += 1;
            }
        }
        self.ordinal += 1;
        let id = self.ordinal;
        self.io_mut(direction).submitted_jobs += 1;
        self.jobs.push_back(Job {
            id,
            worker,
            direction,
            remaining: (keys.len() * self.block_bytes) as f64,
            keys,
            submitted: self.now,
            ready: self.now + self.config.latency_to_first_byte_ms,
        });
        Some(id)
    }

    pub(crate) fn job_keys(&self, id: u64) -> Vec<HostBlockKey> {
        self.jobs.iter().find(|j| j.id == id).unwrap().keys.clone()
    }

    fn io_mut(&mut self, direction: Direction) -> &mut G3IoStats {
        match direction {
            Direction::Read => &mut self.stats.read,
            Direction::Write => &mut self.stats.write,
        }
    }

    fn event_time(&self, job: &Job, counts: &ReadyCounts) -> f64 {
        if job.ready > self.now {
            job.ready
        } else {
            self.now + job.remaining / counts.rate(&self.config, job)
        }
    }

    pub(crate) fn next_deadline(&self, worker: usize) -> Option<f64> {
        let counts = ReadyCounts::new(self.jobs.iter(), self.now);
        self.jobs
            .iter()
            .map(|job| self.event_time(job, &counts))
            .chain(self.completed[&worker].iter().map(|done| done.at_ms))
            .min_by(f64::total_cmp)
    }

    pub(crate) fn advance(&mut self, target: f64) {
        let target = target.max(self.now);
        loop {
            let counts = ReadyCounts::new(self.jobs.iter(), self.now);
            let event = self
                .jobs
                .iter()
                .map(|j| self.event_time(j, &counts))
                .min_by(f64::total_cmp);
            let next = event.unwrap_or(target).min(target);
            let elapsed = next - self.now;
            let rates = self
                .jobs
                .iter()
                .map(|j| {
                    if j.ready <= self.now {
                        (
                            counts.rate(&self.config, j),
                            self.event_time(j, &counts) <= next,
                        )
                    } else {
                        (0.0, false)
                    }
                })
                .collect::<Vec<_>>();
            for (job, (rate, completes)) in self.jobs.iter_mut().zip(rates) {
                // The selected completion boundary is authoritative. Repeated
                // byte subtraction at a large timestamp can leave a residual
                // whose duration is below one clock ULP and would never drain.
                if completes {
                    job.remaining = 0.0;
                } else {
                    job.remaining = (job.remaining - elapsed * rate).max(0.0);
                }
            }
            self.now = next;
            let mut finished = false;
            let mut index = 0;
            while index < self.jobs.len() {
                if self.jobs[index].ready <= self.now && self.jobs[index].remaining == 0.0 {
                    let job = self.jobs.remove(index).unwrap();
                    self.finish(job);
                    finished = true;
                } else {
                    index += 1;
                }
            }
            if event.is_none_or(|time| time > target) {
                break;
            }
            if self.now == target && !finished {
                // Advancing time can make first-byte waiters eligible at this boundary.
                let counts = ReadyCounts::new(self.jobs.iter(), self.now);
                if self
                    .jobs
                    .iter()
                    .all(|job| self.event_time(job, &counts) > target)
                {
                    break;
                }
            }
        }
    }

    fn finish(&mut self, job: Job) {
        self.completion_epoch += 1;
        let pool = self.pool(job.worker);
        self.ordinal += 1;
        for key in &job.keys {
            let entry = self.entries.get_mut(&(pool, *key)).unwrap();
            match job.direction {
                Direction::Write => entry.pending = false,
                Direction::Read => {
                    entry.pins -= 1;
                    self.stats.cross_worker_read_blocks += u64::from(entry.writer != job.worker);
                }
            }
            entry.touched = self.ordinal;
        }
        let bytes = (job.keys.len() * self.block_bytes) as u64;
        let now = self.now;
        let io = self.io_mut(job.direction);
        io.completed_jobs += 1;
        io.completed_bytes += bytes;
        io.transfer_ms += now - job.submitted;
        self.completed
            .get_mut(&job.worker)
            .unwrap()
            .push(Completion {
                id: job.id,
                direction: job.direction,
                keys: job.keys,
                at_ms: now,
            });
    }

    pub(crate) fn take_completed(&mut self, worker: usize, now: f64) -> Vec<Completion> {
        self.advance(now);
        let mut result = Vec::new();
        self.completed.get_mut(&worker).unwrap().retain(|done| {
            if done.at_ms <= now {
                result.push(done.clone());
                false
            } else {
                true
            }
        });
        result
    }

    #[cfg(test)]
    pub(crate) fn cancel(&mut self, id: u64, now: f64) {
        self.advance(now.max(self.now));
        if let Some(index) = self.jobs.iter().position(|job| job.id == id) {
            let job = self.jobs.remove(index).unwrap();
            let pool = self.pool(job.worker);
            for key in &job.keys {
                if job.direction == Direction::Write {
                    self.entries.remove(&(pool, *key));
                } else {
                    self.entries.get_mut(&(pool, *key)).unwrap().pins -= 1;
                }
            }
            self.io_mut(job.direction).cancelled_jobs += 1;
        }
    }

    pub(crate) fn snapshot(&self) -> G3Stats {
        let mut stats = self.stats.clone();
        stats.pending_blocks = self.entries.values().filter(|e| e.pending).count();
        stats.resident_blocks = self.entries.len() - stats.pending_blocks;
        stats
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
