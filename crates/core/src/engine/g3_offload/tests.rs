// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn config(scope: G3Scope) -> G3OffloadConfig {
    G3OffloadConfig {
        scope,
        num_g3_blocks: 8,
        latency_to_first_byte_ms: 0.0,
        read_bandwidth_gbps: 0.0,
        write_bandwidth_gbps: 0.0,
        shared_read_bandwidth_gbps: 0.0,
        shared_write_bandwidth_gbps: 0.0,
    }
}

fn key(n: u64) -> HostBlockKey {
    HostBlockKey::new(n)
}

#[test]
fn oversized_writes_retain_the_leading_blocks_that_fit() {
    for scope in [G3Scope::WorkerLocal, G3Scope::ClusterShared] {
        let mut cfg = config(scope);
        cfg.num_g3_blocks = 2;
        cfg.write_bandwidth_gbps = 0.001;
        let shared = G3Tier::new(cfg, 2, 1_000).unwrap();
        let mut tier = shared.lock().unwrap();
        let cohort = [key(3), key(1), key(2)];
        let write = tier.submit(0, Direction::Write, &cohort, 0.0).unwrap();
        assert_eq!(tier.job_keys(write), cohort[..2]);
        assert_eq!(tier.snapshot().pending_blocks, 2);
        let done = tier.take_completed(0, 2.0);
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].keys, cohort[..2]);
        assert_eq!(tier.snapshot().write.completed_bytes, 2_000);
        assert_eq!(tier.snapshot().resident_blocks, 2);
        assert_eq!(tier.probe(0, key(2)), Probe::Miss);
        let read = tier.submit(0, Direction::Read, &cohort[..2], 2.0).unwrap();
        assert_eq!(tier.take_completed(0, 2.0)[0].id, read);
        assert_eq!(tier.snapshot().read.completed_bytes, 2_000);
    }
}

#[test]
fn partial_writes_preserve_pending_pinned_and_existing_prefix_blocks() {
    for scope in [G3Scope::WorkerLocal, G3Scope::ClusterShared] {
        let mut cfg = config(scope);
        cfg.num_g3_blocks = 4;
        let shared = G3Tier::new(cfg, 1, 1_000).unwrap();
        let mut tier = shared.lock().unwrap();
        tier.submit(0, Direction::Write, &[key(1), key(2), key(9)], 0.0)
            .unwrap();
        tier.take_completed(0, 0.0);
        tier.config.read_bandwidth_gbps = 0.001;
        tier.config.write_bandwidth_gbps = 0.001;
        let read = tier.submit(0, Direction::Read, &[key(2)], 0.0).unwrap();
        tier.submit(0, Direction::Write, &[key(3)], 0.0).unwrap();
        // Only the unrelated resident block 9 can be evicted. Block 1 is
        // already part of this prefix; block 2 is pinned and block 3 pending.
        let write = tier
            .submit(0, Direction::Write, &[key(1), key(4), key(5)], 0.0)
            .unwrap();
        assert_eq!(tier.job_keys(write), vec![key(4)]);
        assert_eq!(tier.probe(0, key(1)), Probe::Resident);
        assert_eq!(tier.probe(0, key(2)), Probe::Resident);
        assert_eq!(tier.probe(0, key(3)), Probe::Pending);
        assert_eq!(tier.probe(0, key(9)), Probe::Miss);
        assert_eq!(tier.snapshot().evictions, 1);
        tier.cancel(write, 0.0);
        tier.cancel(read, 0.0);
        tier.take_completed(0, 1.0);
        assert_eq!(tier.snapshot().resident_blocks, 3);
        assert_eq!(tier.snapshot().pending_blocks, 0);
        assert_eq!(tier.entries[&(0, key(2))].pins, 0);
    }
}

#[test]
fn ready_count_snapshot_visits_each_job_once_and_reuses_counts_for_rates() {
    use std::cell::Cell;
    for n in [16, 128, 1024] {
        let jobs = (0..n)
            .map(|id| Job {
                id: id as u64,
                worker: id % 4,
                direction: if id % 2 == 0 {
                    Direction::Read
                } else {
                    Direction::Write
                },
                keys: vec![],
                submitted: 0.0,
                ready: if id < n / 2 { 0.0 } else { 1.0 },
                remaining: 1_000_000.0,
            })
            .collect::<Vec<_>>();
        let visits = Cell::new(0);
        let counts = ReadyCounts::new(jobs.iter().inspect(|_| visits.set(visits.get() + 1)), 0.0);
        assert_eq!(visits.get(), n);
        assert_eq!(counts.total, [n / 4, n / 4]);
        let mut cfg = config(G3Scope::ClusterShared);
        cfg.read_bandwidth_gbps = 1.0;
        cfg.write_bandwidth_gbps = 1.0;
        for job in &jobs[..n / 2] {
            assert_eq!(counts.rate(&cfg, job), 1_000_000.0 / (n / 8) as f64);
        }
        assert_eq!(
            visits.get(),
            n,
            "rate evaluation has no access to the job queue"
        );
        let ready = ReadyCounts::new(jobs.iter(), 1.0);
        assert_eq!(ready.total, [n / 2, n / 2]);
        assert_eq!(
            ready.rate(&cfg, &jobs[0]),
            counts.rate(&cfg, &jobs[0]) / 2.0
        );
    }
}

#[test]
fn ready_counts_refresh_when_completion_and_first_byte_share_a_boundary() {
    let mut cfg = config(G3Scope::ClusterShared);
    cfg.write_bandwidth_gbps = 1.0;
    let shared = G3Tier::new(cfg, 1, 1_000_000).unwrap();
    let mut tier = shared.lock().unwrap();
    tier.submit(0, Direction::Write, &[key(1)], 0.0).unwrap();
    tier.config.latency_to_first_byte_ms = 1.0;
    tier.submit(0, Direction::Write, &[key(2)], 0.0).unwrap();
    let first = tier.take_completed(0, 1.0);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].keys, vec![key(1)]);
    assert_eq!(first[0].at_ms, 1.0);
    assert_eq!(tier.next_deadline(0), Some(2.0));
    let second = tier.take_completed(0, 2.0);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].keys, vec![key(2)]);
    assert_eq!(second[0].at_ms, 2.0);
    assert_eq!(tier.snapshot().write.transfer_ms, 3.0);
    assert_eq!(tier.next_deadline(0), None);
}

#[test]
fn config_defaults_and_explicit_overrides_round_trip() {
    let minimal = serde_json::json!({"scope": "worker_local", "num_g3_blocks": 32});
    let cfg: G3OffloadConfig = serde_json::from_value(minimal).unwrap();
    assert_eq!(cfg.latency_to_first_byte_ms, 0.1);
    assert_eq!(cfg.read_bandwidth_gbps, 10.0);
    assert_eq!(cfg.write_bandwidth_gbps, 10.0);
    assert_eq!(cfg.shared_read_bandwidth_gbps, 80.0);
    assert_eq!(cfg.shared_write_bandwidth_gbps, 80.0);
    for value in [0.0, 3.0] {
        let explicit = serde_json::json!({
            "scope": "cluster_shared", "num_g3_blocks": 32,
            "latency_to_first_byte_ms": value,
            "read_bandwidth_gbps": value, "write_bandwidth_gbps": value,
            "shared_read_bandwidth_gbps": value, "shared_write_bandwidth_gbps": value
        });
        let cfg: G3OffloadConfig = serde_json::from_value(explicit.clone()).unwrap();
        cfg.validate().unwrap();
        assert_eq!(serde_json::to_value(cfg).unwrap(), explicit);
    }
}

#[test]
fn sixteen_workers_apply_shared_cap_only_to_shared_scope() {
    for scope in [G3Scope::WorkerLocal, G3Scope::ClusterShared] {
        for direction in [Direction::Read, Direction::Write] {
            for (worker_bw, shared_bw) in [(10.0, 80.0), (10.0, 0.0), (0.0, 80.0), (0.0, 0.0)] {
                let mut cfg = config(scope);
                cfg.num_g3_blocks = 16;
                let shared = G3Tier::new(cfg, 16, 10_000_000).unwrap();
                let mut tier = shared.lock().unwrap();
                if direction == Direction::Read {
                    for worker in 0..16 {
                        tier.submit(worker, Direction::Write, &[key(worker as u64)], 0.0)
                            .unwrap();
                    }
                    for worker in 0..16 {
                        tier.take_completed(worker, 0.0);
                    }
                }
                tier.config.read_bandwidth_gbps = worker_bw;
                tier.config.write_bandwidth_gbps = worker_bw;
                tier.config.shared_read_bandwidth_gbps = shared_bw;
                tier.config.shared_write_bandwidth_gbps = shared_bw;
                for worker in 0..16 {
                    tier.submit(worker, direction, &[key(worker as u64)], 0.0)
                        .unwrap();
                }
                let worker_ms: f64 = if worker_bw == 0.0 { 0.0 } else { 1.0 };
                let shared_ms = if scope == G3Scope::ClusterShared && shared_bw > 0.0 {
                    2.0
                } else {
                    0.0
                };
                let expected = worker_ms.max(shared_ms);
                for worker in 0..16 {
                    let done = tier.take_completed(worker, expected);
                    assert_eq!(done.len(), 1);
                    assert_eq!(done[0].at_ms, expected);
                }
                assert_eq!(tier.io_mut(direction).completed_bytes, 160_000_000);
            }
        }
    }
}

#[test]
fn scopes_have_independent_local_capacity_and_runs_are_isolated() {
    for scope in [G3Scope::WorkerLocal, G3Scope::ClusterShared] {
        let mut cfg = config(scope);
        cfg.num_g3_blocks = 3;
        let shared = G3Tier::new(cfg.clone(), 2, 100).unwrap();
        let mut r = shared.lock().unwrap();
        r.submit(0, Direction::Write, &[key(1)], 0.0).unwrap();
        assert_eq!(r.probe(0, key(1)), Probe::Pending);
        r.advance(0.0);
        assert_eq!(
            r.probe(1, key(1)),
            if scope == G3Scope::WorkerLocal {
                Probe::Miss
            } else {
                Probe::Resident
            }
        );
        assert_eq!(
            G3Tier::new(cfg, 2, 100)
                .unwrap()
                .lock()
                .unwrap()
                .probe(0, key(1)),
            Probe::Miss
        );
        r.submit(0, Direction::Write, &[key(2), key(3)], 0.0)
            .unwrap();
        r.advance(0.0);
        r.register_worker(9).unwrap();
        r.submit(9, Direction::Write, &[key(4), key(5), key(6)], 0.0)
            .unwrap();
        r.advance(0.0);
        assert_eq!(
            r.snapshot().resident_blocks,
            if scope == G3Scope::WorkerLocal { 6 } else { 3 }
        );
        assert_eq!(
            r.snapshot().evictions,
            if scope == G3Scope::WorkerLocal { 0 } else { 3 }
        );
    }
}

#[test]
fn retiring_workers_drain_only_their_io_and_never_reuse_ids() {
    for scope in [G3Scope::WorkerLocal, G3Scope::ClusterShared] {
        let mut cfg = config(scope);
        cfg.latency_to_first_byte_ms = 1.0;
        let shared = G3Tier::new(cfg, 0, 100).unwrap();
        let mut tier = shared.lock().unwrap();
        tier.register_worker(2).unwrap();
        tier.register_worker(8).unwrap();
        tier.submit(2, Direction::Write, &[key(1)], 0.0).unwrap();
        assert!(tier.unregister_worker(2).is_err());
        tier.advance(1.0);
        assert!(
            tier.has_work(2),
            "completed but undelivered I/O still owns pins"
        );
        assert!(tier.unregister_worker(2).is_err());
        assert_eq!(tier.take_completed(2, 1.0).len(), 1);
        tier.submit(2, Direction::Read, &[key(1)], 1.0).unwrap();
        tier.submit(8, Direction::Write, &[key(2)], 1.5).unwrap();
        if scope == G3Scope::ClusterShared {
            tier.submit(8, Direction::Read, &[key(1)], 1.5).unwrap();
        }
        assert!(tier.unregister_worker(2).is_err());
        assert_eq!(tier.take_completed(2, 2.0).len(), 1);
        assert!(tier.has_work(8));
        tier.unregister_worker(2).unwrap();
        if scope == G3Scope::ClusterShared {
            assert_eq!(
                tier.entries[&(0, key(1))].pins,
                1,
                "retired writer's block is still pinned by a foreign reader"
            );
        }
        assert!(!tier.completed.contains_key(&2));
        assert!(tier.register_worker(2).is_err());
        tier.register_worker(20).unwrap();
        assert_eq!(
            tier.probe(20, key(1)),
            if scope == G3Scope::WorkerLocal {
                Probe::Miss
            } else {
                Probe::Resident
            }
        );
        if scope == G3Scope::ClusterShared {
            tier.submit(20, Direction::Read, &[key(1)], 2.0).unwrap();
            tier.take_completed(20, 3.0);
            assert_eq!(tier.snapshot().cross_worker_read_blocks, 2);
        }
        tier.take_completed(8, 3.0);
        tier.unregister_worker(8).unwrap();
        tier.unregister_worker(20).unwrap();
        assert!(tier.completed.is_empty());
        assert!(tier.jobs.is_empty());
        assert_eq!(tier.snapshot().pending_blocks, 0);
        assert_eq!(
            tier.snapshot().resident_blocks,
            if scope == G3Scope::WorkerLocal { 0 } else { 2 }
        );
    }
}

#[test]
fn unregister_waits_for_all_same_timestamp_completions() {
    let mut cfg = config(G3Scope::WorkerLocal);
    cfg.latency_to_first_byte_ms = 1.0;
    let shared = G3Tier::new(cfg, 1, 100).unwrap();
    let mut tier = shared.lock().unwrap();
    for block in [1, 2] {
        tier.submit(0, Direction::Write, &[key(block)], 0.0)
            .unwrap();
    }
    tier.advance(1.0);
    assert!(tier.unregister_worker(0).is_err());
    let done = tier.take_completed(0, 1.0);
    assert_eq!(done.len(), 2);
    assert!(done.iter().all(|completion| completion.at_ms == 1.0));
    assert_eq!(tier.snapshot().resident_blocks, 2);
    tier.unregister_worker(0).unwrap();
    assert_eq!(tier.snapshot().resident_blocks, 0);
    assert!(tier.completed.is_empty());
}

#[test]
fn duplicate_payload_keeps_first_writer_and_counts_each_probe() {
    let shared = G3Tier::new(config(G3Scope::ClusterShared), 2, 100).unwrap();
    let mut r = shared.lock().unwrap();
    assert_eq!(r.probe(1, key(1)), Probe::Miss);
    r.submit(0, Direction::Write, &[key(1)], 0.0).unwrap();
    r.advance(0.0);
    assert!(r.submit(1, Direction::Write, &[key(1)], 0.0).is_none());
    for _ in 0..3 {
        assert_eq!(r.probe(1, key(1)), Probe::Resident);
    }
    r.submit(1, Direction::Read, &[key(1)], 0.0).unwrap();
    r.advance(0.0);
    let stats = r.snapshot();
    assert_eq!((stats.lookup_hits, stats.lookup_probes), (3, 4));
    assert_eq!(stats.write.completed_bytes, 100);
    assert_eq!(stats.read.completed_bytes, 100);
    assert_eq!(stats.cross_worker_read_blocks, 1);
}

#[test]
fn read_pin_blocks_eviction_and_cancel_releases_capacity() {
    let mut cfg = config(G3Scope::ClusterShared);
    cfg.num_g3_blocks = 1;
    cfg.read_bandwidth_gbps = 0.001;
    let shared = G3Tier::new(cfg, 2, 1_000_000).unwrap();
    let mut r = shared.lock().unwrap();
    r.submit(0, Direction::Write, &[key(1)], 0.0).unwrap();
    r.advance(0.0);
    let read = r.submit(1, Direction::Read, &[key(1)], 0.0).unwrap();
    assert!(r.submit(0, Direction::Write, &[key(2)], 0.0).is_none());
    r.cancel(read, 10.0);
    r.submit(0, Direction::Write, &[key(2)], 10.0).unwrap();
    r.advance(10.0);
    assert_eq!(r.probe(0, key(1)), Probe::Miss);
    assert_eq!(r.snapshot().evictions, 1);
    assert_eq!(r.snapshot().read.completed_bytes, 0);
    assert_eq!(r.snapshot().read.cancelled_jobs, 1);
}

#[test]
fn bandwidth_shares_respect_worker_and_shared_caps() {
    for (worker_bw, shared_bw, same_worker, expected) in [
        (1.0, 0.0, false, 1.0),
        (1.0, 0.0, true, 2.0),
        (0.0, 1.0, false, 2.0),
    ] {
        let mut cfg = config(G3Scope::ClusterShared);
        cfg.write_bandwidth_gbps = worker_bw;
        cfg.shared_write_bandwidth_gbps = shared_bw;
        let shared = G3Tier::new(cfg, 2, 1_000_000).unwrap();
        let mut r = shared.lock().unwrap();
        r.submit(0, Direction::Write, &[key(1)], 0.0).unwrap();
        r.submit(usize::from(!same_worker), Direction::Write, &[key(2)], 0.0)
            .unwrap();
        assert_eq!(r.next_deadline(0), Some(expected));
        r.advance(expected);
        assert_eq!(r.snapshot().write.completed_jobs, 2);
        assert_eq!(r.snapshot().write.transfer_ms, 2.0 * expected);
    }
}

#[test]
fn reads_and_writes_overlap_with_independent_bandwidth_after_first_byte_latency() {
    let mut cfg = config(G3Scope::ClusterShared);
    cfg.latency_to_first_byte_ms = 2.0;
    cfg.read_bandwidth_gbps = 1.0;
    cfg.write_bandwidth_gbps = 1.0;
    let shared = G3Tier::new(cfg, 2, 1_000_000).unwrap();
    let mut tier = shared.lock().unwrap();
    tier.submit(0, Direction::Write, &[key(1)], 0.0).unwrap();
    tier.take_completed(0, 3.0);
    tier.submit(0, Direction::Read, &[key(1)], 3.0).unwrap();
    tier.submit(1, Direction::Write, &[key(2)], 3.0).unwrap();
    tier.advance(4.0);
    assert_eq!(tier.snapshot().read.completed_jobs, 0);
    assert_eq!(tier.snapshot().write.completed_jobs, 1);
    let read = tier.take_completed(0, 6.0);
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].at_ms, 6.0);
    assert_eq!(tier.probe(1, key(2)), Probe::Resident);
    let write = tier.take_completed(1, 6.0);
    assert_eq!(write.len(), 1);
    assert_eq!(write[0].at_ms, 6.0);
    let stats = tier.snapshot();
    assert_eq!(stats.read.transfer_ms, 3.0);
    assert_eq!(stats.write.transfer_ms, 6.0);
}

#[test]
fn first_byte_waiters_do_not_share_bandwidth_until_ready() {
    let mut cfg = config(G3Scope::ClusterShared);
    cfg.latency_to_first_byte_ms = 2.0;
    cfg.write_bandwidth_gbps = 1.0;
    let shared = G3Tier::new(cfg, 1, 1_000_000).unwrap();
    let mut tier = shared.lock().unwrap();
    tier.submit(0, Direction::Write, &[key(1)], 0.0).unwrap();
    tier.submit(0, Direction::Write, &[key(2)], 0.5).unwrap();
    // First job moves 0.5 MB alone from 2 to 2.5 ms, then shares
    // 1 GB/s with the second job until it finishes at 3.5 ms.
    let first = tier.take_completed(0, 3.5);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].keys, vec![key(1)]);
    assert_eq!(first[0].at_ms, 3.5);
    let second = tier.take_completed(0, 4.0);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].at_ms, 4.0);
}

#[test]
fn late_cancel_preserves_other_job_latency_and_new_submissions_use_watermark() {
    let mut cfg = config(G3Scope::ClusterShared);
    cfg.latency_to_first_byte_ms = 100.0;
    let shared = G3Tier::new(cfg, 2, 100).unwrap();
    let mut r = shared.lock().unwrap();
    let first = r.submit(0, Direction::Write, &[key(1)], 0.0).unwrap();
    r.submit(1, Direction::Write, &[key(2)], 10.0).unwrap();
    r.advance(50.0);
    r.cancel(first, 10.0);
    assert_eq!(r.next_deadline(1), Some(110.0));
    r.advance(150.0);
    assert_eq!(r.take_completed(1, 150.0)[0].at_ms, 110.0);
    assert_eq!(r.snapshot().write.transfer_ms, 100.0);
    r.submit(0, Direction::Write, &[key(3)], 10.0).unwrap();
    assert_eq!(r.next_deadline(0), Some(250.0));
}

#[test]
fn large_timestamp_finishes_selected_event_without_byte_residual_spin() {
    let mut cfg = config(G3Scope::ClusterShared);
    cfg.write_bandwidth_gbps = 10.0;
    let shared = G3Tier::new(cfg, 1, 4096).unwrap();
    let mut r = shared.lock().unwrap();
    r.submit(0, Direction::Write, &[key(1)], 10000.0).unwrap();
    let deadline = r.next_deadline(0).unwrap();
    r.advance(deadline);
    assert_eq!(r.take_completed(0, deadline).len(), 1);
    assert_eq!(r.snapshot().write.completed_bytes, 4096);
    assert!(r.next_deadline(0).is_none());
}

#[test]
fn cancel_after_foreign_advance_and_before_delivery_is_terminal_once() {
    let mut cfg = config(G3Scope::ClusterShared);
    cfg.read_bandwidth_gbps = 1.0;
    let shared = G3Tier::new(cfg, 2, 1_000_000).unwrap();
    let mut r = shared.lock().unwrap();
    r.submit(0, Direction::Write, &[key(1)], 0.0).unwrap();
    r.take_completed(0, 0.0);
    let read = r.submit(1, Direction::Read, &[key(1)], 0.0).unwrap();
    r.advance(50.0);
    r.cancel(read, 10.0);
    assert_eq!(r.take_completed(1, 50.0).len(), 1);
    assert!(r.take_completed(1, 50.0).is_empty());
    assert_eq!(r.snapshot().read.completed_jobs, 1);
    assert_eq!(r.snapshot().read.cancelled_jobs, 0);
    assert_eq!(r.entries[&(0, key(1))].pins, 0);
}
