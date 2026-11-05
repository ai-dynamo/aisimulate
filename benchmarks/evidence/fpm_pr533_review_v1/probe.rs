// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private release example: canonical online fitting costs and Rust allocations.
//! Copy into crates/core/examples/fpm_prefix_probe.rs in each isolated snapshot.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::time::Instant;

use aisimulate_core::{ForwardPassMetrics, ForwardPassPerfModel, ForwardPassPerfModelConfig};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

struct MeasuredAllocator;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DEALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REQUESTED: AtomicU64 = AtomicU64::new(0);
static LIVE_DELTA: AtomicI64 = AtomicI64::new(0);
static PEAK_DELTA: AtomicI64 = AtomicI64::new(0);

fn change_live(delta: i64) {
    let live = LIVE_DELTA.fetch_add(delta, Ordering::Relaxed) + delta;
    PEAK_DELTA.fetch_max(live, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for MeasuredAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ACTIVE.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            REQUESTED.fetch_add(layout.size() as u64, Ordering::Relaxed);
            change_live(layout.size() as i64);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() && ACTIVE.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            REQUESTED.fetch_add(layout.size() as u64, Ordering::Relaxed);
            change_live(layout.size() as i64);
        }
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let replacement = unsafe { System.realloc(pointer, layout, size) };
        if !replacement.is_null() && ACTIVE.load(Ordering::Relaxed) {
            REALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            REQUESTED.fetch_add(size as u64, Ordering::Relaxed);
            change_live(size as i64 - layout.size() as i64);
        }
        replacement
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if ACTIVE.load(Ordering::Relaxed) {
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            change_live(-(layout.size() as i64));
        }
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: MeasuredAllocator = MeasuredAllocator;

#[derive(Serialize)]
struct AllocationSample {
    allocations: u64,
    reallocations: u64,
    deallocations: u64,
    requested_bytes: u64,
    peak_additional_live_requested_bytes: i64,
    net_live_requested_bytes: i64,
}

fn begin_allocations() {
    assert!(!ACTIVE.load(Ordering::Relaxed));
    ALLOCATIONS.store(0, Ordering::Relaxed);
    REALLOCATIONS.store(0, Ordering::Relaxed);
    DEALLOCATIONS.store(0, Ordering::Relaxed);
    REQUESTED.store(0, Ordering::Relaxed);
    LIVE_DELTA.store(0, Ordering::Relaxed);
    PEAK_DELTA.store(0, Ordering::Relaxed);
    ACTIVE.store(true, Ordering::Relaxed);
}

fn end_allocations() -> AllocationSample {
    ACTIVE.store(false, Ordering::Relaxed);
    AllocationSample {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        reallocations: REALLOCATIONS.load(Ordering::Relaxed),
        deallocations: DEALLOCATIONS.load(Ordering::Relaxed),
        requested_bytes: REQUESTED.load(Ordering::Relaxed),
        peak_additional_live_requested_bytes: PEAK_DELTA.load(Ordering::Relaxed),
        net_live_requested_bytes: LIVE_DELTA.load(Ordering::Relaxed),
    }
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn memory_kib() -> Value {
    json!({"rss_kib": null, "peak_rss_kib": null})
}

fn semantic_diagnostics(mut value: Value) -> Value {
    fn strip(value: &mut Value) {
        match value {
            Value::Object(object) => {
                object.remove("history_grouping_ms");
                object.remove("training_state_bytes");
                for child in object.values_mut() {
                    strip(child);
                }
            }
            Value::Array(values) => values.iter_mut().for_each(strip),
            _ => {}
        }
    }
    strip(&mut value);
    value
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        6,
        "probe SEED QUERIES TAIL|all UPDATES timing|allocation"
    );
    let seed = Path::new(&args[1]);
    let query_bytes = std::fs::read(&args[2]).unwrap();
    let mut observations: Vec<Vec<ForwardPassMetrics>> =
        serde_json::from_slice(&query_bytes).unwrap();
    let updates: usize = args[4].parse().unwrap();
    assert!(updates > 0 && updates <= observations.len());
    let allocation = args[5] == "allocation";
    assert!(allocation || args[5] == "timing");
    let tail = if args[3] == "all" {
        None
    } else {
        Some(args[3].parse::<usize>().unwrap())
    };
    let config_value = json!({
        "model": "benchmark/private-model", "system": "benchmark_cpu",
        "backend": "vllm", "worker_type": "decode", "estimation_mode": "fpm_learned",
        "estimator_config": {
            "correction": {"enabled": false},
            "fpm_learned": {
                "checkpoint_path": seed, "adaptation": "online", "update_mode": "structural",
                "recent_window": 64, "update_every": 1,
                "history_weight": 0.1, "history_weight_ramp_observations": 32,
                "history_bin_grouping": true, "mutable_tree_tail": tail,
                "derivative_refresh": "eager", "eager_refresh_interval": 32,
                "split_sampling_rate": 0.5, "split_rank_tolerance": 0.01
            }
        }
    });
    let config: ForwardPassPerfModelConfig = serde_json::from_value(config_value.clone()).unwrap();
    let setup = Instant::now();
    let mut model = ForwardPassPerfModel::best_available(config).unwrap();
    let setup_ns = setup.elapsed().as_nanos() as u64;
    let initial: Vec<f64> = observations
        .iter()
        .map(|query| model.estimate_forward_pass_time_ms(query).unwrap().unwrap())
        .collect();
    assert!(
        initial
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    );
    for (query, &value) in observations.iter_mut().zip(&initial) {
        query[0].wall_time = 1.3 * value / 1000.0;
    }
    let initial_diagnostics = model.learned_online_diagnostics().unwrap();
    let before_memory = memory_kib();
    let mut durations_ns = Vec::with_capacity(updates);
    let mut allocations = Vec::with_capacity(updates);
    let mut trajectory = Vec::with_capacity(updates);
    for query in observations.iter().take(updates) {
        trajectory.push(model.estimate_forward_pass_time_ms(query).unwrap().unwrap());
        if allocation {
            begin_allocations();
        }
        let start = Instant::now();
        let result = model.tune_with_fpms(std::slice::from_ref(query));
        let elapsed = start.elapsed().as_nanos() as u64;
        if allocation {
            allocations.push(end_allocations());
        }
        result.unwrap();
        durations_ns.push(elapsed);
    }
    let after_memory = memory_kib();
    let final_predictions: Vec<f64> = observations
        .iter()
        .map(|query| model.estimate_forward_pass_time_ms(query).unwrap().unwrap())
        .collect();
    let artifact = model.export_learned_artifact().unwrap();
    let diagnostics = model.learned_online_diagnostics().unwrap();
    assert_eq!(
        diagnostics["stores"]["pure_decode"]["accepted_observations"],
        updates
    );
    assert_eq!(
        diagnostics["stores"]["pure_decode"]["publications"],
        updates
    );
    let semantic = semantic_diagnostics(diagnostics.clone());
    let executable = std::env::current_exe().unwrap();
    println!(
        "{}",
        json!({
            "schema_version": 1, "phase": args[5], "updates": updates,
            "config": config_value, "binary_sha256": sha(&std::fs::read(executable).unwrap()),
            "seed_manifest_sha256": sha(&std::fs::read(seed.join("manifest.json")).unwrap()),
            "queries_sha256": sha(&query_bytes), "setup_ns": setup_ns,
            "durations_ns": durations_ns, "allocations": allocations,
            "before_memory": before_memory, "after_memory": after_memory,
            "initial_diagnostics": initial_diagnostics, "final_diagnostics": diagnostics,
            "semantic_diagnostics": semantic,
            "semantic_diagnostics_sha256": sha(&serde_json::to_vec(&semantic).unwrap()),
            "initial_predictions_sha256": sha(&serde_json::to_vec(&initial).unwrap()),
            "pre_update_trajectory_sha256": sha(&serde_json::to_vec(&trajectory).unwrap()),
            "final_predictions_sha256": sha(&serde_json::to_vec(&final_predictions).unwrap()),
            "final_artifact_sha256": sha(artifact.as_bytes()),
            "prediction_count": initial.len(), "artifact_bytes": artifact.len()
        })
    );
}
