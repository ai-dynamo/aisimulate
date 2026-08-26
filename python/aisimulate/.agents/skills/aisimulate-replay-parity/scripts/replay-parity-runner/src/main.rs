// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use aisimulate_core::replay::loadgen::{Trace, WorkloadDriver};
use aisimulate_core::replay::{
    CanonicalReplayCoverage, CanonicalReplayRecord, ReplayCaptureOptions, ReplayDeterminism,
    ReplayEngineConfig, ReplayEngineFactory, ReplayRequestPool, ReplayRoutingOutcome,
    ReplayRuntimeInput, ReplaySpec, ReplayTerminalStatus, ReplayTopology, Replayer,
};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CampaignConfig {
    row_name: String,
    source_revision: String,
    trace_file: PathBuf,
    trace_sha256: String,
    trace_rows: usize,
    trace_block_size: usize,
    arrival_speedup_ratio: f64,
    trace_upstream_repository: String,
    trace_upstream_commit: String,
    trace_upstream_path: String,
    trace_full_sha256: String,
    trace_full_rows: usize,
    trace_slice_start: usize,
    #[serde(default = "one_iteration")]
    iterations: usize,
    #[serde(default = "default_true")]
    capture_canonical: bool,
    #[serde(default)]
    write_full_report: bool,
    #[serde(default = "empty_object")]
    metadata: Value,
    #[serde(default = "empty_object")]
    expected: Value,
    spec: ReplaySpec,
}

fn one_iteration() -> usize {
    1
}

fn default_true() -> bool {
    true
}

fn empty_object() -> Value {
    Value::Object(Map::new())
}

fn usage(program: &OsString) -> String {
    format!(
        "usage: {} CONFIG_JSON OUTPUT_DIRECTORY",
        Path::new(program).display()
    )
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

fn validate_sha256(value: &str, name: &str) -> Result<()> {
    ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{name} must be a 64-character hexadecimal SHA-256"
    );
    Ok(())
}

fn count_nonempty_lines(bytes: &[u8]) -> usize {
    bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
        .count()
}

fn validate_expected_subset(expected: &Value, actual: &Value, path: &str) -> Result<()> {
    match (expected, actual) {
        (Value::Object(expected), Value::Object(actual)) => {
            for (key, expected_value) in expected {
                let child_path = format!("{path}/{key}");
                let actual_value = actual
                    .get(key)
                    .with_context(|| format!("expected result is missing {child_path}"))?;
                validate_expected_subset(expected_value, actual_value, &child_path)?;
            }
            Ok(())
        }
        _ => {
            ensure!(
                expected == actual,
                "golden expectation mismatch at {path}: expected {expected}, got {actual}"
            );
            Ok(())
        }
    }
}

fn engine_block_size(spec: &ReplaySpec) -> Result<usize> {
    let engine: ReplayEngineConfig = serde_json::from_value(spec.engine.clone())
        .context("invalid ReplayEngineConfig in campaign spec")?;
    let block_size = match &spec.topology {
        ReplayTopology::Aggregated { .. } => engine.rank.block_size,
        ReplayTopology::Disaggregated { .. } => engine
            .prefill
            .as_ref()
            .map_or(engine.rank.block_size, |role| role.rank.block_size),
    };
    ensure!(block_size > 0, "engine block size must be positive");
    Ok(block_size)
}

fn canonical_metadata(config: &CampaignConfig) -> Result<Value> {
    let normalized_engine: ReplayEngineConfig = serde_json::from_value(config.spec.engine.clone())
        .context("invalid ReplayEngineConfig in campaign metadata")?;
    let mut metadata = config
        .metadata
        .as_object()
        .cloned()
        .context("campaign metadata must be a JSON object")?;
    metadata.insert(
        "campaign".to_string(),
        Value::String("aisimulate.engine-replay-parity.v1".to_string()),
    );
    metadata.insert(
        "row_name".to_string(),
        Value::String(config.row_name.clone()),
    );
    metadata.insert(
        "composition".to_string(),
        json!({"placement": "round_robin", "scaling": "none"}),
    );
    metadata.insert(
        "workload".to_string(),
        json!({
            "format": "mooncake",
            "sha256": config.trace_sha256,
            "rows": config.trace_rows,
            "block_size": config.trace_block_size,
            "arrival_speedup_ratio": config.arrival_speedup_ratio,
            "upstream": {
                "repository": config.trace_upstream_repository,
                "commit": config.trace_upstream_commit,
                "path": config.trace_upstream_path,
                "full_sha256": config.trace_full_sha256,
                "full_rows": config.trace_full_rows,
            },
            "slice": {
                "start_zero_based": config.trace_slice_start,
                "rows": config.trace_rows,
            },
        }),
    );
    metadata.insert(
        "topology".to_string(),
        serde_json::to_value(&config.spec.topology)?,
    );
    metadata.insert(
        "engine".to_string(),
        serde_json::to_value(normalized_engine)?,
    );
    Ok(Value::Object(metadata))
}

fn prepare_output_directory(path: &Path) -> Result<()> {
    if path.exists() {
        ensure!(path.is_dir(), "{} is not a directory", path.display());
        ensure!(
            fs::read_dir(path)?.next().is_none(),
            "output directory {} is not empty",
            path.display()
        );
    } else {
        fs::create_dir_all(path).with_context(|| format!("failed to create {}", path.display()))?;
    }
    Ok(())
}

fn qualification_summary(
    config: &CampaignConfig,
    report: &aisimulate_core::replay::ReplayReport,
    canonical_sha256: Option<&str>,
    measured_process_ms: f64,
    config_sha256: &str,
    binary_sha256: &str,
) -> Value {
    let mut prefill_workers = BTreeSet::new();
    let mut decode_workers = BTreeSet::new();
    let mut routing_counts = BTreeMap::<&str, usize>::new();
    let mut terminal_counts = BTreeMap::<&str, usize>::new();
    let mut requests_with_reuse = 0usize;
    let mut handoffs_complete = 0usize;
    let mut readmissions = 0usize;
    let mut max_pressure_records_per_request = 0usize;
    let mut requests_with_short_output = 0usize;
    let mut requested_output_tokens = 0usize;

    for request in &report.per_request {
        if let Some(worker) = request.prefill_worker_idx {
            prefill_workers.insert(worker);
        }
        if let Some(worker) = request.decode_worker_idx {
            decode_workers.insert(worker);
        }
        if request.reused_input_tokens > 0 {
            requests_with_reuse += 1;
        }
        requested_output_tokens += request.requested_output_length;
        if request.output_length != request.requested_output_length {
            requests_with_short_output += 1;
        }
        if request.prefill_admit_ms.is_some()
            && request.source_held_ms.is_some()
            && request.destination_reserved_ms.is_some()
            && request.destination_activated_ms.is_some()
            && request.decode_admit_ms.is_some()
            && request.source_released_ms.is_some()
        {
            handoffs_complete += 1;
        }
        readmissions += request.readmission_count;
        max_pressure_records_per_request =
            max_pressure_records_per_request.max(request.pressure_record_ordinals.len());
        for routing in &request.routing_history {
            let pool = match routing.pool {
                ReplayRequestPool::Agg => "agg",
                ReplayRequestPool::Prefill => "prefill",
                ReplayRequestPool::Decode => "decode",
            };
            let outcome = match routing.outcome {
                ReplayRoutingOutcome::Immediate => "immediate",
                ReplayRoutingOutcome::Queued => "queued",
            };
            *routing_counts
                .entry(match (pool, outcome) {
                    ("agg", "immediate") => "agg_immediate",
                    ("agg", "queued") => "agg_queued",
                    ("prefill", "immediate") => "prefill_immediate",
                    ("prefill", "queued") => "prefill_queued",
                    ("decode", "immediate") => "decode_immediate",
                    ("decode", "queued") => "decode_queued",
                    _ => unreachable!(),
                })
                .or_default() += 1;
        }
        *terminal_counts
            .entry(match request.terminal_status {
                ReplayTerminalStatus::Completed => "completed",
                ReplayTerminalStatus::Rejected => "rejected",
                ReplayTerminalStatus::Canceled => "canceled",
                ReplayTerminalStatus::Failed => "failed",
            })
            .or_default() += 1;
    }

    let pressure = report.runtime_evidence.pressure.as_ref();
    let kv_ingest = report.runtime_evidence.kv_ingest.as_ref();
    json!({
        "row_name": config.row_name,
        "source_revision": config.source_revision,
        "config_sha256": config_sha256,
        "runner_binary_sha256": binary_sha256,
        "canonical_sha256": canonical_sha256,
        "completed_requests": report.request_counts.completed_requests,
        "num_requests": report.request_counts.num_requests,
        "total_input_tokens": report.request_counts.total_input_tokens,
        "total_output_tokens": report.request_counts.total_output_tokens,
        "requested_output_tokens": requested_output_tokens,
        "requests_with_short_output": requests_with_short_output,
        "all_requests_full_output": requests_with_short_output == 0,
        "terminal_counts": terminal_counts,
        "requests_with_reuse": requests_with_reuse,
        "prefix_cache_reused_ratio": report.prefix_cache_reused_ratio,
        "first_admission_prefix_cache_reused_ratio": report.first_admission_prefix_cache_reused_ratio,
        "prefill_workers": prefill_workers,
        "decode_workers": decode_workers,
        "routing_counts": routing_counts,
        "handoffs_complete": handoffs_complete,
        "readmissions": readmissions,
        "max_pressure_records_per_request": max_pressure_records_per_request,
        "vllm_preemptions": pressure.map_or(0, |value| value.vllm_preemptions_total),
        "sglang_retractions": pressure.map_or(0, |value| value.sglang_retractions_total),
        "all_pressure_records_readmitted": pressure.is_none_or(|value| {
            value.records.iter().all(|record| record.readmitted_at_ms.is_some())
        }),
        "pressure_records": pressure.map_or(0, |value| value.records.len()),
        "lifecycle_operations": report.runtime_evidence.lifecycle_operations.len(),
        "kv_ingest_batches": kv_ingest.map_or(0, |value| value.batches),
        "kv_ingest_events": kv_ingest.map_or(0, |value| value.events),
        "virtual_duration_ms": report.throughput.duration_ms,
        "reported_wall_time_ms": report.throughput.wall_time_ms,
        "measured_process_ms": measured_process_ms,
    })
}

fn main() -> Result<()> {
    let mut args = env::args_os();
    let program = args
        .next()
        .unwrap_or_else(|| OsString::from("replay-parity-runner"));
    let config_path = args.next().context(usage(&program))?;
    let output_dir = args.next().context(usage(&program))?;
    if args.next().is_some() {
        bail!(usage(&program));
    }
    let config_path = PathBuf::from(config_path);
    let output_dir = PathBuf::from(output_dir);
    prepare_output_directory(&output_dir)?;

    let config_bytes = fs::read(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config_sha256 = sha256_bytes(&config_bytes);
    let config: CampaignConfig = serde_json::from_slice(&config_bytes)
        .with_context(|| format!("invalid campaign config {}", config_path.display()))?;
    ensure!(
        !config.row_name.trim().is_empty(),
        "row_name must be nonempty"
    );
    ensure!(
        !config.source_revision.trim().is_empty(),
        "source_revision must be nonempty"
    );
    ensure!(config.iterations > 0, "iterations must be positive");
    ensure!(
        config.trace_block_size > 0,
        "trace_block_size must be positive"
    );
    ensure!(config.trace_rows > 0, "trace_rows must be positive");
    ensure!(
        config.trace_full_rows >= config.trace_rows
            && config.trace_slice_start <= config.trace_full_rows - config.trace_rows,
        "trace slice exceeds the declared full trace"
    );
    ensure!(
        !config.trace_upstream_repository.trim().is_empty()
            && !config.trace_upstream_commit.trim().is_empty()
            && !config.trace_upstream_path.trim().is_empty(),
        "trace upstream repository, commit, and path must be nonempty"
    );
    validate_sha256(&config.trace_sha256, "trace_sha256")?;
    validate_sha256(&config.trace_full_sha256, "trace_full_sha256")?;
    ensure!(
        config.arrival_speedup_ratio.is_finite() && config.arrival_speedup_ratio > 0.0,
        "arrival_speedup_ratio must be finite and positive"
    );
    ensure!(
        config.spec.requests.is_empty(),
        "campaign spec.requests must be empty"
    );
    ensure!(
        config.spec.max_in_flight.is_none(),
        "golden trace qualification uses authored trace timestamps, not max_in_flight"
    );
    ensure!(
        config.spec.adapters.placement.provider == "round_robin",
        "campaign requires round_robin placement"
    );
    ensure!(
        config.spec.adapters.scaling.provider == "none",
        "campaign requires scaling provider none"
    );
    config.spec.validate()?;

    let trace_bytes = fs::read(&config.trace_file)
        .with_context(|| format!("failed to read {}", config.trace_file.display()))?;
    let observed_trace_sha256 = sha256_bytes(&trace_bytes);
    ensure!(
        observed_trace_sha256 == config.trace_sha256,
        "trace SHA-256 mismatch: expected {}, got {}",
        config.trace_sha256,
        observed_trace_sha256
    );
    let observed_trace_rows = count_nonempty_lines(&trace_bytes);
    ensure!(
        observed_trace_rows == config.trace_rows,
        "trace row-count mismatch: expected {}, got {}",
        config.trace_rows,
        observed_trace_rows
    );
    let trace = Trace::from_mooncake(&config.trace_file, config.trace_block_size)?
        .normalize_session_starts()?
        .speed_up_timing(config.arrival_speedup_ratio)?;
    let engine_block_size = engine_block_size(&config.spec)?;
    let metadata = canonical_metadata(&config)?;
    let capture = ReplayCaptureOptions {
        capture_per_request: true,
        capture_lifecycle_evidence: true,
        capture_canonical_evidence: true,
        determinism: ReplayDeterminism::CanonicalV1,
    };
    let binary_sha256 = sha256_file(&env::current_exe()?)?;

    let mut canonical_lines = Vec::new();
    let mut summaries = Vec::new();
    let mut full_report = None;
    let mut first_canonical_line: Option<Vec<u8>> = None;
    for iteration in 0..config.iterations {
        let workload = WorkloadDriver::new_trace(trace.clone(), engine_block_size)?;
        let started = Instant::now();
        let report = Replayer::new(config.spec.clone(), ReplayEngineFactory::new())?
            .with_runtime_input(ReplayRuntimeInput::Workload(workload))
            .with_capture_options(capture)
            .run()?;
        let measured_process_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let canonical_line = if config.capture_canonical {
            let coverage = CanonicalReplayCoverage::from_report(&report, capture);
            let line =
                CanonicalReplayRecord::build(&report, metadata.clone(), &coverage, Value::Null)?
                    .into_json_line()?;
            if let Some(first) = first_canonical_line.as_ref() {
                ensure!(
                    line == *first,
                    "canonical output changed between iterations 0 and {iteration}"
                );
            } else {
                first_canonical_line = Some(line.clone());
            }
            Some(line)
        } else {
            None
        };
        let canonical_sha256 = canonical_line.as_deref().map(sha256_bytes);
        let summary = qualification_summary(
            &config,
            &report,
            canonical_sha256.as_deref(),
            measured_process_ms,
            &config_sha256,
            &binary_sha256,
        );
        validate_expected_subset(&config.expected, &summary, "/expected")?;
        summaries.push(summary);
        if let Some(line) = canonical_line {
            canonical_lines.extend_from_slice(&line);
        }
        if config.write_full_report {
            full_report = Some(json!({
                "summary": serde_json::to_value(&report)?,
                "per_request": report.per_request,
                "runtime_evidence": report.runtime_evidence,
            }));
        }
    }

    ensure!(
        sha256_file(&config.trace_file)? == config.trace_sha256,
        "trace changed during campaign run"
    );
    if config.capture_canonical {
        fs::write(output_dir.join("canonical.jsonl"), canonical_lines)?;
    }
    let mut summary_jsonl = Vec::new();
    for summary in &summaries {
        serde_json::to_writer(&mut summary_jsonl, summary)?;
        summary_jsonl.push(b'\n');
    }
    fs::write(output_dir.join("qualification.jsonl"), summary_jsonl)?;
    if let Some(report) = full_report {
        fs::write(
            output_dir.join("report.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
    }
    println!("{}", serde_json::to_string(summaries.last().unwrap())?);
    Ok(())
}
