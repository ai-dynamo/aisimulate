// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo request-trace-v1 loading without a dependency on Dynamo crates.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail, ensure};
use flate2::read::MultiGzDecoder;
use serde::Deserialize;

use super::trace::assign_dependency_component_play_ids;
use super::{
    AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION, AgenticDependency,
    AgenticDependencyRelation, AgenticDependencyTrigger, AgenticHashIdScope, AgenticMooncakeHeader,
    AgenticMooncakeRow, AgenticSourceProvenance, AgenticTrace, MooncakeRow, Trace,
};

#[derive(Debug, Clone, PartialEq)]
pub enum DynamoRequestTrace {
    Standard(Trace),
    Agentic(AgenticTrace),
}

#[derive(Debug, Clone, Deserialize)]
struct Record {
    schema: String,
    event_time_unix_ms: u64,
    #[serde(default)]
    agent_context: Option<AgentContext>,
    #[serde(default)]
    request: Option<RequestMetrics>,
}

#[derive(Debug, Clone, Deserialize)]
struct AgentContext {
    session_id: String,
    #[serde(default)]
    parent_session_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RequestMetrics {
    request_id: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    request_received_ms: Option<u64>,
    #[serde(default)]
    total_time_ms: Option<f64>,
    replay: ReplayMetrics,
}

#[derive(Debug, Clone, Deserialize)]
struct ReplayMetrics {
    trace_block_size: usize,
    input_length: usize,
    input_sequence_hashes: Vec<u64>,
}

#[derive(Debug, Clone)]
struct RequestEntry {
    start_ms: i64,
    end_ms: i64,
    agent_context: Option<AgentContext>,
    request: RequestMetrics,
}

impl DynamoRequestTrace {
    pub fn from_request_trace_files(
        paths: &[PathBuf],
        expected_block_size: Option<usize>,
    ) -> Result<Self> {
        ensure!(!paths.is_empty(), "Dynamo trace requires at least one path");
        let mut entries = load_entries(paths)?;
        let contextual = entries
            .iter()
            .filter(|entry| entry.agent_context.is_some())
            .count();
        if contextual != 0 && contextual != entries.len() {
            bail!("Dynamo request trace cannot mix standard and agentic requests");
        }
        let block_size = entries[0].request.replay.trace_block_size;
        ensure!(block_size > 0, "embedded trace block size must be positive");
        if entries
            .iter()
            .any(|entry| entry.request.replay.trace_block_size != block_size)
        {
            bail!("Dynamo request trace contains mixed trace block sizes");
        }
        if let Some(expected) = expected_block_size {
            ensure!(
                expected == block_size,
                "trace_block_size {expected} does not match embedded Dynamo request trace block size {block_size}"
            );
        }
        entries.sort_by(|left, right| {
            (left.start_ms, left.end_ms, &left.request.request_id).cmp(&(
                right.start_ms,
                right.end_ms,
                &right.request.request_id,
            ))
        });
        if contextual == 0 {
            lower_standard(entries, block_size).map(Self::Standard)
        } else {
            lower_agentic(entries, block_size).map(Self::Agentic)
        }
    }
}

fn open_reader(path: &PathBuf) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader: Box<dyn Read> = if path.extension().and_then(|value| value.to_str()) == Some("gz") {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(Box::new(BufReader::new(reader)))
}

fn parse_record(line: &str) -> Result<Option<Record>> {
    let value: serde_json::Value = serde_json::from_str(line)?;
    let Some(object) = value.as_object() else {
        return Ok(None);
    };
    let event = object.get("event").unwrap_or(&value);
    let event_type = event
        .get("event_type")
        .and_then(serde_json::Value::as_str)
        .or_else(|| object.get("event_type").and_then(serde_json::Value::as_str));
    if event_type == Some("request_payload") {
        return Ok(None);
    }
    if event_type != Some("request_end") {
        return Ok(None);
    }
    let record: Record = serde_json::from_value(event.clone())?;
    ensure!(
        record.schema == "dynamo.request.trace.v1",
        "unsupported Dynamo request trace schema {:?}",
        record.schema
    );
    Ok(Some(record))
}

fn load_entries(paths: &[PathBuf]) -> Result<Vec<RequestEntry>> {
    let mut entries = Vec::new();
    let mut request_ids = HashSet::new();
    for path in paths {
        for (line_index, line) in open_reader(path)?.lines().enumerate() {
            let line = line
                .with_context(|| format!("failed to read {}:{}", path.display(), line_index + 1))?;
            if line.trim().is_empty() {
                continue;
            }
            let Some(record) = parse_record(&line).with_context(|| {
                format!("failed to parse {}:{}", path.display(), line_index + 1)
            })?
            else {
                continue;
            };
            let request = record
                .request
                .context("request_end is missing request metrics")?;
            ensure!(
                !request.request_id.trim().is_empty(),
                "request_id must be nonempty"
            );
            ensure!(
                request_ids.insert(request.request_id.clone()),
                "duplicate request_id {:?}",
                request.request_id
            );
            let start_ms = request
                .request_received_ms
                .unwrap_or(record.event_time_unix_ms) as i64;
            let duration_ms = request.total_time_ms.unwrap_or(0.0);
            ensure!(
                duration_ms.is_finite() && duration_ms >= 0.0,
                "request duration must be finite and nonnegative"
            );
            entries.push(RequestEntry {
                start_ms,
                end_ms: start_ms.saturating_add(duration_ms.round() as i64),
                agent_context: record.agent_context,
                request,
            });
        }
    }
    ensure!(
        !entries.is_empty(),
        "Dynamo trace contains no request_end records"
    );
    Ok(entries)
}

fn lower_standard(entries: Vec<RequestEntry>, block_size: usize) -> Result<Trace> {
    let first_start = entries
        .iter()
        .map(|entry| entry.start_ms)
        .min()
        .ok_or_else(|| anyhow!("Dynamo trace contains no requests"))?;
    let rows = entries
        .into_iter()
        .map(|entry| -> Result<MooncakeRow> {
            Ok(MooncakeRow {
                request_id: Some(entry.request.request_id),
                input_length: Some(entry.request.replay.input_length),
                output_length: Some(
                    usize::try_from(
                        entry
                            .request
                            .output_tokens
                            .context("missing output_tokens")?,
                    )
                    .context("output_tokens does not fit usize")?,
                ),
                hash_ids: Some(entry.request.replay.input_sequence_hashes),
                timestamp: Some((entry.start_ms - first_start) as f64),
                ..Default::default()
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Trace::from_mooncake_rows(rows, block_size)
}

fn lower_agentic(entries: Vec<RequestEntry>, block_size: usize) -> Result<AgenticTrace> {
    let first_start = entries
        .iter()
        .map(|entry| entry.start_ms)
        .min()
        .ok_or_else(|| anyhow!("Dynamo trace contains no requests"))?;
    let mut by_session: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, entry) in entries.iter().enumerate() {
        let context = entry
            .agent_context
            .as_ref()
            .context("agentic request is missing agent_context")?;
        ensure!(
            !context.session_id.trim().is_empty(),
            "session_id must be nonempty"
        );
        by_session
            .entry(context.session_id.clone())
            .or_default()
            .push(index);
    }
    for indices in by_session.values_mut() {
        indices.sort_by_key(|index| {
            let entry = &entries[*index];
            (
                entry.start_ms,
                entry.end_ms,
                entry.request.request_id.clone(),
            )
        });
    }
    let mut previous = vec![None; entries.len()];
    for indices in by_session.values() {
        for pair in indices.windows(2) {
            previous[pair[1]] = Some(pair[0]);
        }
    }
    // A child session's first request waits for the latest parent request that
    // began before it. This preserves the causal spawn edge without importing
    // Dynamo's producer-side data-gen crate.
    for indices in by_session.values() {
        let first = indices[0];
        let Some(parent_session) = entries[first]
            .agent_context
            .as_ref()
            .and_then(|context| context.parent_session_id.as_ref())
        else {
            continue;
        };
        if let Some(parent_indices) = by_session.get(parent_session)
            && let Some(parent) = parent_indices
                .iter()
                .copied()
                .filter(|index| entries[*index].start_ms <= entries[first].start_ms)
                .max_by_key(|index| entries[*index].start_ms)
        {
            previous[first] = Some(parent);
        }
    }
    let mut rows = entries
        .iter()
        .enumerate()
        .map(|(index, entry)| -> Result<AgenticMooncakeRow> {
            let context = entry
                .agent_context
                .as_ref()
                .expect("validated agent context");
            let dependencies =
                previous[index]
                    .map(|parent| {
                        let relation = if entries[parent].agent_context.as_ref().is_some_and(
                            |parent_context| parent_context.session_id != context.session_id,
                        ) {
                            AgenticDependencyRelation::Spawn
                        } else {
                            AgenticDependencyRelation::Sequence
                        };
                        AgenticDependency {
                            request_id: entries[parent].request.request_id.clone(),
                            trigger: AgenticDependencyTrigger::Completion,
                            delay_ms: entry.start_ms.saturating_sub(entries[parent].end_ms) as f64,
                            relation,
                        }
                    })
                    .into_iter()
                    .collect();
            Ok(AgenticMooncakeRow {
                request_id: entry.request.request_id.clone(),
                play_id: "dynamo-request-trace".to_string(),
                session_id: context.session_id.clone(),
                model: entry
                    .request
                    .model
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                input_length: Some(entry.request.replay.input_length),
                output_length: Some(
                    usize::try_from(
                        entry
                            .request
                            .output_tokens
                            .context("missing output_tokens")?,
                    )
                    .context("output_tokens does not fit usize")?,
                ),
                output_token_ids: None,
                hash_ids: Some(entry.request.replay.input_sequence_hashes.clone()),
                not_before_ms: if previous[index].is_none() {
                    (entry.start_ms - first_start) as f64
                } else {
                    0.0
                },
                priority: None,
                strict_priority: None,
                policy_class: None,
                dependencies,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    assign_dependency_component_play_ids(&mut rows, "dynamo-play");
    AgenticTrace::from_agentic_mooncake_rows(
        AgenticMooncakeHeader {
            schema: AGENTIC_MOONCAKE_SCHEMA.to_string(),
            version: AGENTIC_MOONCAKE_VERSION,
            block_size,
            hash_id_scope: AgenticHashIdScope::Local,
            source: AgenticSourceProvenance {
                format: "dynamo.request.trace.v1".to_string(),
                digest: format!("requests:{}", rows.len()),
            },
        },
        rows,
    )
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use serde_json::json;
    use tempfile::NamedTempFile;

    use super::*;

    fn request(id: &str, start_ms: u64, session_id: Option<&str>) -> serde_json::Value {
        let mut value = json!({
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": start_ms + 10,
            "request": {
                "request_id": id,
                "output_tokens": 2,
                "request_received_ms": start_ms,
                "total_time_ms": 10,
                "replay": {
                    "trace_block_size": 4,
                    "input_length": 4,
                    "input_sequence_hashes": [11]
                }
            }
        });
        if let Some(session_id) = session_id {
            value["agent_context"] = json!({"session_id": session_id});
        }
        value
    }

    fn trace_file(rows: &[serde_json::Value]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for row in rows {
            writeln!(file, "{}", serde_json::to_string(row).unwrap()).unwrap();
        }
        file
    }

    #[test]
    fn loads_standard_multi_file_trace() {
        let first = trace_file(&[request("a", 100, None)]);
        let second = trace_file(&[request("b", 120, None)]);
        let loaded = DynamoRequestTrace::from_request_trace_files(
            &[first.path().to_path_buf(), second.path().to_path_buf()],
            Some(4),
        )
        .unwrap();
        let DynamoRequestTrace::Standard(trace) = loaded else {
            panic!("expected standard trace");
        };
        assert_eq!(trace.sessions.len(), 2);
        assert_eq!(trace.sessions[0].first_arrival_timestamp_ms, Some(0.0));
        assert_eq!(trace.sessions[1].first_arrival_timestamp_ms, Some(20.0));
    }

    #[test]
    fn loads_agentic_trace_as_dependency_graph() {
        let file = trace_file(&[
            request("a", 100, Some("session")),
            request("b", 120, Some("session")),
        ]);
        let loaded =
            DynamoRequestTrace::from_request_trace_files(&[file.path().to_path_buf()], Some(4))
                .unwrap();
        let DynamoRequestTrace::Agentic(trace) = loaded else {
            panic!("expected agentic trace");
        };
        assert_eq!(trace.node_count(), 2);
        assert_eq!(trace.play_count(), 1);
    }

    #[test]
    fn independent_agent_sessions_become_independent_plays() {
        let file = trace_file(&[
            request("a", 100, Some("session-a")),
            request("b", 120, Some("session-b")),
        ]);
        let loaded =
            DynamoRequestTrace::from_request_trace_files(&[file.path().to_path_buf()], Some(4))
                .unwrap();
        let DynamoRequestTrace::Agentic(trace) = loaded else {
            panic!("expected agentic trace");
        };
        assert_eq!(trace.node_count(), 2);
        assert_eq!(trace.play_count(), 2);
    }
}
