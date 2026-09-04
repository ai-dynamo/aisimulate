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
    event_type: String,
    event_time_unix_ms: u64,
    #[serde(default)]
    agent_context: Option<AgentContext>,
    #[serde(default)]
    request: Option<RequestMetrics>,
    #[serde(default)]
    tool: Option<ToolMetrics>,
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

#[derive(Debug, Clone, Deserialize)]
struct ToolMetrics {
    tool_call_id: String,
    tool_class: String,
    #[serde(default)]
    claude: Option<ClaudeToolMetrics>,
    #[serde(default)]
    started_at_unix_ms: Option<u64>,
    #[serde(default)]
    ended_at_unix_ms: Option<u64>,
    #[serde(default)]
    duration_ms: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ClaudeToolMetrics {
    source_request_id: String,
    #[serde(default)]
    consumer_request_id: Option<String>,
    #[serde(default)]
    child_session_id: Option<String>,
    execution_mode: String,
}

#[derive(Debug, Clone)]
struct RequestEntry {
    start_ms: i64,
    end_ms: i64,
    agent_context: Option<AgentContext>,
    request: RequestMetrics,
}

#[derive(Debug, Clone)]
struct ToolEntry {
    session_id: String,
    tool_call_id: String,
    tool_class: String,
    claude: Option<ClaudeToolMetrics>,
}

#[derive(Debug, Default)]
struct LoadedEntries {
    requests: Vec<RequestEntry>,
    tools: Vec<ToolEntry>,
}

impl DynamoRequestTrace {
    pub fn from_request_trace_files(
        paths: &[PathBuf],
        expected_block_size: Option<usize>,
    ) -> Result<Self> {
        ensure!(!paths.is_empty(), "Dynamo trace requires at least one path");
        let loaded = load_entries(paths)?;
        let mut entries = loaded.requests;
        let contextual = entries
            .iter()
            .filter(|entry| entry.agent_context.is_some())
            .count();
        if contextual != 0 && contextual != entries.len() {
            bail!("Dynamo request trace cannot mix requests with and without agent_context");
        }
        let block_size = entries[0].request.replay.trace_block_size;
        ensure!(block_size > 0, "embedded trace block size must be positive");
        if entries
            .iter()
            .any(|entry| entry.request.replay.trace_block_size != block_size)
        {
            bail!("mixed replay trace_block_size values are not supported");
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
            lower_agentic(entries, loaded.tools, block_size).map(Self::Agentic)
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
    if matches!(event_type, Some("request_payload" | "tool_start")) {
        return Ok(None);
    }
    ensure!(
        matches!(event_type, Some("request_end" | "tool_end" | "tool_error")),
        "request trace supports request_end, terminal tool events, request_payload, and tool_start; got {event_type:?}"
    );
    let record: Record = serde_json::from_value(event.clone())?;
    ensure!(
        record.schema == "dynamo.request.trace.v1",
        "unsupported Dynamo request trace schema {:?}",
        record.schema
    );
    ensure!(
        Some(record.event_type.as_str()) == event_type,
        "record event_type changed while decoding"
    );
    Ok(Some(record))
}

fn load_entries(paths: &[PathBuf]) -> Result<LoadedEntries> {
    let mut loaded = LoadedEntries::default();
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
            if record.event_type == "request_end" {
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
                let (start_ms, end_ms) = request_times(record.event_time_unix_ms, &request)?;
                loaded.requests.push(RequestEntry {
                    start_ms,
                    end_ms,
                    agent_context: record.agent_context,
                    request,
                });
            } else if let Some(tool) = tool_entry(record)? {
                loaded.tools.push(tool);
            }
        }
    }
    ensure!(
        !loaded.requests.is_empty(),
        "Dynamo trace contains no request_end records"
    );
    Ok(loaded)
}

fn request_times(event_time_unix_ms: u64, request: &RequestMetrics) -> Result<(i64, i64)> {
    let total_ms = request
        .total_time_ms
        .map(|value| {
            ensure!(
                value.is_finite() && value >= 0.0,
                "request duration must be finite and nonnegative"
            );
            Ok(value.round() as u64)
        })
        .transpose()?;
    let end_ms = match (request.request_received_ms, total_ms) {
        (Some(start), Some(duration)) => start.saturating_add(duration),
        _ => event_time_unix_ms,
    };
    let start_ms = request
        .request_received_ms
        .unwrap_or_else(|| event_time_unix_ms.saturating_sub(total_ms.unwrap_or(0)));
    Ok((saturating_i64(start_ms), saturating_i64(end_ms)))
}

fn tool_entry(record: Record) -> Result<Option<ToolEntry>> {
    let Some(context) = record.agent_context else {
        return Ok(None);
    };
    let Some(tool) = record.tool else {
        return Ok(None);
    };
    ensure!(
        !context.session_id.trim().is_empty(),
        "tool session_id must be nonempty"
    );
    ensure!(
        !tool.tool_call_id.trim().is_empty(),
        "tool_call_id must be nonempty"
    );
    ensure!(
        !tool.tool_class.trim().is_empty(),
        "tool_class must be nonempty"
    );
    if let Some(duration) = tool.duration_ms {
        ensure!(
            duration.is_finite() && duration >= 0.0,
            "tool duration must be finite and nonnegative"
        );
    }
    let end_ms = saturating_i64(tool.ended_at_unix_ms.unwrap_or(record.event_time_unix_ms));
    let start_ms = tool
        .started_at_unix_ms
        .map(saturating_i64)
        .or_else(|| {
            tool.duration_ms
                .map(|duration| end_ms.saturating_sub(duration.round() as i64))
        })
        .unwrap_or(end_ms);
    ensure!(end_ms >= start_ms, "tool end time precedes start time");
    Ok(Some(ToolEntry {
        session_id: context.session_id,
        tool_call_id: tool.tool_call_id,
        tool_class: tool.tool_class,
        claude: tool.claude,
    }))
}

fn saturating_i64(value: u64) -> i64 {
    value.min(i64::MAX as u64) as i64
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

fn lower_agentic(
    entries: Vec<RequestEntry>,
    tools: Vec<ToolEntry>,
    block_size: usize,
) -> Result<AgenticTrace> {
    let first_start = entries
        .iter()
        .map(|entry| entry.start_ms)
        .min()
        .ok_or_else(|| anyhow!("Dynamo trace contains no requests"))?;
    let id_to_index = entries
        .iter()
        .enumerate()
        .map(|(index, entry)| (entry.request.request_id.clone(), index))
        .collect::<HashMap<_, _>>();
    let mut by_session: HashMap<String, Vec<usize>> = HashMap::new();
    let mut parent_by_session: HashMap<String, String> = HashMap::new();
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
        if let Some(parent) = context.parent_session_id.as_ref() {
            match parent_by_session.get(&context.session_id) {
                Some(existing) if existing != parent => bail!(
                    "session {:?} has conflicting parent_session_id values {:?} and {:?}",
                    context.session_id,
                    existing,
                    parent
                ),
                Some(_) => {}
                None => {
                    parent_by_session.insert(context.session_id.clone(), parent.clone());
                }
            }
        }
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
    let mut dependencies = vec![Vec::<AgenticDependency>::new(); entries.len()];
    for indices in by_session.values() {
        for pair in indices.windows(2) {
            push_dependency(
                &mut dependencies[pair[1]],
                dependency_between(
                    &entries,
                    pair[0],
                    pair[1],
                    AgenticDependencyTrigger::Completion,
                    AgenticDependencyRelation::Sequence,
                ),
            );
        }
    }

    let mut explicit_tool_by_child: HashMap<String, &ToolEntry> = HashMap::new();
    for tool in &tools {
        let Some(claude) = tool.claude.as_ref() else {
            continue;
        };
        ensure!(
            matches!(claude.execution_mode.as_str(), "blocking" | "background"),
            "tool {:?} ({}) has unsupported execution_mode {:?}",
            tool.tool_call_id,
            tool.tool_class,
            claude.execution_mode
        );
        for request_id in [
            Some(claude.source_request_id.as_str()),
            claude.consumer_request_id.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            let request_index = id_to_index.get(request_id).with_context(|| {
                format!(
                    "tool {:?} references unknown request_id {:?}",
                    tool.tool_call_id, request_id
                )
            })?;
            let request_session = &entries[*request_index]
                .agent_context
                .as_ref()
                .expect("validated agent context")
                .session_id;
            ensure!(
                request_session == &tool.session_id,
                "tool {:?} request {:?} belongs to session {:?}, expected {:?}",
                tool.tool_call_id,
                request_id,
                request_session,
                tool.session_id
            );
        }
        let Some(child_session) = claude.child_session_id.as_ref() else {
            continue;
        };
        if !by_session.contains_key(child_session) {
            continue;
        }
        ensure!(
            explicit_tool_by_child
                .insert(child_session.clone(), tool)
                .is_none(),
            "multiple tool events reference child session {:?}",
            child_session
        );
    }

    for (child_session, parent_session) in &parent_by_session {
        let child_indices = by_session
            .get(child_session)
            .expect("child session must have requests");
        let parent_indices = by_session.get(parent_session).with_context(|| {
            format!(
                "child session {:?} references unknown parent session {:?}",
                child_session, parent_session
            )
        })?;
        let first_child = child_indices[0];
        let last_child = *child_indices
            .iter()
            .max_by_key(|index| {
                let entry = &entries[**index];
                (entry.end_ms, entry.start_ms, &entry.request.request_id)
            })
            .expect("child session must be nonempty");

        if let Some(tool) = explicit_tool_by_child.get(child_session) {
            let claude = tool.claude.as_ref().expect("explicit tool has metadata");
            let parent_spawn = id_to_index[&claude.source_request_id];
            ensure!(
                parent_indices.contains(&parent_spawn),
                "tool {:?} source request {:?} is not in parent session {:?}",
                tool.tool_call_id,
                claude.source_request_id,
                parent_session
            );
            push_dependency(
                &mut dependencies[first_child],
                dependency_between(
                    &entries,
                    parent_spawn,
                    first_child,
                    AgenticDependencyTrigger::Dispatch,
                    AgenticDependencyRelation::Spawn,
                ),
            );
            if let Some(consumer) = claude.consumer_request_id.as_ref() {
                let parent_join = id_to_index[consumer];
                ensure!(
                    parent_indices.contains(&parent_join),
                    "tool {:?} consumer request {:?} is not in parent session {:?}",
                    tool.tool_call_id,
                    consumer,
                    parent_session
                );
                push_dependency(
                    &mut dependencies[parent_join],
                    dependency_between(
                        &entries,
                        last_child,
                        parent_join,
                        AgenticDependencyTrigger::Completion,
                        AgenticDependencyRelation::Join,
                    ),
                );
            }
            continue;
        }

        if let Some(parent_spawn) = parent_indices
            .iter()
            .copied()
            .filter(|index| entries[*index].start_ms <= entries[first_child].start_ms)
            .max_by_key(|index| entries[*index].start_ms)
        {
            push_dependency(
                &mut dependencies[first_child],
                dependency_between(
                    &entries,
                    parent_spawn,
                    first_child,
                    AgenticDependencyTrigger::Dispatch,
                    AgenticDependencyRelation::Spawn,
                ),
            );
        }
        if let Some(parent_join) = parent_indices
            .iter()
            .copied()
            .filter(|index| entries[*index].start_ms >= entries[last_child].end_ms)
            .min_by_key(|index| entries[*index].start_ms)
        {
            push_dependency(
                &mut dependencies[parent_join],
                dependency_between(
                    &entries,
                    last_child,
                    parent_join,
                    AgenticDependencyTrigger::Completion,
                    AgenticDependencyRelation::Join,
                ),
            );
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
            let request_dependencies = dependencies[index].clone();
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
                not_before_ms: if request_dependencies.is_empty() {
                    (entry.start_ms - first_start) as f64
                } else {
                    0.0
                },
                recorded_api_time_ms: None,
                priority: None,
                strict_priority: None,
                policy_class: None,
                dependencies: request_dependencies,
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
                digest: format!("requests:{};tools:{}", rows.len(), tools.len()),
            },
        },
        rows,
    )
}

fn dependency_between(
    entries: &[RequestEntry],
    source: usize,
    target: usize,
    trigger: AgenticDependencyTrigger,
    relation: AgenticDependencyRelation,
) -> AgenticDependency {
    let source_time = match trigger {
        AgenticDependencyTrigger::Dispatch => entries[source].start_ms,
        AgenticDependencyTrigger::Completion => entries[source].end_ms,
    };
    AgenticDependency {
        request_id: entries[source].request.request_id.clone(),
        trigger,
        delay_ms: entries[target].start_ms.saturating_sub(source_time) as f64,
        relation,
    }
}

fn push_dependency(dependencies: &mut Vec<AgenticDependency>, dependency: AgenticDependency) {
    if dependencies.iter().any(|existing| {
        existing.request_id == dependency.request_id && existing.trigger == dependency.trigger
    }) {
        return;
    }
    dependencies.push(dependency);
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

    fn child_request(
        id: &str,
        start_ms: u64,
        session_id: &str,
        parent_session_id: &str,
    ) -> serde_json::Value {
        let mut value = request(id, start_ms, Some(session_id));
        value["agent_context"]["parent_session_id"] = json!(parent_session_id);
        value
    }

    fn child_tool(
        source_request_id: &str,
        consumer_request_id: Option<&str>,
        child_session_id: &str,
        execution_mode: &str,
    ) -> serde_json::Value {
        json!({
            "schema": "dynamo.request.trace.v1",
            "event_type": "tool_end",
            "event_time_unix_ms": 115,
            "agent_context": {"session_id": "parent"},
            "tool": {
                "tool_call_id": "tool-1",
                "tool_class": "agent",
                "started_at_unix_ms": 110,
                "ended_at_unix_ms": 115,
                "claude": {
                    "source_request_id": source_request_id,
                    "consumer_request_id": consumer_request_id,
                    "child_session_id": child_session_id,
                    "execution_mode": execution_mode
                }
            }
        })
    }

    fn dependencies<'a>(trace: &'a AgenticTrace, request_id: &str) -> &'a [AgenticDependency] {
        trace
            .nodes()
            .iter()
            .find(|node| node.request_id() == request_id)
            .expect("request must exist")
            .dependencies()
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
    fn missing_duration_uses_request_end_event_time() {
        let mut first = request("a", 100, Some("session"));
        first["event_time_unix_ms"] = json!(150);
        first["request"]
            .as_object_mut()
            .unwrap()
            .remove("total_time_ms");
        let file = trace_file(&[first, request("b", 160, Some("session"))]);

        let DynamoRequestTrace::Agentic(trace) =
            DynamoRequestTrace::from_request_trace_files(&[file.path().to_path_buf()], Some(4))
                .unwrap()
        else {
            panic!("expected agentic trace");
        };

        let dependency = &dependencies(&trace, "b")[0];
        assert_eq!(dependency.request_id, "a");
        assert_eq!(dependency.delay_ms, 10.0);
    }

    #[test]
    fn blocking_child_preserves_parent_sequence_spawn_and_join() {
        let file = trace_file(&[
            request("parent-source", 100, Some("parent")),
            child_tool(
                "parent-source",
                Some("parent-consumer"),
                "child",
                "blocking",
            ),
            child_request("child-first", 120, "child", "parent"),
            child_request("child-last", 150, "child", "parent"),
            request("parent-consumer", 200, Some("parent")),
        ]);

        let DynamoRequestTrace::Agentic(trace) =
            DynamoRequestTrace::from_request_trace_files(&[file.path().to_path_buf()], Some(4))
                .unwrap()
        else {
            panic!("expected agentic trace");
        };

        assert!(
            dependencies(&trace, "child-first")
                .iter()
                .any(|dependency| {
                    dependency.request_id == "parent-source"
                        && dependency.trigger == AgenticDependencyTrigger::Dispatch
                        && dependency.relation == AgenticDependencyRelation::Spawn
                })
        );
        let consumer = dependencies(&trace, "parent-consumer");
        assert!(consumer.iter().any(|dependency| {
            dependency.request_id == "parent-source"
                && dependency.relation == AgenticDependencyRelation::Sequence
        }));
        assert!(consumer.iter().any(|dependency| {
            dependency.request_id == "child-last"
                && dependency.trigger == AgenticDependencyTrigger::Completion
                && dependency.relation == AgenticDependencyRelation::Join
        }));
    }

    #[test]
    fn background_child_launches_without_implicit_parent_join() {
        let file = trace_file(&[
            request("parent-source", 100, Some("parent")),
            child_tool("parent-source", None, "child", "background"),
            child_request("child", 120, "child", "parent"),
            request("parent-next", 130, Some("parent")),
        ]);

        let DynamoRequestTrace::Agentic(trace) =
            DynamoRequestTrace::from_request_trace_files(&[file.path().to_path_buf()], Some(4))
                .unwrap()
        else {
            panic!("expected agentic trace");
        };

        assert!(dependencies(&trace, "child").iter().any(|dependency| {
            dependency.request_id == "parent-source"
                && dependency.trigger == AgenticDependencyTrigger::Dispatch
                && dependency.relation == AgenticDependencyRelation::Spawn
        }));
        assert_eq!(dependencies(&trace, "parent-next").len(), 1);
        assert_eq!(
            dependencies(&trace, "parent-next")[0].request_id,
            "parent-source"
        );
    }

    #[test]
    fn timestamp_fallback_infers_spawn_and_last_child_join() {
        let file = trace_file(&[
            request("parent-source", 100, Some("parent")),
            child_request("child-first", 120, "child", "parent"),
            child_request("child-last", 150, "child", "parent"),
            request("parent-consumer", 200, Some("parent")),
        ]);

        let DynamoRequestTrace::Agentic(trace) =
            DynamoRequestTrace::from_request_trace_files(&[file.path().to_path_buf()], Some(4))
                .unwrap()
        else {
            panic!("expected agentic trace");
        };

        assert!(
            dependencies(&trace, "child-first")
                .iter()
                .any(|dependency| {
                    dependency.request_id == "parent-source"
                        && dependency.relation == AgenticDependencyRelation::Spawn
                })
        );
        assert!(
            dependencies(&trace, "parent-consumer")
                .iter()
                .any(|dependency| {
                    dependency.request_id == "child-last"
                        && dependency.relation == AgenticDependencyRelation::Join
                })
        );
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
