// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming trace metadata for host-memory admission in `aisimulate.resources`.
//!
//! Admission bounds a finite Mooncake-family or Dynamo trace replay before the
//! loader materializes it. This pass reads each JSON value in turn, counts the
//! hash, output and prompt totals those loaders retain, and skips every other
//! value without allocating it.

use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use flate2::read::MultiGzDecoder;
use serde::Deserialize;
use serde::de::{
    self, DeserializeOwned, DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess,
    Visitor,
};

pub(super) const INVALID_LENGTH: &str = "trace token lengths must be nonnegative integers";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TraceRows {
    /// `mooncake`, `mooncake-delta` and `agentic_mooncake` rows.
    Mooncake,
    /// Dynamo request-trace records, direct or wrapped in `event`.
    Dynamo,
}

/// Mirrors `_TraceCounters` in `aisimulate.resources`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct TraceCounters {
    pub hashes: u64,
    pub authored_outputs: u64,
    pub outputs: u64,
    pub largest_input: u64,
    pub total_input: u64,
    pub requests: u64,
}

/// Stop once the running lower bound exceeds the live headroom.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Headroom {
    pub limit: Option<i64>,
    pub retains_source_hashes: bool,
    pub materializes_inputs: bool,
}

impl Headroom {
    /// Mirrors `_TraceCounters.lower_bound` in `aisimulate.resources`.
    fn exceeded_by(&self, counters: &TraceCounters) -> bool {
        let Some(limit) = self.limit else {
            return false;
        };
        let [hashes, authored, outputs, largest, total] = [
            counters.hashes,
            counters.authored_outputs,
            counters.outputs,
            counters.largest_input,
            counters.total_input,
        ]
        .map(i128::from);
        let loaded = if self.retains_source_hashes {
            8 * hashes + 4 * authored
        } else {
            0
        };
        let prepared = if self.materializes_inputs {
            (4 * hashes).max(4 * (total + outputs))
        } else {
            4 * (hashes + outputs)
        };
        loaded.max(prepared).max(4 * largest) > i128::from(limit)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TraceScanOutcome {
    Complete,
    /// The lower bound exceeded the headroom; later rows were not inspected.
    Stopped,
    /// A standard Dynamo request followed an agentic one, or the reverse.
    MixedAgentContext,
}

#[derive(Debug)]
pub(super) enum TraceScanError {
    Open(std::io::Error),
    Invalid(String),
}

/// Count `paths` in order. Mooncake rows without an input length are sized as
/// their hash count times `omitted_input_block_size`.
pub(super) fn scan_trace(
    rows: TraceRows,
    paths: &[impl AsRef<Path>],
    omitted_input_block_size: u64,
    headroom: Headroom,
) -> Result<(TraceCounters, TraceScanOutcome), TraceScanError> {
    let mut counters = TraceCounters::default();
    let mut agentic = None;
    for path in paths {
        let path = path.as_ref();
        let file = File::open(path).map_err(TraceScanError::Open)?;
        let reader: Box<dyn Read> =
            if rows == TraceRows::Dynamo && path.extension().is_some_and(|ext| ext == "gz") {
                Box::new(MultiGzDecoder::new(file))
            } else {
                Box::new(file)
            };
        let reader = BufReader::with_capacity(64 * 1024, reader);
        let outcome = match rows {
            TraceRows::Mooncake => each_value(reader, |Row(row)| {
                row.map_or(Ok(TraceScanOutcome::Complete), |row| {
                    add_mooncake(&mut counters, row, omitted_input_block_size, headroom)
                })
            }),
            TraceRows::Dynamo => each_value(reader, |Row(row)| {
                row.map_or(Ok(TraceScanOutcome::Complete), |row| {
                    add_dynamo(&mut counters, &mut agentic, row, headroom)
                })
            }),
        }?;
        if outcome != TraceScanOutcome::Complete {
            return Ok((counters, outcome));
        }
    }
    Ok((counters, TraceScanOutcome::Complete))
}

/// Visit whitespace-separated JSON values until one ends the scan.
fn each_value<V: DeserializeOwned>(
    reader: impl Read,
    mut add: impl FnMut(V) -> Result<TraceScanOutcome, TraceScanError>,
) -> Result<TraceScanOutcome, TraceScanError> {
    for value in serde_json::Deserializer::from_reader(reader).into_iter::<V>() {
        let value = value.map_err(|error| TraceScanError::Invalid(error.to_string()))?;
        let outcome = add(value)?;
        if outcome != TraceScanOutcome::Complete {
            return Ok(outcome);
        }
    }
    Ok(TraceScanOutcome::Complete)
}

fn add_mooncake(
    counters: &mut TraceCounters,
    row: MooncakeRow,
    omitted_input_block_size: u64,
    headroom: Headroom,
) -> Result<TraceScanOutcome, TraceScanError> {
    if row.invalid_length {
        return Err(TraceScanError::Invalid(INVALID_LENGTH.into()));
    }
    let length = row
        .input
        .unwrap_or_else(|| row.hashes.saturating_mul(omitted_input_block_size));
    counters.largest_input = counters.largest_input.max(length);
    counters.total_input = counters.total_input.saturating_add(length);
    counters.hashes += row.hashes;
    counters.authored_outputs += row.authored_outputs;
    counters.outputs = counters.outputs.saturating_add(row.outputs);
    counters.requests += u64::from(row.is_request);
    Ok(stop_if_exceeded(counters, headroom))
}

fn add_dynamo(
    counters: &mut TraceCounters,
    agentic: &mut Option<bool>,
    row: DynamoRow,
    headroom: Headroom,
) -> Result<TraceScanOutcome, TraceScanError> {
    let fields = if row.wrapped { row.wrapper } else { row.direct };
    if !fields.request_end {
        return Ok(TraceScanOutcome::Complete);
    }
    // The first request selects standard or agentic mode for every file.
    let contextual = fields.agent_context.unwrap_or(false);
    if agentic.is_some_and(|mode| mode != contextual) {
        return Ok(TraceScanOutcome::MixedAgentContext);
    }
    *agentic = Some(contextual);
    let (Length::Valid(outputs), Length::Valid(input)) =
        (fields.output_tokens, fields.input_length)
    else {
        return Err(TraceScanError::Invalid(INVALID_LENGTH.into()));
    };
    counters.hashes += fields.hashes;
    counters.outputs = counters.outputs.saturating_add(outputs);
    counters.largest_input = counters.largest_input.max(input);
    counters.requests += 1;
    Ok(stop_if_exceeded(counters, headroom))
}

fn stop_if_exceeded(counters: &TraceCounters, headroom: Headroom) -> TraceScanOutcome {
    if headroom.exceeded_by(counters) {
        TraceScanOutcome::Stopped
    } else {
        TraceScanOutcome::Complete
    }
}

/// Accept every scalar JSON value as `$value`.
macro_rules! accept_scalars {
    ($value:expr) => {
        fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E> {
            Ok($value)
        }

        fn visit_i64<E>(self, _: i64) -> Result<Self::Value, E> {
            Ok($value)
        }

        fn visit_u64<E>(self, _: u64) -> Result<Self::Value, E> {
            Ok($value)
        }

        fn visit_f64<E>(self, _: f64) -> Result<Self::Value, E> {
            Ok($value)
        }

        fn visit_str<E>(self, _: &str) -> Result<Self::Value, E> {
            Ok($value)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok($value)
        }
    };
}

/// The shape of an inspected JSON value. Containers are skipped unallocated.
enum Shape {
    Integer(u64),
    Negative,
    Float(f64),
    Null,
    Object,
    RequestEnd,
    Other,
}

impl Shape {
    fn is_number(&self) -> bool {
        matches!(self, Self::Integer(_) | Self::Negative | Self::Float(_))
    }

    fn length(&self) -> Length {
        match *self {
            Self::Integer(value) => Length::Valid(value),
            // serde_json reads an integer literal beyond u64 as a float. Keep
            // it above every headroom, like the unbounded Python integer.
            Self::Float(value) if value >= u64::MAX as f64 => Length::Valid(u64::MAX),
            _ => Length::Invalid,
        }
    }
}

impl<'de> Deserialize<'de> for Shape {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ShapeVisitor;

        impl<'de> Visitor<'de> for ShapeVisitor {
            type Value = Shape;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("any JSON value")
            }

            fn visit_bool<E>(self, _: bool) -> Result<Shape, E> {
                Ok(Shape::Other)
            }

            fn visit_i64<E>(self, value: i64) -> Result<Shape, E> {
                Ok(u64::try_from(value).map_or(Shape::Negative, Shape::Integer))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Shape, E> {
                Ok(Shape::Integer(value))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Shape, E> {
                Ok(Shape::Float(value))
            }

            fn visit_str<E>(self, value: &str) -> Result<Shape, E> {
                Ok(if value == "request_end" {
                    Shape::RequestEnd
                } else {
                    Shape::Other
                })
            }

            fn visit_unit<E>(self) -> Result<Shape, E> {
                Ok(Shape::Null)
            }

            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Shape, A::Error> {
                IgnoredAny.visit_seq(seq)?;
                Ok(Shape::Other)
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Shape, A::Error> {
                IgnoredAny.visit_map(map)?;
                Ok(Shape::Object)
            }
        }

        deserializer.deserialize_any(ShapeVisitor)
    }
}

#[derive(Clone, Copy, Debug, Default)]
enum Length {
    /// Missing, or not a nonnegative integer.
    #[default]
    Invalid,
    Valid(u64),
}

/// The number of numeric items directly inside an array value.
struct NumberCount(u64);

impl<'de> Deserialize<'de> for NumberCount {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct CountVisitor;

        impl<'de> Visitor<'de> for CountVisitor {
            type Value = NumberCount;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("any JSON value")
            }

            accept_scalars!(NumberCount(0));

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<NumberCount, A::Error> {
                let mut count = 0;
                while let Some(item) = seq.next_element::<Shape>()? {
                    count += u64::from(item.is_number());
                }
                Ok(NumberCount(count))
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<NumberCount, A::Error> {
                IgnoredAny.visit_map(map)?;
                Ok(NumberCount(0))
            }
        }

        deserializer.deserialize_any(CountVisitor)
    }
}

/// Field names that Mooncake rows and Dynamo records are counted by.
enum Key {
    HashIds,
    OutputTokenIds,
    InputLength,
    InputTokens,
    OutputLength,
    OutputTokens,
    Event,
    EventType,
    AgentContext,
    Request,
    Replay,
    InputSequenceHashes,
    Other,
}

impl<'de> Deserialize<'de> for Key {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct KeyVisitor;

        impl Visitor<'_> for KeyVisitor {
            type Value = Key;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an object key")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Key, E> {
                Ok(match value {
                    "hash_ids" => Key::HashIds,
                    "output_token_ids" => Key::OutputTokenIds,
                    "input_length" => Key::InputLength,
                    "input_tokens" => Key::InputTokens,
                    "output_length" => Key::OutputLength,
                    "output_tokens" => Key::OutputTokens,
                    "event" => Key::Event,
                    "event_type" => Key::EventType,
                    "agent_context" => Key::AgentContext,
                    "request" => Key::Request,
                    "replay" => Key::Replay,
                    "input_sequence_hashes" => Key::InputSequenceHashes,
                    _ => Key::Other,
                })
            }
        }

        deserializer.deserialize_identifier(KeyVisitor)
    }
}

/// Fields of a JSON object that the scan inspects.
trait ObjectFields {
    fn field<'de, A: MapAccess<'de>>(&mut self, key: Key, map: &mut A) -> Result<(), A::Error>;
}

/// Read an object into `T`, or skip any other JSON value; yields whether it was an object.
struct Fields<'a, T>(&'a mut T);

impl<'de, T: ObjectFields> DeserializeSeed<'de> for Fields<'_, T> {
    type Value = bool;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<bool, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de, T: ObjectFields> Visitor<'de> for Fields<'_, T> {
    type Value = bool;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("any JSON value")
    }

    accept_scalars!(false);

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<bool, A::Error> {
        IgnoredAny.visit_seq(seq)?;
        Ok(false)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<bool, A::Error> {
        while let Some(key) = map.next_key()? {
            self.0.field(key, &mut map)?;
        }
        Ok(true)
    }
}

/// One top-level JSON value; only objects are rows.
struct Row<T>(Option<T>);

impl<'de, T: ObjectFields + Default> Deserialize<'de> for Row<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut row = T::default();
        let is_object = Fields(&mut row).deserialize(deserializer)?;
        Ok(Row(is_object.then_some(row)))
    }
}

fn skip<'de, A: MapAccess<'de>>(map: &mut A) -> Result<(), A::Error> {
    map.next_value::<IgnoredAny>().map(drop)
}

#[derive(Default)]
struct MooncakeRow {
    /// Any hash, input or output field makes the object a request.
    is_request: bool,
    invalid_length: bool,
    /// The last input length wins; every output length is added.
    input: Option<u64>,
    outputs: u64,
    hashes: u64,
    authored_outputs: u64,
}

impl ObjectFields for MooncakeRow {
    fn field<'de, A: MapAccess<'de>>(&mut self, key: Key, map: &mut A) -> Result<(), A::Error> {
        match key {
            Key::HashIds => {
                self.is_request = true;
                self.hashes += map.next_value::<NumberCount>()?.0;
            }
            Key::OutputTokenIds => self.authored_outputs += map.next_value::<NumberCount>()?.0,
            Key::InputLength | Key::InputTokens => {
                self.is_request = true;
                match map.next_value::<Shape>()?.length() {
                    Length::Valid(length) => self.input = Some(length),
                    Length::Invalid => self.invalid_length = true,
                }
            }
            Key::OutputLength | Key::OutputTokens => {
                self.is_request = true;
                match map.next_value::<Shape>()?.length() {
                    Length::Valid(length) => self.outputs = self.outputs.saturating_add(length),
                    Length::Invalid => self.invalid_length = true,
                }
            }
            _ => skip(map)?,
        }
        Ok(())
    }
}

#[derive(Default)]
struct DynamoRow {
    /// An `event` field selects the wrapped record, whatever its value.
    wrapped: bool,
    direct: DynamoFields,
    wrapper: DynamoFields,
}

impl ObjectFields for DynamoRow {
    fn field<'de, A: MapAccess<'de>>(&mut self, key: Key, map: &mut A) -> Result<(), A::Error> {
        if let Key::Event = key {
            self.wrapped = true;
            map.next_value_seed(Fields(&mut self.wrapper)).map(drop)
        } else {
            self.direct.field(key, map)
        }
    }
}

/// A record's fields; repeated fields keep their last value and add hashes.
#[derive(Default)]
struct DynamoFields {
    request_end: bool,
    /// An object marks an agentic request and null a standard one.
    agent_context: Option<bool>,
    output_tokens: Length,
    input_length: Length,
    hashes: u64,
}

impl ObjectFields for DynamoFields {
    fn field<'de, A: MapAccess<'de>>(&mut self, key: Key, map: &mut A) -> Result<(), A::Error> {
        match key {
            Key::EventType => self.request_end = matches!(map.next_value()?, Shape::RequestEnd),
            Key::AgentContext => match map.next_value()? {
                Shape::Object => self.agent_context = Some(true),
                Shape::Null => self.agent_context = Some(false),
                _ => {}
            },
            Key::Request => drop(map.next_value_seed(Fields(&mut DynamoRequest(self)))?),
            _ => skip(map)?,
        }
        Ok(())
    }
}

/// `request`, holding `output_tokens` and `replay`.
struct DynamoRequest<'a>(&'a mut DynamoFields);

impl ObjectFields for DynamoRequest<'_> {
    fn field<'de, A: MapAccess<'de>>(&mut self, key: Key, map: &mut A) -> Result<(), A::Error> {
        match key {
            Key::OutputTokens => self.0.output_tokens = map.next_value::<Shape>()?.length(),
            Key::Replay => drop(map.next_value_seed(Fields(&mut DynamoReplay(self.0)))?),
            _ => skip(map)?,
        }
        Ok(())
    }
}

/// `request.replay`, holding `input_length` and `input_sequence_hashes`.
struct DynamoReplay<'a>(&'a mut DynamoFields);

impl ObjectFields for DynamoReplay<'_> {
    fn field<'de, A: MapAccess<'de>>(&mut self, key: Key, map: &mut A) -> Result<(), A::Error> {
        match key {
            Key::InputLength => self.0.input_length = map.next_value::<Shape>()?.length(),
            Key::InputSequenceHashes => self.0.hashes += map.next_value::<NumberCount>()?.0,
            _ => skip(map)?,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn scan(
        rows: TraceRows,
        contents: &str,
        limit: Option<i64>,
    ) -> (TraceCounters, TraceScanOutcome) {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        let headroom = Headroom {
            limit,
            ..Headroom::default()
        };
        scan_trace(rows, &[file.path()], 16, headroom).unwrap()
    }

    #[test]
    fn mooncake_rows_count_direct_numbers_and_size_omitted_prompts() {
        let contents = concat!(
            r#"{"hash_ids":[1,"2",[3],{"h":4},5.5],"output_length":2,"output_tokens":3}"#,
            "\n",
            r#"{"input_length":5,"input_tokens":7,"input_length":9,"output_token_ids":[1,2],"other":[[1]]}"#,
            " [1] 2 \"s\" null ",
            r#"{"output_length":18446744073709551616}{"meta":{}}"#,
        );
        let (counters, outcome) = scan(TraceRows::Mooncake, contents, None);
        assert_eq!(outcome, TraceScanOutcome::Complete);
        assert_eq!(
            counters,
            TraceCounters {
                hashes: 2,
                authored_outputs: 2,
                outputs: u64::MAX,
                largest_input: 32,
                total_input: 41,
                requests: 3,
            }
        );
    }

    #[test]
    fn dynamo_records_follow_their_wrapper_and_stop_at_the_headroom() {
        let direct = r#"{"event_type":"request_end","agent_context":null,"request":{"output_tokens":3,"replay":{"input_length":8,"input_sequence_hashes":[1,2]}}}"#;
        let wrapped = r#"{"event_type":"request_end","event":{"event_type":"request_end","request":{"output_tokens":1,"replay":{"input_length":16,"input_sequence_hashes":[3]}}}}"#;
        let payload = r#"{"event_type":"request_payload","request":{"output_tokens":-1}}"#;
        let contents = [payload, direct, wrapped, "not JSON"].join("\n");
        // The 16-token wrapped prompt lifts the bound past 40; the invalid last line is never read.
        let (counters, outcome) = scan(TraceRows::Dynamo, &contents, Some(40));
        assert_eq!(outcome, TraceScanOutcome::Stopped);
        assert_eq!(
            (counters.requests, counters.hashes, counters.outputs),
            (2, 3, 4)
        );
        assert_eq!(counters.largest_input, 16);

        let agentic = direct.replace("null", r#"{"session_id":"s"}"#);
        let (_, outcome) = scan(TraceRows::Dynamo, &[direct, &agentic].join("\n"), None);
        assert_eq!(outcome, TraceScanOutcome::MixedAgentContext);
    }
}
