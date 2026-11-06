// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Weka request lengths for host-resource admission.
//!
//! The Python resource planner bounds a Dynamo-stack Weka replay before the
//! trace is loaded. It needs each request's input and output lengths and the
//! corpus block size, so this scan skips hash arrays and every other field
//! without materializing them. It keeps the planner's field rules, which are
//! more lenient than the importer's: only `n` and `s` rows in a top-level
//! `requests` array, or in such a row's own `requests` array, count, and only
//! their lengths are validated.

use std::fmt;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::Path;

use serde::Deserialize;
use serde::de::{self, DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};

const TOKEN_LENGTH_ERROR: &str = "trace token lengths must be nonnegative integers";
const POSITIVE_BLOCK_SIZE_ERROR: &str = "Weka trace requires a positive block_size";
const MIXED_BLOCK_SIZE_ERROR: &str = "Weka corpus mixes block sizes";

/// Request lengths read from one Weka source file.
#[derive(Debug, PartialEq)]
pub(crate) struct WekaLengthScan {
    /// The supplied block size, or the file's first top-level `block_size`.
    /// `None` only when no valid first block size exists; `error` says why.
    pub(crate) block_size: Option<u64>,
    /// `(input, output)` lengths of counted requests in source order, up to
    /// the first error.
    pub(crate) lengths: Vec<(u64, u64)>,
    /// The first validation or JSON error.
    pub(crate) error: Option<String>,
}

/// Read the request lengths of every JSON value in one Weka source file.
///
/// With no supplied block size, the file's first top-level `block_size` must
/// be a positive integer; it is found even after a length error, as long as
/// the JSON before it parses. Later numeric block sizes must match. A
/// document with a JSON syntax error contributes none of its requests.
pub(crate) fn scan_weka_request_lengths(
    path: &Path,
    block_size: Option<u64>,
) -> io::Result<WekaLengthScan> {
    let reader = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut scan = WekaLengthScan {
        block_size,
        lengths: Vec::new(),
        error: None,
    };
    let mut counting = true;
    for document in serde_json::Deserializer::from_reader(reader).into_iter::<Document>() {
        let events = match document {
            Ok(Document(events)) => events,
            Err(error) => {
                if counting || scan.block_size.is_none() {
                    scan.error = Some(error.to_string());
                }
                return Ok(scan);
            }
        };
        for event in events {
            match event {
                Event::BlockSize(value) => match scan.block_size {
                    None => match value {
                        Scalar::Unsigned(size) if size > 0 => scan.block_size = Some(size),
                        // The block size is read before any request is counted,
                        // so its error replaces an earlier length error.
                        Scalar::Unsigned(_) => {
                            return Ok(block_size_error(POSITIVE_BLOCK_SIZE_ERROR));
                        }
                        _ => return Ok(block_size_error(TOKEN_LENGTH_ERROR)),
                    },
                    Some(expected) => {
                        if counting && value.differs_from(expected) {
                            scan.error = Some(MIXED_BLOCK_SIZE_ERROR.to_string());
                            counting = false;
                        }
                    }
                },
                Event::Request(input, output) if counting => match (input, output) {
                    (Some(input), Some(output)) => scan.lengths.push((input, output)),
                    _ => {
                        scan.error = Some(TOKEN_LENGTH_ERROR.to_string());
                        counting = false;
                    }
                },
                Event::Request(..) => {}
            }
        }
        if !counting && scan.block_size.is_some() {
            return Ok(scan);
        }
    }
    if scan.block_size.is_none() {
        return Ok(block_size_error(POSITIVE_BLOCK_SIZE_ERROR));
    }
    Ok(scan)
}

fn block_size_error(message: &str) -> WekaLengthScan {
    WekaLengthScan {
        block_size: None,
        lengths: Vec::new(),
        error: Some(message.to_string()),
    }
}

/// One top-level JSON value's block sizes and request rows, in source order.
/// A row is recorded when its object closes, after any rows nested in it.
struct Document(Vec<Event>);

enum Event {
    BlockSize(Scalar),
    /// An `n` or `s` row's lengths; `None` is a missing or invalid length.
    Request(Option<u64>, Option<u64>),
}

impl<'de> Deserialize<'de> for Document {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut events = Vec::new();
        Walker {
            level: Level::Document,
            events: &mut events,
        }
        .deserialize(deserializer)?;
        Ok(Self(events))
    }
}

#[derive(Clone, Copy)]
enum Level {
    Document,
    Requests { nested: bool },
    Row { nested: bool },
}

/// Visits one value at a known depth; values of any other shape are skipped.
struct Walker<'a> {
    level: Level,
    events: &'a mut Vec<Event>,
}

impl<'de> DeserializeSeed<'de> for Walker<'_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Walker<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E: de::Error>(self, _: bool) -> Result<(), E> {
        Ok(())
    }

    fn visit_i64<E: de::Error>(self, _: i64) -> Result<(), E> {
        Ok(())
    }

    fn visit_u64<E: de::Error>(self, _: u64) -> Result<(), E> {
        Ok(())
    }

    fn visit_f64<E: de::Error>(self, _: f64) -> Result<(), E> {
        Ok(())
    }

    fn visit_str<E: de::Error>(self, _: &str) -> Result<(), E> {
        Ok(())
    }

    fn visit_unit<E: de::Error>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        let Level::Requests { nested } = self.level else {
            while seq.next_element::<IgnoredAny>()?.is_some() {}
            return Ok(());
        };
        let events = self.events;
        while seq
            .next_element_seed(Walker {
                level: Level::Row { nested },
                events: &mut *events,
            })?
            .is_some()
        {}
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let events = self.events;
        match self.level {
            Level::Document => {
                while let Some(key) = map.next_key::<Key>()? {
                    match key {
                        Key::BlockSize => events.push(Event::BlockSize(map.next_value()?)),
                        Key::Requests => map.next_value_seed(Walker {
                            level: Level::Requests { nested: false },
                            events: &mut *events,
                        })?,
                        _ => {
                            map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
            }
            Level::Row { nested } => {
                // Repeated keys keep their last value.
                let (mut request, mut input, mut output) = (false, None, None);
                while let Some(key) = map.next_key::<Key>()? {
                    match key {
                        Key::Type => {
                            request = matches!(map.next_value()?, Scalar::RequestType);
                        }
                        Key::In => input = map.next_value::<Scalar>()?.length(),
                        Key::Out => output = map.next_value::<Scalar>()?.length(),
                        Key::Requests if !nested => map.next_value_seed(Walker {
                            level: Level::Requests { nested: true },
                            events: &mut *events,
                        })?,
                        _ => {
                            map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                if request {
                    events.push(Event::Request(input, output));
                }
            }
            Level::Requests { .. } => {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
            }
        }
        Ok(())
    }
}

enum Key {
    BlockSize,
    Requests,
    Type,
    In,
    Out,
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

            fn visit_str<E: de::Error>(self, key: &str) -> Result<Key, E> {
                Ok(match key {
                    "block_size" => Key::BlockSize,
                    "requests" => Key::Requests,
                    "type" => Key::Type,
                    "in" => Key::In,
                    "out" => Key::Out,
                    _ => Key::Other,
                })
            }
        }

        deserializer.deserialize_str(KeyVisitor)
    }
}

/// The parts of a scalar field the planner checks.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Scalar {
    Unsigned(u64),
    /// A negative or non-integer number, or an integer beyond `u64`.
    Number(f64),
    /// The string `n` or `s`.
    RequestType,
    Other,
}

impl Scalar {
    fn length(self) -> Option<u64> {
        match self {
            Self::Unsigned(length) => Some(length),
            _ => None,
        }
    }

    /// Whether a later block size disagrees; non-numbers are not compared.
    fn differs_from(self, expected: u64) -> bool {
        match self {
            Self::Unsigned(value) => value != expected,
            Self::Number(value) => value != expected as f64,
            Self::RequestType | Self::Other => false,
        }
    }
}

impl<'de> Deserialize<'de> for Scalar {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ScalarVisitor;

        impl<'de> Visitor<'de> for ScalarVisitor {
            type Value = Scalar;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value")
            }

            fn visit_bool<E: de::Error>(self, _: bool) -> Result<Scalar, E> {
                Ok(Scalar::Other)
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Scalar, E> {
                Ok(u64::try_from(value).map_or(Scalar::Number(value as f64), Scalar::Unsigned))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Scalar, E> {
                Ok(Scalar::Unsigned(value))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Scalar, E> {
                Ok(Scalar::Number(value))
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Scalar, E> {
                Ok(if matches!(value, "n" | "s") {
                    Scalar::RequestType
                } else {
                    Scalar::Other
                })
            }

            fn visit_unit<E: de::Error>(self) -> Result<Scalar, E> {
                Ok(Scalar::Other)
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Scalar, A::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(Scalar::Other)
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Scalar, A::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(Scalar::Other)
            }
        }

        deserializer.deserialize_any(ScalarVisitor)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use serde_json::json;

    use super::*;

    fn scan(text: &str, block_size: Option<u64>) -> WekaLengthScan {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(text.as_bytes()).unwrap();
        scan_weka_request_lengths(file.path(), block_size).unwrap()
    }

    fn lines(documents: &[serde_json::Value]) -> String {
        documents
            .iter()
            .map(|document| format!("{document}\n"))
            .collect()
    }

    fn counted(block_size: u64, lengths: &[(u64, u64)]) -> WekaLengthScan {
        WekaLengthScan {
            block_size: Some(block_size),
            lengths: lengths.to_vec(),
            error: None,
        }
    }

    fn failed(block_size: Option<u64>, lengths: &[(u64, u64)], error: &str) -> WekaLengthScan {
        WekaLengthScan {
            block_size,
            lengths: lengths.to_vec(),
            error: Some(error.to_string()),
        }
    }

    #[test]
    fn counts_normal_streaming_and_nested_rows_regardless_of_field_order() {
        // Raw text keeps `block_size` last; `json!` would sort the keys.
        let play = r#"{
            "id": "play",
            "requests": [
                {"hash_ids": [1, 2], "out": 2, "in": 5, "type": "s"},
                {"type": "n", "in": 4, "out": 0, "hash_ids": []},
                {
                    "requests": [{"type": "s", "in": 9, "out": 3, "hash_ids": [9]}],
                    "type": "subagent",
                    "total_tokens": 1000000000000
                }
            ],
            "totals": {"in": 1000000000000, "out": "ignored"},
            "block_size": 4
        }"#;
        assert_eq!(scan(play, None), counted(4, &[(5, 2), (4, 0), (9, 3)]));
    }

    #[test]
    fn nested_rows_precede_their_parent_and_deeper_rows_are_ignored() {
        let play = json!({
            "block_size": 4,
            "requests": [{
                "type": "n", "in": 2, "out": 2,
                "requests": [{
                    "type": "s", "in": 1, "out": 1,
                    "requests": [{"type": "s", "in": 7, "out": 7}],
                }],
            }],
        });
        assert_eq!(scan(&lines(&[play]), None), counted(4, &[(1, 1), (2, 2)]));
    }

    #[test]
    fn only_request_rows_validate_their_lengths() {
        // Raw text keeps repeated keys, which take their last value.
        let play = r#"{
            "block_size": 4,
            "type": "s", "in": -1,
            "requests": [
                {"type": "subagent", "in": "x", "out": -1, "requests": {"type": "s", "in": -1}},
                {"in": -1, "out": 1},
                {"type": "x", "in": 1.5, "out": 1},
                {"type": "s", "in": 3, "out": 1, "type": ["n"]},
                {"type": "x", "in": -1, "out": 1, "type": "n", "in": 5},
                [{"type": "s", "in": -1, "out": 1}],
                7,
                {"type": "s", "in": 3, "out": 1, "in": 4}
            ]
        }"#;
        assert_eq!(scan(play, None), counted(4, &[(5, 1), (4, 1)]));
    }

    #[test]
    fn invalid_request_lengths_stop_counting() {
        for invalid in [
            json!(-1),
            json!(1.5),
            json!("1"),
            json!(null),
            json!(true),
            json!([1]),
        ] {
            for field in ["in", "out"] {
                let mut row = json!({"type": "s", "in": 2, "out": 2});
                row[field] = invalid.clone();
                let documents = [
                    json!({"block_size": 4, "requests": [{"type": "n", "in": 1, "out": 1}, row]}),
                    json!({"requests": [{"type": "s", "in": 3, "out": 3}]}),
                ];
                assert_eq!(
                    scan(&lines(&documents), None),
                    failed(Some(4), &[(1, 1)], TOKEN_LENGTH_ERROR)
                );
            }
        }
        let missing = json!({"block_size": 4, "requests": [{"type": "s", "in": 2}]});
        assert_eq!(
            scan(&lines(&[missing]), None),
            failed(Some(4), &[], TOKEN_LENGTH_ERROR)
        );
    }

    #[test]
    fn later_numeric_block_sizes_must_match() {
        // Requests before a mismatched block size in the same play still count.
        let documents = [
            r#"{"block_size": 4, "requests": [{"type": "s", "in": 1, "out": 1}]}"#,
            r#"{"block_size": "8", "requests": [{"type": "s", "in": 2, "out": 2}]}"#,
            r#"{"requests": [{"type": "s", "in": 3, "out": 3}], "block_size": 8}"#,
            r#"{"block_size": 4, "requests": [{"type": "s", "in": 4, "out": 4}]}"#,
        ]
        .join("\n");
        assert_eq!(
            scan(&documents, None),
            failed(Some(4), &[(1, 1), (2, 2), (3, 3)], MIXED_BLOCK_SIZE_ERROR)
        );
        // A supplied size comes from an earlier file of the same corpus.
        assert_eq!(
            scan(&documents, Some(8)),
            failed(Some(8), &[], MIXED_BLOCK_SIZE_ERROR)
        );
        let float = json!({"block_size": 4.0, "requests": [{"type": "s", "in": 1, "out": 1}]});
        assert_eq!(scan(&lines(&[float]), Some(4)), counted(4, &[(1, 1)]));
    }

    #[test]
    fn the_first_block_size_must_be_positive() {
        let row = json!({"type": "s", "in": 1, "out": 1});
        for (block_size, error) in [
            (Some(json!(0)), POSITIVE_BLOCK_SIZE_ERROR),
            (Some(json!(-4)), TOKEN_LENGTH_ERROR),
            (Some(json!("4")), TOKEN_LENGTH_ERROR),
            (Some(json!(4.0)), TOKEN_LENGTH_ERROR),
            (None, POSITIVE_BLOCK_SIZE_ERROR),
        ] {
            let mut play = json!({"requests": [row]});
            if let Some(block_size) = block_size {
                play["block_size"] = block_size;
            }
            assert_eq!(scan(&lines(&[play]), None), failed(None, &[], error));
        }
        // The first block size is found after a later document, and after a
        // length error that stops counting.
        let documents = [
            json!({"requests": [row, {"type": "s", "in": -1, "out": 1}, row]}),
            json!({"requests": [row], "block_size": 4}),
        ];
        assert_eq!(
            scan(&lines(&documents), None),
            failed(Some(4), &[(1, 1)], TOKEN_LENGTH_ERROR)
        );
        let documents = [
            json!({"requests": [{"type": "s", "in": -1, "out": 1}]}),
            json!({"block_size": 0}),
        ];
        assert_eq!(
            scan(&lines(&documents), None),
            failed(None, &[], POSITIVE_BLOCK_SIZE_ERROR)
        );
    }

    #[test]
    fn reads_every_value_in_a_file() {
        let play = json!({"block_size": 4, "requests": [{"type": "s", "in": 1, "out": 1}]});
        let text = format!("{play}{play} 7 [{play}] \"x\"\n\n{play:#}");
        assert_eq!(scan(&text, None), counted(4, &[(1, 1), (1, 1), (1, 1)]));
    }

    #[test]
    fn skips_hash_arrays_without_validating_them() {
        let mut hashes: Vec<serde_json::Value> = (0..100_000_u64).map(Into::into).collect();
        hashes.extend([json!(-1), json!("x"), json!(1e30), json!({"in": -1})]);
        let play = json!({
            "block_size": 64,
            "hash_ids": [-1],
            "requests": [{"type": "s", "in": 100, "out": 2, "hash_ids": hashes}],
        });
        assert_eq!(scan(&lines(&[play]), None), counted(64, &[(100, 2)]));
    }

    #[test]
    fn syntax_errors_keep_earlier_documents() {
        let play = json!({"block_size": 4, "requests": [{"type": "s", "in": 1, "out": 1}]});
        let truncated = r#"{"block_size": 4, "requests": [{"type": "s", "in": 2, "out": 2}, {"#;
        for tail in ["not JSON", truncated] {
            let result = scan(&format!("{play}\n{tail}"), None);
            assert_eq!((result.block_size, result.lengths), (Some(4), vec![(1, 1)]));
            assert!(result.error.is_some());
        }
        // Without a block size, the syntax error is the reason.
        let result = scan(&format!("{}\n{truncated}", json!({"requests": []})), None);
        assert_eq!((result.block_size, result.lengths), (None, vec![]));
        assert!(result.error.unwrap().contains("EOF"));
    }

    #[test]
    fn missing_files_are_io_errors() {
        let directory = tempfile::tempdir().unwrap();
        let error = scan_weka_request_lengths(&directory.path().join("missing.jsonl"), None);
        assert_eq!(error.unwrap_err().kind(), io::ErrorKind::NotFound);
    }
}
