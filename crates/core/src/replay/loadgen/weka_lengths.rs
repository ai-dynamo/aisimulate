// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Weka request lengths for host-resource admission.
//!
//! The Python resource planner bounds a Dynamo-stack Weka replay before the
//! trace is loaded. It needs only each request's input and output lengths, so
//! this scan skips hash arrays and every other field without materializing
//! them. It reports each request as it is read, so the planner can stop at its
//! headroom cutoff. It keeps the planner's field rules, which are more lenient
//! than the importer's: only `n` and `s` rows in a top-level `requests` array,
//! or in such a row's own `requests` array, count, and only their lengths and
//! the top-level block size are validated.

use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::ControlFlow;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde::de::{self, DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};

const TOKEN_LENGTH_ERROR: &str = "trace token lengths must be nonnegative integers";
const MIXED_BLOCK_SIZE_ERROR: &str = "Weka corpus mixes block sizes";

/// Report each counted request's `(input, output)` lengths in source order.
///
/// Every top-level JSON value in the file is read, and values other than
/// objects are skipped. A numeric top-level `block_size` must equal
/// `block_size`. A request is reported when its object closes, so a request
/// nested in another row precedes that row. When `on_request` breaks, the scan
/// stops without reading further input and returns `Ok`.
pub(crate) fn scan_weka_requests(
    path: &Path,
    block_size: u64,
    on_request: &mut dyn FnMut(u64, u64) -> ControlFlow<()>,
) -> Result<()> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut scan = Scan {
        block_size,
        on_request,
        stopped: false,
    };
    let mut values = 0;
    loop {
        // One deserializer per value carries the scan state; skip the
        // whitespace between values.
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            return Ok(());
        }
        let blank = buffer
            .iter()
            .take_while(|byte| matches!(byte, b' ' | b'\t' | b'\n' | b'\r'))
            .count();
        if blank > 0 {
            reader.consume(blank);
            continue;
        }
        values += 1;
        let result = Walker {
            level: Level::Document,
            scan: &mut scan,
        }
        .deserialize(&mut serde_json::Deserializer::from_reader(&mut reader));
        if scan.stopped {
            return Ok(());
        }
        result.with_context(|| format!("JSON value {values}"))?;
    }
}

struct Scan<'a> {
    block_size: u64,
    on_request: &'a mut dyn FnMut(u64, u64) -> ControlFlow<()>,
    stopped: bool,
}

#[derive(Clone, Copy)]
enum Level {
    Document,
    Requests { nested: bool },
    Row { nested: bool },
}

/// Visits one value at a known depth; values of any other shape are skipped.
struct Walker<'s, 'a> {
    level: Level,
    scan: &'s mut Scan<'a>,
}

impl<'de> DeserializeSeed<'de> for Walker<'_, '_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Walker<'_, '_> {
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
        let scan = self.scan;
        while seq
            .next_element_seed(Walker {
                level: Level::Row { nested },
                scan: &mut *scan,
            })?
            .is_some()
        {}
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let scan = self.scan;
        match self.level {
            Level::Document => {
                while let Some(key) = map.next_key::<Key>()? {
                    match key {
                        Key::BlockSize => {
                            if map.next_value::<Scalar>()?.differs_from(scan.block_size) {
                                return Err(de::Error::custom(MIXED_BLOCK_SIZE_ERROR));
                            }
                        }
                        Key::Requests => map.next_value_seed(Walker {
                            level: Level::Requests { nested: false },
                            scan: &mut *scan,
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
                            scan: &mut *scan,
                        })?,
                        _ => {
                            map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                if request {
                    let (Some(input), Some(output)) = (input, output) else {
                        return Err(de::Error::custom(TOKEN_LENGTH_ERROR));
                    };
                    if (scan.on_request)(input, output).is_break() {
                        // Abort the parse; the caller sees `stopped`, not this error.
                        scan.stopped = true;
                        return Err(de::Error::custom("scan stopped"));
                    }
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

    /// Whether a block size disagrees; non-numbers are not compared.
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

    /// Every reported request, and the error if the scan failed.
    fn scan(text: &str, block_size: u64) -> (Vec<(u64, u64)>, Option<String>) {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(text.as_bytes()).unwrap();
        let mut requests = Vec::new();
        let result = scan_weka_requests(file.path(), block_size, &mut |input, output| {
            requests.push((input, output));
            ControlFlow::Continue(())
        });
        (requests, result.err().map(|error| format!("{error:#}")))
    }

    fn counted(text: &str, block_size: u64) -> Vec<(u64, u64)> {
        let (requests, error) = scan(text, block_size);
        assert_eq!(error, None);
        requests
    }

    fn failed(text: &str, block_size: u64, expected: &str) -> Vec<(u64, u64)> {
        let (requests, error) = scan(text, block_size);
        let error = error.expect("scan should fail");
        assert!(error.contains(expected), "{error}");
        requests
    }

    fn lines(documents: &[serde_json::Value]) -> String {
        documents
            .iter()
            .map(|document| format!("{document}\n"))
            .collect()
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
        assert_eq!(counted(play, 4), [(5, 2), (4, 0), (9, 3)]);
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
        assert_eq!(counted(&lines(&[play]), 4), [(1, 1), (2, 2)]);
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
        assert_eq!(counted(play, 4), [(5, 1), (4, 1)]);
    }

    #[test]
    fn invalid_request_lengths_stop_the_scan() {
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
                assert_eq!(failed(&lines(&documents), 4, TOKEN_LENGTH_ERROR), [(1, 1)]);
            }
        }
        let missing = json!({"block_size": 4, "requests": [{"type": "s", "in": 2}]});
        assert!(failed(&lines(&[missing]), 4, TOKEN_LENGTH_ERROR).is_empty());
    }

    #[test]
    fn numeric_block_sizes_must_match() {
        // Requests before a mismatched block size in the same play still count.
        let documents = [
            r#"{"block_size": 4, "requests": [{"type": "s", "in": 1, "out": 1}]}"#,
            r#"{"block_size": "8", "requests": [{"type": "s", "in": 2, "out": 2}]}"#,
            r#"{"requests": [{"type": "s", "in": 3, "out": 3}], "block_size": 8}"#,
            r#"{"block_size": 4, "requests": [{"type": "s", "in": 4, "out": 4}]}"#,
        ]
        .join("\n");
        assert_eq!(
            failed(&documents, 4, MIXED_BLOCK_SIZE_ERROR),
            [(1, 1), (2, 2), (3, 3)]
        );
        assert!(failed(&documents, 8, MIXED_BLOCK_SIZE_ERROR).is_empty());
        let float = json!({"block_size": 4.0, "requests": [{"type": "s", "in": 1, "out": 1}]});
        assert_eq!(counted(&lines(&[float]), 4), [(1, 1)]);
    }

    #[test]
    fn reads_every_value_in_a_file() {
        let play = json!({"block_size": 4, "requests": [{"type": "s", "in": 1, "out": 1}]});
        let text = format!("{play}{play} 7 [{play}] \"x\"\n\n{play:#}");
        assert_eq!(counted(&text, 4), [(1, 1), (1, 1), (1, 1)]);
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
        assert_eq!(counted(&lines(&[play]), 64), [(100, 2)]);
    }

    #[test]
    fn syntax_errors_follow_the_requests_before_them() {
        let play = json!({"block_size": 4, "requests": [{"type": "s", "in": 1, "out": 1}]});
        assert_eq!(
            failed(&format!("{play}\nnot JSON"), 4, "JSON value 2"),
            [(1, 1)]
        );
        let truncated = r#"{"block_size": 4, "requests": [{"type": "s", "in": 2, "out": 2}, {"#;
        assert_eq!(
            failed(&format!("{play}\n{truncated}"), 4, "EOF"),
            [(1, 1), (2, 2)]
        );
    }

    #[test]
    fn breaking_stops_before_the_rest_of_the_input_is_read() {
        let text = r#"{"block_size": 4, "requests": [{"type": "s", "in": 1, "out": 1}, not JSON"#;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(text.as_bytes()).unwrap();
        let mut calls = 0;
        let result = scan_weka_requests(file.path(), 4, &mut |_, _| {
            calls += 1;
            ControlFlow::Break(())
        });
        // Parsing the malformed tail would have failed the scan.
        assert!(result.is_ok());
        assert_eq!(calls, 1);
    }

    #[test]
    fn missing_files_fail() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("missing.jsonl");
        let error = scan_weka_requests(&path, 4, &mut |_, _| ControlFlow::Continue(()));
        assert!(format!("{:#}", error.unwrap_err()).contains("missing.jsonl"));
    }
}
