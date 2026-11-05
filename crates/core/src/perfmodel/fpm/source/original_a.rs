// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Original-A manifests contain bulk observations, unlike bounded bundle and
//! catalogue metadata. Retain only their dependency references while parsing
//! and verifying the complete stream.

use std::collections::BTreeSet;
use std::fmt;
use std::io::{BufReader, Read};
use std::path::Path;

use serde::de::{Error, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};

use super::{LearnedDependency, absolute, check_hash, invalid, io};
use crate::AicError;

pub(super) struct Reference {
    pub(super) path: String,
    pub(super) sha256: String,
    pub(super) checkpoint_manifest_sha256: Option<String>,
}

impl<'de> Deserialize<'de> for Reference {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ReferenceVisitor;
        impl<'de> Visitor<'de> for ReferenceVisitor {
            type Value = Reference;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an original-A dependency reference")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Reference, M::Error> {
                let mut path = None;
                let mut sha256 = None;
                let mut checkpoint_manifest_sha256 = None;
                let mut seen = BTreeSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !seen.insert(key.clone()) {
                        return Err(M::Error::custom(format!("duplicate field {key}")));
                    }
                    match key.as_str() {
                        "path" => path = Some(map.next_value()?),
                        "sha256" => sha256 = Some(map.next_value()?),
                        "checkpoint_manifest_sha256" => {
                            checkpoint_manifest_sha256 = map.next_value()?;
                        }
                        _ => {
                            map.next_value::<Discard>()?;
                        }
                    }
                }
                Ok(Reference {
                    path: path.ok_or_else(|| M::Error::missing_field("path"))?,
                    sha256: sha256.ok_or_else(|| M::Error::missing_field("sha256"))?,
                    checkpoint_manifest_sha256,
                })
            }
        }
        deserializer.deserialize_map(ReferenceVisitor)
    }
}

/// One map projection serves the root and provenance maps. Every other value
/// is consumed with bounded nesting state, not converted to a JSON Value.
#[derive(Default)]
struct Fields<const ROOT: bool> {
    sources: Option<Vec<Reference>>,
    seed_artifacts: Option<Vec<Reference>>,
    provenance: Option<Box<Fields<false>>>,
}

impl<'de, const ROOT: bool> Deserialize<'de> for Fields<ROOT> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FieldsVisitor<const ROOT: bool>;
        impl<'de, const ROOT: bool> Visitor<'de> for FieldsVisitor<ROOT> {
            type Value = Fields<ROOT>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an original-A reference object")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Fields<ROOT>, M::Error> {
                let mut fields = Fields::default();
                let mut seen = BTreeSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !seen.insert(key.clone()) {
                        return Err(M::Error::custom(format!("duplicate field {key}")));
                    }
                    match key.as_str() {
                        "sources" if !ROOT => fields.sources = Some(map.next_value()?),
                        "seed_artifacts" if ROOT => fields.seed_artifacts = Some(map.next_value()?),
                        "provenance" if ROOT => fields.provenance = Some(map.next_value()?),
                        _ => {
                            map.next_value::<Discard>()?;
                        }
                    }
                }
                Ok(fields)
            }
        }
        deserializer.deserialize_map(FieldsVisitor::<ROOT>)
    }
}

/// Preserve duplicate-key rejection even within discarded observations. Each
/// array element is released before the next is read; only open-object keys
/// and the JSON parser's bounded-depth stack remain live.
struct Discard;
impl<'de> Deserialize<'de> for Discard {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct DiscardVisitor;
        impl<'de> Visitor<'de> for DiscardVisitor {
            type Value = Discard;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value")
            }
            fn visit_unit<E: Error>(self) -> Result<Discard, E> {
                Ok(Discard)
            }
            fn visit_bool<E: Error>(self, _: bool) -> Result<Discard, E> {
                Ok(Discard)
            }
            fn visit_i64<E: Error>(self, _: i64) -> Result<Discard, E> {
                Ok(Discard)
            }
            fn visit_u64<E: Error>(self, _: u64) -> Result<Discard, E> {
                Ok(Discard)
            }
            fn visit_f64<E: Error>(self, _: f64) -> Result<Discard, E> {
                Ok(Discard)
            }
            fn visit_str<E: Error>(self, _: &str) -> Result<Discard, E> {
                Ok(Discard)
            }
            fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Discard, S::Error> {
                while seq.next_element::<Discard>()?.is_some() {}
                Ok(Discard)
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Discard, M::Error> {
                let mut seen = BTreeSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !seen.insert(key.clone()) {
                        return Err(M::Error::custom(format!("duplicate field {key}")));
                    }
                    map.next_value::<Discard>()?;
                }
                Ok(Discard)
            }
        }
        deserializer.deserialize_any(DiscardVisitor)
    }
}

struct HashedFile {
    file: std::fs::File,
    digest: Sha256,
    byte_length: u64,
}

impl Read for HashedFile {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = self.file.read(buffer)?;
        self.digest.update(&buffer[..count]);
        self.byte_length += count as u64;
        Ok(count)
    }
}

pub(super) struct References {
    pub(super) sources: Vec<Reference>,
    pub(super) seeds: Vec<Reference>,
    pub(super) manifest: LearnedDependency,
}

pub(super) fn read(path: &Path, expected: &str) -> Result<References, AicError> {
    let file = std::fs::File::open(path).map_err(|error| io(path, error))?;
    let mut reader = BufReader::with_capacity(
        65536,
        HashedFile {
            file,
            digest: Sha256::new(),
            byte_length: 0,
        },
    );
    let fields = serde_json::from_reader::<_, Fields<true>>(&mut reader);
    // A parse failure must not stop verification of later input bytes.
    std::io::copy(&mut reader, &mut std::io::sink()).map_err(|error| io(path, error))?;
    let hashed = reader.into_inner();
    let sha256 = format!("{:x}", hashed.digest.finalize());
    check_hash(expected, &sha256)?;
    let fields = fields.map_err(|error| invalid(format!("{}: {error}", path.display())))?;
    let sources = fields
        .provenance
        .and_then(|p| p.sources)
        .ok_or_else(|| invalid("original A requires source and seed references"))?;
    let seeds = fields
        .seed_artifacts
        .ok_or_else(|| invalid("original A requires source and seed references"))?;
    Ok(References {
        sources,
        seeds,
        manifest: LearnedDependency {
            path: absolute(path)?,
            byte_length: hashed.byte_length,
            sha256,
            purpose: "original_a_manifest".into(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf, String) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("original-a.json");
        std::fs::write(&path, bytes).unwrap();
        (directory, path, format!("{:x}", Sha256::digest(bytes)))
    }

    #[test]
    fn original_a_projection_retains_only_exact_references_and_hashes_all_bytes() {
        let bytes = br#"{
          "sources": ["unrelated root metadata"],
          "provenance": {"sources": [{"path":"observations.json","sha256":"source",
            "extra":{"seed_artifacts":[false,null,1,"metadata"]}}]},
          "seed_artifacts": [{"path":"seed/model.json","sha256":"seed",
            "checkpoint_manifest_sha256":"checkpoint"}],
          "accepted_iterations":[[{"wall_time":0.01,"dp_rank":0}]]
        }   
        "#;
        let (_directory, path, hash) = fixture(bytes);
        let projected = read(&path, &hash).unwrap();
        assert_eq!(projected.manifest.sha256, hash);
        assert_eq!(projected.manifest.byte_length, bytes.len() as u64);
        assert_eq!(projected.sources.len(), 1);
        assert_eq!(projected.sources[0].path, "observations.json");
        assert_eq!(projected.sources[0].sha256, "source");
        assert_eq!(projected.seeds.len(), 1);
        assert_eq!(projected.seeds[0].path, "seed/model.json");
        assert_eq!(
            projected.seeds[0].checkpoint_manifest_sha256.as_deref(),
            Some("checkpoint")
        );
        let mut changed = bytes.to_vec();
        changed.push(b' ');
        std::fs::write(&path, &changed).unwrap();
        assert!(
            read(&path, &hash)
                .err()
                .unwrap()
                .to_string()
                .contains("checksum mismatch")
        );
    }

    #[test]
    fn original_a_projection_rejects_duplicates_in_kept_and_discarded_fields() {
        for body in [
            r#""provenance":{"sources":[],"sources":[]},"seed_artifacts":[]"#,
            r#""provenance":{"sources":[{"path":"a","path":"b","sha256":"c"}]},"seed_artifacts":[]"#,
            r#""provenance":{"sources":[{"path":"a","sha256":"b","extra":1,"extra":2}]},"seed_artifacts":[]"#,
            r#""provenance":{"sources":[{"path":"a","sha256":"b","extra":{"nested":1,"nested":2}}]},"seed_artifacts":[]"#,
            r#""provenance":{"sources":[]},"seed_artifacts":[{"path":"a","sha256":"b","extra":{"nested":1,"nested":2}}]"#,
            r#""provenance":{"sources":[]},"seed_artifacts":[],"accepted_iterations":[[{"wall_time":1,"wall_time":2}]]"#,
            r#""provenance":{"sources":[]},"seed_artifacts":[],"extra":1,"extra":2"#,
        ] {
            let bytes = format!("{{{body}}}");
            let (_directory, path, hash) = fixture(bytes.as_bytes());
            let error = read(&path, &hash).err().unwrap().to_string();
            assert!(error.contains("duplicate field"), "{error}");
        }
    }

    #[test]
    fn original_a_parse_failure_still_checks_unread_tail_and_required_references() {
        let mut malformed = b"{!".to_vec();
        malformed.extend(std::iter::repeat_n(b' ', 2 * 65536));
        let (_directory, path, hash) = fixture(&malformed);
        let error = read(&path, &hash).err().unwrap().to_string();
        assert!(error.contains("key must be a string"), "{error}");
        malformed.push(b' ');
        std::fs::write(&path, &malformed).unwrap();
        let error = read(&path, &hash).err().unwrap().to_string();
        assert!(error.contains("checksum mismatch"), "{error}");
        for bytes in [
            r#"{"provenance":{"sources":[]}}"#,
            r#"{"provenance":{"sources":[{"path":"a"}]},"seed_artifacts":[]}"#,
            r#"{"provenance":{"sources":[]},"seed_artifacts":[]} false"#,
        ] {
            let (_directory, path, hash) = fixture(bytes.as_bytes());
            assert!(read(&path, &hash).is_err());
        }
    }

    #[test]
    fn original_a_streaming_does_not_relax_small_metadata_limit() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("bundle.json");
        std::fs::File::create(&path)
            .unwrap()
            .set_len(super::super::MANIFEST_LIMIT + 1)
            .unwrap();
        let error = super::super::read_json(&path).unwrap_err().to_string();
        assert!(error.contains("manifest exceeds 16 MiB"), "{error}");
    }
}
