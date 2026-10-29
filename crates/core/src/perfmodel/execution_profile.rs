// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! External, immutable operation graphs bound to a compiled baseline and data.

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::common::enums::DatabaseMode;
use crate::operators::Op;
use crate::operators::measured_composite::invalid;
use crate::perfmodel::engine::spec::EngineSpec;
use crate::{
    AicError, EstimationMode, ForwardPassFallbackPolicy, ForwardPassPerfModelConfig,
    ForwardPassWorkerType,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionProfileSelection {
    pub path: PathBuf,
    /// Required content pin; loading a changed file is an error, never a refresh.
    pub sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RetainedFile {
    path: PathBuf,
    sha256: String,
    size_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionProfile {
    schema_version: u32,
    /// Exact ordinary compiled graph and identity, excluding its storage root.
    basis: EngineSpec,
    worker_type: ForwardPassWorkerType,
    ops: Vec<Op>,
    /// Aggregated profiles use `ops` for prefill and this graph for decode.
    generation_ops: Option<Vec<Op>>,
    /// Every file needed to load the private systems root, including hardware
    /// YAML, version/source metadata and all retained operation measurements.
    retained_files: Vec<RetainedFile>,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub(crate) fn validate_config(config: &ForwardPassPerfModelConfig) -> Result<(), AicError> {
    let Some(selection) = &config.estimator_config.op_level.execution_profile else {
        return Ok(());
    };
    if config.estimation_mode != EstimationMode::OpLevel
        || config.fallback_policy != ForwardPassFallbackPolicy::Deny
        || config.database_mode != DatabaseMode::Silicon
        || config.enable_shared_layer != Some(false)
        || config.estimator_config.correction.enabled
        || config.nextn != 0
        || config.speculation.is_some()
        || config.fpm_profile.is_some()
        || config.encoder_parallel.is_some()
        || config.decoder_replay
        || config.pp != 1
        || config.attention_dp != 1
        || config.cp_size.unwrap_or(1) != 1
        || config.dcp.unwrap_or(1) != 1
        || config
            .backend_version
            .as_deref()
            .is_none_or(|s| s.is_empty() || s == "latest" || s == "next")
        || !selection.path.is_absolute()
        || !is_digest(&selection.sha256)
    {
        return Err(invalid(
            "requires explicit op_level/SILICON, fallback deny, shared layer and correction disabled, PP1/DP1/CP1/DCP1, no speculation or vision encoder, a literal backend version, and an absolute path with a lowercase SHA-256 pin",
        ));
    }
    Ok(())
}

fn regular_path(root: &Path, relative: &Path) -> Result<PathBuf, AicError> {
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(invalid(
            "retained paths must be canonical bundle-relative paths",
        ));
    }
    let mut path = root.to_path_buf();
    for part in relative.components() {
        path.push(part);
        let metadata = std::fs::symlink_metadata(&path).map_err(invalid)?;
        if metadata.file_type().is_symlink() {
            return Err(invalid(
                "symlinks are not permitted inside a profile bundle",
            ));
        }
    }
    if !path.is_file() {
        return Err(invalid("retained input is not a regular file"));
    }
    Ok(path)
}

impl ExecutionProfileSelection {
    fn read(&self) -> Result<ExecutionProfile, AicError> {
        let bytes = std::fs::read(&self.path).map_err(invalid)?;
        if !is_digest(&self.sha256) || digest(&bytes) != self.sha256 {
            return Err(invalid("profile SHA-256 does not match its pin"));
        }
        let profile: ExecutionProfile = serde_json::from_slice(&bytes).map_err(invalid)?;
        if !matches!(profile.schema_version, 1 | 2)
            || profile.basis.schema_version != crate::ENGINE_SPEC_SCHEMA_VERSION
            || profile.ops.is_empty()
            || profile.retained_files.is_empty()
        {
            return Err(invalid(
                "unsupported profile schema, phase or empty graph/bundle",
            ));
        }
        if profile.worker_type == ForwardPassWorkerType::Aggregated {
            if profile.schema_version == 1 {
                return Err(invalid("aggregated profiles require schema_version 2"));
            }
            if profile.generation_ops.as_ref().is_none_or(Vec::is_empty) {
                return Err(invalid("aggregated profiles require generation_ops"));
            }
        } else if profile.generation_ops.is_some() {
            return Err(invalid("generation_ops require an aggregated profile"));
        }
        Ok(profile)
    }

    /// Admit against the actual compiler result, then copy verified bytes to an
    /// engine-owned root. A fresh root also isolates lazy/global database caches.
    pub(crate) fn apply(
        &self,
        mut spec: EngineSpec,
        systems_root: &Path,
        worker_type: ForwardPassWorkerType,
    ) -> Result<(EngineSpec, tempfile::TempDir), AicError> {
        let mut profile = self.read()?;
        if profile.worker_type != worker_type {
            return Err(invalid(
                "profile worker_type differs from the requested worker",
            ));
        }
        let source_root = spec
            .engine
            .systems_path
            .clone()
            .unwrap_or_else(|| systems_root.to_path_buf());
        spec.engine.systems_path = None;
        profile.basis.engine.systems_path = None;
        if spec != profile.basis {
            let at = first_difference(
                &serde_json::to_value(&spec).map_err(invalid)?,
                &serde_json::to_value(&profile.basis).map_err(invalid)?,
            );
            let at = if at.is_empty() {
                "an unserialized field".to_string()
            } else {
                at
            };
            return Err(invalid(format!(
                "compiled baseline differs from the profile basis at {at}; compile the basis \
                 against the retained inventory (the files in retained_files) or requalify the profile"
            )));
        }
        if spec.engine.database_mode != DatabaseMode::Silicon
            || spec.engine.enable_shared_layer != Some(false)
            || spec.engine.tolerate_dirless_version
            || spec.engine.forward_model.as_deref() != Some("op_level")
        {
            return Err(invalid(
                "baseline requires op_level SILICON with primary sources only",
            ));
        }
        let context = profile.worker_type != ForwardPassWorkerType::Decode;
        validate_replacement(&profile.ops, &spec, context)?;
        if let Some(generation) = &profile.generation_ops {
            validate_replacement(generation, &spec, false)?;
        }
        let snapshot = tempfile::Builder::new()
            .prefix("aisimulate-op-profile-")
            .tempdir()
            .map_err(invalid)?;
        let mut seen = BTreeSet::new();
        for file in profile.retained_files {
            if !seen.insert(file.path.clone()) || !is_digest(&file.sha256) {
                return Err(invalid("duplicate retained path or invalid file digest"));
            }
            let source = regular_path(&source_root, &file.path)?;
            let bytes = std::fs::read(&source).map_err(invalid)?;
            if bytes.len() as u64 != file.size_bytes || digest(&bytes) != file.sha256 {
                return Err(invalid(format!(
                    "retained input changed: {}",
                    file.path.display()
                )));
            }
            let destination = snapshot.path().join(&file.path);
            std::fs::create_dir_all(destination.parent().unwrap()).map_err(invalid)?;
            std::fs::write(destination, bytes).map_err(invalid)?;
        }
        // A pinned YAML must not redirect lazy readers outside the snapshot.
        let hardware = regular_path(
            snapshot.path(),
            Path::new(&format!("{}.yaml", spec.engine.system_name)),
        )?;
        let hardware: serde_yaml::Value =
            serde_yaml::from_slice(&std::fs::read(hardware).map_err(invalid)?).map_err(invalid)?;
        let data_dir = hardware["data_dir"]
            .as_str()
            .ok_or_else(|| invalid("hardware YAML requires data_dir"))?;
        for path in [
            Some(data_dir),
            Some(spec.engine.backend_version.as_deref().unwrap_or("")),
            hardware["misc"]["nccl_version"].as_str(),
            hardware["misc"]["oneccl_version"].as_str(),
        ]
        .into_iter()
        .flatten()
        {
            if path.is_empty()
                || Path::new(path)
                    .components()
                    .any(|part| !matches!(part, Component::Normal(_)))
            {
                return Err(invalid(
                    "hardware data_dir, backend version and communication versions must stay inside the bundle",
                ));
            }
        }
        if context {
            spec.context_ops = profile.ops;
            spec.generation_ops = profile.generation_ops.unwrap_or_default();
        } else {
            spec.generation_ops = profile.ops;
            spec.context_ops.clear();
        }
        spec.engine.extra.insert(
            "execution_profile_phase".into(),
            match profile.worker_type {
                ForwardPassWorkerType::Prefill => "prefill",
                ForwardPassWorkerType::Decode => "decode",
                ForwardPassWorkerType::Aggregated => "aggregated",
            }
            .into(),
        );
        spec.engine
            .extra
            .insert("execution_profile_sha256".into(), self.sha256.clone());
        // Keep v1 EngineSpec hashes compatible with existing saved model states.
        if profile.schema_version == 2 {
            spec.engine
                .extra
                .insert("execution_profile_schema_version".into(), "2".into());
        }
        spec.engine.systems_path = Some(snapshot.path().to_path_buf());
        Ok((spec, snapshot))
    }
}

fn validate_replacement(ops: &[Op], spec: &EngineSpec, context: bool) -> Result<(), AicError> {
    let baseline = if context {
        &spec.context_ops
    } else {
        &spec.generation_ops
    };
    let weights = |ops: &[Op]| ops.iter().map(Op::weight_bytes).sum::<f64>();
    if !weights(ops).is_finite() || (weights(baseline) - weights(ops)).abs() > 1.0 {
        return Err(invalid(
            "replacement graph must preserve the baseline resident weights",
        ));
    }
    validate_ops(ops, spec, context)
}

/// Path of the first field where the actual compiled spec differs from the
/// pinned basis, so authors see what to requalify. A frequent case is an
/// attention `lane_order` compiled against a different table inventory.
fn first_difference(actual: &serde_json::Value, pinned: &serde_json::Value) -> String {
    use serde_json::Value::{Array, Object};
    fn join(head: &str, tail: &str) -> String {
        match tail.chars().next() {
            None => head.to_string(),
            Some('[') => format!("{head}{tail}"),
            Some('(') => format!("{head} {tail}"),
            _ => format!("{head}.{tail}"),
        }
    }
    fn clip(value: &serde_json::Value) -> String {
        let text = value.to_string();
        if text.chars().count() > 120 {
            format!("{}...", text.chars().take(120).collect::<String>())
        } else {
            text
        }
    }
    match (actual, pinned) {
        (Object(x), Object(y)) => {
            let Some(key) = x
                .keys()
                .chain(y.keys())
                .find(|key| x.get(*key) != y.get(*key))
            else {
                return String::new();
            };
            match (x.get(key), y.get(key)) {
                (Some(v), Some(w)) => join(key, &first_difference(v, w)),
                _ => format!("{key} (present on one side only)"),
            }
        }
        (Array(x), Array(y)) => match x.iter().zip(y).position(|(v, w)| v != w) {
            Some(i) => join(&format!("[{i}]"), &first_difference(&x[i], &y[i])),
            None => format!("(compiled length {}, profile length {})", x.len(), y.len()),
        },
        _ => format!("(compiled {}, profile {})", clip(actual), clip(pinned)),
    }
}

fn validate_ops(ops: &[Op], spec: &EngineSpec, context: bool) -> Result<(), AicError> {
    let mut names = BTreeSet::new();
    for op in ops {
        if !names.insert(op.name()) {
            return Err(invalid("duplicate operation name"));
        }
        match op {
            Op::MeasuredComposite(op) => {
                op.validate()?;
                if op.is_context != context
                    || op.system != spec.engine.system_name
                    || op.backend != spec.engine.backend.as_str()
                    || Some(op.version.as_str()) != spec.engine.backend_version.as_deref()
                {
                    return Err(invalid("measured scope identity or phase mismatch"));
                }
            }
            Op::Overlap(op) => {
                validate_ops(&op.group_a, spec, context)?;
                validate_ops(&op.group_b, spec, context)?;
            }
            Op::Fallback(_)
            | Op::TokenScale(_)
            | Op::FpmForward(_)
            | Op::Vision(_)
            | Op::Dsv41Stage(_)
            | Op::Glm53Primitive(_)
            | Op::Glm53Ffn(_) => {
                return Err(invalid(
                    "unsupported composite in external execution profile",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::first_difference;

    #[test]
    fn first_difference_names_the_path_and_both_values() {
        assert_eq!(
            first_difference(
                &json!({"engine": {"tp_size": 1}}),
                &json!({"engine": {"tp_size": 2}})
            ),
            "engine.tp_size (compiled 1, profile 2)"
        );
        assert_eq!(
            first_difference(
                &json!({"context_ops": [{"Gemm": {"lane_order": ["fa3", "fla"]}}]}),
                &json!({"context_ops": [{"Gemm": {"lane_order": ["x", "fa3", "fla"]}}]}),
            ),
            "context_ops[0].Gemm.lane_order[0] (compiled \"fa3\", profile \"x\")"
        );
        assert_eq!(
            first_difference(&json!({"ops": [1, 2]}), &json!({"ops": [1]})),
            "ops (compiled length 2, profile length 1)"
        );
        assert_eq!(
            first_difference(&json!({"extra": {}}), &json!({"extra": {"foo": "bar"}})),
            "extra.foo (present on one side only)"
        );
        assert_eq!(first_difference(&json!({"a": 1}), &json!({"a": 1})), "");
    }
}
