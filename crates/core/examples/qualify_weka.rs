// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::ffi::OsString;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use aisimulate_core::replay::loadgen::{AgenticGraphBuilder, WekaImporter};
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

const USAGE: &str = "usage: qualify_weka <weka-json-or-jsonl-path> [--materialize <output.jsonl>]";

fn main() -> Result<()> {
    let (path, materialized) = parse_arguments(env::args_os().skip(1))?;
    let summary = qualify(&path, materialized.as_deref())?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn parse_arguments(
    mut arguments: impl Iterator<Item = OsString>,
) -> Result<(PathBuf, Option<PathBuf>)> {
    let path = arguments.next().context(USAGE)?.into();
    let materialized = match arguments.next() {
        None => None,
        Some(flag) if flag == "--materialize" => Some(arguments.next().context(USAGE)?.into()),
        Some(_) => bail!(USAGE),
    };
    if arguments.next().is_some() {
        bail!(USAGE);
    }
    Ok((path, materialized))
}

fn qualify(path: &Path, materialized: Option<&Path>) -> Result<Value> {
    let importer = WekaImporter::open(path)?;
    let mut builder = AgenticGraphBuilder::new(importer.header().clone())?;
    let mut output = materialized
        .map(|path| {
            let parent = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            tempfile::NamedTempFile::new_in(parent)
        })
        .transpose()?;
    let mut writer = output
        .as_mut()
        .map(|output| BufWriter::new(output.as_file_mut()));
    if let Some(writer) = writer.as_mut() {
        serde_json::to_writer(&mut *writer, importer.header())?;
        writeln!(writer)?;
    }
    let summary = importer.for_each_row(|row| {
        if let Some(writer) = writer.as_mut() {
            serde_json::to_writer(&mut *writer, &row)?;
            writeln!(writer)?;
        }
        builder.push(row)
    })?;
    if let Some(writer) = writer.as_mut() {
        writer.flush()?;
    }
    drop(writer);
    let graph = builder.finish()?;
    if graph.node_count() != summary.requests || graph.play_count() != summary.plays {
        bail!(
            "validated graph cardinality differs from importer summary: graph={}/{}, importer={}/{}",
            graph.play_count(),
            graph.node_count(),
            summary.plays,
            summary.requests
        );
    }
    if let (Some(output), Some(path)) = (output, materialized) {
        output
            .persist_noclobber(path)
            .with_context(|| format!("writing materialized graph to {}", path.display()))?;
    }
    Ok(json!({
        "source_digest": summary.header.source.digest,
        "block_size": summary.header.block_size,
        "files": summary.files,
        "plays": summary.plays,
        "requests": summary.requests,
        "raw_zero_outputs": summary.raw_zero_outputs,
        "weka_nested_timestamp_basis": summary.nested_timestamp_basis,
        "graph_digest": graph.graph_digest(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisimulate_core::replay::loadgen::{load_agentic_mooncake, load_weka_agentic_graph};

    #[test]
    fn materialized_output_reloads_without_changing_the_qualification_summary() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/e2e/configs/unified_cli/fixtures/traces/weka-relative.json");
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("agentic-v2.jsonl");
        let baseline = qualify(&source, None).unwrap();
        assert_eq!(qualify(&source, Some(&destination)).unwrap(), baseline);
        let direct = load_weka_agentic_graph(&source, Some(4)).unwrap();
        let reloaded = load_agentic_mooncake(&destination, 4).unwrap();
        assert_eq!(direct.identity(), reloaded.identity());
        assert_eq!(direct.nodes(), reloaded.nodes());
    }
}
