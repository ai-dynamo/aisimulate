// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;

use aisimulate_core::replay::loadgen::{AgenticGraphBuilder, WekaImporter};
use anyhow::{Context, Result, bail};
use serde_json::json;

fn main() -> Result<()> {
    let mut arguments = env::args_os().skip(1);
    let path = arguments
        .next()
        .context("usage: qualify_weka <weka-json-or-jsonl-path>")?;
    if arguments.next().is_some() {
        bail!("usage: qualify_weka <weka-json-or-jsonl-path>");
    }

    let importer = WekaImporter::open(&path)?;
    let mut builder = AgenticGraphBuilder::new(importer.header().clone())?;
    let summary = importer.for_each_row(|row| builder.push(row))?;
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
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "source_digest": summary.header.source.digest,
            "block_size": summary.header.block_size,
            "files": summary.files,
            "plays": summary.plays,
            "requests": summary.requests,
            "raw_zero_outputs": summary.raw_zero_outputs,
            "weka_nested_timestamp_basis": summary.nested_timestamp_basis,
            "graph_digest": graph.graph_digest(),
        }))?
    );
    Ok(())
}
