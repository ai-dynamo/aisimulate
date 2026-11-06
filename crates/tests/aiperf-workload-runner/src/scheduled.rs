// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduled source preparation through the owning native input adapters.

use aiperf_engine_input::dataset_input::{
    BuiltinRunnerDatasetInputAdapterResolver, DatasetInputAdapterResolver, DatasetInputContext,
    PreparedDatasetInput,
};
use aiperf_exchange_model::TextTokenizer;
use anyhow::{Context, Result, ensure};
use std::sync::Arc;

/// Load synthetic and ordinary conversation sources without reimplementing acquisition or composition.
pub(crate) fn prepare(
    run: &aiperf_config::model::BenchmarkRun,
    dataset: &serde_json::value::RawValue,
    models: &aiperf_protocol::protocol_dto::ModelsSpec,
    tokenizer: &dyn TextTokenizer,
    cancellation: aiperf_clock_runtime::cancellation::RunCancellation,
) -> Result<PreparedDatasetInput> {
    let authored: serde_json::Value = serde_json::from_str(dataset.get())?;
    ensure!(
        ["images", "audio", "video", "rankings"]
            .iter()
            .all(|field| authored.get(*field).is_none_or(serde_json::Value::is_null)),
        "scheduled simulator proof currently requires text-only sources"
    );
    let registry = aiperf_endpoints::EndpointRegistry::builtin()?;
    let endpoint_id = aiperf_registry::registry_id::RegistryId::new("chat")?;
    let endpoint = registry.prepare(
        &endpoint_id,
        aiperf_endpoints::RawEndpointConfig {
            streaming: true,
            use_server_token_count: false,
            ..Default::default()
        },
    )?;
    let formats = aiperf_dataset_loaders::LoaderRegistry::with_core_formats()?;
    let context = DatasetInputContext {
        cancellation: Some(cancellation),
        dataset_formats: &formats,
        models,
        run_rng_root: aiperf_rng::RngRoot::new(run.random_seed),
        tokenizer,
        rankings: false,
        endpoint_descriptor: endpoint.descriptor(),
        trace_prompt_storage: Arc::new(
            aiperf_dataset_loaders::compose::MaterializedTracePromptStorage,
        ),
        media_generator_factory: Arc::new(
            aiperf_dataset_synthesis::generator::NativeSyntheticMediaGeneratorFactory::default(),
        ),
        // The explicit builtin tokenizer has no chat template. The shared
        // adapter itself adds authored cache-bust overhead during composition.
        prompt_budget: Default::default(),
        trace_window: run.cfg.trace_timestamp_window().map_or_else(
            Default::default,
            |(start, end)| aiperf_dataset::TraceWindow {
                start_ms: start.map(|value| value as f64),
                end_ms: end.map(|value| value as f64),
            },
        ),
        profiling_request_cap: run.cfg.profiling_request_cap(),
        load_timing: None,
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("creating scheduled source preparation runtime")?;
    let prepared = tokio::task::LocalSet::new().block_on(&runtime, async {
        BuiltinRunnerDatasetInputAdapterResolver::new()
            .load(dataset, &context)
            .await
    })?;
    Ok(prepared)
}
