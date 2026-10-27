// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! State snapshot behavior: every model kind restores bit-exactly through the
//! canonical construction path, continues tuning exactly like the saving
//! model, and every malformed input fails closed without panicking.
//!
//! Expected values are never recomputed from production formulas. Each check
//! compares a restored model with the uninterrupted original on the same
//! inputs (`f64::to_bits`, errors included), and every round trip includes a
//! cold control that must differ, proving the comparison can fail.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parquet::data_type::{DoubleType, Int64Type};
use parquet::file::{properties::WriterProperties, writer::SerializedFileWriter};
use parquet::schema::parser::parse_message_type;

use super::*;
use crate::fpm::tests::TEST_MODEL;
use crate::{
    BackendKind, FpmLearnedConfig, LearnedAdaptation, LearnedDerivativeRefresh, LearnedUpdateMode,
    LinearFitConfig, RegressionFeatureAxis, RegressionFitKind, RegressionUpdatePolicy,
    ScheduledRequestMetrics, SplineFitConfig, SplineSearchConfig,
};

type Iteration = Vec<ForwardPassMetrics>;
type Loader<'a> = &'a dyn Fn(ForwardPassPerfModelConfig) -> Result<ForwardPassPerfModel, AicError>;

const AGG: ForwardPassWorkerType = ForwardPassWorkerType::Aggregated;
const DECODE: ForwardPassWorkerType = ForwardPassWorkerType::Decode;

// ---------------------------------------------------------------------------
// Deterministic inputs
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn between(&mut self, low: u32, high: u32) -> u32 {
        low + (self.next() % u64::from(high - low)) as u32
    }
}

fn rank(prefill: u32, decode: u32, kv: u32, wall_ms: f64, requests: bool) -> ForwardPassMetrics {
    let past = if decode > 0 { kv / decode } else { 0 };
    let mut scheduled = ScheduledRequestMetrics {
        num_prefill_requests: u32::from(prefill > 0),
        sum_prefill_tokens: prefill,
        sum_prefill_kv_tokens: prefill / 4,
        num_decode_requests: decode,
        sum_decode_kv_tokens: if decode > 0 { kv } else { 0 },
        ..Default::default()
    };
    if requests {
        let mut extend = Vec::new();
        let mut kv_lengths = Vec::new();
        if prefill > 0 {
            extend.push(u64::from(prefill));
            kv_lengths.push(u64::from(prefill / 4));
        }
        for _ in 0..decode {
            extend.push(1);
            kv_lengths.push(u64::from(past));
        }
        scheduled.extend_lengths = Some(extend);
        scheduled.past_kv_lengths = Some(kv_lengths);
    }
    ForwardPassMetrics {
        wall_time: wall_ms / 1000.0,
        scheduled_requests: scheduled,
        ..Default::default()
    }
}

/// A smooth, mildly nonlinear latency surface with deterministic noise.
fn latency(prefill: u32, decode: u32, kv: u32, noise: u64) -> f64 {
    let (p, d, k) = (f64::from(prefill), f64::from(decode), f64::from(kv));
    2.0 + 4e-4 * p + 2e-5 * k + 1e-2 * d + 1e-8 * p * p + (noise % 97) as f64 * 1e-3
}

/// Iterations for one worker role. Aggregated rotates through pure decode,
/// pure prefill, locally mixed and (optionally) cross-rank compositions.
/// Coordinates exceed the default fixed correction limits (8192 tokens,
/// batch 512, 2M KV tokens) so hybrid's dynamic branch is exercised.
fn stream(
    worker: ForwardPassWorkerType,
    seed: u64,
    count: usize,
    cross_rank: bool,
    requests: bool,
) -> Vec<Iteration> {
    let mut lcg = Lcg(seed);
    (0..count)
        .map(|index| {
            let prefill = lcg.between(64, 16_384);
            let decode = lcg.between(1, 700);
            let kv = decode * lcg.between(64, 4_096);
            let noise = lcg.next();
            let shape = match worker {
                ForwardPassWorkerType::Prefill => 1,
                ForwardPassWorkerType::Decode => 0,
                ForwardPassWorkerType::Aggregated => index % if cross_rank { 4 } else { 3 },
            };
            match shape {
                0 => vec![rank(0, decode, kv, latency(0, decode, kv, noise), requests)],
                1 => vec![rank(prefill, 0, 0, latency(prefill, 0, 0, noise), requests)],
                2 => vec![rank(
                    prefill,
                    decode,
                    kv,
                    latency(prefill, decode, kv, noise),
                    requests,
                )],
                _ => vec![
                    rank(prefill, 0, 0, latency(prefill, 0, 0, noise) / 2.0, requests),
                    rank(0, decode, kv, latency(prefill, decode, kv, noise), requests),
                ],
            }
        })
        .collect()
}

/// Inside and outside every fixed correction range, plus idle and invalid
/// queries whose errors must also match.
fn queries(worker: ForwardPassWorkerType, requests: bool) -> Vec<Iteration> {
    let mut out = vec![vec![rank(0, 0, 0, 0.0, requests)]];
    for (decode, kv) in [(1, 64), (32, 40_000), (256, 600_000), (600, 2_400_000)] {
        if worker != ForwardPassWorkerType::Prefill {
            out.push(vec![rank(0, decode, kv, 0.0, requests)]);
        }
        for prefill in [128, 4_096, 12_000] {
            match worker {
                ForwardPassWorkerType::Prefill => {
                    out.push(vec![rank(prefill, 0, 0, 0.0, requests)])
                }
                ForwardPassWorkerType::Decode => {}
                ForwardPassWorkerType::Aggregated => {
                    out.push(vec![rank(prefill, 0, 0, 0.0, requests)]);
                    out.push(vec![rank(prefill, decode, kv, 0.0, requests)]);
                    out.push(vec![
                        rank(prefill, 0, 0, 0.0, requests),
                        rank(0, decode, kv, 0.0, requests),
                    ]);
                }
            }
        }
    }
    if worker == ForwardPassWorkerType::Decode {
        // Prefill work on a decode worker is an error for regression and
        // learned roles; native accepts it. Either outcome must match.
        out.push(vec![rank(512, 0, 0, 0.0, requests)]);
    }
    out
}

fn predictions(model: &ForwardPassPerfModel, queries: &[Iteration]) -> Vec<String> {
    queries
        .iter()
        .map(|query| {
            format!(
                "{:?}",
                model
                    .estimate_forward_pass_time_ms(query)
                    .map(|value| value.map(f64::to_bits))
                    .map_err(|error| error.to_string())
            )
        })
        .collect()
}

fn without_provenance(mut diagnostics: ForwardPassPerfDiagnostics) -> ForwardPassPerfDiagnostics {
    diagnostics.provenance = None;
    diagnostics
}

/// Online learner diagnostics without the two fields that are not state, as
/// for existing online checkpoints: `training_state_bytes`, an allocation
/// estimate from vector capacities that differs between a grown and a freshly
/// decoded learner, and `history_grouping_ms`, the wall-clock time this
/// process spent grouping history rows at load (skipped by serde).
fn online_diagnostics(model: &ForwardPassPerfModel) -> Option<serde_json::Value> {
    let mut value = model.learned_online_diagnostics()?;
    if let Some(object) = value.as_object_mut() {
        object.remove("training_state_bytes");
    }
    if let Some(stores) = value.get_mut("stores").and_then(|s| s.as_object_mut()) {
        for store in stores.values_mut() {
            if let Some(store) = store.as_object_mut() {
                store.remove("training_state_bytes");
                store.remove("history_grouping_ms");
            }
        }
    }
    Some(value)
}

/// Every observable the saving model exposes, excluding provenance.
fn observables(model: &ForwardPassPerfModel, queries: &[Iteration]) -> String {
    format!(
        "{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}",
        predictions(model, queries),
        without_provenance(model.diagnostics()),
        model.regression_store_diagnostics(),
        model.readiness_report(),
        online_diagnostics(model),
        model.learned_selector_diagnostics(),
        model.learned_metadata(),
        model.min_correction_factor().map(f64::to_bits),
        model.max_correction_factor().map(f64::to_bits),
        model.avg_correction_factor().map(f64::to_bits),
    )
}

fn continue_config(mut config: ForwardPassPerfModelConfig) -> ForwardPassPerfModelConfig {
    config
        .estimator_config
        .state_snapshot
        .as_mut()
        .unwrap()
        .on_tune = StateSnapshotTuning::Continue;
    config
}

/// Files whose bytes must be identical between two snapshots of models with
/// identical state (manifests and configs embed their own paths).
fn state_files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    [
        STATE_FILE,
        "learned/online/state.bin",
        "learned/online/model.json",
        STABLE_ARTIFACT_FILE,
        LEARNED_ARTIFACT_FILE,
    ]
    .into_iter()
    .filter_map(|name| {
        std::fs::read(dir.join(name))
            .ok()
            .map(|bytes| (name.to_owned(), bytes))
    })
    .collect()
}

/// Save after `warm`, restore through `load`, compare everything, check
/// `on_tune: reject`, continue both models through `next` in chunks, and
/// compare predictions after every chunk and final state bytes.
fn check_round_trip(
    mut original: ForwardPassPerfModel,
    warm: &[Iteration],
    next: &[Iteration],
    queries: &[Iteration],
    load: Loader<'_>,
    mut cold: ForwardPassPerfModel,
) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let untuned = predictions(&cold, queries);
    original.tune_with_fpms(warm).unwrap();
    // Cold controls compare estimates only: diagnostics such as sample counts
    // differ trivially and would make a vacuous control pass.
    assert_ne!(
        predictions(&original, queries),
        untuned,
        "warm-up must change estimates, or the round trip cannot detect lost state",
    );
    let pinned = original
        .save_state_snapshot(root.join("saved"), &StateSnapshotOptions::default())
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<ForwardPassPerfModelConfig>(
            &std::fs::read(root.join("saved").join(CONFIG_FILE)).unwrap()
        )
        .unwrap(),
        pinned,
    );
    let expected = observables(&original, queries);

    let mut frozen = load(pinned.clone()).unwrap();
    assert_eq!(observables(&frozen, queries), expected);
    // Saved-config round trip: the restored provenance is a fixed point.
    if let Some(provenance) = frozen.provenance() {
        assert_eq!(provenance.config, pinned);
    }
    let before = state_files(&root.join("saved"));
    for error in [
        frozen.tune_with_fpms(&next[..1]).unwrap_err(),
        frozen.flush_updates().unwrap_err(),
    ] {
        assert!(error.to_string().contains("on_tune='reject'"), "{error}");
    }
    assert_eq!(observables(&frozen, queries), expected);
    // Re-saving a restored model reproduces the state bytes exactly.
    frozen
        .save_state_snapshot(root.join("resaved"), &StateSnapshotOptions::default())
        .unwrap();
    assert_state_files(&root.join("resaved"), &before);

    let mut resumed = load(continue_config(pinned)).unwrap();
    for chunk in next.chunks(7) {
        original.tune_with_fpms(chunk).unwrap();
        resumed.tune_with_fpms(chunk).unwrap();
        cold.tune_with_fpms(chunk).unwrap();
        assert_eq!(
            predictions(&resumed, queries),
            predictions(&original, queries)
        );
    }
    assert_ne!(
        predictions(&cold, queries),
        predictions(&original, queries),
        "a cold model fed only the continuation must estimate differently, or this \
         comparison cannot detect lost state",
    );
    assert_eq!(
        observables(&resumed, queries),
        observables(&original, queries)
    );
    original
        .save_state_snapshot(
            root.join("original-final"),
            &StateSnapshotOptions::default(),
        )
        .unwrap();
    resumed
        .save_state_snapshot(root.join("resumed-final"), &StateSnapshotOptions::default())
        .unwrap();
    assert_state_files(
        &root.join("resumed-final"),
        &state_files(&root.join("original-final")),
    );
    dir
}

fn assert_state_files(dir: &Path, expected: &[(String, Vec<u8>)]) {
    let actual = state_files(dir);
    assert_eq!(
        actual.iter().map(|(name, _)| name).collect::<Vec<_>>(),
        expected.iter().map(|(name, _)| name).collect::<Vec<_>>()
    );
    for ((name, actual), (_, expected)) in actual.iter().zip(expected) {
        assert!(actual == expected, "{name} differs in {}", dir.display());
    }
}

// ---------------------------------------------------------------------------
// Regression
// ---------------------------------------------------------------------------

fn regression_config(
    worker: ForwardPassWorkerType,
    edit: impl FnOnce(&mut ForwardPassPerfModelConfig),
) -> ForwardPassPerfModelConfig {
    let mut config =
        ForwardPassPerfModelConfig::new("test/model", "test", BackendKind::Vllm, worker);
    config.estimation_mode = EstimationMode::FpmRegression;
    edit(&mut config);
    config
}

fn best(config: ForwardPassPerfModelConfig) -> Result<ForwardPassPerfModel, AicError> {
    ForwardPassPerfModel::best_available(config)
}

fn check_regression(
    worker: ForwardPassWorkerType,
    requests: bool,
    edit: impl Fn(&mut ForwardPassPerfModelConfig),
) {
    let config = regression_config(worker, &edit);
    // Warm without cross-rank work: that store stays empty, so its infinite
    // bound sentinels are saved, restored and then first populated.
    let warm = stream(worker, 11, 90, false, requests);
    let next = stream(worker, 12, 40, true, requests);
    check_round_trip(
        best(config.clone()).unwrap(),
        &warm,
        &next,
        &queries(worker, requests),
        &best,
        best(config).unwrap(),
    );
}

#[test]
fn regression_linear_round_trips_for_dedicated_and_aggregated_roles() {
    check_regression(DECODE, false, |_| {});
    check_regression(AGG, false, |_| {});
}

#[test]
fn regression_error_threshold_round_trips_lazy_update_clocks() {
    check_regression(AGG, false, |config| {
        config.estimator_config.fpm_regression.fit.linear = Some(LinearFitConfig {
            update_policy: RegressionUpdatePolicy::ErrorThreshold {
                relative_tolerance: 0.02,
                absolute_tolerance_ms: 0.01,
                window: 6,
                trigger: 2,
                cooldown: 3,
                startup_observations: 4,
            },
            ..LinearFitConfig::default()
        });
    });
}

#[test]
fn selected_regression_round_trips_one_three_and_six_dimensions() {
    use RegressionFeatureAxis::*;
    check_regression(DECODE, false, |config| {
        config.estimator_config.fpm_regression.fit.linear = Some(LinearFitConfig {
            feature_axes: vec![Attention],
            ..LinearFitConfig::default()
        });
    });
    // A sampling-only axis exercises unrecoverable bucket coordinates.
    check_regression(AGG, false, |config| {
        config.estimator_config.fpm_regression.fit.linear = Some(LinearFitConfig {
            feature_axes: vec![Attention, Moe, Count],
            non_negative: false,
            ..LinearFitConfig::default()
        });
        config.estimator_config.fpm_regression.sampling.axes = vec![Attention, Moe, LogCount];
        config
            .estimator_config
            .fpm_regression
            .sampling
            .bins_per_axis = vec![3, 3, 2];
    });
    check_regression(DECODE, true, |config| {
        config.estimator_config.fpm_regression.fit.linear = Some(LinearFitConfig {
            feature_axes: vec![Attention, Moe, Count, Extend, Past, MaxPast],
            ..LinearFitConfig::default()
        });
    });
}

#[test]
fn spline_regression_round_trips_periodic_and_adaptive_search_state() {
    check_regression(DECODE, false, |config| {
        config.estimator_config.fpm_regression.fit.kind = RegressionFitKind::Spline;
        config.estimator_config.fpm_regression.fit.spline = Some(SplineFitConfig {
            knots_per_axis: 3,
            search: SplineSearchConfig::periodic(16),
        });
        config.estimator_config.fpm_regression.fit.rebuild_interval = Some(11);
    });
    check_regression(AGG, false, |config| {
        config.estimator_config.fpm_regression.fit.kind = RegressionFitKind::Spline;
        config.estimator_config.fpm_regression.fit.spline = Some(SplineFitConfig {
            knots_per_axis: 2,
            search: SplineSearchConfig::adaptive(8, 2, 0.01, 0.0, 8),
        });
    });
}

// ---------------------------------------------------------------------------
// Native (fixture engine; native compilation needs the Python feature)
// ---------------------------------------------------------------------------

/// One shared fixture engine: loading its tables dominates native test time.
fn fixture_engine() -> Arc<Engine> {
    static ENGINE: std::sync::OnceLock<Arc<Engine>> = std::sync::OnceLock::new();
    Arc::clone(ENGINE.get_or_init(crate::fpm::tests::fixture_engine))
}

/// A minimal systems root for the fingerprint: hardware YAML, a shared
/// top-level manifest, data under `data_dir`, and an excluded subtree.
fn fake_systems_root(dir: &Path) -> PathBuf {
    let root = dir.join("systems");
    let data = root.join("data/b200_sxm/vllm/0.24.0");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(root.join("data/b200_sxm/gemm/vllm/0.24.0")).unwrap();
    std::fs::create_dir_all(root.join("profiles")).unwrap();
    std::fs::write(root.join("b200_sxm.yaml"), "data_dir: data/b200_sxm\n").unwrap();
    std::fs::write(root.join("perf_data_reuse_manifest.yaml"), "reuse: []\n").unwrap();
    std::fs::write(data.join("gemm_perf.parquet"), b"gemm rows").unwrap();
    std::fs::write(
        root.join("data/b200_sxm/gemm/vllm/0.24.0/gemm_perf.parquet"),
        b"family rows",
    )
    .unwrap();
    std::fs::write(root.join("profiles/unrelated.json"), b"{}").unwrap();
    root
}

fn native_config(root: &Path, mode: CorrectionMode) -> ForwardPassPerfModelConfig {
    let mut config =
        ForwardPassPerfModelConfig::new(TEST_MODEL, "b200_sxm", BackendKind::Vllm, AGG);
    config.estimation_mode = EstimationMode::OpLevel;
    config.backend_version = Some("0.24.0".into());
    config.systems_paths = vec![root.to_path_buf()];
    config.estimator_config.correction.mode = mode;
    config.estimator_config.correction.min_observations = 3;
    config
}

/// Stand-in for native selection: the hand-built fixture engine with the
/// same provenance `select` records. The fingerprinted root is the
/// configured root; the fixture engine itself reads the packaged data.
fn native_select(
    engine: Arc<Engine>,
) -> impl Fn(ForwardPassPerfModelConfig, Option<bool>) -> Result<ForwardPassPerfModel, AicError> {
    move |config, _| {
        let mut model = ForwardPassPerfModel::from_engine(
            Arc::clone(&engine),
            config.estimator_config.correction_options(),
        );
        model.is_correction_enabled = config.estimator_config.correction.enabled;
        if let Some(root) = config.systems_paths.first() {
            model.set_data_fingerprint(perf_db::loaded(
                root,
                &config.system,
                config
                    .estimator_config
                    .fpm_interpolation
                    .fpm_parquet_path
                    .as_deref(),
            ));
        }
        model.provenance = Some(ForwardPassPerfProvenance {
            selected_systems_root: config.systems_paths.first().cloned(),
            requested_estimation_mode: config.estimation_mode,
            selected_estimation_mode: config.estimation_mode,
            selection_failures: Vec::new(),
            config,
        });
        Ok(model)
    }
}

fn native_load_with(
    engine: Arc<Engine>,
) -> impl Fn(ForwardPassPerfModelConfig) -> Result<ForwardPassPerfModel, AicError> {
    move |config| {
        let (config, registered) = config.resolve_with_registration()?;
        restore(config, registered, &native_select(Arc::clone(&engine)))
    }
}

fn native_model(root: &Path, mode: CorrectionMode) -> ForwardPassPerfModel {
    let config = native_config(root, mode).resolve().unwrap();
    native_select(fixture_engine())(config, None).unwrap()
}

/// Observed latency = native estimate x a smooth workload-dependent factor,
/// so correction ratios stay mostly inside the default bounds.
fn native_stream(model: &ForwardPassPerfModel, seed: u64, count: usize) -> Vec<Iteration> {
    let mut iterations = stream(AGG, seed, count, true, false);
    for (index, iteration) in iterations.iter_mut().enumerate() {
        let native = model
            .estimate_forward_pass_time_ms(iteration)
            .unwrap()
            .unwrap();
        let factor = 0.7 + (index % 9) as f64 * 0.11;
        for rank in iteration.iter_mut() {
            rank.wall_time = native * factor / 1000.0;
        }
    }
    iterations
}

#[test]
fn native_corrections_round_trip_for_fixed_dynamic_and_hybrid_policies() {
    let dir = tempfile::tempdir().unwrap();
    let root = fake_systems_root(dir.path());
    let load = native_load_with(fixture_engine());
    for mode in [
        CorrectionMode::Auto,
        CorrectionMode::DynamicLocal,
        CorrectionMode::Hybrid,
    ] {
        let warm = native_stream(&native_model(&root, mode), 21, 60);
        let next = native_stream(&native_model(&root, mode), 22, 28);
        let queries = queries(AGG, false);
        let saved = check_round_trip(
            native_model(&root, mode),
            &warm,
            &next,
            &queries,
            &load,
            native_model(&root, mode),
        );
        let saved = saved.path().join("saved");
        let manifest = Manifest::parse(&std::fs::read(saved.join(MANIFEST_FILE)).unwrap()).unwrap();
        let native = manifest.native.unwrap();
        assert_eq!(native.engine_spec_sha256, fixture_engine().spec_sha256());
        assert!(!native.packed);
        // Hardware YAML, the shared manifest and both data files; not profiles/.
        assert_eq!(native.perf_db.file_count, 4);
        if mode == CorrectionMode::Hybrid {
            // Queries beyond every fixed limit select the dynamic branch: a
            // restored model must reproduce a factor different from 1.0.
            let model = load(
                serde_json::from_slice(&std::fs::read(saved.join(CONFIG_FILE)).unwrap()).unwrap(),
            )
            .unwrap();
            let outside = vec![rank(12_000, 600, 2_400_000, 0.0, false)];
            let detailed = model.estimate_forward_pass_detailed(&outside).unwrap();
            assert_ne!(detailed.correction_factor, Some(1.0));
        }
    }
}

#[test]
fn native_snapshot_rejects_changed_perf_data_and_engine_spec() {
    let dir = tempfile::tempdir().unwrap();
    let root = fake_systems_root(dir.path());
    let mut model = native_model(&root, CorrectionMode::Hybrid);
    model
        .tune_with_fpms(&native_stream(&model, 31, 20))
        .unwrap();
    let pinned = model
        .save_state_snapshot(dir.path().join("snap"), &StateSnapshotOptions::default())
        .unwrap();
    let load = native_load_with(fixture_engine());
    load(pinned.clone()).unwrap();

    // A different compiled graph from the same configuration is rejected.
    let mut spec = crate::perfmodel::engine::spec::EngineSpec::new(
        crate::fpm::tests::fixture_engine_config(),
        crate::fpm::tests::context_ops(),
        crate::fpm::tests::generation_ops()[..1].to_vec(),
    );
    spec.engine.kv_block_size = Some(64);
    let db = fixture_engine().database().clone();
    let other = Arc::new(Engine::build(spec, db).unwrap());
    let error = native_load_with(other)(pinned.clone()).unwrap_err();
    assert!(
        error.to_string().contains("engine specification differs"),
        "{error}"
    );

    // Data changes, additions (which can change source resolution) and
    // removals are all fingerprint mismatches; excluded subtrees are not.
    std::fs::write(root.join("profiles/unrelated.json"), b"changed").unwrap();
    load(pinned.clone()).unwrap();
    for (path, contents) in [
        (
            "data/b200_sxm/vllm/0.24.0/gemm_perf.parquet",
            Some(&b"gemm rowz"[..]),
        ),
        (
            "data/b200_sxm/vllm/0.23.0/gemm_perf.parquet",
            Some(&b"sibling"[..]),
        ),
        ("perf_data_reuse_manifest.yaml", None),
    ] {
        let path = root.join(path);
        let original = std::fs::read(&path).ok();
        let new_parent = (!path.parent().unwrap().exists()).then(|| path.parent().unwrap());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        match contents {
            Some(bytes) => std::fs::write(&path, bytes).unwrap(),
            None => std::fs::remove_file(&path).unwrap(),
        }
        let error = load(pinned.clone()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("differs from the snapshot fingerprint")
                || matches!(error, AicError::StateSnapshotIo { .. }),
            "{error}"
        );
        match original {
            Some(bytes) => std::fs::write(&path, bytes).unwrap(),
            None => std::fs::remove_file(&path).unwrap(),
        }
        if let Some(parent) = new_parent {
            // An empty version directory still changes source resolution.
            assert!(load(pinned.clone()).is_err());
            std::fs::remove_dir(parent).unwrap();
        }
    }
    load(pinned.clone()).unwrap();

    // This process loaded the root before the change below, so even a
    // snapshot of the data now on disk cannot restore here.
    std::fs::write(root.join("perf_data_reuse_manifest.yaml"), "reuse: [x]\n").unwrap();
    let current = perf_db::fingerprint(&root, "b200_sxm", None).unwrap();
    let repinned = repin(&pinned, |manifest| {
        manifest["native"]["perf_db"] = serde_json::to_value(&current).unwrap();
    });
    let error = load(repinned).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("before it changed; restore in a new process"),
        "{error}"
    );
}

#[test]
fn packed_perf_db_is_self_contained_and_verified() {
    let dir = tempfile::tempdir().unwrap();
    let root = fake_systems_root(dir.path());
    let mut model = native_model(&root, CorrectionMode::Fixed);
    model
        .tune_with_fpms(&native_stream(&model, 41, 30))
        .unwrap();
    let pinned = model
        .save_state_snapshot(
            dir.path().join("packed"),
            &StateSnapshotOptions {
                include_perf_db: true,
            },
        )
        .unwrap();
    let packed = dir.path().join("packed").join(perf_db::PACKED_SYSTEMS_DIR);
    assert_eq!(
        pinned.systems_paths,
        vec![snapshot::normalized_absolute(&packed).unwrap()]
    );
    assert!(
        packed
            .join("data/b200_sxm/gemm/vllm/0.24.0/gemm_perf.parquet")
            .is_file()
    );
    assert!(!packed.join("profiles").exists());
    // The original root may disappear; the packed copy is pinned instead.
    std::fs::remove_dir_all(&root).unwrap();
    let load = native_load_with(fixture_engine());
    let restored = load(pinned.clone()).unwrap();
    let queries = queries(AGG, false);
    assert_eq!(
        predictions(&restored, &queries),
        predictions(&model, &queries)
    );
    std::fs::write(
        packed.join("b200_sxm.yaml"),
        "data_dir: data/b200_sxm\n# edit\n",
    )
    .unwrap();
    let error = load(pinned).unwrap_err();
    assert!(error.to_string().contains("fingerprint"), "{error}");
}

/// FPM interpolation with an external parquet and no data directory: the
/// fingerprint records the absent directory and the parquet with its
/// sidecar, packing copies both, and data that changes after the model
/// loaded it can no longer be saved.
#[test]
fn external_fpm_data_is_fingerprinted_at_load_and_packed() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("systems");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("b200_sxm.yaml"), "data_dir: data/b200_sxm\n").unwrap();
    let parquet = dir.path().join("measured/fpm.parquet");
    std::fs::create_dir_all(parquet.parent().unwrap()).unwrap();
    std::fs::write(&parquet, b"rows").unwrap();
    std::fs::write(parquet.with_extension("metadata.json"), b"{}").unwrap();
    let mut config = native_config(&root, CorrectionMode::Hybrid);
    config.estimation_mode = EstimationMode::FpmInterpolation;
    config.estimator_config.fpm_interpolation.fpm_parquet_path = Some(parquet.clone());
    let select = native_select(fixture_engine());
    let mut model = select(config.resolve().unwrap(), None).unwrap();
    model
        .tune_with_fpms(&native_stream(&model, 43, 40))
        .unwrap();
    let reference = model
        .save_state_snapshot(
            dir.path().join("reference"),
            &StateSnapshotOptions::default(),
        )
        .unwrap();
    let packed = model
        .save_state_snapshot(
            dir.path().join("packed"),
            &StateSnapshotOptions {
                include_perf_db: true,
            },
        )
        .unwrap();
    let packed_parquet = dir
        .path()
        .join("packed")
        .join(perf_db::PACKED_FPM_DIR)
        .join("fpm.parquet");
    assert_eq!(
        packed.estimator_config.fpm_interpolation.fpm_parquet_path,
        Some(packed_parquet.clone())
    );
    assert!(packed_parquet.with_extension("metadata.json").is_file());
    assert_eq!(
        reference
            .estimator_config
            .fpm_interpolation
            .fpm_parquet_path,
        Some(parquet.clone())
    );
    let manifest =
        Manifest::parse(&std::fs::read(dir.path().join("reference").join(MANIFEST_FILE)).unwrap())
            .unwrap();
    let native = manifest.native.unwrap();
    assert_eq!(native.perf_db.file_count, 1);
    assert!(native.perf_db.fpm_parquet.unwrap().sidecar_sha256.is_some());
    let load = native_load_with(fixture_engine());
    let queries = queries(AGG, false);
    let expected = predictions(&model, &queries);
    assert_eq!(
        predictions(&load(reference.clone()).unwrap(), &queries),
        expected
    );

    // Changing the parquet or its sidecar after load invalidates the
    // reference snapshot and any further save from this process.
    for path in [parquet.clone(), parquet.with_extension("metadata.json")] {
        let original = std::fs::read(&path).unwrap();
        std::fs::write(&path, b"changed").unwrap();
        let error = load(reference.clone()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("differs from the snapshot fingerprint"),
            "{error}"
        );
        let error = model
            .save_state_snapshot(
                dir.path().join("after-change"),
                &StateSnapshotOptions::default(),
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("changed after this process loaded it"),
            "{error}"
        );
        assert!(!dir.path().join("after-change").exists());
        // The packed copy is independent of the source.
        assert_eq!(
            predictions(&load(packed.clone()).unwrap(), &queries),
            expected
        );
        std::fs::write(&path, original).unwrap();
    }
    // Creating the data directory changes source resolution.
    std::fs::create_dir_all(root.join("data/b200_sxm")).unwrap();
    assert!(load(reference.clone()).is_err());
    std::fs::remove_dir(root.join("data/b200_sxm")).unwrap();
    assert_eq!(predictions(&load(reference).unwrap(), &queries), expected);

    // A model without a load-time fingerprint cannot be saved.
    let mut bare = ForwardPassPerfModel::from_engine(fixture_engine(), Default::default());
    bare.provenance = model.provenance.clone();
    let error = bare
        .save_state_snapshot(dir.path().join("bare"), &StateSnapshotOptions::default())
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("no load-time performance-data fingerprint"),
        "{error}"
    );
}

// ---------------------------------------------------------------------------
// Learned
// ---------------------------------------------------------------------------

const FOREST_FEATURES: [&str; 3] = [
    "num_decode_requests",
    "sum_decode_kv_tokens",
    "sum_prefill_tokens",
];

fn stump(feature: i32, threshold: f64, left: f64, right: f64) -> serde_json::Value {
    serde_json::json!({
        "left": [1, -1, -1], "right": [2, -1, -1], "feature": [feature, -1, -1],
        "threshold": [threshold, 0.0, 0.0], "value": [0.0, left, right],
        "missing_left": [true, true, true]
    })
}

/// A small three-feature forest. Omitting `skip` leaves a store untrained.
fn forest_artifact(
    worker: ForwardPassWorkerType,
    skip: Option<ForwardPassRegressionWorkloadKind>,
    identity: Option<serde_json::Value>,
) -> String {
    let mut stores = serde_json::Map::new();
    for (index, kind) in workload_kinds_for(worker).iter().enumerate() {
        if Some(*kind) == skip {
            continue;
        }
        stores.insert(
            serde_json::to_value(kind).unwrap().as_str().unwrap().into(),
            serde_json::json!({
                "baseline": 5.0 + index as f64,
                "trees": [
                    stump(2, 1024.0, 0.5, 3.0),
                    stump(1, 50_000.0, 0.25, 2.5),
                    stump(0, 64.0, 0.125, 1.0),
                ]
            }),
        );
    }
    let mut metadata = serde_json::json!({"trainer": "snapshot-test"});
    if let Some(identity) = identity {
        metadata["identity"] = identity;
    }
    serde_json::json!({
        "schema": "aic_fpm_learned_forward_perf", "schema_version": 1,
        "worker_type": worker, "target": "ms", "features": FOREST_FEATURES,
        "stores": stores, "metadata": metadata
    })
    .to_string()
}

fn learned_frozen_config(
    worker: ForwardPassWorkerType,
    artifact: &Path,
    mode: CorrectionMode,
) -> ForwardPassPerfModelConfig {
    let mut config =
        ForwardPassPerfModelConfig::new("learned/model", "test-gpu", BackendKind::Vllm, worker);
    config.estimation_mode = EstimationMode::FpmLearned;
    config.estimator_config.fpm_learned.artifact_path = Some(artifact.to_path_buf());
    config.estimator_config.fpm_learned.adaptation = LearnedAdaptation::Frozen;
    config.estimator_config.correction.enabled = true;
    config.estimator_config.correction.mode = mode;
    config.estimator_config.correction.min_observations = 3;
    config
}

#[test]
fn frozen_forest_corrections_round_trip_and_copy_the_artifact() {
    let dir = tempfile::tempdir().unwrap();
    let artifact = dir.path().join("forest.json");
    std::fs::write(
        &artifact,
        forest_artifact(
            AGG,
            Some(ForwardPassRegressionWorkloadKind::CrossRankAggregated),
            None,
        ),
    )
    .unwrap();
    for mode in [CorrectionMode::Hybrid, CorrectionMode::DynamicLocal] {
        let config = learned_frozen_config(AGG, &artifact, mode);
        let saved = check_round_trip(
            best(config.clone()).unwrap(),
            &stream(AGG, 51, 70, true, false),
            &stream(AGG, 52, 30, true, false),
            &queries(AGG, false),
            &best,
            best(config).unwrap(),
        );
        // The snapshot holds the artifact's exact bytes, so the original may change.
        assert_eq!(
            std::fs::read(saved.path().join("saved").join(LEARNED_ARTIFACT_FILE)).unwrap(),
            std::fs::read(&artifact).unwrap()
        );
        let report = best(learned_frozen_config(AGG, &artifact, mode))
            .unwrap()
            .readiness_report();
        assert_eq!(
            report.stores[2],
            ForwardPassStoreReadiness {
                workload_kind: ForwardPassRegressionWorkloadKind::CrossRankAggregated,
                ready: false,
                not_ready_reason: Some(ForwardPassStoreNotReadyReason::UntrainedForestStore),
                retained_observations: 0,
            }
        );
    }
}

#[test]
fn frozen_forest_fixed_corrections_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let artifact = dir.path().join("forest.json");
    std::fs::write(&artifact, forest_artifact(AGG, None, None)).unwrap();
    let config = learned_frozen_config(AGG, &artifact, CorrectionMode::Fixed);
    check_round_trip(
        best(config.clone()).unwrap(),
        &stream(AGG, 53, 70, true, false),
        &stream(AGG, 54, 30, true, false),
        &queries(AGG, false),
        &best,
        best(config).unwrap(),
    );
}

/// A real one-feature training checkpoint (constant seed forest fitted to two
/// rows), extended from the causal-selector test fixture.
fn training_checkpoint(
    role: ForwardPassWorkerType,
    adaptation: LearnedAdaptation,
) -> (tempfile::TempDir, ForwardPassPerfModelConfig) {
    let dir = tempfile::tempdir().unwrap();
    let mut config =
        ForwardPassPerfModelConfig::new("selector-test", "test-gpu", BackendKind::Vllm, role);
    config.estimation_mode = EstimationMode::FpmLearned;
    let learned = &mut config.estimator_config.fpm_learned;
    learned.adaptation = adaptation;
    learned.checkpoint_path = Some(dir.path().to_owned());
    learned.update_mode = LearnedUpdateMode::LeafOnly;
    learned.update_every = 2;
    learned.recent_window = 8;
    learned.history_weight = None;
    learned.history_weight_ramp_observations = 0;
    learned.history_bin_grouping = false;
    learned.derivative_refresh = LearnedDerivativeRefresh::Lazy;
    learned.eager_refresh_interval = 3;
    learned.split_sampling_rate = 1.0;
    learned.split_rank_tolerance = 0.0;
    config.estimator_config.correction.enabled = adaptation != LearnedAdaptation::Online;
    config.estimator_config.correction.min_observations = 1;
    config.estimator_config.correction.sampling.bins_per_axis = [2, 2];
    if adaptation == LearnedAdaptation::CausalSelector {
        config.estimator_config.selector.warmup = 2;
        config.estimator_config.selector.margin = 0.01;
    }
    let rows = dir.path().join("rows.parquet");
    let schema = Arc::new(parse_message_type("message seed { REQUIRED INT64 row_id (UINT_64); REQUIRED DOUBLE y; REQUIRED DOUBLE wall_time_ms; REQUIRED DOUBLE x0; }").unwrap());
    let mut writer = SerializedFileWriter::new(
        std::fs::File::create(&rows).unwrap(),
        schema,
        Arc::new(WriterProperties::builder().build()),
    )
    .unwrap();
    let mut group = writer.next_row_group().unwrap();
    let mut column = group.next_column().unwrap().unwrap();
    column
        .typed::<Int64Type>()
        .write_batch(&[0, 1], None, None)
        .unwrap();
    column.close().unwrap();
    for values in [[10.0, 10.0], [10.0, 10.0], [0.0, 1.0]] {
        let mut column = group.next_column().unwrap().unwrap();
        column
            .typed::<DoubleType>()
            .write_batch(&values, None, None)
            .unwrap();
        column.close().unwrap();
    }
    group.close().unwrap();
    writer.close().unwrap();
    let hash = |path: &Path| snapshot::sha256_hex(&std::fs::read(path).unwrap());
    let store = serde_json::json!({"baseline": 10.0, "trees": [{
        "left": [-1], "right": [-1], "feature": [-1], "threshold": [0.0],
        "value": [0.0], "missing_left": [true]
    }]});
    let seed_store = serde_json::json!({
        "rows_file": "rows.parquet", "rows_sha256": hash(&rows), "row_count": 2,
        "bin_thresholds": [[0.5]], "n_bins_non_missing": [2], "missing_values_bin_idx": 255,
        "loss": "squared_error", "learning_rate": 1.0, "l2_regularization": 0.0,
        "min_samples_leaf": 1, "max_depth": null, "max_leaf_nodes": 2
    });
    let mut stores = serde_json::Map::new();
    let mut seeds = serde_json::Map::new();
    for kind in workload_kinds_for(role) {
        let name: String = serde_json::to_value(kind).unwrap().as_str().unwrap().into();
        stores.insert(name.clone(), store.clone());
        seeds.insert(name, seed_store.clone());
    }
    let identity = canonical_learned_identity(&config).unwrap();
    let artifact = serde_json::json!({
        "schema": "aic_fpm_learned_forward_perf", "schema_version": 1,
        "worker_type": role, "target": "ms", "features": ["num_decode_requests"],
        "stores": stores, "metadata": {"identity": identity}
    });
    let artifact_path = dir.path().join("model.json");
    std::fs::write(&artifact_path, artifact.to_string()).unwrap();
    let manifest = serde_json::json!({
        "schema": "aisimulate_fpm_training_checkpoint", "schema_version": 1,
        "artifact_file": "model.json", "artifact_sha256": hash(&artifact_path),
        "identity": identity, "features": ["num_decode_requests"], "target": "ms",
        "shrinkage": "included_in_leaf_values", "stores": seeds
    });
    std::fs::write(dir.path().join("manifest.json"), manifest.to_string()).unwrap();
    (dir, config)
}

fn write_rows(path: &Path, wall_ms: &[f64], features: &[&[f64]]) {
    let columns: String = (0..features.len())
        .map(|index| format!(" REQUIRED DOUBLE x{index};"))
        .collect();
    let schema = Arc::new(
        parse_message_type(&format!(
            "message seed {{ REQUIRED INT64 row_id (UINT_64); REQUIRED DOUBLE y; REQUIRED DOUBLE wall_time_ms;{columns} }}"
        ))
        .unwrap(),
    );
    let mut writer = SerializedFileWriter::new(
        std::fs::File::create(path).unwrap(),
        schema,
        Arc::new(WriterProperties::builder().build()),
    )
    .unwrap();
    let mut group = writer.next_row_group().unwrap();
    let mut column = group.next_column().unwrap().unwrap();
    let ids: Vec<i64> = (0..wall_ms.len() as i64).collect();
    column
        .typed::<Int64Type>()
        .write_batch(&ids, None, None)
        .unwrap();
    column.close().unwrap();
    for values in [wall_ms, wall_ms]
        .into_iter()
        .chain(features.iter().copied())
    {
        let mut column = group.next_column().unwrap().unwrap();
        column
            .typed::<DoubleType>()
            .write_batch(values, None, None)
            .unwrap();
        column.close().unwrap();
    }
    group.close().unwrap();
    writer.close().unwrap();
}

/// A two-feature checkpoint for public default learner settings. Its seed
/// tree splits on the decode-request count, which does not explain the seed
/// rows (latency follows KV tokens), so a structural refresh can replace it.
fn structural_checkpoint(
    role: ForwardPassWorkerType,
    adaptation: LearnedAdaptation,
) -> (tempfile::TempDir, ForwardPassPerfModelConfig) {
    let dir = tempfile::tempdir().unwrap();
    // Deserialize the public schema so every omitted control takes its default.
    let config: ForwardPassPerfModelConfig = serde_json::from_value(serde_json::json!({
        "model": "structural-test", "system": "test-gpu", "backend": "vllm",
        "worker_type": role, "estimation_mode": "fpm_learned",
        "estimator_config": {
            "correction": {"enabled": adaptation != LearnedAdaptation::Online},
            "fpm_learned": {"checkpoint_path": dir.path(), "adaptation": adaptation},
        },
    }))
    .unwrap();
    let rows = dir.path().join("rows.parquet");
    write_rows(
        &rows,
        &[3.0, 9.0, 3.0, 9.0],
        &[
            &[8.0, 8.0, 256.0, 256.0],
            &[20_000.0, 400_000.0, 20_000.0, 400_000.0],
        ],
    );
    let hash = |path: &Path| snapshot::sha256_hex(&std::fs::read(path).unwrap());
    let store = serde_json::json!({"baseline": 0.0, "trees": [{
        "left": [1, -1, -1], "right": [2, -1, -1], "feature": [0, -1, -1],
        "threshold": [64.5, 0.0, 0.0], "value": [0.0, 6.0, 6.0],
        "missing_left": [true, true, true]
    }]});
    let seed_store = serde_json::json!({
        "rows_file": "rows.parquet", "rows_sha256": hash(&rows), "row_count": 4,
        "bin_thresholds": [[64.5], [100_000.5]], "n_bins_non_missing": [2, 2],
        "missing_values_bin_idx": 255, "loss": "squared_error", "learning_rate": 1.0,
        "l2_regularization": 0.0, "min_samples_leaf": 1, "max_depth": null,
        "max_leaf_nodes": 2
    });
    let mut stores = serde_json::Map::new();
    let mut seeds = serde_json::Map::new();
    for kind in workload_kinds_for(role) {
        let name: String = serde_json::to_value(kind).unwrap().as_str().unwrap().into();
        stores.insert(name.clone(), store.clone());
        seeds.insert(name, seed_store.clone());
    }
    let features = ["num_decode_requests", "sum_decode_kv_tokens"];
    let identity = canonical_learned_identity(&config).unwrap();
    let artifact = serde_json::json!({
        "schema": "aic_fpm_learned_forward_perf", "schema_version": 1,
        "worker_type": role, "target": "ms", "features": features,
        "stores": stores, "metadata": {"identity": identity}
    });
    let artifact_path = dir.path().join("model.json");
    std::fs::write(&artifact_path, artifact.to_string()).unwrap();
    let manifest = serde_json::json!({
        "schema": "aisimulate_fpm_training_checkpoint", "schema_version": 1,
        "artifact_file": "model.json", "artifact_sha256": hash(&artifact_path),
        "identity": identity, "features": features, "target": "ms",
        "shrinkage": "included_in_leaf_values", "stores": seeds
    });
    std::fs::write(dir.path().join("manifest.json"), manifest.to_string()).unwrap();
    (dir, config)
}

fn rebuilt_nodes(model: &ForwardPassPerfModel) -> u64 {
    model.learned_online_diagnostics().unwrap()["stores"]
        .as_object()
        .unwrap()
        .values()
        .map(|store| store["rebuilt_nodes"].as_u64().unwrap())
        .sum()
}

/// The online learner fits only once a batch fills, so chunked feedback in
/// the continuation phase exercises pending rows across the save boundary.
fn learned_stream(worker: ForwardPassWorkerType, seed: u64, count: usize) -> Vec<Iteration> {
    stream(worker, seed, count, true, false)
}

#[test]
fn frozen_forest_from_a_checkpoint_round_trips_through_an_exported_artifact() {
    let (_dir, config) = training_checkpoint(DECODE, LearnedAdaptation::Frozen);
    let saved = check_round_trip(
        best(config.clone()).unwrap(),
        &learned_stream(DECODE, 61, 40),
        &learned_stream(DECODE, 62, 20),
        &queries(DECODE, false),
        &best,
        best(config.clone()).unwrap(),
    );
    let restored = best(
        serde_json::from_slice(
            &std::fs::read(saved.path().join("saved").join(CONFIG_FILE)).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    // Exported without an identity, the forest keeps its checkpoint-verified status.
    assert_eq!(
        restored.learned_online_diagnostics().unwrap()["identity_verified"],
        true
    );
}

#[test]
fn online_forest_round_trips_through_the_runtime_checkpoint_with_pending_rows() {
    for role in [DECODE, AGG] {
        let (_dir, config) = training_checkpoint(role, LearnedAdaptation::Online);
        // An odd warm-up leaves one admitted row pending in a two-row batch.
        check_round_trip(
            best(config.clone()).unwrap(),
            &learned_stream(role, 71, 23),
            &learned_stream(role, 72, 21),
            &queries(role, false),
            &best,
            best(config).unwrap(),
        );
    }
}

#[test]
fn causal_selector_round_trips_both_experts_calibration_and_every_selector_key() {
    for role in [DECODE, AGG] {
        let (_dir, config) = training_checkpoint(role, LearnedAdaptation::CausalSelector);
        let saved = check_round_trip(
            best(config.clone()).unwrap(),
            &learned_stream(role, 81, 33),
            &learned_stream(role, 82, 21),
            &queries(role, false),
            &best,
            best(config).unwrap(),
        );
        let manifest =
            Manifest::parse(&std::fs::read(saved.path().join("saved/manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest.kind, StateSnapshotKind::CausalSelector);
        assert!(manifest.file(STABLE_ARTIFACT_FILE).is_some());
        assert!(
            manifest
                .file(&format!("{ONLINE_CHECKPOINT_DIR}/manifest.json"))
                .is_some()
        );
    }
}

#[test]
fn online_and_causal_selector_round_trip_with_default_learner_settings() {
    // Default eager refreshes run every 32 batches per store (batches of one
    // online row, four for the selector's online expert), so the warm-up is
    // long enough for structural rebuilds before the save. The continuation
    // stays shorter than the recent window (64 rows online, 8 for the
    // selector's expert) so the cold control cannot converge to the same
    // window and its estimates must differ.
    for (adaptation, warm, next) in [
        (LearnedAdaptation::Online, 40, 23),
        (LearnedAdaptation::CausalSelector, 140, 6),
    ] {
        for role in [DECODE, AGG] {
            let (_dir, config) = structural_checkpoint(role, adaptation);
            // The fixture must exercise the public defaults for its adaptation.
            let mut expected = FpmLearnedConfig {
                adaptation,
                ..FpmLearnedConfig::default()
            };
            if adaptation == LearnedAdaptation::CausalSelector {
                expected.recent_window = 8;
                expected.history_weight = Some(0.25);
                expected.history_weight_ramp_observations = 8;
                expected.update_every = 4;
            }
            let mut actual = config.estimator_config.fpm_learned.clone();
            actual.checkpoint_path = None;
            assert_eq!(actual, expected);
            let fresh = best(config.clone()).unwrap();
            let stores = if role == AGG { 4 } else { 1 };
            let saved = check_round_trip(
                fresh,
                &learned_stream(role, 93, stores * warm),
                &learned_stream(role, 94, stores * next),
                &queries(role, false),
                &best,
                best(config).unwrap(),
            );
            let restored = best(
                serde_json::from_slice(
                    &std::fs::read(saved.path().join("saved").join(CONFIG_FILE)).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
            assert!(
                rebuilt_nodes(&restored) > 0,
                "{adaptation:?} {role:?}: the warm-up must include a structural rebuild"
            );
        }
    }
}

#[test]
fn from_learned_round_trips_without_identity_and_keeps_its_options() {
    let dir = tempfile::tempdir().unwrap();
    let artifact = dir.path().join("forest.json");
    std::fs::write(&artifact, forest_artifact(AGG, None, None)).unwrap();
    let source = artifact.to_str().unwrap();
    let options = ForwardPassPerfOptions {
        min_observations: 3,
        bucket_shape: Some([3, 2]),
        max_num_tokens: 4096,
        ..ForwardPassPerfOptions::default()
    };
    let build = || ForwardPassPerfModel::from_learned(source, options.clone()).unwrap();
    let saved = check_round_trip(
        build(),
        &stream(AGG, 91, 50, true, false),
        &stream(AGG, 92, 30, true, false),
        &queries(AGG, false),
        &best,
        build(),
    );
    let saved = saved.path().join("saved");
    let summary = ForwardPassPerfModel::inspect_state_snapshot(&saved).unwrap();
    assert_eq!(summary.kind, StateSnapshotKind::LearnedWithoutIdentity);
    assert_eq!(summary.identity, None);
    assert_eq!(summary.topology, None);
    let config: ForwardPassPerfModelConfig =
        serde_json::from_slice(&std::fs::read(saved.join(CONFIG_FILE)).unwrap()).unwrap();
    let restored = best(config.clone()).unwrap();
    assert_eq!(restored.options(), &options);
    assert!(restored.provenance().is_none());
    // Placeholder identity fields are not interpreted, but estimator controls
    // must still match the saved options.
    let mut changed = config;
    changed.estimator_config.correction.max_num_tokens = 8192;
    let error = best(changed).unwrap_err();
    assert!(
        error.to_string().contains("correction.max_num_tokens"),
        "{error}"
    );
}

// ---------------------------------------------------------------------------
// Fail-closed restoration
// ---------------------------------------------------------------------------

fn saved_regression(dir: &Path) -> ForwardPassPerfModelConfig {
    let mut model = best(regression_config(AGG, |_| {})).unwrap();
    model
        .tune_with_fpms(&stream(AGG, 101, 80, true, false))
        .unwrap();
    model
        .save_state_snapshot(dir.join("snap"), &StateSnapshotOptions::default())
        .unwrap()
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// Rewrite the manifest and re-pin the configuration to its new digest, so a
/// test reaches the check behind the digest instead of the digest itself.
fn repin(
    config: &ForwardPassPerfModelConfig,
    edit: impl FnOnce(&mut serde_json::Value),
) -> ForwardPassPerfModelConfig {
    let root = config
        .estimator_config
        .state_snapshot
        .as_ref()
        .unwrap()
        .path
        .clone();
    let mut manifest = read_json(&root.join(MANIFEST_FILE));
    edit(&mut manifest);
    let bytes = serde_json::to_vec(&manifest).unwrap();
    std::fs::write(root.join(MANIFEST_FILE), &bytes).unwrap();
    let mut config = config.clone();
    config
        .estimator_config
        .state_snapshot
        .as_mut()
        .unwrap()
        .sha256 = Some(snapshot::sha256_hex(&bytes));
    config
}

/// Replace `state.bin` and re-pin both digests.
fn replace_state(config: &ForwardPassPerfModelConfig, bytes: &[u8]) -> ForwardPassPerfModelConfig {
    let root = config
        .estimator_config
        .state_snapshot
        .as_ref()
        .unwrap()
        .path
        .clone();
    std::fs::write(root.join(STATE_FILE), bytes).unwrap();
    repin(config, |manifest| {
        for file in manifest["files"].as_array_mut().unwrap() {
            if file["path"] == STATE_FILE {
                file["sha256"] = snapshot::sha256_hex(bytes).into();
                file["size_bytes"] = bytes.len().into();
            }
        }
    })
}

/// Edit the decoded state through its JSON form (all values here are finite).
fn mutate_state(
    config: &ForwardPassPerfModelConfig,
    edit: impl FnOnce(&mut serde_json::Value),
) -> ForwardPassPerfModelConfig {
    let root = config
        .estimator_config
        .state_snapshot
        .as_ref()
        .unwrap()
        .path
        .clone();
    let state: StateFile =
        snapshot::decode_state(&std::fs::read(root.join(STATE_FILE)).unwrap()).unwrap();
    let mut value = serde_json::to_value(&state).unwrap();
    edit(&mut value);
    let state: StateFile = serde_json::from_value(value).unwrap();
    replace_state(config, &snapshot::encode_state(&state).unwrap())
}

fn expect_error(config: ForwardPassPerfModelConfig, needle: &str) {
    let error = best(config).unwrap_err().to_string();
    assert!(error.contains(needle), "expected {needle:?} in {error}");
}

#[test]
fn snapshot_configuration_requires_explicit_strict_selection_and_matching_digests() {
    let dir = tempfile::tempdir().unwrap();
    let pinned = saved_regression(dir.path());
    for edit in [
        |config: &mut ForwardPassPerfModelConfig| config.estimation_mode = EstimationMode::Auto,
        |config: &mut ForwardPassPerfModelConfig| {
            config.fallback_policy = ForwardPassFallbackPolicy::Allow
        },
        |config: &mut ForwardPassPerfModelConfig| {
            config.fallback_policy = ForwardPassFallbackPolicy::LegacyRegression
        },
    ] {
        let mut config = pinned.clone();
        edit(&mut config);
        expect_error(
            config,
            "requires an explicit estimation_mode and fallback_policy='deny'",
        );
    }
    let mut config = pinned.clone();
    config
        .estimator_config
        .state_snapshot
        .as_mut()
        .unwrap()
        .sha256 = Some("0".repeat(64));
    expect_error(config, "sha256 does not match manifest.json");
    let mut config = pinned.clone();
    config
        .estimator_config
        .state_snapshot
        .as_mut()
        .unwrap()
        .sha256 = Some("abc".into());
    expect_error(config, "64 hexadecimal characters");
    let mut config = pinned.clone();
    config
        .estimator_config
        .state_snapshot
        .as_mut()
        .unwrap()
        .path = dir.path().join("missing");
    // Snapshot I/O is never reported as missing performance data.
    assert!(matches!(
        best(config).unwrap_err(),
        AicError::StateSnapshotIo { source, .. } if source.kind() == std::io::ErrorKind::NotFound
    ));

    // A manifest changed after normalization is caught by restoration itself.
    let (resolved, registered) = pinned.clone().resolve_with_registration().unwrap();
    repin(&pinned, |manifest| {
        manifest["files"][0]["size_bytes"] = 1.into()
    });
    let error = restore(resolved, registered, &ForwardPassPerfModel::select).unwrap_err();
    assert!(
        error.to_string().contains("changed after normalization"),
        "{error}"
    );
}

#[test]
fn identity_and_estimator_control_mismatches_are_rejected_with_their_location() {
    let dir = tempfile::tempdir().unwrap();
    let pinned = saved_regression(dir.path());
    let mut config = pinned.clone();
    config.estimator_config.fpm_regression.min_observations = 6;
    expect_error(
        config,
        "pinned identity at estimator_config.fpm_regression.min_observations",
    );
    let mut config = pinned.clone();
    config.tp = 2;
    expect_error(config, "pinned identity at tp");
    let mut config = pinned.clone();
    config.worker_type = DECODE;
    expect_error(config, "worker_type or estimation_mode differs");
    let mut config = pinned.clone();
    config.estimation_mode = EstimationMode::OpLevel;
    assert!(best(config).is_err());
}

#[test]
fn incompatible_versions_and_manifest_shapes_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let pinned = saved_regression(dir.path());
    for (edit, needle) in [
        (
            Box::new(|m: &mut serde_json::Value| m["schema_version"] = 2.into())
                as Box<dyn FnOnce(&mut serde_json::Value)>,
            "unsupported manifest schema",
        ),
        (
            Box::new(|m: &mut serde_json::Value| m["state_layout_version"] = 99.into()),
            "state layout version 99",
        ),
        (
            Box::new(|m: &mut serde_json::Value| {
                m["producer"]["aisimulate_core_version"] = "0.0.1".into()
            }),
            "written by aisimulate-core 0.0.1",
        ),
        (
            Box::new(|m: &mut serde_json::Value| {
                m["producer"]["engine_spec_schema_version"] = 1.into()
            }),
            "engine spec schema 1",
        ),
        (
            Box::new(|m: &mut serde_json::Value| m["future_field"] = true.into()),
            "unknown field",
        ),
        // A newer release may add fields; its versions are reported first.
        (
            Box::new(|m: &mut serde_json::Value| {
                m["schema_version"] = 2.into();
                m["future_field"] = true.into();
                m["kind"] = "future_kind".into();
            }),
            "unsupported manifest schema aisimulate_fpm_state_snapshot version 2",
        ),
        (
            Box::new(|m: &mut serde_json::Value| {
                m["state_layout_version"] = 2.into();
                m["files"][0]["future_field"] = true.into();
            }),
            "state layout version 2 is not supported",
        ),
        (
            Box::new(|m: &mut serde_json::Value| {
                m["producer"]["aisimulate_core_version"] = "99.0.0".into();
                m["producer"]["future_field"] = true.into();
                m["future_field"] = true.into();
            }),
            "written by aisimulate-core 99.0.0",
        ),
        (
            Box::new(|m: &mut serde_json::Value| m["producer"]["future_field"] = true.into()),
            "unknown field",
        ),
        (
            Box::new(|m: &mut serde_json::Value| m["kind"] = "native".into()),
            "kind contradicts",
        ),
        (
            Box::new(|m: &mut serde_json::Value| m["identity"] = serde_json::Value::Null),
            "only from_learned snapshots may omit",
        ),
        (
            Box::new(|m: &mut serde_json::Value| m["files"][0]["path"] = "../state.bin".into()),
            "unsafe snapshot path",
        ),
        (
            Box::new(|m: &mut serde_json::Value| m["files"] = serde_json::json!([])),
            "does not list state.bin",
        ),
        (
            Box::new(|m: &mut serde_json::Value| {
                m["diagnostics"]["retained_observations"] = 1.into()
            }),
            "restored diagnostics differ",
        ),
    ] {
        let snapshot = tempfile::tempdir().unwrap();
        copy_dir(
            &pinned
                .estimator_config
                .state_snapshot
                .as_ref()
                .unwrap()
                .path,
            &snapshot.path().join("s"),
        );
        let mut config = pinned.clone();
        config
            .estimator_config
            .state_snapshot
            .as_mut()
            .unwrap()
            .path = snapshot.path().join("s");
        expect_error(repin(&config, edit), needle);
    }
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &to.join(entry.file_name()));
        } else {
            std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }
}

#[test]
fn corrupt_truncated_or_oversized_state_is_rejected_without_panicking() {
    let dir = tempfile::tempdir().unwrap();
    let pinned = saved_regression(dir.path());
    let root = pinned
        .estimator_config
        .state_snapshot
        .as_ref()
        .unwrap()
        .path
        .clone();
    let original = std::fs::read(root.join(STATE_FILE)).unwrap();

    // Digest checks come first.
    std::fs::write(root.join(STATE_FILE), b"tampered").unwrap();
    expect_error(
        pinned.clone(),
        "does not match its manifest size or SHA-256",
    );

    let mut trailing = original.clone();
    trailing.push(0);
    expect_error(
        replace_state(&pinned, &trailing),
        "corrupt or truncated state.bin",
    );
    expect_error(
        replace_state(&pinned, &original[..original.len() / 2]),
        "corrupt or truncated state.bin",
    );
    // Oversized inputs are rejected from their recorded or actual size, before
    // any of their bytes are read or allocated (sparse files, no disk use).
    let recorded = |size: u64| {
        repin(&pinned, |manifest| {
            for file in manifest["files"].as_array_mut().unwrap() {
                if file["path"] == STATE_FILE {
                    file["size_bytes"] = size.into();
                }
            }
        })
    };
    expect_error(recorded(u64::MAX), "above the 4294967296-byte limit");
    let oversized = snapshot::MAX_STATE_BYTES + 1;
    std::fs::File::options()
        .write(true)
        .open(root.join(STATE_FILE))
        .unwrap()
        .set_len(oversized)
        .unwrap();
    expect_error(recorded(oversized), "above the 4294967296-byte limit");
    std::fs::write(root.join(STATE_FILE), &original).unwrap();
    let manifest = std::fs::read(root.join(MANIFEST_FILE)).unwrap();
    let file = std::fs::File::options()
        .append(true)
        .open(root.join(MANIFEST_FILE))
        .unwrap();
    file.set_len(snapshot::MAX_MANIFEST_BYTES + 1).unwrap();
    expect_error(pinned.clone(), "exceeds 67108864 bytes");
    std::fs::write(root.join(MANIFEST_FILE), &manifest).unwrap();
    // A declared length far beyond the bytes fails without allocating it.
    let mut huge = 1u32.to_le_bytes().to_vec();
    for variant in 0..4u32 {
        huge.truncate(4);
        huge.extend(variant.to_le_bytes());
        huge.extend([0xff; 32]);
        assert!(snapshot::decode_state::<StateFile>(&huge).is_err());
    }
    // The original bytes, re-pinned, still restore.
    best(replace_state(&pinned, &original)).unwrap();
}

/// Flip bytes of `state.bin` behind re-pinned digests and restore with
/// `on_tune: continue`. Each outcome must be an error or a model that keeps
/// serving: query, tune, query, report and re-save, all without panicking.
/// Digests, not invariants, detect value changes that keep every invariant.
fn fuzz_state(
    label: &str,
    pinned: &ForwardPassPerfModelConfig,
    load: Loader<'_>,
    next: &[Iteration],
    queries: &[Iteration],
) {
    let root = pinned
        .estimator_config
        .state_snapshot
        .as_ref()
        .unwrap()
        .path
        .clone();
    let original = std::fs::read(root.join(STATE_FILE)).unwrap();
    let out = tempfile::tempdir().unwrap();
    let (mut rejected, mut accepted, mut panics) = (0, 0, Vec::new());
    let step = original.len() / 48 + 1;
    for index in (0..original.len()).step_by(step) {
        for mask in [0x5a_u8, 0x80] {
            let mut bytes = original.clone();
            bytes[index] ^= mask;
            let config = continue_config(replace_state(pinned, &bytes));
            let trial = out.path().join(format!("{index}-{mask}"));
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let Ok(mut model) = load(config) else {
                    return false;
                };
                let _ = predictions(&model, queries);
                let _ = model.tune_with_fpms(next);
                let _ = predictions(&model, queries);
                let _ = (model.readiness_report(), model.diagnostics());
                let _ = model.save_state_snapshot(&trial, &StateSnapshotOptions::default());
                true
            }));
            match outcome {
                Ok(false) => rejected += 1,
                Ok(true) => accepted += 1,
                Err(_) => panics.push(format!("byte {index} ^ {mask:#04x}")),
            }
        }
    }
    replace_state(pinned, &original);
    assert!(panics.is_empty(), "{label}: panics at {panics:?}");
    assert!(rejected > 0, "{label}: no corruption reached a check");
    assert!(rejected + accepted > 0);
}

fn tuned_snapshot(
    mut model: ForwardPassPerfModel,
    warm: &[Iteration],
    dir: &Path,
    name: &str,
) -> ForwardPassPerfModelConfig {
    model.tune_with_fpms(warm).unwrap();
    model
        .save_state_snapshot(dir.join(name), &StateSnapshotOptions::default())
        .unwrap()
}

#[test]
fn corrupted_state_of_every_kind_fails_closed_or_keeps_serving_in_continue_mode() {
    use RegressionFeatureAxis::*;
    let dir = tempfile::tempdir().unwrap();
    type Edit = Box<dyn Fn(&mut ForwardPassPerfModelConfig)>;
    let error_threshold: Edit = Box::new(|config| {
        config.estimator_config.fpm_regression.fit.linear = Some(LinearFitConfig {
            update_policy: RegressionUpdatePolicy::ErrorThreshold {
                relative_tolerance: 0.02,
                absolute_tolerance_ms: 0.01,
                window: 6,
                trigger: 2,
                cooldown: 3,
                startup_observations: 4,
            },
            ..LinearFitConfig::default()
        });
    });
    let selected: Edit = Box::new(|config| {
        config.estimator_config.fpm_regression.fit.linear = Some(LinearFitConfig {
            feature_axes: vec![Attention, Moe, Count, Extend, Past, MaxPast],
            ..LinearFitConfig::default()
        });
    });
    let periodic: Edit = Box::new(|config| {
        config.estimator_config.fpm_regression.fit.kind = RegressionFitKind::Spline;
        config.estimator_config.fpm_regression.fit.spline = Some(SplineFitConfig {
            knots_per_axis: 3,
            search: SplineSearchConfig::periodic(16),
        });
    });
    let adaptive: Edit = Box::new(|config| {
        config.estimator_config.fpm_regression.fit.kind = RegressionFitKind::Spline;
        config.estimator_config.fpm_regression.fit.spline = Some(SplineFitConfig {
            knots_per_axis: 2,
            search: SplineSearchConfig::adaptive(8, 2, 0.01, 0.0, 8),
        });
    });
    let regressions: Vec<(&str, ForwardPassWorkerType, bool, Edit)> = vec![
        ("linear", AGG, false, Box::new(|_| {})),
        ("error_threshold", AGG, false, error_threshold),
        ("selected", DECODE, true, selected),
        ("spline_periodic", DECODE, false, periodic),
        ("spline_adaptive", AGG, false, adaptive),
    ];
    for (label, worker, requests, edit) in regressions {
        let config = regression_config(worker, edit);
        let warm = stream(worker, 11, 90, false, requests);
        let pinned = tuned_snapshot(best(config).unwrap(), &warm, dir.path(), label);
        let next = stream(worker, 12, 12, true, requests);
        fuzz_state(label, &pinned, &best, &next, &queries(worker, requests));
    }

    let root = fake_systems_root(dir.path());
    let load = native_load_with(fixture_engine());
    for mode in [
        CorrectionMode::Fixed,
        CorrectionMode::DynamicLocal,
        CorrectionMode::Hybrid,
    ] {
        let model = native_model(&root, mode);
        let (warm, next) = (native_stream(&model, 21, 60), native_stream(&model, 22, 12));
        let name = format!("native-{mode:?}");
        let pinned = tuned_snapshot(model, &warm, dir.path(), &name);
        fuzz_state(&name, &pinned, &load, &next, &queries(AGG, false));
    }

    let artifact = dir.path().join("forest.json");
    std::fs::write(&artifact, forest_artifact(AGG, None, None)).unwrap();
    for mode in [
        CorrectionMode::Fixed,
        CorrectionMode::DynamicLocal,
        CorrectionMode::Hybrid,
    ] {
        let config = learned_frozen_config(AGG, &artifact, mode);
        let name = format!("learned-{mode:?}");
        let warm = stream(AGG, 51, 70, true, false);
        let pinned = tuned_snapshot(best(config).unwrap(), &warm, dir.path(), &name);
        let next = stream(AGG, 52, 12, true, false);
        fuzz_state(&name, &pinned, &best, &next, &queries(AGG, false));
    }

    for (adaptation, role) in [
        (LearnedAdaptation::Online, AGG),
        (LearnedAdaptation::CausalSelector, DECODE),
    ] {
        let (_checkpoint, config) = training_checkpoint(role, adaptation);
        let name = format!("{adaptation:?}-{role:?}");
        let warm = learned_stream(role, 81, 33);
        let pinned = tuned_snapshot(best(config).unwrap(), &warm, dir.path(), &name);
        let next = learned_stream(role, 82, 8);
        fuzz_state(&name, &pinned, &best, &next, &queries(role, false));
    }

    let model = ForwardPassPerfModel::from_learned(
        artifact.to_str().unwrap(),
        ForwardPassPerfOptions {
            min_observations: 3,
            ..ForwardPassPerfOptions::default()
        },
    )
    .unwrap();
    let warm = stream(AGG, 91, 50, true, false);
    let pinned = tuned_snapshot(model, &warm, dir.path(), "from_learned");
    let next = stream(AGG, 92, 12, true, false);
    fuzz_state("from_learned", &pinned, &best, &next, &queries(AGG, false));
}

/// The online learner keeps its own checkpoint inside the snapshot. Corrupt
/// its state behind every re-pinned digest (online manifest, snapshot file
/// record, pinned `source_sha256`, manifest digest) and continue tuning.
#[test]
fn corrupted_online_checkpoint_behind_repinned_digests_never_panics() {
    for (adaptation, role) in [
        (LearnedAdaptation::Online, AGG),
        (LearnedAdaptation::CausalSelector, DECODE),
    ] {
        let (_checkpoint, config) = training_checkpoint(role, adaptation);
        let dir = tempfile::tempdir().unwrap();
        let pinned = tuned_snapshot(
            best(config).unwrap(),
            &learned_stream(role, 501, 23),
            dir.path(),
            "snap",
        );
        let root = dir.path().join("snap");
        let state_path = root.join(ONLINE_CHECKPOINT_DIR).join("state.bin");
        let manifest_path = root.join(ONLINE_CHECKPOINT_DIR).join("manifest.json");
        let original = std::fs::read(&state_path).unwrap();
        let online_manifest = read_json(&manifest_path);
        let snapshot_manifest = read_json(&root.join(MANIFEST_FILE));
        let next = learned_stream(role, 502, 9);
        let queries = queries(role, false);
        let out = tempfile::tempdir().unwrap();
        let (mut rejected, mut panics) = (0, Vec::new());
        for index in (0..original.len()).step_by(original.len() / 40 + 1) {
            let mut bytes = original.clone();
            bytes[index] ^= 0x5a;
            std::fs::write(&state_path, &bytes).unwrap();
            let mut online = online_manifest.clone();
            online["state_sha256"] = snapshot::sha256_hex(&bytes).into();
            let online_bytes = serde_json::to_vec(&online).unwrap();
            std::fs::write(&manifest_path, &online_bytes).unwrap();
            let online_sha = snapshot::sha256_hex(&online_bytes);
            let mut manifest = snapshot_manifest.clone();
            for file in manifest["files"].as_array_mut().unwrap() {
                if file["path"] == format!("{ONLINE_CHECKPOINT_DIR}/manifest.json") {
                    file["sha256"] = online_sha.clone().into();
                    file["size_bytes"] = online_bytes.len().into();
                }
            }
            manifest["identity"]["estimator_config"]["fpm_learned"]["source_sha256"] =
                online_sha.clone().into();
            let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
            std::fs::write(root.join(MANIFEST_FILE), &manifest_bytes).unwrap();
            let mut config = continue_config(pinned.clone());
            config.estimator_config.fpm_learned.source_sha256 = Some(online_sha);
            config
                .estimator_config
                .state_snapshot
                .as_mut()
                .unwrap()
                .sha256 = Some(snapshot::sha256_hex(&manifest_bytes));
            let trial = out.path().join(index.to_string());
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let Ok(mut model) = best(config) else {
                    return false;
                };
                let _ = predictions(&model, &queries);
                let _ = model.tune_with_fpms(&next);
                let _ = model.flush_updates();
                let _ = predictions(&model, &queries);
                let _ = model.learned_online_diagnostics();
                let _ = model.save_state_snapshot(&trial, &StateSnapshotOptions::default());
                true
            }));
            match outcome {
                Ok(false) => rejected += 1,
                Ok(true) => {}
                Err(_) => panics.push(index),
            }
        }
        assert!(
            panics.is_empty(),
            "{adaptation:?}: panics at bytes {panics:?}"
        );
        assert!(
            rejected > 0,
            "{adaptation:?}: re-pinning never reached the decoder"
        );
    }
}

#[test]
fn state_invariant_violations_are_rejected_before_they_can_panic() {
    let dir = tempfile::tempdir().unwrap();
    let pinned = saved_regression(dir.path());
    let store = |value: &mut serde_json::Value, index: usize| -> serde_json::Value {
        value["state"]["Regression"]["stores"]["stores"][index][1]["Linear"].clone()
    };
    let set_store = |value: &mut serde_json::Value, index: usize, store: serde_json::Value| {
        value["state"]["Regression"]["stores"]["stores"][index][1]["Linear"] = store;
    };
    type Edit = Box<dyn Fn(&mut serde_json::Value)>;
    let cases: Vec<(Edit, &str)> = vec![
        (
            Box::new(|value| {
                let stores = &mut value["state"]["Regression"]["stores"]["stores"];
                let first = stores[0].clone();
                stores[0] = stores[1].clone();
                stores[1] = first;
            }),
            "regression store kinds differ",
        ),
        (
            Box::new(|value| {
                value["state"]["Regression"]["stores"]["stores"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            }),
            "regression store kinds differ",
        ),
        (
            Box::new(move |value| {
                // The last key keeps the encoded keys strictly increasing.
                let mut linear = store(value, 0);
                let buckets = linear["samples"]["buckets"].as_array_mut().unwrap();
                let last = buckets.len() - 1;
                buckets[last][0] = serde_json::json!([9, 0]);
                set_store(value, 0, linear);
            }),
            "bucket key is outside the grid",
        ),
        (
            Box::new(move |value| {
                let mut linear = store(value, 0);
                let buckets = linear["samples"]["buckets"].as_array_mut().unwrap();
                let moved = buckets[1][1][0].clone();
                buckets[0][1].as_array_mut().unwrap().push(moved);
                buckets[1][1].as_array_mut().unwrap().remove(0);
                set_store(value, 0, linear);
            }),
            "wrong bucket key",
        ),
        (
            Box::new(move |value| {
                let mut linear = store(value, 0);
                linear["samples"]["total_observations"] = 1.into();
                set_store(value, 0, linear);
            }),
            "count does not match",
        ),
        (
            Box::new(move |value| {
                let mut linear = store(value, 0);
                linear["samples"]["axis_max"][0] = linear["samples"]["axis_min"][0].clone();
                set_store(value, 0, linear);
            }),
            "outside the sample bounds",
        ),
        (
            Box::new(move |value| {
                let mut linear = store(value, 0);
                linear["samples"]["buckets_per_axis"] = serde_json::json!([4]);
                set_store(value, 0, linear);
            }),
            "sample grid differs",
        ),
        (
            Box::new(move |value| {
                let mut linear = store(value, 0);
                linear["recursive"]["count"] = 3.into();
                set_store(value, 0, linear);
            }),
            "statistics count differs",
        ),
        (
            Box::new(move |value| {
                let mut linear = store(value, 0);
                linear["min_observations"] = 2.into();
                set_store(value, 0, linear);
            }),
            "min_observations differs",
        ),
        (
            Box::new(move |value| {
                let mut linear = store(value, 0);
                linear["update"]["accepted"] = 4.into();
                set_store(value, 0, linear);
            }),
            "eager regression updates carry lazy-update state",
        ),
        (
            Box::new(move |value| {
                let mut linear = store(value, 0);
                linear["fit"]["coefficients"][0] = (-1.0).into();
                set_store(value, 0, linear);
            }),
            "published regression fit",
        ),
        (
            Box::new(|value| {
                value["layout_version"] = 2.into();
            }),
            "layout version differs",
        ),
    ];
    for (edit, needle) in cases {
        let snapshot = tempfile::tempdir().unwrap();
        copy_dir(
            &pinned
                .estimator_config
                .state_snapshot
                .as_ref()
                .unwrap()
                .path,
            &snapshot.path().join("s"),
        );
        let mut config = pinned.clone();
        config
            .estimator_config
            .state_snapshot
            .as_mut()
            .unwrap()
            .path = snapshot.path().join("s");
        let root = snapshot.path().join("s");
        let state: StateFile =
            snapshot::decode_state(&std::fs::read(root.join(STATE_FILE)).unwrap()).unwrap();
        let mut value = serde_json::to_value(&state).unwrap();
        edit(&mut value);
        let bytes = match serde_json::from_value::<StateFile>(value) {
            Ok(state) => snapshot::encode_state(&state).unwrap(),
            // A shape the encoder cannot even represent is a decode failure.
            Err(_) => b"not a state".to_vec(),
        };
        let error = best(replace_state(&config, &bytes))
            .unwrap_err()
            .to_string();
        assert!(error.contains(needle), "expected {needle:?} in {error}");
    }

    // Valid state of another estimator kind is rejected after decoding.
    let artifact = tempfile::tempdir().unwrap();
    let forest = artifact.path().join("forest.json");
    std::fs::write(&forest, forest_artifact(AGG, None, None)).unwrap();
    let learned = best(learned_frozen_config(AGG, &forest, CorrectionMode::Fixed)).unwrap();
    learned
        .save_state_snapshot(
            artifact.path().join("learned"),
            &StateSnapshotOptions::default(),
        )
        .unwrap();
    let foreign = std::fs::read(artifact.path().join("learned").join(STATE_FILE)).unwrap();
    expect_error(
        replace_state(&pinned, &foreign),
        "saved state does not match the configured estimator",
    );
}

#[test]
fn correction_spline_and_selector_invariants_are_rejected() {
    // Dynamic correction rows outside their bounds would panic in the next
    // rebuild; cached factors must equal a recomputation.
    let dir = tempfile::tempdir().unwrap();
    let artifact = dir.path().join("forest.json");
    std::fs::write(&artifact, forest_artifact(DECODE, None, None)).unwrap();
    let mut model = best(learned_frozen_config(
        DECODE,
        &artifact,
        CorrectionMode::Hybrid,
    ))
    .unwrap();
    model
        .tune_with_fpms(&stream(DECODE, 111, 30, false, false))
        .unwrap();
    let pinned = model
        .save_state_snapshot(dir.path().join("hybrid"), &StateSnapshotOptions::default())
        .unwrap();
    let decode_dynamic = |value: &mut serde_json::Value| -> serde_json::Value {
        value["state"]["Learned"]["corrections"]["decode"]["Hybrid"]["dynamic"].clone()
    };
    for (edit, needle) in [
        (
            Box::new(move |value: &mut serde_json::Value| {
                let mut dynamic = decode_dynamic(value);
                dynamic["primary"][0]["log_x"][0] = 99.0.into();
                value["state"]["Learned"]["corrections"]["decode"]["Hybrid"]["dynamic"] = dynamic;
            }) as Box<dyn Fn(&mut serde_json::Value)>,
            "outside its bounds",
        ),
        (
            Box::new(move |value: &mut serde_json::Value| {
                let mut dynamic = decode_dynamic(value);
                dynamic["fallback"] = 1.75.into();
                value["state"]["Learned"]["corrections"]["decode"]["Hybrid"]["dynamic"] = dynamic;
            }),
            "factors differ from their retained rows",
        ),
        (
            Box::new(move |value: &mut serde_json::Value| {
                let mut dynamic = decode_dynamic(value);
                dynamic["next_admission"] = 3.into();
                value["state"]["Learned"]["corrections"]["decode"]["Hybrid"]["dynamic"] = dynamic;
            }),
            "dynamic correction",
        ),
        (
            Box::new(|value: &mut serde_json::Value| {
                let fixed =
                    value["state"]["Learned"]["corrections"]["decode"]["Hybrid"]["fixed"].clone();
                value["state"]["Learned"]["corrections"]["decode"] =
                    serde_json::json!({"Fixed": fixed});
            }),
            "correction policy differs",
        ),
        (
            Box::new(|value: &mut serde_json::Value| {
                let fixed = &mut value["state"]["Learned"]["corrections"]["decode"]["Hybrid"]["fixed"]
                    ["samples"]["buckets"][0][1][0][1]["correction_factor"];
                *fixed = 9.0.into();
            }),
            "outside its admitted range",
        ),
    ] {
        let snapshot = tempfile::tempdir().unwrap();
        copy_dir(&dir.path().join("hybrid"), &snapshot.path().join("s"));
        // The configuration keeps reading the original copy's artifact.
        let mut config = pinned.clone();
        config
            .estimator_config
            .state_snapshot
            .as_mut()
            .unwrap()
            .path = snapshot.path().join("s");
        expect_error(mutate_state(&config, edit), needle);
    }

    // Spline knot vectors that exceed the basis width would panic on update.
    let mut spline = best(regression_config(DECODE, |config| {
        config.estimator_config.fpm_regression.fit.kind = RegressionFitKind::Spline;
        config.estimator_config.fpm_regression.fit.spline = Some(SplineFitConfig {
            knots_per_axis: 2,
            search: SplineSearchConfig::periodic(16),
        });
    }))
    .unwrap();
    spline
        .tune_with_fpms(&stream(DECODE, 121, 60, false, false))
        .unwrap();
    let pinned = spline
        .save_state_snapshot(dir.path().join("spline"), &StateSnapshotOptions::default())
        .unwrap();
    fn spline_store(value: &mut serde_json::Value) -> &mut serde_json::Value {
        &mut value["state"]["Regression"]["stores"]["stores"][0][1]["Spline"]
    }
    for (edit, needle) in [
        (
            Box::new(move |value: &mut serde_json::Value| {
                spline_store(value)["recursive"]["knots"][0] =
                    serde_json::json!([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
            }) as Box<dyn Fn(&mut serde_json::Value)>,
            "spline knots are invalid",
        ),
        (
            Box::new(move |value: &mut serde_json::Value| {
                spline_store(value)["recursive"]["knots"][0] = serde_json::json!([0.5, 0.25]);
            }),
            "spline knots are invalid",
        ),
        (
            Box::new(move |value: &mut serde_json::Value| {
                spline_store(value)["recursive"]["p"] = 9.into();
            }),
            "basis width",
        ),
        (
            Box::new(move |value: &mut serde_json::Value| {
                spline_store(value)["maximum"][0] = 1e12.into();
            }),
            "retained-range guard",
        ),
        (
            Box::new(move |value: &mut serde_json::Value| {
                spline_store(value)["last_search"] = 1_000_000.into();
            }),
            "search clock",
        ),
        (
            Box::new(move |value: &mut serde_json::Value| {
                spline_store(value)["monitor"]["bad"] = 1.into();
            }),
            "periodic spline search carries adaptive monitor state",
        ),
    ] {
        let snapshot = tempfile::tempdir().unwrap();
        copy_dir(&dir.path().join("spline"), &snapshot.path().join("s"));
        let mut config = pinned.clone();
        config
            .estimator_config
            .state_snapshot
            .as_mut()
            .unwrap()
            .path = snapshot.path().join("s");
        expect_error(mutate_state(&config, edit), needle);
    }

    // Selector evidence must be consistent with its configuration.
    let (_checkpoint, config) = training_checkpoint(DECODE, LearnedAdaptation::CausalSelector);
    let mut selector = best(config).unwrap();
    selector
        .tune_with_fpms(&stream(DECODE, 131, 12, false, false))
        .unwrap();
    let pinned = selector
        .save_state_snapshot(
            dir.path().join("selector"),
            &StateSnapshotOptions::default(),
        )
        .unwrap();
    for (edit, needle) in [
        (
            Box::new(|value: &mut serde_json::Value| {
                value["state"]["CausalSelector"]["selector"][1]["streak"] = 50.into();
            }) as Box<dyn Fn(&mut serde_json::Value)>,
            "selector evidence is inconsistent",
        ),
        (
            Box::new(|value: &mut serde_json::Value| {
                // Pure-prefill evidence on a decode-only forest.
                value["state"]["CausalSelector"]["selector"][9]["observations"] = 1.into();
            }),
            "without admitted feedback",
        ),
    ] {
        let snapshot = tempfile::tempdir().unwrap();
        copy_dir(&dir.path().join("selector"), &snapshot.path().join("s"));
        let mut config = pinned.clone();
        config
            .estimator_config
            .state_snapshot
            .as_mut()
            .unwrap()
            .path = snapshot.path().join("s");
        expect_error(mutate_state(&config, edit), needle);
    }
}

#[test]
fn learned_files_are_pinned_and_snapshot_destinations_are_never_replaced() {
    let (_checkpoint, config) = training_checkpoint(DECODE, LearnedAdaptation::CausalSelector);
    let model = best(config).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let pinned = model
        .save_state_snapshot(dir.path().join("snap"), &StateSnapshotOptions::default())
        .unwrap();
    let root = dir.path().join("snap");
    let stable = root.join(STABLE_ARTIFACT_FILE);
    let original = std::fs::read(&stable).unwrap();
    std::fs::write(&stable, b"{}").unwrap();
    expect_error(pinned.clone(), "learned/stable.json does not match");
    std::fs::write(&stable, &original).unwrap();
    std::fs::write(root.join("learned/online/state.bin"), b"x").unwrap();
    assert!(best(pinned.clone()).is_err());

    // An existing destination, even empty, is an error and is left intact.
    let error = model
        .save_state_snapshot(&root, &StateSnapshotOptions::default())
        .unwrap_err();
    assert!(error.to_string().contains("already exists"), "{error}");
    std::fs::create_dir(dir.path().join("empty")).unwrap();
    assert!(
        model
            .save_state_snapshot(dir.path().join("empty"), &StateSnapshotOptions::default())
            .is_err()
    );
    // Failed saves leave no staging directories behind.
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .filter(|name| name.to_string_lossy().starts_with(".fpm-state-snapshot-"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
    // Native models need provenance to be restorable.
    let error = ForwardPassPerfModel::from_engine(fixture_engine(), Default::default())
        .save_state_snapshot(
            dir.path().join("no-provenance"),
            &StateSnapshotOptions::default(),
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("no construction provenance"),
        "{error}"
    );
}

// ---------------------------------------------------------------------------
// Readiness, topology and inspection
// ---------------------------------------------------------------------------

#[test]
fn readiness_reports_stores_reasons_and_corrections_for_every_kind() {
    use ForwardPassRegressionWorkloadKind::*;
    let mut regression = best(regression_config(AGG, |_| {})).unwrap();
    let cold = regression.readiness_report();
    assert!(!cold.ready);
    assert_eq!(cold.worker_type, Some(AGG));
    assert!(cold.corrections.is_empty());
    assert!(cold.stores.iter().all(|store| {
        store.not_ready_reason == Some(ForwardPassStoreNotReadyReason::InsufficientData)
    }));
    // Identical decode features: enough samples, but no usable fit.
    let degenerate = vec![vec![rank(0, 8, 800, 3.0, false)]; 6];
    regression.tune_with_fpms(&degenerate).unwrap();
    let report = regression.readiness_report();
    assert_eq!(report.stores[0].workload_kind, PureDecode);
    assert_eq!(report.stores[0].retained_observations, 6);
    assert_eq!(
        report.stores[0].not_ready_reason,
        Some(ForwardPassStoreNotReadyReason::NoUsableFit)
    );
    regression
        .tune_with_fpms(&stream(AGG, 141, 60, false, false))
        .unwrap();
    let report = regression.readiness_report();
    assert!(report.ready);
    assert_eq!(
        report
            .stores
            .iter()
            .map(|store| (store.workload_kind, store.ready))
            .collect::<Vec<_>>(),
        vec![
            (PureDecode, true),
            (ContainsLocallyMixed, true),
            (CrossRankAggregated, false),
            (PurePrefill, true),
        ]
    );
    let regression_diagnostics = regression.regression_store_diagnostics();
    for (store, legacy) in report.stores.iter().zip(&regression_diagnostics) {
        assert_eq!(
            (
                store.workload_kind,
                store.ready,
                store.retained_observations
            ),
            (
                legacy.workload_kind,
                legacy.ready,
                legacy.retained_observations
            )
        );
    }

    let dir = tempfile::tempdir().unwrap();
    let root = fake_systems_root(dir.path());
    let mut native = native_model(&root, CorrectionMode::Auto);
    let report = native.readiness_report();
    assert!(report.stores.iter().all(|store| store.ready));
    assert_eq!(report.corrections.len(), 3);
    assert!(report.corrections.iter().all(|c| c.enabled && !c.ready));
    native
        .tune_with_fpms(&native_stream(&native, 151, 30))
        .unwrap();
    let report = native.readiness_report();
    assert_eq!(
        report.corrections[1].query_type,
        ForwardPassCorrectionQueryType::Decode
    );
    assert!(report.corrections[1].ready);
    assert!(report.corrections[1].retained_observations >= 3);

    let (_checkpoint, config) = training_checkpoint(AGG, LearnedAdaptation::CausalSelector);
    let selector = best(config).unwrap();
    let report = selector.readiness_report();
    assert_eq!(report.corrections.len(), 12);
    assert_eq!(report.corrections[0].workload_kind, Some(PureDecode));
}

#[test]
fn topology_and_inspection_expose_the_pinned_identity_without_loading() {
    let dir = tempfile::tempdir().unwrap();
    let root = fake_systems_root(dir.path());
    let mut model = native_model(&root, CorrectionMode::Auto);
    model
        .tune_with_fpms(&native_stream(&model, 161, 12))
        .unwrap();
    let topology = model.deployment_topology().unwrap();
    assert_eq!(topology.worker_type, AGG);
    assert_eq!((topology.tp, topology.pp, topology.attention_dp), (1, 1, 1));
    assert_eq!(
        topology.correction_limits,
        Some(ForwardPassCorrectionLimits {
            max_num_tokens: 8192,
            max_batch_size: 512,
            max_kv_tokens: 2_000_000
        })
    );
    model
        .save_state_snapshot(dir.path().join("snap"), &StateSnapshotOptions::default())
        .unwrap();
    let summary = ForwardPassPerfModel::inspect_state_snapshot(dir.path().join("snap")).unwrap();
    assert_eq!(summary.kind, StateSnapshotKind::Native);
    assert_eq!(summary.topology, Some(topology));
    assert_eq!(summary.readiness, model.readiness_report());
    assert_eq!(summary.state_layout_version, STATE_SNAPSHOT_LAYOUT_VERSION);
    let regression = best(regression_config(DECODE, |_| {})).unwrap();
    assert_eq!(
        regression.deployment_topology().unwrap().correction_limits,
        None
    );
}

#[test]
fn saved_configuration_survives_json_and_float_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let mut model = best(regression_config(AGG, |config| {
        config.estimator_config.correction.factor_bounds.min = Some(0.1 + 0.2);
        config
            .estimator_config
            .fpm_regression
            .fit
            .singular_ridge_scale = 1.0 / 3.0;
    }))
    .unwrap();
    // The cross-rank store stays empty: its infinite bound sentinels must survive.
    model
        .tune_with_fpms(&stream(AGG, 171, 40, false, false))
        .unwrap();
    let pinned = model
        .save_state_snapshot(dir.path().join("snap"), &StateSnapshotOptions::default())
        .unwrap();
    let text = serde_json::to_string(&pinned).unwrap();
    let reloaded: ForwardPassPerfModelConfig = serde_json::from_str(&text).unwrap();
    assert_eq!(reloaded, pinned);
    let restored = best(reloaded).unwrap();
    assert_eq!(restored.provenance().unwrap().config, pinned);
    // A relative path resolves against the working directory, like learned sources.
    let mut relative = pinned.clone();
    let snapshot = &mut relative.estimator_config.state_snapshot.as_mut().unwrap();
    snapshot.sha256 = None;
    let cwd = std::env::current_dir().unwrap();
    snapshot.path = pathdiff(&dir.path().join("snap"), &cwd);
    let resolved = relative.resolve().unwrap();
    assert_eq!(resolved, pinned.clone().resolve().unwrap());
    let state: StateFile =
        snapshot::decode_state(&std::fs::read(dir.path().join("snap").join(STATE_FILE)).unwrap())
            .unwrap();
    let SnapshotState::Regression { stores } = state.state else {
        panic!("regression state")
    };
    let encoded = format!("{stores:?}");
    assert!(
        encoded.contains("inf"),
        "empty store keeps infinite sentinels: {encoded}"
    );
}

/// Relative path from `base` to `path` using `..` components.
fn pathdiff(path: &Path, base: &Path) -> PathBuf {
    let path: Vec<_> = path.components().collect();
    let base: Vec<_> = base.components().collect();
    let common = path.iter().zip(&base).take_while(|(a, b)| a == b).count();
    let mut out = PathBuf::new();
    for _ in common..base.len() {
        out.push("..");
    }
    for component in &path[common..] {
        out.push(component);
    }
    out
}

// ---------------------------------------------------------------------------
// Format stability
// ---------------------------------------------------------------------------

/// Independent mirror of the version-1 `state.bin` layout for the regression,
/// fixed/dynamic/hybrid correction and selector envelopes. A change to any
/// encoded field, order, type or variant fails this decode: bump
/// `STATE_SNAPSHOT_LAYOUT_VERSION` and update the mirror together.
mod layout_v1 {
    use serde::Deserialize;

    type Map<K, V> = Vec<(K, V)>;

    #[derive(Deserialize)]
    pub(super) struct StateFile {
        pub layout_version: u32,
        pub state: State,
    }

    #[derive(Deserialize)]
    pub(super) enum State {
        Native {
            corrections: Workload,
        },
        Regression {
            stores: Vec<(u32, Store)>,
        },
        Learned {
            corrections: Workload,
        },
        CausalSelector {
            corrections: Vec<(u32, Workload)>,
            selector: [Evidence; 12],
        },
    }

    #[derive(Deserialize)]
    pub(super) struct Workload {
        pub prefill: Correction,
        pub decode: Correction,
        pub mixed: Correction,
    }

    #[derive(Deserialize)]
    pub(super) enum Correction {
        Fixed(Fixed),
        Dynamic(Dynamic),
        Hybrid(Box<(Fixed, Dynamic)>),
    }

    #[derive(Deserialize)]
    pub(super) struct Samples<T> {
        pub buckets: Map<Vec<usize>, Vec<(Vec<f64>, T)>>,
        pub total_observations: usize,
        pub axis_min: Vec<f64>,
        pub axis_max: Vec<f64>,
        pub fixed_bounds: bool,
        pub buckets_per_axis: Vec<usize>,
        pub max_observations: usize,
    }

    #[derive(Deserialize)]
    pub(super) struct Fixed {
        pub samples: Samples<f64>,
        pub min_observations: usize,
        pub min_faster: Option<f64>,
        pub max_slower: Option<f64>,
    }

    #[derive(Deserialize)]
    pub(super) struct Row {
        pub admission: usize,
        pub log_x: [f64; 2],
        pub ratio: f64,
    }

    #[derive(Deserialize)]
    pub(super) struct Dynamic {
        pub ndim: usize,
        pub bins: [usize; 2],
        pub log_bounds: Option<[[f64; 2]; 2]>,
        pub primary: Vec<Row>,
        pub recent: Vec<Row>,
        pub primary_capacity: usize,
        pub recent_capacity: usize,
        pub max_age: usize,
        pub next_admission: usize,
        pub min_observations: usize,
        pub lower: Option<f64>,
        pub upper: Option<f64>,
        pub fallback: f64,
        pub factors: Map<[usize; 2], f64>,
    }

    #[derive(Deserialize)]
    pub(super) enum Store {
        Linear(Linear),
    }

    #[derive(Deserialize)]
    pub(super) struct Observation {
        pub raw_x: [f64; 2],
        pub observed_ms: f64,
    }

    #[derive(Deserialize)]
    pub(super) struct Fit {
        pub intercept: f64,
        pub coefficients: [f64; 2],
        pub means: [f64; 2],
        pub scales: [f64; 2],
        pub active: [bool; 2],
    }

    #[derive(Deserialize)]
    pub(super) struct Recursive {
        pub count: usize,
        pub means: [f64; 3],
        pub scatter: [[f64; 3]; 3],
        pub mutations_since_rebuild: usize,
        pub damaged_downdate: bool,
        pub ridge_scale: f64,
        pub rebuild_interval: Option<usize>,
        pub non_negative: bool,
    }

    #[derive(Deserialize)]
    pub(super) struct Update {
        pub errors: Vec<bool>,
        pub bad: usize,
        pub accepted: usize,
        pub age: usize,
    }

    #[derive(Deserialize)]
    pub(super) struct Linear {
        pub samples: Samples<Observation>,
        pub min_observations: usize,
        pub fit: Option<Fit>,
        pub recursive: Recursive,
        pub update: Update,
    }

    #[derive(Deserialize)]
    pub(super) struct Evidence {
        pub observations: u64,
        pub stable_ewma: f64,
        pub online_ewma: f64,
        pub active: u32,
        pub streak: u64,
        pub switches: u64,
    }
}

fn decode_v1(path: &Path) -> layout_v1::StateFile {
    use bincode::Options;
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .deserialize(&std::fs::read(path).unwrap())
        .expect("state.bin no longer matches layout v1: bump STATE_SNAPSHOT_LAYOUT_VERSION")
}

#[test]
fn state_layout_v1_matches_an_independent_decoder() {
    assert_eq!(
        STATE_SNAPSHOT_LAYOUT_VERSION, 1,
        "update the layout_v1 mirror and this golden with the version"
    );
    let dir = tempfile::tempdir().unwrap();
    let mut regression = best(regression_config(DECODE, |config| {
        config.estimator_config.fpm_regression.fit.rebuild_interval = Some(17);
    }))
    .unwrap();
    regression
        .tune_with_fpms(&stream(DECODE, 181, 70, false, false))
        .unwrap();
    regression
        .save_state_snapshot(
            dir.path().join("regression"),
            &StateSnapshotOptions::default(),
        )
        .unwrap();
    let state = decode_v1(&dir.path().join("regression").join(STATE_FILE));
    assert_eq!(state.layout_version, 1);
    let layout_v1::State::Regression { stores } = state.state else {
        panic!("regression variant index changed")
    };
    assert_eq!(stores.len(), 1);
    // `ForwardPassRegressionWorkloadKind::PureDecode` is variant 0.
    assert_eq!(stores[0].0, 0);
    let layout_v1::Store::Linear(linear) = &stores[0].1;
    assert_eq!(linear.samples.total_observations, 64);
    assert_eq!(linear.samples.max_observations, 64);
    assert_eq!(linear.samples.buckets_per_axis, [4, 4]);
    assert!(!linear.samples.fixed_bounds);
    assert_eq!(
        linear
            .samples
            .buckets
            .iter()
            .map(|(_, b)| b.len())
            .sum::<usize>(),
        64
    );
    assert_eq!(linear.recursive.count, 64);
    assert_eq!(linear.recursive.rebuild_interval, Some(17));
    assert_eq!(linear.recursive.ridge_scale, 1e-9);
    assert!(linear.recursive.non_negative);
    assert_eq!(linear.min_observations, 5);
    assert!(linear.fit.is_some());

    let artifact = dir.path().join("forest.json");
    std::fs::write(&artifact, forest_artifact(AGG, None, None)).unwrap();
    for (mode, expected) in [
        (CorrectionMode::Fixed, "fixed"),
        (CorrectionMode::DynamicLocal, "dynamic"),
        (CorrectionMode::Hybrid, "hybrid"),
    ] {
        let mut learned = best(learned_frozen_config(AGG, &artifact, mode)).unwrap();
        learned
            .tune_with_fpms(&stream(AGG, 191, 40, true, false))
            .unwrap();
        let path = dir.path().join(expected);
        learned
            .save_state_snapshot(&path, &StateSnapshotOptions::default())
            .unwrap();
        let layout_v1::State::Learned { corrections } = decode_v1(&path.join(STATE_FILE)).state
        else {
            panic!("learned variant index changed")
        };
        let observed = match &corrections.decode {
            layout_v1::Correction::Fixed(fixed) => {
                assert_eq!(fixed.min_observations, 3);
                assert_eq!((fixed.min_faster, fixed.max_slower), (Some(0.5), Some(2.0)));
                assert!(fixed.samples.fixed_bounds);
                "fixed"
            }
            layout_v1::Correction::Dynamic(dynamic) => {
                assert_eq!((dynamic.ndim, dynamic.bins), (2, [4, 4]));
                assert_eq!(
                    (dynamic.primary_capacity, dynamic.recent_capacity),
                    (48, 16)
                );
                assert_eq!(dynamic.max_age, 128);
                assert!(dynamic.next_admission > 0 && dynamic.log_bounds.is_some());
                "dynamic"
            }
            layout_v1::Correction::Hybrid(pair) => {
                assert_eq!(pair.1.recent_capacity, 16);
                "hybrid"
            }
        };
        assert_eq!(observed, expected);
    }
    let (_checkpoint, config) = training_checkpoint(DECODE, LearnedAdaptation::CausalSelector);
    let mut selector = best(config).unwrap();
    selector
        .tune_with_fpms(&stream(DECODE, 201, 9, false, false))
        .unwrap();
    selector
        .save_state_snapshot(
            dir.path().join("selector"),
            &StateSnapshotOptions::default(),
        )
        .unwrap();
    let layout_v1::State::CausalSelector {
        corrections,
        selector,
    } = decode_v1(&dir.path().join("selector").join(STATE_FILE)).state
    else {
        panic!("selector variant index changed")
    };
    assert_eq!(corrections.len(), 1);
    // PureDecode store, decode query: key 1 in store-major order.
    assert_eq!(selector[1].observations, 9);
    assert!(selector.iter().enumerate().all(|(index, evidence)| {
        index == 1 || (evidence.observations == 0 && evidence.active == 0)
    }));
    assert!(selector[1].stable_ewma.is_finite() && selector[1].online_ewma >= 0.0);
    assert!(selector[1].switches <= 9 && selector[1].streak < 2);

    // Manifest shape: the file inventory each kind pins.
    let files = |name: &str| -> Vec<String> {
        read_json(&dir.path().join(name).join(MANIFEST_FILE))["files"]
            .as_array()
            .unwrap()
            .iter()
            .map(|file| file["path"].as_str().unwrap().to_owned())
            .collect()
    };
    assert_eq!(files("regression"), [STATE_FILE]);
    assert_eq!(files("hybrid"), [LEARNED_ARTIFACT_FILE, STATE_FILE]);
    assert_eq!(
        files("selector"),
        [
            "learned/online/manifest.json",
            STABLE_ARTIFACT_FILE,
            STATE_FILE
        ]
    );
    let manifest = read_json(&dir.path().join("regression").join(MANIFEST_FILE));
    assert_eq!(manifest["schema"], "aisimulate_fpm_state_snapshot");
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(manifest["producer"]["state_encoding"], "bincode1_fixint");
    assert_eq!(
        manifest["producer"]["aisimulate_core_version"],
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(manifest["kind"], "regression");
}
