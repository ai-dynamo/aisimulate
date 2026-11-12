# Op database parity: HF-layout tree versus packaged (AISIM-2063)

Local, offline, read-only. Tree built locally from commit `242e71a3f` by
`tools/perf_database/publish_op_latency_tree.py build` (`tree-build.json`; 1,595 files, 625 manifests). Loaded with
`aisim_core.data.op_tree.load_op_latency_tree`, compared by
`tools/perf_database/op_database_parity.py run` (`parity.json`) and `... drill` (`rollback-drill.json`).
Machine: Linux x86_64, CPU only, 5 fresh processes per source for the timing medians. Nothing was uploaded and no
default changed.

| Check | Result |
| --- | --- |
| Rows and hashes | 1,597 packaged files under `systems/data`: 4 empty `.gitkeep` placeholders not published; 643 byte-identical; 944 op tables with equal Arrow row hash, equal row counts (17,360,711 rows) and equal column types; 6 system-level README/report/provenance files outside every version manifest are published byte-identical in the tree but are not runtime inputs, so the loaded root omits them; 0 problems, 0 unaccounted files. |
| Lookup metadata | `perf_data_reuse_manifest.yaml` and `query_versions.yaml` rebuilt from the Parquet tables are byte-identical to the packaged files. |
| Predictions | 11 usable model/system/backend scenarios (Qwen3-32B on h200 vllm, h100 trtllm, b200 sglang, gb300 trtllm, l40s vllm; Llama-3.1-8B a100 trtllm; Llama-3.1-70B h100 vllm; Qwen3-30B-A3B h200 vllm; Qwen3-235B-A22B b200 trtllm; DeepSeek-V3 b200 sglang and h200 trtllm), op-level estimator through `best_available`, 3 prefill and 3 decode shapes each. 924 values (totals plus per-op latency, energy, source and fallback lists) compared: 0 mismatches, max absolute difference 0.0 ms. The tree run selected only the loaded root. |
| Offline | Predictions ran with sockets blocked (`connect` and DNS raise) and `HF_HUB_OFFLINE`, `AISIM_DATA_OFFLINE` set. `ensure_data` with `DataPin(local_root=tree, offline=True)` wired one entry; a family-less selection fails with `data-conflict`; an offline HF pin with an empty cache raises `MissingDataError` and leaves the configuration unchanged (no packaged fallback). |
| Load cost (one-time) | 625 entries, 1,587 runtime files, 105.8 MB verified and copied in 1.0 s; resulting root 118.1 MB with specs (copy, not hard link). |
| Startup, one scenario, median of 5 | Wall 1.232 s packaged vs 1.247 s HF-layout; model build 0.872 vs 0.878 s; first prediction 0.220 vs 0.220 s; process peak RSS 137.6 vs 137.5 MiB. Differences are within run-to-run noise. |
| Sweep, 11 scenario builds, 66 predictions and 66 diagnostics calls in one process | 6.42 s packaged vs 6.61 s HF-layout; peak RSS 251.8 vs 251.8 MiB. |
| Rollback drill | With the pin unset (default path) predictions equal the pinned ones (924 values, 0.0 ms) and come from the packaged root; pin variables are inert when unset. |

Not covered: the real dataset revision `ai-dynamo/aisim-data@07f0080d1ab09d36c7f8b4197b5b8b6e7490cd5b` (all numbers above are measured on the locally built tree, not on that revision; the parity run must be repeated on it before any default changes), a real Hugging Face download (the dataset is not published; the repo had no catalog on 2026-11-07),
Rust consumers reading the loaded root without Python (a `cargo test` pass against it), the wheel built without
packaged data, GPU-backed accuracy, and the FPM and E2E categories. The startup figures exclude the one-time load
and the first-use download. The repository's existing per-configuration `materialize_systems_root` cannot assemble
the whole tree because an op-latency selection without `operation_family` matches many entries (data-conflict);
`load_op_latency_tree` covers that case.
