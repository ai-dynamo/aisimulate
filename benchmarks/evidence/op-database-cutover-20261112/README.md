# Op database cutover parity: pinned Hugging Face revision versus packaged tables

Post-flip go/no-go evidence for defaulting op-latency data to `ai-dynamo/aisim-data` and dropping the packaged
tables. `parity-real.json` is the output of `python/aisim/tools/perf_database/op_database_parity.py run` run on the
commit before the packaged tables were deleted: the packaged `systems/data` of this branch's base against the real
downloaded revision `07d52645fb2b021337dc9ce79fce8b5a526bfc51` (catalog SHA-256 `c0a0a65b...`, not a locally built
tree), loaded with `aisim_core.data.op_tree.load_op_latency_tree`. Local paths are replaced by `<scratch>`.

| Gate | Result |
| --- | --- |
| rows | true. Every packaged file has a loaded counterpart: 902 byte-identical files, 1,441 tables with equal Arrow row hash (31,873,230 rows), 4 empty placeholders and 6 non-runtime README/report files outside every manifest. |
| lookup | true. `perf_data_reuse_manifest.yaml` and `query_versions.yaml` rebuilt from the Parquet tables are byte-identical. |
| predictions | true. 11 model/system/backend scenarios, op-level estimator through `best_available`, 924 values compared, 0 mismatches, max absolute difference 0.0 ms. |
| offline | true. Predictions ran with sockets blocked; an offline pin with an empty cache raises `MissingDataError` and never falls back. |

One-time load: 884 entries, 2,343 files, 182.8 MB verified in about 1.4 s; the composed root is 195 MB with specs.
Timing and RSS in the file come from a single repeat and are within noise.

The converter fix that made the refreshed tree possible (`reuse-manifest` analysis statistics are optional in
`aisim-op-reuse-manifest/v1`) is the first commit of the cutover PR.
