# Op-latency publish tree audit (AISIM-2340)

Local build only; nothing was uploaded. Source: packaged op data at commit
`52d2c6136d781d97c398035601b502e53679daec` (`simonec/unified-data`), builder
`python/aisim/tools/perf_database/publish_op_latency_tree.py build` (converter module `op_latency_publish.py` next to it), audit
`... audit` (re-reads every source and destination file from disk).

- Source map: `source_map.parquet` (schema `aisim-op-source-map/v1`), one row per source file with source and
  destination path, git blob, SHA-256, bytes and the content hash used. `audit.json` is the audit result.
- 1,595 files published from 1,599 source files; the 4 skipped are empty `.gitkeep` placeholders.
  625 version manifests (`aisim-artifact-manifest/v1`, category `op-latency`), plus category/dataset indexes and the
  coverage catalog, are generated.
- Hash audit: 0 problems, 0 unmapped source files, 0 stray destination files.

| Basis | Files | Check |
| --- | ---: | --- |
| `file-bytes` (collection_meta.yaml, reuse.yaml, errors json, README/report, `.parquet.source`) | 649 | source SHA-256 equals destination SHA-256 |
| `arrow-rows` (op tables, 17,360,711 rows) | 944 | SHA-256 of the Arrow IPC rows (no schema metadata) equal on both sides; Arrow `equals` also true |
| `embedded-source` (`perf_data_reuse_manifest`, `query_versions` YAML to Parquet) | 2 | verbatim YAML embedded in file metadata; its SHA-256 equals the source file's SHA-256 |

Op tables cannot stay byte-identical: none carried a schema ID and 166 used Snappy, so they are rewritten as
zstd Parquet stamped `aisim-op-latency/v1`. Row hashes show the rows did not change.

Reuse: all 59 `reuse.yaml` declarations resolve to a published donor manifest and table
(`catalog/op-latency-reuse.parquet`).

Size: source 124.1 MB (121.1 MB tables) to 111.0 MB published (`data/op-latency` 110.3 MB incl. 2.3 MB manifests;
`catalog/` 0.68 MB). 2,239 files in total.

Not covered here: Rust/Python predictor parity on the restamped tables (a later gate, not part of this change).
