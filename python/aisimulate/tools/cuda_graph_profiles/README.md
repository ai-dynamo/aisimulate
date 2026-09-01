# CUDA graph profile maintenance

This tool reproduces the packaged vLLM CUDA graph reservation database from a
small, human-reviewed set of InfX artifacts. It never checks raw artifacts into
the repository.

## Contract

- `infx_sources.yaml` is the reviewed source manifest.
- `infx_sources.lock.json` pins the successful run attempt, head SHA, artifact
  ID/name, and every extracted-file SHA256.
- `cuda_graph_profiles.parquet` is the measurement source of truth. Generated
  review reports live under `reports/v1`; only the Parquet, metadata, and model
  artifacts are packaged.
- The pre-KV vLLM estimate is eligible for lookup and training. The later actual
  graph-pool size is diagnostic only.
- Concurrency is provenance. It is not profile identity or a model feature.
- A semantic duplicate with more than 5% reservation spread fails publication.

The generated database and reports contain no raw logs, credentials, request
records, or internal filesystem paths.
Parser tests use short synthetic, redacted log snippets rather than source logs.

## Reproduce

Authenticate `gh` for `SemiAnalysisAI/InferenceX`, then run from
`python/aisimulate`:

```bash
cache_dir=$(mktemp -d)
python -m tools.cuda_graph_profiles --cache-dir "$cache_dir" reproduce
```

The command resolves approved sources, downloads them into the temporary cache,
verifies locked checksums, parses single-node and nested multinode logs,
publishes the database and reports, trains the deterministic ridge artifact,
and validates all checksums.

Individual stages are also available:

```bash
python -m tools.cuda_graph_profiles --cache-dir "$cache_dir" resolve
python -m tools.cuda_graph_profiles --cache-dir "$cache_dir" download
python -m tools.cuda_graph_profiles --cache-dir "$cache_dir" parse
python -m tools.cuda_graph_profiles --cache-dir "$cache_dir" train
python -m tools.cuda_graph_profiles --cache-dir "$cache_dir" validate
```

`download` replaces only the artifact-ID directories under the supplied cache.
Delete the temporary cache after review.

## Review checklist

- Confirm every new run attempt completed successfully.
- Review source mapping and identity reconciliation.
- Review training exclusions, especially actual-only legacy logs.
- Confirm the validation report and model gates.
- Inspect the Parquet diff; never add raw artifact files.
- Add parser fixtures when vLLM changes a log format.
