# FastAFD measured MoE stage profiles

A profile is a JSON document owned by AISimulate that carries exact measurements of
the complete decode MoE stage, produced by FastAFD's MegaMoE model benchmarks.
AISimulate reads the document and never imports, installs, or runs FastAFD; no
FastAFD change is required to load a profile. Lookup is exact-match only, and a
missing measurement is an error, not an estimate.

## Document

```json
{
  "schema": "aisimulate.fastafd-moe-stage.v1",
  "lookup_policy": "exact-only",
  "source": {
    "repository": "https://github.com/<owner>/FastAFD",
    "commit": "<40-character git commit>",
    "benchmark": "fastafd.megamoe-colocated-model-benchmark.v2",
    "boundary": "<the benchmark's own description of what the measured interval covers>"
  },
  "entries": [
    {
      "key": {
        "model_path": "Qwen/Qwen3-235B-A22B-FP8",
        "system": "gb200",
        "stage": "agg",
        "topology": "ep8",
        "logical_batch_per_source_rank": 48,
        "mtp_nextn": 0,
        "microbatches": 1,
        "moe_layers": 94,
        "routed_topk": 8,
        "moe_precision": "w4a8_mxfp4_mxfp8",
        "moe_backend": "megamoe"
      },
      "model_profile": "qwen3_235b_fp4",
      "latency_ms": 12.37,
      "measurement": {
        "scope": "complete_moe_stage",
        "method": "cuda_event",
        "method_version": "torch 2.10.0+cu130; cuda 13.0",
        "statistic": "p50",
        "sample_count": 40,
        "cv_percent": 0.56,
        "routing": "balanced",
        "source_tree_sha256": "<SHA-256 of the source tree that ran>",
        "raw_artifact": "qwen3_235b_fp4_8ep_t48_n0_balanced_mega.json",
        "raw_sha256": "<SHA-256 of that raw result file>"
      },
      "validation": {"stable": true, "correctness": null}
    }
  ]
}
```

`source` names the code that produced every entry: the repository URL and commit,
the raw result schema that identifies the benchmark procedure, and the benchmark's
measurement boundary. The commit is not required to belong to the upstream FastAFD
repository; a fork or a modified tree is recorded as such, and each entry's
`source_tree_sha256` identifies the exact tree, so uncommitted modifications are
visible as a tree that differs from the commit.

The `key` identifies the model, hardware, stage (`agg` for a colocated EP group,
`afd` for an attention/FFN split), topology (`ep<N>` or `<A>A<F>F`), per-source-rank
batch, MTP degree, microbatches, measured MoE layer count, top-k, MoE precision, and
MoE backend. `latency_ms` is the p50 over `sample_count` timed iterations of the
stage for all `moe_layers` layers of one decode step. `routing` records the expert
routing distribution the benchmark used; `cv_percent` is the dispersion of those
iterations. `stable` is the benchmark's own stability gate and must be true;
`correctness` is the matched-output check against a reference backend, `null` when
the benchmark ran no reference.

## Building a profile

`python -m aisimulate.fastafd_profile <results_dir> --repository <url> --output <profile.json>`
converts one directory of raw benchmark results (one benchmark, one commit) into a
profile. Results with another routing distribution (`--routing`, default
`balanced`), an unstable timing, or a failed correctness check are skipped and
counted. Repeated measurements of one key are an error; keep one run per directory.
The raw results, the benchmark revision, and the procedure stay with the profile so
that every entry can be traced to its evidence.
