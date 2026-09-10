# Replicated indexer identity correction v2

This version corrects `geometry.index_n_heads` from 8 to 32 in the six archived
GB300 TP4 module tables. It contains **no new measurements**. All 4,080 rows
retain their original latency bits, source/configuration/image identity, timing
scope, kernel witness, sample counts, KV initialization, and execution profile.
Only the geometry label changes in the 3,670 attention rows. The 410 other rows,
all baseline tables, and hardware YAML files are unchanged. Original tables,
source receipts, sampling manifests, raw observations, and reports remain at
their original paths.

The SOL review identified that the SDK had divided indexer heads by TP while
the timed SGLang module replicated all 32 heads. The old recorder checked query
heads and other attention dimensions, but omitted indexer heads and projection
shapes. It obtained labels from the SDK manifest rather than from those loaded
dimensions. The corrected model and the new collector guard agree on 32 heads,
including a 4096-by-1280 query projection and 32-by-5120 head-weight projection.
The guard runs before instrumentation in both module and forward collection,
and validates both prefill and decode manifests.

## Evidence and limits

[source-proof.json](source-proof.json) binds all six original tables and eleven
archived source manifests. Each row's original `source_sha256` is the canonical
digest of that captured SGLang source manifest. All eleven record the actual
indexer source SHA-256
`187768ce4429c9b73e5ef85a05ce137a0e8db3b375e5ebb1ea356acaa64e86bc`.
That file is byte-identical to
[SGLang's pinned indexer implementation](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py#L203),
which retains all index heads and uses replicated projections.

This proves the captured indexer file's identity. It does not assert that the
entire loaded model matches the upstream revision: the captured
`srt/models/deepseek_v4.py` hash is
`d69b85051bcf4535993d9c2a6625a1e86386e99e1954bb9c2c25e47cd577a8a9`,
which differs from that upstream file. The subsequent
[actual runtime audit](actual-runtime-audit.json) binds the separately captured
installed model file to that exact archived hash: its indexer constructor passes
the original configuration directly to `DeepseekV41Indexer`, with no TP division.
The three archived native runners also match their recorded producer hashes;
their instrumentation changes mHC timing streams and the output projection's
reduction boundary, and does not assign indexer heads or projection dimensions.
This is a source audit, rather than a new measurement of a loaded module's
shapes. The new runtime dimension guard did not run during historical collection.

The [derivation receipt](derivation.json) records original and derived file
hashes and, for every row, original/derived canonical row hashes and the
unchanged IEEE-754 latency bits. It also binds the derivation program, source
proof, copied artifacts, and corrected consumer manifests. Sidecars retain the
original measurement producer, runtime, and case-plan identities; only the
derived table's data hash is updated. The extra receipt identifies the metadata
transformation separately from the original collection provenance.

## Consumer roots and reproduction

| Original cohort | Corrected systems root within this directory |
| --- | --- |
| Initial pilot | `initial/{full,decoder_bounded}/systems` |
| Formal study | `study/{full,decoder_bounded}/systems` |
| Prefix refinement | `prefix-refinement/{full,decoder_bounded}/systems` |

The two adjacent `*-manifest.json` files contain the corrected production
consumer geometry. The runtime version remains `0.0.0.dev0`; `indexer-identity-v2`
is an artifact revision. Corrected predictions must use these new roots.
Historical eight-head tables deliberately fail strict corrected-model lookup.
Historical prediction reports describe their original model and data versions;
new prediction comparisons must be written separately.

From the repository root, with PyArrow and PyYAML available, obtain the pinned
indexer file above and run the derivation into a fresh directory:

```bash
python data/experimental/deepseek-v41/gb300-silicon/indexer-identity-v2/derive.py \
  --indexer-source /path/to/dsv41_sparse.py \
  --output-dir /path/to/fresh-derivation
```

The program verifies the exact source hash and every frozen original table and
source receipt before copying data. It refuses to overwrite a derived dataset
or manifest, rejects unexpected original geometry and key collisions, and
checks all other columns and latency bits after Parquet serialization. The
checked-in CPU contract tests independently recheck every row and copied file,
compare the corrected manifests with the production model, and exercise strict
native SILICON queries against both corrected and historical identities.
