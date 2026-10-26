# MSA continuation 510808560: independent acceptance review

**PASS for the exact selected collection scope.** All **7,478 new callbacks** have matching native completion logs, checkpoint done entries and unique finite positive Parquet measurements. There are **zero new failures or unattempted selected cases**. This permits closing the collection lane; it does not publish data or extend numerical qualification.

| KV cache | GEMM | Seed rows preserved | New rows | Total measured | Inherited failures preserved |
|---|---|---:|---:|---:|---:|
| BF16 | BF16 | 2,332 | 0 | 2,332 | 496 |
| FP8 | BF16 | 2,332 | 0 | 2,332 | 452 |
| BF16 | FP8-block | 1,850 | 482 | 2,332 | 452 |
| FP8 | FP8-block | 0 | 2,332 | 2,332 | 0 |
| BF16 | NVFP4 | 0 | 2,332 | 2,332 | 0 |
| FP8 | NVFP4 | 0 | 2,332 | 2,332 | 0 |
| **Total** | | **6,514** | **7,478** | **13,992** | **1,400** |

The original **17,568** identities remain accounted as **13,992 measured + 1,400 inherited failed + 2,176 excluded unattempted**. The failed set is byte-for-byte equivalent as identities to the frozen seed: 44 score-address failures from job510058975 and 1,356 native CSR-ratio failures from job510388495. No failure was erased or retried. The single error-summary object is the expected synthetic `UnresolvedFailures` for those 1,400 records, not a fresh case failure.

Validation completed:

- Parquet and retained CSV each contain exactly **13,992 unique full identities**, with no duplicates or invalid latencies. Every Parquet latency equals its retained CSV value; range **0.0212–28.4238 ms**.
- All **6,514 inherited CSV records are unchanged**. All 7,478 new row keys exactly match the frozen continuation plan and its native completion log lines.
- Final checkpoint: 13,992 done, 1,400 failed, zero pending attempts. Retained pre-finalization snapshots contain the same terminal sets and 15,392 pending attempts. The finalizer correctly closed those attempts after writing the sidecar.
- Sidecar schema validation passes. Recomputed collector closure is `sha256:e2eb81ce4bf514c4188e02bea3a75f6d1adb5135513e51515d3650bc1f0493c4`; recomputed attested-case hash is `sha256:f719cd74c30eb8d1eac7a30129a0411e88b3b9d43d6dbd15795011f88d5e2f9c`.
- Retained transaction hashes match the final Parquet, raw CSV and sidecar. No active transaction journal remains. Exported Parquet and sidecar copies match the raw result directory exactly.
- Job trace matches source `4b14219a22891ec643fe1a18c4ed0141cc98a36e`, runner `4c7fff85c8e8fa420afe1ad332fd3692b6ae1128`, approved image and manifest. Runtime preflight confirms vLLM0.31.0; serialization preflight confirms Pandas2.3.3/PyArrow24.0.0.
- ZIP and tar hashes were independently rechecked: `69a1062e4d6e54db5679e943551f4321c068a9ccaf96ab8950cf58edcad3da14` and `db85329941ddecb4500507c76ee5cf346ac5983332ae1b5e55f4386c70a8cc21`.

Preserve per-row provenance when publishing: **184 rows** retain job510058975/source9b4de9d, **6,330 rows** retain job510388495/source362953, and only **7,478 rows** were measured by this continuation/source4b142. Both original seed provenance files and the exact exclusions are unchanged. The schema1 aggregate finalizer reference does not replace this mapping.

No continuous per-case clock/power telemetry was recorded, so constant clocks are not established. Existing numerical checks retain their bounded scope. The remaining 3,576 original-grid gaps stay explicit; no defaults, optional publisher or merge action was taken.
