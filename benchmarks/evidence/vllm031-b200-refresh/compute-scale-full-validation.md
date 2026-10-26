# B200 compute-scale full collection validation

**Pass bounded artifact validation: 1,628 completed callbacks, zero failures, 3,256 valid paired rows. Preserve the four clipped zeros. No collection rerun or row replacement is required by this evidence.**

- GitLab pipeline **76775056**, child **76775781**, job **510166928**; Slurm **5314088** completed.
- AISim source `46bbd3472aa919ca63213a450c6336761621106c`; runner `932f69168edc4e149f013cb729f502c88b88356b`.
- Allocation: one B200 for **399 seconds / 0.11083 GPU-hours**, from the campaign scheduler record.
- No data PR, default selection, raw artifact or scheduler state was changed by this validation.

| Check | Result |
|---|---|
| Exact pinned getter | 74 M values × 22 K values = 1,628 distinct callbacks |
| Checkpoint | 1,628 done; zero failed; zero unresolved; no missing or extra planned IDs |
| `computescale_perf.parquet` | 1,628 unique full keys; finite, nonnegative latency; four zeros |
| `scale_matrix_perf.parquet` | 1,628 unique full keys; finite, strictly positive latency |
| Paired coverage | Identical M/K sets; 3,256 total rows |
| Metadata | Exact framework, version, B200, kernel source, FP8 dtype, collector reference and complete status |
| Integrity | No duplicate keys, nulls or nonfinite values; retained CSVs exactly match Parquet; exported copies are byte-identical |
| Archive | ZIP CRC passes; both tar copies match; every extracted member matches its archive |
| Historical failure baseline | Original 0.25 campaign: 0/1,628 failed; both original compute-scale failure shards empty |

Recomputed using the pinned generator, getter, case-ID helper and provenance code:

- Case-plan hash: `sha256:08d764dbdc12e324a3a26e0c3d1b7e3359dbc43cd8bc36c09b20f94980e85092`.
- Collector closure hash: `sha256:83a18ea8c872bedbd7bf51d6f16133b69cfb43dc96ab1253b8d3576aa538fd50` across 40 files.
- Both hashes match both table metadata entries. Failure rates come from reports/checkpoints, not Parquet row counts.

## Four valid clipped zeros

The native contract is `compute_scale = max(0, dynamic/5 − static/5)` and `scale_matrix = static/5`. Compute-scale is a subtraction estimate between different quantizers, not a standalone kernel latency. The full artifact retains the estimate and static companion; it does **not** retain the original raw dynamic duration.

| M | K | Full overhead (µs) | Full static companion (µs) | New diagnostic signed differences, three batches (µs) |
|---:|---:|---:|---:|---|
| 32,768 | 32 | 0 | 21.444267 | −0.003201, +0.023466, −0.005335 |
| 16,384 | 32 | 0 | 12.401066 | +0.018133, −0.022400, +0.090668 |
| 8,192 | 32 | 0 | 7.492267 | +0.408534, +0.346667, +0.246400 |
| 4,353 | 32 | 0 | 5.054933 | +0.059733, +0.077867, +0.023467 |

Diagnostic job **2388720** completed all **12/12 callbacks**, preserving **24 finite positive raw component durations**, CUDA graph use and five-call normalization. All **24 native quantized-output/scale checks** have zero maximum error. Exact result keys, source, runtime-manifest hash, helper hash, three batches and four selected original case IDs were independently verified. Archive SHA256 is `29d6b4dd80d4e29edb57552ae98dd312999139a11e52ce19ba4ad700ba6bf444`.

The diagnostic allocation ended **FAILED after the probes**: its wrapper fell through to the generic GEMM runner and raised `KeyError('cases')`. The parent fixed the missing exit. Preserve the failed terminal record and the successful probe records separately; rerunning successful probes is unnecessary. The inherited plan includes the original 16-case inventory, but `selected_case_ids=[8,9,10,11]` selects four cases per batch; the effective count is 12.

These new durations are independent diagnostics, not recovered original components. Positive differences in a later run do not invalidate or replace the full-run zeros. No ratio against zero is used.

## Source, numerical and runtime limits

- Earlier job **2385206** has 48 successful sampled callbacks across 16 shapes and **96 native numerical checks**, all with zero output/scale error. Its collector module, timer, shape generator, grid and vLLM utility source are byte-identical between `53fb6dd3` and collection source `46bbd347`. The runtime manifest hash is the same `7e72f1e6…95eb593`.
- Current main checked at `dcadbc5087bc039bbe2672e7888440b792813720`. Getter, callback and timing-helper ASTs match; grid/helper/generator bytes match. Main still admits only vLLM ≤0.25.0 and has no campaign runtime manifest. The qualification branch supplies target 0.31 admission. This integration prerequisite remains separate from artifact validity.
- Auto-collection used host driver **615.71.10**, reporting CUDA UMD **13.4**. Personal numerical/component runs used driver **580.82.07**, with pinned container CUDA **13.0**. Same pinned image/source does not establish identical host runtime conditions.
- Auto logs requested **1965-MHz SM / 3996-MHz memory** clocks. The command returned “All done” with a deprecation warning; there is no recorded per-callback realized-clock trace or GPU UUID. One startup snapshot shows 40°C and 191 W / 1000 W. Do not claim a verified sustained clock lock.
- Live preflight records installed vLLM 0.31.0 and x86_64. Immutable image/framework source identity is supplied by the checked campaign manifest. The file named `collector_git_version.txt` contains the Git tool version, not a source SHA; checkout logs, metadata and closure hashing establish the collector source.
- Numerical qualification covers sampled tensors, not every full-grid shape, GEMM correctness or end-to-end prediction accuracy. The full run does not erase earlier reuse decisions: 15 sampled compute-scale keys remain approved, while the `M32768/K65536` six-repeat variability record remains attached to its fresh data. Static results retain 13 sampled reuse approvals and three drift failures.

The paired artifacts are ready for parent review. Publication still requires the existing integration, exact-head CI, signature and promotion checks. Detailed cases, hashes, latencies, component records and source evidence are in [the validation JSON](compute-scale-full-validation.json).

Sources: [native paired callback](https://github.com/ai-dynamo/aisimulate/blob/46bbd3472aa919ca63213a450c6336761621106c/python/aisimulate/collector/vllm/collect_computescale.py#L32), [registered shape expansion](https://github.com/ai-dynamo/aisimulate/blob/46bbd3472aa919ca63213a450c6336761621106c/python/aisimulate/collector/case_generator.py#L1759), [collector/case provenance hashes](https://github.com/ai-dynamo/aisimulate/blob/46bbd3472aa919ca63213a450c6336761621106c/python/aisimulate/collector/provenance.py#L184).
