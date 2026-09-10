# Prefix refinement v1

This is a separate, frozen study of discrete attention-prefix coverage. The
original pilot, 126-configuration calibration, 38-configuration holdout,
precision-v2 repeat, tables, and reports remain unchanged. This directory does
not make the previous unsupported predictions retroactively supported.

The current table keys prefix exactly and interpolates only its query/KV axis.
For bounded prefill, the late attention layers see
`effective_prefix = original_prefix + max(query - 128, 0)`. Ten original bounded
holdouts lacked the resulting prefix curves. A separate, diagnostic serving
run also encountered batch-one full prefill at prefix 256. These observations
motivate this explicitly targeted refinement; they are not new independent
accuracy evidence.

## Frozen sampling design

The new bounded calibration adds 18 prefill configurations, all with query 128:

| Batch | Original prefix | Count |
| --- | --- | ---: |
| 1 | 64, 192, 576, 768, 1600, 1792 | 6 |
| 2 | 64, 192, 576, 1600 | 4 |
| 1 and 2 | 2, 130, 514, 1538 | 8 |

The last eight configurations support fresh query-130 holdouts on both sides of
the decoder replay boundary. Calibration uses one warmup and three measured
repetitions per configuration: 54 actual measured model invocations. There is
no new full-profile calibration and no new baseline measurement.

The count of 18 is a **conditional minimum for this fixed target and the
current exact-prefix consumer**. Ten distinct `(batch, late-layer effective
prefix)` curves are missing for the original bounded holdouts; the new
query-130 target adds eight other missing curves. One homogeneous model
invocation supplies only one such late-layer prefix curve for its batch.
Consequently at least 18 new homogeneous configurations are necessary to
populate these 18 curves without changing the consumer. This is not a
minimum sample count for arbitrary V4.1 workloads, other interpolation
contracts, or overall statistical accuracy.

The 46 fresh forward holdouts in **each** profile are:

| Phase | Batch | Query or past KV | Original prefix | Count |
| --- | --- | --- | --- | ---: |
| Prefill | 1 and 2 | query 31, 80, 126 | 0, 128, 512, 1536 | 24 |
| Prefill | 1 | query 257, 320 | 0 | 2 |
| Prefill | 1 | query 64, 96 | 256 | 2 |
| Prefill | 1 and 2 | query 130 | 0, 128, 512, 1536 | 8 |
| Decode | 1 and 2 | past KV 80, 160, 320, 640, 1920 | n/a | 10 |

Every configuration is distinct from the original calibration, original
holdouts, current calibration, and selected pilot reuse. Each has one warmup
and ten measured repetitions, with the component recorder absent. There are
92 new profile/configuration pairs and 920 measured forward invocations;
neither TP ranks nor repetitions are independent workload configurations.
Together with calibration, the campaign records 974 measured invocations per
rank and 110 warmup invocations: `18*3 + 92*10` measured and `18 + 92` warmups.
Four TP rank records describe each of those same invocations; they are not
four independent sample points. The manifests were frozen before any of
these measurements, and all 46/profile holdouts are disjoint from old and new
calibration and the original holdouts.
Decode seeds exactly the manifest's past KV, then measures native inclusive
KV equal to past KV plus one.

These choices keep batch at one or two, total new prefill tokens at most 512,
and native context at most 2048. They cover short extensions, the replay
boundary, and both compression ratios beyond the candidate budget. They do
not qualify batch three, heterogeneous extensions, arbitrary prefix curves,
longer contexts, new corpora, or a different serving kernel configuration.
The grid is a bounded response to identified prefix gaps, not a claim of
uniform coverage over all legal inputs.

## Restricted reuse and source precedence

For each profile, only attention rows projected from the two original pilot
configurations `(batch=1, prefix=256, query=3 or 128)` are eligible for reuse.
Admission first rebuilds the complete pilot table from all four raw ranks and
checks it against the previously published table. Native source, checkpoint
config, image digest, text/tokenizer provenance, eager execution, local timing
scope, and each component's dispatch identity must match the formal study.

The pilot's default collective settings differ from the formal study. This
reuse is restricted to the attention module's local compute interval, whose
collective is outside the timed module. It does not assert equal full-forward
runtime configurations. The independent new forwards retain the formal
unfused communication and shared-expert settings.

Physical keys are merged in order: original formal calibration, new bounded
calibration, then selected pilot attention. First source wins; duplicates
within a source are rejected. Original formal values and all original
baseline tables remain byte-identical. No holdout component timings are
collected or admitted. The predicted combined table sizes are 848 full and
948 bounded module keys, including respectively 12 and 10 additional pilot
keys. The bounded calibration contributes another 108 new keys. Both profiles
retain their original 80 baseline keys.

The frozen key projection gives 46/46 new holdouts with every required curve
present and every query/KV coordinate inside the observed curve range. This
is a geometric projection; actual strict SILICON query coverage is checked
after raw-data admission. It does not relax any consumer interpolation or
provenance rules.

## Execution, admission, and reproduction

`plan-receipt.json` freezes the canonical manifest digests, immutable runtime
identity, repeat counts, and hashes of the original artifacts. GPU execution
uses the original frozen native producer from commit
`6853dab695eede0ca02ea3fb901312766b0d08c6`, the same source image and tokenizer
text as the existing study. The public source and upstream attribution are
already recorded in the adjacent [study documentation](../study/README.md)
and repository notices. Allocation, cache, and model-storage paths stay in
private campaign receipts.

Run these commands from the repository root with its development environment:

```bash
export PYTHONPATH=python/aisimulate/src:python/aisimulate
python data/experimental/deepseek-v41/gb300-silicon/prefix-refinement-v1/refinement.py check-plan
pytest -p no:timeout data/experimental/deepseek-v41/gb300-silicon/prefix-refinement-v1/test_refinement.py
```

Use the checked-in `calibration-plan.json` with the original native runner,
`--warmup 1 --iterations 3`, bounded manifest and native replay flag. Run each
profile's `heldout-plan.json` with `--forward-only --warmup 1 --iterations 10`.
Do not enable baseline collection for this refinement. All three attempts
must have separate fresh output directories. The native arguments must
otherwise exactly match the preserved original calibration/precision
execution contracts, including both disabled CUDA graph backends, explicit
unfused all-reduce, and disabled shared-expert fusion.

Admission requires every planned warmup and measured invocation, each TP rank
exactly once, complete module membership, finite outputs, consistent input
and dispatch provenance, and a fresh completion receipt. Forward timing is
the native synchronized benchmark wall interval including preparation,
forward and sampling. It includes all model layers and shared Engram hashing;
it is distinct from both GPU-timed serving FPM and HTTP end-to-end latency.

```bash
python data/experimental/deepseek-v41/gb300-silicon/prefix-refinement-v1/refinement.py build \
  --calibration "$CALIBRATION_ATTEMPT" \
  --full-forward "$FULL_FORWARD_ATTEMPT" \
  --bounded-forward "$BOUNDED_FORWARD_ATTEMPT"
python data/experimental/deepseek-v41/gb300-silicon/prefix-refinement-v1/refinement.py verify
```

`build` validates all three attempts before publishing separate full/bounded
systems overlays. `verify` reproduces those tables from the original formal,
selected pilot and new calibration evidence, verifies unchanged baselines,
re-aggregates the independent forwards, and queries every module key with
strict SILICON and shared-source fallback disabled. Per repetition, the
observation is the maximum across the four TP ranks; the reported point is
the median of those rank maxima. Failed and incomplete attempts remain
private evidence and cannot become an accepted output. Predictions and
accuracy results are a separate report; no correction is fitted here.
