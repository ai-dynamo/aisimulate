# Ascend 910C measurement contribution proposal

This draft offers concrete Huawei Ascend measurement data for maintainer review.
Would the project accept Ascend 910C operator datasets and their typed consumers?
Feedback on the preferred data location, contract and upstreaming sequence is
requested before expanding this into production hardware support.

## Included evidence

- `ascend_vector_perf.parquet`: 160 measured attention-RoPE and split-Q/K-norm
  rows from SGLang `76f9213a411018547f4fd6a75f36feaa4d6bed58` on Ascend.
  There are 112 eager rows and 48 captured-graph rows. Two unstable eager
  attempts were excluded by the original collection campaign.
- `collection_meta.yaml`: original runtime/image identity, collector and case
  hashes. The recorded `torch: 2.10.0+cpu` package label is retained verbatim;
  execution used torch_npu, not CPU timings.
- `hardware-calibration.yaml`: the fork's measured Atlas 800T A3 calibration.
  A device is one runtime-visible chip: 16 chips on eight dual-chip boards.
  Compute coefficients are observed throughput, not advertised peaks.
- `manifest.json`: exact artifact hashes and immutable source paths.

Rows retain execution mode, dimensions, token count, kernel source, signature,
latency, CV, sample count, numerical-check count and runtime/source hashes.
The collection calls serving kernels with NPU event timing; it does not train
timings against end-to-end serving throughput.

## Integration boundary

These artifacts are intentionally in `benchmarks/evidence`, outside production
data discovery. This PR does not register a selectable hardware platform or
claim an upstream GLM/Qwen predictor works. The fork's custom
`glm52_modelslim_vectors/v1` table is not a drop-in replacement for CUDA data.
Its Rust query code and model composition need a separately reviewed port
through the canonical performance-model API.

The copied hardware file includes GLM-specific runtime-memory calibration and
references to the fork's broader evidence; it must not become a universal Qwen
memory model. Raw traces and all excluded-case records are not bundled here.
Their review and reproducible collection instructions remain prerequisites
for promoting this sample into a supported upstream dataset.

This sample does not establish whole-model accuracy, communication coverage,
performance on other Ascend products, Qwen Graph coverage or kernel speedups.
We propose subsequent changes for hardware identity, typed operator consumers,
collector parity and broader datasets, each with independent validation.

## Reproduce the artifact checks

With PyArrow and PyYAML installed, run from the repository root:

```bash
python benchmarks/evidence/ascend910c-data-proposal/validate.py
```

This checks artifact hashes, mode counts, unique measurement identities,
finite positive timings and collection row totals. It does not rerun NPU
measurements or validate prediction accuracy.

## Source and attribution

The files are copied without changes from
https://github.com/xuhuan51/aisimulate-910c/tree/8e583af366a33013748b06a8b09c665a396bf6e7,
which retains the history of https://github.com/yaleyoou/aisimulate-910c.
Original paths are in `manifest.json`; the fork's license is Apache-2.0.
Existing NVIDIA copyright notices and contributor attribution are retained.
This proposal does not claim authorship of the entire Ascend adaptation.
See the repository's `THIRD_PARTY_NOTICES.md` entry.
