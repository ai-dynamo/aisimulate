# B300 batch-52 native serving-padding evidence

Job **1024034** on AWS PDX Slurm B300 `pool0-0231` completed successfully.
Collector source was signed commit `74adc4e62`; source/harness archive hashes
are in `input-sha256.txt`. Runtime image:
`images/sglang-0521-2dbe4c7f.sqsh`, SGLang 0.5.21; the native class source hash
is recorded in each result. The probe uses eight local heads, FP8 KV, BF16
projections and TP8-local geometry with two dummy model layers.

The harness invokes **the installed `DecodeCudaGraphRunner.load_batch`
method**, using its native `DecodeInputBuffers` and registry, for 52 real
requests and a captured bucket of 56. It observes the method's metadata
dispatch view rather than executing a full-model graph. Collector padding
matches the native input IDs, positions, request-pool indices, GPU/CPU sequence
lengths, cache locations and sequence-length sum at both sampled histories,
in full-indexer and skip-indexer modes: **four checks passed**.

The four dummy slots use the actual backend's **sequence-length sentinel 1**
and request indices 0. They are not four additional full-history requests.
The module graph executes 56 slots and the persisted row retains 52 real
requests. No generic FPM batch-rounding rule is introduced.

| Mode | Past history | Padded B52, ms | Exact B52 diagnostic, ms | Exact B56 control, ms |
|---|---:|---:|---:|---:|
| Full indexer | 4,096 | 0.0867 | 0.0863 | 0.0872 |
| Full indexer | 16,384 | 0.1044 | 0.1049 | 0.1066 |
| Skip indexer | 4,096 | 0.0605 | 0.0602 | 0.0615 |
| Skip indexer | 16,384 | 0.0602 | 0.0612 | 0.0618 |

All four processes passed and produced 16 probe rows. Their measured
collection calls total **55.853 seconds = 0.015515 GPU-hours**, excluding
allocation time. The scheduler's 4m26s job duration is not used for collection
cost. These are bounded single observations, not a statistical equivalence
claim across the historical 44 batch-52 rows.

All results are **probe-only**. Exact-shape diagnostic rows cannot enter the
performance database. Existing batch-52 data needs recollection with the
corrected native-padding collector and fresh provenance before claiming this
boundary for that published data.

Reproduction uses the included `padded_worker.py` and `padded.sbatch`, an exact
signed collector archive, the stated native image, and the existing GLM
config. Remote result root:
`/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec/results/pr664-padded-serving-20261103/1024034`.

This validates native load-batch buffer preparation and sampled module graph
execution. It does **not** establish full-model serving replay, TP8 E2E
accuracy, or numerical attention correctness.

## Published artifact representation

`executed-artifacts.zip` preserves the original text outputs and executed
harness/script bytes. Readable copies normalize line endings/trailing whitespace;
the two script copies add SPDX headers only. Measurement values and executable
statements are unchanged. `artifact-hashes.json` records original and published
hashes; the original harness hash remains in `input-sha256.txt`.

## GLM attention geometry source

Pinned SGLang source `e00930c5489053f26d86b179cee0d087f846acbb`,
`python/sglang/srt/models/deepseek_v2.py`, `DeepseekV2DecoderLayer` at lines
2552–2562 passes explicit `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`,
`q_lora_rank` and `kv_lora_rank` into `DeepseekV2AttentionMLA`. Its constructor
at lines 1993–1998 stores those dimensions. Generic `head_dim` is not the
attention-dimension selector on this path. This source evidence supports the
model-profile equivalence check; unrelated quantization configurations are not
assumed equivalent.
