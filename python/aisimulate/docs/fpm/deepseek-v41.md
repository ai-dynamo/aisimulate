<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1 FPM collection

The collection path supports the text backbone with pure TP4, native checkpoint
precision, GPU-resident Engram, eager execution, and DSpark disabled.
Historical [GB200 and GB300 experimental evidence](https://github.com/ai-dynamo/aisimulate/tree/dd1fa97add17d3d74f580f4c3e0566c1b5f11827/data/experimental/deepseek-v41)
is preserved separately from the product code. The GB200 calibration is not
admitted for serving prediction: ordinary-serving validation exposed a large
latency mismatch. Experimental systems overlays do not change curated defaults.

## Execution identity

Schema 7 appends four strings to the existing exact model/backend/topology
identity: `model_config_sha256`, `execution_profile`, `engram_residency`, and
`input_modality`. The config hash uses canonical JSON after the same quantization
normalization used by the SDK. V4.1 uses `hbm_tp_sharded` and `text`; its profile
is `full` or `decoder_bounded`. Changing the config or profile cannot borrow
another cell. Schema 6 loads with the legacy values `""`, `full`, `none`, `text`.
Publication upgrades a validated schema-6 pair in memory before emitting schema 7.

The shared model retains the entire resident-weight inventory even when a replay
profile executes fewer decoder tokens. FPM interpolation uses the original
stage-aware SOL graph. Vision and speculative decoding remain outside this
campaign's measurement contract.

When a native FPM table labels FMHA by its cache precision, select that table
with `fpm_fmha_dtype: "fp8"` in native engine/replay JSON, or
`ModelConfig(fpm_fmha_quant_mode=FMHAQuantMode.fp8, forward_model="fpm")`.
This option requires `forward_model="fpm"` and changes only the exact FPM cell
selector. The checkpoint's analytical attention graph, interpolation SOL
anchors, and memory inventory remain unchanged. `activation_dtype` retains its
existing arithmetic-override meaning; it is not a substitute for this selector.
The historical GB200 comparison uses the table selector with no activation
override, and its optional op-level SOL comparison uses checkpoint precision.

`--fpm-decoder-replay` describes true bounded decoder execution. The current
vLLM route rejects it because the verified preview executes the full backbone.
Prefix-cache/SWA tail recomputation does not establish true decoder replay.
Replay OFF data must never be relabeled as ON data.

For Decoder ON, the FPM v1 telemetry API rejects an iteration with multiple
prefill requests and fresh prefill tokens. Its prompt-length variance cannot
prove equal current extends when requests have different cached prefixes or
completed chunks. Single-prefill and decode-only telemetry remain supported;
explicit homogeneous static inputs retain their separate table-query path.
This admission limit also applies to whole-forward FPM engines before lookup.
Historical reports retain their original predictor identities and coverage;
their supported counts do not describe this stricter current admission rule.

## Slurm transport

`--fpm-executor slurm` runs inside a caller-owned Slurm allocation with Pyxis.
It requires an explicit `--fpm-slurm-container-image` and accepts repeated
`--fpm-slurm-container-mount SOURCE:TARGET` arguments. The frozen Generator
manifest determines the node count; the allocated count must agree. A shared
campaign directory must be visible at the same path on each node.

The transport reuses Generator scripts, native result validation, attempt
identity, recovery, and formal publication. It launches one engine per node
with explicit ranks and a common master, and collects under stable `nodeNNNN`
execution-unit names. Cleanup cancels only receipted, campaign-specific steps
and verifies their exit; it never cancels the caller's allocation. Kubernetes
remains the default transport.

## Qualification and publication gates

Use a digest-pinned ARM64 image for GB200. The official preview recipe is at
[vLLM recipes commit ce19de141d448df3e739de977da0830e9a175de5](https://github.com/vllm-project/recipes/blob/ce19de141d448df3e739de977da0830e9a175de5/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml).
The model config revision is
[fb2764a5cf321eaa5070ca8f9e892818f477c16d](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/fb2764a5cf321eaa5070ca8f9e892818f477c16d).
Image support for loading the model does not prove that its Dynamo instrumentation
satisfies the Collector's native FPM contract. The ordinary preflight must pass.

Engram makes token history part of the workload. Qualification therefore needs
a reproducible tokenizer-generated text corpus and real model-computed KV for
both cached prefills and decode. V4.1 cached-prefill and decode publication reject `fake_fallback`,
legacy, and skipped-warm provenance; the consumer rejects them too. A producer
must earn the `real_kv` marker through execution. The initial canary should use
small batch/context limits, then expand the matrix only after native artifacts,
config identity, input provenance, and timing checks pass.

Record the immutable checkpoint/image/instrumentation revisions, corpus hash,
resolved engine settings, native artifacts and measured coverage with the data.
Keep internal cluster names, account names, filesystem paths, and raw operational
logs in private campaign storage.


The pinned producer is packaged under
`collector/fpm_forward/runtime/dsv41/`. It executes seed chunks on the same
request and block tables as the measured step, retaining Engram history and
compressor ring state. Source hashes cover the installed ARM vLLM files and
Dynamo scheduler/point/FPM definitions. `ai-dynamo-runtime==1.4.2` has passed
an actual import check with Dynamo source `54960177085413259859c88bd34ed0734d4c2ea9`.
This import check does not establish GPU numerical or timing correctness.

Its initial native grid is bounded to batch 2, per-request context 2048, total
scheduled prefill tokens 512, and global benchmark warmup 0. Use model limit
2050 to include the native decode context-2048 endpoint. Every completed point
archives actual prompt/output tokens in an adjacent JSONL file; the Collector
checks the file checksum, point coverage, per-request computed counts, and
completed seed witness. Unsupported or incomplete points fail the campaign.
The original mixed English/Chinese fixture is reproducible and is not a
representative production workload; Engram locality and routing sensitivity
remain limits on generalizing any resulting curve.

## Pinned Hugging Face profiles

FPM datasets are external artifacts. A checked-in pin manifest binds a Hugging
Face dataset commit to each system YAML, native parquet table, and metadata
sidecar by SHA256. Do not replace its immutable revision with `main` or a PR ref.
To materialize an overlay for the existing native reader:

```python
from pathlib import Path
import aisimulate_core
from aisimulate_core.sdk.fpm_dataset import materialize_fpm_profile
from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel

pin = Path(aisimulate_core.__file__).parent / "systems/dsv41_fpm_hf.json"
systems_path = materialize_fpm_profile(pin, "gb300-tp4-full", allow_unqualified=True)
config = ForwardPassPerfModelConfig(
    model="deepseek-ai/DeepSeek-V4.1-Flash", system="gb300", backend="sglang",
    worker_type="aggregated", backend_version="0.0.0.dev0",  # Historical identity.
    systems_paths=(str(systems_path),), tp=4, moe_tp_size=4, moe_ep_size=1,
    estimation_mode="fpm_interpolation", fpm_fmha_quant_mode="fp8",
    enable_shared_layer=False, strict_provenance=True,
)
model = RustForwardPassPerfModel.best_available(config)
print(model.estimate_forward_pass_time_ms({
    "version": 1, "wall_time": 0.0,
    "scheduled_requests": {
        "num_prefill_requests": 0, "sum_prefill_tokens": 0, "sum_prefill_kv_tokens": 0,
        "num_decode_requests": 1, "sum_decode_kv_tokens": 2048,
        "var_prefill_length": 0.0, "var_decode_kv_tokens": 0.0,
    },
}))
```

The packaged `systems/dsv41_fpm_hf.json` pins the three principal #158 tables
(GB300 TP4 `full` / `decoder_bounded`, and quarantined GB200 TP4 `full`) to
[HF dataset commit b35883ee5f8b4a82e24844a7e872056aff64287f](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/tree/b35883ee5f8b4a82e24844a7e872056aff64287f).
The dataset change is reviewed in [HF PR #11](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/discussions/11).
Historical component tables and observations remain in that dataset revision;
they are not substituted for the principal prediction tables.

The equivalent staging command, from this repository, prints the verified overlay:

```sh
python -m aisimulate_core.sdk.fpm_dataset \
  python/aisimulate/src/aisimulate_core/systems/dsv41_fpm_hf.json \
  gb300-tp4-full --allow-unqualified
```

Use `--local-files-only` (or the same Python keyword) after the first download to
run without network access. Every cache use rechecks hashes; corrupted files
fail instead of silently selecting another table. The native reader still owns
schema, real-KV provenance, execution-profile, model-config, and exact-cell
validation. This staging step does not alter interpolation or precision.

Historical #158 profiles retain their original development or quarantined
status, including the archived backend identity. Reproducing these experiments
requires `--allow-unqualified` (Python: `allow_unqualified=True`), which emits a
warning and does not grant serving admission. In particular, moving GB200 data
to Hugging Face does not resolve its ordinary-serving latency mismatch.

For restricted datasets, staging uses `HF_TOKEN` or the token saved by `hf auth
login` (`HF_HOME` / `HF_TOKEN_PATH` are respected). Credentials are never stored
in manifests or passed to redirected CDN requests. Set
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` to download anonymously. No additional Hub SDK
is required for this loader.
