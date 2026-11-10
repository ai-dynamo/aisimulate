# Collector v2 Case Files

Collector v2 plans collection from model YAML instead of treating the
collector as a flat op list. The whole declaration surface is:

```text
plan cases = dedup( expand(base_ops sweep grids) ∪ expand(model_case_values shapes) )
runnable   = plan cases  −  what collector/skips.py skips
```

There is intentionally no per-case selector or exception rule engine. Cases
that cannot run on some framework version fail at runtime and are recorded
with a classification and a (model, dtype) group label in the failure log; a
whole group failing is surfaced in the collection summary as a fix-me signal.
See `.claude/rules/collector/failure_handling.md` for the escalation rules.

## File Layout

- `base_ops/<op>.yaml`: shared recipe library. Only recipes named by a model's
  `base_ops` (or declared universal through the base file's `model_ops`) are
  activated.
- `models/<architecture>_cases.yaml`: architecture-specific op activation,
  model path aliases, and model shape values.
- `../skips.py`: the one list of cases a collection leaves out (see
  "Skipped Cases" below).

## Model Case Files

Model files are keyed by HuggingFace architecture name and list every model
path alias that should resolve to that architecture plan:

```yaml
schema_version: 1
architecture: Qwen3MoeForCausalLM
model_path: Qwen/Qwen3-235B-A22B
model_paths:
  - Qwen/Qwen3-30B-A3B
  - Qwen/Qwen3-235B-A22B
include_base: true
base_ops:
  - attention_context
  - attention_generation
  - gemm
```

`model_path` is the default representative model. The top-level `model_paths`
list resolves support-matrix names to this architecture plan; it does not by
itself multiply kernel cases.

### Op Sections

Op sections activate ops for the plan. `cases: all` is the only meaningful
value — there are no selectors:

```yaml
model_ops:
  - gemm
  - moe

all_frameworks_op_cases:
  moe:
    cases: all

framework_specific_op_cases:
  sglang:
    mla_context:
      cases: all
```

Use `all_frameworks_op_cases` when the op applies to every backend. Use
`framework_specific_op_cases` when only one backend should collect the op.

### Model Dimensions

Model-specific op dimensions live in `model_case_values`:

```yaml
model_case_values:
  moe:
    - model_paths:
        - Qwen/Qwen3-235B-A22B
        - Qwen/Qwen3-235B-A22B-FP8
      hidden_size: 4096
      inter_size: 1536
      topk: 8
      num_experts: 128
  mla:
    - model_path: deepseek-ai/DeepSeek-V3
      num_heads: 128
      q_lora_rank: 1536
      kv_lora_rank: 512
      qk_nope_head_dim: 128
      qk_rope_head_dim: 64
      v_head_dim: 128
```

The collector loads these values by op name and honors `COLLECTOR_MODEL_PATH`,
so a targeted run collects one model without editing Python. Two deliberately
different multi-name forms:

- `model_aliases` resolves multiple artifact names to one canonical physical
  case. Use it only for shape-only ops where the artifact does not affect the
  invoked kernel or persisted key.
- `model_paths` expands one physical case per path. Use it only when the model
  name changes runtime behavior, quantization policy, activation, or module
  loading.

Keeping these meanings separate prevents checkpoint suffixes from multiplying
the same shape by every independently swept quantization mode.

### Structural Attention Profiles

Attention correlations belong to the model, not to a global Cartesian product.
Store a native topology under `model_case_values.attention`:

```yaml
model_case_values:
  attention:
    - model_path: openai/gpt-oss-120b
      model_aliases: [openai/gpt-oss-20b]
      num_attention_heads: 64
      num_key_value_heads: 8
      head_dim: 64
      window_sizes: [0, 128]
      tensor_parallel_sizes: [1, 2, 4, 8, 16, 32, 64]
```

The generator expands only valid TP shards of that tuple. It never crosses one
model's query/KV heads with another model's head dimension or window.

Attention is planned once for every CUDA backend (`collector/op_plans.py`).
The base grid in `base_ops/attention_*.yaml` is generic full attention at
head_dim 128; every other head_dim and window comes from a model profile, and
full runs deduplicate the base and model shapes. A profile also carries the
model's attention properties: `v_head_dim`, `has_attention_sink`, and SGLang's
`sglang_runtime_window_size`, `sglang_attention_chunk_size` and
`sglang_scaling`. Each backend's rows take what its run function supports.
Layers with different properties (a sliding-window layer with its own runtime
window) get their own row. A targeted run with an explicit profile uses only
that model profile.

A profile declares no attention backend. SGLang's backend per architecture and
SM is `SGLANG_ATTENTION_BACKENDS` in `collector/case_generator.py`, each entry
citing the SGLang override that makes the choice; the plan reads it for the
dispatch and the SGLang worker checks it against what serving resolves on the
device, through the architecture's own model when SGLang registers an override
for it and through a model without one otherwise. vLLM rows of a model-planned
shape name the `[architecture, model]` of each profile that plans it; the vLLM
worker builds a model's own config, and selects under it, only for an
architecture vLLM has a config hook for, and runs every other row on the
generic config.

## Base Op Files

Shared sweep recipes live in per-op files under `base_ops/<op>.yaml`. For
cross-model ops such as MoE, MLA, Mamba2, GDN, and MHC, the base op file owns
the token counts, batch/sequence sweeps, parallelism sizes, routing
distributions, and generator constraints. Model YAML stores model dimensions;
base op YAML stores the reusable sweep policy.

MoE is planned once for every CUDA backend (`collector/op_plans.py`). The
shared quant-mode axis is `common_case_values.moe.quantization_modes`; a mode
may keep a checkpoint-specific format to the models that ship it
(`allowed_model_paths`) and carry a `module_config`. A model row declares its
checkpoint's modes in `quantization_modes`, the same for every backend; a row
that declares none plans every mode. Quant-sensitive checkpoint artifacts use
separate model rows even when all geometry fields are identical.
`framework_quantization.<backend>` holds only a backend's `module_config` (and
vllm_xpu's own policy). A mode an SM or one backend cannot run is an entry in
`collector/skips.py`.

Artifact-keyed vLLM MoE rows also require a checked-in
`src/aisimulate_core/model_configs/<org>--<model>_config.json`. The collector
uses that local structural config to reproduce serving routing without a Hub
download.

When frameworks invoke different routed-expert geometry for the same artifact
(vLLM routes Nemotron-H experts through a latent width), declare one row per
geometry; every backend plans both. A MoE row may not declare `frameworks`.
vLLM cannot run routed experts with TP and EP both above 1, so
`collector/skips.py` leaves `parallel: tp_ep` out of vLLM's MoE plan.

`include_base: true` means "include the small universal base set" declared by
base-file `model_ops` (currently dense attention and GEMM). Use an explicit
model `base_ops` list for auxiliary recipes, `framework_specific_base_ops` for
framework-only recipes, and artifact-keyed `model_specific_base_ops` when one
concrete checkpoint needs an op (e.g. the static-FP8 Qwen3 artifact opting
into `compute_scale`).

## Skipped Cases

`collector/skips.py` is the one place that keeps cases out of a collection.
Each entry names what it skips (`backend`, `op`, `sm`, `model`, `dtype` or any
case field), and the comment above it says why:

```python
# SGLang: the collector times fp8_block GEMMs through DeepGEMM, which has no
# SM89 or SM120 kernel, so every case would fail.
{"backend": "sglang", "op": "gemm", "dtype": "fp8_block", "sm": [89, 120]},
# TRT-LLM cannot serve MiMo-V2-Flash: it registers no model class for it.
{"backend": "trtllm", "model": "MiMoV2FlashForCausalLM"},
```

Add an entry only when the cases would be mismeasured, would hang, would fail
as a whole block, or come from a model the backend cannot serve; edit or
delete it when the reason stops being true. Everything else runs, and its
failures are recorded and compared with the previous run.

An op plan (`OP_PLANS` in `collector/op_plans.py`) applies the list when it
expands, and `collect.py --plan-only --sm N` reports what it left out under
which entry. For other ops, `collect.py` applies it around the getter and to
the generated cases before queueing (typed fields such as `moe_type`,
`gemm_type`, `dtype`, `kv_cache_dtype` and the FP8 flags). A `model` entry
drops that model's rows before cases are merged, so a shape another model
also needs is still collected.

## Running Subsets (healing)

Subset selection is a runtime concern, never persisted to YAML:

```bash
# one model on one GPU type
python3 collect.py --backend sglang --model-path <model> --gpu b200_sxm

# substring filter over generated cases (repeatable, OR semantics)
python3 collect.py --backend trtllm --ops moe --case-filter "tp=4"

# finish an interrupted run
python3 collect.py --backend trtllm --resume
```

## Adding New Coverage

Add one architecture file for a new architecture, or add a model path alias to
an existing architecture file when the model uses the same case plan. Add or
edit one `base_ops/<op>.yaml` file when common op sweeps change. Add a new op
collector only when no existing op can generate the needed points. See
`.claude/rules/collector/case_authoring.md` for the full rules.


## Optional long-context points

Model attention and MLA/DSA/MSA module rows can declare additive numerical
points, consumed only with `--include-long-context` (default off). This extends
model-correlated coverage without increasing every geometry's base grid. Existing
normal points, including long-context anchors, remain enabled when the flag is
off. MoE token, KDA width and NCCL payload additions are independent of this flag.

| Field | Value and meaning |
|---|---|
| `model_case_values.attention[].long_context_cases.context` | Exact `[batch, full_lookup_sequence, 0]` points |
| `model_case_values.attention[].long_context_cases.generation` | Exact `[batch, past_length, 0]` points |
| `model_case_values.mla_module[].long_context_cases.context` | Exact `[batch, new_query_length, cached_prefix_length]` points |
| `model_case_values.mla_module[].long_context_cases.generation` | Exact `[batch, past_length, 0]` points |
| `long_context_tensor_parallel_sizes` | Optional real TP shards for these added module points only, when a legacy normal row has no TP declaration; otherwise inherits `tensor_parallel_sizes` |

The same `long_context_cases` structure is supported on an existing
`framework_specific_model_case_values.mla_module` row. Such a model geometry
contributes to the shared case plan, with backend availability still owned by
`collector/skips.py`. Numeric tuples do not express predicates or independent
sweep axes. For example, on an existing sparse module model row:

```yaml
long_context_cases:
  context:
    - [1, 32768, 917504]
  generation:
    - [32, 1048575, 0]
```

`past_length` is the decode past-KV length, excluding the new token.
`cached_prefix_length` counts already cached prefill tokens, excluding the new
query. Context validates `query + prefix <= max_position_embeddings`; decode validates
`past + 1 <= max_position_embeddings`, including nested text configs. Invalid
metadata raises; missing model configs or context limits never become guessed
coverage. Dense context uses the full lookup sequence because that is the current lookup
contract, even for a short cached query. Shared native indexing constraints
still apply; device capacity is refused at runtime, not used to vary the plan.

Module additions use their model's declared TP shards and native projection
GEMM. Resolve projection precision for the selected artifact before choosing a
canonical alias owner; a full plan unions the projection lanes of all declared
artifacts. MiniMax NVFP4's MXFP8 attention projections use the SDK's `fp8_block`
approximation while retaining the canonical MiniMax measurement owner.
Their workload points are unioned before deduplication. Runtime-sensitive
cases share an owner only when native attention semantics agree, not merely
because the models have the same head count. Unknown quantization remains part
of that comparison; new incompatible collisions raise before collection. Existing normal points and their ownership
remain unchanged. Plan rows retain exact manifests and existing IDs; no perf
schema, filename or consumer key changes are introduced.


### Choosing coverage

Use three evidence sources for every extension; record exact source revisions
and identify what remains unmeasured:

| Input | Evidence | What it establishes |
|---|---|---|
| Model architecture/config | Packaged checkpoint config and existing `model_case_values` | Operator family, correlated geometry, legal TP shards and context validity limit |
| Effective serving recipe | Runtime logs and launch configuration, cross-checked against pinned framework source | Local/global token mapping, TP/DP/EP, precision, selected backend and speculative width |
| Target workload | Profiling and warmup request traces; per-rank forwards when available | Context demand and observed joint batch/query/cached-prefix shapes |

AgentX supplies serving and workload evidence for the current declarations; the
collector has no live dependency on AgentX. Label each added point as observed,
configuration-derived or chosen interpolation support. Request input plus actual
output establishes context demand, not physical GPU batches. Launch allowances
are bounds, not observed forwards. Use per-rank records to claim observed joint
shapes; otherwise state that the tuples are coverage choices. Context limits
validate these choices rather than generate a Cartesian sweep. Do not substitute
another model or checkpoint when its identity cannot be resolved.

Campaign inventories, model-by-model decisions, benchmark results and review
history are retained with the local evidence outside this PR. They do not change
the declaration or serving-parity requirements above.

### Runtime equivalence and checkpoint capacity

Keep the normal planner's representative when native attention semantics agree.
For GLM attention modules, tokenizer and MLP metadata are outside the measured
module; explicit native qk/v dimensions override redundant `head_dim`. An
expert-only quantization recipe is ignored for attention equivalence only when
it excludes every self-attention module or explicitly quantizes only MLP modules.
Unknown quantization and changed attention/indexer/RoPE dimensions remain
significant; incompatible persisted-key collisions raise.

Attention equivalence does not establish equal startup cost or KV capacity.
SGLang initializes two full dummy model layers by default, so BF16 and NVFP4
checkpoints can have different resident weights and capacity refusals. Targeted
runs retain their own checkpoint. Use `--model-path` when checkpoint-specific
capacity is required, and keep its initialization and group-cost evidence
separate. A refusal remains a coverage gap, not a zero-cost measurement.

### Deferred Kimi-K3 MLA extension

Kimi-K3's `mla_use_nope: true` selects a different native contract. At pinned
vLLM commit [`ced6857afa0ea7b2e3f0846a62e1394e90f15607`](https://github.com/vllm-project/vllm/commit/ced6857afa0ea7b2e3f0846a62e1394e90f15607),
[`KimiDecoderLayer`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/kimi_k3/nvidia/model.py#L930)
constructs native [`MultiHeadLatentAttention`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/kimi_k3/nvidia/mla.py#L136)
with NoPE, fused cache insertion and the configured output gate. The generic
DeepSeek module is not interchangeable with it.

Before adding optional Kimi MLA points, preserve native registration/cache
binding, absorbed weights, metadata, FP8 policy and the real forward contract.
Align the output-gate/projection timing boundary with the consumer. Qualify short
and long prefill/decode, BF16/FP8 cache, local heads including 12/24, graph/eager
behavior and numerical/serving parity.

No optional Kimi MLA points are declared yet. Normal coverage is preserved
without certifying native parity for existing generic rows. SGLang's standalone
MLA route and vLLM's dense fallback are not qualified by this module extension.
Keep the required native extension deferred/unmeasured rather than reporting
zero GPU-hours or complete 1M coverage. Independent KDA widths remain enabled.
