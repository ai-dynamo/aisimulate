# opharness TODO (out of scope for the current work — recorded, not scheduled)

Owner decisions 2026-09-25 (tianhaox): the items below are NOT part of the
op-probe harness work. They stay here so the matrix cells they explain are
traceable; do not re-raise them in status reports.

## SDK modeling gaps (matrix cells "generator rejects")

| Architecture | Repos in the roster | Structure (from the Hub configs) | Closest existing SDK family |
|---|---|---|---|
| `Glm5NextForConditionalGeneration` | zai-org/GLM-5.3-Flash, -BF16, nvidia/GLM-5.3-Flash-NVFP4 | GLM-5.3 DSA (kv_lora 512, indexer topk 2048, mHC) with 34/45 layers swapped to KDA linear attention (`linear_attn_config`, 0-based lists), 288-expert MoE, MTP | DEEPSEEKV32 / GLM-5.3 + KIMIK3 (KDA layers) |
| `Qwen4ExpForConditionalGeneration` | Qwen/Qwen3.8-Flash-Next, -FP8, nvidia/-NVFP4 | Qwen3.5 GDN hybrid (interval 4, 512 experts x 640, MTP) plus n-gram embedding (`ngram_*`, `ple_*`) and `indexer_*` fields | QWEN35 (extension) |
| `InklingForConditionalGeneration` | baerquant/Inkling, Inkling-Small, baerquants/*-NVFP4 | 66-layer GQA MoE (256 experts top-6, 2 shared) with SWA layers (`local_layer_ids`, `swa_*`), short conv, hmlp vision encoder, audio | none (new) |
| `MiMoV2ForCausalLM` | XiaomiMiMo/MiMo-V2.6-Pro-RL, -Flash-RL | MiMo-V2-Flash lineage: SWA 128 + full attention hybrid, 384-expert MoE, head_dim 192 / v 128, vision + audio | HYBRIDMOE (MiMoV2FlashForCausalLM) |

Framework side (pinned images): vllm 0.29.0 loads Qwen4Exp, Inkling and
MiMoV2 natively; no pinned framework knows Glm5Next. Identity records for
these architectures need both the SDK model and, for Glm5Next, a framework
pin bump — neither is scheduled.

## Harness components not built

- `e2e_align` (onboard_model step 10): prediction vs live measurement on the
  golden deployment. Needs real weights and a probe GPU that matches an SDK
  `systems/data` entry; the current probe box (H20, proxied as h200_sxm)
  satisfies neither. The grader is implemented (`components/e2e_align.py`),
  the campaign is not.
