#!/bin/bash
# sglang 0.5.21 gates (2026-10-01): captures re-taken in the v0.5.21 image as opcov_<cap>@0.5.21.json.
HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd ${AIS_PROBE_WORKSPACE:-.}
export AIS_PROBE_WORKSPACE=${AIS_PROBE_WORKSPACE:-.} AIS_SM=${AIS_SM:-$(python3 -c "import yaml;print('sm%d'%yaml.safe_load(open('$HARNESS/targets.yaml'))['platform']['sm'])")}
PD=$HARNESS/components/path_diff.py
OUT=$HARNESS/results/pathdiff/$AIS_SM/sglang-0.5.21
mkdir -p $OUT
run() { local cap=$1 name=$2 repo=$3 kv=$4 hint=$5 kvarg=(); [ -n "$kv" ] && kvarg=(--kv-dtype "$kv")
  [ -f facts/pathdiff/opcov_$cap@0.5.21.json ] || { printf '%-36s MISSING CAPTURE\n' "$name"; return; }
  python3 $PD --diff --capture-file facts/pathdiff/opcov_$cap@0.5.21.json --repo "$repo" --framework sglang --version 0.5.21 "${kvarg[@]}" --op-hint "$hint" --save-verdict $OUT/$name.json 2>/dev/null | python3 -c "
import sys,json;t=sys.stdin.read()
if '{' not in t: print('%-36s'%'$name','NO SERVING RECORD', t.strip()[:80]); sys.exit()
d=json.loads(t[t.index('{'):]); print('%-36s'%'$name', d['verdict'].upper().ljust(9), 'only_col=',d['collector_only_signal'], 'drift=',{k:v['collector_only_kernels'][:3] for k,v in (d['kernel_drift'] or {}).items()}, '| col=',d['collector_backends'],'| srv=',d['serving_record']['id'],'| err=',(d.get('capture_run_error') or '')[:80])"; }
ATT='attn|attention|flash|fwd|triton'
DSA='mla|dsa|nsa|attn|indexer|flash|mqa|sparse|gemm|deepgemm|quant|topk'
MSA='attn|sparse|msa|topk|index|score|merge|block'
GEMM='gemm|linear|proj|deepgemm|cutlass|cublas|quant'
MOE='moe|expert|grouped|fused_moe|deepgemm|topk|marlin|routing|ep_'
GDN='gdn|delta|conv1d|linear|mamba|gated|recurr'
run sgl_attn_ctx      attn_ctx_Llama-3.1-8B          meta-llama/Meta-Llama-3.1-8B auto "$ATT"
run sgl_attn_gen      attn_gen_Llama-3.1-8B          meta-llama/Meta-Llama-3.1-8B auto "$ATT"
# Gemma-4 head_dim 512 context: the profile's sglang_backends map pins triton
# on SM90 (collect_attn's default table would say fa3); capture = op_smoke
# --case-index 168 (s/caps/launch_sgl_gemma4_hd512.sh), serving = TritonAttnBackend record
run sgl_attn_ctx_gemma4_hd512 attn_ctx_hd512_gemma-4-26B-A4B google/gemma-4-26B-A4B auto "$ATT"
# single-cell (seq 4096 = the probe's isl) capture: the whole-sweep capture unions
# short cells that run dense FA3 with the sparse path (review 2026-09-25)
run sgl_dsa_ctx_s4096 dsa_ctx_bf16_DeepSeek-V3.2     deepseek-ai/DeepSeek-V3.2   auto "$DSA"
run sgl_dsa_gen       dsa_gen_bf16_DeepSeek-V3.2     deepseek-ai/DeepSeek-V3.2   auto "$DSA"
# fp8-KV (sglang kv variant fp8_e4m3, framework-mode records 2026-09-24).
# Context is length-conditional in sglang (dense FA3 + NO indexer below the
# dense threshold; indexer + fp8 sparse above), so the ctx capture is ONE cell
# at the record's isl (s/caps/sgl_dsa_ctx_fp8_s4096.py) — a whole-sweep
# capture unions both regimes and can never match a single record.
run sgl_dsa_ctx_fp8_s4096 dsa_ctx_fp8_DeepSeek-V3.2  deepseek-ai/DeepSeek-V3.2   fp8  "$DSA"
run sgl_dsa_gen_fp8   dsa_gen_fp8_DeepSeek-V3.2      deepseek-ai/DeepSeek-V3.2   fp8  "$DSA"
# short-isl regime: s=512 prefix-0 cell (AIC_DSA_CONTEXT_PREFIX_LENS=0) vs the
# isl-512 fp8-KV probe (not a plan record -> --serving-raw)
# NOT GRADED at 0.5.21: the serving side of this cell is a 0.5.16 one-off raw (facts/reprobe_nopc/dsv32_sgl_fp8_isl512_v2.json);
# an isl-512 fp8-KV probe on 0.5.21 is required before this short-isl verdict means anything.
# python3 $PD --diff --capture-file facts/pathdiff/opcov_sgl_dsa_ctx_fp8_s512_p0@0.5.21.json --repo deepseek-ai/DeepSeek-V3.2 --framework sglang --version 0.5.21 --kv-dtype fp8 --isl 512 --serving-raw facts/reprobe_nopc/dsv32_sgl_fp8_isl512_v2.json --op-hint "$DSA" --save-verdict $OUT/dsa_ctx_fp8_s512_DeepSeek-V3.2.json 2>/dev/null | python3 -c "
# import sys,json;t=sys.stdin.read();d=json.loads(t[t.index('{'):]);print('%-36s'%'dsa_ctx_fp8_s512_DeepSeek-V3.2', d['verdict'].upper().ljust(9), 'only_col=',d['collector_only_signal'], 'drift=',d['kernel_drift'])"
run sgl_msa_ctx       msa_ctx_MiniMax-M3             MiniMaxAI/MiniMax-M3        auto "$MSA"
run sgl_msa_gen       msa_gen_MiniMax-M3             MiniMaxAI/MiniMax-M3        auto "$MSA"
run sgl_mla_ctx       mla_ctx_DeepSeek-V3            deepseek-ai/DeepSeek-V3     auto "$DSA"
run sgl_mla_gen       mla_gen_DeepSeek-V3            deepseek-ai/DeepSeek-V3     auto "$DSA"
run sgl_gemm_fp8block gemm_fp8block_DeepSeek-V3      deepseek-ai/DeepSeek-V3     auto "$GEMM"
run sgl_moe_fp8block  moe_fp8block_DeepSeek-V3       deepseek-ai/DeepSeek-V3     auto "$MOE"
run sgl_gdn_ctx       gdn_ctx_Qwen3.5-0.8B           Qwen/Qwen3.5-0.8B           auto "$GDN"
run sgl_gdn_gen       gdn_gen_Qwen3.5-0.8B           Qwen/Qwen3.5-0.8B           auto "$GDN"

# --- 2026-10-01 gate backfill (collector pin -> 0.5.21): the 15 registry families
# that had no gate on sglang. Serving records: the 0.5.21 plan (isl 4096, kv auto).
DSV4='dsv4|dsa|csa|hca|compress|indexer|mqa|sparse|flash|attn|mla|topk|gemm|deepgemm|quant'
KDA='kda|delta|conv1d|linear|attn_res|gated|recurr'
ENC='vision|vit|patch|encoder|flash|attn'
MLA='mla|attn|flash|bmm|proj|indexer|rope|concat|gemm|gemvx|cutlass|deepgemm|quant'
GLM5='mla|dsa|nsa|attn|indexer|flash|mqa|sparse|gemm|deepgemm|quant|topk'
M=sgl-project/DeepSeek-V4-Flash-FP8
run sgl_dsv4_csa_ctx          dsv4_csa_ctx_DeepSeek-V4-Flash-FP8          $M auto "$DSV4"
run sgl_dsv4_csa_gen          dsv4_csa_gen_DeepSeek-V4-Flash-FP8          $M auto "$DSV4"
run sgl_dsv4_hca_ctx          dsv4_hca_ctx_DeepSeek-V4-Flash-FP8          $M auto "$DSV4"
run sgl_dsv4_hca_gen          dsv4_hca_gen_DeepSeek-V4-Flash-FP8          $M auto "$DSV4"
run sgl_dsv4_csa_attn         dsv4_csa_attn_DeepSeek-V4-Flash-FP8         $M auto "$DSV4"
run sgl_dsv4_hca_attn         dsv4_hca_attn_DeepSeek-V4-Flash-FP8         $M auto "$DSV4"
run sgl_dsv4_paged_mqa_logits dsv4_paged_mqa_logits_DeepSeek-V4-Flash-FP8 $M auto "$DSV4"
run sgl_glm5_dsa_attn         glm5_dsa_attn_GLM-5                         zai-org/GLM-5 auto "$GLM5"
run sgl_glm5_mqa_logits       glm5_mqa_logits_GLM-5                       zai-org/GLM-5 auto "$GLM5"
run sgl_glm5_topk             glm5_topk_GLM-5                             zai-org/GLM-5 auto "$GLM5"
run sgl_kda_ctx               kda_ctx_Kimi-K3                             moonshotai/Kimi-K3 auto "$KDA"
run sgl_kda_gen               kda_gen_Kimi-K3                             moonshotai/Kimi-K3 auto "$KDA"
run sgl_mla_bmm               mla_bmm_gen_DeepSeek-V3                     deepseek-ai/DeepSeek-V3 auto "$MLA"
# Qwen3-VL on sglang: the plan probe is text-only (prefill_api shows only
# HybridAttnBackend.forward; no vision tower ran), so the record has no attention-role
# evidence for the encoder (vllm proves this family from its profile_run, which feeds
# dummy multimodal input). Explained deviation until the sglang probe exercises the
# vision tower — written next to the facts, not into the gate directory.
OUT_EXPLAINED=facts/pathdiff/explained_sgl_0521; mkdir -p $OUT_EXPLAINED
OUT=$OUT_EXPLAINED run sgl_encoder_attn_qwen3vl  encoder_attn_Qwen3-VL-2B   Qwen/Qwen3-VL-2B-Instruct auto "$ENC"
