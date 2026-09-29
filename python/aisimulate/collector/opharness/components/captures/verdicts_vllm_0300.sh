#!/bin/bash
cd ${AIS_PROBE_WORKSPACE:-.}
export AIS_PROBE_WORKSPACE=${AIS_PROBE_WORKSPACE:-.} AIS_SM=sm90
PD=ais/python/aisimulate/collector/opharness/components/path_diff.py
OUT=ais/python/aisimulate/collector/opharness/results/pathdiff/sm90/vllm-0.30.0
OUT_EXPLAINED=facts/pathdiff/explained_0300; mkdir -p $OUT_EXPLAINED
run() { local cap=$1 name=$2 repo=$3 kv=$4 hint=$5 kvarg=(); [ -n "$kv" ] && kvarg=(--kv-dtype "$kv")
  [ -n "$SERVING_RAW" ] && kvarg+=(--serving-raw "$SERVING_RAW")
  [ -f facts/pathdiff/opcov_$cap@0.30.0.json ] || { printf '%-40s MISSING CAPTURE\n' "$name"; return; }
  python3 $PD --diff --capture-file facts/pathdiff/opcov_$cap@0.30.0.json --repo "$repo" --framework vllm --version 0.30.0 "${kvarg[@]}" --op-hint "$hint" --save-verdict $OUT/$name.json 2>/dev/null | python3 -c "
import sys,json;t=sys.stdin.read()
if '{' not in t: print('%-40s'%'$name','NO SERVING RECORD', t.strip()[:80]); sys.exit()
d=json.loads(t[t.index('{'):])
print('%-40s'%'$name', d['verdict'].upper().ljust(9), 'missing=',d.get('missing_roles'), 'only_col=',d.get('only_col'), 'drift=',{k:v['collector_only_kernels'][:2] for k,v in (d.get('kernel_drift') or {}).items()})"; }
GEMM='gemm|linear|proj|nvjet|cublas|deepgemm|scaled_mm|quant|cutlass'
MOE='moe|expert|grouped|fused_moe|deepgemm|topk|marlin|routing'
MLA='mla|attn|flash|bmm|proj|indexer|rope|concat|gemm|deepgemm|quant'
GDN='gdn|delta|conv1d|linear|mamba|gated|recurrent'
run gemm_fp8block      gemm_fp8block_DeepSeek-V3.2      deepseek-ai/DeepSeek-V3.2 fp8  "$GEMM"
run gemm_bf16          gemm_bf16_Llama-3.1-8B           meta-llama/Meta-Llama-3.1-8B "" "$GEMM"
run gemm_fp8           gemm_fp8_Llama-3.1-70B-FP8       nvidia/Llama-3.1-70B-Instruct-FP8 auto "$GEMM"
run moe_fp8block_dsv3  moe_fp8block_DeepSeek-V3.2       deepseek-ai/DeepSeek-V3.2 fp8  "$MOE"
run moe_bf16_gemma4    moe_bf16_gemma4-26B-A4B          google/gemma-4-26B-A4B ""      "$MOE"
run moe_mxfp4_gptoss   moe_mxfp4_gpt-oss-120b           openai/gpt-oss-120b ""         "$MOE"
run mla_ctx_bf16       mla_ctx_bf16_DeepSeek-V3         deepseek-ai/DeepSeek-V3 auto   "$MLA"
OUT=$OUT_EXPLAINED run mla_gen_bf16 mla_gen_bf16_DeepSeek-V3 deepseek-ai/DeepSeek-V3 auto "$MLA"
run mla_ctx_fp8        mla_ctx_fp8_DeepSeek-R1          deepseek-ai/DeepSeek-R1 fp8    "$MLA"
run mla_gen_fp8        mla_gen_fp8_DeepSeek-R1          deepseek-ai/DeepSeek-R1 fp8    "$MLA"
run mla_bmm            mla_bmm_gen_DeepSeek-V3          deepseek-ai/DeepSeek-V3 auto   "$MLA"
run gdn_ctx            gdn_ctx_Qwen3.5-0.8B             Qwen/Qwen3.5-0.8B auto         "$GDN"
run gdn_gen            gdn_gen_Qwen3.5-0.8B             Qwen/Qwen3.5-0.8B auto         "$GDN"
OUT=$OUT_EXPLAINED run compute_scale compute_scale_Llama-3.1-70B-FP8 nvidia/Llama-3.1-70B-Instruct-FP8 auto 'quant|scale|per_token|fp8'
DSV4='dsv4|dsa|csa|hca|compress|indexer|mqa|sparse|flash|attn|mla|topk|gemm|deepgemm|quant'
MHC='mhc|hc_|tilelang|prenorm|norm'
KDA='kda|delta|conv1d|linear|attn_res|gated|recurr'
M=sgl-project/DeepSeek-V4-Flash-FP8
run dsv4_csa_ctx           dsv4_csa_ctx_DeepSeek-V4-Flash-FP8      $M fp8 "$DSV4"
run dsv4_csa_gen           dsv4_csa_gen_DeepSeek-V4-Flash-FP8      $M fp8 "$DSV4"
run dsv4_hca_ctx           dsv4_hca_ctx_DeepSeek-V4-Flash-FP8      $M fp8 "$DSV4"
run dsv4_hca_gen           dsv4_hca_gen_DeepSeek-V4-Flash-FP8      $M fp8 "$DSV4"
run dsv4_hca_attn          dsv4_hca_attn_DeepSeek-V4-Flash-FP8     $M fp8 "$DSV4"
run dsv4_paged_mqa_logits  dsv4_paged_mqa_logits_DeepSeek-V4-Flash-FP8 $M fp8 "$DSV4"
ATTN='attn|attention|flash|fwd|unified'
DSA='mla|dsa|attn|indexer|flash|mqa|sparse|gemm|proj|deepgemm|quant'
ENC='vision|vit|patch|encoder|flash|attn'
GEMM_M1='gemm|linear|proj|nvjet|cublas|cutlass|skinny'
run attn_ctx               attn_ctx_Llama-3.1-8B                   meta-llama/Meta-Llama-3.1-8B "" "$ATTN"
run attn_gen               attn_gen_Llama-3.1-8B                   meta-llama/Meta-Llama-3.1-8B "" "$ATTN"
run encoder_attn_qwen3vl   encoder_attn_Qwen3-VL-2B                Qwen/Qwen3-VL-2B-Instruct auto "$ENC"
run dsa_ctx_bf16           dsa_ctx_bf16_DeepSeek-V3.2              deepseek-ai/DeepSeek-V3.2 auto "$DSA"
run dsa_ctx_bf16_s4096     dsa_ctx_bf16_s4096_DeepSeek-V3.2        deepseek-ai/DeepSeek-V3.2 auto "$DSA"
# isl 512 is below the DSA sparse threshold (dense FA3, no indexer) — graded against a dedicated isl-512 serving
# raw of the same record (probe_vllm --isl 512), as on 0.29; the isl-4096 record would compare sparse vs dense.
SERVING_RAW=facts/reprobe_nopc/04cea4da0eff_isl512.json run dsa_ctx_fp8 dsa_ctx_fp8_s512_DeepSeek-V3.2 deepseek-ai/DeepSeek-V3.2 fp8 "$DSA"
run dsa_ctx_fp8_s4096      dsa_ctx_fp8_s4096_DeepSeek-V3.2         deepseek-ai/DeepSeek-V3.2 fp8  "$DSA"
run dsa_gen_fp8            dsa_gen_fp8_DeepSeek-V3.2               deepseek-ai/DeepSeek-V3.2 fp8  "$DSA"
run gemm_bf16_m1           gemm_bf16_m1_Llama-3.1-8B               meta-llama/Meta-Llama-3.1-8B "" "$GEMM_M1"
run gemm_fp8block_m1       gemm_fp8block_m1_DeepSeek-V3.2          deepseek-ai/DeepSeek-V3.2 fp8  "$GEMM_M1"
run kda_gen                kda_gen_Kimi-K3                         moonshotai/Kimi-K3 auto "$KDA"
MHC='mhc|hyper|hc_|residual|reduce|store|norm|rms|manifold'
# mhc: EXPLAINED, not a gate — the collector measures the no-norm mhc_pre variant by producer/consumer contract
# (SDK bills attn_norm separately; collect_mhc_module._mhc_pre), serving 0.30 runs the with_norm big_fuse variant.
OUT=$OUT_EXPLAINED run mhc_pre_post mhc_DeepSeek-V4-Flash-FP8 sgl-project/DeepSeek-V4-Flash-FP8 fp8 "$MHC"
MSA='attn|sparse|msa|topk|index|score|merge'
run msa_ctx                msa_ctx_MiniMax-M3                      MiniMaxAI/MiniMax-M3 auto "$MSA"
run msa_gen                msa_gen_MiniMax-M3                      MiniMaxAI/MiniMax-M3 auto "$MSA"
run kda_ctx                kda_ctx_Kimi-K3                         moonshotai/Kimi-K3 auto "$KDA"
