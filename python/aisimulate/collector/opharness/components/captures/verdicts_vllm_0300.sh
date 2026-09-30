#!/bin/bash
# SM-parametric (2026-09-30): AIS_SM selects the taxonomy AND the per-SM verdict dir;
# paths resolve from this script's location, not from a checkout inside the workspace.
HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd ${AIS_PROBE_WORKSPACE:-.}
export AIS_PROBE_WORKSPACE=${AIS_PROBE_WORKSPACE:-.} AIS_SM=${AIS_SM:-sm90}
PD=$HARNESS/components/path_diff.py
OUT=$HARNESS/results/pathdiff/$AIS_SM/vllm-0.30.0; mkdir -p $OUT
OUT_EXPLAINED=facts/pathdiff/explained_0300; mkdir -p $OUT_EXPLAINED
# Per-SM platform floors: `FLOOR_SM=<sm> FLOOR_NOTE="<why>" run ...` declares that on that SM the gate has no serving
# instance by framework fact (recorded as a platform-floor verdict file, excluded from path_aligned, still counted as
# declared coverage by workflow_check.declared_gates). Other SMs grade the gate normally.
run() { local cap=$1 name=$2 repo=$3 kv=$4 hint=$5 kvarg=(); [ -n "$kv" ] && kvarg=(--kv-dtype "$kv")
  if [ -n "$FLOOR_SM" ] && [ "$FLOOR_SM" = "$AIS_SM" ]; then
    python3 - "$OUT/$name.json" "$name" "$repo" "$AIS_SM" "$FLOOR_NOTE" <<'PY'
import json,sys,time; out,name,repo,sm,note=sys.argv[1:]
json.dump({"verdict":"platform-floor","gate_name":name,"repo":repo,"framework":"vllm","version":"0.30.0","sm":sm,
           "note":note,"graded_at":time.strftime("%Y-%m-%dT%H:%M:%S")}, open(out,"w"), indent=1)
PY
    printf '%-40s PLATFORM-FLOOR (%s): %s\n' "$name" "$AIS_SM" "$FLOOR_NOTE"; return; fi
  [ -n "$SERVING_RAW" ] && kvarg+=(--serving-raw "$SERVING_RAW")
  [ -f facts/pathdiff/opcov_$cap@0.30.0.json ] || { printf '%-40s MISSING CAPTURE\n' "$name"; return; }
  python3 $PD --diff --capture-file facts/pathdiff/opcov_$cap@0.30.0.json --repo "$repo" --framework vllm --version 0.30.0 "${kvarg[@]}" --op-hint "$hint" --save-verdict $OUT/$name.json 2>/dev/null | python3 -c "
import sys,json;t=sys.stdin.read()
if '{' not in t: print('%-40s'%'$name','NO SERVING RECORD', t.strip()[:80]); sys.exit()
d=json.loads(t[t.index('{'):])
print('%-40s'%'$name', d['verdict'].upper().ljust(9), 'missing=',d.get('missing_roles'), 'only_col=',d.get('only_col'), 'drift=',{k:v['collector_only_kernels'][:2] for k,v in (d.get('kernel_drift') or {}).items()})"; }
GEMM='gemm|linear|proj|nvjet|gemvx|cublas|deepgemm|scaled_mm|quant|cutlass'
MOE='moe|expert|grouped|fused_moe|deepgemm|topk|marlin|routing'
MLA='mla|attn|flash|bmm|proj|indexer|rope|concat|gemm|gemvx|cutlass|deepgemm|quant'
GDN='gdn|delta|conv1d|linear|mamba|gated|recurrent'
run gemm_fp8block      gemm_fp8block_DeepSeek-V3.2      deepseek-ai/DeepSeek-V3.2 fp8  "$GEMM"
run gemm_bf16          gemm_bf16_Llama-3.1-8B           meta-llama/Meta-Llama-3.1-8B auto "$GEMM"
# gemm_fp8 (dynamic per-token activation row) vs a ModelOpt STATIC per-tensor checkpoint: on CC>=100 vLLM routes
# the static scheme to FlashInferFP8ScaledMMLinearKernel and the dynamic one to Cutlass, so the SDK's derived
# fp8_static rests on a different GEMM kernel there. Owner decision 2026-09-30 (tianhaox): governed approximation,
# not a collected row — see collector/evidence_exceptions.yaml (fp8_static on SM100+/SM120) -> explained deviation.
OUT=$OUT_EXPLAINED run gemm_fp8 gemm_fp8_Llama-3.1-70B-FP8 nvidia/Llama-3.1-70B-Instruct-FP8 auto "$GEMM"
run moe_fp8block_dsv3  moe_fp8block_DeepSeek-V3.2       deepseek-ai/DeepSeek-V3.2 fp8  "$MOE"
run moe_bf16_gemma4    moe_bf16_gemma4-26B-A4B          google/gemma-4-26B-A4B auto    "$MOE"
run moe_mxfp4_gptoss   moe_mxfp4_gpt-oss-120b           openai/gpt-oss-120b auto       "$MOE"
run mla_ctx_bf16       mla_ctx_bf16_DeepSeek-V3         deepseek-ai/DeepSeek-V3 auto   "$MLA"
OUT=$OUT_EXPLAINED run mla_gen_bf16 mla_gen_bf16_DeepSeek-V3 deepseek-ai/DeepSeek-V3 auto "$MLA"
FLOOR_SM=sm120 FLOOR_NOTE="CC 12.0: the only dense-MLA backend TRITON_MLA needs 102400 B smem for fp8-KV decode, the card allows 101376 B (findings sm120_triton_mla_fp8kv_smem); no fp8-KV MLA serving record can exist" run mla_ctx_fp8        mla_ctx_fp8_DeepSeek-R1          deepseek-ai/DeepSeek-R1 fp8    "$MLA"
FLOOR_SM=sm120 FLOOR_NOTE="CC 12.0: the only dense-MLA backend TRITON_MLA needs 102400 B smem for fp8-KV decode, the card allows 101376 B (findings sm120_triton_mla_fp8kv_smem); no fp8-KV MLA serving record can exist" run mla_gen_fp8        mla_gen_fp8_DeepSeek-R1          deepseek-ai/DeepSeek-R1 fp8    "$MLA"
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
GEMM_M1='gemm|linear|proj|nvjet|gemvx|cublas|cutlass|skinny'
run attn_ctx               attn_ctx_Llama-3.1-8B                   meta-llama/Meta-Llama-3.1-8B auto "$ATTN"
run attn_gen               attn_gen_Llama-3.1-8B                   meta-llama/Meta-Llama-3.1-8B auto "$ATTN"
run encoder_attn_qwen3vl   encoder_attn_Qwen3-VL-2B                Qwen/Qwen3-VL-2B-Instruct auto "$ENC"
run dsa_ctx_bf16           dsa_ctx_bf16_DeepSeek-V3.2              deepseek-ai/DeepSeek-V3.2 auto "$DSA"
run dsa_ctx_bf16_s4096     dsa_ctx_bf16_s4096_DeepSeek-V3.2        deepseek-ai/DeepSeek-V3.2 auto "$DSA"
# isl 512 is below the DSA sparse threshold (dense FA3, no indexer) — graded against a dedicated isl-512 serving
# raw of the same record (probe_vllm --isl 512), as on 0.29; the isl-4096 record would compare sparse vs dense.
SERVING_RAW=facts/reprobe_nopc/04cea4da0eff_isl512.json run dsa_ctx_fp8 dsa_ctx_fp8_s512_DeepSeek-V3.2 deepseek-ai/DeepSeek-V3.2 fp8 "$DSA"
run dsa_ctx_fp8_s4096      dsa_ctx_fp8_s4096_DeepSeek-V3.2         deepseek-ai/DeepSeek-V3.2 fp8  "$DSA"
run dsa_gen_fp8            dsa_gen_fp8_DeepSeek-V3.2               deepseek-ai/DeepSeek-V3.2 fp8  "$DSA"
run gemm_bf16_m1           gemm_bf16_m1_Llama-3.1-8B               meta-llama/Meta-Llama-3.1-8B auto "$GEMM_M1"
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
