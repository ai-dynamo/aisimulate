#!/bin/bash
cd "${AIS_PROBE_WORKSPACE:?set AIS_PROBE_WORKSPACE to the probe workspace}"
export AIS_SM=${AIS_SM:-sm90}
PD=ais/python/aisimulate/collector/opharness/components/path_diff.py
OUT=ais/python/aisimulate/collector/opharness/results/pathdiff/sm90/sglang-0.5.16
mkdir -p $OUT
run() { local cap=$1 name=$2 repo=$3 kv=$4 hint=$5 kvarg=(); [ -n "$kv" ] && kvarg=(--kv-dtype "$kv")
  [ -f facts/pathdiff/opcov_$cap.json ] || { printf '%-36s MISSING CAPTURE\n' "$name"; return; }
  python3 $PD --diff --capture-file facts/pathdiff/opcov_$cap.json --repo "$repo" --framework sglang --version 0.5.16 "${kvarg[@]}" --op-hint "$hint" --save-verdict $OUT/$name.json 2>/dev/null | python3 -c "
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
run sgl_dsa_ctx       dsa_ctx_bf16_DeepSeek-V3.2     deepseek-ai/DeepSeek-V3.2   auto "$DSA"
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
python3 $PD --diff --capture-file facts/pathdiff/opcov_sgl_dsa_ctx_fp8_s512_p0.json --repo deepseek-ai/DeepSeek-V3.2 --framework sglang --version 0.5.16 --kv-dtype fp8 --isl 512 --serving-raw facts/reprobe_nopc/dsv32_sgl_fp8_isl512_v2.json --op-hint "$DSA" --save-verdict $OUT/dsa_ctx_fp8_s512_DeepSeek-V3.2.json 2>/dev/null | python3 -c "
import sys,json;t=sys.stdin.read();d=json.loads(t[t.index('{'):]);print('%-36s'%'dsa_ctx_fp8_s512_DeepSeek-V3.2', d['verdict'].upper().ljust(9), 'only_col=',d['collector_only_signal'], 'drift=',d['kernel_drift'])"
run sgl_msa_ctx       msa_ctx_MiniMax-M3             MiniMaxAI/MiniMax-M3        auto "$MSA"
run sgl_msa_gen       msa_gen_MiniMax-M3             MiniMaxAI/MiniMax-M3        auto "$MSA"
run sgl_mla_ctx       mla_ctx_DeepSeek-V3            deepseek-ai/DeepSeek-V3     auto "$DSA"
run sgl_mla_gen       mla_gen_DeepSeek-V3            deepseek-ai/DeepSeek-V3     auto "$DSA"
run sgl_gemm_fp8block gemm_fp8block_DeepSeek-V3      deepseek-ai/DeepSeek-V3     auto "$GEMM"
run sgl_moe_fp8block  moe_fp8block_DeepSeek-V3       deepseek-ai/DeepSeek-V3     auto "$MOE"
run sgl_gdn_ctx       gdn_ctx_Qwen3.5-0.8B           Qwen/Qwen3.5-0.8B           auto "$GDN"
run sgl_gdn_gen       gdn_gen_Qwen3.5-0.8B           Qwen/Qwen3.5-0.8B           auto "$GDN"
