#!/bin/bash
cd ${AIS_PROBE_WORKSPACE:-.}
export AIS_PROBE_WORKSPACE=${AIS_PROBE_WORKSPACE:-.} AIS_SM=sm90
PD=ais/python/aisimulate/collector/opharness/components/path_diff.py
OUT=ais/python/aisimulate/collector/opharness/results/pathdiff/sm90/trtllm-1.3.0rc23
mkdir -p $OUT
run() { local cap=$1 name=$2 repo=$3 kv=$4 hint=$5 kvarg=(); [ -n "$kv" ] && kvarg=(--kv-dtype "$kv")
  [ -f facts/pathdiff/opcov_$cap.json ] || { printf '%-36s MISSING CAPTURE\n' "$name"; return; }
  python3 $PD --diff --capture-file facts/pathdiff/opcov_$cap.json --repo "$repo" --framework trtllm --version 1.3.0rc23 "${kvarg[@]}" --op-hint "$hint" --save-verdict $OUT/$name.json 2>/dev/null | python3 -c "
import sys,json;t=sys.stdin.read()
if '{' not in t: print('%-36s'%'$name','NO SERVING RECORD', t.strip()[:80]); sys.exit()
d=json.loads(t[t.index('{'):]); print('%-36s'%'$name', d['verdict'].upper().ljust(9), 'only_col=',d['collector_only_signal'], 'drift=',{k:v['collector_only_kernels'][:3] for k,v in (d['kernel_drift'] or {}).items()}, '| col=',d['collector_backends'],'| srv=',d['serving_record']['id'],'| err=',(d.get('capture_run_error') or '')[:80])"; }
ATT='attn|attention|fmha|mha|flash|xqa'
DSA='mla|dsa|nsa|attn|indexer|flash|mqa|sparse|gemm|deepgemm|quant|topk|fmha'
GEMM='gemm|linear|proj|deepgemm|cutlass|cublas|quant'
MOE='moe|expert|grouped|fused_moe|deepgemm|topk|routing|cutlass'
GDN='gdn|delta|conv1d|linear|mamba|gated|recurr'
# kv "auto" pins the rendered (bf16-KV) serving record now that trtllm has
# fp8-KV variant records too (probe_trtllm --kv-dtype, 2026-09-24)
run trt_attn_ctx      attn_ctx_Llama-3.1-8B       meta-llama/Meta-Llama-3.1-8B auto "$ATT"
run trt_attn_gen      attn_gen_Llama-3.1-8B       meta-llama/Meta-Llama-3.1-8B auto "$ATT"
run trt_dsa_ctx       dsa_ctx_bf16_DeepSeek-V3.2  deepseek-ai/DeepSeek-V3.2   auto "$DSA"
run trt_dsa_gen       dsa_gen_bf16_DeepSeek-V3.2  deepseek-ai/DeepSeek-V3.2   auto "$DSA"
run trt_mla_ctx       mla_ctx_bf16_DeepSeek-V3    deepseek-ai/DeepSeek-V3     auto "$DSA"
run trt_mla_gen       mla_gen_bf16_DeepSeek-V3    deepseek-ai/DeepSeek-V3     auto "$DSA"
run trt_gemm_fp8block gemm_fp8block_DeepSeek-V3   deepseek-ai/DeepSeek-V3     auto "$GEMM"
run trt_moe_fp8block  moe_fp8block_DeepSeek-V3    deepseek-ai/DeepSeek-V3     auto "$MOE"
run trt_gdn_ctx       gdn_ctx_Qwen3.5-0.8B        Qwen/Qwen3.5-0.8B           auto "$GDN"
# fp8-KV variants: single-cell captures (s/caps/trt_kv/*.py: attention
# use_fp8_kv_cache=True; dsa/mla kv 'fp8' + gemm fp8_block) vs the fp8-KV
# variant records (kv_cache_config.dtype fp8, KV manager resolved FP8)
run trt_attn_ctx_fp8kv attn_ctx_fp8_Llama-3.1-8B  meta-llama/Meta-Llama-3.1-8B fp8  "$ATT"
run trt_attn_gen_fp8kv attn_gen_fp8_Llama-3.1-8B  meta-llama/Meta-Llama-3.1-8B fp8  "$ATT"
# not a gate on rc23: the sparse FlashMLA asserts bf16 KV, so no fp8-KV DSA serving record can exist
# (findings trtllm_kv_dtype_variants_2026_09_24); the cell is a recorded framework wall, not an alignment question
# run trt_dsa_ctx_fp8kv  dsa_ctx_fp8_DeepSeek-V3.2  deepseek-ai/DeepSeek-V3.2   fp8  "$DSA"
# not a gate on rc23: the sparse FlashMLA asserts bf16 KV, so no fp8-KV DSA serving record can exist
# (findings trtllm_kv_dtype_variants_2026_09_24); the cell is a recorded framework wall, not an alignment question
# run trt_dsa_gen_fp8kv  dsa_gen_fp8_DeepSeek-V3.2  deepseek-ai/DeepSeek-V3.2   fp8  "$DSA"
run trt_mla_ctx_fp8kv  mla_ctx_fp8_DeepSeek-V3    deepseek-ai/DeepSeek-V3     fp8  "$DSA"
run trt_mla_gen_fp8kv  mla_gen_fp8_DeepSeek-V3    deepseek-ai/DeepSeek-V3     fp8  "$DSA"
# the fp8-context-FMHA attention cell (use_fp8_context_fmha=True) is captured
# too (opcov_trt_attn_ctx_fp8kv_fp8fmha); whichever cell serving's kernel
# names match is the gate entry above — see findings trtllm_kv_variants
# FLA fused_recurrent row is the deliberate fallback lane; FlashInfer decode row matches — explained, not a gate verdict
# run trt_gdn_gen       gdn_gen_Qwen3.5-0.8B        Qwen/Qwen3.5-0.8B           ""   "$GDN"
