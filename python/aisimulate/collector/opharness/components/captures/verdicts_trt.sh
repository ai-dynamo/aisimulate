#!/bin/bash
cd "${AIS_PROBE_WORKSPACE:?set AIS_PROBE_WORKSPACE to the probe workspace}"
export AIS_SM=${AIS_SM:-sm90}
PD=ais/python/aisimulate/collector/opharness/components/path_diff.py
OUT=ais/python/aisimulate/collector/opharness/results/pathdiff/sm90/trtllm-1.3.0rc23
mkdir -p $OUT
run() { local cap=$1 name=$2 repo=$3 kv=$4 hint=$5 kvarg=(); [ -n "$kv" ] && kvarg=(--kv-dtype "$kv")
  [ -f facts/pathdiff/opcov_$cap.json ] || { printf '%-36s MISSING CAPTURE\n' "$name"; return; }
  python3 $PD --diff --capture-file facts/pathdiff/opcov_$cap.json --repo "$repo" --framework trtllm --version 1.3.0rc23 "${kvarg[@]}" --op-hint "$hint" --save-verdict $OUT/$name.json 2>/dev/null | python3 -c "
import sys,json;t=sys.stdin.read()
if not t.strip(): print('%-36s'%'$name','NO SERVING RECORD'); sys.exit()
d=json.loads(t[t.index('{'):]); print('%-36s'%'$name', d['verdict'].upper().ljust(9), 'only_col=',d['collector_only_signal'], 'drift=',{k:v['collector_only_kernels'][:3] for k,v in (d['kernel_drift'] or {}).items()}, '| col=',d['collector_backends'],'| srv=',d['serving_record']['id'],'| err=',(d.get('capture_run_error') or '')[:80])"; }
ATT='attn|attention|fmha|mha|flash|xqa'
DSA='mla|dsa|nsa|attn|indexer|flash|mqa|sparse|gemm|deepgemm|quant|topk|fmha'
GEMM='gemm|linear|proj|deepgemm|cutlass|cublas|quant'
MOE='moe|expert|grouped|fused_moe|deepgemm|topk|routing|cutlass'
GDN='gdn|delta|conv1d|linear|mamba|gated|recurr'
run trt_attn_ctx      attn_ctx_Llama-3.1-8B       meta-llama/Meta-Llama-3.1-8B ""   "$ATT"
run trt_attn_gen      attn_gen_Llama-3.1-8B       meta-llama/Meta-Llama-3.1-8B ""   "$ATT"
run trt_dsa_ctx       dsa_ctx_bf16_DeepSeek-V3.2  deepseek-ai/DeepSeek-V3.2   ""   "$DSA"
run trt_dsa_gen       dsa_gen_bf16_DeepSeek-V3.2  deepseek-ai/DeepSeek-V3.2   ""   "$DSA"
run trt_mla_ctx       mla_ctx_bf16_DeepSeek-V3    deepseek-ai/DeepSeek-V3     ""   "$DSA"
run trt_mla_gen       mla_gen_bf16_DeepSeek-V3    deepseek-ai/DeepSeek-V3     ""   "$DSA"
run trt_gemm_fp8block gemm_fp8block_DeepSeek-V3   deepseek-ai/DeepSeek-V3     ""   "$GEMM"
run trt_moe_fp8block  moe_fp8block_DeepSeek-V3    deepseek-ai/DeepSeek-V3     ""   "$MOE"
run trt_gdn_ctx       gdn_ctx_Qwen3.5-0.8B        Qwen/Qwen3.5-0.8B           ""   "$GDN"
# FLA fused_recurrent row is the deliberate fallback lane; FlashInfer decode row matches — explained, not a gate verdict
# run trt_gdn_gen       gdn_gen_Qwen3.5-0.8B        Qwen/Qwen3.5-0.8B           ""   "$GDN"
