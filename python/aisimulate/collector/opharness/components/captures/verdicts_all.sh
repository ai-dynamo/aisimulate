#!/bin/bash
# capture -> (verdict name, serving repo, kv, op_hint)
cd "${AIS_PROBE_WORKSPACE:?set AIS_PROBE_WORKSPACE to the probe workspace}"
export AIS_SM=${AIS_SM:-sm90}
PD=ais/python/aisimulate/collector/opharness/components/path_diff.py
OUT=ais/python/aisimulate/collector/opharness/results/pathdiff/sm90/vllm-0.29.0
run() { local cap=$1 name=$2 repo=$3 kv=$4 hint=$5 kvarg=(); [ -n "$kv" ] && kvarg=(--kv-dtype "$kv")
  [ -f facts/pathdiff/opcov_$cap.json ] || { printf '%-36s MISSING CAPTURE\n' "$name"; return; }
  python3 $PD --diff --capture-file facts/pathdiff/opcov_$cap.json --repo "$repo" --framework vllm --version 0.29.0 "${kvarg[@]}" --op-hint "$hint" --save-verdict $OUT/$name.json 2>/dev/null | python3 -c "
import sys,json;d=json.load(sys.stdin)
print('%-36s'%'$name', d['verdict'].upper().ljust(9), 'only_col=',d['collector_only_signal'], 'drift=',{k:v['collector_only_kernels'][:3] for k,v in (d['kernel_drift'] or {}).items()}, '| col=',d['collector_backends'], '| err=',d.get('capture_run_error'))"; }
GEMM='gemm|linear|proj|nvjet|cublas|deepgemm|scaled_mm|quant|cutlass'
MOE='moe|expert|grouped|fused_moe|deepgemm|topk|marlin|routing'
# the MLA module collectors include the projections (DeepGEMM fp8 block GEMM on
# DeepSeek fp8 checkpoints); under CUDA graphs the serving GEMM kernels sit
# under quant spans, so the hint must admit them or they read as collector-only
MLA='mla|attn|flash|bmm|proj|indexer|rope|concat|gemm|deepgemm|quant'
GDN='gdn|delta|conv1d|linear|mamba|gated|recurrent'
run gemm_fp8block      gemm_fp8block_DeepSeek-V3.2      deepseek-ai/DeepSeek-V3.2 fp8  "$GEMM"
run gemm_bf16          gemm_bf16_Llama-3.1-8B           meta-llama/Meta-Llama-3.1-8B "" "$GEMM"
# per-tensor fp8 serving instance = nvidia/Llama-3.1-70B-Instruct-FP8 (modelopt
# static fp8 -> ModelOptFp8LinearMethod, cutlass_3x_gemm_sm90_fp8). vLLM 0.29
# loads Qwen/Qwen3-32B-FP8-Static-PerTensor UNQUANTIZED (its config has no
# quant_method and vllm ignores the modelopt hf_quant_config for it) — not a
# valid instance; see findings full_reprobe_framework_mode_2026_09_24.
run gemm_fp8           gemm_fp8_Llama-3.1-70B-FP8       nvidia/Llama-3.1-70B-Instruct-FP8 auto "$GEMM"
run moe_fp8block_dsv3  moe_fp8block_DeepSeek-V3.2       deepseek-ai/DeepSeek-V3.2 fp8  "$MOE"
run moe_bf16_gemma4    moe_bf16_gemma4-26B-A4B          google/gemma-4-26B-A4B ""      "$MOE"
run moe_mxfp4_gptoss   moe_mxfp4_gpt-oss-120b           openai/gpt-oss-120b ""         "$MOE"
run mla_ctx_bf16       mla_ctx_bf16_DeepSeek-V3         deepseek-ai/DeepSeek-V3 auto   "$MLA"
# mla_gen_bf16 vs DeepSeek-V3 is an EXPLAINED divergence (bf16-weight MLA cell:
# fused_a_gemm runs only for bf16 q_a/kv_a weights; every DeepSeek-V3 serving
# artifact is fp8 -> no serving instance; consumed via the gemm_quant axis).
# Kept OUT of the gate dir on purpose — write its report next to the facts.
OUT_EXPLAINED=facts/pathdiff/explained; mkdir -p $OUT_EXPLAINED
OUT=$OUT_EXPLAINED run mla_gen_bf16 mla_gen_bf16_DeepSeek-V3 deepseek-ai/DeepSeek-V3 auto "$MLA"
run mla_ctx_fp8        mla_ctx_fp8_DeepSeek-R1          deepseek-ai/DeepSeek-R1 fp8    "$MLA"
run mla_gen_fp8        mla_gen_fp8_DeepSeek-R1          deepseek-ai/DeepSeek-R1 fp8    "$MLA"
run mla_bmm            mla_bmm_gen_DeepSeek-V3          deepseek-ai/DeepSeek-V3 auto   "$MLA"
run gdn_ctx            gdn_ctx_Qwen3.5-0.8B             Qwen/Qwen3.5-0.8B auto         "$GDN"
run gdn_gen            gdn_gen_Qwen3.5-0.8B             Qwen/Qwen3.5-0.8B auto         "$GDN"
# compute_scale = vllm's scaled_fp8_quant (per-tensor/per-token) kernels; the
# serving instance is a per-tensor fp8 checkpoint (Qwen3-32B-FP8 static —
# same kernel names as the gemm_fp8 verdict). DeepSeek-V3.2 is fp8 BLOCK and
# runs per_token_group_quant instead, so it was never a valid pairing
# (2026-09-24: the first no-collector-signal-rule recompute exposed it).
# compute_scale (standalone scaled_fp8_quant kernels) has NO serving instance in
# framework mode: torch.compile's norm_quant / act_quant fusion passes fold the
# activation quant into Inductor kernels (triton_*_fused__to_copy_clamp_cutlass_
# scaled_mm_* on Llama-3.1-70B-FP8) — the standalone kernel is eager-only.
# Kept OUT of the gate; report written next to the facts for the record.
OUT=$OUT_EXPLAINED run compute_scale compute_scale_Llama-3.1-70B-FP8 nvidia/Llama-3.1-70B-Instruct-FP8 auto 'quant|scale|per_token|fp8'
