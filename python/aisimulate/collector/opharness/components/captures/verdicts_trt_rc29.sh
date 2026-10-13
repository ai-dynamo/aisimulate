#!/bin/bash
# trtllm 1.3.0rc29 gates (2026-10-01): captures re-taken in the rc29 image as opcov_<cap>@1.3.0rc29.json;
# SM-parametric like verdicts_vllm_0300.sh. The rc23 script stays as the previous pin's record.
HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # resolve from this file BEFORE cd-ing into the workspace
cd ${AIS_PROBE_WORKSPACE:-.}
export AIS_PROBE_WORKSPACE=${AIS_PROBE_WORKSPACE:-.} AIS_SM=${AIS_SM:-$(python3 -c "import yaml;print('sm%d'%yaml.safe_load(open('$HARNESS/targets.yaml'))['platform']['sm'])")}
PD=$HARNESS/components/path_diff.py
OUT=$HARNESS/results/pathdiff/$AIS_SM/trtllm-1.3.0rc29
mkdir -p $OUT
run() { local cap=$1 name=$2 repo=$3 kv=$4 hint=$5 kvarg=(); [ -n "$kv" ] && kvarg=(--kv-dtype "$kv")
  # Per-SM platform floors (same contract as verdicts_vllm_0300.sh): FLOOR_SM=<sm>[,<sm>...] FLOOR_NOTE="<why>" run ...
  # records a platform-floor verdict (no serving instance on that SM by framework fact / capacity), excluded from path_aligned,
  # still counted as declared coverage by workflow_check.declared_gates.
  if [ -n "$FLOOR_SM" ] && [[ ",$FLOOR_SM," == *",$AIS_SM,"* ]]; then
    python3 - "$OUT/$name.json" "$name" "$repo" "$AIS_SM" "$FLOOR_NOTE" <<'PY'
import json,sys,time; out,name,repo,sm,note=sys.argv[1:]
json.dump({"verdict":"platform-floor","gate_name":name,"repo":repo,"framework":"trtllm","version":"1.3.0rc29","sm":sm,
           "note":note,"graded_at":time.strftime("%Y-%m-%dT%H:%M:%S")}, open(out,"w"), indent=1)
PY
    printf '%-36s PLATFORM-FLOOR (%s): %s\n' "$name" "$AIS_SM" "$FLOOR_NOTE"; return; fi
  [ -f facts/pathdiff/opcov_$cap@1.3.0rc29.json ] || { printf '%-36s MISSING CAPTURE\n' "$name"; return; }
  python3 $PD --diff --capture-file facts/pathdiff/opcov_$cap@1.3.0rc29.json --repo "$repo" --framework trtllm --version 1.3.0rc29 "${kvarg[@]}" --op-hint "$hint" --save-verdict $OUT/$name.json 2>/dev/null | python3 -c "
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
FLOOR_SM=sm89 FLOOR_NOTE="TRT-LLM 1.3.0rc29 on Ada: DeepSeek-V3.2 sparse MLA needs DeepGEMM SM90+/SM10x attention kernels (attention.hpp arch assert) on top of the V3 MoE/MLA floors; no serving record can exist" run trt_dsa_ctx       dsa_ctx_bf16_DeepSeek-V3.2  deepseek-ai/DeepSeek-V3.2   auto "$DSA"
FLOOR_SM=sm89 FLOOR_NOTE="TRT-LLM 1.3.0rc29 on Ada: DeepSeek-V3.2 sparse MLA needs DeepGEMM SM90+/SM10x attention kernels (attention.hpp arch assert) on top of the V3 MoE/MLA floors; no serving record can exist" run trt_dsa_gen       dsa_gen_bf16_DeepSeek-V3.2  deepseek-ai/DeepSeek-V3.2   auto "$DSA"
FLOOR_SM=sm89 FLOOR_NOTE="TRT-LLM 1.3.0rc29 on Ada: DeepSeek-V3/R1 cannot boot — no MoE implementation serves FP8 block-scales below SM90/SM120 (CutlassFusedMoE SM90/SM120, TritonFusedMoE SM90, Marlin NVFP4 only; findings sm89_trtllm_rc29_probe_2026_10_04) AND MLA has no FMHA kernel (attentionOp.cpp:3234, re-verified 2026-10-04); no serving record can exist" run trt_mla_ctx       mla_ctx_bf16_DeepSeek-V3    deepseek-ai/DeepSeek-V3     auto "$DSA"
FLOOR_SM=sm89 FLOOR_NOTE="TRT-LLM 1.3.0rc29 on Ada: DeepSeek-V3/R1 cannot boot — no MoE implementation serves FP8 block-scales below SM90/SM120 (CutlassFusedMoE SM90/SM120, TritonFusedMoE SM90, Marlin NVFP4 only; findings sm89_trtllm_rc29_probe_2026_10_04) AND MLA has no FMHA kernel (attentionOp.cpp:3234, re-verified 2026-10-04); no serving record can exist" run trt_mla_gen       mla_gen_bf16_DeepSeek-V3    deepseek-ai/DeepSeek-V3     auto "$DSA"
FLOOR_SM=sm89 FLOOR_NOTE="TRT-LLM 1.3.0rc29 on Ada: DeepSeek-V3/R1 cannot boot — no MoE implementation serves FP8 block-scales below SM90/SM120 (CutlassFusedMoE SM90/SM120, TritonFusedMoE SM90, Marlin NVFP4 only; findings sm89_trtllm_rc29_probe_2026_10_04) AND MLA has no FMHA kernel (attentionOp.cpp:3234, re-verified 2026-10-04); no serving record can exist" run trt_gemm_fp8block gemm_fp8block_DeepSeek-V3   deepseek-ai/DeepSeek-V3     auto "$GEMM"
FLOOR_SM=sm89 FLOOR_NOTE="TRT-LLM 1.3.0rc29 on Ada: DeepSeek-V3/R1 cannot boot — no MoE implementation serves FP8 block-scales below SM90/SM120 (CutlassFusedMoE SM90/SM120, TritonFusedMoE SM90, Marlin NVFP4 only; findings sm89_trtllm_rc29_probe_2026_10_04) AND MLA has no FMHA kernel (attentionOp.cpp:3234, re-verified 2026-10-04); no serving record can exist" run trt_moe_fp8block  moe_fp8block_DeepSeek-V3    deepseek-ai/DeepSeek-V3     auto "$MOE"
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
FLOOR_SM=sm89 FLOOR_NOTE="TRT-LLM 1.3.0rc29 on Ada: DeepSeek-V3/R1 cannot boot — no MoE implementation serves FP8 block-scales below SM90/SM120 (CutlassFusedMoE SM90/SM120, TritonFusedMoE SM90, Marlin NVFP4 only; findings sm89_trtllm_rc29_probe_2026_10_04) AND MLA has no FMHA kernel (attentionOp.cpp:3234, re-verified 2026-10-04); no serving record can exist" run trt_mla_ctx_fp8kv  mla_ctx_fp8_DeepSeek-V3    deepseek-ai/DeepSeek-V3     fp8  "$DSA"
FLOOR_SM=sm89 FLOOR_NOTE="TRT-LLM 1.3.0rc29 on Ada: DeepSeek-V3/R1 cannot boot — no MoE implementation serves FP8 block-scales below SM90/SM120 (CutlassFusedMoE SM90/SM120, TritonFusedMoE SM90, Marlin NVFP4 only; findings sm89_trtllm_rc29_probe_2026_10_04) AND MLA has no FMHA kernel (attentionOp.cpp:3234, re-verified 2026-10-04); no serving record can exist" run trt_mla_gen_fp8kv  mla_gen_fp8_DeepSeek-V3    deepseek-ai/DeepSeek-V3     fp8  "$DSA"
# the fp8-context-FMHA attention cell (use_fp8_context_fmha=True) is captured
# too (opcov_trt_attn_ctx_fp8kv_fp8fmha); whichever cell serving's kernel
# names match is the gate entry above — see findings trtllm_kv_variants
# FLA fused_recurrent row is the deliberate fallback lane; FlashInfer decode row matches — explained, not a gate verdict
# run trt_gdn_gen       gdn_gen_Qwen3.5-0.8B        Qwen/Qwen3.5-0.8B           ""   "$GDN"

# --- 2026-10-01 gate backfill: DSV4 families (owner decision 3). Serving = the rc29
# plan records (isl 4096, kv auto; Hopper DSV4 runs on the fp8_ds_mla pool customization).
DSV4='dsv4|dsa|csa|hca|compress|indexer|mqa|sparse|flash|attn|mla|topk|gemm|deepgemm|quant|fmha'
M=sgl-project/DeepSeek-V4-Flash-FP8
FLOOR_SM=sm89 FLOOR_NOTE="TRT-LLM 1.3.0rc29 on Ada: 'DeepSeek-V4 requires Hopper or newer GPUs' (model init RuntimeError); no serving record can exist" run trt_dsv4_csa_ctx  dsv4_csa_ctx_DeepSeek-V4-Flash-FP8  $M auto "$DSV4"
FLOOR_SM=sm89 FLOOR_NOTE="TRT-LLM 1.3.0rc29 on Ada: 'DeepSeek-V4 requires Hopper or newer GPUs' (model init RuntimeError); no serving record can exist" run trt_dsv4_csa_gen  dsv4_csa_gen_DeepSeek-V4-Flash-FP8  $M auto "$DSV4"
FLOOR_SM=sm89 FLOOR_NOTE="TRT-LLM 1.3.0rc29 on Ada: 'DeepSeek-V4 requires Hopper or newer GPUs' (model init RuntimeError); no serving record can exist" run trt_dsv4_hca_ctx  dsv4_hca_ctx_DeepSeek-V4-Flash-FP8  $M auto "$DSV4"
FLOOR_SM=sm89 FLOOR_NOTE="TRT-LLM 1.3.0rc29 on Ada: 'DeepSeek-V4 requires Hopper or newer GPUs' (model init RuntimeError); no serving record can exist" run trt_dsv4_hca_gen  dsv4_hca_gen_DeepSeek-V4-Flash-FP8  $M auto "$DSV4"

# Qwen3-VL encoder (2026-10-02): serving side = probes/vision_trtllm.py sidecar
# (synthetic image through Qwen3VLModel.mm_encoder.visual, merged as profile_run).
ENC='vision|vit|patch|encoder|flash|attn|fmha'
run trt_encoder_attn_qwen3vl  encoder_attn_Qwen3-VL-2B  Qwen/Qwen3-VL-2B-Instruct auto "$ENC"

# --- 2026-10-02 backfill, remaining families with a serving record on rc29
MAMBA='mamba|ssm|chunk|scan|conv1d|selective|causal|state'
MLA='mla|attn|flash|bmm|proj|indexer|rope|concat|gemm|gemvx|cutlass|deepgemm|quant|nvjet|cublas'
run trt_gdn_gen       gdn_gen_Qwen3.5-0.8B           Qwen/Qwen3.5-0.8B             auto "$GDN"
FLOOR_SM=sm89 FLOOR_NOTE="TRT-LLM 1.3.0rc29 on Ada: DeepSeek-V3/R1 cannot boot — no MoE implementation serves FP8 block-scales below SM90/SM120 (CutlassFusedMoE SM90/SM120, TritonFusedMoE SM90, Marlin NVFP4 only; findings sm89_trtllm_rc29_probe_2026_10_04) AND MLA has no FMHA kernel (attentionOp.cpp:3234, re-verified 2026-10-04); no serving record can exist" run trt_mla_bmm       mla_bmm_gen_DeepSeek-V3        deepseek-ai/DeepSeek-V3       auto "$MLA"
run trt_mamba2_ctx    mamba2_ctx_Nemotron-H-56B      nvidia/Nemotron-H-56B-Base-8K auto "$MAMBA"
run trt_mamba2_gen    mamba2_gen_Nemotron-H-56B      nvidia/Nemotron-H-56B-Base-8K auto "$MAMBA"
MSA='attn|sparse|msa|topk|index|score|merge|block'
FLOOR_SM=sm90,sm100,sm103,sm120 FLOOR_NOTE="no serving record on this SM yet: its MiniMax-M3 probes ran on the 2-sparse-layer dummy, which cannot boot on trtllm (the KV cache manager allocates index-K for layers 3..N by checkpoint convention while the model follows the cut config; findings minimax_m3_dummy_dense_head_2026_10_05) — re-probe with the fixed m3 dummy (3-layer dense head + 1 sparse layer) and drop this SM from the list; sm89 already grades for real" run trt_msa_ctx msa_ctx_MiniMax-M3 MiniMaxAI/MiniMax-M3 auto "$MSA"
FLOOR_SM=sm90,sm100,sm103,sm120 FLOOR_NOTE="no serving record on this SM yet: its MiniMax-M3 probes ran on the 2-sparse-layer dummy, which cannot boot on trtllm (the KV cache manager allocates index-K for layers 3..N by checkpoint convention while the model follows the cut config; findings minimax_m3_dummy_dense_head_2026_10_05) — re-probe with the fixed m3 dummy (3-layer dense head + 1 sparse layer) and drop this SM from the list; sm89 already grades for real" run trt_msa_gen msa_gen_MiniMax-M3 MiniMaxAI/MiniMax-M3 auto "$MSA"
