#!/bin/bash
cd ${AIS_PROBE_WORKSPACE:-.}
export AIS_PROBE_WORKSPACE=${AIS_PROBE_WORKSPACE:-.} AIS_SM=sm90
PD=ais/python/aisimulate/collector/opharness/components/path_diff.py
OUT=ais/python/aisimulate/collector/opharness/results/pathdiff/sm90/vllm-0.29.0
run() { local cap=$1 name=$2 repo=$3 kv=$4 hint=$5 kvarg=(); [ -n "$kv" ] && kvarg=(--kv-dtype "$kv")
  [ -f facts/pathdiff/opcov_$cap.json ] || { printf '%-40s MISSING CAPTURE\n' "$name"; return; }
  python3 $PD --diff --capture-file facts/pathdiff/opcov_$cap.json --repo "$repo" --framework vllm --version 0.29.0 "${kvarg[@]}" --op-hint "$hint" --save-verdict $OUT/$name.json 2>/dev/null | python3 -c "
import sys,json;t=sys.stdin.read()
if '{' not in t: print('%-40s'%'$name','NO SERVING RECORD', t.strip()[:80]); sys.exit()
d=json.loads(t[t.index('{'):])
print('%-40s'%'$name', d['verdict'].upper().ljust(9), 'only_col=',d['collector_only_signal'], 'drift=',{k:v['collector_only_kernels'][:3] for k,v in (d['kernel_drift'] or {}).items()}, '| col=',d['collector_backends'], '| err=',d.get('capture_run_error'))"; }
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
# explained decomposition (collector measures pre/post unfused; SDK composes attn_norm separately) — not a gate verdict
# run mhc_pre_post           mhc_module_DeepSeek-V4-Flash-FP8        $M fp8 "$MHC"
# Kimi-K3: since the generator caps the KDA decode batch at 256 (vllm.rule,
# owner decision 2026-09-25) the framework-mode probe passes; the serving
# record is selected by the default rule again (no --serving-raw).
run kda_ctx                kda_ctx_Kimi-K3                         moonshotai/Kimi-K3 auto "$KDA"
# collector also measures the Triton fallback lane (spec/mixed batches); fused row matches serving — not a gate verdict
# run kda_gen                kda_gen_Kimi-K3                         moonshotai/Kimi-K3 auto "$KDA"
