# Current predictor interpretation for prospective GB200 v6

This clarification applies to fresh predictions from FPM head
`3627568d809774eb693abf3551a2ed36ad1dc032`, native extension SHA256
`f54a6015f6a97212e018615d909f1814f729d9b25df8b64e86802b7c5630e163`.
It must accompany any fresh `compare_e2e.py` output. Source hashes are recorded
in the adjacent preparation source inventory and in the comparison output.
No historical report, prediction, observed time, or comparison source is edited.

The unchanged comparison script emits an old limitation saying that both
SGLang and vLLM charge a separate first-output decode after prefill. That
generalization is stale for the current SGLang scheduler. Its completed
prefill pass calls `simulate_prefill_first_tokens` at prefill completion and
adds no separate decode duration for the first token. Source:
`crates/core/src/engine/scheduler/sglang/core.rs:739` and
`crates/core/src/engine/scheduler/sglang/decode.rs:373`.

The vLLM aggregated replay used for this GB200 comparison still charges the
decode predictor when it emits the first output after completing prefill.
The zero-duration branch in `emit_ready_tokens` applies only to a dedicated
`WorkerType::Prefill` worker. The comparison's `vllm_engine` does not select that
worker; the default is `Aggregated`. Source:
`data/experimental/deepseek-v41/verification-plan/compare_e2e.py:163`,
`crates/core/src/engine/scheduler/vllm/core.rs:2538`, and
`crates/core/src/engine/config.rs:455`.

Accordingly, this GB200 report retains the vLLM first-output modeling
approximation. Do not subtract a decode time from the fresh predictions,
claim that both schedulers have been corrected, or recalculate old accuracy
tables under the new source identity. The observed native interval comparison
and the complete HTTP replay comparison remain different comparisons with
their own source-bound outputs and coverage.

The new measurement uses native prefix retention128 and requires gate512.
Any resulting performance claim is conditional on that actual policy, this
single physical lifecycle, the frozen GB200 calibration, and native
per-dispatch zero-pressure eligibility. It does not establish default
retention0 performance or native allocator memory accuracy.
