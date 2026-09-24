# Native GLM IndexPool repaired-wheel functional qualification

These original wrappers call the public vLLM LLM/EngineArgs/sleep/enqueue/wake_up/wait_for_completion/collective_rpc APIs and observe actual native V2 InputBatch execution. No Req, attention metadata, cache, model tensor input, state buffer or scheduler counter is hand-built or modified. Upstream API/source revision: vllm-project/vllm@ced6857afa0ea7b2e3f0846a62e1394e90f15607, paths vllm/entrypoints/llm.py, vllm/engine/arg_utils.py, vllm/v1/worker/gpu/{model_runner,input_batch,states}.py and vllm/v1/core/sched/scheduler.py. Source URLs are https://github.com/vllm-project/vllm/tree/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm . Upstream copyright contributors to the vLLM project; Apache-2.0 LICENSE included. No upstream implementation is copied into these original probe scripts.

The patched wheel has actual distribution version0.30.0+glm53kpool.bf5f6b0e689d and SHAa3b63cb3c95cf976f717077102e8172a33501c7d05092bc84cc58e3aaef47d36. Its independent source build receipt is included. This is not a stock-runtime support claim. expected-runtime.json freezes its modified helper and the stock helper separately, 23 other source files, all19 exact image-derived native binaries, both original checkpoint config hashes/revisions and the candidate wheel digest. Candidate private installation passed CPU604941; complete Engine correctness is initially NOT_EVALUATED.

Run each profile in a separate fresh process/container and output directory, on the actual checkpoint and TP cell. Use PYTHONPATH=PAYLOAD:PRIVATE_CANDIDATE for candidate; use PYTHONPATH=PAYLOAD with the immutable stock image for stock. Leave all DYN_FPM_* real-KV/benchmark overlays and AISIM_GLM53_PURPOSE unset. Use allocation-private compile/cache directories set before imports, and VLLM_WORKER_MULTIPROC_METHOD=spawn.

First perform actual installed-source/API/EngineArgs/config CPU gates (no GPU or weights are loaded):

    python PAYLOAD/probe.py --model MODEL --checkpoint fp8 --tp 4 --runtime candidate --mode split --policy production --output NEW --cpu-preflight

Repeat the gate for stock/reference.

Then GPU profiles, with identical model/TP/policy/input source:

    python PAYLOAD/probe.py --model MODEL --checkpoint fp8 --tp 4 --runtime stock --mode reference --policy production --output STOCK_REFERENCE
    python PAYLOAD/probe.py --model MODEL --checkpoint fp8 --tp 4 --runtime candidate --mode reference --policy production --output CANDIDATE_REFERENCE
    python PAYLOAD/probe.py --model MODEL --checkpoint fp8 --tp 4 --runtime candidate --mode split --policy production --output CANDIDATE_SPLIT

`--policy eager` is a separately identified alternative and must be compared only against the same policy. Production requires actual completed native FULL decode graph dispatch; eager requires NONE. This is a correctness probe, not a timing producer. No benchmark/stock-unqualified guard is disabled.

The public long_prefill_token_threshold4097, max_num_batched_tokens16398 and disabled prefix caching/mamba_cache_mode=none make the ordinary scheduler naturally split native real requests. Cases include singleton P4097/Q3 andQ4, both in a heterogeneous B2 batch, all four resulting pool tails in a B4 batch with Q1/2/3/4, and B2 two-chunk prefixes8194 with finalQ2/3. The native partition is verified from the workers, never assumed from arguments. The reference public threshold0 processes each exact prompt in one prefill. Every case runs twice with the original frozen text tokenized by the actual pinned tokenizer, followed by32 greedy tokens.

The read-only worker extension records actual query token IDs, native prefix lengths/state slots, request IDs, selected graph mode and actual sampled IDs. It marks completion only after the original model execute and original native sampling both return and CUDA synchronization completes. The verifier checks all TP ranks, full same-request prefix/decode chains, exact source/binary/version/physicalGPU/settings receipts, identical repeated outputs, and the required heterogeneous native target batches. Intermediate prefill samples are preserved but never mistaken for committed request output.

Finally:

    python PAYLOAD/validate.py --stock-reference STOCK_REFERENCE --candidate-reference CANDIDATE_REFERENCE --candidate-split CANDIDATE_SPLIT --output NEW_REPORT.json

It recomputes every native chain, compares stock versus candidate one-shot output and candidate one-shot versus split output token-for-token, preserves first differences and original logprobs, and exits nonzero on any mismatch. A pass applies only to this checkpoint/TP/policy and frozen functional geometry suite. All performance/MAPE/full8cell accuracy status stays NOT_EVALUATED. Keep all failures and earlier immutable probes.

Qualification v3 uses the pinned public scheduling-only admission barrier for **all three profiles**: `LLM.sleep(level=0, mode="keep")`, `enqueue`, `wake_up(tags=["scheduling"])`, then `wait_for_completion`. The native APIs leave weights and KV/state memory resident. The source manifest `cohort-source.json` pins the actual implementation; the CPU gate checks its installed bytes. Native returned internal request IDs are recorded separately from final external IDs and bound through the actual assignment witness. Every cohort must also appear together in the first completed worker prefill, and the original heterogeneous tail targets must still appear together. No scheduler counters or requests are edited.

This follows upstream `vllm/entrypoints/llm.py:727-729` and `vllm/v1/engine/core.py:879-935` at the immutable revision above. Historical GPU 605360 used `generate`, which permitted the independent engine process to start its first request before remaining prompts arrived. All 20 candidate split outputs matched the candidate reference for 32 tokens, but the required B2/B4 target cohorts were staggered. That historical qualification remains failed; v3 does not relabel or overwrite it. Native references with different cohort admission policies cannot be combined into a single v3 qualification result.
