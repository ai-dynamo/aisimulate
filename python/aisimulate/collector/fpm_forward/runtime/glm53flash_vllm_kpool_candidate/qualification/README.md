# Native GLM IndexPool repaired-wheel functional qualification

These original wrappers call the public vLLM LLM/EngineArgs/sleep/enqueue/wake_up/wait_for_completion/collective_rpc APIs and observe actual native V2 InputBatch execution. No Req, attention metadata, cache, model tensor input, state buffer or scheduler counter is hand-built or modified. Upstream API/source revision: vllm-project/vllm@ced6857afa0ea7b2e3f0846a62e1394e90f15607, paths vllm/entrypoints/llm.py, vllm/engine/arg_utils.py, vllm/v1/worker/gpu/{model_runner,input_batch,states}.py and vllm/v1/core/sched/scheduler.py. Source URLs are https://github.com/vllm-project/vllm/tree/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm . Upstream copyright contributors to the vLLM project; Apache-2.0 LICENSE included. No upstream implementation is copied into these original probe scripts.

The patched wheel has actual distribution version0.30.0+glm53kpool.bf5f6b0e689d and SHAa3b63cb3c95cf976f717077102e8172a33501c7d05092bc84cc58e3aaef47d36. Its independent source build receipt is included. This is not a stock-runtime support claim. expected-runtime.json freezes its modified helper and the stock helper separately, 23 other source files, all19 exact image-derived native binaries, both original checkpoint config hashes/revisions and the candidate wheel digest. Candidate private installation passed CPU604941. The frozen production-policy native Engine functional suite has now passed and been reviewed for all four FP8/NVFP4 TP2/TP4 cells; the scope and immutable summary are recorded below.

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


## Reviewed four-cell result

| Checkpoint | TP | Original GPU job | Profiles | Exact comparisons | Result |
| --- | --- | --- | --- | --- | --- |
| FP8 | 2 | 608011 | 3 | 40 | Passed, no differences |
| FP8 | 4 | 606669 | 3 | 40 | Passed, no differences |
| NVFP4 | 2 | 608013 | 3 | 40 | Passed, no differences |
| NVFP4 | 4 | 608014 | 3 | 40 | Passed, no differences |

Each profile completed 20 requests and 32 greedy output tokens per request.
The three profiles are stock/reference, candidate/reference and candidate/split,
all using the same v3 native cohort protocol. Current strict revalidation and
an independent original prompt/output/finish comparison both passed all four
cells. Original preflight and native receipt bytes are preserved.

`admission-summary.json` has SHA256
`d43dfdcfabe870cc51983fa41fada4897b4d84d64ac57fafe2236e7753435e67`. It binds the
current validator source commit `aae82cc48f2e05a1a077893ba42152d8708771de`, the
original frozen validator SHA256
`839af86e26a41a14a79ed301c71e495c571d16970f6c37d0802a30fd7d3fb5e2`, all original
raw locations and hashes, and the actual source/wheel/19-binary/checkpoint
identities. Its final remote locations were populated and hash-verified on
private Lustre before these URIs were included. Current-validator additions
are identified by `revalidation_added_fields`; no historical receipt is
rewritten to appear originally stricter.

The approved scope is
`native_Engine_functional_correctness_for_frozen_geometry_suite_only`. Stock
retained-tail rejection remains active. Formal campaign hardware/capacity
qualification, eight-configuration collection coverage, performance accuracy
and graph Ops support remain NOT_EVALUATED. The historical failed probes
above are retained; their status is not changed by these later successful jobs.
