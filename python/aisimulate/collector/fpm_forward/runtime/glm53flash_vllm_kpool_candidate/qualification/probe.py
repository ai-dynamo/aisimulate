# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordinary native Engine requests; this probe does not produce performance data."""

import argparse
import hashlib
import json
import os
from pathlib import Path

if __package__:
    from .worker_probe import verify_runtime
else:
    from worker_probe import verify_runtime

CASES = {
    "single_q3": [4100],
    "single_q4": [4101],
    "heterogeneous_b2": [4100, 4101],
    "four_tails_b4": [4098, 4099, 4100, 4101],
    "two_chunks_b2": [8196, 8197],
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", choices=("fp8", "nvfp4"), default="fp8")
    p.add_argument("--tp", type=int, choices=(2, 4), default=4)
    p.add_argument("--runtime", choices=("stock", "candidate"), required=True)
    p.add_argument("--mode", choices=("reference", "split"), required=True)
    p.add_argument("--policy", choices=("production", "eager"), default="production")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu-preflight", action="store_true")
    return p.parse_args()


def engine_kwargs(args, expected):
    return dict(
        model=args.model,
        revision=expected["checkpoints"][args.checkpoint]["revision"],
        tokenizer_revision=expected["checkpoints"][args.checkpoint]["revision"],
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        dtype="bfloat16",
        kv_cache_dtype="fp8_e4m3",
        max_model_len=131079,
        max_num_seqs=4,
        max_num_batched_tokens=16398,
        long_prefill_token_threshold=4097 if args.mode == "split" else 0,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        async_scheduling=False,
        distributed_executor_backend="mp",
        enforce_eager=args.policy == "eager",
        cudagraph_metrics=True,
        language_model_only=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
        worker_extension_cls="worker_probe.QualificationWorker",
    )


def save(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)


def cohort_sources():
    """Verify the native documented scheduling-only admission API."""
    import vllm

    package = Path(vllm.__file__).resolve().parent.parent
    sources = json.loads(Path(__file__).with_name("cohort-source.json").read_text())["sources"]
    result = {}
    for source in sources:
        actual = hashlib.sha256((package / source["path"]).read_bytes()).hexdigest()
        if actual != source["sha256"]:
            raise RuntimeError("native cohort admission source differs")
        result[source["path"]] = actual
    for method in ("sleep", "enqueue", "wake_up", "wait_for_completion"):
        if not callable(getattr(vllm.LLM, method, None)):
            raise RuntimeError("native public cohort admission API is missing")
    return result


def generate_cohort(llm, inputs, params, output, case, repetition):
    """Queue a whole cohort through the public API before native scheduling."""
    if llm.llm_engine.has_unfinished_requests():
        raise RuntimeError("previous native cohort is still active")

    def record(name, **fields):
        row = {"case": case, "repetition": repetition, "event": name, **fields}
        with (output / "cohort-admission.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    llm.sleep(level=0, mode="keep")
    if llm.llm_engine.is_sleeping() is not True:
        raise RuntimeError("native scheduling-only pause did not complete")
    record("scheduling_paused", level=0, mode="keep")
    ids = llm.enqueue(inputs, params, use_tqdm=False)
    if len(ids) != len(inputs) or len(set(ids)) != len(inputs) or not llm.llm_engine.is_sleeping():
        raise RuntimeError("native cohort did not enqueue completely while paused")
    record(
        "cohort_enqueued",
        native_request_ids=ids,
        prompt_sha256=[
            hashlib.sha256(json.dumps(item["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest()
            for item in inputs
        ],
    )
    llm.wake_up(tags=["scheduling"])
    if llm.llm_engine.is_sleeping() is not False:
        raise RuntimeError("native scheduling-only resume did not complete")
    record("scheduling_resumed", tags=["scheduling"])
    outputs = llm.wait_for_completion(use_tqdm=False)
    if llm.llm_engine.has_unfinished_requests() or len(outputs) != len(ids):
        raise RuntimeError("native completed requests differ from enqueued cohort")
    record("cohort_completed", external_request_ids=[row.request_id for row in outputs])
    return outputs


def main():
    args = parse()
    root = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=False)
    expected = json.loads((root / "expected-runtime.json").read_text())
    if any(os.environ.get(k) == "1" for k in ("DYN_FPM_GLM53FLASH_REAL_KV", "DYN_FPM_DSV41_REAL_KV")) or os.environ.get(
        "AISIM_GLM53_PURPOSE"
    ):
        raise RuntimeError(
            "ordinary Engine correctness probe must not load benchmark schedulers or Ops instrumentation"
        )
    runtime = verify_runtime(expected, args.runtime)
    config = json.loads((Path(args.model) / "config.json").read_text())
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if config_sha != expected["checkpoints"][args.checkpoint]["config_sha256"]:
        raise RuntimeError("actual model config differs from the immutable checkpoint")
    kwargs = engine_kwargs(args, expected)
    from vllm.engine.arg_utils import EngineArgs

    actual_args = EngineArgs(**kwargs)
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if __package__:
        from .worker_probe import QualificationWorker, install_request_id_witness, request_identity_sources
    else:
        from worker_probe import QualificationWorker, install_request_id_witness, request_identity_sources

    for name in ("prepare_inputs", "execute_model", "sample", "sample_tokens"):
        if not callable(getattr(GPUModelRunner, name, None)):
            raise RuntimeError("missing actual native V2 API")
    if not callable(QualificationWorker.install_repair_qualification_observer):
        raise RuntimeError("missing worker extension")
    save(
        args.output / "preflight.json",
        {
            "status": "passed",
            "runtime": runtime,
            "checkpoint_config_sha256": config_sha,
            "request_identity_protocol": "native_assign_request_id_v1",
            "request_identity_source_sha256": request_identity_sources(),
            "cohort_admission_protocol": "native_scheduling_pause_enqueue_v1",
            "cohort_admission_source_sha256": cohort_sources(),
            "public_engine_args": kwargs,
            "actual_engine_args_class": type(actual_args).__module__ + "." + type(actual_args).__name__,
            "gpu_execution": False,
        },
    )
    if args.cpu_preflight:
        return
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if torch.cuda.device_count() != args.tp:
        raise RuntimeError("visible GPU count differs from requested pure TP")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=kwargs["tokenizer_revision"], local_files_only=True)
    corpus = (root / "input.txt").read_text()
    prompts = {}
    for case, lengths in CASES.items():
        for i, n in enumerate(lengths):
            passage = f"Independent request {case}, item {i}.\n" + corpus
            tokens = tokenizer.encode(passage * 300, add_special_tokens=True)
            if len(tokens) < n:
                raise RuntimeError("original text does not provide enough actual tokenizer tokens")
            prompts[(case, i)] = tokens[:n]
    llm = LLM(**kwargs)
    try:
        install_request_id_witness(llm.llm_engine.input_processor, args.output)
        save(
            args.output / "effective-native-config.json",
            {
                "config": str(llm.llm_engine.vllm_config),
                "cache_config": str(llm.llm_engine.vllm_config.cache_config),
                "scheduler_config": str(llm.llm_engine.vllm_config.scheduler_config),
                "compilation_config": str(llm.llm_engine.vllm_config.compilation_config),
            },
        )
        installed = llm.collective_rpc(
            "install_repair_qualification_observer", timeout=300, args=(str(args.output), expected, args.runtime)
        )
        save(args.output / "worker-installation.json", installed)
        results = []
        for case, lengths in CASES.items():
            for rep in range(2):
                inputs = [{"prompt_token_ids": prompts[(case, i)]} for i in range(len(lengths))]
                outputs = generate_cohort(
                    llm,
                    inputs,
                    SamplingParams(temperature=0, seed=0, max_tokens=32, min_tokens=32, ignore_eos=True, logprobs=5),
                    args.output,
                    case,
                    rep,
                )
                if len(outputs) != len(inputs):
                    raise RuntimeError("native Engine did not complete every actual request")
                for i, out in enumerate(outputs):
                    tokens = prompts[(case, i)]
                    if out.prompt_token_ids != tokens or not out.finished or len(out.outputs) != 1:
                        raise RuntimeError("native final request differs from supplied prompt")
                    answer = out.outputs[0]
                    if len(answer.token_ids) != 32:
                        raise RuntimeError("native Engine did not produce32 greedy output tokens")
                    logprobs = (
                        None
                        if answer.logprobs is None
                        else [
                            {
                                str(token): {
                                    "logprob": float(lp.logprob),
                                    "rank": lp.rank,
                                    "decoded_token": lp.decoded_token,
                                }
                                for token, lp in position.items()
                            }
                            for position in answer.logprobs
                        ]
                    )
                    row = {
                        "case": case,
                        "repetition": rep,
                        "item": i,
                        "request_id": out.request_id,
                        "prompt_token_ids": tokens,
                        "output_token_ids": list(answer.token_ids),
                        "text": answer.text,
                        "logprobs": logprobs,
                        "finish_reason": answer.finish_reason,
                    }
                    results.append(row)
                    with (args.output / "outputs.jsonl").open("a") as f:
                        f.write(json.dumps(row, sort_keys=True) + "\n")
        completed = llm.collective_rpc("finish_repair_qualification_observer", timeout=300)
        save(args.output / "worker-completion.json", completed)
        if __package__:
            from .validate import validate_native
        else:
            from validate import validate_native

        evidence = validate_native(args.output, args.tp, args.mode, args.policy, args.runtime)
        save(
            args.output / "native-receipt.json",
            {
                "status": "native_requests_and_histories_verified",
                "runtime_kind": args.runtime,
                "mode": args.mode,
                "policy": args.policy,
                "checkpoint": args.checkpoint,
                "tp": args.tp,
                "evidence": evidence,
                "correctness_comparison": "NOT_EVALUATED",
                "accuracy_acceptance": "NOT_EVALUATED",
            },
        )
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
