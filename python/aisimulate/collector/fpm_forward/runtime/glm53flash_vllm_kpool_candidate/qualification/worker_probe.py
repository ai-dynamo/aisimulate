# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original read-only wrappers around the pinned native V2 worker APIs.

See README.md for exact upstream APIs/source revision. No request, cache,
attention metadata, scheduling counter or native compute is replaced.
"""

import functools
import hashlib
import importlib.metadata
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def token_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def verify_runtime(expected, runtime):
    import vllm

    version = expected["versions"][runtime]
    if vllm.__version__ != version or importlib.metadata.version("vllm") != version:
        raise RuntimeError("actual loaded package/version metadata differs from frozen runtime")
    package = Path(vllm.__file__).resolve().parent.parent
    source_pins = dict(expected["source_pins"])
    source_pins[expected["helper_path"]] = expected["helper_sha256"][runtime]
    actual = {}
    for name, wanted in {**source_pins, **expected["native_binaries"]}.items():
        actual[name] = digest(package / name)
        if actual[name] != wanted:
            raise RuntimeError(f"actual loaded source/native binary differs: {name}")
    return {
        "version": version,
        "package_root": str(package),
        "loaded_files": actual,
        "candidate_wheel_sha256": expected["candidate_wheel_sha256"] if runtime == "candidate" else None,
    }


def request_identity_sources():
    """Verify the native parent APIs whose real assignments are observed."""
    import vllm

    package = Path(vllm.__file__).resolve().parent.parent
    sources = json.loads(Path(__file__).with_name("request-id-source.json").read_text())["sources"]
    result = {}
    for source in sources:
        actual = digest(package / source["path"])
        if actual != source["sha256"]:
            raise RuntimeError("native request identity source differs from pinned contract")
        result[source["path"]] = actual
    return result


def install_request_id_witness(processor, directory):
    """Call original native assignment, retaining its actual before/after IDs."""
    path = Path(directory) / "request-id-map.jsonl"
    path.touch(exist_ok=False)
    original = processor.assign_request_id

    @functools.wraps(original)
    def assign(request):
        external = request.request_id
        result = original(request)
        if request.external_req_id != external:
            raise RuntimeError("native request assignment did not preserve the external ID")
        with path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "external_request_id": external,
                        "native_request_id": request.request_id,
                        "prompt_sha256": token_digest(request.prompt_token_ids),
                        "original_assignment_returned": True,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        return result

    processor.assign_request_id = assign


class QualificationWorker:
    def install_repair_qualification_observer(self, directory, expected, runtime):
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        runner = self.model_runner
        if type(runner) is not GPUModelRunner or hasattr(runner, "_repair_qualification_probe"):
            raise RuntimeError("qualification requires the actual unobserved V2 runner")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        rank = get_tensor_model_parallel_rank()
        identity = verify_runtime(expected, runtime)
        index = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(index)
        hardware = {
            "name": torch.cuda.get_device_name(index),
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": prop.total_memory,
            "cuda_device_index": index,
            "uuid": str(getattr(prop, "uuid", "")),
        }
        if "GB300" not in hardware["name"].upper() or hardware["compute_capability"] != [10, 3]:
            raise RuntimeError("qualification requires actual GB300/sm103")
        with (root / f"worker-rank-{rank}.json").open("x") as f:
            json.dump(
                {
                    "tp_rank": rank,
                    "runtime": identity,
                    "hardware": hardware,
                    "native_runner": type(runner).__module__ + "." + type(runner).__name__,
                    "native_config": str(runner.vllm_config),
                    "settings": {
                        "tp_size": runner.parallel_config.tensor_parallel_size,
                        "ep_enabled": runner.parallel_config.enable_expert_parallel,
                        "async_scheduling": runner.scheduler_config.async_scheduling,
                        "prefix_caching": runner.cache_config.enable_prefix_caching,
                        "mamba_cache_mode": runner.cache_config.mamba_cache_mode,
                        "kv_dtype": runner.cache_config.cache_dtype,
                        "max_model_len": runner.max_model_len,
                        "long_prefill_token_threshold": runner.scheduler_config.long_prefill_token_threshold,
                        "max_num_batched_tokens": runner.scheduler_config.max_num_batched_tokens,
                        "speculative": runner.speculative_config is not None,
                        "enforce_eager": runner.model_config.enforce_eager,
                    },
                },
                f,
                indent=2,
            )
        trace = root / f"forward-rank-{rank}.jsonl"
        prompts = root / f"prompts-rank-{rank}.jsonl"
        trace.open("x").close()
        prompts.open("x").close()
        state = {"pending": None, "sampled": None, "count": 0, "prompts": set()}
        original_prepare = runner.prepare_inputs
        original_execute = runner.execute_model
        original_sample = runner.sample
        original_sample_tokens = runner.sample_tokens

        def append(path, value):
            with path.open("a") as f:
                f.write(json.dumps(value, sort_keys=True) + "\n")

        @functools.wraps(original_prepare)
        def prepare(scheduler_output, batch_state, batch_desc, *args, **kwargs):
            batch = original_prepare(scheduler_output, batch_state, batch_desc, *args, **kwargs)
            if state["pending"] is not None:
                raise RuntimeError("new native inputs before previous native GPU/sample completion")
            ids = list(batch.req_ids)
            lengths = list(map(int, batch.num_scheduled_tokens))
            prefixes = list(map(int, batch.num_computed_tokens_np))
            slots = list(map(int, batch.idx_mapping_np))
            tokens = list(map(int, batch.input_ids[: batch.num_tokens].detach().cpu().tolist()))
            seqs = list(map(int, batch.seq_lens[: len(ids)].detach().cpu().tolist()))
            if (
                batch.num_draft_tokens
                or lengths != [scheduler_output.num_scheduled_tokens[x] for x in ids]
                or seqs != [p + q for p, q in zip(prefixes, lengths, strict=True)]
                or len(tokens) != sum(lengths)
            ):
                raise RuntimeError("actual native V2 input geometry differs from native scheduler")
            records = []
            offset = 0
            for rid, q, p, slot, prefilling in zip(ids, lengths, prefixes, slots, batch.is_prefilling_np, strict=True):
                plen = int(runner.req_states.prompt_len.np[slot])
                prompt = list(map(int, runner.req_states.all_token_ids.gpu[slot, :plen].detach().cpu().tolist()))
                if rid not in state["prompts"]:
                    append(
                        prompts, {"request_id": rid, "prompt_token_ids": prompt, "prompt_sha256": token_digest(prompt)}
                    )
                    state["prompts"].add(rid)
                records.append(
                    {
                        "request_id": rid,
                        "prefix": p,
                        "query": q,
                        "native_slot": slot,
                        "is_prefilling": bool(prefilling),
                        "prompt_length": plen,
                        "prompt_sha256": token_digest(prompt),
                        "query_token_ids": tokens[offset : offset + q],
                    }
                )
                offset += q
            state["count"] += 1
            state["pending"] = {
                "invocation": state["count"],
                "tp_rank": rank,
                "requests": records,
                "native_mode": batch_desc.cg_mode.name,
                "actual_padded_tokens": batch.num_tokens_after_padding,
                "model_execute_returned": False,
                "gpu_completed": False,
            }
            return batch

        @functools.wraps(original_execute)
        def execute(scheduler_output, *args, **kwargs):
            dummy = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
            result = original_execute(scheduler_output, *args, **kwargs)
            if not dummy and scheduler_output.total_num_scheduled_tokens:
                if state["pending"] is None:
                    raise RuntimeError("real native execution bypassed observed InputBatch")
                state["pending"]["model_execute_returned"] = True
            return result

        @functools.wraps(original_sample)
        def sample(*args, **kwargs):
            result = original_sample(*args, **kwargs)
            if state["pending"] is not None:
                if state["sampled"] is not None:
                    raise RuntimeError("native sampler repeated before receipt completion")
                state["sampled"] = result[0].sampled_token_ids.detach().cpu().tolist()
            return result

        @functools.wraps(original_sample_tokens)
        def sample_tokens(*args, **kwargs):
            result = original_sample_tokens(*args, **kwargs)
            row = state["pending"]
            if row is not None:
                sampled = state["sampled"]
                if (
                    not row["model_execute_returned"]
                    or sampled is None
                    or len(sampled) != len(row["requests"])
                    or any(len(x) != 1 for x in sampled)
                ):
                    raise RuntimeError("native model/logits/sample completion lacks exact request correspondence")
                torch.cuda.synchronize()
                for req, values in zip(row["requests"], sampled, strict=True):
                    req["sampled_token_id"] = int(values[0])
                row["gpu_completed"] = True
                append(trace, row)
                state["pending"] = state["sampled"] = None
            return result

        runner.prepare_inputs = prepare
        runner.execute_model = execute
        runner.sample = sample
        runner.sample_tokens = sample_tokens
        runner._repair_qualification_probe = state
        return {"tp_rank": rank, "status": "installed_read_only_native_wrappers", "hardware": hardware}

    def finish_repair_qualification_observer(self):
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        state = self.model_runner._repair_qualification_probe
        torch.cuda.synchronize()
        if state["pending"] is not None or state["sampled"] is not None:
            raise RuntimeError("qualification ends with unfinished native execution")
        return {
            "tp_rank": get_tensor_model_parallel_rank(),
            "completed_native_forwards": state["count"],
            "requests": len(state["prompts"]),
        }
