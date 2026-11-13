# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SGLang collector helpers for runtime token limits and KV allocation."""

import importlib
import inspect
from collections.abc import Callable, Iterable
from contextlib import contextmanager


def kv_pool_capacity_tokens(model_runner) -> int | None:
    """Best-effort token capacity from the actual SGLang KV allocator."""
    allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
    if allocator is None:
        return None
    # Hybrid SWA allocators report min(full, swa) from available_size().
    # Logical sequence capacity follows the full-attention pool instead.
    if hasattr(allocator, "full_available_size"):
        try:
            return int(allocator.full_available_size())
        except Exception:
            return None
    if hasattr(allocator, "available_size"):
        try:
            return int(allocator.available_size())
        except Exception:
            return None
    for attr in ("max_total_num_tokens", "size", "pool_size"):
        value = getattr(allocator, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def swa_kv_pool_capacity_tokens(model_runner) -> int | None:
    """Best-effort token capacity of SGLang's hybrid SWA KV pool."""
    allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
    available_size = getattr(allocator, "swa_available_size", None)
    if not callable(available_size):
        return None
    try:
        return int(available_size())
    except Exception:
        return None


def required_kv_tokens(batch_size: int, seq_len: int, prefix_len: int, *, is_prefill: bool) -> int:
    """Return total KV tokens needed by a collector case."""
    return batch_size * (seq_len + prefix_len) if is_prefill else batch_size * seq_len


def kv_pool_page_size(model_runner) -> int:
    """Page size of the SGLang KV allocator (1 for token-level allocators)."""
    allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
    page = getattr(allocator, "page_size", None)
    return int(page) if isinstance(page, int) and page > 0 else 1


def required_kv_alloc_tokens(
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    page_size: int,
    *,
    is_prefill: bool,
) -> int:
    """KV tokens SGLang's paged ``alloc_extend`` must find free for this case.

    The naive :func:`required_kv_tokens` (``bs*(sl+prefix)``) only counts logical
    KV tokens, but SGLang's PAGED allocator rounds EACH request's KV span up to a
    full page independently. With a large page (e.g. 256) and many requests, even
    tiny ``seq_len`` forces ``batch_size`` whole pages: a bs=512 / sl=1 decode
    needs 512 pages = ``512 * 256`` slots, which can exceed the KV pool even
    though ``bs*sl`` is tiny — so ``alloc_extend`` fails with "Prefill out of
    memory" while the naive check passes. This returns that true paged
    requirement so such shapes are skipped BEFORE launching. With ``page_size==1``
    it equals the logical span plus the fresh decode token (a no-op for
    token-level page rounding).

    Decode also allocates one fresh token per request. When ``seq_len`` lands
    exactly on a page boundary, that token requires one additional page for
    every request.

    NOTE: this is the real allocation size only, NOT SGLang's eviction
    over-estimate (``extend_num_tokens + bs*page_size``). The collector clears
    the pool between shapes, so the cache is empty and nothing needs eviction;
    adding that slack would over-skip valid shapes (e.g. bs=256/sl<=256).
    """
    if batch_size <= 0:
        return 0
    per_req = required_kv_tokens(batch_size, seq_len, prefix_len, is_prefill=is_prefill) // batch_size
    if not is_prefill:
        per_req += 1
    pages_per_req = (per_req + page_size - 1) // page_size
    return batch_size * pages_per_req * page_size


def required_swa_kv_alloc_tokens(
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    page_size: int,
    window_size: int,
    *,
    is_prefill: bool,
) -> int:
    """Page-rounded hybrid SWA allocation needed by one collector shape."""
    if batch_size <= 0:
        return 0
    past_len = prefix_len if is_prefill else seq_len
    fresh_len = seq_len if is_prefill else 1
    window_start = max(0, past_len - window_size)
    window_start = (window_start // page_size) * page_size
    swa_tail_len = past_len - window_start
    rounded_tail = ((swa_tail_len + page_size - 1) // page_size) * page_size
    rounded_past = ((past_len + page_size - 1) // page_size) * page_size
    rounded_total = ((past_len + fresh_len + page_size - 1) // page_size) * page_size
    return batch_size * (rounded_tail + rounded_total - rounded_past)


# The floor of the request slots SGLang derives when max_running_requests is unset:
# max(min(int(token_capacity / context_len * 512), 4096), 2048)
# (KVCacheConfigurator.resolve_max_num_reqs, srt/mem_cache/kv_cache_configurator.py:3119-3132 at v0.5.23).
SGLANG_MIN_REQUEST_SLOTS = 2048


def request_slots(batch_sizes: Iterable[int]) -> int:
    """ServerArgs.max_running_requests for a module runner that times these batch sizes.

    Unset, SGLang derives the slots from the KV pool's token capacity (the formula
    above) and allocates req_to_token as (slots + 1) x (context_len + 4) int32
    (ReqToTokenPool, srt/mem_cache/memory_pool.py:388-408; _build_default_req_pool,
    kv_cache_configurator.py:1755-1777) outside the KV byte budget: only the
    1 - mem_fraction_static slack covers it (_profile_available_bytes, :2990-3060).
    A module dummy keeps two of the model's layers, so its pool holds many times the
    tokens serving's does and the estimate lands on 4096. For a 1M-token context that
    is 16 GiB, more than the slack of an 80 GB device, and a pool SGLang clamps to its
    profiled capacity then overcommits the device. The floor is what a server of the
    full model resolves wherever it holds at most four context windows of KV, and it
    is not below any decode CUDA-graph batch size (serving_memory_defaults), so the
    graph buckets stay SGLang's (memory_hook.py:182-188).
    """
    return max([SGLANG_MIN_REQUEST_SLOTS, *batch_sizes])


def sglang_dsa_mqa_logits_chunking_supported() -> bool:
    """Return whether the imported SGLang DSA/NSA indexer chunks MQA logits."""
    candidates = (
        "sglang.srt.layers.attention.dsa.dsa_indexer",
        "sglang.srt.layers.attention.nsa.nsa_indexer",
    )
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            indexer_cls = getattr(module, "Indexer", None)
            should_chunk = getattr(indexer_cls, "_should_chunk_mqa_logits", None)
            get_topk = getattr(indexer_cls, "_get_topk_ragged", None)
            if should_chunk is None or get_topk is None:
                continue
            source = "\n".join(
                (
                    inspect.getsource(should_chunk),
                    inspect.getsource(get_topk),
                )
            )
        except Exception:
            continue

        has_chunk_decision = "_should_chunk_mqa_logits" in source
        has_chunk_loop = "while start < q_offset" in source
        has_chunk_kernel = "logits_chunk" in source
        if has_chunk_decision and has_chunk_loop and has_chunk_kernel:
            return True
    # sglang>=0.5.21 restructured the contract: the free-memory row chunking moved
    # out of Indexer into dsa_indexer_kpool._mqa_logits_row_chunks (:85, driven by
    # mqa_logits_should_chunk / mqa_logits_rows_per_chunk / mqa_logits_budget_bytes)
    # and the SM100 batch>num_sms split lives in Indexer._get_topk_ragged's local
    # _chunked_fp8_paged_mqa_logits (dsa_indexer.py:1027). Same behaviour the
    # collector relies on (logits never exceed the free-memory budget), new shape.
    try:
        kpool = importlib.import_module("sglang.srt.layers.attention.dsa.dsa_indexer_kpool")
        indexer_mod = importlib.import_module("sglang.srt.layers.attention.dsa.dsa_indexer")
        indexer_cls = getattr(indexer_mod, "Indexer", None)
        row_chunks = getattr(kpool, "_mqa_logits_row_chunks", None)
        if indexer_cls is not None and row_chunks is not None and getattr(indexer_cls, "_get_topk_ragged", None):
            source = inspect.getsource(row_chunks)
            if "mqa_logits_should_chunk" in source and "mqa_logits_rows_per_chunk" in source:
                return True
    except Exception:
        pass
    raise RuntimeError("SGLang DSA MQA-logits chunking source contract was not detected")


def dsa_indexer_prefill_shape_is_supported(batch_size: int, seq_len: int) -> bool:
    """Validate dimensions without inferring the native forward mode.

    A one-token EXTEND remains prefill. SGLang 0.5.14's native TC runner
    bucket-pads it (49e384ce, prefill_cuda_graph_runner.py:480-524). At 0.5.21
    the collector times DSA prefill eagerly (prefill graphs off, as serving runs
    a one-token extend too: bucket 4 is more than twice its size,
    prefill_cuda_graph_runner.py:157,1286-1290), and the eager DSA prefill
    indexer illegal-accessed on that shape. FIXME(re-verify): admit seq_len=1
    once a 0.5.21 run shows the eager one-token extend no longer faults.
    """
    return batch_size > 0 and seq_len > 1


def required_prefill_extend_tokens(batch_size: int, seq_len: int) -> int:
    """Return the number of fresh prefill tokens in a collector case."""
    return batch_size * seq_len


def serving_memory_defaults(tp_size: int) -> tuple[int, int]:
    """(chunked_prefill_size, decode CUDA-graph max_bs) SGLang derives from GPU memory.

    The defaults of a server launched without either flag: the memory table of
    _handle_gpu_memory_settings (srt/arg_groups/memory_hook.py:89-180 at
    v0.5.23; MiB from get_device_memory_capacity, srt/utils/common.py:990-1017).
    ``tp_size`` is the server's tensor-parallel world, which moves the decode
    batch size on GPUs below 160 GiB. The max_running_requests cap (:182-188)
    does not apply: a default server leaves that limit unset.
    """
    from sglang.srt.utils.common import get_device_memory_capacity

    gpu_mem = get_device_memory_capacity()
    if gpu_mem is None:
        return 4096, 160
    if gpu_mem < 20 * 1024:
        return 2048, 8
    if gpu_mem < 35 * 1024:
        return 4096, 48 if tp_size < 4 else 160
    if gpu_mem < 60 * 1024:
        return 4096, 32 if tp_size < 4 else 160
    if gpu_mem < 160 * 1024:
        return 8192, 256 if tp_size < 4 else 512
    return 16384, 512


def decode_autotune_max_tokens(tp_size: int) -> int:
    """The token count of the one dummy forward SGLang's default warm-up autotunes in.

    Warm-up runs a decode-shaped dummy forward at the decode CUDA-graph max
    batch size under FlashInfer's autotuner (base_runner.py:279-285,
    decode_cuda_graph_runner.py:704-716, flashinfer_autotune.py:354-362); the
    extend-shaped pass at the prefill ceiling is opt-in
    (SGLANG_FLASHINFER_AUTOTUNE_EXTEND, srt/environ.py:1282;
    flashinfer_autotune.py:412-443).
    """
    return serving_memory_defaults(tp_size)[1]


def runtime_chunk_size(model_runner) -> int:
    server_args = getattr(model_runner, "server_args", None)
    sglang_chunk = getattr(server_args, "chunked_prefill_size", None) if server_args else None
    if not (isinstance(sglang_chunk, int) and sglang_chunk > 0) and server_args is not None:
        # sglang>=0.5.21: derived values are read through the resolving view
        # (arg_groups/model_override_base.py:120 resolving_view; one_batch.load_model
        # reads cfg.mem_fraction_static the same way), not off the raw record.
        try:
            from sglang.srt.arg_groups.model_override_base import resolving_view

            sglang_chunk = getattr(resolving_view(server_args), "chunked_prefill_size", None)
        except ImportError:
            pass
    if isinstance(sglang_chunk, int) and sglang_chunk > 0:
        return sglang_chunk
    raise RuntimeError("SGLang did not initialize server_args.chunked_prefill_size")


# A point the device cannot run is refused with a reason that starts with this
# classification, before anything is allocated for it.
KERNEL_LIMIT = "FIXME(kernel-limit)"


def over_chunk(shape: tuple, chunk: int) -> str | None:
    """Why serving would not run this (batch, seq_len, prefix_len) prefill as one forward, or None."""
    batch_size, seq_len, _prefix_len = shape
    tokens = required_prefill_extend_tokens(batch_size, seq_len)
    if tokens <= chunk:
        return None
    return (
        f"{KERNEL_LIMIT}: does not fit: {tokens} new tokens exceed the chunked_prefill_size={chunk} "
        "SGLang uses on this device, and serving chunks this prefill"
    )


def over_kv_pool(shape: tuple, page_size: int, capacity: int, is_prefill: bool) -> str | None:
    """Why this (batch, seq_len, prefix_len) point's KV cache cannot fit SGLang's KV pool, or None."""
    needed = required_kv_alloc_tokens(*shape, page_size, is_prefill=is_prefill)
    if needed <= capacity:
        return None
    return f"{KERNEL_LIMIT}: does not fit: the point needs {needed} KV tokens, the KV pool holds {capacity}"


def refuse(shapes: list, refused: dict, reason_for: Callable[[tuple], str | None]) -> None:
    """Record the reason of every point not yet refused that ``reason_for`` refuses."""
    for shape in shapes:
        if shape not in refused:
            reason = reason_for(shape)
            if reason is not None:
                refused[shape] = reason


def refusal_summary(refused: dict, total: int) -> str:
    """One classified line for the points this device refused (the subprocess's last error line)."""
    shown = "; ".join(
        f"b={b} s={s} prefix={p}: {reason.removeprefix(f'{KERNEL_LIMIT}: ')}"
        for (b, s, p), reason in list(refused.items())[:3]
    )
    more = f"; and {len(refused) - 3} more" if len(refused) > 3 else ""
    return f"{KERNEL_LIMIT}: {len(refused)}/{total} point(s) refused: {shown}{more}"


def kernel_limit_error(lines: list[str]) -> str | None:
    """The kernel-limit error a failed measurement subprocess ended on, or None.

    Points the device refused end the subprocess on one classified exception line,
    the first unindented line of its last traceback. torch.distributed prefixes
    each line of a rank's output with "[rankN]: ".
    """
    plain = [line.split("]: ", 1)[1] if line.startswith("[rank") and "]: " in line else line for line in lines]
    tracebacks = [index for index, line in enumerate(plain) if line.startswith("Traceback ")]
    if not tracebacks:
        return None
    error = next((line for line in plain[tracebacks[-1] + 1 :] if not line.startswith(" ")), "")
    return error.removeprefix("RuntimeError: ") if error.startswith(f"RuntimeError: {KERNEL_LIMIT}") else None


def chunked_alloc_extend(orig_alloc_extend: Callable, chunk_size: int) -> Callable:
    """Wrap ``alloc_extend`` using SGLang's runtime chunk size."""
    # Lazy: keep this module importable in torch-free unit-test environments.
    import torch

    def wrapped(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens):
        bs = prefix_lens.shape[0]
        if extend_num_tokens <= chunk_size or bs == 0:
            return orig_alloc_extend(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens)

        extend_per_req = (seq_lens_cpu - prefix_lens_cpu).tolist()
        chunk_size_per_req = max(1, chunk_size // bs)

        cur_prefix = prefix_lens.clone()
        cur_prefix_cpu = prefix_lens_cpu.clone()
        cur_last_loc = last_loc.clone()
        advanced = [0] * bs
        per_req_indices: list[list] = [[] for _ in range(bs)]

        while True:
            chunk_extends = [min(chunk_size_per_req, extend_per_req[i] - advanced[i]) for i in range(bs)]
            chunk_total = sum(chunk_extends)
            if chunk_total == 0:
                break

            chunk_tensor = torch.tensor(chunk_extends, dtype=cur_prefix_cpu.dtype)
            new_seq_cpu = cur_prefix_cpu + chunk_tensor
            new_seq = new_seq_cpu.to(seq_lens.device)

            indices = orig_alloc_extend(cur_prefix, cur_prefix_cpu, new_seq, new_seq_cpu, cur_last_loc, chunk_total)
            if indices is None:
                return None

            offset = 0
            for i in range(bs):
                n = chunk_extends[i]
                if n > 0:
                    req_chunk = indices[offset : offset + n]
                    per_req_indices[i].append(req_chunk)
                    offset += n
                    cur_last_loc[i] = req_chunk[-1]
                    advanced[i] += n
            cur_prefix = new_seq
            cur_prefix_cpu = new_seq_cpu

        final = [torch.cat(lst) for lst in per_req_indices if lst]
        if not final:
            return torch.empty((0,), dtype=torch.int64, device=prefix_lens.device)
        return torch.cat(final)

    return wrapped


@contextmanager
def temporarily_chunked_alloc_extend(model_runner, extend_num_tokens: int):
    """Patch SGLang ``alloc_extend`` while a synthetic collector batch is built."""
    chunk_size = runtime_chunk_size(model_runner)
    allocator = model_runner.token_to_kv_pool_allocator
    saved_alloc_extend = None
    if extend_num_tokens > chunk_size:
        saved_alloc_extend = allocator.alloc_extend
        allocator.alloc_extend = chunked_alloc_extend(saved_alloc_extend, chunk_size=chunk_size)

    try:
        yield
    finally:
        if saved_alloc_extend is not None:
            allocator.alloc_extend = saved_alloc_extend


def alloc_prefix_indices(model_runner, batch_size: int, prefix_len: int) -> list:
    """Allocate per-request prefix KV indices using SGLang's runtime chunk size."""
    # Lazy: see chunked_alloc_extend.
    import torch

    device = getattr(model_runner, "device", "cuda")
    if prefix_len <= 0:
        return [torch.empty((0,), dtype=torch.int64, device=device) for _ in range(batch_size)]

    allocator = model_runner.token_to_kv_pool_allocator
    alloc_swa_tail = getattr(allocator, "alloc_extend_swa_tail", None)
    window_size = getattr(getattr(model_runner, "model_config", None), "window_size", None)
    page_size = getattr(allocator, "page_size", None)
    if callable(alloc_swa_tail) and isinstance(window_size, int) and window_size > 0:
        if not isinstance(page_size, int) or page_size <= 1:
            raise RuntimeError("SGLang SWA tail allocation requires a paged KV allocator")
        window_start = max(0, prefix_len - window_size)
        window_start = (window_start // page_size) * page_size
        swa_tail_len = prefix_len - window_start
        per_request = []
        for _ in range(batch_size):
            prefix_lens_cpu = torch.zeros(1, dtype=torch.int64)
            seq_lens_cpu = torch.full((1,), prefix_len, dtype=torch.int64)
            prefix_lens = prefix_lens_cpu.to(device, non_blocking=True)
            seq_lens = seq_lens_cpu.to(device, non_blocking=True)
            last_loc = torch.full((1,), -1, dtype=torch.int64, device=device)
            indices = alloc_swa_tail(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                prefix_len,
                swa_tail_len,
            )
            if indices is None:
                raise RuntimeError(
                    f"failed to allocate SWA-tail prefix cache: prefix_len={prefix_len}, swa_tail_len={swa_tail_len}"
                )
            per_request.append(indices.contiguous())
        return per_request

    chunk_size = runtime_chunk_size(model_runner)
    alloc_extend = allocator.alloc_extend
    if batch_size * prefix_len > chunk_size:
        alloc_extend = chunked_alloc_extend(alloc_extend, chunk_size=chunk_size)

    prefix_lens_cpu = torch.zeros(batch_size, dtype=torch.int64)
    seq_lens_cpu = torch.full((batch_size,), prefix_len, dtype=torch.int64)
    prefix_lens = prefix_lens_cpu.to(device, non_blocking=True)
    seq_lens = seq_lens_cpu.to(device, non_blocking=True)
    last_loc = torch.full((batch_size,), -1, dtype=torch.int64, device=device)

    flat = alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        batch_size * prefix_len,
    )
    if flat is None:
        raise RuntimeError(f"failed to allocate prefix cache: batch_size={batch_size}, prefix_len={prefix_len}")
    return [flat[i * prefix_len : (i + 1) * prefix_len].contiguous() for i in range(batch_size)]
