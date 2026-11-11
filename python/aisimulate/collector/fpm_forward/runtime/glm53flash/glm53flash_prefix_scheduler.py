# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified code derived from ai-dynamo/dynamo (Apache-2.0, https://github.com/ai-dynamo/dynamo),
# components/src/dynamo/vllm/instrumented_scheduler.py at three immutable revisions:
# - 54960177085413259859c88bd34ed0734d4c2ea9: InstrumentedScheduler benchmark state machine
#   (requests injected into and scheduled by the parent vLLM scheduler);
# - 99dae1f53e5a0534c274223ff9daa97f0b1fc2ea (shipped as ai-dynamo 1.5.0.dev20260917): prefix-cache
#   real-seed staging (_bench_realseed_stage_point / _bench_realseed_pending_step: seed shot,
#   untimed warm shot, validated measured shot), ported onto the 54960177 overlay;
# - 9abd2ef0d908c4bb323e1b8956ea76a5772a1d41 (release/1.7.0; the method is unchanged since the vLLM v0.31.0
#   bump 395f02405c1dd1835ab73a39f7a5984ae5fd4552): queued-request classification over vLLM's ``waiting`` +
#   ``kv_holding_waiting`` queues (_compute_queued), overriding the 54960177 method that reads the removed
#   ``Scheduler.skipped_waiting``.
# Modified: GLM-5.3-Flash default-serving hybrid state (prefix caching on, Mamba "align" mode),
# seeded token inputs, align-4 prefill chunk-start checks, repeated measurements and compact
# deferred evidence. Upstream license: LICENSE.
# Runtime base: ai-dynamo 1.7.0.dev20261109 (scheduler file ai-dynamo/dynamo@9abd2ef0d908c4bb323e1b8956ea76a5772a1d41).
"""GLM-5.3-Flash native FPM producer on the default serving configuration.

Timing is the unchanged Dynamo ``InstrumentedScheduler`` FPM ``wall_time``:
prefill = schedule() end to update_from_output() arrival of the measured step;
steady decode = consecutive output arrivals (second decode step). This module
never builds a ``SchedulerOutput``: every request -- prefix seed, warmup and
measured -- is added to the native waiting queue and scheduled by vLLM's own
scheduler, so prefix-cache lookup, Mamba align-mode chunking, KDA/conv, MLA
and IndexPool state handling are the serving code paths.

State construction (prefix caching on):

* prefill ``(new_i, prefix_i)``: one seed request per slot computes exactly
  ``prefix_i`` real tokens (cached by the native prefix cache); each repetition
  then submits ``prefix_i`` + a repetition-unique suffix, which must hit exactly
  ``prefix_i`` (validated from the native measured FPM, never assumed).
* decode ``context_i`` (the true context, not moved): the prompt is
  ``context_i - 1`` tokens and a seed request computes its first
  ``4 * floor((context_i - 2) / 4)`` tokens, so the measured shot's prefill chunk
  starts on the 4-token grid (only its final chunk may end off the grid); each
  repetition submits the prompt with ``max_tokens = 3`` and the second pure
  decode step (all B requests, KV read ``sum(context_i)``) is the measured
  steady sample.

No host work is added between steps of a repetition: the only per-step work is
the native Dynamo FPM bookkeeping plus a reference to the native
``cudagraph_stats`` object and of the native ``SchedulerOutput``. Validation,
medians and evidence serialization run after a repetition's requests have
finished, and the compact evidence record of a point (prompt specifications,
not token arrays) is written after all of its repetitions.

Stock vLLM v0.31.0 does not complete IndexPool entries at a cached/chunked
prefill start that is not a multiple of the pool width (4), and the sources do
not show that v0.33.0 removed the defect. Every prefill
prefix and new-token length and every decode seed is therefore a multiple of 4, and every scheduled
prefill chunk start (seed, warmup and measured shots) is recorded from the
native ``SchedulerOutput`` and checked after its stage; an unaligned start
fails the run instead of being measured.

Inputs are the seeded token stream of ``collector.glm53flash_attention_tokens``
(seed 53, ordinary BPE vocabulary of the pinned tokenizer), not a text corpus.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict
from itertools import chain
from pathlib import Path

import dynamo.vllm.instrumented_scheduler as native
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import get_tokenizer
from vllm.v1.request import Request, RequestStatus

DYNAMO_SHA = "54960177085413259859c88bd34ed0734d4c2ea9"
# The scheduler file that runs, which is not derived code: the ai-dynamo 1.7.0.dev20261109 build carries it.
DYNAMO_RUNTIME_SHA = "9abd2ef0d908c4bb323e1b8956ea76a5772a1d41"
DYNAMO_RUNTIME_DISTRIBUTION = "ai-dynamo 1.7.0.dev20261109"
REAL_SEED_SOURCE = "ai-dynamo/dynamo@99dae1f53e5a0534c274223ff9daa97f0b1fc2ea instrumented_scheduler._bench_realseed_*"
QUEUED_PORT_SOURCE = "ai-dynamo/dynamo@9abd2ef0d908c4bb323e1b8956ea76a5772a1d41 instrumented_scheduler._compute_queued"
VLLM_SHA = "4ba787ec9959e566d3b587a796dff07673324da5"
# Pure-TP sizes accepted for both checkpoints. Whether a size fits a system is
# decided by the native runtime (weight load and KV allocation fail observably),
# not by this producer.
TP_SIZES = (1, 2, 4, 8)
# IndexPool width: stock v0.31.0 requires pool-aligned prefill chunk starts. The rule stays for v0.33.0, whose sources
# do not show the defect removed.
CHUNK_ALIGN = 4
STATE_PROTOCOL = "glm53flash_prefix_cache_real_seed_v1"
INPUT_SOURCE = "seeded_tokens"
TIMING_BOUNDARY = "dynamo_vllm_instrumented_scheduler_fpm_wall_time"
WARMUP_REPEATS = 5
MEASUREMENT_REPEATS = 10
# The reduction of the measured repetitions: the upper median of the wall times of every repetition after the first,
# which is Dynamo's adjacent_upper_median (ai-dynamo/dynamo@9abd2ef0d9 instrumented_scheduler.py:7445-7460). The merge
# in the engine accepts no other median (worker_factory.py:432-444). Nine repetitions are an odd count, so the value is
# their median. The collector's measurement evidence reader keeps the label and the three values before it, and a test
# keeps them equal.
REDUCTION = "upper_median_after_first_repetition"
ESTIMATE_METHOD = "adjacent_upper_median"
MAX_REJECTED_REPETITIONS = 10
MAX_BATCH = 32
MAX_CONTEXT = 131072
# The context headroom that the producer has always required, in positions beyond the deepest measured context.
CONTEXT_HEADROOM = 7
MAX_NEW = 8192
STAGE_TIMEOUT_SECONDS = 1800.0
PREFILL_REAL_SEED_REASON = "prefill_real_seed"
DECODE_REAL_KV_REASON = "kvwarm_real_kv"


def pure_tensor_parallel(parallel, speculative_config) -> bool:
    """Whether a vLLM parallel configuration is the collected pure-TP layout."""
    return (
        parallel.tensor_parallel_size in TP_SIZES
        and parallel.pipeline_parallel_size == 1
        and parallel.data_parallel_size == 1
        and not parallel.use_ubatching
        and parallel.prefill_context_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and not parallel.enable_expert_parallel
        and speculative_config is None
    )


def glm_checkpoint_dir(model: str, revision: str) -> Path:
    """Folder of the loaded checkpoint. It holds the ``config.json`` and ``tokenizer.json`` that this run reads.

    A folder launch is that folder. A launch by Hub ID keeps the ID in the vLLM model configuration, so the files
    come from the local Hugging Face cache at the loaded revision. The lookup downloads nothing.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("DYN_FPM_TOKENIZER_REVISION must be the 40-hex revision of the loaded checkpoint")
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(model, "config.json", revision=revision)
    if not isinstance(cached, str):
        # None is a file that the cache does not know. A file that is known to be absent comes back as a sentinel.
        raise ValueError(
            "GLM-5.3-Flash checkpoint files are absent from the local Hugging Face cache for the loaded revision"
        )
    return Path(cached).parent


def glm_pinned_model(raw_config: dict, revision: str) -> tuple[str, str]:
    """The pinned checkpoint of the loaded configuration, as ``(model, pinned revision)``.

    The pins name the checkpoints that the profiles were collected on. Two rules apply to ``revision``, the revision
    that was loaded. A revision that is a pin names its model, so the loaded configuration must equal the configuration
    of that model. A revision label that contradicts the loaded files would select another model without notice.
    Any other revision of the same files has the same configuration, so it is accepted when its configuration equals
    the configuration of a pinned model.
    """
    from aisimulate_core.sdk.glm53flash import MODEL_REVISIONS
    from aisimulate_core.sdk.utils import get_model_config_from_model_path

    models_by_pin = {pinned_revision: model for model, pinned_revision in MODEL_REVISIONS.items()}
    if revision in models_by_pin:
        model = models_by_pin[revision]
        if get_model_config_from_model_path(model)["raw_config"] != raw_config:
            raise ValueError(
                f"GLM-5.3-Flash checkpoint at pinned revision {revision} does not match the configuration of {model}"
            )
        return model, revision
    for model, pinned_revision in MODEL_REVISIONS.items():
        if get_model_config_from_model_path(model)["raw_config"] == raw_config:
            return model, pinned_revision
    raise ValueError("loaded GLM-5.3-Flash checkpoint matches no pinned model configuration")


def glm_context_policy(max_model_len: int, measured: int) -> dict:
    """Context limits of one run. The runtime context is the configured ``max_model_len``.

    The runtime context must be at least the measured context plus ``CONTEXT_HEADROOM`` positions, the headroom that
    the producer has always required. A larger runtime context also meets this rule, so a run can configure the
    context that a deployment serves and measure a lower limit.
    """
    if max_model_len < measured + CONTEXT_HEADROOM:
        raise ValueError("GLM vLLM runtime context must be at least the measured context plus seven internal positions")
    return {
        "measured_context_limit": measured,
        "runtime_context_length": max_model_len,
        "native_admission_headroom": CONTEXT_HEADROOM,
    }


def token_slice(pool: list[int], length: int, offset: int) -> list[int]:
    """Rotate and repeat the seeded token stream deterministically."""
    if not pool or length < 0:
        raise ValueError("non-empty token pool and nonnegative length required")
    size = len(pool)
    start = offset % size
    repeats, tail = divmod(start + length, size)
    stream = pool * repeats + pool[:tail]
    return stream[start : start + length]


def prompt_spec(segments: list[tuple[int, int]]) -> list[list[int]]:
    """Compact, exactly reconstructible prompt: ``[[offset, length], ...]``."""
    return [[int(offset), int(length)] for offset, length in segments]


def build_prompt(pool: list[int], spec: list[list[int]]) -> list[int]:
    tokens: list[int] = []
    for offset, length in spec:
        tokens.extend(token_slice(pool, length, offset))
    return tokens


def prefill_chunk_starts(scheduler_output, prompt_lengths: dict[str, int]) -> list[list]:
    """``[[request_id, start, tokens], ...]`` of every prefill chunk in one native step.

    Starts are the native ``num_computed_tokens`` at schedule time
    (``NewRequestData`` / ``CachedRequestData``); a chunk is a prefill chunk when
    it starts before the prompt end. Requests outside this producer are ignored.
    """
    scheduled = scheduler_output.num_scheduled_tokens
    rows = [(new.req_id, new.num_computed_tokens) for new in scheduler_output.scheduled_new_reqs]
    cached = scheduler_output.scheduled_cached_reqs
    rows.extend(zip(cached.req_ids, cached.num_computed_tokens, strict=True))
    chunks = []
    for req_id, start in rows:
        prompt = prompt_lengths.get(req_id)
        if prompt is not None and start < prompt:
            chunks.append([req_id, int(start), int(scheduled[req_id])])
    return chunks


_DECODE_CONTEXTS: dict | None = None


def explicit_decode_contexts() -> dict[tuple[int, int], list[int]]:
    """Frozen per-request decode contexts keyed by ``(batch_size, total_kv_read_tokens)``.

    Dynamo's explicit decode points carry totals only and split them evenly
    (``_balanced_partition``). Executed align-4 geometry splits on the 4-token
    grid instead, so the frozen split is supplied next to the points file
    (``DYN_FPM_GLM53FLASH_DECODE_CONTEXTS``: ``[[batch, total, [contexts...]], ...]``).
    """
    global _DECODE_CONTEXTS
    if _DECODE_CONTEXTS is None:
        path = os.environ.get("DYN_FPM_GLM53FLASH_DECODE_CONTEXTS")
        table = {}
        for batch, total, contexts in json.loads(Path(path).read_text()) if path else []:
            key = (int(batch), int(total))
            if key in table or len(contexts) != batch or sum(contexts) != total or min(contexts) < 1:
                raise ValueError(f"invalid or duplicate explicit decode contexts for {key}")
            table[key] = [int(c) for c in contexts]
        _DECODE_CONTEXTS = table
    return _DECODE_CONTEXTS


def scheduled_dict(metrics) -> dict:
    return {
        "num_prefill_requests": int(metrics.num_prefill_requests),
        "sum_prefill_tokens": int(metrics.sum_prefill_tokens),
        "sum_prefill_kv_tokens": int(metrics.sum_prefill_kv_tokens),
        "num_decode_requests": int(metrics.num_decode_requests),
        "sum_decode_kv_tokens": int(metrics.sum_decode_kv_tokens),
    }


class Glm53FlashPrefixSeedScheduler(native.InstrumentedScheduler):
    # ------------------------------------------------------------------
    # Initialization and identity
    # ------------------------------------------------------------------
    def _bench_init(self, config):
        self._glm_point = None
        self._glm_stage = None
        self._glm_stage_deadline = 0.0
        self._glm_rows = None
        self._glm_repetitions = []
        self._glm_rejected = []
        self._glm_rep_index = 0
        self._glm_rep_record = None
        self._glm_rep_dispatches = []
        self._glm_rep_scheduled = []
        self._glm_step_outputs = []
        self._glm_prompt_lengths = {}
        self._glm_seed = None
        self._glm_point_records = {}
        self._glm_evidence_count = 0
        self._glm_evidence_digest = hashlib.sha256()
        self._glm_request_set = f"{os.environ.get('FPM_RUN_ID', 'glm53flash')}-{uuid.uuid4().hex}"
        self._glm_tokens = None
        self._glm_input = None
        self._glm_identity = None
        measured = int(os.environ.get("DYN_FPM_GLM53FLASH_MEASURED_CONTEXT", MAX_CONTEXT))
        if not 1 <= measured <= MAX_CONTEXT:
            raise ValueError("GLM measured context limit must be between 1 and 131072")
        self._glm_context_policy = glm_context_policy(config.model_config.max_model_len, measured)
        super()._bench_init(config)
        if not self._bench_active:
            raise ValueError("GLM-5.3-Flash prefix-seed producer requires native benchmark mode")
        if config.model_config.enforce_eager:
            raise ValueError("GLM-5.3-Flash FPM requires the native CUDA graph policy")
        if not config.observability_config.cudagraph_metrics:
            raise ValueError("GLM-5.3-Flash FPM requires actual CUDA graph dispatch metrics")
        cache = config.cache_config
        if not cache.enable_prefix_caching:
            raise ValueError("default-serving GLM FPM requires native prefix caching (vLLM default)")
        self._glm_serving = {
            "enable_prefix_caching": bool(cache.enable_prefix_caching),
            "mamba_cache_mode": getattr(cache, "mamba_cache_mode", None),
            "prefix_match_unit": getattr(cache, "prefix_match_unit", None),
            "hash_block_size": int(self._bench_hash_block_size),
            "block_size": int(self.block_size),
            "need_mamba_block_aligned_split": bool(getattr(self, "need_mamba_block_aligned_split", False)),
            "cache_dtype": getattr(cache, "cache_dtype", None),
            "async_scheduling": bool(getattr(config.scheduler_config, "async_scheduling", False)),
            # Whether vLLM 0.33.0 can save a prefill checkpoint inside the last chunk decides the planned points. It
            # holds the alignment of the first Mamba group only with checkpoint blocks (scheduler.py:437-449).
            "mamba_prefill_checkpoint": {
                "blocks": bool(self.mamba_has_prefill_checkpoint_blocks),
                "alignment": (
                    int(self.mamba_prefill_checkpoint_alignment) if self.mamba_has_prefill_checkpoint_blocks else None
                ),
            },
        }
        expected_unit = os.environ.get("DYN_FPM_GLM53FLASH_PREFIX_MATCH_UNIT")
        if expected_unit is not None and self._glm_serving["hash_block_size"] != int(expected_unit):
            raise ValueError("native prefix-match unit differs from the frozen campaign identity")
        if self._glm_serving["async_scheduling"]:
            raise ValueError("GLM FPM campaign freezes synchronous scheduling (--no-async-scheduling)")
        parallel = config.parallel_config
        self._glm_tp_size = parallel.tensor_parallel_size
        revision = os.environ.get("DYN_FPM_TOKENIZER_REVISION", "")
        if not pure_tensor_parallel(parallel, config.speculative_config):
            raise ValueError("GLM-5.3-Flash requires pure TP1/2/4/8, DP/PP/CP 1, no EP/ubatching/speculation")
        if self.connector is not None or self.ec_connector is not None:
            raise ValueError("GLM-5.3-Flash FPM forbids KV/encoder connectors")
        if self._bench_explicit_points is None:
            raise ValueError("GLM-5.3-Flash requires a frozen explicit sampling manifest")
        if self._bench_config.warmup_iterations != 0:
            raise ValueError("native global warmup must be zero; the producer runs five warmups per point")
        if not self._bench_config.collect_imbalanced:
            raise ValueError("frozen executed geometries require DYN_FPM_BENCH_COLLECT_IMBALANCED=1")
        from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, execution_identity
        from aisimulate_core.sdk.glm53flash import Glm53FlashConfig
        from aisimulate_core.sdk.utils import get_model_config_from_model_path

        # vLLM loads the weights of the revision in its model configuration, and this producer reads the configuration
        # and the tokenizer of DYN_FPM_TOKENIZER_REVISION. A launch by Hub ID without --revision loads the default
        # branch.
        loaded = config.model_config.revision
        if loaded is not None and loaded != revision:
            raise ValueError(
                f"GLM-5.3-Flash launch revision {loaded!r} differs from DYN_FPM_TOKENIZER_REVISION {revision!r}"
            )
        checkpoint_dir = glm_checkpoint_dir(config.model_config.model, revision)
        raw_config = get_model_config_from_model_path(str(checkpoint_dir))["raw_config"]
        Glm53FlashConfig.from_text_config(raw_config["text_config"])
        if (
            config.model_config.architecture != "Glm5NextForConditionalGeneration"
            or config.model_config.hf_config.model_type != "glm5_next"
        ):
            raise ValueError("loaded model must be GLM-5.3-Flash")
        self._glm_identity = dict(
            zip(
                EXECUTION_COLUMNS,
                execution_identity(raw_config, backend="vllm", input_modality="text"),
                strict=True,
            )
        )
        pinned_model, pinned_revision = glm_pinned_model(raw_config, revision)
        if config.scheduler_config.max_num_batched_tokens % CHUNK_ALIGN:
            raise ValueError("the native token budget must be a multiple of the IndexPool width (4)")
        from collector import glm53flash_attention_tokens as seeded

        tokenizer_json = checkpoint_dir / "tokenizer.json"
        tokenizer_sha = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
        if tokenizer_sha != seeded.TOKENIZER_JSON_SHA256:
            raise ValueError(f"tokenizer.json {tokenizer_sha} is not the pinned GLM-5.3-Flash tokenizer")
        tokenizer = get_tokenizer(
            config.model_config.tokenizer,
            tokenizer_mode=config.model_config.tokenizer_mode,
            trust_remote_code=config.model_config.trust_remote_code,
            revision=revision,
        )
        low, high = seeded.ordinary_vocab_range(tokenizer)
        if (low, high) != (seeded.VOCAB_LOW, seeded.VOCAB_HIGH):
            raise ValueError(f"tokenizer ordinary vocabulary {(low, high)} differs from the pinned range")
        count = MAX_CONTEXT + MAX_BATCH * seeded.REQUEST_STRIDE
        self._glm_tokens = seeded.generate(seeded.SEED, count, low, high)
        self._glm_input = {
            "source": INPUT_SOURCE,
            "generator": seeded.GENERATOR,
            "generator_version": seeded.GENERATOR_VERSION,
            "seed": seeded.SEED,
            "count": count,
            "vocab_low": low,
            "vocab_high_exclusive": high,
            "tokenizer_json_sha256": tokenizer_sha,
            "tokenizer_revision": revision,
            "pinned_model": pinned_model,
            "pinned_revision": pinned_revision,
            "checkpoint_dir": str(checkpoint_dir),
            "token_ids_sha256": hashlib.sha256(
                json.dumps(self._glm_tokens, separators=(",", ":")).encode()
            ).hexdigest(),
            "unique_token_count": len(set(self._glm_tokens)),
            "sampling": "prompt = concatenated [offset, length] rotations of the seeded stream (see prompt specs)",
            "chunk_start_alignment": CHUNK_ALIGN,
            "decode_contexts_sha256": (
                hashlib.sha256(Path(os.environ["DYN_FPM_GLM53FLASH_DECODE_CONTEXTS"]).read_bytes()).hexdigest()
                if os.environ.get("DYN_FPM_GLM53FLASH_DECODE_CONTEXTS")
                else None
            ),
        }

    @classmethod
    def _bench_decode_context_lengths(cls, total_kv_read_tokens, batch_size):
        explicit = explicit_decode_contexts()
        if not explicit:
            return super()._bench_decode_context_lengths(total_kv_read_tokens, batch_size)
        contexts = explicit.get((int(batch_size), int(total_kv_read_tokens)))
        if contexts is None:
            raise ValueError(
                f"decode point B={batch_size} KV={total_kv_read_tokens} has no frozen per-request contexts"
            )
        return list(contexts)

    def _bench_eager_warmup_points(self):
        return []  # five warmup repetitions run per exact geometry

    def _kvwarm_warm_eligible(self):
        return True

    def _kvwarm_prepare(self, mode):
        # ai-dynamo/dynamo@9abd2ef0d9 _bench_build_grid calls this at line 3260 to plan native warm-up stages and to
        # reorder the decode grid. This producer seeds every point through its own stages, so nothing is planned.
        return None

    def _bench_cache_fake_prefixes(self, *args, **kwargs):
        raise RuntimeError("synthetic prefix KV is forbidden for GLM-5.3-Flash")

    def _bench_inject_fake_decode(self, *args, **kwargs):
        raise RuntimeError("synthetic decode KV is forbidden for GLM-5.3-Flash")

    def _bench_new_request_counts_as_decode(self, req_id):
        # Every request here is an ordinary new request admitted by the native
        # scheduler; its first step is a (cache-hit) prefill.
        return False

    def _bench_blocks_per_req(self, num_tokens, *, has_cache_hit=False, apply_admission_cap=False):
        # Native capacity semantics of the pinned managers (hybrid, align mode).
        return sum(
            manager.get_num_blocks_to_allocate(
                request_id="__glm53flash_capacity_probe__",
                num_tokens=num_tokens,
                new_computed_blocks=[],
                total_computed_tokens=0,
                num_local_computed_tokens=0,
                num_tokens_main_model=num_tokens,
                apply_admission_cap=apply_admission_cap,
                # v0.33.0 KVCacheManager.allocate_slots (kv_cache_manager.py:529):
                # max(num_prompt_tokens, num_tokens - 1).
                prefill_end=num_tokens,
            )
            for manager in self.kv_cache_manager.coordinator.single_type_managers
        )

    def _bench_prefill_kv_read_lengths(self, total, batch, partition=None, rows=None):
        if rows is not None:
            lengths = [int(prefix) for _, prefix in rows]
        elif partition is not None:
            raise ValueError("GLM default-serving FPM uses explicit executed rows, not partitions")
        else:
            quotient, remainder = divmod(total, batch)
            lengths = [quotient + int(index < remainder) for index in range(batch)]
        if len(lengths) != batch or sum(lengths) != total or min(lengths) < 0:
            raise ValueError("prefix rows differ from requested token totals")
        return lengths

    def _bench_prefill_point_feasible(
        self, total_prefill_tokens, batch_size, total_kv_read_tokens, partition=None, rows=None
    ):
        if not (
            1 <= batch_size <= min(MAX_BATCH, self._bench_capacity_limit("max_num_running_reqs"))
            and batch_size
            <= total_prefill_tokens
            <= min(MAX_NEW, self._bench_capacity_limit("max_num_scheduled_tokens"))
        ):
            return False
        try:
            prefixes = self._bench_prefill_kv_read_lengths(total_kv_read_tokens, batch_size, partition, rows)
            queries = self._bench_prefill_new_token_lengths(total_prefill_tokens, batch_size, partition, rows)
        except ValueError:
            return False
        limit = min(self._glm_context_policy["measured_context_limit"], self._bench_capacity_limit("max_model_len"))
        if len(queries) != batch_size or sum(queries) != total_prefill_tokens or min(queries) < 1:
            return False
        lengths = [prefix + query for prefix, query in zip(prefixes, queries, strict=True)]
        if max(lengths) > limit:
            return False
        required = sum(self._bench_blocks_per_req(length) for length in lengths)
        return required <= self._bench_grid_usable_blocks(batch_size)

    def _bench_build_grid(self):
        built = self._bench_grid_built
        super()._bench_build_grid()
        if built:
            return
        unit = self._glm_serving["hash_block_size"]
        # vLLM 0.33.0 sets these launch attributes in Scheduler.__init__ (scheduler.py:218, :331, :342-345, :350,
        # :437-456). The align-mode rule holds only without EAGLE block drop, hash lookahead tokens and partial hash
        # hits, and with a scheduler block equal to the block of the cache configuration, which its chunk cut uses
        # (:572-591).
        for field, value, modeled in (
            ("use_eagle_block_drop", self.use_eagle_block_drop, not self.use_eagle_block_drop),
            ("num_hash_lookahead_tokens", self.num_hash_lookahead_tokens, self.num_hash_lookahead_tokens == 0),
            ("mamba_partial_cache_hit", self.mamba_partial_cache_hit, not self.mamba_partial_cache_hit),
            ("block_size", self.block_size, self.block_size == self.cache_config.block_size),
        ):
            if not modeled:
                raise ValueError(
                    f"GLM-5.3-Flash collection does not model this launch in its align-mode rule: {field}={value}"
                )
        # vLLM 0.33.0 saves a prefill checkpoint inside the last chunk only if every Mamba group holds a checkpoint
        # block (scheduler.py:445-449, :593-604).
        alignment = self.mamba_prefill_checkpoint_alignment if self.mamba_has_prefill_checkpoint_blocks else None
        hash_block = self.hash_block_size
        # vLLM 0.33.0 splits prefill chunks only in Mamba align mode (scheduler.py:432-434, :860-861, :1350-1351), on
        # the block of the cache configuration (:572).
        block = self.cache_config.block_size if self.need_mamba_block_aligned_split else None
        for point in self._bench_grid:
            if point.point_type == "prefill":
                prefixes, queries = self._glm_prefill_rows(point)
                if any(prefix % unit for prefix in prefixes):
                    raise ValueError(
                        f"benchmark_id={point.benchmark_id}: prefix not on the native prefix-match grid ({unit}); "
                        "the frozen manifest must carry the executed geometry"
                    )
                if any(length % CHUNK_ALIGN for length in (*prefixes, *queries)):
                    raise ValueError(
                        f"benchmark_id={point.benchmark_id}: prefix/new-token lengths must be multiples of "
                        f"{CHUNK_ALIGN} (kpool_align4); the frozen manifest must carry the executed geometry"
                    )
                for prefix, query in zip(prefixes, queries, strict=True):
                    stop = (
                        None
                        if block is None
                        else self._glm_align_stop(
                            prefix, prefix + query, block, hash_block=hash_block, checkpoint_alignment=alignment
                        )
                    )
                    if stop is not None:
                        # A repetition accepts one native FPM, so a split point would be rejected until the run fails.
                        raise ValueError(
                            f"benchmark_id={point.benchmark_id}: vLLM's Mamba align mode stops the prefill of {query} "
                            f"new tokens after a prefix of {prefix} at {stop}, a boundary of its {block}-token blocks, "
                            "so the point runs in more than one step and the producer measures one"
                        )
            else:
                contexts = self._bench_decode_context_lengths(point.total_kv_read_tokens, point.batch_size)
                if min(contexts) < 3:
                    raise ValueError("GLM real decode requires per-request context >= 3; no clamping")
                # The native decode check bounds a context by the runtime context, which can exceed the measured
                # context. The artifact states the measured context as kvwarm.max_context, and the prefill check bounds
                # prefix plus new tokens by it too.
                limit = self._glm_context_policy["measured_context_limit"]
                if max(contexts) > limit:
                    raise ValueError(
                        f"GLM-5.3-Flash decode context {max(contexts)} exceeds the measured context limit {limit}"
                    )
                # The native decode check bounds the batch by the running-request limit, which can exceed MAX_BATCH.
                # The artifact states MAX_BATCH as kvwarm.max_batch, and the prefill check bounds the batch by it too.
                if point.batch_size > MAX_BATCH:
                    raise ValueError(
                        f"GLM-5.3-Flash decode batch {point.batch_size} exceeds the producer limit {MAX_BATCH}"
                    )
                # A repetition prefills each prompt beyond its cached seed, and it is accepted only if its first decode
                # step holds every request. vLLM v0.33.0 schedules at most max_num_batched_tokens tokens in one step
                # (scheduler.py:722, :1295, :1570-1571). The suffixes of two or more requests above that budget spread
                # over several steps, so every repetition is rejected: the cached prefix cannot grow beyond the seed.
                # One request may prefill its suffix over several steps (scheduler.py:607-619): no other request can
                # decode before it, and _bench_should_record_scheduled records no step with a prefill request.
                budget = self.scheduler_config.max_num_batched_tokens
                prompts = [self._glm_decode_prompt(context) for context in contexts]
                total = sum(prompt - self._glm_decode_seed(prompt, unit) for prompt in prompts)
                if point.batch_size > 1 and total > budget:
                    context = contexts[0] if len(set(contexts)) == 1 else contexts
                    raise ValueError(
                        f"GLM-5.3-Flash decode point of {point.batch_size} requests at context {context} needs "
                        f"{total} suffix tokens in one step, above the token budget {budget}"
                    )

    @staticmethod
    def _glm_align_stop(start, end, block, *, hash_block, checkpoint_alignment):
        """Where vLLM 0.33.0 in Mamba align mode stops the last prefill chunk from ``start`` to the prompt end ``end``.

        It holds without EAGLE, without hash lookahead tokens and without partial hash hits. None means one step. The
        planner's ``glm53flash_points.glm_align_stop`` has the same body, because this module cannot import it.
        """
        # The last cacheable position is (end - 1) // block * block (scheduler.py:579-584,
        # kv_cache_interface.py:1169-1186). With a prefill checkpoint block of ``checkpoint_alignment``, a chunk that
        # passes is_mamba_prefill_checkpoint_valid saves the checkpoint inside it (kv_cache_interface.py:1189-1210) and
        # takes neither that stop nor the stop at the next boundary after a start inside a block (scheduler.py:593-606,
        # :644-651). The chunk stops at the first remaining stop strictly inside it (:661). The rule leaves out the
        # shared-prefix junction stop (:632-658). vLLM sets a junction only when the prefix hit of one cache group is
        # longer than the hit that the request uses (kv_cache_manager.py:319-325). The chunk then starts before the
        # planned prefix, and the producer rejects a repetition whose prefix differs from the planned one.
        last_cache_position = (end - 1) // block * block
        internal = (
            checkpoint_alignment is not None
            and start % hash_block == 0
            and -(-end // block) - 2 > (start - 1) // block
            and start + hash_block <= last_cache_position
            and start < last_cache_position < end
            and (last_cache_position - start) % checkpoint_alignment == 0
        )
        if internal:
            return None
        stops = (last_cache_position, (start // block + 1) * block if start % block else 0)
        return min((stop for stop in stops if start < stop < end), default=None)

    @staticmethod
    def _glm_decode_prompt(context):
        """The prompt precedes one sampled token and the measured second decode step."""
        return context - 1

    @staticmethod
    def _glm_decode_seed(prompt, unit):
        """Largest hit-aligned prefix strictly shorter than the prompt."""
        return (prompt - 1) // unit * unit

    def _glm_decode_step(self, context):
        """1-based pure decode step whose per-request KV read equals ``context``."""
        return context - self._glm_decode_prompt(context) + 1

    def _glm_prefill_rows(self, point):
        prefixes = self._bench_prefill_kv_read_lengths(
            point.total_kv_read_tokens, point.batch_size, point.partition, point.rows
        )
        queries = self._bench_prefill_new_token_lengths(
            point.total_prefill_tokens, point.batch_size, point.partition, point.rows
        )
        return prefixes, queries

    # ------------------------------------------------------------------
    # Recording: only the measured repetition, only the native FPM
    # ------------------------------------------------------------------
    def _bench_should_record_scheduled(self, scheduled):
        if self._glm_stage != "measure" or self._glm_point is None:
            return False
        if self._glm_point.point_type == "prefill":
            return scheduled.num_prefill_requests > 0
        return scheduled.num_decode_requests > 0 and scheduled.num_prefill_requests == 0

    def _update_from_output(self, scheduler_output, model_runner_output):
        result = super()._update_from_output(scheduler_output, model_runner_output)
        if self._glm_stage is not None and scheduler_output.total_num_scheduled_tokens > 0:
            # References only: chunk starts are checked after the stage.
            self._glm_step_outputs.append(scheduler_output)
            if self._glm_stage == "measure":
                self._glm_rep_dispatches.append(getattr(model_runner_output, "cudagraph_stats", None))
                self._glm_rep_scheduled.append(scheduler_output.total_num_scheduled_tokens)
        return result

    def _compute_queued(self):
        # Ported from QUEUED_PORT_SOURCE: vLLM v0.31.0 replaced Scheduler.skipped_waiting by kv_holding_waiting, and
        # v0.33.0 keeps both waiting queues (scheduler.py:232, :236).
        prefill = native.WelfordAccumulator()
        decode_kv = native.WelfordAccumulator()
        for request in chain(self.waiting, self.kv_holding_waiting):
            if request.status in (RequestStatus.PREEMPTED, RequestStatus.WAITING_FOR_REMOTE_KVS):
                decode_kv.add(request.num_computed_tokens)
            else:
                prefill.add(request.num_tokens)
        return native.QueuedRequestMetrics(
            num_prefill_requests=prefill.n,
            sum_prefill_tokens=prefill.s,
            var_prefill_length=prefill.variance(),
            num_decode_requests=decode_kv.n,
            sum_decode_kv_tokens=decode_kv.s,
            var_decode_kv_tokens=decode_kv.variance(),
        )

    def _glm_take_chunk_starts(self, stage):
        """Check every prefill chunk start of the finished stage; returns compact evidence."""
        steps = [prefill_chunk_starts(output, self._glm_prompt_lengths) for output in self._glm_step_outputs]
        self._glm_step_outputs = []
        unaligned = [chunk for chunks in steps for chunk in chunks if chunk[1] % CHUNK_ALIGN]
        if unaligned:
            raise RuntimeError(
                f"benchmark_id={self._glm_point.benchmark_id}: {stage} scheduled prefill chunks at starts not "
                f"divisible by {CHUNK_ALIGN} (stock vLLM IndexPool defect path): {unaligned[:8]}"
            )
        return [[[start, tokens] for _, start, tokens in chunks] for chunks in steps if chunks]

    # ------------------------------------------------------------------
    # State machine (every request is natively scheduled)
    # ------------------------------------------------------------------
    def schedule(self, throttle_prefills=False):
        # Native Dynamo returns an empty output during DECODE_SWEEP while
        # benchmark requests are active (its decode points use synthetic
        # outputs). Here decode requests are ordinary native requests, so
        # every phase advances the state machine and then schedules natively.
        if not (self._bench_active and self._bench_phase == native._BenchPhase.DECODE_SWEEP):
            return super().schedule(throttle_prefills)
        try:
            if self._bench_step() is not None:
                raise RuntimeError("GLM prefix-seed producer never builds a SchedulerOutput")
        except Exception as error:
            self._bench_abort(error)
            raise
        return self._schedule_and_record_time(throttle_prefills)

    def _bench_step_prefill(self):
        return self._glm_step("prefill")

    def _bench_step_decode(self):
        return self._glm_step("decode")

    def _glm_alive(self):
        return any(rid in self.requests for rid in self._bench_active_req_ids)

    def _glm_inject(self, prompts, *, max_tokens, salts, role):
        ids = []
        for prompt, salt in zip(prompts, salts, strict=True):
            req_id = f"{self._glm_request_set}-{role}-{self._bench_seq}"
            self._bench_seq += 1
            request = Request(
                request_id=req_id,
                prompt_token_ids=prompt,
                sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0),
                pooling_params=None,
                block_hasher=self._bench_block_hasher,
                cache_salt=salt,
            )
            self._glm_prompt_lengths[req_id] = len(prompt)
            self.add_request(request)
            self._bench_active_req_ids.add(req_id)
            ids.append(req_id)
        self._glm_stage_deadline = time.monotonic() + STAGE_TIMEOUT_SECONDS
        return ids

    def _glm_step(self, point_type):
        if self._glm_point is None:
            if self._bench_stop_at_timeout_boundary(point_type):
                return None
            point = self._bench_pop_next(point_type)
            if point is None:
                self._bench_phase = (
                    native._BenchPhase.DECODE_SWEEP
                    if point_type == "prefill" and self._bench_config.mode == "agg"
                    else native._BenchPhase.DONE
                )
                return None
            self._glm_begin_point(point)
            return None
        if self._glm_alive():
            if time.monotonic() >= self._glm_stage_deadline:
                raise RuntimeError(f"GLM {self._glm_stage} stage timed out; no fallback")
            return None
        self._bench_cleanup_requests()
        if self._glm_stage == "seed":
            self._glm_seed["completed_unix_ns"] = time.time_ns()
            self._glm_seed["prefill_chunks"] = self._glm_take_chunk_starts("seed")
            self._glm_start_repetition()
        elif self._glm_stage == "measure":
            if self._glm_finish_repetition():
                self._glm_finish_point()
            else:
                self._glm_start_repetition()
        return None

    def _glm_slot_offset(self, point, slot):
        return 131 * slot + 17 * point.benchmark_id

    def _glm_begin_point(self, point):
        self._glm_point_order = getattr(self, "_glm_point_order", -1) + 1
        self._glm_point = point
        self._bench_current_point = None
        self._bench_current_fpms = []
        self._glm_repetitions = []
        self._glm_rejected = []
        self._glm_rep_index = 0
        salt_base = f"{self._glm_request_set}-p{point.benchmark_id}"
        self._glm_salts = [f"{salt_base}-s{slot}" for slot in range(point.batch_size)]
        if point.point_type == "prefill":
            prefixes, queries = self._glm_prefill_rows(point)
            self._glm_rows = {"prefix": prefixes, "new": queries}
            seed_lengths = prefixes
            reason = PREFILL_REAL_SEED_REASON if point.total_kv_read_tokens > 0 else None
        else:
            contexts = self._bench_decode_context_lengths(point.total_kv_read_tokens, point.batch_size)
            unit = self._glm_serving["hash_block_size"]
            prompt_lengths = [self._glm_decode_prompt(context) for context in contexts]
            self._glm_rows = {
                "context": contexts,
                "prompt": prompt_lengths,
                "measured_decode_step": self._glm_decode_step(contexts[0]),
            }
            seed_lengths = [self._glm_decode_seed(length, unit) for length in prompt_lengths]
            reason = DECODE_REAL_KV_REASON
        if reason is not None and reason not in point.sample_reasons:
            point.sample_reasons.append(reason)
        self._glm_seed = {
            "lengths": list(seed_lengths),
            "specs": [],
            "request_ids": [],
            "started_unix_ns": time.time_ns(),
        }
        slots = [slot for slot, length in enumerate(seed_lengths) if length > 0]
        if not slots:
            self._glm_start_repetition()
            return
        specs = [prompt_spec([(self._glm_slot_offset(point, slot), seed_lengths[slot])]) for slot in slots]
        self._glm_seed["specs"] = specs
        self._glm_stage = "seed"
        self._glm_seed["request_ids"] = self._glm_inject(
            [build_prompt(self._glm_tokens, spec) for spec in specs],
            max_tokens=1,
            salts=[self._glm_salts[slot] for slot in slots],
            role="seed",
        )

    def _glm_repetition_prompts(self, point, attempt):
        specs, salts = [], []
        if point.point_type == "decode":
            for slot, length in enumerate(self._glm_rows["prompt"]):
                specs.append(prompt_spec([(self._glm_slot_offset(point, slot), length)]))
                salts.append(self._glm_salts[slot])
            return specs, salts
        for slot, (prefix, new) in enumerate(zip(self._glm_rows["prefix"], self._glm_rows["new"], strict=True)):
            base = self._glm_slot_offset(point, slot)
            if prefix:
                # Same seeded prefix; a repetition-unique suffix (shifted
                # rotation) so earlier repetitions cannot extend the hit.
                suffix_offset = base + prefix + 1 + 7 * (attempt + 1)
                specs.append(prompt_spec([(base, prefix), (suffix_offset, new)]))
                salts.append(self._glm_salts[slot])
            else:
                specs.append(prompt_spec([(base + 53 * attempt, new)]))
                salts.append(f"{self._glm_salts[slot]}-r{attempt}")
        return specs, salts

    def _glm_start_repetition(self):
        point = self._glm_point
        attempt = len(self._glm_repetitions) + len(self._glm_rejected)
        specs, salts = self._glm_repetition_prompts(point, attempt)
        self._bench_current_point = point
        self._bench_current_fpms = []
        self._bench_expected_fpms = self._glm_rows["measured_decode_step"] if point.point_type == "decode" else 1
        self._glm_rep_dispatches = []
        self._glm_rep_scheduled = []
        self._glm_rep_record = {
            "attempt": attempt,
            "prompt_specs": specs,
            "salts": salts,
            "started_unix_ns": time.time_ns(),
            "point_order": self._glm_point_order,
        }
        self._glm_stage = "measure"
        prompts = [build_prompt(self._glm_tokens, spec) for spec in specs]
        self._glm_rep_record["request_ids"] = self._glm_inject(
            prompts,
            max_tokens=self._bench_expected_fpms + 1 if point.point_type == "decode" else 1,
            salts=salts,
            role=f"r{attempt}",
        )

    def _glm_rejection(self, point, fpms):
        expected_count = self._glm_rows["measured_decode_step"] if point.point_type == "decode" else 1
        if len(fpms) != expected_count:
            return f"recorded {len(fpms)} native FPMs, expected {expected_count}"
        reason = self._bench_fpm_validation_failure(point, fpms[-1])
        if reason is not None:
            return reason
        scheduled = fpms[-1]["scheduled_requests"]
        if point.point_type == "prefill" and scheduled.get("num_decode_requests", 0):
            return "measured prefill step is mixed with decode"
        if point.point_type == "decode":
            first = fpms[0]["scheduled_requests"]
            if first.get("num_decode_requests") != point.batch_size or first.get("sum_decode_kv_tokens") != sum(
                self._glm_rows["prompt"]
            ):
                return "first decode step is not the all-B lockstep step"
        return None

    def _glm_finish_repetition(self) -> bool:
        point = self._glm_point
        fpms = [dict(fpm) for fpm in self._bench_current_fpms]
        dispatches = []
        for stats in self._glm_rep_dispatches:
            dispatches.append(None if stats is None else asdict(stats))
        record = {
            **self._glm_rep_record,
            "ended_unix_ns": time.time_ns(),
            "prefill_chunks": self._glm_take_chunk_starts(f"repetition {self._glm_rep_record['attempt']}"),
            "fpms": fpms,
            "scheduled_token_counts": list(self._glm_rep_scheduled),
            "dispatches": dispatches,
        }
        reason = self._glm_rejection(point, fpms)
        if reason is None and any(item is None for item in dispatches[-len(fpms) :]):
            reason = "missing actual graph dispatch receipt"
        self._bench_current_fpms = []
        if reason is not None:
            record["rejection_reason"] = reason
            self._glm_rejected.append(record)
            if len(self._glm_rejected) > MAX_REJECTED_REPETITIONS:
                raise RuntimeError(
                    f"benchmark_id={point.benchmark_id}: too many rejected repetitions; last reason: {reason}"
                )
            return False
        index = len(self._glm_repetitions)
        record["repetition"] = index
        record["role"] = "warmup" if index < WARMUP_REPEATS else "measurement"
        self._glm_repetitions.append(record)
        return len(self._glm_repetitions) == WARMUP_REPEATS + MEASUREMENT_REPEATS

    def _bench_measurement_protocol(self):
        # ai-dynamo/dynamo@9abd2ef0d9 instrumented_scheduler.py:4236-4281 names the synthetic content of the native
        # prompts and one launch per point. The prompts here are the seeded token stream, and each point holds the
        # measured repetitions of one launch, so independent_repetitions stays 1.
        return {
            **super()._bench_measurement_protocol(),
            "synthetic_content": INPUT_SOURCE,
            "producer": {
                "state_protocol": STATE_PROTOCOL,
                "warmup_repeats": WARMUP_REPEATS,
                "measurement_repeats": MEASUREMENT_REPEATS,
                "reduction": REDUCTION,
            },
        }

    def _glm_finish_point(self):
        point = self._glm_point
        measured = [record for record in self._glm_repetitions if record["role"] == "measurement"]
        # The retained FPM of a repetition is its only prefill FPM or its last (steady) decode FPM.
        retained = [copy.deepcopy(record["fpms"][-1]) for record in measured]
        self._bench_current_point = point
        self._bench_current_fpms = [dict(fpm) for fpm in measured[-1]["fpms"]]
        self._glm_stage = None
        saved = len(self._bench_results)
        self._bench_save_current_point()
        # A point that the native shape validation skips (instrumented_scheduler.py:7613-7628 at 9abd2ef0d9) is
        # recorded as skipped and saves no result, so it has no record to rewrite.
        if len(self._bench_results) != saved:
            self._glm_record_repetitions(point, retained)
        evidence = {
            "benchmark_id": point.benchmark_id,
            "point_type": point.point_type,
            "rows": self._glm_rows,
            "seed": self._glm_seed,
            "repetitions": self._glm_repetitions,
            "rejected_repetitions": self._glm_rejected,
        }
        raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
        path = Path(self._bench_config.output_path).with_suffix(".repetitions.jsonl")
        with path.open("xb" if self._glm_evidence_count == 0 else "ab") as stream:
            stream.write(raw + b"\n")
        self._glm_evidence_digest.update(raw + b"\n")
        self._glm_evidence_count += 1
        self._glm_point_records[point.benchmark_id] = hashlib.sha256(raw).hexdigest()
        self._glm_point = None
        self._glm_seed = None
        self._glm_rows = None
        self._glm_prompt_lengths = {}

    def _glm_record_repetitions(self, point, retained):
        """Make the saved point hold the upper median of the measured repetitions and the raw sample of each one.

        ai-dynamo/dynamo@9abd2ef0d9 instrumented_scheduler.py:7437-7500 copies the FPMs of the last measured repetition
        into ``benchmark_measurement.raw_fpms`` and keeps one FPM. With one attention-DP rank there is no synchronizer
        (:2944-2955), so the point result and the iteration group hold that same FPM list (:7508-7511, :7659-7674).
        Each raw sample keeps its own ``wall_time`` and ``benchmark_sample``, so its timing interval measures it. The
        merge of worker_factory.py:357-366 requires ``sample_index`` to be the position of the sample in ``raw_fpms``,
        so the record numbers the samples by position. The repetition sidecar keeps the index of each FPM in its own
        repetition. The forward indices rise already: each repetition runs after the one before it. The retained wall
        time is ``walls[len(walls) // 2]`` of the sorted samples 1 to 9, as :440-444 computes it.
        """
        result = self._bench_results[-1]
        group = self._bench_iteration_groups[-1]
        if (
            result.point.benchmark_id != point.benchmark_id
            or group["benchmark_id"] != point.benchmark_id
            or len(result.fpms) != 1
            or len(group["rank_results"]) != 1
            or group["rank_results"][0]["fpms"] is not result.fpms
        ):
            raise RuntimeError(f"benchmark_id={point.benchmark_id}: the native save did not keep one FPM of one rank")
        (fpm,) = result.fpms
        measurement = fpm["benchmark_measurement"]
        measurement["expected_internal_samples"] = MEASUREMENT_REPEATS
        for position, sample in enumerate(retained):
            sample["benchmark_sample"]["sample_index"] = position
        indices = list(range(1, MEASUREMENT_REPEATS))
        measurement["raw_fpms"] = retained
        measurement["estimate"] = {"method": ESTIMATE_METHOD, "raw_sample_indices": indices}
        walls = sorted(retained[index]["wall_time"] for index in indices)
        fpm["wall_time"] = walls[len(walls) // 2]
        group["wall_time"] = fpm["wall_time"]

    def _bench_cleanup_requests(self):
        super()._bench_cleanup_requests()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def _bench_write_results(self):
        destination = Path(self._bench_config.output_path)
        staging = destination.with_name(destination.name + ".native")
        # The runtime wrapper treats the final path as completion. Write the
        # native document elsewhere, enrich it, then publish atomically.
        self._bench_config.output_path = str(staging)
        try:
            super()._bench_write_results()
        finally:
            self._bench_config.output_path = str(destination)
        output = json.loads(staging.read_text())
        try:
            self._glm_enrich(output, destination)
        except Exception as error:
            # Publish the native document anyway; readers reject it.
            output["error"] = f"{output.get('error') or ''}; GLM enrichment failed: {error!r}".lstrip("; ")
            output["valid"] = output["usable"] = False
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(output, indent=2))
        os.replace(temporary, destination)
        staging.unlink()

    def _glm_enrich(self, output, destination):
        evidence_path = destination.with_suffix(".repetitions.jsonl")
        if self._glm_evidence_count:
            with evidence_path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != self._glm_evidence_digest.hexdigest():
                raise RuntimeError("GLM repetition evidence changed before publication")
        else:
            digest = None
        output["input_provenance"] = {
            **(self._glm_input or {}),
            "context_policy": self._glm_context_policy,
            "repetition_evidence_manifest": {
                "schema_version": 1,
                "file": evidence_path.name,
                "sha256": digest,
                "records": self._glm_evidence_count,
            },
        }
        output["context_policy"] = self._glm_context_policy
        output["execution_identity"] = self._glm_identity
        output["execution_mode"] = "native_graph_policy"
        output["observation_purpose"] = "fpm"
        output["ops_instrumented"] = False
        output["timing_boundary"] = TIMING_BOUNDARY
        output["serving_config"] = self._glm_serving
        output["kvwarm"] = {
            "enabled": True,
            "warm_eligible": True,
            "skip_reason": None,
            "method": "native_prefix_cache_real_seed",
            "state_protocol": STATE_PROTOCOL,
            "max_batch": MAX_BATCH,
            "max_context": self._glm_context_policy["measured_context_limit"],
        }
        from collector.glm53flash_runtime_identity import vllm_source_manifest_sha256

        output["producer"] = {
            "instrumentation_revision": DYNAMO_SHA,
            "runtime_scheduler_revision": DYNAMO_RUNTIME_SHA,
            "runtime_distribution": DYNAMO_RUNTIME_DISTRIBUTION,
            "real_seed_port_source": REAL_SEED_SOURCE,
            "queued_metrics_port_source": QUEUED_PORT_SOURCE,
            "vllm_package_version": __import__("vllm").__version__,
            "reviewed_scheduler_api_revision": VLLM_SHA,
            "runtime_source_manifest_sha256": vllm_source_manifest_sha256(
                __import__("vllm").__version__, Path(__file__).with_name("runtime-source-sha256.json")
            ),
            "overlay_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "context_policy_version": 1,
            "warmup_repeats": WARMUP_REPEATS,
            "measurement_repeats": MEASUREMENT_REPEATS,
        }
        for item in output["results"]:
            point = item["point"]
            item["kv_seed_regime"] = (
                "real_kv"
                if point["point_type"] == "decode"
                else "real_prefix"
                if point["total_kv_read_tokens"] > 0
                else "not_applicable"
            )
            item["repetition_evidence_sha256"] = self._glm_point_records.get(point["benchmark_id"])
        for group in output["iteration_groups"]:
            group["kv_seed_regime"] = "real_kv"


# Spawned processes may import this module before the native base finishes;
# publish only after class creation (see sitecustomize.py).
if os.environ.get("DYN_FPM_GLM53FLASH_PREFIX_SEED") == "1":
    native.InstrumentedScheduler = Glm53FlashPrefixSeedScheduler
