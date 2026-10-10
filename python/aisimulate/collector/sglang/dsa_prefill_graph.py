# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention-only inputs for SGLang's own prefill compiler and graph runner.

Native API references: sgl-project/sglang at
49e384ce9d304648e9959666ecb8ce8cd98d0deb, python/sglang/srt:
model_executor/model_runner.py:2593-2766 and
model_executor/runner/prefill_cuda_graph_runner.py:111-318,370-427,765-941.
This adapter delegates compilation, capture, padding, metadata and replay to
that runtime. It excludes batch preparation and the model's other layers.
"""

import copy
from contextlib import contextmanager

import torch


def _release_owned_compile_hooks(hooks, previous_keys, module):
    """Release only native compile hooks closing over this adapter's module.

    SGLang 49e384ce compile.py registers a torch bytecode hook but discards
    its removal handle. Removing that owned entry matches PyTorch's native
    RemovableHandle.remove; unrelated hooks must survive this scope.
    """
    for key in set(hooks) - previous_keys:
        hook = hooks[key]
        code = getattr(hook, "__code__", None)
        if code is None:
            continue
        cells = dict(zip(code.co_freevars, hook.__closure__ or (), strict=True))
        if "module" in cells and cells["module"].cell_contents is module:
            del hooks[key]


def graph_token_bucket(model_runner, tokens: int) -> int | None:
    config = model_runner.server_args.cuda_graph_config.prefill
    if config.backend == "disabled":
        return None
    if config.backend != "tc_piecewise":
        raise ValueError(f"Unsupported SGLang module prefill graph backend: {config.backend}")
    return next((size for size in sorted(config.bs or []) if size >= tokens), None)


@contextmanager
def dsa_prefill_graph(model_runner, attention, forward_batch, hidden_states, zero_allocator, *, skip_indexer):
    from sglang.srt.distributed.device_communicators import pynccl_allocator
    from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import set_tc_piecewise_forward_context

    bucket = graph_token_bucket(model_runner, len(hidden_states))
    if bucket is None:
        raise ValueError("Batch is outside SGLang's prefill graph coverage")
    if attention.layer_id != 0:
        raise ValueError("Attention-only prefill graph requires collector test_layer=0")

    hidden = torch.randn((bucket, hidden_states.shape[1]), device=hidden_states.device, dtype=hidden_states.dtype)
    hidden[: len(hidden_states)].copy_(hidden_states)
    logits = torch.empty((bucket, 1), dtype=torch.float32, device=hidden.device)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = attention

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])
            self.hidden = hidden
            self.prev_topk = None

        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch):
            values = self.hidden[: input_ids.shape[0]]
            get_attn_tp_context().set_attn_inputs(AttentionInputs(values, forward_batch, attention.prepare_qkv_latent))
            kwargs = {"prev_topk_indices": self.prev_topk} if skip_indexer else {}
            return self.layers[0].self_attn(
                positions=positions,
                hidden_states=values,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                **kwargs,
            )

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.config = model_runner.model.config
            self.quant_config = getattr(model_runner.model, "quant_config", None)

        @torch.no_grad()
        def forward(self, input_ids, positions, forward_batch):
            output = self.model(input_ids, positions, forward_batch)
            if isinstance(output, tuple):
                output = output[0]
            return LogitsProcessorOutput(next_token_logits=logits[: input_ids.shape[0]], hidden_states=output)

    original_skip = attention.skip_topk
    original_next_skip = attention.next_skip_topk
    owned = copy.copy(model_runner)
    owned.server_args = copy.copy(model_runner.server_args)
    owned.server_args.cuda_graph_config = copy.deepcopy(model_runner.server_args.cuda_graph_config)
    owned.server_args.cuda_graph_config.prefill.bs = [bucket]
    owned.server_args.cuda_graph_config.prefill.max_bs = bucket
    owned.model_config = copy.copy(model_runner.model_config)
    owned.model_config.num_hidden_layers = 1
    owned.model = Outer()
    owned.attention_layers = [attention.attn_mqa]
    owned.moe_layers = [None]
    owned.moe_fusions = [None]
    owned.dsa_indexers = [attention.indexer]

    @torch.no_grad()
    def produce_topk(runner, batch):
        """Prepare the real producer output outside the reuse-layer timing."""
        if not skip_indexer:
            return
        values = hidden[: len(batch.input_ids)]
        with (
            forward_context(ForwardContext(attn_backend=owned.attn_backend)),
            set_tc_piecewise_forward_context(
                batch,
                runner.attention_layers,
                runner.quant_config,
                runner.moe_layers,
                runner.moe_fusions,
                dsa_indexers=runner.dsa_indexers,
            ),
        ):
            attention.skip_topk = False
            attention.next_skip_topk = True
            try:
                get_attn_tp_context().set_attn_inputs(AttentionInputs(values, batch, attention.prepare_qkv_latent))
                output = attention(
                    positions=batch.positions,
                    hidden_states=values,
                    forward_batch=batch,
                    zero_allocator=zero_allocator,
                )
                if not isinstance(output, tuple) or len(output) != 2 or output[1] is None:
                    raise RuntimeError("Native prefill producer returned no topk indices for the reuse layer")
                if owned.model.model.prev_topk is None:
                    owned.model.model.prev_topk = output[1].detach().clone()
                else:
                    owned.model.model.prev_topk.copy_(output[1])
            finally:
                attention.skip_topk = True
                attention.next_skip_topk = False

    class ModuleRunner(PrefillCudaGraphRunner):
        def _run_dummy_forward(self, num_tokens):
            # The native runner owns this batch and its opaque metadata. Only
            # the producer tensor needed by a standalone reuse layer is added.
            batch, backend = self.capture_prepare(num_tokens)
            backend.init_forward_metadata(batch)
            produce_topk(self, batch)
            self._run_forward(batch, num_tokens)

    import torch._dynamo.convert_frame as convert_frame

    previous_hooks = set(convert_frame._bytecode_hooks)
    previous_pool = pynccl_allocator._graph_pool_id
    runner = None
    try:
        attention.skip_topk = skip_indexer
        attention.next_skip_topk = False
        runner = ModuleRunner(owned)
        if not runner.can_run_graph(forward_batch):
            raise RuntimeError("Native SGLang prefill runner rejected the selected module batch")
        with runner.backend.replay_session():
            static_batch = runner.load_batch(forward_batch)
            produce_topk(runner, static_batch)

        print(f"DSA native tc_piecewise: tokens={len(hidden_states)} capture_bucket={bucket} skip={skip_indexer}")
        # Native execute enters these scopes once for the whole model
        # (prefill_cuda_graph_runner.py:866-930). Keep them outside the
        # per-layer timer, as well as Outer/logits processing. The instance
        # forward is the native installed compile.py:190-202 trampoline;
        # its compiled pieces and custom-op dispatch remain in the timer.
        with (
            torch.no_grad(),
            runner.backend.replay_session(),
            forward_context(ForwardContext(attn_backend=owned.attn_backend)),
            set_tc_piecewise_forward_context(
                static_batch,
                runner.attention_layers,
                runner.quant_config,
                runner.moe_layers,
                runner.moe_fusions,
                dsa_indexers=runner.dsa_indexers,
                num_tokens=bucket,
                raw_num_tokens=len(hidden_states),
            ),
        ):

            def replay():
                return owned.model.model.forward(static_batch.input_ids, static_batch.positions, static_batch)

            yield replay
    finally:
        attention.skip_topk = original_skip
        attention.next_skip_topk = original_next_skip
        try:
            if runner is not None:
                runner.backend.cleanup()
        finally:
            _release_owned_compile_hooks(convert_frame._bytecode_hooks, previous_hooks, owned.model.model)
            # The native installer also maintains a per-code Dynamo cache.
            # This adapter's Inner.forward is never installed on the caller's
            # model, so invalidate only its code, not the process-wide cache.
            torch._dynamo.eval_frame.remove_from_cache(Inner.forward.__code__)
            pynccl_allocator.set_graph_pool_id(previous_pool)
