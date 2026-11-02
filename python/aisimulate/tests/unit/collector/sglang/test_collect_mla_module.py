# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from collector.helper import PhaseClock, create_test_case_id, render_phase_totals, run_measurement_subprocess


@pytest.fixture(autouse=True)
def _mock_helper_imports(monkeypatch):
    """Ensure the module can be imported without CUDA / SGLang installed."""

    fake_helper = types.ModuleType("helper")
    fake_helper.get_sm_version = lambda: 90  # default Hopper
    fake_helper.log_perf = lambda **kw: None
    fake_helper.benchmark_with_power = lambda **kw: None
    fake_helper.PhaseClock = PhaseClock
    fake_helper.render_phase_totals = render_phase_totals
    fake_helper.run_measurement_subprocess = run_measurement_subprocess
    monkeypatch.setitem(__import__("sys").modules, "helper", fake_helper)

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.float8_e4m3fn = "float8_e4m3fn"
    fake_torch.cuda = types.SimpleNamespace(
        empty_cache=lambda: None,
        get_device_capability=lambda *_a, **_kw: (9, 0),
        get_device_name=lambda *_a, **_kw: "Fake GPU",
        is_available=lambda: False,
        current_device=lambda: 0,
        set_device=lambda *_a, **_kw: None,
    )
    fake_torch.randn = lambda *_a, **_kw: None
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _import_module():
    """Import collect_mla_module after mocking."""
    import importlib.util

    mod_name = "collector.sglang.collect_mla_module"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name,
        "collector/sglang/collect_mla_module.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestGetPrecisionCombos:
    def test_hopper_sm90(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            combos = mod._get_precision_combos("context")
        assert combos == [
            ("bfloat16", "bfloat16", "bfloat16"),
            ("bfloat16", "fp8", "bfloat16"),
            ("bfloat16", "bfloat16", "fp8_block"),
            ("bfloat16", "fp8", "fp8_block"),
        ]

    def test_ada_sm89_no_fp8(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=89):
            combos = mod._get_precision_combos("context")
        assert combos == [
            ("bfloat16", "bfloat16", "bfloat16"),
            ("bfloat16", "bfloat16", "fp8_block"),
        ]

    def test_blackwell_sm100(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=100):
            combos = mod._get_precision_combos("generation")
        assert ("bfloat16", "fp8", "bfloat16") in combos
        assert len(combos) == 4

    def test_no_phase_difference(self):
        """SGLang precision combos are the same for context and generation."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            ctx = mod._get_precision_combos("context")
            gen = mod._get_precision_combos("generation")
        assert ctx == gen


class TestGetBackends:
    def test_dsa_uses_0514_backend_name(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            assert mod._get_backends("dsa") == "dsa"
        with patch.object(mod, "get_sm_version", return_value=100):
            assert mod._get_backends("dsa") == "dsa"

    def test_mla_hopper(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            assert mod._get_backends("mla") == "fa3"

    def test_mla_blackwell(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=100):
            assert mod._get_backends("mla") == "trtllm_mla"

    def test_mla_older(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=80):
            assert mod._get_backends("mla") == "triton"


class TestGetContextTestCases:
    def test_memory_guard(self):
        """No test case exceeds batch_size * seq_len > 128K."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod.get_context_test_cases("mla"):
                assert case[0] * case[1] <= 128 * 1024

    def test_large_seq_guard(self):
        """seq_len >= 8192 only with batch_size <= 8."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod.get_context_test_cases("dsa"):
                if case[0] >= 8192:
                    assert case[1] <= 8

    def test_format_length(self):
        """Each inner sweep test case has 6 elements."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod.get_context_test_cases("mla"):
                assert len(case) == 6


class TestGetGenerationTestCases:
    def test_memory_guard(self):
        """The generation grid respects its declared token budget."""
        mod = _import_module()
        budget = mod.get_mla_module_sweep_spec("sglang").generation_max_tokens
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod.get_generation_test_cases("dsa"):
                assert case[0] * case[1] <= budget


@pytest.mark.unit
class TestAgenticGenerationCoverage:
    @pytest.mark.parametrize(("past_kv", "capacity"), [(15, 64), (16, 64), (15, 63)])
    def test_decode_bounds_report_errors_before_request_allocation(self, monkeypatch, capsys, past_kv, capacity):
        mod = _import_module()
        imports = {
            "sglang.srt.layers.layer_boundary.adapters.attention": ["AttentionInputs", "get_attn_tp_context"],
            "sglang.srt.managers.schedule_batch": ["Req", "ScheduleBatch"],
            "sglang.srt.mem_cache.cache_init_params": ["CacheInitParams"],
            "sglang.srt.mem_cache.chunk_cache": ["ChunkCache"],
            "sglang.srt.model_executor.forward_batch_info": ["ForwardBatch"],
            "sglang.srt.model_executor.runner": ["model_capture_mode"],
            "sglang.srt.sampling.sampling_params": ["SamplingParams"],
            "sglang.srt.speculative.spec_info": ["SpeculativeAlgorithm"],
            "sglang.srt.utils": ["BumpAllocator"],
        }
        for name, attributes in imports.items():
            native = types.ModuleType(name)
            native.__dict__.update({attribute: object() for attribute in attributes})
            monkeypatch.setitem(sys.modules, name, native)
        requests = []

        def request(**kwargs):
            requests.append(kwargs)
            raise ValueError("request construction reached")

        sys.modules["sglang.srt.managers.schedule_batch"].Req = request
        sys.modules["sglang.srt.sampling.sampling_params"].SamplingParams = types.SimpleNamespace
        monkeypatch.setattr(mod, "_import_sglang_forward_context", lambda: (None, None))
        monkeypatch.setattr(mod.torch, "OutOfMemoryError", MemoryError, raising=False)
        monkeypatch.setattr(mod.torch.cuda, "OutOfMemoryError", MemoryError, raising=False)
        monkeypatch.setattr(
            mod.torch, "randint", lambda *_: types.SimpleNamespace(tolist=lambda: [0] * past_kv), raising=False
        )
        pool = types.SimpleNamespace(clear=lambda: None, page_size=64, available_size=lambda: capacity)
        runner = types.SimpleNamespace(
            model_config=types.SimpleNamespace(context_len=16), req_to_token_pool=pool, token_to_kv_pool_allocator=pool
        )
        assert not mod._run_decode(
            runner,
            None,
            1,
            past_kv,
            8,
            1,
            1,
            "cuda",
            None,
            None,
            "dsa",
            "test",
            "GlmMoeDsaForCausalLM",
            "dsa",
            "0.5.21",
            "Fake GPU",
            "bfloat16",
            "fp8",
            "bfloat16",
            8,
        )
        error = capsys.readouterr().err
        if past_kv == 16:
            assert not requests
            assert "past_kv + 1=17 exceeds model context length=16" in error
        elif capacity < 64:
            assert not requests
            assert "requires 64 KV tokens, exceeding actual pool capacity=63" in error
        else:
            assert len(requests) == 1
            assert "request construction reached" in error

    def test_explicit_generation_batches_do_not_inject_batch_one(self, monkeypatch):
        mod = _import_module()
        monkeypatch.setattr(
            mod, "get_generation_test_cases", lambda attn_type: [[720896, 52, 8, "fp8", "bfloat16", "bfloat16"]]
        )
        monkeypatch.setattr(
            mod,
            "get_mla_module_sweep_spec",
            lambda backend: types.SimpleNamespace(
                generation_batch_sizes=[4, 8, 16, 32, 52], context_prefix_lengths=[1048575]
            ),
        )
        monkeypatch.setattr(mod, "_model_max_position_embeddings", lambda model: 1048576)
        assert mod._generation_manifest("dsa", 8, "fp8", "bfloat16", "bfloat16", "test") == ((52, 720896, 0),)


@pytest.mark.unit
class TestGenerationMemoryManifest:
    @pytest.mark.parametrize(
        "attn_type,kv_dtype,backend,expected",
        [
            ("mla", "bfloat16", None, 2304),
            ("dsa", "bfloat16", None, 2568),
            ("dsa", "fp8", "trtllm", 1416),
            ("dsa", "fp8", "flashmla_kv", 1576),
        ],
    )
    def test_native_pool_bytes_across_loaded_layers(self, monkeypatch, attn_type, kv_dtype, backend, expected):
        mod = _import_module()
        monkeypatch.setenv("SGLANG_TEST_NUM_LAYERS", "2")
        monkeypatch.setenv("SGLANG_LOAD_FORMAT", "dummy")
        assert mod._estimated_generation_kv_bytes_per_token("zai-org/GLM-5.2", attn_type, kv_dtype, backend) == expected

    @pytest.mark.parametrize("layers,load_format", [("0", "dummy"), ("2", "auto")])
    def test_untrimmed_runner_uses_full_layer_count(self, monkeypatch, layers, load_format):
        mod = _import_module()
        monkeypatch.setenv("SGLANG_TEST_NUM_LAYERS", layers)
        monkeypatch.setenv("SGLANG_LOAD_FORMAT", load_format)
        assert mod._estimated_generation_kv_bytes_per_token("zai-org/GLM-5.2", "dsa", "bfloat16", None) == 78 * 1284

    @pytest.mark.parametrize("device_gib,expected_count", [(80, 1), (268, 2)])
    def test_expanded_grid_drops_only_device_memory_shapes(self, monkeypatch, capsys, device_gib, expected_count):
        mod = _import_module()
        monkeypatch.setenv("SGLANG_TEST_NUM_LAYERS", "2")
        monkeypatch.setenv("SGLANG_LOAD_FORMAT", "dummy")
        monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            mod.torch.cuda,
            "get_device_properties",
            lambda _: types.SimpleNamespace(total_memory=device_gib * 2**30),
            raising=False,
        )
        # Explicit expanded fixture: the large cell needs 80.29 GiB of KV
        # allocation alone, including fresh-token and padding pages.
        monkeypatch.setattr(
            mod,
            "get_generation_test_cases",
            lambda _: [
                [720896, 1, 8, "bfloat16", "bfloat16", "bfloat16"],
                [131072, 256, 8, "bfloat16", "bfloat16", "bfloat16"],
            ],
        )
        monkeypatch.setattr(
            mod,
            "get_mla_module_sweep_spec",
            lambda _: types.SimpleNamespace(
                generation_batch_sizes=[256],
                context_prefix_lengths=[],
            ),
        )
        manifest = mod._generation_manifest("dsa", 8, "bfloat16", "bfloat16", "bfloat16", "deepseek-ai/DeepSeek-V3.2")
        assert len(manifest) == expected_count
        assert manifest[0] == (1, 720896, 0)  # context bounds are NOT a planning filter
        output = capsys.readouterr().out
        assert ("dropped 1/2 shapes (memory budget" in output) == (device_gib == 80)

    def test_no_cuda_preserves_plan_without_estimating(self, monkeypatch):
        mod = _import_module()
        monkeypatch.setattr(mod, "_estimated_generation_kv_bytes_per_token", lambda *_: pytest.fail("no device"))
        assert mod._generation_manifest("mla", 8, "bfloat16", "bfloat16", "bfloat16", "unknown")

    def test_generation_getters_bind_nonempty_manifests(self):
        mod = _import_module()
        for getter in (mod.get_dsa_generation_module_test_cases, mod.get_wideep_mla_generation_test_cases):
            cases = getter()
            assert cases
            assert all(len(case) == 12 and case[11] for case in cases)

    def test_manifest_changes_task_identity(self):
        mod = _import_module()
        case = mod.get_dsa_generation_module_test_cases()[0]
        smaller = [*case[:11], case[11][:-1]]
        assert create_test_case_id(case, "generation", mod.__name__) != create_test_case_id(
            smaller, "generation", mod.__name__
        )

    def test_cli_resolves_generation_manifest_before_dispatch(self, monkeypatch):
        mod = _import_module()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "collect_mla_module",
                "--mode",
                "generation",
                "--attn-type",
                "dsa",
                "--model",
                "test",
                "--num-heads",
                "8",
            ],
        )
        monkeypatch.setattr(mod, "_get_precision_combos", lambda _: [("bfloat16", "fp8", "bfloat16")])
        manifest = ((52, 131072, 0),)
        monkeypatch.setattr(mod, "_generation_manifest", lambda *_: manifest)
        dispatched = []
        monkeypatch.setattr(mod, "run_mla_module", lambda **kw: dispatched.append(kw["inner_manifest"]))
        mod.main()
        assert dispatched == [manifest]

    def test_empty_manifests_are_not_queued(self, monkeypatch, capsys):
        mod = _import_module()
        monkeypatch.setattr(mod, "_generation_manifest", lambda *_: ())
        assert mod.get_dsa_generation_module_test_cases() == []
        assert mod.get_wideep_mla_generation_test_cases() == []
        assert "empty generation manifest; group not queued" in capsys.readouterr().out

    def test_worker_executes_manifest_without_replanning(self, monkeypatch):
        mod = _import_module()
        monkeypatch.setattr(mod, "get_generation_test_cases", lambda *_: pytest.fail("must not rebuild queued plan"))
        monkeypatch.setattr(
            mod, "_filter_cases_from_env", lambda *_a, **_kw: pytest.fail("must not filter queued plan")
        )
        monkeypatch.setenv("AIC_DSA_CONTEXT_BATCH_SIZES", "1")
        monkeypatch.setattr(mod, "cleanup_distributed", lambda: None)
        monkeypatch.setattr(mod, "load_model_runner", lambda **_: object())
        captured = []
        monkeypatch.setattr(mod, "run_attention_torch", lambda **kw: captured.extend(kw["test_cases"]) or len(captured))
        manifest = ((52, 131072, 0), (1, 720896, 0))
        mod.run_mla_module("dsa", 8, "test", "fp8", "bfloat16", "bfloat16", False, 0, inner_manifest=manifest)
        assert captured == [(52, 131072, False, 0), (1, 720896, False, 0)]
        with pytest.raises(RuntimeError, match="nonempty planned inner manifest"):
            mod.run_mla_module("dsa", 8, "test", "fp8", "bfloat16", "bfloat16", False, 0)

    def test_generation_manifest_reaches_subprocess(self, monkeypatch):
        mod = _import_module()
        captured = {}

        def run_child(argv, **_kwargs):
            call = ast.Expression(ast.parse(argv[2]).body[-1].value)
            eval(
                compile(call, "<mla-subprocess>", "eval"), {"run_mla_module": lambda *_args, **kw: captured.update(kw)}
            )
            return 0, b""

        monkeypatch.setattr(mod, "run_measurement_subprocess", run_child)
        manifest = ((52, 131072, 0),)
        mod.run_mla_module_worker(
            0,
            0,
            8,
            "fp8",
            "bfloat16",
            "bfloat16",
            "test",
            "dsa",
            None,
            8,
            "trtllm",
            manifest,
            perf_filename="dsa_generation_module_perf.txt",
            skip_indexer=True,
        )
        assert captured["inner_manifest"] == manifest
        assert captured["skip_indexer"] is True


class TestDsaContextPrefixShape:
    def test_single_token_extend_stays_out_while_prefill_is_timed_eagerly(self):
        # 0.5.21: prefill graphs off, and the eager DSA indexer faulted on a one-token extend
        mod = _import_module()
        assert not mod._dsa_context_prefix_shape_is_valid(1, 1, 0)
        assert not mod._dsa_context_prefix_shape_is_valid(32, 1, 1024)

    def test_accepts_multi_token_prefill_with_prefix(self):
        mod = _import_module()
        assert mod._dsa_context_prefix_shape_is_valid(1, 2, 0)
        assert mod._dsa_context_prefix_shape_is_valid(32, 4, 1024)

    def test_rejects_negative_prefix(self):
        mod = _import_module()
        assert not mod._dsa_context_prefix_shape_is_valid(1, 2, -1)


class TestDsaRuntimeLimits:
    def test_total_kv_token_limit_matches_indexer_offset_capacity(self):
        from collector.sglang.runtime_limits import dsa_indexer_total_kv_tokens_supported

        assert dsa_indexer_total_kv_tokens_supported(32, 4, 1_000_000, is_prefill=True)
        assert not dsa_indexer_total_kv_tokens_supported(32, 4, 1_048_575, is_prefill=True)
        assert not dsa_indexer_total_kv_tokens_supported(64, 4, 524_288, is_prefill=True)

    def test_decode_allocation_counts_fresh_token_before_page_rounding(self):
        from collector.sglang.runtime_limits import required_kv_alloc_tokens

        assert required_kv_alloc_tokens(2, 256, 0, 256, is_prefill=True) == 512
        assert required_kv_alloc_tokens(2, 256, 0, 256, is_prefill=False) == 1024

    def test_hybrid_swa_allocation_counts_tail_and_fresh_page(self):
        from collector.sglang.runtime_limits import required_swa_kv_alloc_tokens

        assert required_swa_kv_alloc_tokens(2, 1, 256, 256, 128, is_prefill=True) == 1024
        assert required_swa_kv_alloc_tokens(2, 256, 0, 256, 128, is_prefill=False) == 1024


@pytest.mark.unit
class TestDsaCudaGraph:
    def test_generation_cuda_graph_uses_max_batch_tokens(self):
        mod = _import_module()
        runner = types.SimpleNamespace(
            server_args=types.SimpleNamespace(
                cuda_graph_config=types.SimpleNamespace(
                    decode=types.SimpleNamespace(backend="cudagraphs", bs=None, max_bs=256)
                )
            )
        )

        assert mod._generation_cuda_graph_enabled_for_tokens(runner, 256)
        assert not mod._generation_cuda_graph_enabled_for_tokens(runner, 257)
        assert not mod._generation_cuda_graph_enabled_for_tokens(runner, 0)

    def test_generation_cuda_graph_uses_explicit_batch_list(self):
        mod = _import_module()
        runner = types.SimpleNamespace(
            server_args=types.SimpleNamespace(
                disable_cuda_graph_padding=True,
                cuda_graph_config=types.SimpleNamespace(
                    decode=types.SimpleNamespace(backend="cudagraphs", bs=[1, 4, 16], max_bs=None)
                ),
            )
        )

        assert mod._generation_cuda_graph_enabled_for_tokens(runner, 4)
        assert not mod._generation_cuda_graph_enabled_for_tokens(runner, 8)
        assert not mod._generation_cuda_graph_enabled_for_tokens(runner, 17)

    @pytest.mark.parametrize("batch_size", [4, 8, 16, 32, 52])
    def test_agentx_batches_use_exact_shape_graph_with_default_bucket_coverage(self, batch_size):
        mod = _import_module()
        capture_bs = [1, 2, 4, 8, 12, *range(16, 257, 8)]
        assert 52 not in capture_bs
        runner = types.SimpleNamespace(
            server_args=types.SimpleNamespace(
                disable_cuda_graph_padding=False,
                cuda_graph_config=types.SimpleNamespace(
                    decode=types.SimpleNamespace(backend="cudagraphs", bs=capture_bs, max_bs=256)
                ),
            )
        )
        assert mod._generation_cuda_graph_enabled_for_tokens(runner, batch_size)
        assert not mod._generation_cuda_graph_enabled_for_tokens(runner, 257)

    @pytest.mark.parametrize("disable_padding", [False, True])
    def test_generation_cuda_graph_uses_native_resolved_arguments(self, monkeypatch, disable_padding):
        mod = _import_module()
        native = types.ModuleType("sglang.srt.arg_groups.model_override_base")
        native.resolving_view = lambda _: types.SimpleNamespace(
            cuda_graph_config=types.SimpleNamespace(
                decode=types.SimpleNamespace(backend="cudagraphs", bs=[48, 56], max_bs=56)
            ),
            disable_cuda_graph_padding=disable_padding,
        )
        monkeypatch.setitem(sys.modules, native.__name__, native)
        runner = types.SimpleNamespace(server_args=types.SimpleNamespace())
        assert mod._generation_cuda_graph_enabled_for_tokens(runner, 52) is not disable_padding
        assert mod._generation_cuda_graph_enabled_for_tokens(runner, 56)
        assert not mod._generation_cuda_graph_enabled_for_tokens(runner, 57)

    def test_generation_cuda_graph_can_be_disabled(self):
        mod = _import_module()
        runner = types.SimpleNamespace(
            server_args=types.SimpleNamespace(
                cuda_graph_config=types.SimpleNamespace(
                    decode=types.SimpleNamespace(backend="disabled", bs=[1, 4, 16], max_bs=None)
                )
            )
        )

        assert not mod._generation_cuda_graph_enabled_for_tokens(runner, 4)


class TestDsaSkipIndexer:
    def test_uses_sglang_cross_layer_topk_contract(self):
        source_path = Path("collector/sglang/collect_mla_module.py")
        source = source_path.read_text()
        tree = ast.parse(source, filename=str(source_path))
        functions = {
            node.name: ast.get_source_segment(source, node) for node in tree.body if isinstance(node, ast.FunctionDef)
        }

        prefill = functions["_run_prefill"]
        assert "AttnForwardMethod.MHA_ONE_SHOT" in prefill
        assert "attention_module.next_skip_topk = True" in prefill
        assert '_skip_state["prev_topk"] = warmup_output[1].detach()' in prefill
        assert "attention_module.skip_topk = _skip_uses_dense_mha" in prefill
        prefill_tree = ast.parse(prefill)
        missing_producer = ast.parse(
            "_skip_indexer and native_graph_context is None "
            'and not _skip_uses_dense_mha and _skip_state["prev_topk"] is None',
            mode="eval",
        ).body
        assert any(
            isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(missing_producer)
            for node in ast.walk(prefill_tree)
        )
        assert "sglang_dsa_dense_mha_" in prefill
        assert "dsa_backend.dsa_prefill_impl" in prefill

        decode = functions["_run_decode"]
        assert "attention_module.next_skip_topk = True" in decode
        assert '_skip_state["prev_topk"] = warmup_output[1].detach()' in decode
        assert "return attention_module(" in decode
        assert "model_capture_mode() if use_benchmark_cuda_graph" in decode
        assert "model_runner.attn_backend.dsa_decode_impl" in decode

        assert "register_forward_hook" not in prefill
        assert "register_forward_hook" not in decode


@pytest.mark.unit
class TestBuildModuleTestCases:
    def test_glm52_local_config_uses_transformers_v5_layer_type(self):
        mod = _import_module()

        local_path = Path(mod._resolve_local_model_path("nvidia/GLM-5.2-NVFP4"))
        config = json.loads((local_path / "config.json").read_text())

        assert "deepseek_sparse_attention" not in config["layer_types"]
        assert set(config["layer_types"]) == {"compressed_sparse_attention"}
        assert config["architectures"] == ["GlmMoeDsaForCausalLM"]
        assert config["index_topk_freq"] == 4

    @pytest.mark.parametrize(
        "model_path",
        ["zai-org/GLM-5.3", "zai-org/GLM-5.3-BF16", "nvidia/GLM-5.3-NVFP4"],
    )
    def test_glm53_alias_configs_match_glm52_geometry(self, model_path):
        mod = _import_module()
        geometry_keys = (
            "hidden_size",
            "intermediate_size",
            "moe_intermediate_size",
            "num_experts_per_tok",
            "n_routed_experts",
            "n_shared_experts",
            "scoring_func",
            "routed_scaling_factor",
            "norm_topk_prob",
            "num_hidden_layers",
            "indexer_types",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "index_topk_freq",
            "num_attention_heads",
            "kv_lora_rank",
            "q_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "max_position_embeddings",
        )

        def geometry(path):
            local_path = Path(mod._resolve_local_model_path(path))
            config = json.loads((local_path / "config.json").read_text())
            assert len(config["indexer_types"]) == config["num_hidden_layers"]
            return {key: config[key] for key in geometry_keys}

        assert geometry(model_path) == geometry("zai-org/GLM-5.2")

    def test_module_precision_respects_outer_sm_gate(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=89):
            cases = mod._build_module_test_cases("dsa", "context")
        assert cases
        assert {case[3] for case in cases} == {"bfloat16"}

    def test_module_precision_passes_phase_and_sm_to_yaml(self):
        mod = _import_module()
        calls = []

        def precision_specs(backend, *, phase, sm_version):
            calls.append((backend, phase, sm_version))
            return [
                types.SimpleNamespace(
                    compute_dtype="bfloat16",
                    kv_cache_dtype="bfloat16",
                    gemm_type="bfloat16",
                )
            ]

        with (
            patch.object(mod, "get_sm_version", return_value=89),
            patch.object(mod, "get_mla_module_precision_specs", side_effect=precision_specs),
        ):
            cases = mod._build_module_test_cases("dsa", "generation")

        assert cases
        assert set(calls) == {("sglang", "generation", 89)}

    def test_full_dsa_sweep_canonicalizes_consumer_equivalent_checkpoints(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=100):
            cases = mod._build_module_test_cases("dsa", "context")
        model_paths = {case[6] for case in cases}
        # #1378 canonicalization: one DSA collection per (architecture,
        # DSA-gemm-type) group on the longest-context checkpoint, reused for the
        # consumer-equivalent rest. GLM-5/5.1/5.2/5.3 share identical DSA geometry
        # (index_n_heads=32, index_head_dim=128, index_topk=2048, heads/dims/lora,
        # max_position_embeddings all equal for 5.2/5.3) so GLM-5.2 (bf16) and
        # GLM-5.2-FP8 (fp8_block) are canonical by YAML iteration order; GLM-5/5.1
        # and every NVFP4 checkpoint (which shares the bf16 DSA key) are reused.
        assert model_paths == {
            "deepseek-ai/DeepSeek-V3.2",
            "zai-org/GLM-5.2",
            "zai-org/GLM-5.2-FP8",
        }
        assert "zai-org/GLM-5" not in model_paths
        assert "zai-org/GLM-5-FP8" not in model_paths
        assert not any("NVFP4" in model_path for model_path in model_paths)

    def test_mla_includes_v3_family(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("mla", "context")
        model_paths = {c[6] for c in cases}
        assert model_paths == {
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-V3",
        }

    def test_mla_blackwell_keeps_native_nvfp4_checkpoint(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=100):
            cases = mod._build_module_test_cases("mla", "context")
        assert "nvidia/DeepSeek-V3.1-NVFP4" in {c[6] for c in cases}

    def test_hopper_dsa_uses_bf16_glm_checkpoint(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("dsa", "context")
        model_paths = {case[6] for case in cases}
        # bf16 canonical is GLM-5.2 (first in YAML among 5.2/5.3); GLM-5/5.1 reuse it.
        assert "zai-org/GLM-5.2" in model_paths
        assert "zai-org/GLM-5" not in model_paths
        assert not any("NVFP4" in model_path for model_path in model_paths)

    def test_targeted_quantized_artifact_keeps_requested_checkpoint(self, monkeypatch):
        mod = _import_module()
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/GLM-5-NVFP4")
        with patch.object(mod, "get_sm_version", return_value=100):
            cases = mod._build_module_test_cases("dsa", "context")
        assert cases
        assert {case[6] for case in cases} == {"nvidia/GLM-5-NVFP4"}

    def test_generation_case_format(self):
        """Each DSA decode task includes target TP, backend and exact manifest."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod._build_module_test_cases("dsa", "generation"):
                assert len(case) == 12
                assert case[7] == "dsa"
                assert case[8] is None  # DSA backend resolved at runtime
                assert case[9] in {1, 2, 4, 8}
                assert case[10] == ("flashmla_kv" if case[3] == "fp8" else None)

    def test_blackwell_context_uses_0514_default_backend(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=100):
            cases = mod._build_module_test_cases("dsa", "context")
        assert {case[10] for case in cases if case[3] == "fp8"} == {"trtllm"}

    def test_blackwell_generation_uses_0514_default_backend(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=100):
            cases = mod._build_module_test_cases("dsa", "generation")
        assert {case[10] for case in cases if case[3] == "fp8"} == {"trtllm"}

    def test_skip_indexer_cases_use_glm52_canonical_model(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=100):
            cases = mod.get_dsa_context_module_skip_indexer_test_cases()
        assert cases
        # GLM-5.2 and GLM-5.3 both have index_topk_freq=4; GLM-5/5.1 have freq=1
        # (no skip layers). The skip-indexer op is collected on the GLM-5.2
        # canonicals (bf16 + fp8_block) since 5.2 precedes 5.3 in YAML order;
        # NVFP4 reuses the bf16 key.
        assert {case[6] for case in cases} == {"zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8"}

    def test_glm_sparse_selector_preserves_targeted_artifact(self, monkeypatch):
        _import_module()
        source_path = Path("collector/sglang/glm5_dsa_sparse_modules.py")
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        selector_node = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_selected_glm5_models"
        )
        namespace = {"get_sm_version": lambda: 90}
        exec(compile(ast.Module(body=[selector_node], type_ignores=[]), str(source_path), "exec"), namespace)
        selector = namespace["_selected_glm5_models"]

        monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
        assert selector() == ["zai-org/GLM-5"]
        for model_path in ("nvidia/GLM-5-NVFP4", "nvidia/GLM-5.2-NVFP4"):
            monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
            assert selector() == []

        namespace["get_sm_version"] = lambda: 100
        monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
        assert selector() == ["nvidia/GLM-5.2-NVFP4"]
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/GLM-5-NVFP4")
        assert selector() == ["nvidia/GLM-5-NVFP4"]

        namespace["get_sm_version"] = lambda: 90
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V3.2")
        assert selector() == []

    def test_deduplication(self):
        """One entry per top-level sweep tuple, not per inner (seq, batch) shape."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("dsa", "context")
        keys = {
            (
                c[6],
                c[2],
                c[3],
                c[4],
                c[5],
                c[1],
                c[9],
                c[10],
            )
            for c in cases
        }
        assert len(cases) == len(keys)
        assert all(c[0] == 0 for c in cases)

    def test_dsa_context_is_sharded_by_batch_size(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("dsa", "context")
        batch_sizes = {c[1] for c in cases}
        assert batch_sizes == set(mod.get_mla_module_sweep_spec("sglang").context_batch_sizes)

    # Reads the bundled export files, whatever shape they hold. The MIXED_PRECISION shape is pinned in
    # test_collect_mla_module_native_quant.py with export files written to tmp_path.
    @pytest.mark.parametrize("model_path", ["nvidia/GLM-5-NVFP4", "nvidia/GLM-5.3-NVFP4"])
    def test_targeted_glm5_nvfp4_context_is_not_emitted_on_hopper(self, monkeypatch, model_path):
        mod = _import_module()
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("dsa", "context")
        assert cases == []

    def test_targeted_nvfp4_wideep_mla_is_not_emitted_on_hopper(self, monkeypatch):
        mod = _import_module()
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/DeepSeek-V3.1-NVFP4")
        with patch.object(mod, "get_sm_version", return_value=90):
            assert mod._build_wideep_mla_test_cases("context") == []

    def test_hopper_loader_rejects_nvfp4_before_sglang_initialization(self):
        mod = _import_module()
        with (
            patch.object(mod, "get_sm_version", return_value=90),
            patch.object(mod, "_model_native_gemm_quant", return_value="nvfp4"),
            pytest.raises(ValueError, match=r"no collector-supported native NVFP4.*SM90"),
        ):
            mod.load_model_runner("example/NVFP4", 8, "bfloat16", "dsa")

    def test_placeholder_seq_batch(self):
        """seq_len and batch_size are placeholders (0) — subprocess sweeps internally."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod._build_module_test_cases("mla", "context"):
                assert case[0] == 0  # seq_len placeholder
                assert case[1] == 0  # batch_size placeholder


class TestDsaTpShapeValidation:
    def _runner(self, attn):
        return types.SimpleNamespace(
            model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attn)]))
        )

    def test_accepts_local_rank_projection_shapes(self):
        mod = _import_module()
        attn = types.SimpleNamespace(
            num_heads=8,
            num_local_heads=8,
            qk_nope_head_dim=384,
            qk_rope_head_dim=128,
            v_head_dim=256,
            hidden_size=6144,
            q_b_proj=types.SimpleNamespace(output_size=4096, output_size_per_partition=4096),
            kv_b_proj=types.SimpleNamespace(output_size=5120, output_size_per_partition=5120),
            o_proj=types.SimpleNamespace(input_size=2048, input_size_per_partition=2048, output_size=6144),
        )
        mod._validate_dsa_tp_module_shapes(self._runner(attn), local_num_heads=8, target_tp_size=8)

    def test_rejects_native_projection_shapes_for_tp_rank(self):
        mod = _import_module()
        attn = types.SimpleNamespace(
            num_heads=8,
            num_local_heads=8,
            qk_nope_head_dim=384,
            qk_rope_head_dim=128,
            v_head_dim=256,
            hidden_size=6144,
            q_b_proj=types.SimpleNamespace(output_size=64 * 512, output_size_per_partition=64 * 512),
            kv_b_proj=types.SimpleNamespace(output_size=8 * (384 + 256), output_size_per_partition=8 * (384 + 256)),
            o_proj=types.SimpleNamespace(input_size=8 * 256, input_size_per_partition=8 * 256, output_size=6144),
        )
        with pytest.raises(RuntimeError, match=r"q_b_proj\.output_size"):
            mod._validate_dsa_tp_module_shapes(self._runner(attn), local_num_heads=8, target_tp_size=8)


class TestEntryPoints:
    def test_wideep_mla_context_returns_list(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod.get_wideep_mla_context_test_cases()
        assert isinstance(cases, list)
        assert len(cases) > 0

    def test_wideep_mla_generation_returns_list(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod.get_wideep_mla_generation_test_cases()
        assert isinstance(cases, list)
        assert len(cases) > 0

    def test_dsa_generation_returns_list(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod.get_dsa_generation_module_test_cases()
        assert isinstance(cases, list)
        assert len(cases) > 0


class TestGetMlaBackendList:
    def test_hopper(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            assert mod._get_mla_backend_list() == ["flashinfer", "fa3"]

    def test_blackwell(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=100):
            assert mod._get_mla_backend_list() == ["trtllm_mla"]

    def test_older(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=80):
            assert mod._get_mla_backend_list() == []


class TestBuildWideepMlaTestCases:
    def test_format_length_10(self):
        """Each wideep MLA test case has 9 elements (6 + model_path + attn_type + backend)."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod._build_wideep_mla_test_cases("context"):
                assert len(case) == 9
                assert case[7] == "mla"

    def test_context_filename(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("context")
        assert {c[7] for c in cases} == {"mla"}

    def test_generation_filename(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("generation")
        assert {c[7] for c in cases} == {"mla"}

    def test_only_mla_models(self):
        """Full WideEP MLA keeps one consumer-equivalent V3 representative."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("context")
        model_paths = {c[6] for c in cases}
        assert model_paths == {"deepseek-ai/DeepSeek-V3"}

    def test_sweeps_backends(self):
        """Hopper should sweep flashinfer and fa3 backends."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("context")
        backends = {c[8] for c in cases}
        assert backends == {"flashinfer", "fa3"}

    def test_single_precision_bfloat16(self):
        """All wideep MLA cases use bfloat16 precision (logged as fp8_block/fp8)."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod._build_wideep_mla_test_cases("context"):
                assert case[3] == "bfloat16"  # kv_cache_dtype
                assert case[4] == "bfloat16"  # compute_dtype
                assert case[5] == "bfloat16"  # gemm_type


@pytest.mark.unit
class TestFailClosedOrchestration:
    def test_empty_case_list_raises_before_model_load(self, monkeypatch):
        mod = _import_module()
        monkeypatch.setattr(mod, "get_context_test_cases", lambda _attn_type: [])
        monkeypatch.setattr(mod, "_filter_cases_from_env", lambda cases, **_kwargs: cases)
        monkeypatch.setattr(
            mod,
            "load_model_runner",
            lambda **_kwargs: pytest.fail("empty sweep must fail before loading a model"),
        )

        with pytest.raises(RuntimeError, match=r"MLA module context has no runnable cases"):
            mod.run_mla_module(
                "mla",
                8,
                "deepseek-ai/DeepSeek-V3",
                "bfloat16",
                "bfloat16",
                "bfloat16",
                True,
                0,
                attention_backend="fa3",
            )

    def test_zero_logged_rows_raises(self, monkeypatch):
        mod = _import_module()
        test_case = [128, 1, 8, "bfloat16", "bfloat16", "bfloat16"]
        monkeypatch.setattr(mod, "get_context_test_cases", lambda _attn_type: [test_case])
        monkeypatch.setattr(mod, "_filter_cases_from_env", lambda cases, **_kwargs: cases)
        monkeypatch.setattr(mod, "cleanup_distributed", lambda: None)
        monkeypatch.setattr(mod, "load_model_runner", lambda **_kwargs: object())
        monkeypatch.setattr(mod, "run_attention_torch", lambda **_kwargs: 0)

        with pytest.raises(RuntimeError, match=r"MLA module context persisted no rows"):
            mod.run_mla_module(
                "mla",
                8,
                "deepseek-ai/DeepSeek-V3",
                "bfloat16",
                "bfloat16",
                "bfloat16",
                True,
                0,
                attention_backend="fa3",
            )

    def test_configured_subprocess_timeout_raises(self, monkeypatch, capsys):
        mod = _import_module()

        def timed_out(argv, *, env, cwd, timeout):
            # helper.run_measurement_subprocess kills the child and keeps its output
            assert timeout == 7
            raise mod.subprocess.TimeoutExpired(cmd=argv, timeout=timeout, output=b"partial sweep")

        monkeypatch.setenv("AIC_MLA_MODULE_SUBPROCESS_TIMEOUT_SEC", "7")
        monkeypatch.setattr(mod, "run_measurement_subprocess", timed_out)

        with pytest.raises(RuntimeError, match=r"MLA module context subprocess timed out after 7s"):
            mod._run_mla_subprocess(
                "mla",
                8,
                "deepseek-ai/DeepSeek-V3",
                "bfloat16",
                "bfloat16",
                "bfloat16",
                True,
                0,
                attention_backend="fa3",
            )

        assert "partial sweep" in capsys.readouterr().out


class TestOrdinaryMLAPrecision:
    @pytest.mark.parametrize(
        "gemm_type,dtype,block", [("bfloat16", "bfloat16", None), ("fp8_block", "float8_e4m3fn", [128, 128])]
    )
    def test_uniform_projections(self, gemm_type, dtype, block):
        mod = _import_module()
        projection = types.SimpleNamespace(
            weight=types.SimpleNamespace(dtype=dtype),
            quant_method=types.SimpleNamespace(quant_config=types.SimpleNamespace(weight_block_size=block)),
        )
        attention = types.SimpleNamespace(
            **dict.fromkeys(("fused_qkv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj"), projection)
        )
        mod._validate_mla_projection_precision(attention, gemm_type)

    def test_mixed_projection_rejected_before_writing_row(self):
        mod = _import_module()
        bf16 = types.SimpleNamespace(weight=types.SimpleNamespace(dtype="bfloat16"), quant_method=None)
        fp8 = types.SimpleNamespace(
            weight=types.SimpleNamespace(dtype="float8_e4m3fn"),
            quant_method=types.SimpleNamespace(quant_config=types.SimpleNamespace(weight_block_size=[128, 128])),
        )
        attention = types.SimpleNamespace(fused_qkv_a_proj_with_mqa=bf16, q_b_proj=bf16, kv_b_proj=bf16, o_proj=fp8)
        with pytest.raises(ValueError, match="o_proj executes fp8_block"):
            mod._validate_mla_projection_precision(attention, "bfloat16")


@pytest.mark.unit
class TestOrdinaryMLACli:
    @pytest.mark.parametrize("chunk_size", [None, 16384])
    def test_worker_chunk_size_reaches_subprocess_call(self, monkeypatch, chunk_size):
        mod = _import_module()
        captured = {}

        def capture_call(*_args, **kwargs):
            captured.update(kwargs)

        def run_child(argv, **_kwargs):
            # Execute the generated child call without importing the GPU runtime.
            tree = ast.parse(argv[2])
            call = ast.Expression(tree.body[-1].value)
            eval(compile(call, "<mla-subprocess>", "eval"), {"run_mla_module": capture_call})
            return 0, b""

        monkeypatch.setattr(mod, "run_measurement_subprocess", run_child)
        mod.run_mla_module_worker(
            128,
            1,
            8,
            "bfloat16",
            "bfloat16",
            "bfloat16",
            "test",
            "mla",
            perf_filename="mla_context_module_perf.txt",
            chunked_prefill_size=chunk_size,
        )
        assert captured["ordinary_mla"] is True
        assert captured["chunked_prefill_size"] == chunk_size

    @pytest.mark.parametrize("chunk_size", [0, -1])
    def test_nonpositive_chunk_size_is_rejected(self, chunk_size):
        mod = _import_module()
        with pytest.raises(ValueError, match="chunked_prefill_size must be positive"):
            mod.run_mla_module(
                "mla",
                8,
                "test",
                "bfloat16",
                "bfloat16",
                "bfloat16",
                True,
                0,
                ordinary_mla=True,
                chunked_prefill_size=chunk_size,
            )

    @pytest.mark.parametrize("chunk_size", [None, 16384])
    @pytest.mark.parametrize("tp_size", [4, 8])
    @pytest.mark.parametrize("explicit_tp", [False, True])
    def test_cli_serving_options_reach_model_runner(self, monkeypatch, chunk_size, tp_size, explicit_tp):
        mod = _import_module()
        args = ["collect_mla_module", "--ordinary-mla", "--mode", "context", "--attn-type", "mla", "--model", "test"]
        if explicit_tp:
            args.extend(["--target-tp-size", str(tp_size)])
        else:
            args.extend(["--num-heads", str(128 // tp_size)])
        if chunk_size is not None:
            args.extend(["--chunked-prefill-size", str(chunk_size)])
        monkeypatch.setattr(sys, "argv", args)
        monkeypatch.setattr(mod, "_module_model_native_heads", lambda _model: 128)
        monkeypatch.setattr(mod, "_get_precision_combos", lambda _mode: [("bfloat16", "bfloat16", "bfloat16")])
        monkeypatch.setattr(
            mod, "get_context_test_cases", lambda _type: [[128, 1, 128 // tp_size, "bfloat16", "bfloat16", "bfloat16"]]
        )
        monkeypatch.setattr(mod, "_filter_cases_from_env", lambda cases, **_kwargs: cases)
        monkeypatch.setattr(mod, "cleanup_distributed", lambda: None)

        captured = {}
        timed = {}

        def load_runner(**kwargs):
            captured.update(kwargs)
            return object()

        def run_attention(**kwargs):
            timed.update(kwargs)
            return len(kwargs["test_cases"])

        monkeypatch.setattr(mod, "load_model_runner", load_runner)
        monkeypatch.setattr(mod, "run_attention_torch", run_attention)
        mod.main()
        assert captured["chunked_prefill_size"] == chunk_size
        assert captured["target_tp_size"] == tp_size
        assert captured["head_num"] == 128 // tp_size
        assert timed["target_tp_size"] == tp_size
        assert timed["head_num"] == 128 // tp_size

    @pytest.mark.parametrize(
        "options",
        [
            ["--target-tp-size", "0"],
            ["--target-tp-size", "-1"],
            ["--target-tp-size", "3"],
            ["--target-tp-size", "4", "--num-heads", "16"],
            ["--num-heads", "0"],
            ["--num-heads", "-1"],
            ["--num-heads", "3"],
        ],
    )
    def test_invalid_tp_shape_fails_before_dispatch(self, monkeypatch, options):
        mod = _import_module()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "collect_mla_module",
                "--ordinary-mla",
                "--mode",
                "context",
                "--attn-type",
                "mla",
                "--model",
                "test",
                *options,
            ],
        )
        monkeypatch.setattr(mod, "_module_model_native_heads", lambda _model: 128)
        monkeypatch.setattr(mod, "run_mla_module", lambda **_kw: pytest.fail("must fail before dispatch"))
        with pytest.raises(SystemExit) as exc:
            mod.main()
        assert exc.value.code == 2

    @pytest.mark.parametrize(
        "args",
        [
            ["--mode", "generation", "--attn-type", "mla"],
            ["--mode", "context", "--attn-type", "dsa"],
            ["--mode", "context"],
        ],
    )
    def test_invalid_combination_fails_before_dispatch(self, monkeypatch, capsys, args):
        mod = _import_module()
        monkeypatch.setattr(sys, "argv", ["collect_mla_module", "--ordinary-mla", *args])
        monkeypatch.setattr(mod, "get_mla_module_model_specs", lambda **_kw: pytest.fail("must fail before dispatch"))
        with pytest.raises(SystemExit) as exc:
            mod.main()
        assert exc.value.code == 2
        assert "--ordinary-mla requires --mode context --attn-type mla" in capsys.readouterr().err

    @pytest.mark.parametrize("fail_first", [False, True])
    def test_context_mla_dispatches(self, monkeypatch, capsys, fail_first):
        mod = _import_module()
        monkeypatch.setattr(
            sys,
            "argv",
            ["collect_mla_module", "--ordinary-mla", "--mode", "context", "--attn-type", "mla", "--model", "test"],
        )
        monkeypatch.setattr(mod, "_module_model_native_heads", lambda _model: 8)
        monkeypatch.setattr(
            mod,
            "_get_precision_combos",
            lambda _mode: [("bfloat16", "bfloat16", "bfloat16"), ("bfloat16", "fp8", "bfloat16")],
        )
        calls = []

        def collect(**kwargs):
            calls.append(kwargs)
            if fail_first and len(calls) == 1:
                raise ValueError("projection precision mismatch")

        monkeypatch.setattr(mod, "run_mla_module", collect)
        if fail_first:
            with pytest.raises(SystemExit) as exc:
                mod.main()
            assert exc.value.code == 1
            output = capsys.readouterr()
            assert "projection precision mismatch" in output.err
            assert "COLLECTION FAILED: 1 dispatches failed" in output.out
            assert "ALL TESTS COMPLETED" not in output.out
        else:
            mod.main()
            assert "ALL TESTS COMPLETED" in capsys.readouterr().out
        assert len(calls) > 1
        assert all(call["ordinary_mla"] and call["is_prefill"] and call["attn_type"] == "mla" for call in calls)
