# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual setup function without importing GPU-only vLLM."""

import ast
import functools
import gc
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[3] / "collector/vllm/utils.py"


@pytest.fixture
def harness(tmp_path):
    cleanups = []

    def build(*, fail_at=None):
        events = []
        contexts = []
        error = RuntimeError("native initialization failed")

        def device(value):
            kind, _, index = value.partition(":")
            return SimpleNamespace(type=kind, index=int(index) if index else None)

        @contextmanager
        def config_context(config):
            contexts.append(config)
            try:
                yield
            finally:
                contexts.pop()

        def initialize(**kwargs):
            assert len(contexts) == 1
            parsed = urlparse(kwargs["distributed_init_method"])
            assert parsed.scheme == "file"
            store = Path(unquote(parsed.path))
            assert store.parent.is_dir()
            assert not store.exists(), "Native FileStore must receive a fresh path"
            store.write_text("native store is live")
            events.append(("init", kwargs, store))
            if fail_at == "init":
                raise error

        def model_parallel(tp, pp):
            assert len(contexts) == 1
            assert events[-1][2].read_text() == "native store is live"
            events.append(("model_parallel", tp, pp))
            if fail_at == "model_parallel":
                raise error

        def destroy(name):
            # The real NCCL heartbeat can still read FileStore while native
            # groups are destroyed. Model this access at both shutdown steps.
            assert events[0][2].read_text() == "native store is live"
            events.append(name)
            if fail_at == name:
                raise error

        namespace = {
            "functools": functools,
            "atexit": SimpleNamespace(
                register=lambda callback, *args: cleanups.append(functools.partial(callback, *args))
            ),
            "tempfile": SimpleNamespace(mkdtemp=lambda **kwargs: tempfile.mkdtemp(dir=tmp_path, **kwargs)),
            "shutil": shutil,
            "Path": Path,
            "torch": SimpleNamespace(device=device, cuda=SimpleNamespace(current_device=lambda: 3)),
            "init_distributed_environment": initialize,
            "ensure_model_parallel_initialized": model_parallel,
            "destroy_model_parallel": lambda: destroy("destroy_model_parallel"),
            "destroy_distributed_environment": lambda: destroy("destroy_distributed_environment"),
            "set_current_vllm_config": config_context,
            "VllmConfig": object,
        }
        tree = ast.parse(SOURCE.read_text())
        nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in {"setup_distributed", "_shutdown_distributed"}
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
        return SimpleNamespace(setup=namespace["setup_distributed"], events=events, contexts=contexts, error=error)

    yield build, cleanups
    for cleanup in cleanups:
        cleanup()


def test_workers_on_same_visible_device_use_independent_stores_and_preserve_launcher_env(harness, monkeypatch):
    build, cleanups = harness
    for name, value in {"MASTER_ADDR": "launcher", "MASTER_PORT": "8889", "RANK": "7", "WORLD_SIZE": "8"}.items():
        monkeypatch.setenv(name, value)
    before = dict(os.environ)
    workers = [build(), build()]
    for worker in workers:
        worker.setup("cuda:0")
        worker.setup("cuda:0")
        assert len(worker.events) == 2  # One native init + one TP/PP init.
        kwargs = worker.events[0][1]
        assert kwargs == {
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "distributed_init_method": worker.events[0][2].as_uri(),
        }
        assert worker.events[1] == ("model_parallel", 1, 1)
        assert not worker.contexts
    assert workers[0].events[0][2] != workers[1].events[0][2]
    assert dict(os.environ) == before
    assert len(cleanups) == 2


@pytest.mark.parametrize("device,local_rank", [("cuda:4", 4), ("cuda", 3), ("cpu", 0)])
def test_explicit_device_selects_local_rank_without_launcher_inference(harness, device, local_rank):
    build, _ = harness
    worker = build()
    worker.setup(device)
    assert worker.events[0][1]["local_rank"] == local_rank


def test_store_lives_past_setup_and_garbage_collection_until_exit_cleanup(harness):
    build, cleanups = harness
    worker = build()
    worker.setup("cuda:0")
    store = worker.events[0][2]
    gc.collect()
    assert store.read_text() == "native store is live"
    cleanups.pop(0)()
    assert worker.events[-2:] == ["destroy_model_parallel", "destroy_distributed_environment"]
    assert not store.parent.exists()


@pytest.mark.parametrize("fail_at", ["init", "model_parallel"])
def test_native_failures_propagate_without_retry_and_keep_partial_store_alive(harness, fail_at):
    build, cleanups = harness
    worker = build(fail_at=fail_at)
    with pytest.raises(RuntimeError) as caught:
        worker.setup("cuda:0")
    assert caught.value is worker.error
    assert len(worker.events) == (1 if fail_at == "init" else 2)
    assert not worker.contexts
    gc.collect()
    assert worker.events[0][2].is_file()
    assert len(cleanups) == 1


@pytest.mark.parametrize("fail_at", ["destroy_model_parallel", "destroy_distributed_environment"])
def test_failed_native_teardown_preserves_store_and_reports_original_error(harness, fail_at):
    build, cleanups = harness
    worker = build(fail_at=fail_at)
    worker.setup("cuda:0")
    store = worker.events[0][2]
    with pytest.raises(RuntimeError, match=f"preserving rendezvous at {store.parent}") as caught:
        cleanups.pop(0)()
    assert caught.value.__cause__ is worker.error
    assert worker.events[-1] == fail_at
    if fail_at == "destroy_model_parallel":
        assert "destroy_distributed_environment" not in worker.events
    gc.collect()
    assert store.read_text() == "native store is live"
