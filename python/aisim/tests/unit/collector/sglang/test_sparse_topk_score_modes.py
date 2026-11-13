# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SGLang top-k kernel ops emit the top_last rows only: the sparse modules time their
top-k on written history, so there is no degenerate cost for a flat row to stand for."""

import ast
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit

SGLANG_DIR = Path(__file__).resolve().parents[4] / "collector" / "sglang"
DSV4_SPARSE_SOURCE = SGLANG_DIR / "deepseekv4_sparse_modules.py"
GLM5_SPARSE_SOURCE = SGLANG_DIR / "glm5_dsa_sparse_modules.py"


def _load(source_path: Path, *names: str, namespace: dict | None = None) -> dict:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    loaded = dict(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), loaded)
    return loaded


class _Lengths:
    """A numpy stand-in for the causal-length tensor _bench_topk_shape builds."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def __floordiv__(self, divisor):
        return _Lengths(self.values // divisor)

    def clamp(self, min):  # noqa: A002  (torch's keyword)
        return _Lengths(np.maximum(self.values, min))

    def repeat(self, times):
        return _Lengths(np.tile(self.values, times))

    def max(self):
        return _Lengths(self.values.max())

    def item(self):
        return self.values.item()


def test_dsv4_topk_shapes_return_one_top_last_row():
    torch = SimpleNamespace(arange=lambda start, stop, *, dtype, device: _Lengths(np.arange(start, stop)), int32="i32")
    timed = []

    def bench(seq_lens, device, *, topk_k, variant):
        timed.append((seq_lens.values.tolist(), topk_k, variant))
        return 0.1234567

    shape = _load(DSV4_SPARSE_SOURCE, "_bench_topk_shape", namespace={"torch": torch, "_bench_topk_512": bench})[
        "_bench_topk_shape"
    ]
    config = SimpleNamespace(csa_compress_ratio=4, index_topk=2)
    assert shape(8, 4, 2, config, "cuda:0", variant="v1") == [("v1_top_last", 0.123457)]
    assert timed == [([2, 2, 2, 3, 2, 2, 2, 3], 2, "v1")]
    # Nothing to select when the compressed length is within top-k: no kernel is timed.
    assert shape(0, 4, 1, config, "cuda:0", variant="v2") == [("v2_top_last", 0.0)]
    assert len(timed) == 1


def _worker(source_path, name, written, **fakes):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            OutOfMemoryError=MemoryError, empty_cache=lambda: None, get_device_name=lambda _device: "gpu"
        )
    )
    guarded = _load(
        DSV4_SPARSE_SOURCE,
        "_guarded_bench",
        namespace={"torch": fake_torch, "traceback": SimpleNamespace(print_exc=lambda: None), "Callable": Callable},
    )["_guarded_bench"]
    namespace = {
        "KERNEL_TO_OP_NAME": {"topk": "test_op"},
        "_guarded_bench": guarded,
        "_make_perf_filename": lambda _kernel, output_dir: os.path.join(output_dir, "test.txt"),
        "_write_row": lambda _path, **row: written.append(row["score_mode"]),
        "os": os,
        "torch": fake_torch,
        **fakes,
    }
    return _load(source_path, name, namespace=namespace)[name]


def _dsv4_worker(shape_result, written):
    worker = _worker(
        DSV4_SPARSE_SOURCE,
        "run_dsv4_sparse_kernel_worker",
        written,
        _bench_sparse_kernel_shape=lambda *_args, **_kwargs: shape_result,
        _dsv4_sparse_config=lambda _model: SimpleNamespace(num_attention_heads=64),
    )
    return lambda: worker("model", "topk", 1, 1, [[0, 4096, "v1"]], perf_filename="out.txt")


def _glm5_worker(shape_result, written):
    worker = _worker(
        GLM5_SPARSE_SOURCE,
        "run_glm5_dsa_sparse_kernel_worker",
        written,
        _bench_glm5_sparse_kernel_shape=lambda *_args, **_kwargs: ("kernel", shape_result),
        _glm5_sparse_config=lambda _model: SimpleNamespace(num_attention_heads=64),
    )
    return lambda: worker("model", "topk", 1, [[0, 4096]], perf_filename="out.txt")


@pytest.mark.parametrize(("make_worker", "mode"), [(_dsv4_worker, "v1_top_last"), (_glm5_worker, "top_last")])
def test_topk_workers_persist_the_top_last_row_and_reject_a_flat_one(make_worker, mode):
    written = []
    make_worker([(mode, 1.0)], written)()
    assert written == [mode]

    written.clear()
    flat = mode.replace("top_last", "flat")
    with pytest.raises(RuntimeError, match="expected score modes"):
        make_worker([(flat, 2.0), (mode, 1.0)], written)()
    assert not written


@pytest.mark.parametrize("source_path", [DSV4_SPARSE_SOURCE, GLM5_SPARSE_SOURCE])
def test_no_flat_score_mode_is_left_to_emit(source_path):
    source = source_path.read_text(encoding="utf-8")
    assert '"flat"' not in source and "_flat" not in source


def test_dsv4_rows_keep_the_variant_of_their_phase():
    # The row writer names the two rows a table may hold.
    assert 'score_mode not in ("v1_top_last", "v2_top_last")' in DSV4_SPARSE_SOURCE.read_text(encoding="utf-8")
    # op_plans.Dsv4TopkCase: v1 on context points, v2 on generation points.
    from collector.op_plans import plan

    variants = {point[2] for row in plan("dsv4_csa_topk_calib", "sglang", 100).rows for point in row[-1]}
    assert variants == {"v1", "v2"}
