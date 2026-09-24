# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU loader experiments and rejection controls; no CUDA or GPU evidence."""

import copy
import ctypes
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from collector import glm53flash_graph_libraries
from collector.glm53flash_graph_libraries import _select_provider

pytestmark = pytest.mark.unit


def bindings():
    candidates = [
        {"path": "/first/libcudart.so.13", "load_bias": 1000, "sha256": "a" * 64},
        {"path": "/second/libcudart.so.13", "load_bias": 2000, "sha256": "a" * 64},
    ]
    callers = []
    for name, symbols in (
        (
            "libtorch_cuda.so",
            ("cudaGraphLaunch", "cudaGraphInstantiateWithFlags", "cudaStreamBeginCapture", "cudaStreamEndCapture"),
        ),
        ("libc10_cuda.so", ("cudaStreamGetCaptureInfo", "cudaEventRecord")),
    ):
        callers.append(
            {
                "path": "/torch/lib/" + name,
                "relocations": [
                    {"symbol": symbol, "provider_path": candidates[1]["path"], "provider_load_bias": 2000}
                    for symbol in symbols
                ],
            }
        )
    return candidates, callers


def test_actual_provider_is_selected_even_when_it_is_not_the_first_path():
    candidates, callers = bindings()
    assert _select_provider(candidates, callers) == candidates[1]
    assert _select_provider(list(reversed(candidates)), callers) == candidates[1]


@pytest.mark.parametrize(
    "defect", ["mixed", "unresolved", "wrong_base", "missing_launch", "missing_c10", "different_binary", "duplicate"]
)
def test_unknown_or_mixed_runtime_bindings_fail_closed(defect):
    candidates, callers = copy.deepcopy(bindings())
    if defect == "mixed":
        callers[1]["relocations"][0].update(provider_path=candidates[0]["path"], provider_load_bias=1000)
    elif defect == "unresolved":
        callers[0]["relocations"][0]["provider_path"] = callers[0]["path"]
    elif defect == "wrong_base":
        callers[0]["relocations"][0]["provider_load_bias"] = 9999
    elif defect == "missing_launch":
        callers[0]["relocations"] = callers[0]["relocations"][1:]
    elif defect == "missing_c10":
        callers.pop()
    elif defect == "different_binary":
        candidates[0]["sha256"] = "b" * 64
    else:
        candidates.append(dict(candidates[0]))
    with pytest.raises(RuntimeError):
        _select_provider(candidates, callers)


def test_actual_elf_relocations_distinguish_identical_mapped_instances(tmp_path):
    """Compile original stub libraries whose CUDA-named functions must never run."""
    compiler = shutil.which("cc")
    if not compiler or not shutil.which("readelf") or platform.system() != "Linux":
        pytest.skip("requires Linux, a C compiler and binutils for the actual loader experiment")
    if platform.machine() not in {"aarch64", "x86_64"} or ctypes.sizeof(ctypes.c_void_p) != 8:
        pytest.skip("qualified ELF64 ABI only")
    first, second, torch = (tmp_path / part for part in ("first", "second", "torch"))
    for directory in (first, second, torch / "lib"):
        directory.mkdir(parents=True)
    (torch / "__init__.py").touch()
    symbols = (
        "cudaGraphLaunch",
        "cudaGraphInstantiateWithFlags",
        "cudaStreamBeginCapture",
        "cudaStreamEndCapture",
        "cudaStreamGetCaptureInfo",
    )
    source = tmp_path / "runtime.c"
    source.write_text("#include <stdlib.h>\n" + "\n".join(f"int {name}(void) {{ abort(); }}" for name in symbols))
    runtime = first / "libcudart.so.13"
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(source), "-Wl,-soname,libcudart.so.13", "-o", str(runtime)],
        check=True,
        capture_output=True,
    )
    shutil.copyfile(runtime, second / runtime.name)
    for name, imports in (("libtorch_cuda.so", symbols[:4]), ("libc10_cuda.so", symbols[4:])):
        source = tmp_path / (name + ".c")
        source.write_text(
            "\n".join(f"extern int {symbol}(void);" for symbol in imports)
            + "\nint never_call(void) { return "
            + "+".join(f"{symbol}()" for symbol in imports)
            + "; }\n"
        )
        subprocess.run(
            [
                compiler,
                "-shared",
                "-fPIC",
                str(source),
                str(runtime),
                f"-Wl,-rpath,{first}",
                "-o",
                str(torch / "lib" / name),
            ],
            check=True,
            capture_output=True,
        )
    # An isolated process ensures temporary mapped stubs cannot affect any other
    # test's real framework imports. Merely looking up pointers must not call them.
    script = r"""
import ctypes, json, os, sys
from pathlib import Path
from types import SimpleNamespace
from collector.glm53flash_graph_nodes import _library
root = Path(sys.argv[1])
sys.modules['torch'] = SimpleNamespace(__file__=str(root/'torch/__init__.py'))
handles = [ctypes.CDLL(str(root/'torch/lib'/name)) for name in ('libtorch_cuda.so','libc10_cuda.so')]
handles.append(ctypes.CDLL(str(root/'second/libcudart.so.13')))
library, receipt = _library('cudart')
assert Path(receipt['path']) == root/'first/libcudart.so.13'
binding = receipt['provider_binding']
assert len(binding['candidates']) == 2
assert len({item['load_bias'] for item in binding['candidates']}) == 2
assert len({item['sha256'] for item in binding['candidates']}) == 1
assert sum(len(item['relocations']) for item in binding['callers']) == 5
print(json.dumps({'status':'ACTUAL_ELF_STUB_PROVIDER_VERIFIED_NO_CUDA', 'selected':receipt['path']}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(glm53flash_graph_libraries.__file__).resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["status"] == "ACTUAL_ELF_STUB_PROVIDER_VERIFIED_NO_CUDA"
