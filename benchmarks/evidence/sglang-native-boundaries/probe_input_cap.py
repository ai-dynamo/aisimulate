# SPDX-License-Identifier: Apache-2.0
# Behavioral harness for SGLang schedule_policy.py at the immutable revision below.
# Copyright SGLang Team and contributors, Apache-2.0. Modified as a CPU-only
# selection probe; no upstream implementation is vendored. See THIRD_PARTY_NOTICES.md.
"""Execute pinned native selection with unrelated capacity/tile gates allowed.

This is a source-level check of the input-cap branch, not a full server run.
Requires a local SGLang Git checkout containing SOURCE_REV; no GPU or imports
from the installed serving runtime are needed.
"""

import argparse
import ast
import hashlib
import json
import subprocess
from collections import namedtuple
from enum import Enum
from types import SimpleNamespace

SOURCE_REV = "baa9befa381511cd4d56f83980951e5a45fffc6d"
SOURCE_PATH = "python/sglang/srt/managers/schedule_policy.py"
SOURCE_URL = f"https://github.com/sgl-project/sglang/blob/{SOURCE_REV}/{SOURCE_PATH}"
SOURCE_SHA256 = "03ba021fdd225cc4e7d0d5eb822a725fe6e3907e40157ec6461390f23b7f2cb9"


class AddReqResult(Enum):
    CONTINUE = 0
    OTHER = 1
    NO_TOKEN = 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-checkout", required=True)
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "-C", args.sglang_checkout, "show", f"{SOURCE_REV}:{SOURCE_PATH}"]
    )
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PrefillAdder"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_select_prefill_admission", "budget_state"}
    ]
    assert len(methods) == 2
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    scope = {
        "AddReqResult": AddReqResult,
        "CLIP_MAX_NEW_TOKENS": 4096,
        "_PrefillAdmission": namedtuple(
            "Admission", "prefix_len extend_len max_new_tokens is_chunked"
        ),
    }
    # Only the two selected methods of the hash-verified, pinned Git blob run.
    exec(compile(module, SOURCE_URL, "exec"), scope)  # noqa: S102

    class Harness:
        _select_prefill_admission = scope["_select_prefill_admission"]
        budget_state = scope["budget_state"]
        page_size = 128
        exact_chunk_fill = False
        dllm_config = None
        rem_mamba_slots = None
        memory_budget = SimpleNamespace(has_capacity=lambda: True)

        def ceil_paged_tokens(self, n):
            return (n + 127) // 128 * 128

        def _check_prefill_budget(self, *_args, **_kwargs):
            return True, self.rem_chunk_tokens

        def _check_prefill_tile_budget(self, _tokens):
            return None

    rows = []
    for cap, chunk, prompts, expected in [
        (64, 4096, [128], [128]),
        (256, 4096, [128, 256, 128], [128, 256]),
        (256, None, [128, 256, 128], [128]),
    ]:
        runner = Harness()
        runner.rem_input_tokens, runner.rem_chunk_tokens = cap, chunk
        runner.can_run_list = []
        admitted = []
        for prompt in prompts:
            req = SimpleNamespace(
                prefix_len=0,
                full_untruncated_fill_ids=range(prompt),
                sampling_params=SimpleNamespace(max_new_tokens=1),
            )
            result = runner._select_prefill_admission(
                req,
                total_tokens=prompt + 129,
                host_hit_length=0,
                swa_host_hit_length=0,
                truncation_align_size=None,
            )
            if isinstance(result, AddReqResult):
                break
            admitted.append(result.extend_len)
            runner.can_run_list.append(req)
            charge = runner.ceil_paged_tokens(result.extend_len)
            runner.rem_input_tokens -= charge
            if runner.rem_chunk_tokens is not None:
                runner.rem_chunk_tokens -= charge
            if runner.budget_state() != AddReqResult.CONTINUE:
                break
        assert admitted == expected, (cap, chunk, admitted, expected)
        rows.append(
            {
                "input_cap": cap,
                "chunk_limit": chunk,
                "prompts": prompts,
                "admitted": admitted,
            }
        )
    print(
        json.dumps(
            {
                "source": SOURCE_URL,
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "scope": "native selection and budget_state; capacity/tile gates allowed; no full-server claim",
                "cases": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
