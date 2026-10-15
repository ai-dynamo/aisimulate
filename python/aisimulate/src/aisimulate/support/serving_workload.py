# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze native AIPerf Weka chat requests in its isolated, pinned environment.

Executed as a fixed package resource script with a structured JSON input. No
AISimulate imports or GPU model are needed in the AIPerf environment. Upstream
Weka reconstruction supplies content/history, and upstream Session/ChatEndpoint
supplies the complete request payload. Native Mooncake replay then sends that
payload verbatim with the original timestamp and end-to-start delay.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def materialize(input_path: Path) -> None:
    from aiperf.cli import app
    from aiperf.common import random_generator
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.config.flags.converter import convert_cli_to_aiperf
    from aiperf.dataset.generator.prompt import PromptGenerator
    from aiperf.dataset.loader.weka_trace import WekaTraceLoader
    from aiperf.endpoints.openai_chat import ChatEndpoint
    from aiperf.workers.session_manager import UserSession
    from transformers import AutoTokenizer

    inputs = json.loads(input_path.read_text())
    trace = json.loads(Path(inputs["trace"]).read_text())
    output = Path(inputs["output"])
    # Parsing the real producer options preserves the native loader's options.
    _, bound, _ = app.parse_args(inputs["arguments"], exit_on_error=False, print_error=False)
    config = convert_cli_to_aiperf(bound.arguments["cli_config"]).benchmark
    random_generator.reset()
    random_generator.init(0)
    # The pinned AIPerf offline Hub resolver treats a local directory as a Hub
    # ID. Load the supplied local files directly, then use its native wrapper.
    inner = AutoTokenizer.from_pretrained(inputs["tokenizer"], local_files_only=True, trust_remote_code=False)
    tokenizer = Tokenizer._build_with_kwargs(inner, resolved_name=inputs["tokenizer"])
    template = inner.get_chat_template()
    if not isinstance(template, str) or not template:
        raise ValueError("the target tokenizer must define the serving chat template")
    generator = PromptGenerator(prompts=None, prefix_prompts=None, tokenizer=tokenizer)
    loader = WekaTraceLoader(
        filename=inputs["trace"],
        run=SimpleNamespace(cfg=config, random_seed=0),
        prompt_generator=generator,
    )
    conversations = loader.convert_to_conversations(loader.load_dataset())
    if len(conversations) != 1 or len(conversations[0].turns) != len(trace["requests"]):
        raise ValueError("native AIPerf reconstruction changed the selected play's request graph")
    conversation = conversations[0]
    if conversation.branches:
        raise ValueError("native AIPerf reconstructed branches; this complete play is not yet qualified")
    session = UserSession(
        x_correlation_id="preparation",
        num_turns=len(conversation.turns),
        conversation=conversation,
        context_mode=conversation.context_mode or loader.get_default_context_mode(),
    )
    if session.should_store_response():
        raise ValueError("request materialization requires prerecorded history, independent of generated responses")
    endpoint_info = SimpleNamespace(primary_model_name=inputs["model"], endpoint=config.endpoint)
    endpoint = ChatEndpoint(endpoint_info)
    tokenization = []
    rows = []
    for index, turn in enumerate(conversation.turns):
        if turn.source_outer_idx != index:
            raise ValueError("native AIPerf reordered the selected play")
        session.advance_turn(index)
        payload = endpoint.format_payload(
            SimpleNamespace(
                turns=session.turn_list,
                model_endpoint=endpoint_info,
                system_message=conversation.system_message,
                user_context_message=None,
            )
        )
        # Explicit defaults make target chat tokenization reviewable. They do
        # not override the server's template or enable remote template code.
        payload.update(add_generation_prompt=True, continue_final_message=False)
        if payload.get("ignore_eos") is not True:
            raise ValueError("prepared request must force its requested output length")
        tokens = inner.apply_chat_template(
            payload["messages"], tokenize=True, add_generation_prompt=True, chat_template=template, return_dict=False
        )
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
            raise ValueError("target chat tokenizer did not return an explicit nonempty token sequence")
        rows.append(
            {
                "session_id": trace["id"],
                "timestamp": turn.timestamp,
                "delay": turn.delay,
                "output_length": trace["requests"][index]["out"],
                "payload": payload,
            }
        )
        tokenization.append({"turn_index": index, "input_token_ids": tokens, "input_length": len(tokens)})
    (output / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output / "tokenization.json").write_text(
        json.dumps(
            {
                "chat_template": template,
                "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                "chat_template_kwargs": {"add_generation_prompt": True, "continue_final_message": False},
                "content": "native AIPerf Weka reconstruction from its packaged sonnet corpus, seed 0",
                "history": "prerecorded synthetic history; generated responses do not alter future inputs",
                "requests": tokenization,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    materialize(Path(sys.argv[1]))
