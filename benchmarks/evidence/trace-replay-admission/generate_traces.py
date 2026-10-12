# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the seeded synthetic traces used by the trace-replay admission evidence."""

import argparse
import json
import random
from pathlib import Path

BLOCK = 64


def dynamo(path: Path, requests: int, *, agentic: bool, seed: int = 7) -> None:
    """Standard or agentic Dynamo request_end rows sharing one of 16 prompt prefixes."""
    rng = random.Random(seed)
    prefixes = [[rng.getrandbits(62) for _ in range(rng.randint(40, 120))] for _ in range(16)]
    time = 1_700_000_000_000
    with path.open("w") as out:
        for index in range(requests):
            hashes = prefixes[index % len(prefixes)] + [rng.getrandbits(62) for _ in range(rng.randint(40, 160))]
            input_length = len(hashes) * BLOCK - rng.randint(0, BLOCK - 1)
            time += rng.randint(1, 20)
            row = {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": time + 500,
                "request": {
                    "request_id": f"req-{index}",
                    "output_tokens": rng.randint(64, 512),
                    "request_received_ms": time,
                    "total_time_ms": 500,
                    "replay": {
                        "trace_block_size": BLOCK,
                        "input_length": input_length,
                        "input_sequence_hashes": hashes,
                    },
                },
            }
            if agentic:
                row["agent_context"] = {"session_id": f"s-{index % 512}"}
            out.write(json.dumps(row, separators=(",", ":")) + "\n")


def mooncake(path: Path, requests: int, *, seed: int = 3) -> None:
    rng = random.Random(seed)
    with path.open("w") as out:
        for index in range(requests):
            hashes = [rng.getrandbits(40) for _ in range(rng.randint(80, 280))]
            row = {
                "timestamp": index * 10,
                "input_length": len(hashes) * BLOCK,
                "output_length": rng.randint(64, 512),
                "hash_ids": hashes,
            }
            out.write(json.dumps(row, separators=(",", ":")) + "\n")


def agentic_mooncake(path: Path, requests: int, *, seed: int = 5) -> None:
    rng = random.Random(seed)
    header = {
        "schema": "dynamo.agentic_mooncake",
        "version": 2,
        "block_size": BLOCK,
        "hash_id_scope": "local",
        "source": {"format": "bench", "digest": "x"},
    }
    with path.open("w") as out:
        out.write(json.dumps(header) + "\n")
        for index in range(requests):
            hashes = [rng.getrandbits(40) for _ in range(rng.randint(80, 280))]
            row = {
                "request_id": str(index),
                "play_id": str(index % 512),
                "session_id": str(index % 512),
                "model": "m",
                "input_length": len(hashes) * BLOCK,
                "output_length": rng.randint(64, 512),
                "hash_ids": hashes,
                "not_before_ms": 0,
                "dependencies": [],
            }
            out.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dynamo(args.output_dir / "dynamo-standard-16k.jsonl", 16384, agentic=False)
    dynamo(args.output_dir / "dynamo-standard-65k.jsonl", 65536, agentic=False)
    dynamo(args.output_dir / "dynamo-agentic-16k.jsonl", 16384, agentic=True)
    mooncake(args.output_dir / "mooncake-16k.jsonl", 16384)
    agentic_mooncake(args.output_dir / "agentic-mooncake-16k.jsonl", 16384)


if __name__ == "__main__":
    main()
