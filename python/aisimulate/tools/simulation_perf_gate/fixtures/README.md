# AgentX performance fixture

`agentx.jsonl` contains the complete play
`002001296e8a8c38ad9d7cc436d691afc602` from
[SemiAnalysis cc-traces-weka-062126-256k](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k),
revision `8fecd2fc56694469f758f0afbbb6335ad3043740`, original file `traces.jsonl`.
The upstream dataset authors are SemiAnalysis. The pinned dataset card declares
Apache-2.0; it contains no separate copyright or NOTICE statement.
`DATASET_CARD.md` preserves that card and `LICENSE` contains Apache-2.0.

Modified by NVIDIA for this benchmark: selected one complete play and normalized
JSON whitespace. No request values, hashes, dependencies, or timestamps changed.
There are 129 requests, four subagent groups, 13,391,168 input tokens, and
114,540 output tokens. Maximum input plus output is 255,034 tokens.
The SHA-256 of the checked-in JSONL is
`d65b573413396bb689cf7e1d5c85c50ea970ad7ad5714c0a78d4f75102e5d86d`.

The existing `scripts/qualify_weka_samples.py` pins this play and the upstream
revision. Acquisition used the Hugging Face rows API after checking that the
repository revision matched that pin. CI only reads this local file.
The benchmark projects all source model labels onto the configured Qwen3.5
model; it measures AISimulate host cost, not fidelity to the original service.
