# AgentX performance fixture

`agentx.jsonl` contains four complete plays from
[SemiAnalysis cc-traces-weka-062126-256k](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k),
revision `8fecd2fc56694469f758f0afbbb6335ad3043740`, original file `traces.jsonl`.
The upstream dataset authors are SemiAnalysis. The pinned dataset card declares
Apache-2.0; it contains no separate copyright or NOTICE statement.
`DATASET_CARD.md` preserves that card and `LICENSE` contains Apache-2.0.

The four images in `plots/` are unmodified copies of the same paths in the
[pinned upstream dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/tree/8fecd2fc56694469f758f0afbbb6335ad3043740/plots):
`plots/distributions_log.png`, `plots/distributions_linear.png`,
`plots/subagent_distributions_log.png`, and
`plots/subagent_distributions_linear.png`. They illustrate the full upstream
dataset described by the card, not this benchmark's four-play subset, and are
covered by the same Apache-2.0 license.

Modified by NVIDIA for this benchmark: selected four complete plays and normalized
JSON whitespace. The original play `002001296e8a8c38ad9d7cc436d691afc602`
retains its existing rows-API representation. The three lowest-ID other complete
plays with input plus output lengths at most 262,144 were selected from the
immutable `traces.jsonl`, without changing their request values, hashes,
dependencies, or timestamps.

[`agentx.json`](agentx.json) records the four play IDs, checksum, and expected
counts: **255 requests**, **27,418,752 input tokens**, and **286,917 output tokens**.
Both backend cases replay this file with four lanes; no plays are duplicated.

The original play was acquired through the Hugging Face rows API, which rounds
some floating-point values compared with the raw JSONL. Its values are retained.
The additional plays come directly from the pinned file. CI reads only the local
fixture and manifest.
The benchmark projects all source model labels onto the configured Qwen3.5
model; it measures AISimulate host cost, not fidelity to the original service.
