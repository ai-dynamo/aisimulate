# One-time SGLang KV-routing backtest

Use the new four-lane, four-play AgentX SGLang case from PR #398 with two prefill and two decode workers. Build and install AISimulate and the pinned Dynamo runtime separately for each source revision.

- Fixed baseline: `988edba8732af6f45990d0cb567c001119121566` (#397 head).
- Slow candidate: `9093044260aa703c7e231ea54021940083843ada` (#397 merge base).
- Dynamo: `def3b79b15c266805540a678dd400aeb6ccada1d`.
- Benchmark adapter: inherited unchanged from `1eac344d84734a06f9ac5d29e4e29f17daa2820e`, overlaid onto both worker checkouts after building the exact source wheels.

Run one five-round comparison. Keep the existing threshold: more than 10% and 100 ms slower in at least four rounds. No warmups, repetitions, per-request capture, profiling, or workload changes. A detected regression is the expected result. Retain invalid results or missed detection without tuning.

This branch changes only the one-time workflow and this record. It is not intended for merge. Qualification and normal CI use the production workflow on PR #398.
