# FastAFD measured MoE profiles

AISimulate reads external measurements; it does not import, install, or run FastAFD when loading a profile. A profile uses the `aisimulate.fastafd-moe-stage.v1` schema and exact-match lookup. Missing measurements are errors, not estimates.

The profile must identify `https://github.com/hao-ai-lab/FastAFD` and an immutable 40-character source commit. Every entry identifies the model, system, stage, topology, workload, MoE precision, and backend. It also records the measurement method and tool version, p50 sample count, the procedure and its SHA-256, the raw artifact and its SHA-256, and validation evidence. The only accepted latency scope is `complete_moe_stage`.

The producer must verify that the measured interval covers the complete modeled MoE stage under the recorded topology. A client response time, coordinator interval, or per-rank GPU activity span is not interchangeable with that stage latency. FastAFD's current official graph-level traces do not by themselves establish a complete per-step MoE stage measurement; no profile should be published from them without additional validated evidence.

Collect performance baselines separately from CUPTI profiling runs. Keep raw traces, procedure, workload, GPU and NIC topology, selected communication devices, and clocks alongside the profile so that each measurement can be reproduced and reviewed. The hashes identify the declared evidence; loading a profile does not independently verify those files or the remote commit.
