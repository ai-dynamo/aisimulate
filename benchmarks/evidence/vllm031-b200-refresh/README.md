# vLLM 0.31.0 B200 refresh evidence

- Draft scope: 300 fresh MLA pre-BMM cases. Post-BMM remains under qualification; the shared table is partial.
- Collection: [pipeline 76699456](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-auto-collector/-/pipelines/76699456), job509597082, source079052e6f73984fd73540de724d0d99b4de34442.
- Pinned runtime: adjacent manifest; official vLLM0.31 image digest3fb023cc987cc7c2bca7ca14c1e0aff5ae06bd9431bf07773731ce7e8a682460.
- Coverage: 300 planned, 300 passed, zero failed/unattempted/duplicate identities. Historical0.25 matched cases also have zero failures.
- Numerical probes: Slurm2384643, three native eager/CUDA-graph shapes versus FP64; all passed.
- Consumer validation: all 300 exact rows match both Python orchestration and native op queries with `shared_layer=False`. The old public `query()` API has retired on this main revision. Run `python/aisimulate/.venv/bin/python benchmarks/evidence/vllm031-b200-refresh/validate_prebmm.py`.
- Data sanity: all seven collector-data checks pass. Remaining B200 campaign scopes are pending. No defaults are promoted; this is op measurement evidence, not end-to-end accuracy evidence.
- Reuse-manifest generation scanned the full tree; only the affected `mla_bmm_perf/vllm_torch_bmm` group is updated here. Existing unrelated manifest drift is recorded separately.
