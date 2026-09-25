# Reduced native classification evidence

`glm53flash_native_classification_observed.json` contains metadata observed by
this project's unchanged `2c3b741fe88d2a95d4faa471233f62a5cebaf892` producer in
failed native runs 626499 and 626500. Its `originals` object records each full
original artifact's path, byte count and SHA256. The full original failure
records remain unchanged outside the repository. This is our observed run data,
licensed under the repository's Apache-2.0 license; no external implementation
or CUDA headers were copied.

The NONE reduction preserves the original outer execution scope, the first
scoped call of each of the three function-control API names, and one actual
kernel launch with its correlated activity. The graph reduction selects the
first original node of each observed type 0/1/2, their direct clone callbacks,
the exact executable creation callback and edges between selected nodes.
Provider identities and source D2D GetParams evidence remain original. Unselected
nodes, scopes and events are omitted, so this is explicitly TEST_ONLY and does
not satisfy a complete model inventory or performance qualification.

Tests query synthetic native type results and construct explicitly TEST_ONLY
replay intervals to exercise strict source/clone/replay joins. Those results are
not part of the original evidence. The complete original NONE trace was also
reparsed separately as a local diagnostic; neither operation admits the failed
run or changes its measurement provenance.

API semantics are referenced, not copied, from NVIDIA CUDA Toolkit 13.0.2:

- [Driver Entry Point Access](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-runtime-api/group__CUDART__DRIVER__ENTRY__POINT.html)
- [Execution Control](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-runtime-api/group__CUDART__EXECUTION.html)
- [Runtime/Driver Interactions](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-runtime-api/group__CUDART__DRIVER.html)

The existing root THIRD_PARTY_NOTICES.md CUDA/CUPTI and Kineto entries cover the
independently authored parser's API/trace-format references. The immutable
Kineto revision is `094d3c1d072362d0a919a77299459eee94f97931`, pinned by Torch
`cf30153c4c131c8164ee7798e5022d810682e2cb`.
