# DeepSeek V4.1 eager startup rationale

- Source: [original collector startup report](https://github.com/ai-dynamo/aisimulate/blob/2718db360b3d832062d63e28c2a3fae58a21874c/python/aisimulate/collector/fpm_forward/runtime/dsv41/README.md#L3-L8), commit `2718db360b3d832062d63e28c2a3fae58a21874c`.
- Scope: the reported GB200 TP4 bring-up loaded the model, then graph compilation exceeded the allocation time limit and the run was terminated. The same change introduced the collector's eager-only requirement.
- Evidence limit: this is a historical source report, not a new GPU reproduction or a confirmed compiler crash. It does not establish that every compiled configuration fails.
- Current policy: the DeepSeek V4.1 collection path uses eager execution; compiled and CUDA-graph paths need separate qualification.
