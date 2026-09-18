# Support-matrix model roster

The generation roster for the published
[Legacy AIC Support Matrix](https://ai-dynamo.org/aisimulate/support-matrix/)
is curated separately from the model configurations bundled with AISimulate.

A bundled model remains available for explicit SDK and CLI use even after it
is retired from default matrix generation. This keeps historical workflows and
offline model loading working without requiring every superseded release to
occupy the full model/system/backend/version cross-product.

## Inclusion policy

Default matrix entries should represent at least one of the following:

- a current flagship model or checkpoint;
- a distinct architecture or operation pipeline;
- a precision variant with materially different runtime or performance-data
  requirements; or
- a compatibility case that protects an actively supported backend path.

Before adding a model, confirm that its runtime path is viable on at least one
matrix system/backend/version combination. Deterministic unsupported paths
must remain explicitly classified rather than reported as passing.

## Current NVIDIA NVFP4 additions

The current expansion covers Qwen3.5 and Qwen3.6, Gemma 4, Kimi K2.6 and K2.7
Code, DeepSeek V4 Flash and Pro, Nemotron-3 Nano, Nemotron-3.5 Lightning, and
MiniMax M3. Their bundled Hugging Face configs and quantization metadata remain
the source of truth for architecture and precision selection, including
mixed-precision checkpoints.

## Multimodal encoder coverage

The support matrix automatically exercises a checkpoint's vision encoder when
the checkpoint declares one through a non-empty `vision_config` or AIC's
multimodal architecture registry, and AIC implements that encoder. The
canonical workload is **one 1024 x 1024 image per request**. The same
image-bearing run covers the language backbone; multimodal checkpoints do not
receive a second, redundant text-only run.

An encoder-supported PASS means that the agg or disagg run used the canonical
image workload and produced strictly positive encoder latency and encoder
memory for every result row. `ImageHeight`, `ImageWidth`, and `NumImages` in the
generated CSV, plus the matching replay-command arguments, record that workload.

If a checkpoint declares `vision_config` but AIC cannot normalize its encoder
configuration, the row fails with an `ENCODER_UNSUPPORTED` reason. If
normalization succeeds but execution emits no positive encoder evidence, the
row fails with `ENCODER_NOT_EXERCISED`. It must not inherit PASS from a
successful text-backbone-only estimate. Text-only checkpoints keep their
existing workload and leave the image metadata empty.

If AIC implements the encoder but the system/backend/version database has no
`encoder_attention` perf data, the image workload cannot be answered there. The
row is classified `FRAMEWORK_INCOMPATIBLE` with an `ENCODER_DATA_UNAVAILABLE`
reason and a replayable preflight command; the text backbone is not run, the
row is not retried, and the image metadata records the canonical workload that
could not be exercised.

## Retired from default generation

The following bundled configs remain usable explicitly but are superseded in
the default matrix:

- GLM-5 and GLM-5.1 in BF16, FP8, and NVFP4; GLM-5.2 remains.
- MiniMax-M2.5 in BF16 and NVFP4; MiniMax-M2.7 and MiniMax-M3 remain.
- Llama-3.3-Nemotron-Super-49B-v1 and Nemotron-H-56B-Base-8K;
  Nemotron-3 remains.

The source of truth is `SupportMatrixHFModels` in
`aisimulate_core.sdk.common`. `DefaultHFModels` remains the bundled config
inventory for compatibility and offline loading.
