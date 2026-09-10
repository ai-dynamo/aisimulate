# Original text stress strata

These two texts were authored for this verification study. They are not copied
from a benchmark or production traffic. They carry the repository's Apache-2.0
license; copyright 2026 NVIDIA CORPORATION & AFFILIATES.

`field-notes.txt` is continuous English narrative. `service-records.txt` mixes
English technical prose, Chinese explanations, identifiers and quantities.
They broaden vocabulary and token distributions beyond the initial short
fixture, without making a claim of production representativeness.

Tokenize with the actual pinned serving tokenizer, retain text and encoded
corpus hashes, and draw fresh deterministic offsets. Preserve the same request
geometry when comparing corpora. Within the locality pair, send an identical
real-text prompt twice concurrently for the repeated condition and two distinct
real-text offsets for its control. Observe actual KV reuse. A content-dependent
latency difference includes routing and caching effects and cannot be attributed
uniquely to Engram without a separate isolating experiment.

For each corpus, repeat `boundary-129`, `heldout-single-192`,
`engram-repeated-text`, and `engram-distinct-text`. These are eight additional
corpus/scenario strata per runtime profile, separate from geometric holdouts.
Run a separate warm-up, ten pilot trials, then freeze at least twenty fresh
main trials using the same variance rule as the primary study. Keep the two
corpora separate in statistics. Use seeds outside the primary study; pair them
across decoder OFF/ON. Record completed and missing strata if the campaign
cannot finish them. Current status is prepared inputs, not collected results.
