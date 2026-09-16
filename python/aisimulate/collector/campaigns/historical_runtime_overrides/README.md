# Historical B200 / vLLM 0.25.0 runtime overrides

These are byte-preserved `git diff --binary` records from the four original
campaign source checkouts. They modify only the runtime manifest and the two
compatibility links repaired for those sparse checkouts. They do not contain
operator timing changes. Source commits, patch SHA-256 values and runtime-file
SHA-256 values are recorded in the packaged
`b200_sxm/vllm-0.25.0-collection-report.json`.

They document the historical execution environment. The new campaign runner
**does not apply these patches** and refuses dirty source trees. For new runs,
use the committed runtime manifest with its explicit hash and a clean source
revision supporting that interface. Historical measurements retain their actual
source refs; neither these patches nor the review fix imply a new measurement.
