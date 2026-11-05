# Fresh paired seeded-runtime accuracy comparison

Raw results and reproduction scripts are preserved in the
[immutable qualification checkout](https://github.com/ai-dynamo/aisimulate/tree/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_seeded_revision_20261103).
Run the recorded reproduction commands from that checkout with their pinned
external inputs; this review copy retains the report and archive links.

The candidate matches the merged PR #658 baseline for every prediction and
admission in the pinned 36-case C/BC campaign and the pinned HF smoke case.
All 36 cases were retained, including all thirteen M3 cases. This demonstrates
preserved behavior on these populations, not improved accuracy or a new Gym
protocol.

Both builds used the unchanged evaluator from clean revision
`9f4819e2b77afb4798bc8133c42ccc1d1b045e5c`, the same existing recipe and installed
dependency versions. The recorded baseline native SHA256 is
`d88bc55746664e144057e57e17ee5357c08667ea52861627a7ad6a9949f74cd4`;
the tested candidate is
`7fe75475aca65819a9174942ded464ff49060f082357753bcde5286a2028a0d0`.
The candidate was a working-tree build, not the old HEAD named by its checkout.
The comparison JSON files retain exact runtime source inventories, loaded paths,
dependencies, evaluator hashes, harness hash and input identities.

| Population | Cells | Scored observations | Predicted | Coverage | Prediction-weighted MAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full36 C | 36 | 4,261,480 | 3,846,981 | 90.273356% | 6.814360% |
| Full36 BC | 36 | 2,130,748 | 2,130,726 | 99.998967% | 1.665027% |
| HF smoke C | 1 | 10,270 | 3,239 | 31.538462% | 2.592502% |
| HF smoke BC | 1 | 5,135 | 5,135 | 100.000000% | 1.163558% |

Every value in the table is identical for baseline and candidate. All 6,407,633
scored records have identical prediction availability and floating-point bits.
There are zero prediction errors, tuning errors or failed cells. Training/scoring
membership, observation IDs, targets, event times, workers, native workload
classes, admission decisions and hashes of the complete admitted rank payloads
also match. The 16,818 full36 and 51 HF update submissions match exactly.

MAPE is weighted by the number of available predictions, not by case; coverage
uses every scored observation. Unavailable predictions remain visible. The
existing C initial-fit policy and sparse-store limitations are unchanged. The
HF result covers only the selected three-minute Kimi-K3 configuration prefix,
not the full HF dataset. Synthetic example functionality is separate evidence.

## Inputs and execution

`input-verification.json` rehashes all 42 unique local source files, the full36
spec and the HF catalogue against the pinned expected values. The full36 spec
SHA256 is `de9c1a999e23432354d9a046f56b3976f2fae9eae66d13ff67fbf2382adce888`.
The HF snapshot is `5487a4599a7fbc012c07bcd3699754bdf4a8bef7`, catalogue SHA256
`3e4c06950606af4b84d624f85191e344f40e3a54de7ed926ddfa851243c956af`, with selected
configuration `data/moonshotai--Kimi-K3/gb300/vllm/0.1.dev19262/tep8` and 180-second
prefix. No input was downloaded or regenerated for this comparison.

`commands.json` records both fresh interpreter commands for each population,
environment and private log path. Each selected interpreter loaded its own SDK
and native extension while both imported the same clean evaluator. Numerical
thread counts were one. Concurrent unrelated qualification/build work means
these runs provide no simulator host-cost evidence.

The exact executed `qualify_gym.py` and full36 specification are retained here
as `qualify_gym.py` and `inputs/cases.json`; their SHA256 values are unchanged.
The original executed paths in `commands.json` and result identities remain
execution facts. `reproduction-commands.json` uses the relocated copies with
new output destinations and is explicitly not an executed-run record. It requires
the recorded builds; another native digest creates separate new qualification.
The original input paths in the copied specification remain absolute and refer
to the same verified external files. The relocation map records immutable source
provenance without carrying superseded result directories.

`summarize_accuracy.py` verifies every result/trace digest and compares complete
prediction/admission traces before producing:

- `full36/comparison.json`: 72 cells and exact behavioral checks.
- `full36/learning-curves.csv`: 3,372 paired minute/implementation rows.
- `hf-smoke/comparison.json` and `learning-curves.csv`: the two smoke cells.

Raw observation and prediction traces remain under
`/private/tmp/gbdt-seeded-accuracy-20261103/{baseline,candidate}/`; the repository
contains only input verification, aggregate results, source identities and
execution records. Historical qualification directories were not modified.

The later support-assessment dependency verification and sweeper pin-consumer
fixes are outside this unchanged evaluator/native prediction/update path. They
require their own focused tests. These receipts identify the actually tested
7fe75475 build and must not be relabeled as measurements of a later binary;
final delivery must map any later changes to separately recorded validation.
