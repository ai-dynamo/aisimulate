# HF E2E evidence migration

- Inputs: DB release `db-dump/2026-09-28` and private HF revision
  `8ec77d043d236ae92aa0cee6a9896f8195ea7718`. Exact table hashes are in
  `scripts/e2e_accuracy/source/manifests/hf_evidence_bindings.json`.
- HF v2 selects 1,974 STP and 881 MTP points; 2,829 benchmark IDs overlap the DB
  v3 cohort. Every shared pair passed the non-transport identity comparison.
- The 996 reviewed bindings cover 35 launcher cases, 89 runtime archive cases
  and 872 MTP cases. No raw measurements or logs are included in this report.
- A direct resolver audit found identical recipes for all 35 affected MiniMax
  NVFP4/B300 launcher cases. All 69 parsed runtime records and 20 archived
  workloads match the DB evidence after omitting transport provenance.
- MTP bindings retain log-reuse counts from the complete DB table: 804 points
  have one use; 58 have shared logs; 10 have no identified log. Missing/shared
  observed acceptance remains unresolved. Synthetic acceptance retains its
  explicit configured label. The remaining nine HF-only MTP points have no
  reviewed historical binding.
- These checks establish input/evidence equivalence, not numerical replay
  parity. The combined preview must finish before making that claim. Historical
  DB-only rows absent from the HF current view remain a coverage difference.

The source data is attributed to SemiAnalysis InferenceX. The bindings reuse the
existing reviewed runtime/launcher evidence and its source hashes; they neither
replace that evidence nor grant redistribution rights for the private cache.
