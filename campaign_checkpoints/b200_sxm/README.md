# B200 MSA context resume seed

These are preserved original-image results from GitLab pipeline76760660/job510058975, AISim source9b4de9dbc3fa721af0a3965d16f04d3c5dce89bf. The seed contains184 raw measured rows and the byte-exact checkpoint:184 done,44 failed. No generation results are included.

`msa_context_seed_provenance.json` maps every original row and failure to its source/runtime. The frozen image and manifest are unchanged. MSA measurement/dispatch code is byte-identical; the newer helper only repairs first-finalization duplicate recovery.

Resume only with the reviewed16,704-case context selection outside the confirmed native score-address limit. It restores184 successes and attempts16,520 remaining cases. Preserve all44 failed records. The collector may end with `UnresolvedFailures` even if every newly attempted case passes; this is retained failure evidence, not permission to retry or erase those cases. The original17,568-case context ledger also retains820 unattempted cases blocked by the same native address limit.

This research seed is not a complete production dataset. Do not promote defaults or publish it independently. Keep generated data and per-case attempt accounting separate; the seed provenance must accompany resumed artifacts and any draft data publication.
