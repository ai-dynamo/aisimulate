# E027: forced batch-shared acceptance diagnostic

- Hypothesis: the original SGLang run forces mean accepted length 2.99 with one
  floor/ceil draw per verification batch. AISim samples each request independently.
  This correlation difference may change stalls and request TPOT despite an equal mean.
- Experiment: change only sampling scope on pinned research baseline
  `eddb21246e0a371fe542f7cab4034b58173bce63`; retain acceptance rates `[1, 0.99]`,
  seed, wheel data, measured settings, source corpus, and frozen request identities.
  Do not include the E025 bigram-cache correction in this ablation.
- Predeclared cohort: development lanes 1/8/14 with original per-metric eligibility.
  Lost matches invalidate a win; count recovered matches separately. Hold 4/10/13
  out of tuning. Pool individual request errors and inspect every development lane.
- Qualification: this reproduces the forced policy's marginal distribution and
  batch correlation, not Torch's exact RNG stream. It does not claim ordinary
  speculative decoding should share acceptance. This branch is diagnostic only.
- Source: SGLang v0.5.20 commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`,
  `python/sglang/srt/speculative/spec_utils.py` (`sample_simulated_acc_len` and
  `generate_simulated_accept_index`). Runtime source SHA256
  `f7b6db9995357dd99a221b763a642fda1a26b5ec21bc33cb7e350f2a5bcd6343`.
  Runtime counter evidence: 394 multi-request groups, 390 shared three-token
  increments, four shared two-token increments, zero mixed groups.
- Resource cost: CPU only; no new GPU allocation. Test with 50% conditional
  acceptance to expose both batch outcomes before replaying the measured 99% rate.
- Conclusion: independently audited, no lost frozen observations (4169 TTFT,
  4166 TPOT). TTFT MAPE worsens 88.4103→91.8322%; TPOT worsens
  68.2526→104.8737%. Per-lane TTFT/TPOT: 1 lane 73.6772/18.8789%,
  8 lanes 64.9665/53.3170%, 14 lanes 109.8970/146.3360%. Not an accuracy win;
  reject as a sufficient remedy and preserve the ordinary default. Torch RNG
  sequence differences remain explicit. Do not tune acceptance rates to these scores.
- Evidence: E027-independent-dev-audit.json and E027-provenance.json in the
  private campaign; wheel source remains the original diagnostic commit
  `e55917de8771e52da9ae1df38823d6bc12001ea7`. This follow-up only records results.
  No additional GPU time was used. The running campaign and budget ledger are
  maintained in research PR #690. Next discriminate API CPU blocking from GPU work.
