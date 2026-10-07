# Superseded diagnostic cohort

These files preserve the earlier node-235 measurements and SDK receipts from
job 4764263. They are historical diagnostics, not the final qualified data.
The 14-cell FP8 cohort includes TC-covered prefill whose standalone timing
boundary was not qualified. The final cohort excludes those TC points and
uses new node-079 measurements for native eager prefill and decode.

The earlier six priority-decode samples preceded node 235's later DRAIN flag
at 00:30:26 UTC. That later flag alone does not establish that the earlier
samples were invalid. New measurements supersede them without attributing
between-node timing differences to that flag. Original source identities,
row values and numerical receipts are preserved here; final results are in
the parent directory.
