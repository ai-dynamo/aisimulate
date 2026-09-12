# GB200 TP4 ordinary-serving verification

The frozen FPM predicts all primary observations but has large errors. Read the [complete report](report/README.md) for exact supported/observed pairs, 30-trial confidence intervals, the two precision misses and all cache disagreements.

Results are conditional on native prefix retention128 and gate512. Throughput is measured for finite cohorts; it is not saturation throughput or serving capacity.

![Paired MAPE and WAPE](report/comparison.png)

[Current replay semantics](CURRENT_REPLAY_SEMANTICS.md) distinguish the vLLM Aggregated first-output charge from current SGLang behavior. [Provenance](provenance.json) binds original private reports and the exact prediction/runtime identities.
