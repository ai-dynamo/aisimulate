# vLLM Prefill Scheduling Cadence

vLLM can throttle prefill scheduling in data-parallel deployments so decode
steps remain balanced across ranks. Set the interval in an `aisimulate predict`
configuration under the scheduler for each affected worker role:

```yaml
engine:
  mode: aggregated
  model: nvidia/Kimi-K2.5-NVFP4
  hardware: b200_sxm
  backend: vllm
  workers:
    aggregated:
      parallelism:
        attention_data: 8
      scheduler:
        prefill_schedule_interval: 4
```

```bash
aisimulate predict --config prediction.yaml
```

The Python compiler lowers this field to the rank-level engine setting used by
the replay runtime. Lower-level Runner callers can set the same field directly:

```json
{
  "engine": {
    "dp_size": 4,
    "rank": {
      "backend": "vllm",
      "prefill_schedule_interval": 4
    }
  }
}
```

The default is `1`, which preserves the previous scheduling behavior. Values
above one take effect only for vLLM attention-DP groups. On a non-aligned group
step, local prefill work with more than one token remaining waits while decodes
continue. Connector loads, materialized requests, and requests with at most one
prefill token remaining can still advance. Throttling is temporarily released
when a non-preempting aligned step left queued requests due to scheduler
capacity.

SGLang has a separate
[`prefill_decode_interval`](sglang-prefill-decode-interval.md), defaulting to zero.
Nondefault values of either field on the wrong backend fail validation.

The shared counter resets as soon as AISimulate observes that the full DP group
has drained, including after cancellation and internal-work transitions. vLLM
checks global unfinished state every 32 steps and may run a dummy tail before
resetting. AISimulate does not model that collective tail, so a request arriving
during the upstream tail can observe a different cadence phase.

This follows vLLM's `prefill_schedule_interval` scheduler behavior at commit
`e2fa28594f7baad142a426b0b6a2cfe2c79201c7`.

## Validation

Kimi-K2.5 NVFP4, ISL 8192, and OSL 1024 replay measurements show that interval
4 materially closes the TPOT gap on both B200 and B300. Interval 1 reproduces
the previous AISimulate baseline exactly.

### B200, attention DP 8 / MoE EP 8

| Concurrency | Silicon TPOT | Interval 1 | Gap | Interval 4 | Gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 89.548 ms | 173.787 ms | +94.1% | 96.173 ms | +7.4% |
| 1024 | 143.230 ms | 248.305 ms | +73.4% | 147.367 ms | +2.9% |

Interval 4 also moves output throughput to within 6.0% of silicon at concurrency
512 and within 2.3% at concurrency 1024.

### B300, attention DP 4 / MoE EP 4

| Concurrency | Silicon TPOT | Interval 1 | Gap | Interval 4 | Gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 48.845 ms | 67.754 ms | +38.7% | 57.043 ms | +16.8% |
| 256 | 69.027 ms | 100.508 ms | +45.6% | 75.998 ms | +10.1% |
| 512 | 109.110 ms | 153.701 ms | +40.9% | 115.119 ms | +5.5% |

Interval 4 moves per-GPU output throughput gaps from -27.2%, -30.3%, and
-27.7% to -14.0%, -8.8%, and -4.8%, respectively. TTFT gaps improve from
-20.1%, -21.3%, and -23.4% to -15.0%, -13.3%, and -10.5%. The remaining gap
is largest at low concurrency, so cadence is the dominant effect but not the
only source of error for this case.
