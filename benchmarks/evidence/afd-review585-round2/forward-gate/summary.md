## Forward Prediction Performance (advisory)

**PASS** — 101 of 128 comparisons stable: 66 faster, 35 unchanged, 0 regressions; 27 noisy, 0 invalid.

Base `444c71d55a66b5bc50ea4ddab30e69a024c30525` vs head `d139f3646774dced5be67d3c20f3c51cde11ba72`.

<details>
<summary>⚠️ Noisy comparisons (27)</summary>

| case | cache | base | head | change | rounds |
|---|---|---:|---:|---:|---:|
| `qwen3-32b/silicon/context/bs1-isl8192` | cold | 18.67 µs | 19.08 µs | +2.2% (+0.42 µs) | 1/5 |
| `qwen3-32b/silicon/context/bs4-isl4096` | cold | 20.17 µs | 22.50 µs | +11.6% (+2.33 µs) | 1/5 |
| `qwen3-32b/silicon/generation/bs1-isl1024` | cold | 28.33 µs | 28.75 µs | +1.5% (+0.42 µs) | 1/5 |
| `qwen3-32b/silicon/generation/bs128-isl1024` | cold | 30.17 µs | 30.71 µs | +1.8% (+0.54 µs) | 1/5 |
| `qwen3-32b/empirical/generation/bs128-isl1024` | cold | 74.71 µs | 67.46 µs | -9.7% (-7.25 µs) | 1/5 |
| `qwen3-32b/empirical/generation/bs32-isl8192` | cold | 44.71 µs | 37.46 µs | -16.2% (-7.25 µs) | 1/5 |
| `qwen3-235b-a22b/silicon/context/bs1-isl1024` | cold | 24.46 µs | 25.83 µs | +5.6% (+1.38 µs) | 2/5 |
| `qwen3-235b-a22b/silicon/context/bs1-isl8192` | cold | 16.25 µs | 16.58 µs | +2.0% (+0.33 µs) | 1/5 |
| `qwen3-235b-a22b/silicon/context/bs1-isl32768` | cold | 193.12 µs | 194.08 µs | +0.5% (+0.96 µs) | 1/5 |
| `qwen3-235b-a22b/silicon/generation/bs1-isl1024` | cold | 29.17 µs | 29.00 µs | -0.6% (-0.17 µs) | 2/5 |
| `qwen3-235b-a22b/silicon/generation/bs32-isl8192` | cold | 19.38 µs | 19.04 µs | -1.7% (-0.33 µs) | 1/5 |
| `qwen3-235b-a22b/silicon/generation/bs8-isl32768` | cold | 26.08 µs | 28.62 µs | +9.7% (+2.54 µs) | 1/5 |
| `qwen3-235b-a22b/empirical/context/bs1-isl8192` | cold | 33.00 µs | 31.75 µs | -3.8% (-1.25 µs) | 1/5 |
| `qwen3-235b-a22b/empirical/context/bs4-isl4096` | cold | 21.33 µs | 22.38 µs | +4.9% (+1.04 µs) | 1/5 |
| `qwen3-235b-a22b/empirical/generation/bs1-isl1024` | cold | 52.79 µs | 51.04 µs | -3.3% (-1.75 µs) | 1/5 |
| `qwen3-235b-a22b/empirical/generation/bs32-isl1024` | cold | 65.96 µs | 67.62 µs | +2.5% (+1.67 µs) | 2/5 |
| `qwen3-235b-a22b/empirical/generation/bs128-isl1024` | cold | 68.50 µs | 74.38 µs | +8.6% (+5.88 µs) | 3/5 |
| `qwen3-235b-a22b/empirical/generation/bs32-isl8192` | cold | 40.92 µs | 42.08 µs | +2.9% (+1.17 µs) | 1/5 |
| `qwen3-235b-a22b/empirical/generation/bs8-isl32768` | cold | 63.25 µs | 72.67 µs | +14.9% (+9.42 µs) | 3/5 |
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs128-isl1024-prefix0` | cold | 23.62 µs | 27.92 µs | +18.2% (+4.29 µs) | 1/5 |
| `gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl1024-prefix0` | cold | 32.04 µs | 31.79 µs | -0.8% (-0.25 µs) | 1/5 |
| `gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs128-isl1024-prefix0` | cold | 38.25 µs | 40.83 µs | +6.8% (+2.58 µs) | 1/5 |
| `nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs128-isl1024-prefix0` | cold | 48.29 µs | 56.29 µs | +16.6% (+8.00 µs) | 2/5 |
| `qwen3-32b-prefix-b200_sxm-vllm-0.24.0-tp4-pp1-adp1-mtp1-ep1/silicon/context/bs1-isl8192-prefix4096` | cold | 34.25 µs | 33.67 µs | -1.7% (-0.58 µs) | 1/5 |
| `qwen3-235b-a22b-prefix-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl8192-prefix4096` | cold | 31.62 µs | 33.58 µs | +6.2% (+1.96 µs) | 3/5 |
| `qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl8192-prefix0` | cold | 45.08 µs | 45.58 µs | +1.1% (+0.50 µs) | 1/5 |
| `qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs32-isl1024-prefix0` | cold | 52.21 µs | 61.62 µs | +18.0% (+9.42 µs) | 1/5 |

</details>

<details>
<summary>Full matrix (46 cells)</summary>

| status | model | database | phase | cache | geometric mean | worst point | regressions | noisy | invalid |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| ⚠️ noisy | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | +1.0% | +18.2% | 0 | 1 | 0 |
| ⚠️ noisy | gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | +1.0% | +2.9% | 0 | 1 | 0 |
| ⚠️ noisy | gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | +5.7% | +6.8% | 0 | 1 | 0 |
| ⚠️ noisy | nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | +7.9% | +16.6% | 0 | 1 | 0 |
| ⚠️ noisy | qwen3-235b-a22b | EMPIRICAL | context | cold | -8.1% | +4.9% | 0 | 2 | 0 |
| ⚠️ noisy | qwen3-235b-a22b | EMPIRICAL | generation | cold | +4.9% | +14.9% | 0 | 5 | 0 |
| ⚠️ noisy | qwen3-235b-a22b | SILICON | context | cold | +1.9% | +5.6% | 0 | 3 | 0 |
| ⚠️ noisy | qwen3-235b-a22b | SILICON | generation | cold | +1.9% | +9.7% | 0 | 3 | 0 |
| ⚠️ noisy | qwen3-235b-a22b-prefix-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | -2.9% | +6.2% | 0 | 1 | 0 |
| ⚠️ noisy | qwen3-32b | EMPIRICAL | generation | cold | -5.7% | +2.3% | 0 | 2 | 0 |
| ⚠️ noisy | qwen3-32b | SILICON | context | cold | +4.1% | +11.6% | 0 | 2 | 0 |
| ⚠️ noisy | qwen3-32b | SILICON | generation | cold | -3.4% | +1.8% | 0 | 2 | 0 |
| ⚠️ noisy | qwen3-32b-prefix-b200_sxm-vllm-0.24.0-tp4-pp1-adp1-mtp1-ep1 | SILICON | context | cold | -5.4% | -1.7% | 0 | 1 | 0 |
| ⚠️ noisy | qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | +1.1% | +1.1% | 0 | 1 | 0 |
| ⚠️ noisy | qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | +18.0% | +18.0% | 0 | 1 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp1-pp1-adp8-mtp1-ep8 | SILICON | generation | cold | -6.3% | +0.2% | 0 | 0 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp1-pp1-adp8-mtp1-ep8 | SILICON | generation | warm | +0.3% | +0.3% | 0 | 0 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | -6.4% | -4.2% | 0 | 0 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | -0.7% | -0.4% | 0 | 0 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | +0.4% | +0.5% | 0 | 0 | 0 |
| ✅ stable | deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | +0.8% | +0.8% | 0 | 0 | 0 |
| ✅ stable | deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | +0.1% | +0.2% | 0 | 0 | 0 |
| ✅ stable | deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | -17.5% | -12.9% | 0 | 0 | 0 |
| ✅ stable | deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | +0.5% | +0.8% | 0 | 0 | 0 |
| ✅ stable | gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | -0.2% | +0.0% | 0 | 0 | 0 |
| ✅ stable | gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | -0.8% | -0.8% | 0 | 0 | 0 |
| ✅ stable | nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | -0.8% | +0.8% | 0 | 0 | 0 |
| ✅ stable | nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | -1.5% | -1.3% | 0 | 0 | 0 |
| ✅ stable | nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | -0.5% | -0.5% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b | EMPIRICAL | context | warm | -0.0% | +0.8% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b | EMPIRICAL | generation | warm | +0.2% | +1.5% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b | SILICON | context | warm | -1.0% | -0.8% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b | SILICON | generation | warm | -1.7% | -1.4% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b-prefix-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | -1.5% | -1.4% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | EMPIRICAL | context | cold | -15.7% | -7.0% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | EMPIRICAL | context | warm | -0.5% | -0.4% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | EMPIRICAL | generation | warm | +0.8% | +1.2% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | SILICON | context | warm | -0.3% | +0.0% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | SILICON | generation | warm | -0.8% | -0.5% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b-prefix-b200_sxm-vllm-0.24.0-tp4-pp1-adp1-mtp1-ep1 | SILICON | context | warm | -0.2% | -0.0% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | +1.3% | +1.3% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | +0.9% | +0.9% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | -5.2% | +0.9% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | +0.0% | +0.1% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | -5.5% | -2.4% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | -0.2% | -0.1% | 0 | 0 | 0 |

</details>

This check is advisory and is not required by branch protection.
