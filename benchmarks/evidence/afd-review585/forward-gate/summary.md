## Forward Prediction Performance (advisory)

**PASS** — 78 of 128 comparisons stable: 31 faster, 47 unchanged, 0 regressions; 50 noisy, 0 invalid.

Base `6b57265e03a3844503b6a9f77f29f1f2dd8eea02` vs head `4849d6676280aafe963d795a757d087a24d8f8a7`.

<details>
<summary>⚠️ Noisy comparisons (50)</summary>

| case | cache | base | head | change | rounds |
|---|---|---:|---:|---:|---:|
| `qwen3-32b/silicon/context/bs1-isl8192` | cold | 18.54 µs | 19.17 µs | +3.4% (+0.62 µs) | 1/5 |
| `qwen3-32b/silicon/context/bs1-isl32768` | cold | 153.96 µs | 193.25 µs | +25.5% (+39.29 µs) | 2/5 |
| `qwen3-32b/silicon/context/bs4-isl4096` | cold | 25.12 µs | 23.88 µs | -5.0% (-1.25 µs) | 1/5 |
| `qwen3-32b/silicon/generation/bs32-isl1024` | cold | 31.88 µs | 32.17 µs | +0.9% (+0.29 µs) | 2/5 |
| `qwen3-32b/silicon/generation/bs32-isl8192` | cold | 26.71 µs | 25.67 µs | -3.9% (-1.04 µs) | 1/5 |
| `qwen3-32b/empirical/context/bs1-isl1024` | cold | 61.88 µs | 54.67 µs | -11.6% (-7.21 µs) | 3/5 |
| `qwen3-32b/empirical/context/bs1-isl8192` | cold | 30.38 µs | 32.58 µs | +7.3% (+2.21 µs) | 1/5 |
| `qwen3-32b/empirical/context/bs1-isl32768` | cold | 20.75 µs | 20.12 µs | -3.0% (-0.62 µs) | 1/5 |
| `qwen3-32b/empirical/context/bs4-isl4096` | cold | 28.00 µs | 28.62 µs | +2.2% (+0.62 µs) | 1/5 |
| `qwen3-32b/empirical/generation/bs1-isl1024` | cold | 51.25 µs | 57.42 µs | +12.0% (+6.17 µs) | 2/5 |
| `qwen3-32b/empirical/generation/bs32-isl1024` | cold | 71.29 µs | 72.33 µs | +1.5% (+1.04 µs) | 1/5 |
| `qwen3-32b/empirical/generation/bs128-isl1024` | cold | 63.46 µs | 69.75 µs | +9.9% (+6.29 µs) | 2/5 |
| `qwen3-32b/empirical/generation/bs32-isl8192` | cold | 57.29 µs | 39.88 µs | -30.4% (-17.42 µs) | 1/5 |
| `qwen3-32b/empirical/generation/bs8-isl32768` | cold | 69.96 µs | 72.83 µs | +4.1% (+2.88 µs) | 2/5 |
| `qwen3-235b-a22b/silicon/context/bs1-isl1024` | cold | 27.71 µs | 24.50 µs | -11.6% (-3.21 µs) | 1/5 |
| `qwen3-235b-a22b/silicon/context/bs1-isl32768` | cold | 188.42 µs | 191.46 µs | +1.6% (+3.04 µs) | 1/5 |
| `qwen3-235b-a22b/silicon/context/bs4-isl4096` | cold | 15.29 µs | 18.58 µs | +21.5% (+3.29 µs) | 1/5 |
| `qwen3-235b-a22b/silicon/generation/bs1-isl1024` | cold | 32.58 µs | 34.08 µs | +4.6% (+1.50 µs) | 1/5 |
| `qwen3-235b-a22b/silicon/generation/bs32-isl1024` | cold | 29.75 µs | 29.83 µs | +0.3% (+0.08 µs) | 1/5 |
| `qwen3-235b-a22b/silicon/generation/bs128-isl1024` | cold | 31.54 µs | 32.04 µs | +1.6% (+0.50 µs) | 2/5 |
| `qwen3-235b-a22b/empirical/context/bs1-isl1024` | cold | 56.96 µs | 60.12 µs | +5.6% (+3.17 µs) | 1/5 |
| `qwen3-235b-a22b/empirical/context/bs1-isl8192` | cold | 33.12 µs | 30.92 µs | -6.7% (-2.21 µs) | 1/5 |
| `qwen3-235b-a22b/empirical/context/bs1-isl32768` | cold | 17.46 µs | 21.54 µs | +23.4% (+4.08 µs) | 3/5 |
| `qwen3-235b-a22b/empirical/context/bs4-isl4096` | cold | 25.88 µs | 20.75 µs | -19.8% (-5.12 µs) | 1/5 |
| `qwen3-235b-a22b/empirical/generation/bs1-isl1024` | cold | 55.12 µs | 56.96 µs | +3.3% (+1.83 µs) | 3/5 |
| `qwen3-235b-a22b/empirical/generation/bs32-isl1024` | cold | 70.67 µs | 78.75 µs | +11.4% (+8.08 µs) | 2/5 |
| `qwen3-235b-a22b/empirical/generation/bs128-isl1024` | cold | 75.29 µs | 81.00 µs | +7.6% (+5.71 µs) | 2/5 |
| `qwen3-235b-a22b/empirical/generation/bs32-isl8192` | cold | 40.50 µs | 48.62 µs | +20.1% (+8.12 µs) | 1/5 |
| `qwen3-235b-a22b/empirical/generation/bs8-isl32768` | cold | 59.62 µs | 72.50 µs | +21.6% (+12.88 µs) | 2/5 |
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl1024-prefix0` | cold | 37.29 µs | 38.92 µs | +4.4% (+1.63 µs) | 1/5 |
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl32768-prefix0` | cold | 14.21 µs | 13.92 µs | -2.0% (-0.29 µs) | 1/5 |
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs1-isl1024-prefix0` | cold | 25.00 µs | 25.25 µs | +1.0% (+0.25 µs) | 2/5 |
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs128-isl1024-prefix0` | cold | 24.96 µs | 33.79 µs | +35.4% (+8.83 µs) | 1/5 |
| `deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl1024-prefix0` | cold | 29.54 µs | 34.83 µs | +17.9% (+5.29 µs) | 1/5 |
| `deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl32768-prefix0` | cold | 15.08 µs | 16.79 µs | +11.3% (+1.71 µs) | 2/5 |
| `deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs1-isl1024-prefix0` | cold | 31.17 µs | 27.58 µs | -11.5% (-3.58 µs) | 2/5 |
| `deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs128-isl1024-prefix0` | cold | 40.12 µs | 28.42 µs | -29.2% (-11.71 µs) | 1/5 |
| `qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl1024-prefix0` | cold | 46.71 µs | 45.08 µs | -3.5% (-1.62 µs) | 1/5 |
| `qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl32768-prefix0` | cold | 109.46 µs | 109.21 µs | -0.2% (-0.25 µs) | 2/5 |
| `gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl1024-prefix0` | cold | 33.29 µs | 38.67 µs | +16.1% (+5.38 µs) | 3/5 |
| `gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl32768-prefix0` | cold | 276.21 µs | 288.29 µs | +4.4% (+12.08 µs) | 1/5 |
| `gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs1-isl1024-prefix0` | cold | 48.96 µs | 45.92 µs | -6.2% (-3.04 µs) | 1/5 |
| `nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl1024-prefix0` | cold | 44.00 µs | 44.83 µs | +1.9% (+0.83 µs) | 2/5 |
| `nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs1-isl1024-prefix0` | cold | 82.79 µs | 92.33 µs | +11.5% (+9.54 µs) | 3/5 |
| `nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs128-isl1024-prefix0` | cold | 89.46 µs | 94.83 µs | +6.0% (+5.37 µs) | 1/5 |
| `qwen3-32b-prefix-b200_sxm-vllm-0.24.0-tp4-pp1-adp1-mtp1-ep1/silicon/context/bs1-isl8192-prefix4096` | cold | 35.50 µs | 34.04 µs | -4.1% (-1.46 µs) | 1/5 |
| `qwen3-32b-prefix-b200_sxm-vllm-0.24.0-tp4-pp1-adp1-mtp1-ep1/silicon/context/bs1-isl8192-prefix7168` | cold | 27.12 µs | 25.46 µs | -6.1% (-1.67 µs) | 2/5 |
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp1-pp1-adp8-mtp1-ep8/silicon/generation/bs32-isl1024-prefix0` | cold | 49.92 µs | 50.29 µs | +0.8% (+0.38 µs) | 2/5 |
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp1-pp1-adp8-mtp1-ep8/silicon/generation/bs8-isl32768-prefix0` | cold | 29.96 µs | 34.58 µs | +15.4% (+4.63 µs) | 1/5 |
| `qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl8192-prefix0` | cold | 47.75 µs | 47.25 µs | -1.0% (-0.50 µs) | 1/5 |

</details>

<details>
<summary>Full matrix (46 cells)</summary>

| status | model | database | phase | cache | geometric mean | worst point | regressions | noisy | invalid |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| ⚠️ noisy | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp1-pp1-adp8-mtp1-ep8 | SILICON | generation | cold | +7.8% | +15.4% | 0 | 2 | 0 |
| ⚠️ noisy | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | +1.1% | +4.4% | 0 | 2 | 0 |
| ⚠️ noisy | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | +16.9% | +35.4% | 0 | 2 | 0 |
| ⚠️ noisy | deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | +14.6% | +17.9% | 0 | 2 | 0 |
| ⚠️ noisy | deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | -20.8% | -11.5% | 0 | 2 | 0 |
| ⚠️ noisy | gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | +10.1% | +16.1% | 0 | 2 | 0 |
| ⚠️ noisy | gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | -5.2% | -4.1% | 0 | 1 | 0 |
| ⚠️ noisy | nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | +1.3% | +1.9% | 0 | 1 | 0 |
| ⚠️ noisy | nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | +8.7% | +11.5% | 0 | 2 | 0 |
| ⚠️ noisy | qwen3-235b-a22b | EMPIRICAL | context | cold | -0.6% | +23.4% | 0 | 4 | 0 |
| ⚠️ noisy | qwen3-235b-a22b | EMPIRICAL | generation | cold | +12.6% | +21.6% | 0 | 5 | 0 |
| ⚠️ noisy | qwen3-235b-a22b | SILICON | context | cold | -2.2% | +21.5% | 0 | 3 | 0 |
| ⚠️ noisy | qwen3-235b-a22b | SILICON | generation | cold | -1.2% | +4.6% | 0 | 3 | 0 |
| ⚠️ noisy | qwen3-32b | EMPIRICAL | context | cold | -1.5% | +7.3% | 0 | 4 | 0 |
| ⚠️ noisy | qwen3-32b | EMPIRICAL | generation | cold | -2.0% | +12.0% | 0 | 5 | 0 |
| ⚠️ noisy | qwen3-32b | SILICON | context | cold | +5.1% | +25.5% | 0 | 3 | 0 |
| ⚠️ noisy | qwen3-32b | SILICON | generation | cold | -4.6% | +0.9% | 0 | 2 | 0 |
| ⚠️ noisy | qwen3-32b-prefix-b200_sxm-vllm-0.24.0-tp4-pp1-adp1-mtp1-ep1 | SILICON | context | cold | -5.1% | -4.1% | 0 | 2 | 0 |
| ⚠️ noisy | qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | -1.0% | -1.0% | 0 | 1 | 0 |
| ⚠️ noisy | qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | -1.9% | -0.2% | 0 | 2 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp1-pp1-adp8-mtp1-ep8 | SILICON | generation | warm | +1.6% | +1.8% | 0 | 0 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | +0.5% | +0.5% | 0 | 0 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | +0.3% | +1.0% | 0 | 0 | 0 |
| ✅ stable | deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | +1.5% | +1.7% | 0 | 0 | 0 |
| ✅ stable | deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | -0.7% | -0.7% | 0 | 0 | 0 |
| ✅ stable | gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | +1.8% | +1.9% | 0 | 0 | 0 |
| ✅ stable | gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | -1.1% | -0.3% | 0 | 0 | 0 |
| ✅ stable | nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | +0.4% | +0.4% | 0 | 0 | 0 |
| ✅ stable | nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | +2.5% | +2.8% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b | EMPIRICAL | context | warm | +0.8% | +1.2% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b | EMPIRICAL | generation | warm | +2.1% | +2.3% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b | SILICON | context | warm | -0.1% | +0.5% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b | SILICON | generation | warm | -0.2% | +1.0% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b-prefix-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | +2.1% | +6.9% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b-prefix-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | +1.7% | +1.9% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | EMPIRICAL | context | warm | +0.5% | +0.7% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | EMPIRICAL | generation | warm | +0.2% | +1.0% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | SILICON | context | warm | +1.1% | +1.3% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | SILICON | generation | warm | -0.9% | +0.6% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b-prefix-b200_sxm-vllm-0.24.0-tp4-pp1-adp1-mtp1-ep1 | SILICON | context | warm | +1.3% | +1.3% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | -0.5% | -0.5% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | -4.3% | -4.3% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | -0.2% | -0.2% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | -1.0% | -1.0% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | -3.2% | -1.3% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | +0.7% | +1.5% | 0 | 0 | 0 |

</details>

This check is advisory and is not required by branch protection.
