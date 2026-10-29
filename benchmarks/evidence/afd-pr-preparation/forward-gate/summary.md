## Forward Prediction Performance (advisory)

**PASS** — 70 of 128 comparisons stable: 29 faster, 41 unchanged, 0 regressions; 58 noisy, 0 invalid.

Base `f3a3e3156530574f5e7a10446f19e571b82db871` vs head `51e22a92af3cafd20c2b97a3b938c2d57185329b`.

<details>
<summary>⚠️ Noisy comparisons (58)</summary>

| case | cache | base | head | change | rounds |
|---|---|---:|---:|---:|---:|
| `qwen3-32b/silicon/context/bs1-isl1024` | cold | 32.58 µs | 32.00 µs | -1.8% (-0.58 µs) | 1/5 |
| `qwen3-32b/silicon/context/bs1-isl32768` | cold | 171.00 µs | 152.54 µs | -10.8% (-18.46 µs) | 1/5 |
| `qwen3-32b/silicon/context/bs4-isl4096` | cold | 23.88 µs | 21.25 µs | -11.0% (-2.62 µs) | 1/5 |
| `qwen3-32b/silicon/generation/bs1-isl1024` | cold | 38.83 µs | 32.67 µs | -15.9% (-6.17 µs) | 1/5 |
| `qwen3-32b/silicon/generation/bs32-isl1024` | cold | 32.71 µs | 32.50 µs | -0.6% (-0.21 µs) | 1/5 |
| `qwen3-32b/silicon/generation/bs128-isl1024` | cold | 32.96 µs | 34.58 µs | +4.9% (+1.63 µs) | 2/5 |
| `qwen3-32b/silicon/generation/bs32-isl8192` | cold | 24.67 µs | 23.17 µs | -6.1% (-1.50 µs) | 1/5 |
| `qwen3-32b/silicon/generation/bs8-isl32768` | cold | 39.58 µs | 38.46 µs | -2.8% (-1.12 µs) | 1/5 |
| `qwen3-32b/empirical/context/bs1-isl1024` | cold | 61.12 µs | 59.92 µs | -2.0% (-1.21 µs) | 2/5 |
| `qwen3-32b/empirical/context/bs1-isl8192` | cold | 31.83 µs | 32.62 µs | +2.5% (+0.79 µs) | 3/5 |
| `qwen3-32b/empirical/context/bs1-isl32768` | cold | 21.62 µs | 21.00 µs | -2.9% (-0.62 µs) | 1/5 |
| `qwen3-32b/empirical/context/bs4-isl4096` | cold | 27.50 µs | 30.00 µs | +9.1% (+2.50 µs) | 1/5 |
| `qwen3-32b/empirical/generation/bs1-isl1024` | cold | 55.83 µs | 58.04 µs | +4.0% (+2.21 µs) | 2/5 |
| `qwen3-32b/empirical/generation/bs32-isl1024` | cold | 68.58 µs | 62.67 µs | -8.6% (-5.92 µs) | 1/5 |
| `qwen3-32b/empirical/generation/bs128-isl1024` | cold | 65.67 µs | 69.79 µs | +6.3% (+4.12 µs) | 2/5 |
| `qwen3-32b/empirical/generation/bs32-isl8192` | cold | 48.92 µs | 41.88 µs | -14.4% (-7.04 µs) | 1/5 |
| `qwen3-32b/empirical/generation/bs8-isl32768` | cold | 59.92 µs | 84.00 µs | +40.2% (+24.08 µs) | 1/5 |
| `qwen3-235b-a22b/silicon/context/bs1-isl1024` | cold | 23.46 µs | 24.46 µs | +4.3% (+1.00 µs) | 1/5 |
| `qwen3-235b-a22b/silicon/context/bs1-isl8192` | cold | 16.42 µs | 16.83 µs | +2.5% (+0.42 µs) | 3/5 |
| `qwen3-235b-a22b/silicon/context/bs1-isl32768` | cold | 192.71 µs | 206.00 µs | +6.9% (+13.29 µs) | 2/5 |
| `qwen3-235b-a22b/silicon/context/bs4-isl4096` | cold | 17.38 µs | 16.75 µs | -3.6% (-0.62 µs) | 2/5 |
| `qwen3-235b-a22b/silicon/generation/bs1-isl1024` | cold | 33.54 µs | 33.54 µs | -0.0% (-0.00 µs) | 2/5 |
| `qwen3-235b-a22b/silicon/generation/bs32-isl1024` | cold | 30.75 µs | 32.04 µs | +4.2% (+1.29 µs) | 1/5 |
| `qwen3-235b-a22b/silicon/generation/bs128-isl1024` | cold | 32.08 µs | 32.04 µs | -0.1% (-0.04 µs) | 2/5 |
| `qwen3-235b-a22b/silicon/generation/bs8-isl32768` | cold | 38.21 µs | 39.17 µs | +2.5% (+0.96 µs) | 2/5 |
| `qwen3-235b-a22b/empirical/context/bs1-isl1024` | cold | 68.00 µs | 59.33 µs | -12.7% (-8.67 µs) | 1/5 |
| `qwen3-235b-a22b/empirical/context/bs1-isl8192` | cold | 37.50 µs | 33.33 µs | -11.1% (-4.17 µs) | 1/5 |
| `qwen3-235b-a22b/empirical/context/bs1-isl32768` | cold | 20.29 µs | 19.79 µs | -2.5% (-0.50 µs) | 2/5 |
| `qwen3-235b-a22b/empirical/context/bs4-isl4096` | cold | 19.50 µs | 33.04 µs | +69.4% (+13.54 µs) | 2/5 |
| `qwen3-235b-a22b/empirical/generation/bs1-isl1024` | cold | 60.00 µs | 52.75 µs | -12.1% (-7.25 µs) | 1/5 |
| `qwen3-235b-a22b/empirical/generation/bs128-isl1024` | cold | 75.54 µs | 75.17 µs | -0.5% (-0.38 µs) | 1/5 |
| `qwen3-235b-a22b/empirical/generation/bs32-isl8192` | cold | 54.33 µs | 48.62 µs | -10.5% (-5.71 µs) | 1/5 |
| `qwen3-235b-a22b/empirical/generation/bs8-isl32768` | cold | 78.92 µs | 88.42 µs | +12.0% (+9.50 µs) | 2/5 |
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl1024-prefix0` | cold | 39.71 µs | 41.79 µs | +5.2% (+2.08 µs) | 1/5 |
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl32768-prefix0` | cold | 14.33 µs | 18.12 µs | +26.5% (+3.79 µs) | 2/5 |
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs1-isl1024-prefix0` | cold | 22.75 µs | 32.00 µs | +40.7% (+9.25 µs) | 3/5 |
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs128-isl1024-prefix0` | cold | 32.12 µs | 39.38 µs | +22.6% (+7.25 µs) | 2/5 |
| `deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl1024-prefix0` | cold | 31.79 µs | 34.12 µs | +7.3% (+2.33 µs) | 2/5 |
| `deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl32768-prefix0` | cold | 16.62 µs | 15.29 µs | -8.0% (-1.33 µs) | 2/5 |
| `deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs1-isl1024-prefix0` | cold | 27.92 µs | 27.29 µs | -2.2% (-0.62 µs) | 2/5 |
| `deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs128-isl1024-prefix0` | cold | 32.25 µs | 28.88 µs | -10.5% (-3.38 µs) | 2/5 |
| `qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl1024-prefix0` | cold | 46.75 µs | 46.42 µs | -0.7% (-0.33 µs) | 1/5 |
| `qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs1-isl1024-prefix0` | cold | 50.54 µs | 50.88 µs | +0.7% (+0.33 µs) | 2/5 |
| `gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl1024-prefix0` | cold | 31.54 µs | 34.17 µs | +8.3% (+2.63 µs) | 2/5 |
| `gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl32768-prefix0` | cold | 284.08 µs | 291.00 µs | +2.4% (+6.92 µs) | 1/5 |
| `gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs1-isl1024-prefix0` | cold | 46.21 µs | 54.25 µs | +17.4% (+8.04 µs) | 3/5 |
| `gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs128-isl1024-prefix0` | cold | 46.38 µs | 52.71 µs | +13.7% (+6.33 µs) | 1/5 |
| `nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl1024-prefix0` | cold | 47.00 µs | 46.00 µs | -2.1% (-1.00 µs) | 2/5 |
| `nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl32768-prefix0` | cold | 226.46 µs | 224.54 µs | -0.8% (-1.92 µs) | 1/5 |
| `nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs1-isl1024-prefix0` | cold | 87.04 µs | 95.96 µs | +10.2% (+8.92 µs) | 2/5 |
| `nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs128-isl1024-prefix0` | cold | 96.38 µs | 92.38 µs | -4.2% (-4.00 µs) | 2/5 |
| `qwen3-32b-prefix-b200_sxm-vllm-0.24.0-tp4-pp1-adp1-mtp1-ep1/silicon/context/bs1-isl8192-prefix4096` | cold | 35.96 µs | 37.17 µs | +3.4% (+1.21 µs) | 1/5 |
| `qwen3-32b-prefix-b200_sxm-vllm-0.24.0-tp4-pp1-adp1-mtp1-ep1/silicon/context/bs1-isl8192-prefix7168` | cold | 25.25 µs | 23.33 µs | -7.6% (-1.92 µs) | 1/5 |
| `qwen3-235b-a22b-prefix-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl8192-prefix4096` | cold | 37.12 µs | 35.33 µs | -4.8% (-1.79 µs) | 2/5 |
| `qwen3-235b-a22b-prefix-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl8192-prefix7168` | cold | 30.50 µs | 28.67 µs | -6.0% (-1.83 µs) | 1/5 |
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp1-pp1-adp8-mtp1-ep8/silicon/generation/bs8-isl32768-prefix0` | cold | 37.83 µs | 37.12 µs | -1.9% (-0.71 µs) | 1/5 |
| `qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl8192-prefix0` | cold | 43.88 µs | 45.50 µs | +3.7% (+1.62 µs) | 2/5 |
| `qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs32-isl1024-prefix0` | cold | 62.04 µs | 61.62 µs | -0.7% (-0.42 µs) | 3/5 |

</details>

<details>
<summary>Full matrix (46 cells)</summary>

| status | model | database | phase | cache | geometric mean | worst point | regressions | noisy | invalid |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| ⚠️ noisy | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp1-pp1-adp8-mtp1-ep8 | SILICON | generation | cold | -7.7% | -1.9% | 0 | 1 | 0 |
| ⚠️ noisy | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | +15.4% | +26.5% | 0 | 2 | 0 |
| ⚠️ noisy | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | +31.3% | +40.7% | 0 | 2 | 0 |
| ⚠️ noisy | deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | -0.6% | +7.3% | 0 | 2 | 0 |
| ⚠️ noisy | deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | -6.4% | -2.2% | 0 | 2 | 0 |
| ⚠️ noisy | gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | +5.3% | +8.3% | 0 | 2 | 0 |
| ⚠️ noisy | gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | +15.5% | +17.4% | 0 | 2 | 0 |
| ⚠️ noisy | nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | -1.5% | -0.8% | 0 | 2 | 0 |
| ⚠️ noisy | nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | +2.8% | +10.2% | 0 | 2 | 0 |
| ⚠️ noisy | qwen3-235b-a22b | EMPIRICAL | context | cold | +6.4% | +69.4% | 0 | 4 | 0 |
| ⚠️ noisy | qwen3-235b-a22b | EMPIRICAL | generation | cold | -3.8% | +12.0% | 0 | 4 | 0 |
| ⚠️ noisy | qwen3-235b-a22b | SILICON | context | cold | +2.5% | +6.9% | 0 | 4 | 0 |
| ⚠️ noisy | qwen3-235b-a22b | SILICON | generation | cold | +1.2% | +4.2% | 0 | 4 | 0 |
| ⚠️ noisy | qwen3-235b-a22b-prefix-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | -5.4% | -4.8% | 0 | 2 | 0 |
| ⚠️ noisy | qwen3-32b | EMPIRICAL | context | cold | +1.6% | +9.1% | 0 | 4 | 0 |
| ⚠️ noisy | qwen3-32b | EMPIRICAL | generation | cold | +3.9% | +40.2% | 0 | 5 | 0 |
| ⚠️ noisy | qwen3-32b | SILICON | context | cold | -5.9% | +0.4% | 0 | 3 | 0 |
| ⚠️ noisy | qwen3-32b | SILICON | generation | cold | -4.4% | +4.9% | 0 | 5 | 0 |
| ⚠️ noisy | qwen3-32b-prefix-b200_sxm-vllm-0.24.0-tp4-pp1-adp1-mtp1-ep1 | SILICON | context | cold | -2.3% | +3.4% | 0 | 2 | 0 |
| ⚠️ noisy | qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | +3.7% | +3.7% | 0 | 1 | 0 |
| ⚠️ noisy | qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | -0.7% | -0.7% | 0 | 1 | 0 |
| ⚠️ noisy | qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | +0.0% | +0.8% | 0 | 1 | 0 |
| ⚠️ noisy | qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | -2.2% | +0.7% | 0 | 1 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp1-pp1-adp8-mtp1-ep8 | SILICON | generation | warm | +1.0% | +1.1% | 0 | 0 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | +0.5% | +0.5% | 0 | 0 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | +1.5% | +2.7% | 0 | 0 | 0 |
| ✅ stable | deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | +0.2% | +0.4% | 0 | 0 | 0 |
| ✅ stable | deepseek-v4-flash-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | +0.2% | +0.2% | 0 | 0 | 0 |
| ✅ stable | gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | +0.8% | +0.9% | 0 | 0 | 0 |
| ✅ stable | gpt-oss-120b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | -0.3% | -0.3% | 0 | 0 | 0 |
| ✅ stable | nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | -0.4% | -0.2% | 0 | 0 | 0 |
| ✅ stable | nemotron-3-super-120b-fp8-h100_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | -1.1% | -0.9% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b | EMPIRICAL | context | warm | +0.2% | +0.8% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b | EMPIRICAL | generation | warm | -0.3% | +0.4% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b | SILICON | context | warm | +0.1% | +0.9% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b | SILICON | generation | warm | -0.7% | +0.0% | 0 | 0 | 0 |
| ✅ stable | qwen3-235b-a22b-prefix-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | -0.5% | -0.0% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | EMPIRICAL | context | warm | +1.1% | +1.5% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | EMPIRICAL | generation | warm | +0.2% | +1.0% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | SILICON | context | warm | -0.2% | +1.3% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | SILICON | generation | warm | +0.8% | +1.6% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b-prefix-b200_sxm-vllm-0.24.0-tp4-pp1-adp1-mtp1-ep1 | SILICON | context | warm | +0.7% | +0.9% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | -0.4% | -0.4% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-sglang-0.5.14-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | +0.3% | +0.3% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | +1.3% | +1.6% | 0 | 0 | 0 |
| ✅ stable | qwen3.5-397b-a17b-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | -0.5% | -0.1% | 0 | 0 | 0 |

</details>

This check is advisory and is not required by branch protection.
