# CPU (48 cores) versus GPU (one H100) across datasets

Measured 2026-10-19. `build/efim -t 48` on two EPYC 9224 sockets against `build/G-HUIM -g 32G`
(BFS adaptive, defaults: local subtrees, sparse-database indexes, 16-bit keys) on one H100 PCIe with
`CUDA_VISIBLE_DEVICES=0`. Both count only. Five thresholds per dataset (x8 to x0.7 of the plotted
minutil, or 1% to 0.05% of the database utility); best of two runs; 180 s timeout. Every point
both programs finished returned the same itemset count.

- **Reported** is each program's own total (efim `Total ExecutionTime`, G-HUIM `Total time`, which
  leaves out CUDA context creation). **Wall** is the whole process.
- Bold: the GPU is faster. Speedup = CPU time / GPU time.
- The GPU wall time has a floor of about 0.3-0.5 s (context creation, which loading overlaps) and
  varied between runs (fruithut and the re-run ecommerce rows saw about 0.5 s).
- fruithut ran with its `@` header lines stripped for efim, which rejects them; ecommerce and fruithut
  were re-run after the sparse item id fix (ecommerce took 13.5 s on the GPU before it).

| Dataset | minutil | Itemsets | CPU 48c reported / wall (s) | GPU reported / wall (s) | Speedup reported | Speedup wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BMS | 15,992,000 | 0 | 0.013 / 0.016 | 0.013 / 0.414 | **1.01×** | 0.04× |
| BMS | 7,996,000 | 1 | 0.029 / 0.032 | 0.037 / 0.489 | 0.77× | 0.07× |
| BMS | 3,998,000 | 2 | 0.037 / 0.040 | 0.039 / 0.508 | 0.95× | 0.08× |
| BMS | 1,999,000 | 27,673 | 0.885 / 0.898 | 0.230 / 0.677 | **3.85×** | **1.33×** |
| BMS | 1,399,300 |  | CPU >180 s | GPU out of memory/capacity | – | – |
| accidents | 31,200,000 | 0 | 0.190 / 0.237 | 0.210 / 0.382 | 0.90× | 0.62× |
| accidents | 15,600,000 | 64,256 | 0.246 / 0.337 | 0.225 / 0.402 | **1.09×** | 0.84× |
| accidents | 7,800,000 | 3,393,461 | 0.366 / 0.548 | 0.287 / 0.446 | **1.27×** | **1.23×** |
| accidents | 3,900,000 | 46,139,005 | 0.839 / 1.117 | 0.495 / 0.653 | **1.70×** | **1.71×** |
| accidents | 2,730,000 | 138,136,213 | 1.391 / 1.694 | 0.749 / 0.923 | **1.86×** | **1.84×** |
| chess | 1,360,000 | 0 | 0.013 / 0.016 | 0.008 / 0.290 | **1.61×** | 0.06× |
| chess | 680,000 | 3 | 0.016 / 0.020 | 0.034 / 0.322 | 0.46× | 0.06× |
| chess | 340,000 | 1,961,424 | 0.061 / 0.072 | 0.046 / 0.360 | **1.33×** | 0.20× |
| chess | 170,000 | 226,238,941 | 0.956 / 0.972 | 0.250 / 0.541 | **3.83×** | **1.80×** |
| chess | 118,999 | 1,276,697,444 | 3.154 / 3.172 | 0.870 / 1.194 | **3.62×** | **2.66×** |
| connect | 59,200,000 | 0 | 0.033 / 0.038 | 0.037 / 0.298 | 0.89× | 0.13× |
| connect | 29,600,000 | 0 | 0.056 / 0.060 | 0.050 / 0.295 | **1.11×** | 0.20× |
| connect | 14,800,000 | 17,869 | 0.066 / 0.075 | 0.086 / 0.338 | 0.77× | 0.22× |
| connect | 7,400,000 | 221,648,994 | 1.018 / 1.040 | 0.192 / 0.451 | **5.30×** | **2.31×** |
| connect | 5,180,000 | 1,537,428,775 | 3.718 / 3.747 | 0.579 / 0.874 | **6.42×** | **4.29×** |
| kosarak | 5,760,000 | 6 | 0.175 / 0.193 | 0.196 / 0.372 | 0.89× | 0.52× |
| kosarak | 2,880,000 | 21 | 0.183 / 0.208 | 0.194 / 0.366 | 0.94× | 0.57× |
| kosarak | 1,440,000 | 64 | 0.189 / 0.220 | 0.206 / 0.377 | 0.92× | 0.58× |
| kosarak | 720,000 | 255 | 0.217 / 0.250 | 0.216 / 0.389 | **1.00×** | 0.64× |
| kosarak | 503,999 |  | CPU >180 s | GPU out of memory/capacity | – | – |
| mushroom | 44,000 | 10,423,524 | 0.117 / 0.127 | 0.047 / 0.519 | **2.50×** | 0.24× |
| mushroom | 22,000 | 52,123,537 | 0.206 / 0.216 | 0.049 / 0.516 | **4.21×** | 0.42× |
| mushroom | 11,000 | 195,408,180 | 0.606 / 0.617 | 0.070 / 0.488 | **8.65×** | **1.26×** |
| mushroom | 5,500 | 577,220,590 | 1.498 / 1.510 | 0.116 / 0.592 | **12.92×** | **2.55×** |
| mushroom | 3,849 | 910,649,101 | 2.189 / 2.201 | 0.148 / 0.714 | **14.79×** | **3.08×** |
| pumsb | 77,440,000 | 0 | 0.045 / 0.052 | 0.045 / 0.446 | 1.00× | 0.12× |
| pumsb | 38,720,000 | 0 | 0.074 / 0.081 | 0.056 / 0.415 | **1.32×** | 0.20× |
| pumsb | 19,360,000 | 0 | 0.080 / 0.092 | 0.089 / 0.522 | 0.89× | 0.18× |
| pumsb | 9,680,000 | 800,174 | 0.737 / 0.826 | 0.317 / 0.732 | **2.33×** | **1.13×** |
| pumsb | 6,776,000 |  | CPU >180 s | GPU out of memory/capacity | – | – |
| retail | 11,520 | 704 | 0.073 / 0.080 | 0.085 / 0.383 | 0.86× | 0.21× |
| retail | 5,760 | 2,403 | 0.077 / 0.084 | 0.093 / 0.393 | 0.83× | 0.21× |
| retail | 2,880 | 7,829 | 0.084 / 0.092 | 0.093 / 0.387 | 0.90× | 0.24× |
| retail | 1,440 | 5,779,547 | 0.463 / 0.477 | 0.385 / 0.655 | **1.20×** | 0.73× |
| retail | 1,007 |  | CPU >180 s | GPU out of memory/capacity | – | – |
| chainstore | 26,099,735 | 2 | 0.191 / 0.207 | 0.188 / 0.362 | **1.02×** | 0.57× |
| chainstore | 13,049,867 | 4 | 0.182 / 0.200 | 0.193 / 0.365 | 0.94× | 0.55× |
| chainstore | 5,219,947 | 25 | 0.217 / 0.241 | 0.196 / 0.368 | **1.11×** | 0.65× |
| chainstore | 2,609,973 | 81 | 0.210 / 0.245 | 0.214 / 0.385 | 0.98× | 0.64× |
| chainstore | 1,304,986 | 261 | 0.225 / 0.263 | 0.225 / 0.392 | **1.00×** | 0.67× |
| foodmart | 120,110 | 0 | 0.007 / 0.009 | 0.002 / 0.291 | **3.40×** | 0.03× |
| foodmart | 60,055 | 0 | 0.007 / 0.009 | 0.003 / 0.291 | **2.28×** | 0.03× |
| foodmart | 24,022 | 3 | 0.023 / 0.026 | 0.031 / 0.328 | 0.74× | 0.08× |
| foodmart | 12,011 | 258 | 0.022 / 0.025 | 0.031 / 0.366 | 0.72× | 0.07× |
| foodmart | 6,005 | 6,276 | 0.025 / 0.028 | 0.029 / 0.365 | 0.85× | 0.08× |
| ecommerce | 4,970,137 | 3 | 0.038 / 0.041 | 0.047 / 0.512 | 0.81× | 0.08× |
| ecommerce | 2,485,068 | 18 | 0.039 / 0.042 | 0.050 / 0.510 | 0.78× | 0.08× |
| ecommerce | 994,027 | 16,843 | 0.041 / 0.046 | 0.053 / 0.511 | 0.77× | 0.09× |
| ecommerce | 497,013 | 17,705,372 | 0.485 / 0.493 | 0.063 / 0.530 | **7.71×** | 0.93× |
| ecommerce | 248,506 | 934,815,155 | 2.696 / 2.703 | 0.186 / 0.659 | **14.50×** | **4.10×** |
| fruithut | 2,618,715 | 22 | 0.050 / 0.056 | 0.057 / 0.513 | 0.87× | 0.11× |
| fruithut | 1,309,357 | 86 | 0.051 / 0.058 | 0.062 / 0.507 | 0.82× | 0.11× |
| fruithut | 523,743 | 409 | 0.052 / 0.058 | 0.059 / 0.505 | 0.87× | 0.11× |
| fruithut | 261,871 | 1,397 | 0.053 / 0.061 | 0.063 / 0.505 | 0.85× | 0.12× |
| fruithut | 130,935 | 4,735 | 0.054 / 0.064 | 0.066 / 0.516 | 0.82× | 0.12× |

## Summary

- Heavy queries (seconds on the CPU) go to the GPU: wall 1.8-4.3x and reported 3.6-14.8x on chess,
  connect, mushroom and ecommerce at the lowest thresholds, 1.7-1.9x on accidents.
- Queries the CPU answers in under about 0.3 s are faster on the CPU in wall time, because of the GPU's
  fixed startup. Reported times there are within about 25% either way.
- Sparse data with few itemsets (kosarak, chainstore, retail, fruithut, foodmart) is a tie in reported
  time and a CPU win in wall time.
- At the lowest threshold of BMS, kosarak, pumsb and retail neither finishes: efim exceeds 180 s and
  the BFS miner runs out of level capacity or arena memory, since a level is materialized at once.
