### Stage A

| Rank | Config | Seeds | Catch rate mean ± sd [min–max] | Miss distance mm mean ± sd | Median miss mm | Train time s |
|---:|---|---:|---|---|---:|---:|
| 1 | 25 Hz | 3 | 0.035 ± 0.013 [0.025–0.050] | 95.3 ± 13.5 | 60.3 | 126 |
| 2 | 10 Hz (baseline) | 3 | 0.013 ± 0.003 [0.010–0.015] | 146.9 ± 11.1 | 110.2 | 128 |
| 3 | 5 Hz | 3 | 0.000 ± 0.000 [0.000–0.000] | 303.0 ± 10.3 | 250.8 | 133 |

### Stage B

| Rank | Config | Seeds | Catch rate mean ± sd [min–max] | Miss distance mm mean ± sd | Median miss mm | Train time s |
|---:|---|---:|---|---|---:|---:|
| 1 | predict 2 s (20 steps), execute 0.5 = 1 s (10 steps) (baseline) | 3 | 0.013 ± 0.003 [0.010–0.015] | 146.9 ± 11.1 | 110.2 | 128 |
| 2 | predict 5 s (50 steps), execute 0.25 = 1.3 s (13 steps) | 3 | 0.018 ± 0.008 [0.010–0.025] | 165.1 ± 4.5 | 101.1 | 134 |
| 3 | predict 5 s (50 steps), execute 0.5 = 2.5 s (25 steps) | 3 | 0.008 ± 0.008 [0.000–0.015] | 196.6 ± 3.8 | 100.4 | 136 |
| 4 | predict 2 s (20 steps), execute 0.25 = 0.5 s (5 steps) | 3 | 0.000 ± 0.000 [0.000–0.000] | 189.3 ± 6.5 | 164.5 | 129 |
| 5 | predict 0.5 s (5 steps), execute 0.5 = 0.3 s (3 steps) | 3 | 0.002 ± 0.003 [0.000–0.005] | 333.8 ± 11.8 | 313.2 | 127 |
| 6 | predict 0.5 s (5 steps), execute 0.25 = 0.1 s (1 steps) | 3 | 0.000 ± 0.000 [0.000–0.000] | 2162.6 ± 275.7 | 1994.1 | 129 |

### Stage C

| Rank | Config | Seeds | Catch rate mean ± sd [min–max] | Miss distance mm mean ± sd | Median miss mm | Train time s |
|---:|---|---:|---|---|---:|---:|
| 1 | stack 0.5 s (6 frames) | 3 | 0.007 ± 0.006 [0.000–0.010] | 136.9 ± 4.5 | 111.7 | 136 |
| 2 | none (baseline) | 3 | 0.013 ± 0.003 [0.010–0.015] | 146.9 ± 11.1 | 110.2 | 128 |
| 3 | stack 2 s (21 frames) | 3 | 0.003 ± 0.003 [0.000–0.005] | 151.0 ± 7.7 | 131.7 | 138 |
| 4 | gru 2 s (21 frames) | 3 | 0.010 ± 0.005 [0.005–0.015] | 159.2 ± 13.7 | 114.0 | 222 |

### Stage D

| Rank | Config | Seeds | Catch rate mean ± sd [min–max] | Miss distance mm mean ± sd | Median miss mm | Train time s |
|---:|---|---:|---|---|---:|---:|
| 1 | 1000 demos (baseline) | 3 | 0.013 ± 0.003 [0.010–0.015] | 146.9 ± 11.1 | 110.2 | 128 |
| 2 | 500 demos | 3 | 0.012 ± 0.003 [0.010–0.015] | 153.2 ± 6.8 | 110.6 | 134 |
| 3 | 250 demos | 3 | 0.003 ± 0.003 [0.000–0.005] | 148.4 ± 17.4 | 111.2 | 126 |
| 4 | 125 demos | 3 | 0.002 ± 0.003 [0.000–0.005] | 167.3 ± 9.0 | 142.2 | 121 |

### Factor effects

Catch-rate range counts only when the best and worst values are outside each other's seed spread; otherwise the factor is ordered by its miss-distance range.

- Stage A: catch range 0.035 (resolved), miss range 207.7 mm
- Stage B: catch range 0.018 (resolved), miss range 2015.7 mm
- Stage D: catch range 0.012 (resolved), miss range 20.4 mm
- Stage C: catch range 0.010 (resolved), miss range 22.2 mm

Stage E movers: ['A', 'B']; grid of top-2 values plus best single guess:

- 25 Hz, predict 2 s / exec 0.5, hist none, 1000 demos
- 25 Hz, predict 5 s / exec 0.25, hist none, 1000 demos
- 10 Hz, predict 2 s / exec 0.5, hist none, 1000 demos
- 10 Hz, predict 5 s / exec 0.25, hist none, 1000 demos
- 25 Hz, predict 2 s / exec 0.5, hist stack 0.5s, 1000 demos

### Stage E

| Rank | Config | Seeds | Catch rate mean ± sd [min–max] | Miss distance mm mean ± sd | Median miss mm | Train time s |
|---:|---|---:|---|---|---:|---:|
| 1 | 25 Hz, predict 2 s / exec 0.5, hist stack 0.5s, 1000 demos | 3 | 0.022 ± 0.003 [0.020–0.025] | 80.6 ± 0.1 | 64.9 | 129 |
| 2 | 25 Hz, predict 2 s / exec 0.5, hist none, 1000 demos | 3 | 0.035 ± 0.013 [0.025–0.050] | 95.3 ± 13.5 | 60.3 | 126 |
| 3 | 25 Hz, predict 5 s / exec 0.25, hist none, 1000 demos | 3 | 0.018 ± 0.010 [0.010–0.030] | 136.4 ± 3.4 | 91.3 | 120 |
| 4 | 10 Hz, predict 2 s / exec 0.5, hist none, 1000 demos (baseline) | 3 | 0.013 ± 0.003 [0.010–0.015] | 146.9 ± 11.1 | 110.2 | 128 |
| 5 | 10 Hz, predict 5 s / exec 0.25, hist none, 1000 demos | 3 | 0.018 ± 0.008 [0.010–0.025] | 165.1 ± 4.5 | 101.1 | 134 |
