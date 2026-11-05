# Stage 1 ablation: single-ball interception (D1, MLP trunk, MSE chunk head)

## Chosen config for Stage 2

**25 Hz, predict 2 s / execute 1 s, 0.5 s observation stack, 1,000 demos, 3×512 MLP, MSE chunk head.**

| | Catch rate (10 mm) | Mean miss | Median miss |
|---|---|---|---|
| Stage 1 winner | 2.2% ± 0.3% | **80.6 ± 0.1 mm** | 64.9 mm |
| Same timing, no history | 3.5% ± 1.3% | 95.3 ± 13.5 mm | 60.3 mm |
| Expert at 25 Hz | 100% | 0.38 mm | 0.17 mm |

The two 25 Hz / 2 s configs fall inside each other's seed spread on catch rate, so the ranking rule takes the lower miss. The stacked policy is also the more stable of the two (catch 2.0–2.5% across seeds, miss almost identical). This is the fixed backbone and the ceiling for every two-ball model. It replaces the earlier H = 8 default.

Written to `results/stage2_config.json`.

## Stage 0 checks

The expert, acting only at the policy rate with the 1 ms low-level loop holding the latest command, still catches all 200 test seeds at 5, 10 and 25 Hz. No rate was dropped. The 10 mm catch radius was left in place: the learned baseline is ~1% catch / 147 mm miss, so the factors have room to differ.

| Expert rate | Episodes | Catch rate (10 mm) | Mean miss mm | Median miss mm | Max miss mm | Catch at 5 mm | Catch at 2 mm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 Hz | 200 | 1.000 | 1.07 | 1.11 | 2.18 | 1.000 | 0.995 |
| 10 Hz | 200 | 1.000 | 0.95 | 1.09 | 1.97 | 1.000 | 1.000 |
| 25 Hz | 200 | 1.000 | 0.38 | 0.17 | 1.80 | 1.000 | 1.000 |
| 1000 Hz | 200 | 1.000 | 0.32 | 0.12 | 1.53 | 1.000 | 1.000 |

One timed train-plus-eval (20,000 GPU steps, 200 CPU rollouts, 16 workers) is ~2 min train + ~6 min eval wall. Stages A–D were 42 runs in 73 min; Stage E was 6 new runs in 14 min.

## Stage A — policy rate

| Rank | Config | Seeds | Catch rate mean ± sd [min–max] | Miss distance mm mean ± sd | Median miss mm | Train time s |
|---:|---|---:|---|---|---:|---:|
| 1 | 25 Hz | 3 | 0.035 ± 0.013 [0.025–0.050] | 95.3 ± 13.5 | 60.3 | 126 |
| 2 | 10 Hz (baseline) | 3 | 0.013 ± 0.003 [0.010–0.015] | 146.9 ± 11.1 | 110.2 | 128 |
| 3 | 5 Hz | 3 | 0.000 ± 0.000 [0.000–0.000] | 303.0 ± 10.3 | 250.8 | 133 |

25 Hz is the only rate that both catches and gets close. 5 Hz never catches; the expert can, so this is a model failure, not an excluded rate.

## Stage B — chunking

| Rank | Config | Seeds | Catch rate mean ± sd [min–max] | Miss distance mm mean ± sd | Median miss mm | Train time s |
|---:|---|---:|---|---|---:|---:|
| 1 | predict 2 s (20 steps), execute 0.5 = 1 s (10 steps) (baseline) | 3 | 0.013 ± 0.003 [0.010–0.015] | 146.9 ± 11.1 | 110.2 | 128 |
| 2 | predict 5 s (50 steps), execute 0.25 = 1.3 s (13 steps) | 3 | 0.018 ± 0.008 [0.010–0.025] | 165.1 ± 4.5 | 101.1 | 134 |
| 3 | predict 5 s (50 steps), execute 0.5 = 2.5 s (25 steps) | 3 | 0.008 ± 0.008 [0.000–0.015] | 196.6 ± 3.8 | 100.4 | 136 |
| 4 | predict 2 s (20 steps), execute 0.25 = 0.5 s (5 steps) | 3 | 0.000 ± 0.000 [0.000–0.000] | 189.3 ± 6.5 | 164.5 | 129 |
| 5 | predict 0.5 s (5 steps), execute 0.5 = 0.3 s (3 steps) | 3 | 0.002 ± 0.003 [0.000–0.005] | 333.8 ± 11.8 | 313.2 | 127 |
| 6 | predict 0.5 s (5 steps), execute 0.25 = 0.1 s (1 steps) | 3 | 0.000 ± 0.000 [0.000–0.000] | 2162.6 ± 275.7 | 1994.1 | 129 |

A 0.5 s horizon cannot copy the expert's drive: the base spins in place. Predicting 5 s and executing a quarter of it slightly raises catch vs the 2 s / ½ baseline, but the seed spreads overlap and miss is worse, so 2 s / 1 s stays ranked first.

## Stage C — history

| Rank | Config | Seeds | Catch rate mean ± sd [min–max] | Miss distance mm mean ± sd | Median miss mm | Train time s |
|---:|---|---:|---|---|---:|---:|
| 1 | stack 0.5 s (6 frames) | 3 | 0.007 ± 0.006 [0.000–0.010] | 136.9 ± 4.5 | 111.7 | 136 |
| 2 | none (baseline) | 3 | 0.013 ± 0.003 [0.010–0.015] | 146.9 ± 11.1 | 110.2 | 128 |
| 3 | stack 2 s (21 frames) | 3 | 0.003 ± 0.003 [0.000–0.005] | 151.0 ± 7.7 | 131.7 | 138 |
| 4 | gru 2 s (21 frames) | 3 | 0.010 ± 0.005 [0.005–0.015] | 159.2 ± 13.7 | 114.0 | 222 |

History is a weak factor on its own. A 0.5 s stack is a few centimetres closer; 2 s of stacked frames or a GRU over 2 s does not help.

## Stage D — data size (nested prefixes)

| Rank | Config | Seeds | Catch rate mean ± sd [min–max] | Miss distance mm mean ± sd | Median miss mm | Train time s |
|---:|---|---:|---|---|---:|---:|
| 1 | 1000 demos (baseline) | 3 | 0.013 ± 0.003 [0.010–0.015] | 146.9 ± 11.1 | 110.2 | 128 |
| 2 | 500 demos | 3 | 0.012 ± 0.003 [0.010–0.015] | 153.2 ± 6.8 | 110.6 | 134 |
| 3 | 250 demos | 3 | 0.003 ± 0.003 [0.000–0.005] | 148.4 ± 17.4 | 111.2 | 126 |
| 4 | 125 demos | 3 | 0.002 ± 0.003 [0.000–0.005] | 167.3 ± 9.0 | 142.2 | 121 |

Catch rate is flat from 500 to 1,000; it drops below 250. Keep 1,000.

## Factor effects

Catch-rate range counts only when the best and worst values sit outside each other's seed spread.

| Stage | Catch range | Miss range | Moved results? |
|---|---:|---:|---|
| A rate | 0.035 (resolved) | 208 mm | **yes, strongest** |
| B chunking | 0.018 (resolved) | 2016 mm | **yes** |
| D data | 0.012 (resolved) | 20 mm | weakly |
| C history | 0.010 (resolved) | 22 mm | weakly |

Stage E is the 2 × 2 of the top two values of A and B, plus the best single guess (A+B+C+D winners).

## Stage E — combine

| Rank | Config | Seeds | Catch rate mean ± sd [min–max] | Miss distance mm mean ± sd | Median miss mm | Train time s |
|---:|---|---:|---|---|---:|---:|
| 1 | **25 Hz, predict 2 s / exec 0.5, stack 0.5 s, 1000 demos** | 3 | 0.022 ± 0.003 [0.020–0.025] | 80.6 ± 0.1 | 64.9 | 129 |
| 2 | 25 Hz, predict 2 s / exec 0.5, hist none, 1000 demos | 3 | 0.035 ± 0.013 [0.025–0.050] | 95.3 ± 13.5 | 60.3 | 126 |
| 3 | 25 Hz, predict 5 s / exec 0.25, hist none, 1000 demos | 3 | 0.018 ± 0.010 [0.010–0.030] | 136.4 ± 3.4 | 91.3 | 120 |
| 4 | 10 Hz, predict 2 s / exec 0.5, hist none, 1000 demos (baseline) | 3 | 0.013 ± 0.003 [0.010–0.015] | 146.9 ± 11.1 | 110.2 | 128 |
| 5 | 10 Hz, predict 5 s / exec 0.25, hist none, 1000 demos | 3 | 0.018 ± 0.008 [0.010–0.025] | 165.1 ± 4.5 | 101.1 | 134 |

Raising the rate from 10 Hz to 25 Hz is the whole gain. Lengthening the chunk to 5 s does not help at 25 Hz. Adding 0.5 s of stacked observations on the 25 Hz / 2 s policy trades a noisier 3.5% catch for a stable 2.2% catch that finishes 15 mm closer; that is the Stage 2 pick.

The learned ceiling is far from the expert (100% / 0.4 mm). Stage 2 two-ball models should be judged against **2.2% catch / 81 mm mean miss**, not against the expert.

## Setup as run

**Environment.** `catch_env.py` rebuilds the scene from `robotnik_arm_task.py` (same friction,
contact, gripper and controller parameters) and steps PyBullet directly. `SimInterface.Step`
also computes a mass matrix every step; with zero motor damping and elasticity in
`robotnik_arm.json` it only multiplies a zero vector, so it is skipped. Running the
1 kHz expert in this environment reproduces the recorded native episodes to the millisecond
(13.786 s, 14.885 s, 16.785 s catch times on `datasets/one_ball` test seeds).

**Three rates.**
- Simulation: 1 ms, fixed.
- Policy: 5, 10 or 25 Hz. The first tick is at the third sim step, once three ball samples exist.
- Low-level controller: every sim step, holding the latest command. The wheels use the
  servo's velocity loop, `-30 (qd - cmd)`. The arm uses the expert's own law,
  `kp (q_ref - q) + kd (qd_ref - qd) + g(q)`, with the reference clipped to within
  0.15 rad of the measured angle and to the joint limits, and torque clipped to 30 N m.

**Command.** 18 numbers per policy step: 4 wheel velocities, 7 arm joint references and
7 joint reference velocities. These are exactly what the expert hands to its own low-level
loop. The model predicts arm references relative to the arm angles at query time.

**Observation.** 42 numbers per frame:
- elapsed time
- ball position, velocity and acceleration, estimated from the last three 1 ms samples
  (what the expert sees)
- base position, bearing (sine and cosine), base linear and angular velocity
- arm joint angles and velocities, wheel velocities
- end-effector position

There are no past actions.

**Catch.** The end effector comes within 10 mm of the ball after the ball has left the
5 × 5 m box and while it is back inside, as in `robotnik_arm_task.py`. The base no longer
has to have finished navigating, because a policy has no navigation phase. An episode ends
0.1 s after the first catch (so the closest approach is recorded) or 1 s after the ball
reaches its endpoint.

**Miss distance.** The closest end-effector-to-ball distance in the catch window, per
episode, averaged over the 200 test episodes.

**Data.**
- `gen_data.py` logs 1,000 expert demos at 1 kHz, together with each episode seed.
- Odd-indexed demos receive the existing DART-style noise (wheel 0.08 rad/s, arm torque
  0.2 N m, held 50 ms) with clean labels, so every nested prefix is half noisy and half clean.
- The 200 test seeds come from a separate seed stream (`data/seeds.json`) and are never
  used for training.
- `build_dataset.py` downsamples the logs once to a shared 50 Hz anchor grid. Every policy
  rate divides 50 Hz, so each rate's history frames and chunk labels are exact grid samples.
- Every rate trains on the same 848,652 anchor states. Only the spacing of history and
  labels differs, so faster rates don't get more data.
- Data sizes are prefixes of the demo list: 125, 250, 500 and 1,000.

**Model and training.**
- Trunk: a 3 × 512 ReLU MLP for every config.
- History: none, or stacked frames, or a GRU (hidden size 128) over the frames, whose final
  state is concatenated with the current frame. History spacing is one policy step;
  observations only, no past actions.
- Head: linear, producing `horizon × 18`.
- Loss: plain MSE on per-step, per-dimension standardized chunks.
- Optimizer: AdamW (learning rate 1e-3, weight decay 1e-4), batch 1,024, 20,000 steps,
  cosine schedule.
- Every config uses the same steps, the same data subsets and the same test seeds.
  There is no validation-based selection; models are compared only on rollouts.

**Chunk execution.** Predict `round(predict_s × rate)` steps and execute
`round(fraction × horizon)` of them, at least 1, before querying again.

**Compute.**
- Training runs on the RTX A4500, up to 3 runs at once, with `torch.set_num_threads(1)`.
- Rollouts run in 16 persistent worker processes with 1 BLAS thread each, in batches of
  25 episodes.
- Rollout workers run the policy forward pass in NumPy (`np_policy.py`, matching torch to
  float32 rounding). They don't import torch at all, because each simulator process already
  commits about 3.3 GB on Windows.
