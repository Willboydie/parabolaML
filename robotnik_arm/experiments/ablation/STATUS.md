# Progress summary — single-ball interception (Stage 1)

Last updated 4 Nov 2026. Detailed ablation tables: [STAGE1_RESULTS.md](STAGE1_RESULTS.md). Error diagnosis: `results/diagnose.json`.

## Where we are

Stage 1 is done: timing and backbone for a **plain MSE action-chunk policy** on one ball. That config is frozen as the Stage 2 backbone and as the **single-ball ceiling** every two-ball model has to beat.

| | Catch (10 mm) | Mean miss | Median miss |
|---|---|---|---|
| **Stage 1 winner (learned)** | **2.2% ± 0.3%** | **80.6 ± 0.1 mm** | 64.9 mm |
| Same timing, no history | 3.5% ± 1.3% | 95.3 mm | 60.3 mm |
| Expert at 25 Hz | 100% | 0.38 mm | 0.17 mm |

The policy can finish a catch from a perfect expert pose (0.7 mm miss, 100% catch with 0.5 s of control). Full-episode miss is **compounding error over the ~15 s chase**, not a failure to aim at the end. Two-ball work has not started.

## Best models (share these)

Checkpoints are **not in git** (see `.gitignore`). They live on this machine as ~5 MB files:

```
robotnik_arm/experiments/ablation/results/models/
```

**Winner — use this as the Stage 2 backbone**

| File | What it is |
|---|---|
| `r25_p2_e0.5_hstack0.5s_d1000_s1.pt` | PyTorch checkpoint (training seed 1, highest catch of the three) |
| `r25_p2_e0.5_hstack0.5s_d1000_s1.npz` | NumPy export for CPU rollouts (`np_policy.py`) |

The same config, other seeds (for spread): `..._s0.pt` / `..._s2.pt` and matching `.npz`. All three seeds: 2.0%, 2.5%, 2.0% catch; miss 80.5–80.8 mm.

**Runner-up** (25 Hz, no history — noisier catch, worse miss): `r25_p2_e0.5_hnone_d1000_s{0,1,2}.pt`

To replay a catch locally:

```bash
cd robotnik_arm/experiments/ablation
python show_catch.py --weights results/models/r25_p2_e0.5_hstack0.5s_d1000_s1.npz
```

Load in Python:

```python
# rollouts (no torch)
from np_policy import NumpyPolicy
model = NumpyPolicy("results/models/r25_p2_e0.5_hstack0.5s_d1000_s1.npz")

# training / GPU
from policy import load_policy
model = load_policy("results/models/r25_p2_e0.5_hstack0.5s_d1000_s1.pt")
```

Zip to share (~15 MB for the three winner seeds, both formats):

```text
r25_p2_e0.5_hstack0.5s_d1000_s{0,1,2}.{pt,npz}
```

Also send `results/stage2_config.json` so the hyperparams travel with the weights.

## Exact architecture of the winner (`ChunkPolicy`)

Defined in `policy.py`. History is **stack**, not GRU — there is no recurrent layer in this checkpoint.

```
13 × 42 raw obs  (oldest → newest, padded with the first tick at episode start)
        │
        ▼
per-dim z-score with train obs_mean / obs_std, then clip to ±10
        │
        ▼
flatten → 546
        │
        ▼
Linear 546 → 512, ReLU
Linear 512 → 512, ReLU
Linear 512 → 512, ReLU
        │
        ▼
Linear 512 → 900
        │
        ▼
reshape (50, 18)
        │
        ▼
un-z-score with per-horizon-step act_mean / act_std  →  relative command chunk
```

| | |
|---|---|
| Class | `ChunkPolicy` (`experiments/ablation/policy.py`), `history='stack'` so `self.gru is None` |
| Trainable params | **1,267,076** (seed-1 log: `params: 1267076`) |
| Input | `(B, 13, 42)` observations, oldest first. No past actions. |
| Frames | `1 + round(0.5 × 25) = 13` at 25 Hz (12 prior ticks + current; 40 ms apart) |
| Trunk | 3 × Linear+ReLU, width 512. Input width = `42 × 13 = 546`. |
| Head | single Linear `512 → 50×18 = 900`, no activation |
| Output | `(B, 50, 18)` denormalised command chunk |
| Horizon / execute | 50 predicted steps (2.0 s), execute 25 (1.0 s), then re-query |
| Normalisation | obs: 42-D mean/std over all 848,652 train rows; actions: (50, 18) mean/std from a 50k probe of **relative** chunks. Std floored at 1e-3. Input clip ±10 after z-score. |
| Labels | 18-D expert command at each future tick. Joint refs (dims 4–10) are **relative to arm q at query time**, not at the future tick. Wheels and joint vels are absolute. |
| Decode at rollout | wheels = pred[:4]; `q_des = q_at_query + pred[4:11]` (q frozen for the executed 25 steps); `qd_des = pred[11:18]` |
| Loss | plain MSE on the z-scored chunk, mean over batch × horizon × 18 |
| Init | PyTorch default Linear (Kaiming uniform / zeros bias). No dropout, LayerNorm, residual, or output activation. |

Layer param counts: `546×512+512 = 280,064`; two `512×512+512 = 262,656` each; head `512×900+900 = 461,700`. Buffers (`obs_mean/std`, `act_mean/std`) are not trainable.

**Observation, 42-D, one frame** (`catch_env.OBS_NAMES`):

| idx | name | |
|---|---|---|
| 0 | `time` | elapsed seconds |
| 1–3 | `ball_pos_xyz` | from last three 1 ms samples |
| 4–6 | `ball_vel_xyz` | same |
| 7–9 | `ball_acc_xyz` | same |
| 10–12 | `base_pos_xyz` | |
| 13–14 | `bearing_sin`, `bearing_cos` | |
| 15–17 | `base_linvel_xyz` | |
| 18–20 | `base_angvel_xyz` | |
| 21–27 | `arm_q_0..6` | rad |
| 28–34 | `arm_qd_0..6` | rad/s |
| 35–38 | `wheel_qd_0..3` | rad/s |
| 39–41 | `ee_xyz` | end-effector position |

**Command, 18-D, one horizon step:**

| idx | | units | at rollout |
|---|---|---|---|
| 0–3 | wheel velocity | rad/s | used as-is |
| 4–10 | arm joint reference, **relative** to query-time q | rad | added to `q` at the query that produced the chunk |
| 11–17 | arm joint velocity reference | rad/s | used as-is |

## Frozen hyperparameters (Stage 2 lock)

| | Value |
|---|---|
| Sim rate | 1 ms (fixed) |
| Policy rate | **25 Hz** |
| Predict / execute | **2 s / 1 s** (50-step chunk, execute 25, receding horizon) |
| History | **0.5 s observation stack** (13 frames, no past actions, no GRU) |
| Trunk | 3 × 512 ReLU MLP (546 → 512 → 512 → 512) |
| Head | linear `512 → 900`, MSE on standardised action chunks |
| Action | 18-D: 4 wheel vel (rad/s) + 7 joint refs (relative rad) + 7 joint vel refs |
| Observation | 42-D: time, ball p/v/a, base pose/vel, arm q/qd, wheel qd, EE pos |
| Data | 1,000 one-ball demos (half DART), nested prefixes; 200 held-out test seeds |
| Train | AdamW 1e-3, wd 1e-4, batch 1,024, 20,000 steps, 500-step linear warmup then cosine to 1% of lr, no val selection |
| Low-level | every sim step: wheel `kd=30` velocity servo; arm gravity-compensated PD (`kp=100`, `kd=30`, `max_q_lead=0.15` rad, `max_torque=30` N·m) |
| Catch | 10 mm EE-to-ball after the ball has left and re-entered the 5×5 m box |

Rates the expert still catches at: 5, 10, 25 Hz (none dropped). 5 Hz learned policies never catch.

What actually moved Stage 1: **policy rate** (25 Hz >> 10 Hz >> 5 Hz). Chunking: 0.5 s horizons collapse; 2 s / 1 s beat 5 s on miss. History and data size were weak; 500 vs 1,000 demos is flat.

## Outstanding unknowns

1. **Two-ball (Stage 2).** Not started. Winner above is the ceiling, not a two-ball recipe. Target choice / multimodal heads are still open.
2. **How to stop compounding on the drive.** Hand-off miss is 5 mm at 4 s of control and 22 mm at 8 s. One-chunk EE error jumps from ~1 mm to ~8 mm at mid-drive. Recurrence, a longer executed chunk *at 25 Hz*, or an explicit intercept target are untested answers. A bigger MLP or more than 1,000 demos are unlikely: data already plateaued and Part 1 shows the expert commands are already fit (train MSE 0.018, test 0.029).
3. **Heads other than MSE.** Stage 1 was MSE by design (one right answer). GMM / BET / diffusion wait for two-ball, but a two-ball model that still compounds on the drive will miss for the same reason.
4. **Why 25 Hz helps so much.** Expert catch is already 100% at 5 Hz. The learned 5 Hz policy never leaves the start. Open whether that is copycat of the first in-place turn, or missing high-rate feedback during the drive.
5. **Fair two-ball eval.** Test seeds and demo subsets must stay nested and disjoint from Stage 1’s 200 test seeds if we want a comparable ceiling.

## Code map

| Path | Role |
|---|---|
| `experiments/ablation/catch_env.py` | 1 ms sim, 25 Hz policy, 1 ms hold |
| `experiments/ablation/policy.py` / `np_policy.py` | MLP + MSE chunk head |
| `experiments/ablation/train.py` / `run_ablation.py` | GPU train, 16-way CPU eval |
| `experiments/ablation/diagnose.py` | Parts 1–3 error diagnosis |
| `experiments/ablation/results/models/` | checkpoints (gitignored) |
| `experiments/ablation/results/runs/` | per-seed catch/miss JSON |
| `experiments/ablation/data/seeds.json` | train/test episode seeds |
