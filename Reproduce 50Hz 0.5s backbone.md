# Reproduce the 50 Hz, 0.5 s backbone

This is the one-ball model in the Stage 2 progress report: 50 Hz, predict 1 s, execute 0.5 s, 0.24 s of stacked history, DeepSets encoder, seed 0. On the 200 one-ball test scenes it missed by 24.7 mm and caught 94% inside 50 mm.

The checkpoint name is `r50_p1_e0.5_hstack0.24s_d2000_deepsets_s0`. The script that first trained it is `rate50_diagnostic.py`. The commands below call the same trainer with the same config, without the ceiling rollouts that script runs first.

Every command is run from `robotnik_arm/experiments/ablation`, inside the RoboEnv `ml` environment. Datasets and checkpoints are gitignored, so a fresh clone has to build them.

## What the model is

Same DeepSets shape as the 25 Hz ceiling. The change is the clock.

| | This model | 25 Hz ceiling |
| --- | --- | --- |
| Policy rate | 50 Hz (every 50 Hz grid step) | 25 Hz (every second grid step) |
| Predict | 1.0 s, 50 steps | 2.0 s, 50 steps |
| Execute, then replan | 0.5 s, 25 steps | 1.0 s, 25 steps |
| History | stack, 0.24 s, 13 frames | stack, 0.5 s, 13 frames |
| Demos | 2,000 (1,000 clean + 1,000 offset parks) | same grid |
| Encoder | DeepSets, sum pool | same |
| Head | squared error | same |
| Seed trained | 0 only | 0, 1 and 2 |

Derived from `Config` in `config.py`, and checked before training:

- `stride = 1` grid step (20 ms)
- `horizon = 50`
- `execute = 25`
- `frames = 13`
- `name = r50_p1_e0.5_hstack0.24s_d2000_deepsets`

`features` stays `world`. DeepSets does not feed that 42-D vector in raw. `ballset.py` splits each frame into a 22-D ball vector and a 33-D robot vector. With one ball the sum pool is a sum of one embedding.

`DeepSetsPolicy` in `policy.py`:

- phi: Linear 286 → 256, ReLU, 256 → 256, ReLU, 256 → 128. The 286 is 13 frames × 22 ball features.
- trunk input: 13 × 33 robot features, plus the 128-D embedding (557).
- trunk: three Linear 512 + ReLU, same width as the flat policy.
- head: Linear 512 → 900, reshaped to `(50, 18)`.
- 1,444,868 parameters.

Labels are the expert command over the next 50 grid steps. Arm joint references (command dims 4–10) are stored relative to the arm position at the query tick. Wheels and joint velocities stay absolute. At rollout the arm reference is that query position plus the prediction, held for the 25 executed steps. `PolicyDriver` in `np_policy.py` does this, and the score below uses it.

Training is 20,000 AdamW steps, batch 1024, learning rate 1e-3, weight decay 1e-4, 500-step warmup, then cosine down to 1% of the peak. There is no validation split and no early stopping. The file on disk is the weights after step 20,000. `train.py` runs on CUDA with TF32 enabled. A machine without a CUDA GPU cannot run this trainer as written.

## Environment

From the repository root, with [pixi](https://pixi.sh) installed:

```powershell
git clone --recurse-submodules https://github.com/VModugno/RoboEnv.git RoboEnv
cd RoboEnv
pixi install
pixi install -e ml
pixi run smoke-test
pixi shell -e ml
```

Torch lives in the `ml` environment, together with the simulator. Start Python through that shell. On Windows, launching `.pixi\envs\ml\python.exe` directly skips the OpenMP setting pixi applies, and torch can fail to import.

Then:

```powershell
cd ..\robotnik_arm\experiments\ablation
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

`configs/robotnik_arm.json` and `models/` are read from `RoboEnv/` when they are not beside `robotnik_arm/`. `catch_env.py` finds them. Headless simulation is the default for data generation and scoring.

## Data

Training reads `data/grid50_offset.npz`. That file is the original 1,000 demos in `data/grid50.npz`, then 1,000 demos whose parking goal was shifted. `train.py` keeps the prefix through demo index 1999, which is 1,697,332 rows.

Skip this section when all three of these are already true:

- `data/seeds.json` has `"master": 2026`, 1,000 train seeds and 200 test seeds, with no overlap.
- `data/grid50.npz` is exactly those 1,000 training demos (`demo` max is 999).
- `data/grid50_offset.npz` exists, and `data/offset_parking/dataset.json` reports `grid_rows` 1697332.

Otherwise build them in this order. `data/demos/` must not already contain `demo_1000.npz` or later. `build_dataset.py` packs every contiguous `demo_XXXX.npz`, and `gen_offset_parking.py` refuses a `grid50.npz` that is not the original 1,000.

**1. One thousand expert demos and the fixed seeds.**

```powershell
python gen_data.py --demos 1000 --test 200 --workers 16
```

Writes `data/seeds.json`, `data/demos/demo_0000.npz` … `demo_0999.npz`, and `data/expert_rate_check.json`. Seeds come from `SeedSequence([2026, stream, i])`: train stream 10, test stream 20. Odd-indexed demos are collected with DART action noise and clean labels. The same run also rolls the expert at 5, 10, 25 and 1000 Hz on all 200 test seeds. Those checks are not used by this model. If `expert_rate_check.json` is already complete, they are skipped. This is the long step: about 1,800 full episodes at 1 ms simulation steps.

**2. Downsample onto the 50 Hz grid.**

```powershell
python build_dataset.py --require 1000
```

Writes `data/grid50.npz`. Samples start at simulation step 2 and then every 20th step (`FIRST_TICK_STEP` and `1000 // 50` in `build_dataset.py`).

**3. Append 1,000 offset-parking demos.**

```powershell
python gen_offset_parking.py
```

Needs `data/grid50.npz` and `data/seeds.json`. Scene seeds are `seeds.json` `"train"` in order. After the expert plans, the parking position is moved in the route frame: sideways ~ N(0, 50 mm), along-track ~ N(0, 20 mm), vector capped at 150 mm, heading unchanged. Master seed 2026, offset stream 40. Sixteen worker processes. Logged commands are the commands that were executed.

Writes `data/offset_parking/demo_0000.npz` … `demo_0999.npz`, `data/offset_parking/dataset.json`, and `data/grid50_offset.npz` (the original 1,000 demos, then these 1,000).

The published `dataset.json` records:

| Field | Value |
| --- | --- |
| `n` | 1000 |
| `catch_rate` | 1.0 |
| `median_miss_mm` | 0.248 |
| `grid_rows` | 1697332 |

`grid_rows` is the check that the combined file is the one this model trained on. A catch rate below 1, or a different row count, means the grid is not the published training set.

## Train

Paste this into the ablation directory as `train_rate50_s0.py` and run `python train_rate50_s0.py`. It builds the same `Config` as `primary_cfg()` in `rate50_diagnostic.py` and calls `train.py` with the same arguments that script used for seed 0: 20,000 steps, and the defaults of batch 1024 and learning rate 1e-3.

```python
import json
import os
import subprocess
import sys
from pathlib import Path

from config import Config

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

cfg = Config(
    rate_hz=50,
    predict_s=1.0,
    execute_frac=0.5,
    history="stack",
    history_s=0.24,
    demos=2000,
    encoder="deepsets",
    pool="sum",
)
assert (cfg.horizon, cfg.execute, cfg.frames, cfg.name) == (
    50,
    25,
    13,
    "r50_p1_e0.5_hstack0.24s_d2000_deepsets",
), (cfg.horizon, cfg.execute, cfg.frames, cfg.name)

out = Path("results/models") / f"{cfg.name}_s0.pt"
out.parent.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable,
    "train.py",
    "--config",
    json.dumps(cfg.to_dict()),
    "--seed",
    "0",
    "--out",
    str(out),
    "--steps",
    "20000",
    "--data",
    "data/grid50_offset.npz",
]
raise SystemExit(subprocess.call(cmd))
```

`train.py` overwrites `out` if it is already there. Move an existing checkpoint aside first if you want to keep it. `rate50_diagnostic.py` does the opposite: it reuses `results/models/r50_p1_e0.5_hstack0.24s_d2000_deepsets_s0.pt` when that file exists, and it exits before training unless the three 25 Hz ceiling exports `r25_p2_e0.5_hstack0.5s_d2000_deepsets_s{0,1,2}.npz` are present.

The trainer prints one JSON object at the end. The published run was:

```text
{"train_time_s": 67.6, "final_loss": 0.006968546658754349, "train_rows": 1697332, "params": 1444868}
```

`train_rows` and `params` should match exactly. `train_time_s` follows the GPU. `final_loss` is the last training batch with TF32 on, so another GPU can differ in the later digits. The log is also written by the diagnostic to `results/logs/r50_p1_e0.5_hstack0.24s_d2000_deepsets_s0.train.log`. The direct command above prints to the terminal only.

Seeds 1 and 2 were not trained. The diagnostic stopped after seed 0 because the along-track parking bias did not shrink (`tail_moved` is false in `results/stage2/rate50_diagnostic.json`).

## Score

Paste this as `score_rate50_s0.py` in the same directory and run `python score_rate50_s0.py`. It exports the NumPy weights the way `eval_cfg` does, then scores them with `parking_split.eval_weights` on `seeds.json` `"test"` truncated to 200. That is the pass behind the table in the progress report. Catch at 10 mm is a closest approach under 10 mm. Catch at 50 mm is a closest approach under 50 mm. The along-track numbers compare the learned base with the expert at that closest approach, in the expert's route frame.

```python
import json
from pathlib import Path

from parking_split import eval_weights, load_experts
from policy import export_numpy

name = "r50_p1_e0.5_hstack0.24s_d2000_deepsets_s0"
ckpt = Path("results/models") / f"{name}.pt"
weights = ckpt.with_suffix(".npz")
export_numpy(ckpt, weights)
test = json.loads(Path("data/seeds.json").read_text())["test"][:200]
print(f"expert reference on {len(test)} test seeds", flush=True)
experts = load_experts(test)
print(f"rollout {name}", flush=True)
stats = eval_weights(weights, test, experts)
split = stats["split"]
print(
    f"catch10 {stats['catch_rate']:.3f}  catch50 {stats['catch_rate_50mm']:.3f}  "
    f"miss {stats['miss_mm']:.1f} mm  along {split['base_along_median_mm']:+.1f}  "
    f"ahead {split['frac_ahead']:.3f}",
    flush=True,
)
```

Scoring fans out to 16 simulator processes (`pool_map` in `run_relative.py`). The published seed 0 line, from `results/stage2/rate50_diagnostic.json`, is:

| Measure | Published seed 0 |
| --- | --- |
| Mean miss | 24.7 mm |
| Median miss | 19.3 mm |
| Catch at 10 mm | 17% |
| Catch at 50 mm | 94% |
| Signed along-track, median | +19.4 mm |
| Fraction of episodes ahead of the expert | 66.5% |

A new training run should land near that row. TF32 and GPU reduction order mean the weights will not be bitwise identical to the published file, so a miss a couple of millimetres away is still this model. A miss near the 35 mm ceiling, or a `train_rows` other than 1697332, means the data or the config drifted.

To watch one episode in the GUI, after the `.npz` export:

```powershell
python show_catch.py --weights results/models/r50_p1_e0.5_hstack0.24s_d2000_deepsets_s0.npz --seed 843567668329390999
```

`show_catch.py` reads the rate from the checkpoint, so this replays at 50 Hz. The printed success flag uses the 10 mm catch.

## Files

| Piece | Where |
| --- | --- |
| This config | `primary_cfg` in `rate50_diagnostic.py` |
| Trainer | `train.py` |
| Training grid | `gen_data.py`, `build_dataset.py`, `gen_offset_parking.py` → `data/grid50_offset.npz` |
| Test scenes | `data/seeds.json` key `"test"`, first 200 |
| Checkpoint | `results/models/r50_p1_e0.5_hstack0.24s_d2000_deepsets_s0.pt` |
| Rollout weights | same stem, `.npz`, written by `policy.export_numpy` |
| Published score | `results/stage2/rate50_diagnostic.json`, key `primary.s0` |
| Published train log | `results/logs/r50_p1_e0.5_hstack0.24s_d2000_deepsets_s0.train.log` |
