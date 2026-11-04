# Expert data collection

Use the same RoboEnv Python environment as the simulation. From this folder:

```bash
python collect_expert.py --balls 2 --output datasets/two_balls
python collect_expert.py --balls 1 --output datasets/one_ball
```

Each invocation defaults to **500 complete episodes**: 200 `train_dart`, 200
`train_normal`, 50 `validation`, and 50 `test`. Use a separate output folder for
each ball mode. One-ball mode generates only one trajectory, with target index
zero and no target-choice draw. Two-ball mode validates both trajectories and
randomly chooses one, as in the interactive expert. Both modes preserve the
current target speed/height settings and original robot controller speeds.

These counts include successful catches and episodes that hit the step cap.
Failures are not silently dropped or replaced. Each episode includes an outcome,
reason and final error. Process errors appear in the manifest and return a nonzero
exit code; rerun with `--resume` to retry them. The script does not filter a split
after seeing its results.

Preview the default plan without running or creating files:

```bash
python collect_expert.py --balls 2 --dry-run
```

Small check, followed by a resumed run:

```bash
python collect_expert.py --balls 2 --dart 1 --normal 1 --validation 1 --test 1 --output datasets/smoke
python collect_expert.py --balls 2 --dart 1 --normal 1 --validation 1 --test 1 --output datasets/smoke --resume
```

## Configuration

| Option | Default | Meaning |
|---|---:|---|
| `--balls` | 2 | 1 or 2 targets |
| `--dart` | 200 | Noisy training episodes |
| `--normal` | 200 | Clean training episodes |
| `--validation` | 50 | Clean validation episodes |
| `--test` | 50 | Clean held-out test episodes |
| `--seed` | 2026 | Master seed, reproducible across worker counts |
| `--max-steps` | 60000 | Per-episode cap (60 simulation seconds at the default timestep) |
| `--target-speed` | Random 0.5–0.7 | Optional fixed positive launch speed in m/s |
| `--sample-hz` | 100 | Requested recording rate, rounded to an integer simulation-step stride |
| `--wheel-noise-std` | 0.08 | Wheel-command noise standard deviation, rad/s |
| `--arm-noise-std` | 0.2 | Arm-torque noise standard deviation, N m |
| `--noise-hold-seconds` | 0.05 | How long to hold each perturbation before drawing another |
| `--workers` | 1 | Number of independent episode subprocesses |
| `--episode-timeout` | 600 | Wall-clock seconds allowed for each subprocess |
| `--assets-dir` | Script folder | Folder containing `configs/robotnik_arm.json` and `models/` |
| `--resume` | Off | Verify configuration/source and skip already completed files |

The collector always runs headless. Its explicit settings take precedence over
simulation environment variables, including target speed. It uses the Python interpreter that launches
it. Episode counts can be zero, but at least one must be positive. Do not run two
collectors simultaneously into the same output folder. Separate train/validation/
test episode seeds prevent shared trajectories; never randomly split rows from
one episode across model splits. Keep validation and test out of model fitting.
All records from `train_dart` and `train_normal` are training data.

For example, to use the installed environment here:

```bash
/home/nick/projects/RoboEnv/.pixi/envs/default/bin/python collect_expert.py --balls 2 --workers 2 --output datasets/two_balls
```

Resume requires the original dataset settings and unchanged source files. To
change settings or collect with a new expert implementation, use a fresh folder.
The manifest is updated after every completed job and completed episode files
are written atomically. `.jobs/` stores reproducible job specifications; `logs/`
contains one simulator log per episode. Ctrl+C terminates active workers and
retains completed files for resuming.

## DART-style perturbations

Only `train_dart` receives bounded Gaussian action noise: independently drawn for
four wheel velocity commands and seven arm torques, clipped at three standard
deviations and held for 50 ms by default. Commands are clipped to the expert's
wheel-speed and arm-torque bounds. The initial two observation timesteps and
fixed finger commands are not perturbed. Noise is applied at every simulation
step, independent of the recording stride.

The expert continues to observe the actual disturbed robot state and produces
clean recovery commands. **`y` is that clean command, not the noisy command used
to move the robot.** Both are stored for inspection. Normal, validation and test
episodes use clean commands throughout. This implements configurable fixed-noise
DART-style demonstrations; it does not learn a covariance from a trained policy.

## Files and arrays

Each split contains `episode_00000.npz`, etc., loadable with
`np.load(path, allow_pickle=False)`. `dataset.json` records configuration and source
hashes; `schema.json` maps input feature names to column slices; `manifest.json`
lists completed episodes and outcomes.

- `X`: `(T, 96)`, float32 model inputs listed below.
- `y`: `(T, 11)`, float32 clean expert labels: wheel velocities `[FR, FL, BL, BR]`
  in rad/s followed by seven arm torques in N m, in simulator motor order.
- `applied_action`, `action_noise`: `(T, 11)`, actual action and its difference
  from `y`, after clipping.
- `expert_command`, `applied_command`: `(T, 17)`, including six fixed-finger
  torque command slots (zeros; fingers are position-held by the simulation).
- Named observation fields: `target_positions (T,2,3)`, `target_mask (T,2)`,
  `selected_target (T,)`, `target_history (T,3,2,3)`, `target_history_mask (T,3,2)`,
  `history_time_offsets (T,3)`, `base_pos`, `base_ori` (xyzw), `base_bearing`,
  `base_velocity` (world linear xyz then angular xyz), `joint_pos (T,17)`,
  `joint_vel (T,17)`, `ee_pos`, and `previous_applied_action`.
- Expert diagnostics, separate from `X`: `q_des`, `qd_des`, `expert_phase`
  (`align=0, drive=1, orient=2, done=3`), `expert_ready`, `nav_done`, `intercept`,
  `intercept_time`, `base_goal`, `base_goal_bearing`, and `ee_error`. Unplanned
  interception fields use NaN; observations and actions remain finite.
- `next_joint_pos`, `next_base_pos`, `next_target_positions`: observations after
  **one physics step**, not after the next saved row. At a decimated recording
  rate the next saved row is farther ahead. `simulation_dt` in metadata gives
  the transition interval.
- `time`, `simulation_step`, `terminated`, `truncated`: alignment and episode
  boundaries. The final saved row has `terminated=True` for success, otherwise
  `truncated=True`. Initial three physics steps and the final executed step are
  always included, so spacing is not perfectly uniform at episode boundaries.
- `metadata_json`: JSON string containing seeds, ball count, selected target,
  named joint ordering, timestep/rate, initial trajectory parameters, outcome,
  success flag, stopping reason, final time/error and sample count. True launch
  parameters are for reproducibility/analysis and are not included in `X`.

### Input layout

| Feature | Width |
|---|---:|
| Last three observed positions of both target slots (oldest first) | 18 |
| Target-history validity masks | 6 |
| Observation time offsets relative to current time | 3 |
| Base position xyz | 3 |
| Base quaternion xyzw | 4 |
| Sine and cosine of base bearing | 2 |
| Base linear and angular velocity in world coordinates | 6 |
| All measured motor positions (4 wheels, 7 arm, 6 fingers) | 17 |
| All measured motor velocities | 17 |
| End-effector position xyz | 3 |
| Previous applied wheel/arm command | 11 |
| Selected-target one-hot cue | 2 |
| Active-target mask | 2 |
| Simulation time and physics timestep | 2 |

Missing target slots and unavailable history entries are zero padded and masked.
History is updated at the physics rate even when output is sampled at 100 Hz.
A two-ball scene alone does not reveal the expert's random choice; the selected-
target cue is part of the policy input and must also be provided at inference.
Internal planner/reference state is recorded as diagnostics because the expert
is stateful; a learned controller may also need recurrence or longer history.

```python
import json
import numpy as np

with np.load('datasets/two_balls/train_dart/episode_00000.npz', allow_pickle=False) as episode:
    X, y = episode['X'], episode['y']
    outcome = json.loads(str(episode['metadata_json']))
    print(X.shape, y.shape, outcome['success'])
```

## Missing simulation dependencies

The collector checks worker imports before launching episodes. If your active
`.venv` lacks matplotlib, PyBullet, Pinocchio or simulation_and_control, launch
with the RoboEnv interpreter you use for the simulation instead. On this machine:

```bash
/home/nick/projects/RoboEnv/.pixi/envs/default/bin/python collect_expert.py --balls 1 --output datasets/one_ball --resume
```

The output path must be one shell argument. If a previous run saved no NPZ
episodes, `--resume` can refresh its source/configuration fingerprint, preserving
the old configuration as `dataset.previous.json`. Existing episode data still
requires an exact configuration/source match. Worker failures print their log
path and the end of the traceback as well as recording them in the manifest.
