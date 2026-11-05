# parabolaML

Course labs and the final project: a wheeled manipulator that intercepts ballistic targets.

## Layout

| Path | What it is |
|---|---|
| `labs/` | Weekly exercises. Each script loads `configs/` and `models/` from its own directory. |
| `robotnik_arm/` | Final project: simulation, expert collection, tests, and interception experiments. |
| `RoboEnv/` | Local checkout of the simulator. It has its own git history and pixi environment, and it is not part of this repository. |

```
parabolaML/
├── labs/
│   ├── week1/          simulation, Cartesian control, dynamic regression
│   ├── week2/          nonlinear dynamic regression
│   ├── week3/          Bayesian optimisation of PD gains
│   └── week4/          MLP on MNIST
└── robotnik_arm/
    ├── robotnik_arm_task.py
    ├── collect_expert.py
    ├── docs/
    ├── tests/
    └── experiments/ablation/
```

Generated datasets, checkpoints, virtualenvs, and the `RoboEnv/` checkout stay on disk and out of git.

## Setup

Install [pixi](https://pixi.sh), then clone the simulator into `RoboEnv/` at the root of this repository (this machine already has that checkout):

```bash
git clone --recurse-submodules https://github.com/VModugno/RoboEnv.git RoboEnv
cd RoboEnv && pixi install
pixi run smoke-test
```

Week 4 also needs the `ml` environment: `pixi install -e ml` from `RoboEnv/`. Torch does not fit in the default environment; the reason is in `labs/week4/README.md`.

Lab scripts still expect `configs/` and `models/` beside the script. Each week's README has the copy or link commands. The arm task finds those assets in `RoboEnv/` when they are not beside the script.

## Labs

| Week | Start here |
|---|---|
| 1 | `labs/week1/README.md` |
| 2 | `labs/week2/dynamic_regression_nonlinear.py` |
| 3 | `labs/week3/README.md` |
| 4 | `labs/week4/README.md` |

Run a lab from the RoboEnv environment, for example:

```bash
cd RoboEnv && pixi shell
cd "<path-to-this-repo>/labs/week1"
python induction_simulator.py
```

## Final project

From `robotnik_arm/`, in the RoboEnv environment:

```bash
python robotnik_arm_task.py
python collect_expert.py --balls 2 --output datasets/two_balls
python -m unittest discover -s tests -t .
```

`ROBOENV_TARGET_SEED`, `ROBOENV_HEADLESS`, and the other run settings are documented in `robotnik_arm/README.md`. Data collection is documented in `robotnik_arm/docs/DATA_COLLECTION.md`.

The interception experiments live in `robotnik_arm/experiments/ablation/`. Run those scripts from that directory. Progress and the Stage 1 config are in `STATUS.md` and `STAGE1_RESULTS.md` there.
