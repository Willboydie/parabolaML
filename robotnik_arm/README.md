1) Make sure the folders "/models", "/configs" from RoboEnv are accessible (easiest is to copy them to this folder)
2) Use your RoboEnv environent (in whichever way you have been running the labs: pixi, uv, conda, etc)
3) Run robotnik_arm_task.py

Expected behaviour: pybullet 3D environment with 7dof arm mounted on a 4-wheeled platform. The robot moves to reach a target (red dot) with its end-effector. The simulation terminates when the target is reached, and plots of the robot behaviour are shown.

Key variables in robotnik_arm_task.py:

 - target_pos: current moving target position in world coordinates (x,y,z), shown as a red sphere.

 - base_pos:     wheeled base position (x,y,z)
 - base_bearing: wheeled base bearing (angle in radians)
 - q_mes:        arm joint angles (radians)
 - qd_mes:       arm joint angle velocities (rad/s)

 - wheel_cmds:     angular velocity command for each wheel (rad/s)
 - torque_arm:     command torques for each arm joint
 - torque_fingers: torque for each finger joint of the end-effector grasper. In this work, they should just remain fixed.
 
 Assessment Goal:
  - ML model that produces (wheel_cmds, torque_arm) from measurements (base_pos,base_bearing,q_mes,qd_mes)

Scripted parabolic targets
--------------------------
Two visual balls follow independent random parabolas. Each starts on an X/Y side
face, arcs above the cuboid, re-enters on descent, then holds at the opposite
side face. Their true peaks are random 0.5–1.5 m above the roof. The cuboid is
5 x 5 m, centred on the initial robot XY position, with height 0.07–1.30 m.

Launch speed is independently sampled from 0.50–0.70 m/s. Speed varies along the
parabola. Targets have no mass/collisions and are updated from their scripted
paths each timestep, so physical gravity does not affect them. The robot retains
its normal gravity and original controller speeds (base and arm 0.25 m/s).

Both trajectories are resampled until each has an estimated feasible catch on
the descending arc before its endpoint. The expert randomly selects one of
these two validated targets, then estimates its velocity and
acceleration from three positions across two timesteps, and searches for the
earliest estimated feasible descending in-box catch. It compares approach
angles and standoffs, preferring less base travel and turning at the chosen
time. It can catch at the endpoint if the moving interception is infeasible.
Success requires a prior departure, return inside the box and EE distance below
10 mm. The selected ball/path is orange-red/orange; the other is purple. The
solid grey floor, thick box edges and automatically framed camera are retained.

Run `python robotnik_arm_task.py` in the RoboEnv environment with `configs/`
and `models/` beside the script. Settings:
- `ROBOENV_TARGET_SEED`: optional reproducible pair of random trajectories.
- `ROBOENV_TARGET_SPEED`: optional fixed positive launch speed in m/s.
- `ROBOENV_HEADLESS=1`: disable GUI/plots.
- `ROBOENV_MAX_STEPS`: step cap, default 180000.

Example: `ROBOENV_TARGET_SEED=0 python robotnik_arm_task.py`.
Run checks with `python -m unittest discover -p 'test_*.py'` in this folder.
Changed lines in the two robotnik Python files are marked `#NEW`.

Catchability is checked against the current base pose, EE pose and controller
speed limits, with at least 0.2 s remaining before the endpoint. The 5 x 5 m
volume alone does not guarantee capture. This validates the planner timing
model; simulation slip, joint limits and tracking error can still cause a miss.
Selection is reproducible with `ROBOENV_TARGET_SEED`.

Data collection is available through `collect_expert.py`. The default plan is
200 noisy DART training episodes, 200 clean training episodes, 50 validation
and 50 test episodes. Use `--balls 1` or `--balls 2` and separate output folders.
See [DATA_COLLECTION.md](DATA_COLLECTION.md) for commands, noise settings,
resuming, outcome flags and the complete observation/action schema.
