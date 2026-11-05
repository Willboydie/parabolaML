"""Single-ball interception environment with separate sim, policy and low-level rates.

The physics runs at the simulator's fixed 1 ms step. A driver (the scripted expert or
a learned policy) emits a command at the policy rate: four wheel velocities, seven arm
joint references and seven joint reference velocities. The low-level controller runs
every simulation step, holding the latest command: wheel velocity servo and a
gravity-compensated joint PD law, identical to the expert's own arm law.

``LeanStep`` reproduces ``SimInterface.Step`` for this robot without the per-step mass
matrix: the configuration has zero motor damping and elasticity, so the matrix only
ever multiplies a zero vector.
"""
import os
import sys
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = TASK_DIR.parents[1] / 'RoboEnv'
sys.path.insert(0, str(TASK_DIR))

OBS_DIM = 42
CMD_DIM = 18
OBS_NAMES = (['time'] + [f'ball_pos_{a}' for a in 'xyz'] + [f'ball_vel_{a}' for a in 'xyz']
             + [f'ball_acc_{a}' for a in 'xyz'] + [f'base_pos_{a}' for a in 'xyz'] + ['bearing_sin', 'bearing_cos']
             + [f'base_linvel_{a}' for a in 'xyz'] + [f'base_angvel_{a}' for a in 'xyz']
             + [f'arm_q_{i}' for i in range(7)] + [f'arm_qd_{i}' for i in range(7)]
             + [f'wheel_qd_{i}' for i in range(4)] + [f'ee_{a}' for a in 'xyz'])
assert len(OBS_NAMES) == OBS_DIM

WHEEL_KD = 30.0
FIXED_FINGER_ANGLES = np.full(6, 0.1)
ARM_REACH_LIMIT = 0.85
BASE_NAV_TIMEOUT = 240.0
CAP_AFTER_ENDPOINT = 1.0
# Keep simulating briefly after the first catch so min_dist is the true closest approach.
CATCH_LINGER = 0.1
FIRST_TICK_STEP = 2


class CatchEnv:
    """One headless simulator, restored to the same initial state for every episode."""

    def __init__(self, assets_dir=ASSETS_DIR, use_gui=False):
        os.environ['ROBOENV_HEADLESS'] = '0' if use_gui else '1'
        import io
        import contextlib
        import pinocchio as pin
        from simulation_and_control import pb, PinWrapper
        from robotnik_arm_control import (ArmControllerParams, BaseControllerParams,
                                          configure_gripper_fingers, quaternion2bearing)
        self.pin = pin
        self.quaternion2bearing = quaternion2bearing
        conf = 'robotnik_arm.json'
        with contextlib.redirect_stdout(io.StringIO()):
            sim = pb.SimInterface(conf, conf_file_path_ext=str(assets_dir), use_gui=use_gui)
        sim.SetFloorFriction(1.0)
        client = sim.pybullet_client
        rid = sim.bot[0].bot_pybullet
        for j in range(client.getNumJoints(rid)):
            if 'wheel' in client.getJointInfo(rid, j)[1].decode():
                client.changeDynamics(rid, j, contactStiffness=30000, contactDamping=1000,
                                      anisotropicFriction=[1.0, 0.01, 1.0])
        sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
        configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)
        names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.dyn = PinWrapper(conf, 'pybullet', names, ['pybullet'], False, 0, str(assets_dir))
        self.sim, self.client, self.rid = sim, client, rid
        self.joint_ids = list(sim.bot[0].active_joint_ids)
        self.ee_frame = self.dyn.pin_model.getFrameId('robot_j2s7s300_end_effector')
        self.shoulder_frame = self.dyn.pin_model.getFrameId('robot_j2s7s300_link_2')
        self.dt = sim.GetTimeStep()
        lower, upper = sim.GetBotJointsLimit()
        self.arm_lower, self.arm_upper = np.array(lower[4:11]), np.array(upper[4:11])
        self.arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
        self.base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
        self.torque_mode = client.TORQUE_CONTROL
        self.position_mode = client.POSITION_CONTROL
        self._read_state()
        self.init_base_pos = self.base_pos.copy()
        self.init_bearing = self.bearing
        self.init_ee = self.ee.copy()
        self.q_init_arm = self.q[4:11].copy()
        self.saved_state = client.saveState()
        center = self.init_base_pos
        self.lower = np.array([center[0] - 2.5, center[1] - 2.5, 0.07])
        self.upper = np.array([center[0] + 2.5, center[1] + 2.5, 1.30])

    def reset(self, seed):
        from robotnik_arm_control import make_catchable_pair
        self.client.restoreState(self.saved_state)
        self._read_state()
        targets, _selected, _plans = make_catchable_pair(
            seed, None, self.lower, self.upper, self.init_base_pos, self.init_bearing, self.init_ee,
            self.dt, self.base_params, self.arm_params, target_count=1)
        self.target = targets[0]
        self.has_left = False
        self.min_dist = np.inf

    def _read_state(self):
        c, rid = self.client, self.rid
        states = c.getJointStates(rid, self.joint_ids)
        self.q = np.array([s[0] for s in states])
        self.qd = np.array([s[1] for s in states])
        pos, ori = c.getBasePositionAndOrientation(rid)
        lin, ang = c.getBaseVelocity(rid)
        self.base_pos, self.base_ori = np.array(pos), np.array(ori)
        self.base_vel = np.array(lin + ang)
        self.bearing = self.quaternion2bearing(ori[3], ori[0], ori[1], ori[2])
        q_all = np.concatenate((self.base_pos, self.base_ori, self.q))
        self.q_pin = self.dyn.ReoderJoints2PinVec(q_all, 'pos')
        pin, model, data = self.pin, self.dyn.pin_model, self.dyn.pin_data
        pin.forwardKinematics(model, data, self.q_pin)
        pin.updateFramePlacements(model, data)
        self.ee = data.oMf[self.ee_frame].translation.copy()
        self.gravity_arm = pin.computeGeneralizedGravity(model, data, self.q_pin)[6:13]

    def step(self, wheel_cmd, arm_torque):
        wheel_torque = -WHEEL_KD * (self.qd[:4] - wheel_cmd)
        c = self.client
        c.setJointMotorControlArray(self.rid, self.joint_ids[:11], self.torque_mode,
                                    forces=np.concatenate((wheel_torque, arm_torque)).tolist())
        c.setJointMotorControlArray(self.rid, self.joint_ids[11:17], self.position_mode,
                                    targetPositions=FIXED_FINGER_ANGLES.tolist(), forces=[5.0] * 6)
        c.stepSimulation()
        self._read_state()

    def ball(self, t):
        return self.target.position(t)

    def ball_estimate(self, t):
        """Position, velocity and acceleration from the last three 1 ms samples, as the expert sees them."""
        p0, p1, p2 = (self.target.position(max(0.0, t - k * self.dt)) for k in (2, 1, 0))
        v0, v1 = (p1 - p0) / self.dt, (p2 - p1) / self.dt
        acc = (v1 - v0) / self.dt
        return p2, v1 + 0.5 * acc * self.dt, acc

    def obs(self, t):
        pos, vel, acc = self.ball_estimate(t)
        return np.concatenate(([t], pos, vel, acc, self.base_pos, [np.sin(self.bearing), np.cos(self.bearing)],
                               self.base_vel, self.q[4:11], self.qd[4:11], self.qd[:4], self.ee)).astype(np.float32)

    def arm_torque(self, q_des, qd_des):
        p = self.arm_params
        tau = p.kp * (q_des - self.q[4:11]) + p.kd * (qd_des - self.qd[4:11]) + self.gravity_arm
        return np.clip(tau, -p.max_torque, p.max_torque)

    def clip_reference(self, q_des):
        """Low-level safety limits on any arm reference: lead from the measured angle, then joint limits."""
        p = self.arm_params
        q_arm = self.q[4:11]
        q_des = np.clip(q_des, q_arm - p.max_q_lead, q_arm + p.max_q_lead)
        has_limits = self.arm_lower < self.arm_upper
        return np.where(has_limits, np.clip(q_des, self.arm_lower + p.limit_margin, self.arm_upper - p.limit_margin), q_des)

    def capture_distance(self, t):
        """EE-to-ball distance when a catch would count, else None (ball must have left and re-entered the box)."""
        ball = self.ball(t)
        outside = np.any(ball < self.lower - 1e-6) or np.any(ball > self.upper + 1e-6)
        self.has_left |= bool(outside)
        if not self.has_left or outside:
            return None
        return float(np.linalg.norm(ball - self.ee))


class ExpertDriver:
    """The scripted interception expert from robotnik_arm_task.main, acting only at policy ticks.

    It senses the ball every simulation step (as the policy's 1 ms velocity estimate does)
    but plans and updates commands only at ticks; between ticks the low level holds them.
    """

    def __init__(self):
        from robotnik_arm_control import InterceptionExpert, BaseNavState, base_controller, arm_controller
        self.InterceptionExpert, self.BaseNavState = InterceptionExpert, BaseNavState
        self.base_controller, self.arm_controller = base_controller, arm_controller

    def reset(self, env):
        self.expert = self.InterceptionExpert(env.lower, env.upper)
        self.base_state = self.BaseNavState()
        self.goal_pos, self.goal_bearing = env.init_base_pos[:2].copy(), 0.0
        self.q_des, self.qd_des = env.q_init_arm.copy(), np.zeros(7)
        self.planning_attempted = False
        self.ready = False

    def sense(self, env, t):
        self.ready = self.expert.observe(t, env.ball(t))

    def act(self, env, t, dt_ctrl):
        ex = self.expert
        ball = env.ball(t)
        ready = self.ready
        missed = ex.intercept_time is not None and t > ex.intercept_time + 1.0 and np.linalg.norm(ball - ex.intercept) > 0.01
        if ready and (not self.planning_attempted or missed):
            self.planning_attempted = True
            self.base_state = self.BaseNavState()
            if ex.plan(t, ball, env.base_pos, env.bearing, env.ee, env.base_params, env.arm_params):
                self.goal_pos, self.goal_bearing = ex.goal_pos, ex.goal_bearing
        ready = ready and ex.intercept is not None
        wheel = self.base_controller(self.base_state, env.base_pos, env.bearing, self.goal_pos, self.goal_bearing,
                                     t, env.base_params) if ready else np.zeros(4)
        _tau, self.q_des, self.qd_des = self.arm_controller(
            env.dyn, env.q_pin, env.q[4:11], env.qd[4:11], self.q_des, env.q_init_arm,
            ex.intercept if ready else ball, env.ee, env.ee_frame, env.shoulder_frame,
            env.arm_lower, env.arm_upper, dt_ctrl, env.arm_params, reach_active=ready and self.base_state.done)
        return np.concatenate((wheel, self.q_des, self.qd_des)), ready


class DartNoise:
    """Bounded Gaussian wheel/torque perturbation held for 50 ms, as in collect_expert.py."""

    def __init__(self, seed, wheel_std=0.08, arm_std=0.2, hold=0.05):
        self.rng = np.random.default_rng(seed)
        self.std = np.r_[np.full(4, wheel_std), np.full(7, arm_std)]
        self.hold, self.until, self.value = hold, -np.inf, np.zeros(11)

    def __call__(self, t):
        if t >= self.until:
            self.value = np.clip(self.rng.normal(size=11), -3, 3) * self.std
            self.until = t + self.hold
        return self.value


def rollout(env, driver, seed, rate_hz, radius=0.01, stop_on_catch=True, noise=None, log=False,
            until=None):
    """Run one episode. The driver acts every 1/rate_hz seconds from the third step on.

    Returns a dict with success, min capture-window distance and catch time; with
    ``log`` also 1 kHz arrays of observations, commands and applied actions.
    """
    env.reset(seed)
    driver.reset(env)
    stride = int(round(1.0 / (rate_hz * env.dt)))
    assert abs(stride * env.dt * rate_hz - 1.0) < 1e-9, 'policy period must be a whole number of sim steps'
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    ready = False
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    wheel_limit = (env.base_params.max_lin_vel + env.base_params.max_ang_vel * env.base_params.wheel_base_width / 2) / env.base_params.wheel_radius
    rows = dict(obs=[], cmd=[], applied=[], ready=[]) if log else None
    catch_time = None
    step = 0
    while True:
        t = step * env.dt
        if hasattr(driver, 'sense'):
            driver.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            cmd, ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        if noise is not None and ready:
            applied = applied + noise(t)
            applied[:4] = np.clip(applied[:4], -wheel_limit, wheel_limit)
            applied[4:] = np.clip(applied[4:], -env.arm_params.max_torque, env.arm_params.max_torque)
        dist = env.capture_distance(t)
        if dist is not None:
            env.min_dist = min(env.min_dist, dist)
            if catch_time is None and dist < radius:
                catch_time = t
        if log:
            rows['obs'].append(env.obs(t))
            rows['cmd'].append(cmd.astype(np.float32))
            rows['applied'].append(applied.astype(np.float32))
            rows['ready'].append(ready)
        if until is not None and t + 1e-12 >= until:
            break
        if (catch_time is not None and stop_on_catch and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        env.step(applied[:4], applied[4:])
        step += 1
    result = dict(seed=int(seed), success=catch_time is not None, min_dist=float(env.min_dist),
                  catch_time=catch_time, steps=step + 1, duration=float(env.target.duration))
    if log:
        result['arrays'] = {k: np.asarray(v) for k, v in rows.items()}
    return result
