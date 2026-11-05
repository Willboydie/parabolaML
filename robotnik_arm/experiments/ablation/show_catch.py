"""Replay a learned catch in the PyBullet GUI and write stills at launch, mid-flight and catch."""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np

from catch_env import (ASSETS_DIR, CATCH_LINGER, CAP_AFTER_ENDPOINT, FIRST_TICK_STEP,
                       CatchEnv, rollout)
from np_policy import NumpyPolicy, PolicyDriver

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / 'results' / 'models' / 'r25_p2_e0.5_hnone_d1000_s2.npz'
# Closest 25 Hz catch on the 200 test seeds: 2.53 mm at 15.42 s.
DEFAULT_SEED = 843567668329390999


def still(client, path, look_at, distance=3.2, yaw=50, pitch=-28):
    import pybullet as p
    view = client.computeViewMatrixFromYawPitchRoll(look_at, distance, yaw, pitch, 0, 2)
    proj = client.computeProjectionMatrixFOV(55, 1.4, 0.05, 20)
    w, h = 980, 700
    _, _, rgb, _, _ = client.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER)
    img = np.reshape(rgb, (h, w, 4))[:, :, :3]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imsave(path, img)


def decorate(env):
    target = env.target
    peak = -target.velocity[2] / target.acceleration[2]
    high = float(target.position(peak)[2])
    look = [float(env.init_base_pos[0]), float(env.init_base_pos[1]), high / 2]
    env.client.resetDebugVisualizerCamera(max(10.0, 2.0 * high + 3.0), 45, -40, look)
    env.client.configureDebugVisualizer(env.client.COV_ENABLE_GUI, 0)
    target.draw(env.client, draw_box=True, path_color=[1.0, 0.65, 0.0])
    body = target.spawn(env.client, color=(1.0, 0.15, 0.0, 1.0))
    return body, look


def play(env, driver, seed, rate, body, look, shots, speed, linger):
    env.reset(seed)
    if body is None:
        body, look = decorate(env)
    else:
        env.target.draw(env.client, draw_box=True, path_color=[1.0, 0.65, 0.0])
    driver.reset(env)
    stride = int(round(1.0 / (rate * env.dt)))
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    catch_time, step = None, 0
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    taken = set()
    while True:
        t = step * env.dt
        env.client.resetBasePositionAndOrientation(body, env.ball(t).tolist(), [0, 0, 0, 1])
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            cmd, _ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        dist = env.capture_distance(t)
        if dist is not None:
            env.min_dist = min(env.min_dist, dist)
            if catch_time is None and dist < 0.01:
                catch_time = t
                env.client.addUserDebugText(f'CATCH  {dist * 1000:.1f} mm  t={t:.2f}s',
                                            (env.ee + np.array([0, 0, 0.25])).tolist(), [0, 1, 0], 1.4)
                env.client.addUserDebugPoints([env.ee.tolist()], [[0, 1, 0]], pointSize=12)
        focus = ((env.ee + env.ball(t)) / 2).tolist()
        for name, when in shots:
            if name not in taken and t + 1e-9 >= when:
                dist = 2.2 if name == 'catch' else 3.4
                still(env.client, ROOT / 'results' / f'catch_{name}.png', focus, distance=dist)
                taken.add(name)
        if (catch_time is not None and t >= catch_time + linger) or t >= end_time:
            if 'catch' not in taken:
                still(env.client, ROOT / 'results' / 'catch_catch.png',
                      ((env.ee + env.ball(t)) / 2).tolist(), distance=2.2)
            break
        env.step(applied[:4], applied[4:])
        if speed > 0:
            time.sleep(env.dt / speed)
        step += 1
    return dict(success=catch_time is not None, catch_time=catch_time, min_dist=env.min_dist, look=look, body=body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--speed', type=float, default=2.5, help='Realtime multiplier; 0 is as fast as possible')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    model = NumpyPolicy(args.weights)
    env = CatchEnv(use_gui=not args.headless)
    driver = PolicyDriver(model)
    shots = [('launch', 0.05), ('mid', 8.0), ('catch', 15.4)]
    out = play(env, driver, args.seed, model.cfg.rate_hz, None, None, shots, 0 if args.headless else args.speed,
               linger=2.5 if not args.headless else CATCH_LINGER)
    print(f"{args.weights.name} seed {args.seed}: success={out['success']} "
          f"t={out['catch_time']} min={out['min_dist'] * 1000:.2f} mm")
    if not args.headless:
        time.sleep(8)


if __name__ == '__main__':
    main()
