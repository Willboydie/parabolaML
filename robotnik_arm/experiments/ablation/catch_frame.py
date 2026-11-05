"""Spread of the expert catch point in the base frame, across the 1,000 demos."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
N = 1000


def summarize(name, pts):
    center = pts.mean(0)
    dist = np.linalg.norm(pts - center, axis=1)
    print(f'{name}: n={len(pts)}')
    print(f'  mean forward {center[0]*1000:+.0f} mm  left {center[1]*1000:+.0f} mm  up {center[2]*1000:+.0f} mm')
    print(f'  std  forward {pts[:,0].std()*1000:.0f} mm  left {pts[:,1].std()*1000:.0f} mm  up {pts[:,2].std()*1000:.0f} mm')
    print(f'  distance from the mean point: median {np.median(dist)*1000:.0f} mm  '
          f'p90 {np.percentile(dist,90)*1000:.0f} mm  max {dist.max()*1000:.0f} mm')
    print(f'  farther than 50 mm from the mean: {np.mean(dist > 0.05):.0%}')
    horiz = np.linalg.norm(pts[:, :2] - center[:2], axis=1)
    print(f'  horizontal only: median {np.median(horiz)*1000:.0f} mm  '
          f'p90 {np.percentile(horiz,90)*1000:.0f} mm')


def main():
    metas = []
    for i in range(N):
        with np.load(DATA / 'demos' / f'demo_{i:04d}.npz') as z:
            metas.append(json.loads(str(z['meta'])))
    with np.load(DATA / 'grid50.npz') as z:
        obs = z['obs']
        starts = z['starts']
        lengths = z['lengths']
    ball_pts, ee_pts, dart = [], [], []
    missed = 0
    for i, meta in enumerate(metas):
        if not meta.get('success') or meta.get('catch_time') is None:
            missed += 1
            continue
        sl = slice(int(starts[i]), int(starts[i] + lengths[i]))
        rows = obs[sl]
        k = int(np.argmin(np.abs(rows[:, 0] - meta['catch_time'])))
        row = rows[k]
        yaw = np.arctan2(row[13], row[14])
        forward = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-forward[1], forward[0]])
        base = row[10:13]
        def frame(point):
            d = point - base
            return np.array([np.dot(d[:2], forward), np.dot(d[:2], left), d[2]])
        ball_pts.append(frame(row[1:4]))
        ee_pts.append(frame(row[39:42]))
        dart.append(bool(meta.get('dart', i % 2)))
    ball_pts = np.asarray(ball_pts)
    ee_pts = np.asarray(ee_pts)
    dart = np.asarray(dart)
    hand = np.linalg.norm(ball_pts - ee_pts, axis=1)
    print(f'skipped {missed} without a catch; hand-to-ball at the logged catch: '
          f'median {np.median(hand)*1000:.2f} mm')
    summarize('all demos, ball', ball_pts)
    summarize('clean demos, ball', ball_pts[~dart])
    summarize('DART demos, ball', ball_pts[dart])
    summarize('all demos, end effector', ee_pts)


if __name__ == '__main__':
    main()
