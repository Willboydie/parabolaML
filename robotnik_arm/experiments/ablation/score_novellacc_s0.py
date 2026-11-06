"""Score the no-velocity one-ball backbone on the 200 held-out seeds."""
import json
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

import torch

from parking_split import eval_weights, load_experts
from policy import export_numpy


def main():
    name = 'r50_p1_e0.5_hstack0.24s_d2000_deepsets_novellacc_s0'
    ckpt = Path('results/models') / f'{name}.pt'
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    weight = blob['state_dict']['phi.0.weight']
    if blob['config'].get('ball_vel_acc') is not False or tuple(weight.shape) != (256, 130):
        raise SystemExit(
            f'checkpoint is not the position-only ball encoder: {weight.shape} '
            f'{blob["config"].get("ball_vel_acc")}')

    weights = ckpt.with_suffix('.npz')
    export_numpy(ckpt, weights)
    test = json.loads(Path('data/seeds.json').read_text())['test'][:200]
    print(f'expert reference on {len(test)} test seeds', flush=True)
    experts = load_experts(test)
    print(f'rollout {name}', flush=True)
    stats = eval_weights(weights, test, experts)
    split = stats['split']
    print(
        f"catch10 {stats['catch_rate']:.3f}  catch50 {stats['catch_rate_50mm']:.3f}  "
        f"miss {stats['miss_mm']:.1f} mm  median {stats['median_miss_mm']:.1f} mm  "
        f"along {split['base_along_median_mm']:+.1f}  ahead {split['frac_ahead']:.3f}",
        flush=True,
    )
    out = Path('results/stage2/rate50_novellacc.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        name=name,
        dropped='ball world velocity, world acceleration, base-frame velocity, base-frame acceleration',
        kept_ball='world position, base-frame position, ball minus end effector, oracle flag',
        ball_dim=10,
        frames=13,
        phi_in=130,
        params=blob['meta']['params'],
        train_rows=blob['meta']['train_rows'],
        final_loss=blob['meta']['final_loss'],
        train_time_s=blob['meta']['train_time_s'],
        test=stats,
    )
    out.write_text(json.dumps(report, indent=2))
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
