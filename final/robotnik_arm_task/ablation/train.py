"""Train one config with one seed on the GPU: MLP trunk, plain MSE on normalized action chunks.

Every config gets the same number of gradient steps and batch size; there is no early
stopping and no model selection on validation loss.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import torch

from policy import ChunkPolicy, Config

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='JSON Config fields')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data', type=Path, default=DATA / 'grid50.npz')
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = Config(**json.loads(args.config))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()

    with np.load(args.data) as z:
        obs = torch.from_numpy(z['obs']).to(dev)
        cmd = torch.from_numpy(z['cmd']).to(dev)
        demo = torch.from_numpy(z['demo']).to(dev)
        starts, lengths = z['starts'], z['lengths']
        q_col = int(z['arm_q_col'])
    assert cfg.demos <= len(starts), f'only {len(starts)} demos available'
    n_rows = int(starts[cfg.demos - 1] + lengths[cfg.demos - 1])
    obs, cmd, demo = obs[:n_rows], cmd[:n_rows], demo[:n_rows]
    ep_start = torch.from_numpy(starts).to(dev)[demo]
    ep_last = torch.from_numpy(starts + lengths - 1).to(dev)[demo]
    s, H, F = cfg.stride, cfg.horizon, cfg.frames
    hist_off = torch.arange(F - 1, -1, -1, device=dev) * s
    chunk_off = torch.arange(H, device=dev) * s

    def batch(idx):
        h = torch.maximum(idx[:, None] - hist_off, ep_start[idx][:, None])
        c = torch.minimum(idx[:, None] + chunk_off, ep_last[idx][:, None])
        x = obs[h]
        y = cmd[c].clone()
        y[:, :, 4:11] -= obs[idx, q_col:q_col + 7][:, None]
        return x, y

    model = ChunkPolicy(cfg).to(dev)
    with torch.no_grad():
        probe = torch.randint(0, n_rows, (min(n_rows, 50000),), device=dev)
        x, y = batch(probe)
        feat = obs[:, 1:] if cfg.drop_time else obs
        model.obs_mean.copy_(feat.mean(0))
        model.obs_std.copy_(feat.std(0).clamp_min(1e-3))
        model.act_mean.copy_(y.mean(0))
        model.act_std.copy_(y.std(0).clamp_min(1e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warmup) * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * min(1.0, k / args.steps)))))
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    losses = []
    t_data = time.time() - t0
    for step in range(args.steps):
        idx = torch.randint(0, n_rows, (args.batch,), device=dev, generator=gen)
        x, y = batch(idx)
        loss = torch.mean((model(x) - (y - model.act_mean) / model.act_std) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            losses.append((step, loss.item()))
    train_time = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(config=cfg.to_dict(), seed=args.seed, steps=args.steps, batch=args.batch, lr=args.lr,
                train_rows=n_rows, train_time_s=train_time, load_time_s=t_data, final_loss=losses[-1][1],
                losses=losses, params=sum(p.numel() for p in model.parameters()))
    torch.save(dict(config=cfg.to_dict(), state_dict={k: v.cpu() for k, v in model.state_dict().items()}, meta=meta),
               args.out)
    print(json.dumps({k: meta[k] for k in ('train_time_s', 'final_loss', 'train_rows', 'params')}))


if __name__ == '__main__':
    main()
