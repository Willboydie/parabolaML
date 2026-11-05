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

from ballset import ball_features, robot_features
from config import policy_frames
from policy import ChunkPolicy, Config, DeepSetsPolicy, gaussian_mixture_nll

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


def apply_oracle_flag(balls, chosen, oracle):
    """Set the oracle channel on ``chosen``. ``balls`` is (B, N, 22) or (B, F, N, 22)."""
    if not oracle:
        return balls
    balls = balls.clone()
    balls[..., 21] = 0
    for slot in range(balls.shape[-2]):
        take = chosen == slot
        if balls.ndim == 3:
            balls[take, slot, 21] = 1
        else:
            balls[take, :, slot, 21] = 1
    return balls


def deepsets_forward(model, frames, chosen):
    """frames (B, F, N, 42), chosen (B,). Returns the normalized action chunk."""
    balls = apply_oracle_flag(ball_features(frames), chosen, model.cfg.oracle)
    balls = balls.permute(0, 2, 1, 3).contiguous()
    robot = robot_features(frames[:, :, 0])
    return model.forward_set(robot, balls)


def mixture_inputs(model, frames, chosen, two_ball):
    """Ball and robot tensors for the mixture head. Oracle flag stays 0 unless configured."""
    if two_ball:
        balls = apply_oracle_flag(ball_features(frames), chosen, model.cfg.oracle)
        balls = balls.permute(0, 2, 1, 3).contiguous()
        robot = robot_features(frames[:, :, 0])
    else:
        robot = robot_features(frames)
        balls = ball_features(frames).unsqueeze(1)
    return robot, balls


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
        obs_np = z['obs']
        two_ball = obs_np.ndim == 3
        obs = torch.from_numpy(np.ascontiguousarray(obs_np)).to(dev)
        cmd = torch.from_numpy(np.ascontiguousarray(z['cmd'])).to(dev)
        demo = torch.from_numpy(z['demo']).to(dev)
        starts, lengths = z['starts'], z['lengths']
        q_col = int(z['arm_q_col'])
        chosen = None
        if two_ball:
            if 'chosen' not in z.files:
                raise SystemExit('two-ball grid has no chosen-ball column')
            chosen = torch.from_numpy(np.ascontiguousarray(z['chosen'])).to(dev).long()
    if two_ball and cfg.encoder != 'deepsets':
        raise SystemExit('two-ball data needs the deepsets encoder')
    assert cfg.demos <= len(starts), f'only {len(starts)} demos available'
    n_rows = int(starts[cfg.demos - 1] + lengths[cfg.demos - 1])
    obs, cmd, demo = obs[:n_rows], cmd[:n_rows], demo[:n_rows]
    if chosen is not None:
        chosen = chosen[:n_rows]
        if int(chosen.shape[0]) != n_rows:
            raise SystemExit('chosen-ball column does not match the observation rows')
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
        q_now = obs[idx, 0, q_col:q_col + 7] if two_ball else obs[idx, q_col:q_col + 7]
        y[:, :, 4:11] -= q_now[:, None]
        return x, y, (None if chosen is None else chosen[idx])

    deepsets = cfg.encoder == 'deepsets'
    model = (DeepSetsPolicy(cfg) if deepsets else ChunkPolicy(cfg)).to(dev)
    with torch.no_grad():
        probe = torch.randint(0, n_rows, (min(n_rows, 50000),), device=dev)
        x, y, _ch = batch(probe)
        if deepsets:
            # One mean and std per ball channel, pooled over every ball and frame.
            # The one-ball grid has a single ball, so the pool is over rows.
            # Two-ball stats pool both slots, after the oracle flag is written.
            if two_ball:
                balls = apply_oracle_flag(ball_features(obs), chosen, cfg.oracle)
                balls = balls.reshape(-1, balls.shape[-1])
                robot = robot_features(obs[:, 0])
            else:
                balls = ball_features(obs)
                robot = robot_features(obs)
            model.ball_mean.copy_(balls.mean(0))
            model.ball_std.copy_(balls.std(0).clamp_min(1e-3))
            model.robot_mean.copy_(robot.mean(0))
            model.robot_std.copy_(robot.std(0).clamp_min(1e-3))
        else:
            feat = policy_frames(obs, cfg)
            model.obs_mean.copy_(feat.mean(0))
            model.obs_std.copy_(feat.std(0).clamp_min(1e-3))
        model.act_mean.copy_(y.mean(0))
        model.act_std.copy_(y.std(0).clamp_min(1e-3))
    # Fixed rows for the diagnostic loss. They stay in the training set; the loss
    # is recorded and never used to pick a checkpoint.
    probe_idx = torch.randperm(n_rows, generator=torch.Generator().manual_seed(12345))[:4096].to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warmup) * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * min(1.0, k / args.steps)))))
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    losses, diags = [], []
    t_data = time.time() - t0

    mdn = cfg.head == 'mdn'

    def run_model(frames, chosen_rows):
        if two_ball:
            return deepsets_forward(model, frames, chosen_rows)
        return model(frames)

    def run_mixture(frames, chosen_rows):
        robot, balls = mixture_inputs(model, frames, chosen_rows, two_ball)
        return model.mixture(robot, balls)

    for step in range(args.steps):
        idx = torch.randint(0, n_rows, (args.batch,), device=dev, generator=gen)
        x, y, ch = batch(idx)
        target = (y - model.act_mean) / model.act_std
        if mdn:
            log_pi, mu, sigma = run_mixture(x, ch)
            loss = gaussian_mixture_nll(log_pi, mu, sigma, target)
        else:
            pred = run_model(x, ch)
            loss = torch.mean((pred - target) ** 2)
        if not torch.isfinite(loss):
            raise SystemExit(f'loss became non-finite at step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        record = step % 100 == 0 or step == args.steps - 1
        if deepsets and record:
            grad_sq = torch.zeros((), device=dev)
            weight_sq = torch.zeros((), device=dev)
            for p in model.parameters():
                weight_sq = weight_sq + p.detach().float().pow(2).sum()
                if p.grad is not None:
                    grad_sq = grad_sq + p.grad.detach().float().pow(2).sum()
            diags.append(dict(step=step, loss=loss.item(), grad_norm=float(grad_sq.sqrt()),
                              weight_norm=float(weight_sq.sqrt())))
        if mdn:
            preclip = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if deepsets and record:
                diags[-1]['grad_norm_preclip'] = float(preclip)
        opt.step()
        sched.step()
        if deepsets and record:
            diags[-1]['lr'] = float(opt.param_groups[0]['lr'])
        if step % 500 == 0 or step == args.steps - 1:
            losses.append((step, loss.item()))
            if deepsets:
                with torch.no_grad():
                    px, py, pch = batch(probe_idx)
                    if mdn:
                        log_pi, mu, sigma = run_mixture(px, pch)
                        norm_y = (py - model.act_mean) / model.act_std
                        diags[-1]['probe_loss'] = float(gaussian_mixture_nll(log_pi, mu, sigma, norm_y))
                        pi = log_pi.exp()
                        diags[-1]['max_pi'] = float(pi.max(dim=-1).values.mean())
                        diags[-1]['pi_entropy'] = float(-(pi * log_pi).sum(dim=-1).mean())
                        picked = pi.argmax(dim=-1)
                        diags[-1]['component_frac'] = [float((picked == k).float().mean()) for k in range(pi.shape[-1])]
                        mode = mu[torch.arange(mu.shape[0], device=dev), picked]
                        err = (mode * model.act_std + model.act_mean) - py
                        diags[-1]['mode_mse'] = float(err.float().pow(2).mean())
                        print(f"step {step} nll {loss.item():.1f} max_pi {diags[-1]['max_pi']:.3f} "
                              f"entropy {diags[-1]['pi_entropy']:.3f}", flush=True)
                    else:
                        diags[-1]['probe_loss'] = float(torch.mean(
                            (run_model(px, pch) - (py - model.act_mean) / model.act_std) ** 2))
    train_time = time.time() - t0
    probe_error = None
    if deepsets and not mdn:
        with torch.no_grad():
            px, py, pch = batch(probe_idx)
            err = (run_model(px, pch) * model.act_std + model.act_mean) - py
            sq = err.float().pow(2)
            per_step = sq.mean(dim=(0, 2))
            probe_error = dict(
                wheels=float(sq[..., 0:4].mean()),
                arm_q=float(sq[..., 4:11].mean()),
                arm_qd=float(sq[..., 11:18].mean()),
                by_chunk_step=[float(v) for v in per_step],
                note='squared error in raw command units on 4096 training rows; not a held-out split',
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(config=cfg.to_dict(), seed=args.seed, steps=args.steps, batch=args.batch, lr=args.lr,
                train_rows=n_rows, train_time_s=train_time, load_time_s=t_data, final_loss=losses[-1][1],
                losses=losses, params=sum(p.numel() for p in model.parameters()))
    if deepsets:
        meta['diagnostics'] = diags
        meta['probe_error'] = probe_error
    torch.save(dict(config=cfg.to_dict(), state_dict={k: v.cpu() for k, v in model.state_dict().items()}, meta=meta),
               args.out)
    print(json.dumps({k: meta[k] for k in ('train_time_s', 'final_loss', 'train_rows', 'params')}))


if __name__ == '__main__':
    main()
