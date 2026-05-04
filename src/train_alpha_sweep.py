#!/usr/bin/env python3
"""
T15: Alpha-sweep training script.
Trains one MAPPO model per alpha in [0.0, 0.1, ..., 1.0].
Each model is saved to models/alpha_X.XX/ with its own log.

Usage:
    python train_alpha_sweep.py [--epochs 500] [--horizon 64] [--device cpu]

Design decisions:
    - fixed_alpha locks env.alpha throughout training (curriculum cannot override).
    - Each model gets its own save_dir so checkpoints never collide.
    - Alpha IS in the observation (OBS_DIM=13), so each model learns alpha-conditioned policy.
    - This is preferred over a single model because training distribution matches eval alpha.
    - Parallelism: runs are sequential by default; set PARALLEL=True for multiprocessing.
"""

import os
import argparse
import numpy as np

import params
from mappo_agent import train as mappo_train


def train_single_alpha(alpha, epochs, horizon, device, base_dir):
    """Train one MAPPO model with fixed alpha. Returns save_dir."""
    alpha_tag = f'alpha_{alpha:.2f}'
    save_dir  = os.path.join(base_dir, alpha_tag)
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Training alpha={alpha:.2f}  ->  {save_dir}')
    print(f'{"="*60}')

    mappo_train(
        env_device=device,
        epochs=epochs,
        horizon=horizon,
        fixed_alpha=alpha,
        save_dir=save_dir,
        save_every=max(1, epochs // 5),  # 5 intermediate checkpoints
    )
    return save_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',  type=int,   default=500)
    parser.add_argument('--horizon', type=int,   default=64)
    parser.add_argument('--device',  type=str,   default='cpu')
    parser.add_argument('--alpha_min', type=float, default=0.0)
    parser.add_argument('--alpha_max', type=float, default=1.0)
    parser.add_argument('--alpha_step', type=float, default=0.1)
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(base_dir, exist_ok=True)

    alpha_values = np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2,
                             args.alpha_step).tolist()
    # Round to avoid floating-point drift (e.g., 0.30000000004)
    alpha_values = [round(a, 2) for a in alpha_values]

    print(f'Alpha sweep: {alpha_values}')
    print(f'Epochs per model: {args.epochs}  Horizon: {args.horizon}')
    print(f'Total models: {len(alpha_values)}')
    print(f'Estimated total env steps: {len(alpha_values) * args.epochs * args.horizon:,}')

    completed = []
    for alpha in alpha_values:
        try:
            save_dir = train_single_alpha(alpha, args.epochs, args.horizon,
                                          args.device, base_dir)
            completed.append((alpha, save_dir))
            print(f'[DONE] alpha={alpha:.2f} saved to {save_dir}')
        except Exception as e:
            print(f'[FAIL] alpha={alpha:.2f}: {e}')
            import traceback; traceback.print_exc()

    print(f'\n{"="*60}')
    print(f'Alpha sweep complete. {len(completed)}/{len(alpha_values)} models trained.')
    for a, d in completed:
        print(f'  alpha={a:.2f}  ->  {d}/mappo_final.pth')


if __name__ == '__main__':
    main()