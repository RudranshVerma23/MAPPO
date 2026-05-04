#!/usr/bin/env python3
"""
Power Sweep Evaluation Script — FIXED VERSION
Bug fixes mirror alpha_sweep_fixed.py:
  1. episode_td/mi averaged per step (not last-step overwrite)
  2. Reward normalized per-UAV per-step (not 32×NUM_UAVS inflated)
  3. Negative TD guard
  4. env.current_stage=2 for full action space at eval
"""

import sys
import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment_v2 import EnvironmentV2
from mappo_agent import Actor, total_td_mi
from dueling_dqn_multiagent import MultiAgentDuelingDQN
import params


# ─────────────────────────────────────────────
# Shared helpers (same as alpha_sweep_fixed)
# ─────────────────────────────────────────────

def _unwrap_obs(td_field):
    if torch.is_tensor(td_field):
        return td_field
    for key in ('observation', 'global_state'):
        try:
            v = td_field[key]
            if torch.is_tensor(v):
                return v
        except Exception:
            pass
    return td_field


def _load_mappo_actor(model_path):
    if not os.path.exists(model_path):
        print(f"  [WARN] MAPPO model not found: {model_path}")
        return None
    try:
        ckpt = torch.load(model_path, map_location='cpu')
        actor = Actor(params.OBS_DIM, params.ACTION_DIM)
        actor.load_state_dict(ckpt.get('actor', ckpt))
        actor.eval()
        return actor
    except Exception as e:
        print(f"  [WARN] MAPPO checkpoint incompatible: {e}")
        return None


def _load_dqn_agents(base_models):
    agent = MultiAgentDuelingDQN(device='cpu')
    ok = 0
    for u in range(params.NUM_UAVS):
        pth = os.path.join(base_models, f'dqn_multiagent_uav{u}_final.pth')
        if os.path.exists(pth):
            try:
                agent.agents[u].load_state_dict(torch.load(pth, map_location='cpu'))
                ok += 1
            except Exception as e:
                print(f"  [WARN] Could not load DQN UAV{u}: {e}")
    if ok != params.NUM_UAVS:
        print("  [WARN] DQN checkpoints incomplete/incompatible; DQN results may be invalid.")
    return agent


# ─────────────────────────────────────────────
# Single episode evaluation
# ─────────────────────────────────────────────

def evaluate_episode(algo_name, env, actor_or_agent, horizon=64):
    """
    Returns per-step-averaged metrics for one episode.
    All values are per-step averages — NOT sums.
    """
    td_env = env.reset()

    obs_raw = td_env['observation']
    if isinstance(obs_raw, dict):
        obs = obs_raw['observation']
    else:
        obs = _unwrap_obs(obs_raw)
    if not torch.is_tensor(obs):
        obs = torch.tensor(obs, dtype=torch.float32)

    step_tds, step_mis, step_rewards = [], [], []

    for _ in range(horizon):
        if algo_name == 'mappo':
            actor = actor_or_agent
            with torch.no_grad():
                a1, a0 = actor(obs)
                action_tensor = a1 / (a1 + a0)         # Beta mean

        else:
            dqn = actor_or_agent
            trajs, pwrs = [], []
            obs_np = obs.numpy()
            for u in range(params.NUM_UAVS):
                obs_u = torch.tensor(obs_np[u], dtype=torch.float32)
                a = dqn.select_action(obs_u, u)
                ti, pi = dqn.decode_action(a)
                trajs.append(dqn.traj_to_vel(ti))
                pwrs.append(dqn.power_to_alloc(pi))
            action_tensor = torch.cat([torch.stack(trajs), torch.stack(pwrs)], dim=1)

        out = env.step({'action': action_tensor})

        # per-UAV mean reward for THIS step
        reward = out['reward']
        step_reward = (
            reward.mean().item() if torch.is_tensor(reward)
            else float(np.mean(reward))
        )
        step_rewards.append(step_reward)

        # TD and MI for this step
        try:
            total_td_val, total_mi_val, _, _, _ = total_td_mi(
                out['rates'], out['mis']
            )
            td_val = max(0.0, float(total_td_val.item()))  # clamp negative
            mi_val = float(total_mi_val.item())
        except Exception:
            td_val, mi_val = 0.0, 0.0

        step_tds.append(td_val)
        step_mis.append(mi_val)

        # next observation
        next_raw = out['observation']
        if isinstance(next_raw, dict):
            obs = next_raw['observation']
        else:
            obs = _unwrap_obs(next_raw)
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32)

    return {
        'avg_td':     float(np.mean(step_tds)),
        'avg_mi':     float(np.mean(step_mis)),
        'avg_reward': float(np.mean(step_rewards)),
    }


def evaluate_algorithm(algo_name, env, actor_or_agent, num_episodes, horizon=64):
    ep_tds, ep_mis, ep_rewards = [], [], []
    for ep in range(num_episodes):
        r = evaluate_episode(algo_name, env, actor_or_agent, horizon)
        ep_tds.append(r['avg_td'])
        ep_mis.append(r['avg_mi'])
        ep_rewards.append(r['avg_reward'])
        if (ep + 1) % 10 == 0:
            print(f"    Episode {ep+1}/{num_episodes} — "
                f"TD={r['avg_td']:.3f}s  "
                f"MI={r['avg_mi']:.3e}bpcu  "
                f"R={r['avg_reward']:.4f}")

    return {
        'avg_td':     float(np.mean(ep_tds)),
        'std_td':     float(np.std(ep_tds)),
        'avg_mi':     float(np.mean(ep_mis)),
        'std_mi':     float(np.std(ep_mis)),
        'avg_reward': float(np.mean(ep_rewards)),
        'std_reward': float(np.std(ep_rewards)),
    }


# ─────────────────────────────────────────────
# Power sweep
# ─────────────────────────────────────────────

def evaluate_power_sweep(pmax_dbm_values, num_eval_episodes=50, horizon=64):
    base_models = os.path.join(os.path.dirname(__file__), '..', 'models')
    mappo_path  = os.path.join(base_models, 'mappo_final.pth')

    results = {
        'power_dbm': [],
        'mappo_avg_td': [], 'mappo_std_td': [],
        'mappo_avg_mi': [], 'mappo_std_mi': [],
        'mappo_avg_reward': [], 'mappo_std_reward': [],
        'dqn_avg_td': [], 'dqn_std_td': [],
        'dqn_avg_mi': [], 'dqn_std_mi': [],
        'dqn_avg_reward': [], 'dqn_std_reward': [],
    }

    # Load models once (they don't change across power sweep)
    print("Loading models...")
    actor    = _load_mappo_actor(mappo_path)
    dqn_agent = _load_dqn_agents(base_models)
    dqn_agent.eval_mode()

    # Save original params to restore later
    orig_dbm    = params.P_MAX_DBM
    orig_linear = params.P_MAX_LINEAR

    for pmax_dbm in pmax_dbm_values:
        pmax_linear = 10 ** (pmax_dbm / 10) * 1e-3   # dBm → Watts

        print(f"\n{'='*60}")
        print(f"Pmax = {pmax_dbm} dBm  ({pmax_linear*1e3:.2f} mW)")
        print(f"{'='*60}")

        # Update params AND create fresh env for this power level
        params.P_MAX_DBM    = pmax_dbm
        params.P_MAX_LINEAR = pmax_linear

        env = EnvironmentV2()
        env.current_stage = 2    # full action space (3D flight + power control)
        env.alpha = 0.5          # balanced for power sweep (adjust as needed)

        # Sanity check: print what the env will use
        print(f"  env.alpha={env.alpha}, P_MAX_LINEAR={params.P_MAX_LINEAR:.4e} W")

        if actor is not None:
            print(f"\n  Evaluating MAPPO...")
            mappo_metrics = evaluate_algorithm(
                'mappo', env, actor, num_eval_episodes, horizon
            )
        else:
            mappo_metrics = {k: 0.0 for k in [
                'avg_td','std_td','avg_mi','std_mi','avg_reward','std_reward']}

        print(f"\n  Evaluating DQN...")
        dqn_metrics = evaluate_algorithm(
            'dqn', env, dqn_agent, num_eval_episodes, horizon
        )

        results['power_dbm'].append(pmax_dbm)
        for key in ['avg_td','std_td','avg_mi','std_mi','avg_reward','std_reward']:
            results[f'mappo_{key}'].append(mappo_metrics[key])
            results[f'dqn_{key}'].append(dqn_metrics[key])

        print(f"\n  MAPPO — TD: {mappo_metrics['avg_td']:.3f}s | "
              f"MI: {mappo_metrics['avg_mi']:.3e} bpcu | "
              f"R: {mappo_metrics['avg_reward']:.4f}")
        print(f"  DQN   — TD: {dqn_metrics['avg_td']:.3f}s | "
              f"MI: {dqn_metrics['avg_mi']:.3e} bpcu | "
              f"R: {dqn_metrics['avg_reward']:.4f}")

    # Restore
    params.P_MAX_DBM    = orig_dbm
    params.P_MAX_LINEAR = orig_linear

    return results


# ─────────────────────────────────────────────
# Save & Plot
# ─────────────────────────────────────────────

def save_results_csv(results, output_file='power_sweep_results_fixed.csv'):
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    fieldnames = [
        'power_dbm',
        'mappo_avg_td', 'mappo_std_td',
        'mappo_avg_mi', 'mappo_std_mi',
        'mappo_avg_reward', 'mappo_std_reward',
        'dqn_avg_td', 'dqn_std_td',
        'dqn_avg_mi', 'dqn_std_mi',
        'dqn_avg_reward', 'dqn_std_reward',
    ]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(results['power_dbm'])):
            writer.writerow({k: results[k][i] for k in fieldnames})
    print(f"\nResults saved to {output_path}")


def plot_results(results, output_dir='.'):
    powers = results['power_dbm']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Power Sweep Evaluation (Fixed)', fontsize=13, fontweight='bold')

    panels = [
        ('avg_td',     'std_td',     'Avg TD per step (s)',       'TD vs UAV Power'),
        ('avg_mi',     'std_mi',     'Avg MI per step (bpcu)',     'MI vs UAV Power'),
        ('avg_reward', 'std_reward', 'Avg Reward per-UAV per-step','Reward vs UAV Power'),
    ]

    for ax, (avg_key, std_key, ylabel, title) in zip(axes, panels):
        for tag, color, marker, label in [
            ('mappo', 'steelblue', 'o', 'MAPPO'),
            ('dqn',   'darkorange', 's', 'Dueling DQN'),
        ]:
            avg = np.array(results[f'{tag}_{avg_key}'])
            std = np.array(results[f'{tag}_{std_key}'])
            ax.plot(powers, avg, f'{marker}-', color=color, label=label, lw=2, ms=6)
            ax.fill_between(powers, avg - std, avg + std, color=color, alpha=0.2)

        ax.set_xlabel('UAV Transmit Power (dBm)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if 'TD' in title:
            ax.set_ylim(bottom=0)   # TD cannot be negative
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, 'power_sweep_fixed.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out}")
    plt.close()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Match paper: 7.5 to 22.5 dBm in 2.5 dB steps
    pmax_values = np.arange(7.5, 23.0, 2.5).tolist()

    print("\n" + "="*60)
    print("POWER SWEEP EVALUATION  (Fixed)")
    print("="*60)
    print(f"Power range : {pmax_values[0]} – {pmax_values[-1]} dBm")
    print(f"Episodes    : 50 per power level")
    print("="*60)

    results = evaluate_power_sweep(pmax_values, num_eval_episodes=50, horizon=64)
    save_results_csv(results)
    plot_results(results)
    print("\nPower sweep complete.")