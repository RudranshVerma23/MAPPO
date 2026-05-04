#!/usr/bin/env python3
"""
Alpha Sweep Evaluation Script — FIXED VERSION
Bug fixes:
  1. env.alpha is now properly set (was always 1.0)
  2. episode_td/mi now average over steps (was only last step)
  3. Reward normalized per-UAV per-step (was 32*NUM_UAVS inflated)
  4. Separate model per alpha supported (optional, via model_dir pattern)
  5. Negative TD guard added
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
# Helpers
# ─────────────────────────────────────────────

def _unwrap_obs(td_field):
    """Extract raw observation tensor from TensorDict or dict."""
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
    """Load actor from checkpoint. Returns None if not found."""
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
    """Load all per-UAV DQN checkpoints."""
    agent = MultiAgentDuelingDQN(device='cpu')
    ok = 0
    for u in range(params.NUM_UAVS):
        pth = os.path.join(base_models, f'dqn_multiagent_uav{u}_final.pth')
        if os.path.exists(pth):
            try:
                agent.agents[u].load_state_dict(
                    torch.load(pth, map_location='cpu')
                )
                ok += 1
            except Exception as e:
                print(f"  [WARN] Could not load DQN UAV{u}: {e}")
    if ok != params.NUM_UAVS:
        print("  [WARN] DQN checkpoints incomplete/incompatible; DQN results may be invalid.")
    return agent


# ─────────────────────────────────────────────
# Core evaluation function
# ─────────────────────────────────────────────

def evaluate_episode(algo_name, env, actor_or_agent, horizon=64):
    """
    Run ONE evaluation episode.

    Returns per-step-averaged metrics:
      avg_td     : mean transmission delay per step (seconds), positive only
      avg_mi     : mean mutual information per step (bps)
      avg_reward : mean per-UAV per-step reward
    """
    td = env.reset()

    # --- observation extraction ---
    obs_raw = td['observation']
    if isinstance(obs_raw, dict):
        obs_tensor = obs_raw['observation']
    else:
        obs_tensor = _unwrap_obs(obs_raw)
    obs = obs_tensor if torch.is_tensor(obs_tensor) else torch.tensor(obs_tensor, dtype=torch.float32)

    step_tds, step_mis, step_rewards = [], [], []

    for _ in range(horizon):
        # ── action selection ──────────────────────────────────────────────
        if algo_name == 'mappo':
            actor = actor_or_agent
            with torch.no_grad():
                a1, a0 = actor(obs)
                # deterministic: Beta mean = α/(α+β)
                action_tensor = (a1 / (a1 + a0))          # shape (NUM_UAVS, ACTION_DIM)

        else:  # dqn
            dqn = actor_or_agent
            trajs, pwrs = [], []
            obs_np = obs.numpy()
            for u in range(params.NUM_UAVS):
                obs_u = torch.tensor(obs_np[u], dtype=torch.float32)
                a = dqn.select_action(obs_u, u)
                ti, pi = dqn.decode_action(a)
                trajs.append(dqn.traj_to_vel(ti))
                pwrs.append(dqn.power_to_alloc(pi))
            action_tensor = torch.cat(
                [torch.stack(trajs), torch.stack(pwrs)], dim=1
            )

        # ── environment step ──────────────────────────────────────────────
        out = env.step({'action': action_tensor})

        # ── reward: per-UAV average for this step ────────────────────────
        reward = out['reward']
        if torch.is_tensor(reward):
            step_reward = reward.mean().item()          # mean over UAVs, single step
        else:
            step_reward = float(np.mean(reward))
        step_rewards.append(step_reward)

        # ── TD and MI for this step ───────────────────────────────────────
        try:
            total_td_val, total_mi_val, _, _, served = total_td_mi(
                out['rates'], out['mis']
            )
            td_val = float(total_td_val.item())
            mi_val = float(total_mi_val.item())

            # BUG 1 guard: TD must be non-negative
            if td_val < 0:
                print(f"  [WARN] Negative TD encountered ({td_val:.4f}), clamping to 0")
                td_val = 0.0

        except Exception:
            td_val = 0.0
            mi_val = 0.0

        step_tds.append(td_val)
        step_mis.append(mi_val)

        # ── next obs ──────────────────────────────────────────────────────
        next_obs_raw = out['observation']
        if isinstance(next_obs_raw, dict):
            obs = next_obs_raw['observation']
        else:
            obs = _unwrap_obs(next_obs_raw)
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32)

    # average over steps (not sum — prevents horizon inflation)
    return {
        'avg_td':     float(np.mean(step_tds)),
        'avg_mi':     float(np.mean(step_mis)),
        'avg_reward': float(np.mean(step_rewards)),
    }


# ─────────────────────────────────────────────
# Per-algorithm evaluation over N episodes
# ─────────────────────────────────────────────

def evaluate_algorithm(algo_name, env, actor_or_agent, num_episodes, horizon=64):
    ep_tds, ep_mis, ep_rewards = [], [], []

    for ep in range(num_episodes):
        result = evaluate_episode(algo_name, env, actor_or_agent, horizon)
        ep_tds.append(result['avg_td'])
        ep_mis.append(result['avg_mi'])
        ep_rewards.append(result['avg_reward'])

        if (ep + 1) % 10 == 0:
            print(f"    Episode {ep+1}/{num_episodes} — "
                f"TD={result['avg_td']:.3f}s  "
                f"MI={result['avg_mi']:.3e}bpcu  "
                f"R={result['avg_reward']:.4f}")

    return {
        'avg_td':     float(np.mean(ep_tds)),
        'std_td':     float(np.std(ep_tds)),
        'avg_mi':     float(np.mean(ep_mis)),
        'std_mi':     float(np.std(ep_mis)),
        'avg_reward': float(np.mean(ep_rewards)),
        'std_reward': float(np.std(ep_rewards)),
    }


# ─────────────────────────────────────────────
# Alpha sweep
# ─────────────────────────────────────────────

def evaluate_alpha_sweep(
    alpha_values,
    num_eval_episodes=50,
    horizon=64,
    # If you trained a model per alpha value, point to the directory pattern:
    # e.g. models/alpha_0.50/mappo_final.pth
    # Set to None to always use the single trained model (not ideal)
    per_alpha_model_dir=None,
):
    base_models = os.path.join(os.path.dirname(__file__), '..', 'models')

    # Auto-detect per-alpha checkpoints if present.
    if per_alpha_model_dir is None:
        probe = os.path.join(base_models, f'alpha_{alpha_values[0]:.2f}', 'mappo_final.pth')
        if os.path.exists(probe):
            per_alpha_model_dir = base_models
            print(f"[AUTO] Per-alpha checkpoints detected in {base_models}")
        else:
            print("[NOTE] No per-alpha checkpoints detected. Using shared mappo_final.pth.")

    results = {
        'alpha': [],
        'mappo_avg_td': [], 'mappo_std_td': [],
        'mappo_avg_mi': [], 'mappo_std_mi': [],
        'mappo_avg_reward': [], 'mappo_std_reward': [],
        'dqn_avg_td': [], 'dqn_std_td': [],
        'dqn_avg_mi': [], 'dqn_std_mi': [],
        'dqn_avg_reward': [], 'dqn_std_reward': [],
    }

    # Pre-load DQN (doesn't change across alpha)
    print("Loading DQN agents...")
    dqn_agent = _load_dqn_agents(base_models)
    dqn_agent.eval_mode()

    for alpha in alpha_values:
        print(f"\n{'='*60}")
        print(f"Alpha = {alpha:.2f}  (comm={alpha:.2f}, sensing={1-alpha:.2f})")
        print(f"{'='*60}")

        # ── FIX 1: Create env with correct alpha ──────────────────────────
        env = EnvironmentV2()
        env.alpha = alpha           # ← THE CRITICAL FIX: was never set before
        env.current_stage = 2       # unlock full action space for evaluation

        # ── FIX 2: Load per-alpha model if available ──────────────────────
        if per_alpha_model_dir is not None:
            mappo_path = os.path.join(
                per_alpha_model_dir,
                f'alpha_{alpha:.2f}',
                'mappo_final.pth'
            )
        else:
            # Fall back to single model (less ideal — policy ignores alpha)
            mappo_path = os.path.join(base_models, 'mappo_final.pth')
            if alpha != alpha_values[0]:
                print("  [NOTE] Using same model for all alpha — "
                      "for proper sweep, train one model per alpha value.")

        actor = _load_mappo_actor(mappo_path)
        if actor is None:
            null_metrics = {k: 0.0 for k in [
                'avg_td','std_td','avg_mi','std_mi','avg_reward','std_reward']}
            mappo_metrics = null_metrics
        else:
            print(f"\n  Evaluating MAPPO (alpha={alpha:.2f})...")
            mappo_metrics = evaluate_algorithm(
                'mappo', env, actor, num_eval_episodes, horizon
            )

        print(f"\n  Evaluating DQN (alpha={alpha:.2f})...")
        dqn_metrics = evaluate_algorithm(
            'dqn', env, dqn_agent, num_eval_episodes, horizon
        )

        results['alpha'].append(alpha)
        for key in ['avg_td', 'std_td', 'avg_mi', 'std_mi', 'avg_reward', 'std_reward']:
            results[f'mappo_{key}'].append(mappo_metrics[key])
            results[f'dqn_{key}'].append(dqn_metrics[key])

        print(f"\n  MAPPO — TD: {mappo_metrics['avg_td']:.3f}s | "
              f"MI: {mappo_metrics['avg_mi']:.3e} bpcu | "
              f"R: {mappo_metrics['avg_reward']:.4f}")
        print(f"  DQN   — TD: {dqn_metrics['avg_td']:.3f}s | "
              f"MI: {dqn_metrics['avg_mi']:.3e} bpcu | "
              f"R: {dqn_metrics['avg_reward']:.4f}")

    return results


# ─────────────────────────────────────────────
# Save & Plot
# ─────────────────────────────────────────────

def save_results_csv(results, output_file='alpha_sweep_results_fixed.csv'):
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    fieldnames = [
        'alpha',
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
        for i in range(len(results['alpha'])):
            writer.writerow({k: results[k][i] for k in fieldnames})
    print(f"\nResults saved to {output_path}")


def plot_results(results, output_dir='.'):
    alphas = results['alpha']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Alpha Sweep Evaluation (Fixed)', fontsize=13, fontweight='bold')

    # ── TD ────────────────────────────────────────────────────────────────
    ax = axes[0]
    for tag, color, marker, label in [
        ('mappo', 'steelblue', 'o', 'MAPPO'),
        ('dqn',   'darkorange', 's', 'Dueling DQN'),
    ]:
        avg = np.array(results[f'{tag}_avg_td'])
        std = np.array(results[f'{tag}_std_td'])
        ax.plot(alphas, avg, f'{marker}-', color=color, label=label, lw=2, ms=6)
        ax.fill_between(alphas, avg - std, avg + std, color=color, alpha=0.2)

    ax.set_xlabel('Alpha (Communication Weight)')
    ax.set_ylabel('Avg TD per step (seconds)')
    ax.set_title('TD vs Communication Weight')
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)       # TD cannot be negative
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── MI ────────────────────────────────────────────────────────────────
    ax = axes[1]
    for tag, color, marker, label in [
        ('mappo', 'steelblue', 'o', 'MAPPO'),
        ('dqn',   'darkorange', 's', 'Dueling DQN'),
    ]:
        avg = np.array(results[f'{tag}_avg_mi'])
        std = np.array(results[f'{tag}_std_mi'])
        ax.plot(alphas, avg, f'{marker}-', color=color, label=label, lw=2, ms=6)
        ax.fill_between(alphas, avg - std, avg + std, color=color, alpha=0.2)

    ax.set_xlabel('Alpha (Communication Weight)')
    ax.set_ylabel('Avg MI per step (bpcu)')
    ax.set_title('MI vs Communication Weight')
    ax.set_xlim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Reward ────────────────────────────────────────────────────────────
    ax = axes[2]
    for tag, color, marker, label in [
        ('mappo', 'steelblue', 'o', 'MAPPO'),
        ('dqn',   'darkorange', 's', 'Dueling DQN'),
    ]:
        avg = np.array(results[f'{tag}_avg_reward'])
        std = np.array(results[f'{tag}_std_reward'])
        ax.plot(alphas, avg, f'{marker}-', color=color, label=label, lw=2, ms=6)
        ax.fill_between(alphas, avg - std, avg + std, color=color, alpha=0.2)

    ax.set_xlabel('Alpha (Communication Weight)')
    ax.set_ylabel('Avg reward per-UAV per-step')
    ax.set_title('Reward vs Communication Weight')
    ax.set_xlim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, 'alpha_sweep_fixed.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out}")
    plt.close()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    alpha_values = np.arange(0.0, 1.1, 0.1).tolist()

    print("\n" + "="*60)
    print("ALPHA SWEEP EVALUATION  (Fixed)")
    print("="*60)
    print(f"Alpha range : {alpha_values[0]:.2f} – {alpha_values[-1]:.2f}")
    print(f"Episodes    : 50 per alpha")
    print()
    print("NOTE: For a proper tradeoff curve, train one MAPPO model")
    print("per alpha value and set per_alpha_model_dir accordingly.")
    print("="*60)

    results = evaluate_alpha_sweep(
        alpha_values,
        num_eval_episodes=50,
        horizon=64,
        per_alpha_model_dir=None,   # change to your per-alpha model dir if available
    )

    save_results_csv(results)
    plot_results(results)
    print("\nAlpha sweep complete.")