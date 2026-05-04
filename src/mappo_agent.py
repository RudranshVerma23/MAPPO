import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Beta

import params
from curriculum import CurriculumManager
from environment_v2 import EnvironmentV2


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.head = nn.Linear(64, action_dim * 2)

    def forward(self, obs):
        h = self.net(obs)
        out = self.head(h)
        out = out.view(*out.shape[:-1], -1, 2)
        a = torch.nn.functional.softplus(out) + 1.0
        return a[...,0], a[...,1]


class Critic(nn.Module):
    def __init__(self, global_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, g):
        return self.net(g).squeeze(-1)


def sample_action(alpha, beta):
    # alpha,beta shape: (num_agents, action_dim)
    # Use non-reparameterized sampling (policy gradient via log-prob).
    dist = Beta(alpha, beta)
    sample = dist.sample()
    logp = dist.log_prob(sample).sum(dim=-1)
    return sample, logp, dist.entropy().sum(dim=-1)


def total_td_mi(rates, mis):
    rates = torch.as_tensor(rates).detach().float()
    mis = torch.as_tensor(mis).detach().float()
    if rates.dim() > 1:
        rates = rates[0]
    if mis.dim() > 1:
        mis = mis[0]
    valid = rates > 0
    if valid.any():
        td_vals = params.PACKET_SIZE_BITS / (rates[valid] + 1e-12)
        mi_vals = mis[valid]
        total_td = torch.sum(td_vals)
        total_mi = torch.sum(mi_vals)
        avg_td = torch.mean(td_vals)
        avg_mi = torch.mean(mi_vals)
        served = int(valid.sum().item())
    else:
        total_td = torch.tensor(0.0)
        total_mi = torch.tensor(0.0)
        avg_td = torch.tensor(0.0)
        avg_mi = torch.tensor(0.0)
        served = 0
    return total_td, total_mi, avg_td, avg_mi, served


def train(env_device='cpu', epochs=500, horizon=64, ppo_epochs=4, clip_epsilon=0.2,
          lam=0.95, entropy_coeff=0.10, save_every=100,
          fixed_alpha=None, save_dir=None):
    """
    fixed_alpha: if set, env.alpha is locked to this value — curriculum cannot override it.
                 Required for alpha-sweep training (T15).
    save_dir:    directory for model checkpoints and log file. Defaults to project models/logs.
    """
    device = torch.device(env_device)
    env = EnvironmentV2(device=device)
    if fixed_alpha is not None:
        env.alpha = fixed_alpha   # lock alpha; curriculum stage-3 won't overwrite

    actor = Actor(params.OBS_DIM, params.ACTION_DIM).to(device)
    critic = Critic(params.GLOBAL_STATE_DIM).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=5e-4)

    gamma = 0.99

    def _unwrap(td_field):
        try:
            if torch.is_tensor(td_field):
                return td_field
        except Exception:
            pass
        try:
            return td_field['observation']
        except Exception:
            try:
                return td_field['global_state']
            except Exception:
                return td_field

    # logging csv
    import os
    _log_dir   = save_dir if save_dir else os.path.join(os.path.dirname(__file__), '..', 'logs')
    _model_dir = save_dir if save_dir else os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(_log_dir,   exist_ok=True)
    os.makedirs(_model_dir, exist_ok=True)
    log_file = os.path.join(_log_dir, 'mappo_train_log.csv')
    with open(log_file, 'w') as f:
        f.write('epoch,avg_reward,policy_loss,critic_loss,entropy,avg_value,total_td_s,total_mi_bpcu,avg_td_s,avg_mi_bpcu,served_users\n')

    for epoch in range(epochs):
        # ensure curriculum manager present
        if epoch == 0:
            manager = CurriculumManager(env)
        td = env.reset()
        obs = _unwrap(td['observation'])
        obs = obs.to(device) if torch.is_tensor(obs) else torch.tensor(obs, device=device, dtype=torch.float32)

        obs_buf, gstate_buf, actions_buf = [], [], []
        logp_buf, entropy_buf, value_buf, reward_buf = [], [], [], []
        total_td_buf, total_mi_buf = [], []
        avg_td_buf, avg_mi_buf = [], []
        served_buf = []

        for t in range(horizon):
            conc1, conc0 = actor(obs)
            action_sample, logp, ent = sample_action(conc1, conc0)

            out = env.step({'action': action_sample})

            r = out['reward']
            if not torch.is_tensor(r):
                r = torch.tensor(r, device=device, dtype=torch.float32)
            r = r.squeeze(-1).to(device)

            gstate_field = out.get('global_state', None) if hasattr(out, 'get') else out['global_state']
            gstate = gstate_field if torch.is_tensor(gstate_field) else torch.tensor(gstate_field, device=device, dtype=torch.float32)

            v = critic(gstate)

            obs_buf.append(obs)
            gstate_buf.append(gstate)
            actions_buf.append(action_sample)
            logp_buf.append(logp)
            entropy_buf.append(ent)
            value_buf.append(v)
            reward_buf.append(r)

            obs_field = out['observation']
            obs = _unwrap(obs_field)
            obs = obs.to(device) if torch.is_tensor(obs) else torch.tensor(obs, device=device, dtype=torch.float32)

        rewards = torch.stack(reward_buf, dim=0)
        values = torch.stack(value_buf, dim=0)
        old_logps = torch.stack(logp_buf, dim=0)
        entropies = torch.stack(entropy_buf, dim=0)

        with torch.no_grad():
            last_g = gstate_buf[-1]
            last_val = critic(last_g)

        T, U = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(U, device=device)
        next_value = last_val
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae
            next_value = values[t]

        returns = advantages + values

        obs_flat = torch.cat([o for o in obs_buf], dim=0).detach()
        g_flat = torch.cat([g for g in gstate_buf], dim=0).detach()
        actions_flat = torch.cat([a for a in actions_buf], dim=0).detach()
        old_logp_flat = old_logps.view(-1).detach()
        adv_flat = advantages.view(-1).detach()
        ret_flat = returns.view(-1).detach()
        ent_flat = entropies.view(-1)

        policy_loss_val = 0.0
        critic_loss_val = 0.0
        for _ in range(ppo_epochs):
            conc1_new, conc0_new = actor(obs_flat)
            dist_new = Beta(conc1_new, conc0_new)
            new_logp = dist_new.log_prob(actions_flat).sum(dim=-1)
            entropy_new = dist_new.entropy().sum(dim=-1)

            ratio = torch.exp(new_logp - old_logp_flat)
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_flat
            # T7: anneal entropy coeff: 0.10 -> 0.005 over training.
            # High early entropy drives exploration; low late entropy drives convergence.
            annealed_ent = max(entropy_coeff * (0.99 ** epoch), 0.005)
            policy_loss = -torch.mean(torch.min(surr1, surr2)) - annealed_ent * entropy_new.mean()

            actor_opt.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_opt.step()

            values_pred = critic(g_flat).view(-1)
            critic_loss = nn.functional.mse_loss(values_pred, ret_flat)
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_opt.step()

            policy_loss_val += policy_loss.item()
            critic_loss_val += critic_loss.item()

        policy_loss_val /= ppo_epochs
        critic_loss_val /= ppo_epochs

        avg_reward = rewards.mean().item()
        avg_entropy = ent_flat.mean().item()
        avg_value = values.mean().item()

        # Compute QoS satisfaction from last environment output (rates, mis)
        try:
            last_rates = out['rates']
            last_mis = out['mis']
            qos_met = (last_rates >= params.R_TH) & (last_mis >= params.I_TH)
            qos_satisfaction = qos_met.float().mean().item()
            td_step, mi_step, avg_td_step, avg_mi_step, served_step = total_td_mi(last_rates, last_mis)
            total_td_buf.append(td_step.item())
            total_mi_buf.append(mi_step.item())
            avg_td_buf.append(avg_td_step.item())
            avg_mi_buf.append(avg_mi_step.item())
            served_buf.append(served_step)
        except Exception:
            qos_satisfaction = 0.0

        total_td = float(np.mean(total_td_buf)) if total_td_buf else 0.0
        total_mi = float(np.mean(total_mi_buf)) if total_mi_buf else 0.0
        avg_td = float(np.mean(avg_td_buf)) if avg_td_buf else 0.0
        avg_mi = float(np.mean(avg_mi_buf)) if avg_mi_buf else 0.0
        served_users = float(np.mean(served_buf)) if served_buf else 0.0

        # Update curriculum manager
        try:
            manager.update_stage(avg_reward, qos_satisfaction)
            # Restore immediately — curriculum stage-3 sets env.alpha=0.5 internally.
            if fixed_alpha is not None:
                env.alpha = fixed_alpha
        except Exception:
            pass

        with open(log_file, 'a') as f:
            f.write(f"{epoch},{avg_reward:.6e},{policy_loss_val:.6e},{critic_loss_val:.6e},{avg_entropy:.6e},{avg_value:.6e},{total_td:.6e},{total_mi:.6e},{avg_td:.6e},{avg_mi:.6e},{served_users:.2f}\n")

        if epoch % 10 == 0:
            try:
                rates_min, rates_max = last_rates.min().item(), last_rates.max().item()
                mis_min, mis_max = last_mis.min().item(), last_mis.max().item()
                print(f"Epoch {epoch} | Stage {env.current_stage} | Reward {avg_reward:.3e} | QoS {qos_satisfaction*100:.1f}% | AvgTD {avg_td:.3e}s | AvgMI {avg_mi:.3e}bpcu | Rates [{rates_min:.2e},{rates_max:.2e}] | MI [{mis_min:.2e},{mis_max:.2e}]")
            except Exception:
                print(f"Epoch {epoch} | Avg reward {avg_reward:.3e} | PolicyLoss {policy_loss_val:.3e} | CriticLoss {critic_loss_val:.3e} | Ent {avg_entropy:.3e}")

        if epoch % save_every == 0 and epoch > 0:
            torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict(), 'epoch': epoch},
                       os.path.join(_model_dir, f'mappo_checkpoint_{epoch}.pth'))

    torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict()},
               os.path.join(_model_dir, 'mappo_final.pth'))
    return actor, critic   # return models so sweep script can inspect without reloading


if __name__ == '__main__':
    train(env_device='cpu', epochs=500)