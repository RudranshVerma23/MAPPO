"""Multi-Agent Dueling DQN for NOMA-ISAC (Paper-Faithful)."""
import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import params
from environment_v2 import EnvironmentV2

class DuelingDQN(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.adv = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_actions))
    def forward(self, x):
        h = self.fc(x)
        v = self.value(h)
        a = self.adv(h)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

class MultiAgentDuelingDQN:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.env = EnvironmentV2(device=device)
        # Stage 2 from init: full 3D flight + power control.
        # Stage 1 (2D only, equal power) makes power learning impossible — fix.
        self.env.current_stage = 2
        self.env.alpha = 0.5     # balanced objective
        # T6: Reduced from 3 to 2 power levels: 3^5=243 → 2^5=32 combos.
        # Total actions: 7*32=224 (was 1701). Much more learnable with small budget.
        self.n_traj, self.n_power_levels = 7, 2
        self.n_power_combos = self.n_power_levels ** params.K_MAX
        self.n_actions = self.n_traj * self.n_power_combos
        self.agents, self.targets, self.opts = [], [], []
        for u in range(params.NUM_UAVS):
            net = DuelingDQN(params.OBS_DIM, self.n_actions, 128).to(self.device)
            tgt = DuelingDQN(params.OBS_DIM, self.n_actions, 128).to(self.device)
            tgt.load_state_dict(net.state_dict())
            self.agents.append(net)
            self.targets.append(tgt)
            self.opts.append(optim.Adam(net.parameters(), lr=5e-4))
        # T3a: deque per agent — O(1) append/pop, no shared split imbalance.
        # T9: 64 samples per agent per update (was ~10 from shared 32-split-3).
        self.MAX_REPLAY = 20000
        self.batch_size = 64
        self.replays = [collections.deque(maxlen=self.MAX_REPLAY) for _ in range(params.NUM_UAVS)]
        self.gamma, self.eps, self.eps_min, self.eps_decay = 0.99, 1.0, 0.05, 0.995
        # T10 (target update): every 500 steps instead of every 20 epochs
        self.target_update_freq = 500
        self._step_count = 0
        self._train_eps = 1.0   # saved eps for restoring after eval
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, 'dqn_multiagent_train_log.csv')
        Path(self.log_path).write_text('epoch,avg_reward_per_uav,policy_loss,avg_qos,rates_min,rates_max,total_td_s,total_mi_bpcu,avg_td_s,avg_mi_bpcu,served_users\n')

    def eval_mode(self):
        """Greedy policy — zero exploration. Call before evaluation."""
        self._train_eps = self.eps
        self.eps = 0.0
        for net in self.agents:
            net.eval()

    def train_mode(self):
        """Restore training policy. Call after evaluation."""
        self.eps = self._train_eps
        for net in self.agents:
            net.train()
    
    def decode_action(self, a):
        pc, ti = a % self.n_power_combos, a // self.n_power_combos
        pi = []
        for _ in range(params.K_MAX):
            pi.append(pc % self.n_power_levels)
            pc //= self.n_power_levels
        return ti, pi[::-1]
    
    def traj_to_vel(self, ti):
        # Environment expects normalized velocity inputs in [0, 1], not m/s.
        # 0.0 -> -Vmax, 0.5 -> hover, 1.0 -> +Vmax.
        dirs = [
            [0.0, 0.5, 0.5],
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
        ]
        return torch.tensor(dirs[ti], dtype=torch.float32, device=self.device)
    
    def power_to_alloc(self, pi):
        # Emit logits; env applies softmax → power ratios summing to 1.
        # 2 levels: low=0.3, high=1.0 (index 0=low, 1=high).
        levels = [0.3, 1.0]
        pa = torch.zeros(params.K_MAX, device=self.device)
        for k in range(params.K_MAX):
            pa[k] = torch.log(torch.tensor(levels[pi[k]], dtype=torch.float32, device=self.device))
        return pa

    def _total_td_mi(self, rates, mis):
        if torch.is_tensor(rates) and rates.dim() > 1:
            rates = rates[0]
        if torch.is_tensor(mis) and mis.dim() > 1:
            mis = mis[0]
        valid = rates > 0
        if valid.any():
            td_vals = params.PACKET_SIZE_BITS / (rates[valid] + 1e-12)
            mi_vals = mis[valid]
            total_td = torch.sum(td_vals).item()
            total_mi = torch.sum(mi_vals).item()
            avg_td = torch.mean(td_vals).item()
            avg_mi = torch.mean(mi_vals).item()
            served = int(valid.sum().item())
        else:
            total_td = 0.0
            total_mi = 0.0
            avg_td = 0.0
            avg_mi = 0.0
            served = 0
        return total_td, total_mi, avg_td, avg_mi, served
    
    def select_action(self, obs, u):
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            o = obs.unsqueeze(0) if len(obs.shape) == 1 else obs
            q = self.agents[u](o.to(self.device))
            return q.argmax(dim=1).item()
    
    def train(self, epochs=150, horizon=32):
        for ep in range(epochs):
            td = self.env.reset()
            obs_t = td['observation']
            if hasattr(obs_t, 'get'):
                obs_t = obs_t.get('observation', obs_t)
            ep_rew = torch.zeros((params.NUM_UAVS, 1), device=self.device)
            qos_vals, rates_vals, total_td_vals, total_mi_vals, avg_td_vals, avg_mi_vals, served_vals, loss_sum = [], [], [], [], [], [], [], 0.0
            for t in range(horizon):
                acts = [self.select_action(obs_t[u] if torch.is_tensor(obs_t) else torch.tensor(obs_t[u], device=self.device, dtype=torch.float32), u) for u in range(params.NUM_UAVS)]
                trajs = [self.traj_to_vel(self.decode_action(acts[u])[0]) for u in range(params.NUM_UAVS)]
                pwrs = [self.power_to_alloc(self.decode_action(acts[u])[1]) for u in range(params.NUM_UAVS)]
                action_env = torch.cat([torch.stack(trajs), torch.stack(pwrs)], dim=1)
                out = self.env.step({'action': action_env})
                rews, obs_next_t = out['reward'], out['observation']
                if hasattr(obs_next_t, 'get'):
                    obs_next_t = obs_next_t.get('observation', obs_next_t)
                ep_rew += rews
                try:
                    rates, mis = out['rates'], out['mis']
                    qos = (rates >= params.R_TH) & (mis >= params.I_TH)
                    qos_vals.append(qos.float().mean().item())
                    rates_vals.append((rates.min().item(), rates.max().item()))
                    total_td, total_mi, avg_td, avg_mi, served = self._total_td_mi(rates, mis)
                    total_td_vals.append(total_td)
                    total_mi_vals.append(total_mi)
                    avg_td_vals.append(avg_td)
                    avg_mi_vals.append(avg_mi)
                    served_vals.append(served)
                except:
                    pass
                for u in range(params.NUM_UAVS):
                    obs_u_np = obs_t[u].cpu().numpy() if torch.is_tensor(obs_t[u]) else obs_t[u]
                    obs_next_u_np = obs_next_t[u].cpu().numpy() if torch.is_tensor(obs_next_t[u]) else obs_next_t[u]
                    r_u = rews[u, 0].item()
                    done = out['done'][u, 0].item() if torch.is_tensor(out['done'][u, 0]) else False
                    # T3: push directly into per-agent deque (O(1), no shared split)
                    self.replays[u].append((obs_u_np, acts[u], r_u, obs_next_u_np, done))

                # T9: update each agent from its own buffer (64 samples each)
                for u in range(params.NUM_UAVS):
                    if len(self.replays[u]) < self.batch_size:
                        continue
                    batch = random.sample(self.replays[u], self.batch_size)
                    o_b  = torch.tensor([x[0] for x in batch], dtype=torch.float32, device=self.device)
                    a_b  = torch.tensor([x[1] for x in batch], dtype=torch.long,    device=self.device)
                    r_b  = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=self.device)
                    on_b = torch.tensor([x[3] for x in batch], dtype=torch.float32, device=self.device)
                    d_b  = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=self.device)
                    q = self.agents[u](o_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        qn = self.targets[u](on_b).max(1)[0]
                        tq = r_b + self.gamma * qn * (1 - d_b)
                    loss = nn.functional.mse_loss(q, tq)
                    self.opts[u].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agents[u].parameters(), 0.5)
                    self.opts[u].step()
                    loss_sum += loss.item()

                # T10: step-aligned target network sync
                self._step_count += 1
                if self._step_count % self.target_update_freq == 0:
                    for u in range(params.NUM_UAVS):
                        self.targets[u].load_state_dict(self.agents[u].state_dict())

                obs_t = obs_next_t
            self.eps = max(self.eps * self.eps_decay, self.eps_min)
            avg_r = (ep_rew.mean() / horizon).item()
            avg_qos = np.mean(qos_vals) if qos_vals else 0.0
            r_min = min(r[0] for r in rates_vals) if rates_vals else 0.0
            r_max = max(r[1] for r in rates_vals) if rates_vals else 0.0
            total_td = float(np.mean(total_td_vals)) if total_td_vals else 0.0
            total_mi = float(np.mean(total_mi_vals)) if total_mi_vals else 0.0
            avg_td = float(np.mean(avg_td_vals)) if avg_td_vals else 0.0
            avg_mi = float(np.mean(avg_mi_vals)) if avg_mi_vals else 0.0
            served_users = float(np.mean(served_vals)) if served_vals else 0.0
            with open(self.log_path, 'a') as f:
                f.write(f"{ep},{avg_r:.6e},{loss_sum/max(horizon,1):.6e},{avg_qos*100:.1f},{r_min:.2e},{r_max:.2e},{total_td:.6e},{total_mi:.6e},{avg_td:.6e},{avg_mi:.6e},{served_users:.2f}\n")
            if ep % 10 == 0:
                print(f"Epoch {ep} | AvgR/U {avg_r:.3e} | QoS {avg_qos*100:.1f}% | AvgTD {avg_td:.3e}s | AvgMI {avg_mi:.3e}bpcu | Rates [{r_min:.2e}, {r_max:.2e}] | Eps {self.eps:.3f} | Steps {self._step_count}")
        for u in range(params.NUM_UAVS):
            model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            os.makedirs(model_dir, exist_ok=True)
            torch.save(self.agents[u].state_dict(), os.path.join(model_dir, f'dqn_multiagent_uav{u}_final.pth'))

if __name__ == '__main__':
    trainer = MultiAgentDuelingDQN(device='cpu')
    trainer.train(epochs=500, horizon=64)