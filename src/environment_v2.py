import torch
from torch import nn
from tensordict import TensorDict
import params
import math


class EnvironmentV2:
    """Paper-faithful NOMA-ISAC environment.
    - Multi-UAV
    - Rayleigh fading
    - Mobile users (annealed by mobility_factor)
    - DWAHC clustering (simplified density-weighted)
    - Dynamic SIC ordering and per-UAV Eq(14) reward
    """

    def __init__(self, device='cpu', mobility_factor=params.DEFAULT_MOBILITY_FACTOR):
        self.device = device
        self.mobility_factor = mobility_factor
        self.alpha = 1.0
        self.current_stage = 1

    def reset(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        # UAV positions: (NUM_UAVS, 3)
        self.uav_pos = torch.rand((params.NUM_UAVS, 3), device=self.device) * params.AREA_SIZE
        self.uav_pos[:, 2] = torch.distributions.Uniform(params.UAV_ALT_MIN, params.UAV_ALT_MAX).sample(self.uav_pos[:, 2].shape)

        # User positions: (NUM_USERS, 2)
        self.user_pos = torch.rand((params.NUM_USERS, 2), device=self.device) * params.AREA_SIZE

        # Initial clustering using DWAHC
        self.clusters = self._dwahc_clustering(self.uav_pos, self.user_pos)

        obs = self._get_obs()
        td = TensorDict({
            'uav_pos': self.uav_pos,
            'user_pos': self.user_pos,
            'clusters': self.clusters,
            'observation': obs,
        }, batch_size=[])
        return td

    def _get_obs(self):
        # local observation per UAV: normalized pos, dist to centroid, one-hot id, local gains (log10)
        obs = torch.zeros((params.NUM_UAVS, params.OBS_DIM), device=self.device)
        g_all, _ = self.calculate_channel_gains(self.uav_pos, self.user_pos)

        for u in range(params.NUM_UAVS):
            my_pos = self.uav_pos[u]
            agent_id = torch.zeros(params.NUM_UAVS, device=self.device)
            agent_id[u] = 1.0

            user_indices = self.clusters[u]
            mask = (user_indices != -1)
            valid = user_indices[mask]
            if len(valid) > 0:
                assigned = self.user_pos[valid]
                centroid = assigned.mean(dim=0)
                local_gains = torch.log10(g_all[u, valid] + 1e-12)
                pad = torch.zeros(params.K_MAX - len(valid), device=self.device)
                local_gains = torch.cat([local_gains, pad])
            else:
                centroid = my_pos[:2]
                local_gains = torch.zeros(params.K_MAX, device=self.device)

            dist_to_cluster = torch.norm(my_pos[:2] - centroid).reshape(1)
            norm_pos = my_pos / params.AREA_SIZE
            norm_dist = dist_to_cluster / params.AREA_SIZE
            # Append alpha so policy can condition on comm/sensing tradeoff weight.
            alpha_scalar = torch.tensor([self.alpha], device=self.device)
            obs[u] = torch.cat([norm_pos, norm_dist, agent_id, local_gains, alpha_scalar])

        # global state: flattened local obs concatenated
        global_state = obs.view(-1).unsqueeze(0).expand(params.NUM_UAVS, -1)
        return {'observation': obs, 'global_state': global_state}

    def calculate_channel_gains(self, uav_pos, user_pos):
        # distance (U, K)
        u = params.NUM_UAVS
        k = params.NUM_USERS
        diff = uav_pos.unsqueeze(1) - torch.cat([user_pos, torch.zeros(k,1, device=self.device)], dim=-1).unsqueeze(0)
        dist = torch.norm(diff, dim=-1)

        # FSPL at f ~ 2.4 GHz
        c = 3e8
        f = 2.4e9
        pl = 20 * torch.log10(dist + 1e-6) + 20 * math.log10(4 * math.pi * f / c)

        # Rayleigh small-scale fading CN(0,1) -> power |h|^2
        real = torch.randn_like(dist) * math.sqrt(0.5)
        imag = torch.randn_like(dist) * math.sqrt(0.5)
        h_pow = real**2 + imag**2

        g_linear = h_pow * torch.pow(10.0, -pl / 10.0)
        return g_linear, dist

    def _dwahc_clustering(self, uav_pos, user_pos):
        # T17: vectorized DWAHC.
        # rho: (U,) via broadcasted cdist < R — no Python loop over UAVs.
        # cluster slot-fill: loop over U (3), not K (15) — 5x fewer Python iters.
        k = params.NUM_USERS
        u = params.NUM_UAVS
        dist = torch.cdist(user_pos, uav_pos[:, :2])   # (K, U)

        # Vectorized density: dist already computed, reuse. (K, U) < R → sum over K axis.
        R = params.AREA_SIZE / 10.0
        rho = (dist < R).float().sum(dim=0)             # (U,) — no Python loop

        weighted = dist / (1.0 + rho.unsqueeze(0))      # (K, U)
        nearest  = torch.argmin(weighted, dim=1)         # (K,) — each user → UAV idx

        clusters = torch.full((u, params.K_MAX), -1, dtype=torch.long, device=self.device)
        # Loop over UAVs (3), not users (15). For each UAV, gather its users in one shot.
        for ui in range(u):
            assigned = (nearest == ui).nonzero(as_tuple=True)[0]  # user indices for UAV ui
            n = min(len(assigned), params.K_MAX)
            if n > 0:
                clusters[ui, :n] = assigned[:n]

        return clusters

    def compute_isac_metrics(self, uav_pos, user_pos, power_alloc, clusters):
        # power_alloc: (NUM_UAVS, K_MAX) absolute Watts assigned per potential user slot
        g_all, _ = self.calculate_channel_gains(uav_pos, user_pos)
        rates = torch.zeros(params.NUM_USERS, device=self.device)
        mis   = torch.zeros(params.NUM_USERS, device=self.device)

        # T16: vectorized inter-cluster interference — (NUM_UAVS, NUM_USERS) in one shot.
        # inter_if_full[u, k] = sum_{s≠u} g[s,k] * power_alloc[s].sum()
        total_power = power_alloc.sum(dim=1)                    # (U,)
        # g_all: (U, K).  Outer product via einsum: sum over s dimension excluding u.
        # inter_all[u, k] = sum_s (g[s,k] * total_power[s]) - g[u,k]*total_power[u]
        inter_all_full = (g_all * total_power.unsqueeze(1)).sum(dim=0, keepdim=True) \
                         - g_all * total_power.unsqueeze(1)     # (U, K)

        _T2_fs = (params.SYMBOL_DURATION ** 2) * params.SUB_CARRIER_INTERVAL  # scalar constant

        for u in range(params.NUM_UAVS):
            user_indices = clusters[u]
            mask  = (user_indices != -1)
            valid = user_indices[mask]              # local user indices, shape (n,)
            n     = len(valid)
            if n == 0:
                continue

            inter_if    = inter_all_full[u, valid]  # (n,) — no Python loop over UAVs
            g_u_valid   = g_all[u, valid]           # (n,)
            cluster_pwr = power_alloc[u][mask]      # (n,)

            # SIC ordering: ascending effective channel quality
            g_equiv    = g_u_valid / (inter_if + params.NOISE_POWER_LINEAR)
            sorted_idx = torch.argsort(g_equiv)     # (n,) local permutation

            # Reorder to SIC decode sequence
            g_s   = g_u_valid[sorted_idx]           # (n,)
            p_s   = cluster_pwr[sorted_idx]         # (n,)
            ii_s  = inter_if[sorted_idx]            # (n,) inter-cluster IF in sorted order

            # T16 core: intra-cluster IF for position i = sum_{j>i} g_s[j]*p_s[j]
            # = suffix sum of (g_s * p_s) shifted by one.
            prod      = g_s * p_s                                       # (n,)
            cumsum    = torch.cumsum(prod, dim=0)                       # cumsum[i] = sum(prod[0..i])
            total_sum = cumsum[-1]
            # intra_if[i] = total_sum - cumsum[i]  (= sum of prod[i+1..n-1])
            intra_if_s = total_sum - cumsum                             # (n,)

            denom_s = intra_if_s + ii_s + params.NOISE_POWER_LINEAR    # (n,)
            sinr_s  = g_s * p_s / denom_s                              # (n,)

            rates_s = params.BANDWIDTH * torch.log2(1.0 + sinr_s)
            mis_s   = 0.5 * torch.log2(1.0 + (g_s * p_s * _T2_fs) / denom_s)

            # Scatter back to global user indices
            k_globals = valid[sorted_idx]           # global user indices in SIC order
            rates[k_globals] = rates_s
            mis[k_globals]   = mis_s

        return rates, mis

    def _compute_reward_eq14(self, rates, mis):
        # Per-UAV reward as in paper Eq(14)
        rewards = torch.zeros((params.NUM_UAVS,1), device=self.device)
        for u in range(params.NUM_UAVS):
            user_indices = self.clusters[u]
            mask = (user_indices != -1)
            valid = user_indices[mask]
            if len(valid) == 0:
                rewards[u,0] = 0.0
                continue

            rates_u = rates[valid]
            mis_u = mis[valid]

            total_td = torch.sum(params.PACKET_SIZE_BITS / (rates_u + 1e-12))
            comm_term = self.alpha / (total_td * params.TAU_SCALE + 1e-12)
            sens_term = (1.0 - self.alpha) * (torch.sum(mis_u) / params.I_SCALE)

            # penalty exponent: if any user violates R_TH or I_TH
            violation = ((rates_u < params.R_TH) | (mis_u < params.I_TH)).any().item()
            penalty = 1 if violation else 0

            reward = (comm_term + sens_term) / (2.0 ** penalty)
            # Soft clip: prevents geometry-driven spikes (UAV directly above user)
            # from destabilising critic. Median reward ~2; max 10 leaves headroom.
            reward = torch.clamp(reward, max=params.REWARD_CLIP)
            rewards[u,0] = reward

        return rewards

    def step(self, tensordict):
        action = tensordict['action']  # expected shape (NUM_UAVS, ACTION_DIM)

        # map velocity [0,1] -> [-UAV_VEL_MAX, UAV_VEL_MAX]
        vel_raw = action[:, :3]
        vel = (vel_raw * 2.0 - 1.0) * params.UAV_VEL_MAX

        power_raw = action[:, 3:]
        power_ratios = torch.softmax(power_raw, dim=-1)
        power_linear = power_ratios * params.P_MAX_LINEAR

        # stage-specific constraints
        if self.current_stage == 1:
            vel[:,2] = 0.0
            power_linear = torch.full_like(power_linear, params.P_MAX_LINEAR / params.K_MAX)

        # update positions
        self.uav_pos = self.uav_pos + vel
        self.uav_pos[:, :2] = torch.clamp(self.uav_pos[:, :2], 0.0, params.AREA_SIZE)
        self.uav_pos[:, 2] = torch.clamp(self.uav_pos[:, 2], params.UAV_ALT_MIN, params.UAV_ALT_MAX)

        # user mobility
        if self.mobility_factor > 0:
            step = (torch.rand((params.NUM_USERS,2), device=self.device) * 2 - 1)
            step = step * (params.USER_VEL_MAX * self.mobility_factor)
            self.user_pos = torch.clamp(self.user_pos + step, 0.0, params.AREA_SIZE)

        # recompute clusters periodically: simple reassign every step for now
        self.clusters = self._dwahc_clustering(self.uav_pos, self.user_pos)

        rates, mis = self.compute_isac_metrics(self.uav_pos, self.user_pos, power_linear, self.clusters)

        rewards = self._compute_reward_eq14(rates, mis)

        obs = self._get_obs()
        out = TensorDict({
            'uav_pos': self.uav_pos,
            'user_pos': self.user_pos,
            'clusters': self.clusters,
            'rates': rates.unsqueeze(0).expand(params.NUM_UAVS, -1),
            'mis': mis.unsqueeze(0).expand(params.NUM_UAVS, -1),
            'observation': obs['observation'],
            'global_state': obs['global_state'],
            'reward': rewards,
            'done': torch.zeros((params.NUM_UAVS,1), dtype=torch.bool, device=self.device),
            'terminated': torch.zeros((params.NUM_UAVS,1), dtype=torch.bool, device=self.device),
        }, batch_size=[])

        return out

    def _set_seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)