import torch
from torchrl.envs import EnvBase
from torchrl.data import (Unbounded, Bounded, Composite, Categorical)
from tensordict import TensorDict
import params
from typing import Optional


class NomaIsacEnv(EnvBase):
    def __init__(self, device="cpu", mobility_factor=params.DEFAULT_MOBILITY_FACTOR):
        super().__init__(device=device, batch_size=[])
        # self.device = device
        self.mobility_factor = mobility_factor
        self.alpha = 1.0
        self.current_stage = 1
        # 1. Define Observation Spec (Local + Global for CTDE) [cite: 159, 160]
        self.observation_spec = Composite({
            "observation": Unbounded(shape=(params.NUM_UAVS, params.OBS_DIM), device=self.device),
            "global_state": Unbounded(shape=(params.NUM_UAVS, params.GLOBAL_STATE_DIM), device=self.device),
            "rates": Unbounded(shape=(params.NUM_UAVS, params.NUM_USERS), device=self.device),
            "mis": Unbounded(shape=(params.NUM_UAVS, params.NUM_USERS), device=self.device),
            # CRITICAL: Add hidden states here
            "uav_pos": Unbounded(shape=(params.NUM_UAVS, 3), device=self.device),
            "user_pos": Unbounded(shape=(params.NUM_USERS, 2), device=self.device),
            "clusters": Unbounded(shape=(params.NUM_UAVS, params.K_MAX), dtype=torch.long, device=self.device),
        }, shape=[])

        # 2. Define Action Spec (Continuous Velocity + Power) [cite: 162, 164]
        self.action_spec = Composite({
            "action": Bounded(
                low=0, high=1, 
                shape=(params.NUM_UAVS, params.ACTION_DIM),
                device=self.device
            ),
            "sample_log_prob": Unbounded(
                shape=(params.NUM_UAVS,), # One probability scalar per UAV
                device=self.device
            )
        }, shape=[])

        # 3. Define Reward Spec: One reward per agent
        self.reward_spec = Unbounded(
            shape=(params.NUM_UAVS,1), 
            device=self.device
        )

        # 4. Define Done Spec: One flag per agent
        # We use a Composite spec for done/terminated/truncated in modern TorchRL
        self.done_spec = Composite({
            "done": Categorical(
                n=2, 
                shape=(params.NUM_UAVS,1), 
                dtype=torch.bool, 
                device=self.device
            ),
            "terminated": Categorical(
                n=2, 
                shape=(params.NUM_UAVS,1), 
                dtype=torch.bool, 
                device=self.device
            )
        }, shape=[])

    def _get_obs(self, uav_pos, user_pos, clusters):
        obs_list = []
        g_all, _ = self.calculate_channel_gains(uav_pos, user_pos)
        
        for u in range(params.NUM_UAVS):
            my_pos = uav_pos[u]
            agent_id = torch.zeros(params.NUM_UAVS, device=self.device)
            agent_id[u] = 1.0
            
            # Handle padding for centroid calculation
            user_indices = clusters[u]
            user_indices = user_indices[user_indices != -1]
            
            assigned_users = user_pos[user_indices]
            centroid = assigned_users.mean(dim=0) if len(assigned_users) > 0 else my_pos[:2]
            dist_to_cluster = torch.norm(my_pos[:2] - centroid).reshape(1)
            
            norm_pos = my_pos / params.AREA_SIZE
            norm_dist = dist_to_cluster / params.AREA_SIZE

            # Local Channel Gains (Padded to K_MAX for consistent NN input size)
            local_gains = torch.zeros(params.K_MAX, device=self.device)
            if len(user_indices) > 0:
                # This keeps values typically between -5 and 5
                local_gains[:len(user_indices)] = torch.log10(g_all[u, user_indices] + 1e-12)
            
            local_obs = torch.cat([norm_pos, norm_dist, agent_id, local_gains])
            obs_list.append(local_obs)
            
        return torch.stack(obs_list)

    def _reset(self, tensordict=None):
        uav_pos = torch.rand((params.NUM_UAVS, 3), device=self.device) * params.AREA_SIZE
        # Uniformly random between min and max
        uav_pos[:, 2] = torch.distributions.Uniform(params.UAV_ALT_MIN, params.UAV_ALT_MAX).sample(uav_pos[:, 2].shape)
        user_pos = torch.rand((params.NUM_USERS, 2), device=self.device) * params.AREA_SIZE
        
        # Initialize Padded Clusters Tensor: (UAVs, K_MAX) filled with -1
        clusters = torch.full((params.NUM_UAVS, params.K_MAX), -1, dtype=torch.long, device=self.device)
        
        # Initial Clustering: Nearest Neighbor logic [cite: 9, 107]
        dist_matrix = torch.norm(user_pos.unsqueeze(1) - uav_pos[:, :2].unsqueeze(0), dim=-1)
        nearest_uavs = torch.argmin(dist_matrix, dim=1)
        
        for u in range(params.NUM_UAVS):
            indices = (nearest_uavs == u).nonzero(as_tuple=True)[0]
            # Safeguard: only take up to K_MAX users
            indices = indices[:params.K_MAX] 
            clusters[u, :len(indices)] = indices

        obs = self._get_obs(uav_pos, user_pos, clusters)
        # Expand from [1, 36] to [3, 36]
        global_state = obs.view(-1).expand(params.NUM_UAVS, -1)
        
        out = TensorDict({
            "uav_pos": uav_pos,
            "user_pos": user_pos,
            "clusters": clusters, # Now a proper device-aware Tensor
            "observation": obs,
            "global_state": global_state,
            "rates": torch.zeros((params.NUM_UAVS, params.NUM_USERS), device=self.device),
            "mis": torch.zeros((params.NUM_UAVS, params.NUM_USERS), device=self.device),
        }, batch_size=[])
        return out
    
    def calculate_channel_gains(self, uav_pos, user_pos):
        """
        Implements Equation (1): g = H * 10^(-PL/10)[cite: 71].
        """
        # 1. Euclidean Distance Matrix (UAVs, Users)
        # uav_pos: (U, 3), user_pos: (K, 2) -> expand to (U, K, 3)
        diff = uav_pos.unsqueeze(1) - torch.cat([user_pos, torch.zeros(params.NUM_USERS, 1, device=self.device)], dim=-1).unsqueeze(0)
        dist = torch.norm(diff, dim=-1) # Shape: (U, K)

        # 2. Path Loss Model (3GPP Release 15) [cite: 69, 72]
        # Simple log-distance model as a baseline for PL_k^u(t)
        pl = 20 * torch.log10(dist) + 20 * torch.log10(torch.tensor(2.4e9 * 4 * torch.pi / 3e8)) # FSPL at 2.4GHz
        
        # 3. Small-scale Rayleigh Fading [cite: 72]
        # Sampled from CN(0, 1) to capture rapid signal fluctuations
        h_real = torch.randn_like(dist) * torch.sqrt(torch.tensor(0.5))
        h_imag = torch.randn_like(dist) * torch.sqrt(torch.tensor(0.5))
        h_coeff = h_real**2 + h_imag**2 # |H|^2 power gain

        # 4. Final Channel Gain g_k^u(t)
        g_linear = h_coeff * torch.pow(10, -pl / 10)
        return g_linear, dist

    def compute_isac_metrics(self, uav_pos, user_pos, power_alloc, clusters):
        """
        Implements Dynamic SIC Sorting (Eq 6) and Sensing MI (Eq 9).
        """
        g_all, _ = self.calculate_channel_gains(uav_pos, user_pos)
        
        total_rates = torch.zeros(params.NUM_USERS, device=self.device)
        total_mi = torch.zeros(params.NUM_USERS, device=self.device)
        
        for u in range(params.NUM_UAVS):
            # 1. Filter out padding to keep tensors aligned
            user_indices_full = clusters[u]
            mask = (user_indices_full != -1)
            valid_user_indices = user_indices_full[mask]
            
            if len(valid_user_indices) == 0: continue
            
            # 2. Calculate Inter-cluster Interference for valid users only
            inter_interference = torch.zeros(len(valid_user_indices), device=self.device)
            for s in range(params.NUM_UAVS):
                if s == u: continue
                # Interference from UAV s to UAV u's valid users
                inter_interference += (g_all[s, valid_user_indices] * power_alloc[s].sum())

            # 3. Dynamic SIC Decoding Order
            # Equivalent channel gain G = g / (Interference + Noise)
            g_equiv = g_all[u, valid_user_indices] / (inter_interference + params.NOISE_POWER_LINEAR)
            
            # Sort valid users in ascending order: G(1) <= G(2) ...
            sorted_indices = torch.argsort(g_equiv)
            
            # Extract power for this cluster (only the non-padded slice)
            cluster_power = power_alloc[u][mask]
            
            # 4. Calculate SINR, Rates, and MI
            for idx, k_local in enumerate(sorted_indices):
                # Map back to the global user ID using the VALID indices list
                k_global = valid_user_indices[k_local]
                
                # Intra-cluster interference: users with higher equivalent gains
                higher_gain_indices = sorted_indices[idx + 1:]
                intra_interference = (g_all[u, k_global] * cluster_power[higher_gain_indices]).sum()
                
                # SINR (Eq 6)
                denom = intra_interference + inter_interference[k_local] + params.NOISE_POWER_LINEAR
                sinr = (g_all[u, k_global] * cluster_power[k_local]) / denom
                
                # Rate R (Eq 7)
                total_rates[k_global] = params.BANDWIDTH * torch.log2(1 + sinr)

                # Conditional Mutual Information (Eq 9)
                mi_denom = intra_interference + inter_interference[k_local] + params.NOISE_POWER_LINEAR
                mi_val = 0.5 * params.SUB_CARRIER_INTERVAL * torch.log2(
                    1 + (cluster_power[k_local] * (params.SYMBOL_DURATION**2) * params.SUB_CARRIER_INTERVAL) / mi_denom
                )
                total_mi[k_global] = mi_val

        return total_rates, total_mi
    
    def _compute_reward(self, rates, mis):
        # STAGE 1 & 2: Optimize SUM-RATE (Smooth Gradient)
        if self.current_stage <= 2:
            # Normalize by BANDWIDTH to keep magnitude around 1.0-10.0
            objective = 100*torch.sum(rates) / (params.BANDWIDTH * params.NUM_UAVS)
            
        # STAGE 3 & 4: Optimize TD & MI (Fine-Tuning) [cite: 180, 183]
        else:
            total_td = torch.sum(1.0 / (rates + 1e-9))
            comm_term = self.alpha * (1.0 / (total_td * params.TAU_SCALE + 1e-9))
            sens_term = (1.0 - self.alpha) * (torch.sum(mis) / params.I_SCALE)
            objective = comm_term + sens_term

        # Use the exponential penalty we built [cite: 185]
        rate_gap = torch.clamp(params.R_TH - rates.min(), min=0) / params.R_TH
        soft_penalty = torch.exp(-rate_gap)
        
        return (objective * soft_penalty).view(-1).expand(params.NUM_UAVS, 1)
    
    # def _compute_reward(self, rates, mis):
    #     """
    #     Implements Equation (14).
    #     Calculates the system-wide reward with QoS-triggered halving.
    #     """
    #     total_td = torch.sum(1.0 / (rates + 1e-9))
    #     total_mi = torch.sum(mis)
    #     comm_term = self.alpha * (1.0 / (total_td * params.TAU_SCALE + 1e-12))
    #     sens_term = (1.0 - self.alpha) * (total_mi / params.I_SCALE)
    #     weighted_objective = comm_term + sens_term
    #     # violation_rate = (rates < params.R_TH).any()
    #     # violation_mi = (mis < params.I_TH).any()
    #     # penalty_exponent = 1.0 if (violation_rate or violation_mi) else 0.0
    #     # reward = weighted_objective / (2.0 ** penalty_exponent)

    #     rate_gap = torch.clamp(params.R_TH - rates.min(), min=0) / params.R_TH
    #     mi_gap = torch.clamp(params.I_TH - mis.min(), min=0) / params.I_TH

    #     # soft_penalty = 1.0 - 0.5 * (rate_gap + mi_gap) # Penalty factor between 0.5 and 1.0       

    #     penalty_rate = torch.exp(-rate_gap)
    #     penalty_mi = torch.exp(-mi_gap)

    #     # In Stage 1 & 2, only penalize for communication
    #     if self.alpha > 0.9:
    #         soft_penalty = penalty_rate
    #     else:
    #         soft_penalty = penalty_rate * penalty_mi


    #     reward = weighted_objective * soft_penalty
    #     # print(f"DEBUG: Distance Min/Max: {dist.min().item():.2f} / {dist.max().item():.2f}")
    #     # print(f"DEBUG: Avg Rate: {rates.mean().item():.2e}")
    #     # print(f"DEBUG: Comm Term: {comm_term.item():.2e}")
    #     print(f"DEBUG: weighted_obj: {weighted_objective.item():.5e} soft_penalty: {soft_penalty.item():.5e}")
    #     return reward.view(-1).expand(params.NUM_UAVS,1)
    
    def _step(self, tensordict):
        
        # 1. Parse Consolidated Action
        # action shape: (NUM_UAVS, 8) where [0:3] is Velocity and [3:8] is Power
        action = tensordict["action"]
        
        # 2. Map Velocity: [0, 1] -> [-1, 1] -> [-5, 5] m/s
        vel_raw = action[:, :3]
        vel_scaled = (vel_raw * 2.0 - 1.0) * params.UAV_VEL_MAX
        
        # 3. Map NOMA Power: [0, 1] range, then Softmax for cluster-wide sum constraint
        power_raw = action[:, 3:]
        power_ratios = torch.softmax(power_raw, dim=-1) # Ensures Eq (12f) is met
        power_linear = power_ratios * params.P_MAX_LINEAR

        # 4. Update UAV Positions
        if self.current_stage == 1:
            vel_scaled[:, 2] = 0.0  # Zero out altitude change
            power_linear = torch.full_like(power_linear, params.P_MAX_LINEAR / params.K_MAX) # Equal power split
        uav_pos = tensordict["uav_pos"] + vel_scaled
        uav_pos[:, 0:2] = torch.clamp(uav_pos[:, 0:2], 0, params.AREA_SIZE)
        uav_pos[:, 2] = torch.clamp(uav_pos[:, 2], params.UAV_ALT_MIN, params.UAV_ALT_MAX)
        # 5. Update User Positions (The Mobility Logic) 
        user_pos = tensordict["user_pos"]
        # This allows for smooth curriculum annealing
        if self.mobility_factor > 0:
            # Generate random walk direction
            user_move = (torch.rand((params.NUM_USERS, 2), device=self.device) * 2 - 1) 
            # Apply scaled velocity: factor * 0.5 m/s
            user_move = user_move * (params.USER_VEL_MAX * self.mobility_factor)
            user_pos = torch.clamp(user_pos + user_move, 0, params.AREA_SIZE)
        # 6. Execute Physics Engine (Dynamic SIC & ISAC Metrics)
        clusters = tensordict["clusters"] 
        rates, mis = self.compute_isac_metrics(uav_pos, user_pos, power_linear, clusters)
        expanded_rates = rates.unsqueeze(0).expand(params.NUM_UAVS, -1)
        expanded_mis = mis.unsqueeze(0).expand(params.NUM_UAVS, -1)
        # 7. Compute Reward and Next State [cite: 183]
        reward = self._compute_reward(rates, mis)
        next_obs = self._get_obs(uav_pos, user_pos, clusters)
        # Global state for Centralized Critic
        global_state = next_obs.view(1, -1).expand(params.NUM_UAVS, -1)
        
        # print(vel_scaled.abs().mean().item())
        out = TensorDict({
            "uav_pos": uav_pos,
            "user_pos": user_pos,
            "rates": expanded_rates,
            "mis": expanded_mis,
            "clusters": clusters,
            "observation": next_obs,
            "global_state": global_state,
            "reward": reward,
            "done": torch.zeros((params.NUM_UAVS, 1), dtype=torch.bool, device=self.device),
            "terminated": torch.zeros((params.NUM_UAVS, 1), dtype=torch.bool, device=self.device),
        }, batch_size=self.batch_size)
        
        return out

    def _set_seed(self, seed: Optional[int] = None):
        """
        Sets the seed for reproducibility. 
        Required by torchrl.envs.EnvBase.
        """
        if seed is not None:
            torch.manual_seed(seed)
            # If you use numpy for anything in the future:
            # np.random.seed(seed)
        self.seed = seed