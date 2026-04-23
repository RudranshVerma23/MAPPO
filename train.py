import torch
from torch import optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyTensorStorage
from torchrl.objectives import PPOLoss
from torchrl.objectives.value import GAE

import params
from environment import NomaIsacEnv
from models import get_actor_critic
from curriculum import CurriculumManager # <--- Step 1: New Import

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize Environment
    env = NomaIsacEnv(device=device)
    
    # Step 2: Initialize Curriculum Manager
    # This director will monitor 'env' and change its attributes (alpha, mobility)
    manager = CurriculumManager(env)
    
    prob_actor, centralized_critic = get_actor_critic(
        params.OBS_DIM, 
        params.ACTION_DIM, 
        params.GLOBAL_STATE_DIM, 
        device=device
    )

    collector = SyncDataCollector(
        env,
        prob_actor,
        frames_per_batch=2048,
        total_frames=10_000_000,
        reset_at_each_iter=False,
        split_trajs=False,
        device=device,
        storing_device=device,
    )

    sampler = SamplerWithoutReplacement()
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=2048, device=device),
        sampler=sampler,
        batch_size=256,
    )

    loss_module = PPOLoss(
        actor_network=prob_actor,
        critic_network=centralized_critic,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coeff=0.05,
        loss_critic_type="l2"
    )

    loss_module.set_keys(
        reward="reward",
        action="action", 
        sample_log_prob="sample_log_prob",
        value="state_value",
    )

    adv_module = GAE(gamma=0.99, lmbda=0.95, value_network=centralized_critic)
    # optimizer = optim.Adam(loss_module.parameters(), lr=3e-4)
    actor_optimizer = optim.Adam(prob_actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(centralized_critic.parameters(), lr=5e-4)
    prev_batch_last_uav0 = None
    # --- THE TRAINING LOOP ---
    for i, data in enumerate(collector):
        
        # Step 3: Extract Performance Metrics for the Curriculum
        # We look at the 'next' state to see how the UAVs performed after their actions
        with torch.no_grad():
            avg_reward = data["next", "reward"].mean().item()
            
            # Extract Rate and MI tensors from the physics engine output
            rates = data["next", "rates"] 
            mis = data["next", "mis"]
            
            # Step 4: Calculate QoS Satisfaction Rate
            # Following Eq (12h) and (12i): R >= R_th and I >= I_th [cite: 139, 140]
            qos_met = (rates >= params.R_TH) & (mis >= params.I_TH)
            qos_satisfaction = qos_met.float().mean().item()

        # Step 5: Update Curriculum Stage
        # The manager will check if we hit a plateau and met the 90% QoS goal
        manager.update_stage(avg_reward, qos_satisfaction)

        # --- A. Calculate Advantages (The Critic's Job) ---
        with torch.no_grad():
            adv_module(data)

        # --- B. Update the Networks ---
        replay_buffer.extend(data)
        for _ in range(20):
            for mini_batch in replay_buffer:
                loss_vals = loss_module(mini_batch)
                # loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                
                actor_loss = loss_vals["loss_objective"] + loss_vals["loss_entropy"]
                critic_loss = loss_vals["loss_critic"]

                # --- Update Actor ---
                actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)  # keep graph for critic
                torch.nn.utils.clip_grad_norm_(prob_actor.parameters(), 0.5)
                actor_optimizer.step()

                # --- Update Critic ---
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(centralized_critic.parameters(), 0.5)
                critic_optimizer.step()

                # optimizer.zero_grad()
                # loss_value.backward()
                # torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
                # optimizer.step()

        # Calculate Power Entropy here since 'data' contains the actions
        power_raw = data["action"][..., 3:]
        power_ratios = torch.softmax(power_raw, dim=-1)
        
        p = power_ratios + 1e-9
        entropy = -torch.sum(p * torch.log(p), dim=-1).mean()

        # Continuity diagnostics: verify s_{t+1} == next s_t for UAV 0.
        uav0_start = data["uav_pos"][0, 0]
        uav0_end = data["next", "uav_pos"][-1, 0]
        within_batch_err = (data["uav_pos"][1:, 0] - data["next", "uav_pos"][:-1, 0]).abs().max().item()
        cross_batch_err = float("nan") if prev_batch_last_uav0 is None else (uav0_start - prev_batch_last_uav0).abs().max().item()
        prev_batch_last_uav0 = uav0_end.detach().clone()

        print(f"Batch {i} | UAV0 Start Pos(t=0): {uav0_start}")
        print(f"Batch {i} | UAV0 End Pos(t=T): {uav0_end}")
        print(f"Batch {i} | Continuity | within_batch_max_err={within_batch_err:.3e} | cross_batch_max_err={cross_batch_err:.3e}")
        # --- C. Logging Stage-Aware Performance ---
        if i % 10 == 0:
        
            print(f"Batch {i} | Stage: {manager.current_stage} | Reward: {avg_reward:.4f} | QoS: {qos_satisfaction*100:.1f}% | Power Entropy: {entropy.item():.4f}")

if __name__ == "__main__":
    train()