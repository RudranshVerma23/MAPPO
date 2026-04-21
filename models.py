import torch
from torch import nn
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torch.distributions import Beta, Independent
from tensordict.nn import TensorDictModule, TensorDictSequential
import params

class IndependentBeta(Independent):
    """
    Wraps the Beta distribution to treat the last dimension (ACTION_DIM) 
    as a single joint event, summing their log probabilities.
    """
    def __init__(self, concentration1, concentration0):
        super().__init__(Beta(concentration1, concentration0), 1)

class NomaPolicyMapping(nn.Module):
    """
    A dedicated module for mapping features to Beta distribution parameters.
    Ensures all sub-layers (like policy_head) are registered for GPU transfer.
    """
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        # Register the linear head as an attribute of the module
        self.policy_head = nn.Linear(64, action_dim * 2)

    def forward(self, features):
        logits = self.policy_head(features) 
        logits = logits.view(*logits.shape[:-1], self.action_dim, 2)
        # print(logits.mean())
        
        # Use Tanh to keep logits between -1 and 1, then scale
        # This prevents the "5000" explosion and keeps alpha/beta < 11.0
        params_val = torch.nn.functional.softplus(torch.tanh(logits) * 5.0) + 1.0
        return params_val[..., 0], params_val[..., 1]
    
def get_actor_critic(obs_dim, action_dim, global_state_dim, device="cpu"):
    # --- SHARED ACTOR ARCHITECTURE ---
    # Input: Local Obs + Agent ID 
    actor_net = nn.Sequential(
        nn.Linear(obs_dim, 256),
        nn.LayerNorm(256), # Added LayerNorm for training stability
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
    )

    # Output Heads for Beta Distribution (alpha and beta parameters)
    # We need 2 parameters (alpha, beta) for every action dimension
    # Velocity (3) + Power Ratios (5) = 8 action dims -> 16 outputs
    policy_head = nn.Linear(64, action_dim * 2)


    actor_module = TensorDictModule(
        nn.Sequential(actor_net, NomaPolicyMapping(action_dim)),
        in_keys=["observation"],
        out_keys=["concentration1", "concentration0"],
    )

    # Wrap as a Probabilistic Actor using the Beta Distribution
    # Beta is superior for NOMA power (strictly [0, 1])
    prob_actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["concentration1", "concentration0"],
        out_keys=["action"],
        distribution_class=IndependentBeta,
        return_log_prob=True,
    )

    # --- CENTRALIZED CRITIC ARCHITECTURE ---
    # Input: Global State (Concatenated observations of ALL agents)
    critic_net = nn.Sequential(
        nn.Linear(global_state_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1), # Outputs a single scalar V(S)
    )

    centralized_critic = ValueOperator(
        module=critic_net,
        in_keys=["global_state"],
    )

    return prob_actor.to(device), centralized_critic.to(device)