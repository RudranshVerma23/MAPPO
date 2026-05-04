# Environment Constraints
AREA_SIZE = 500.0  # 500m x 500m area [cite: 209]
UAV_ALT_MIN = 20.0  # 
UAV_ALT_MAX = 150.0  # 
UAV_VEL_MAX = 5.0  # 5 m/s 
USER_VEL_MAX = 0.5  # 0.5 m/s 
# Curriculum Control Defaults
DEFAULT_MOBILITY_FACTOR = 0.0  # Starts at 0.0 for Stage 1-3

# Wireless & Sensing Params 
BANDWIDTH = 4e6  # 4 MHz
NOISE_POWER_DBM = -100.0 
NOISE_POWER_LINEAR = 10**(NOISE_POWER_DBM / 10) * 1e-3 # Convert dBm to Watts
SYMBOL_DURATION = 5e-6  # 5 us
SUB_CARRIER_INTERVAL = 0.25e6  # 0.25 MHz
P_MAX_DBM = 29.0
P_MAX_LINEAR = 10**(P_MAX_DBM / 10) * 1e-3 # ~0.8 Watts

# Delay modeling
# Packet size used in TD = L / R. Units: bits.
PACKET_SIZE_BITS = 1e5

# QoS Thresholds
R_TH = 1.5e5   # 0.15 Mbps (150 kbps) - achievable at ~4MHz with SINR>0.03
I_TH = 5e-7    # sensing threshold (bpcu). Calibrated to ~50% of median MI per user
               # after MI formula fix (removed spurious fs multiplier, added channel gain g).

# Optimization Constants [recalibrated after MI formula fix — see diagnostic_physical_stats.csv]
ALPHA = 0.5  # Default comm/sensing tradeoff weight. Also passed into observation.
# TAU_SCALE: so comm_term = alpha/(total_td*TAU_SCALE) ~ 1.0 at median random-policy TD
# Diagnostic median TD: 1.3-1.9 s → TAU_SCALE = alpha/median_td ~ 0.32
TAU_SCALE = 3.20e-1
# I_SCALE: so sens_term = (1-alpha)*(total_mi/I_SCALE) ~ 1.0 at median MI per cluster
# Diagnostic median total_mi: 5-7e-6 bpcu → I_SCALE = (1-alpha)*median_mi ~ 3.07e-6
I_SCALE = 3.07e-6
# Reward clip: prevents geometry spikes (UAV directly above user → TD~0 → reward→∞)
# from destabilising critic. Median reward ~2, so clip at 10 gives 5x headroom.
REWARD_CLIP = 10.0

# System Setup
NUM_UAVS = 3
NUM_USERS = 15  # Total users across the map

# Network Dimensions
K_MAX = 5   # Max users per cluster (fixed-size NN input)
# OBS: [norm_pos(3), dist_centroid(1), agent_id(NUM_UAVS), local_gains(K_MAX), alpha(1)]
# Alpha appended so policy can condition behavior on comm/sensing weight (needed for sweep).
OBS_DIM = 3 + 1 + NUM_UAVS + K_MAX + 1   # = 13
ACTION_DIM = 3 + K_MAX                    # (Velocity xyz, Power Ratios K_MAX) = 8
GLOBAL_STATE_DIM = NUM_UAVS * OBS_DIM     # 3 * 13 = 39