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

# QoS Thresholds 
R_TH = 1.5e5  # 0.15 Mbps (150,000 bps) - more realistic for 4MHz
I_TH = 1.0    # 1.0 bps/Hz or equivalent sensing threshold

# Optimization Constants [cite: 143, 210]
ALPHA = 0.5  # Default weight for TD vs MI
TAU_SCALE = 1.23e-13  # Empirical scaling factor
I_SCALE = 1.67e3    # Empirical scaling factor

# System Setup
NUM_UAVS = 3
NUM_USERS = 15 # Total users across the map

# Network Dimensions
K_MAX = 5  # Max users per cluster for fixed-size NN input
OBS_DIM = 3 + 1 + NUM_UAVS + K_MAX  # (Pos, Dist, ID, Gains) = 12
ACTION_DIM = 3 + K_MAX              # (Velocity, Power Ratios) = 8
GLOBAL_STATE_DIM = NUM_UAVS * OBS_DIM # 3 * 12 = 36