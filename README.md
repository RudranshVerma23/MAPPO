# NOMA-ISAC UAV Networks: MAPPO vs Dueling DQN

Repository for MAPPO and Dueling DQN-based trajectory and power allocation optimization in NOMA-ISAC UAV networks.

---

## Quick Start (5 minutes)

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train MAPPO (single model with default alpha=0.5)
python src/mappo_agent.py

# Train Dueling DQN (multi-agent)
python src/dueling_dqn_multiagent.py

# Train MAPPO for each alpha (0.0, 0.1, ..., 1.0) — recommended for fair comparison
python src/train_alpha_sweep.py --epochs 500 --horizon 64
```

### 3. Evaluate & Plot

```bash
# Alpha sweep (communication vs sensing tradeoff)
python sweeps/alpha_sweep.py --device cpu

# Power sweep (transmit power variation)
python sweeps/power_sweep.py --device cpu
```

**Results saved to:**
- CSV: `sweeps/alpha_sweep_results_fixed.csv`, `sweeps/power_sweep_results_fixed.csv`
- PNG: `alpha_sweep_fixed.png`, `power_sweep_fixed.png`

---

## Repository Structure

```
.
├── src/
│   ├── environment_v2.py          # NOMA-ISAC environment (paper-faithful)
│   ├── params.py                  # Paper constants (bandwidth, thresholds, etc.)
│   ├── mappo_agent.py             # MAPPO trainer (Actor + Critic, PPO+GAE)
│   ├── dueling_dqn_multiagent.py  # Multi-agent Dueling DQN
│   ├── train_alpha_sweep.py       # Train MAPPO per alpha value
│   └── curriculum.py              # Curriculum learning stages
│
├── sweeps/
│   ├── alpha_sweep.py             # Evaluate across alpha: 0.0–1.0 (comm vs sensing)
│   ├── power_sweep.py             # Evaluate across Pmax: 7.5–22.5 dBm
│   └── *_results*.csv             # CSV results from sweep runs
│
├── models/                        # Trained checkpoints (auto-created)
│   ├── mappo_final.pth
│   ├── dqn_multiagent_uav*.pth
│   └── alpha_*/mappo_final.pth    # Per-alpha models from train_alpha_sweep
│
├── logs/                          # Training metrics (auto-created)
│   ├── mappo_train_log.csv
│   └── dqn_multiagent_train_log.csv
│
└── README.md
```

---

## Training Workflows

### Option A: Quick Single-Model Training (⏱️ ~30 min per model)

Train one MAPPO and one DQN with default alpha=0.5:

```bash
python src/mappo_agent.py          # → models/mappo_final.pth
python src/dueling_dqn_multiagent.py  # → models/dqn_multiagent_uav*.pth
```

**Pros:** Fast, low overhead  
**Cons:** Only evaluates at one alpha value

---

### Option B: Fair Comparison (Per-Alpha Training) (⏱️ ~5 hours)

Train separate MAPPO for each alpha (recommended for accurate sweeps):

```bash
python src/train_alpha_sweep.py --epochs 500 --horizon 64 --device cpu
```

This trains 11 models (alpha = 0.0, 0.1, ..., 1.0) and saves them to:
- `models/alpha_0.00/mappo_final.pth`
- `models/alpha_0.10/mappo_final.pth`
- ... etc.

Also trains DQN (once) alongside.

**Pros:** Fair comparison; MAPPO policy is optimized for each alpha  
**Cons:** Longer training time

---

## Evaluation

After training, run sweeps to generate comparison plots:

### Alpha Sweep
Varies alpha (communication weight) from 0.0 to 1.0:
```bash
python sweeps/alpha_sweep.py --device cpu
```

**Output:**
- `sweeps/alpha_sweep_results_fixed.csv` — metrics per alpha (TD, MI, Reward)
- `alpha_sweep_fixed.png` — plots of TD/MI/Reward vs alpha

### Power Sweep
Varies transmit power (Pmax) from 7.5 to 22.5 dBm:
```bash
python sweeps/power_sweep.py --device cpu
```

**Output:**
- `sweeps/power_sweep_results_fixed.csv` — metrics per power level
- `power_sweep_fixed.png` — plots of TD/MI/Reward vs Pmax

---

## Advanced Options

### Custom Training Parameters

```bash
# MAPPO with custom epochs and horizon
python src/mappo_agent.py --epochs 1000 --horizon 128

# Per-alpha sweep with custom settings
python src/train_alpha_sweep.py --epochs 500 --horizon 64 --device gpu

# Sweeps with GPU evaluation
python sweeps/alpha_sweep.py --device cuda
```

### Single-Step Smoke Test

Quickly verify setup works:

```bash
python src/train_alpha_sweep.py --epochs 1 --horizon 1
```

This trains for 1 epoch with 1-step episodes. Should complete in ~10 sec.

---

## Key Metrics

| Metric | Unit | Definition |
|--------|------|------------|
| **TD** | seconds | Sum of packet transmission times across all users |
| **MI** | bpcu | Conditional mutual information (sensing capability) |
| **Reward** | - | Scaled combination: `alpha * (1/TD) + (1-alpha) * MI` |

- **Lower TD** = faster communication ✓
- **Higher MI** = better sensing capability ✓
- **Higher Reward** = better overall performance ✓

---

## Paper Parameters

| Parameter | Value |
|-----------|-------|
| Bandwidth | 4 MHz |
| Noise Power | -100 dBm |
| Max TX Power | 29 dBm |
| Rate Threshold (R_TH) | 0.15 Mbps |
| Sensing Threshold (I_TH) | 10 bps |
| Communication Weight (alpha) | 0.5 (default) |
| Area Size | 500 m² |
| UAV Altitude | 20–150 m |
| UAV Velocity | 5 m/s |
| User Mobility | 0.5 m/s |

---

## Algorithms

### MAPPO (Multi-Agent Proximal Policy Optimization)
- **Action Space:** Continuous (Beta distribution for power allocation)
- **Policy Update:** PPO with ε-clipping (ε=0.2)
- **Advantage:** Generalized Advantage Estimation (GAE, λ=0.95)
- **Entropy Bonus:** 0.15 (for exploration)
- **Training:** 150+ epochs with curriculum learning

### Dueling DQN (Multi-Agent)
- **Action Space:** Discrete (7 directions × 3^K_MAX power levels)
- **Architecture:** Dueling Q-network (value stream + advantage stream)
- **Exploration:** ε-greedy (ε decays during training)
- **Training:** 150+ epochs with curriculum learning

---

## Important Notes

1. **Environment Consistency:** Both MAPPO and DQN use identical environment (`environment_v2.py`).
2. **Fair Evaluation:** For alpha sweeps, use **per-alpha trained models** from `train_alpha_sweep.py` — single model not recommended for fair comparison.
3. **Checkpoint Locations:** 
   - Single MAPPO: `models/mappo_final.pth`
   - Per-alpha MAPPO: `models/alpha_X.XX/mappo_final.pth`
   - Per-UAV DQN: `models/dqn_multiagent_uav{0,1,2}_final.pth`
4. **Deterministic Evaluation:** Sweeps use greedy policies (no exploration).
5. **Output Logs:** Training logs saved to `logs/` (CSV format).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install -r requirements.txt` |
| `CUDA out of memory` | Use `--device cpu` flag or reduce epochs |
| Old checkpoints fail to load | Remove `models/*.pth` and retrain (architecture changed) |
| Sweeps show zero MAPPO results | Verify `models/mappo_final.pth` or `models/alpha_X.XX/` exist |
| Plots not saving | Check write permissions in repository root |

---

## Typical Full Run

```bash
# 1. Setup (1 minute)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Train (5 hours for per-alpha, ~30 min for single models)
python src/train_alpha_sweep.py --epochs 500 --horizon 64

# 3. Evaluate (10 minutes total)
python sweeps/alpha_sweep.py --device cpu
python sweeps/power_sweep.py --device cpu

# 4. View results
# - Open: alpha_sweep_fixed.png, power_sweep_fixed.png
# - Review: sweeps/alpha_sweep_results_fixed.csv, sweeps/power_sweep_results_fixed.csv
```

---

**Status:** Fully runnable. All training and evaluation scripts ready for direct execution.
