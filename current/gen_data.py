# Synthetic Dive Profiles Dataset Generator
# Generates a clean CSV ready for logit/probit modeling.
# Variables include: MaxDepth (m), BottomTime (min), AscentRate (m/min), SafetyStop (0/1), GasMix,
# NDL estimates, NDL violations, plus a binary Incident label.

import numpy as np
import pandas as pd
from pathlib import Path
from math import exp

np.random.seed(42)

N = 2500  # number of dives

# --- Helper: approximate recreational NDL (No-Decompression Limits) by depth (rough, in minutes)
NDL_table = {
    12: 147, 18: 56, 21: 45, 24: 37, 27: 30,
    30: 20, 33: 16, 36: 14, 40: 9, 45: 5, 50: 3, 60: 2
}

def ndl_by_depth(depth_m):
    d = max(12, min(60, depth_m))
    keys = sorted(NDL_table.keys())
    for i in range(len(keys)-1):
        if keys[i] <= d <= keys[i+1]:
            d0, d1 = keys[i], keys[i+1]
            ndl0, ndl1 = NDL_table[d0], NDL_table[d1]
            # linear interpolation
            w = (d - d0) / (d1 - d0)
            return ndl0 * (1 - w) + ndl1 * w
    return NDL_table[keys[-1]]

# --- Simulate MaxDepth (m): mostly shallow, some mid/deep
comp = np.random.choice([0,1,2], size=N, p=[0.7, 0.23, 0.07])
depth = np.empty(N)
depth[comp==0] = np.random.uniform(10, 30, size=(comp==0).sum())
depth[comp==1] = np.random.uniform(30, 40, size=(comp==1).sum())
depth[comp==2] = np.random.uniform(40, 60, size=(comp==2).sum())
depth = np.round(depth, 1)

# --- Simulate BottomTime (min): shorter when deeper
base_time = np.random.normal(42, 10, size=N)
bt = base_time - 0.7*(depth - 12) + np.random.normal(0, 6, size=N)
bt = np.clip(bt, 6, 80)
bt = np.round(bt, 1)

# --- AscentRate (m/min)
ascent = np.random.normal(11.5, 2.8, size=N)
fast_idx = np.random.rand(N) < 0.12
ascent[fast_idx] += np.random.uniform(6, 12, size=fast_idx.sum())
ascent = np.clip(ascent, 5, 30)
ascent = np.round(ascent, 1)

# --- Safety stop probability
logit_p_ss = -0.3 + 0.05*(depth-18)/10 + 0.02*(bt-30)/10
p_ss = 1/(1+np.exp(-logit_p_ss))
safetystop = (np.random.rand(N) < p_ss).astype(int)

# --- Gas mix: Air vs Nitrox32
gas = np.array(["Air"]*N, dtype=object)
nitrox_prob = 0.15 + 0.35*np.exp(-((depth-26)/8)**2)  # bell curve around 26m
nitrox_prob = np.where(depth>40, nitrox_prob*0.2, nitrox_prob)
use_nx = np.random.rand(N) < np.clip(nitrox_prob, 0.02, 0.6)
gas[use_nx] = "Nitrox32"

# --- NDL violation flag
ndl = np.array([ndl_by_depth(d) for d in depth])
ndl_bonus = np.where(gas=="Nitrox32", 1.12, 1.0)
ndl_eff = ndl * ndl_bonus
ndl_violation = (bt > ndl_eff * (0.98 + 0.04*np.random.rand(N))).astype(int)

# --- Diver-level latent risk heterogeneity
diver_risk = np.random.normal(0, 0.6, size=N)

# --- Incident probability (logistic model)
z = (
    -8.0
    + 0.065*depth
    + 0.045*bt
    + 0.09*ascent
    + 0.020*np.maximum(0, depth-30)*(bt/30)
    + 0.8*ndl_violation
    - 0.5*safetystop
    - 0.35*(gas=="Nitrox32").astype(float)
    + diver_risk
)
p_inc = 1/(1+np.exp(-z))
incident = (np.random.rand(N) < p_inc).astype(int)

# --- Assemble DataFrame
df = pd.DataFrame({
    "DiveID": np.arange(1, N+1),
    "MaxDepth_m": depth,
    "BottomTime_min": bt,
    "AscentRate_m_per_min": ascent,
    "SafetyStop": safetystop,
    "GasMix": gas,
    "NDL_min_est": np.round(ndl_eff, 1),
    "NDL_Violation": ndl_violation,
    "Incident": incident,
    "Incident_Prob": np.round(p_inc, 4),
})

# --- Save to CSV
out_path = Path("diving_synthetic_profiles.csv")
df.to_csv(out_path, index=False)

print("Synthetic dataset saved:", out_path)
print(df.head())
