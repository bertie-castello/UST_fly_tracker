import pandas as pd
import matplotlib.pyplot as plt
from step1_fred import *

ENTRY_Z, EXIT_Z, LB = 2.0,0.5,252

def zscore(x: pd.Series, lb=LB) -> pd.Series:
    """Rolling z-score"""
    mu = x.rolling(lb, min_periods=lb//2).mean()
    sigma = x.rolling(lb, min_periods=lb//2).std(ddof=1) # Bessel's correction for unbiasedness
    z = (x - mu) / sigma
    return z

def band_signal(z: pd.Series, entry = ENTRY_Z, exit = EXIT_Z) -> pd.Series:
    # +1 long fly when cheap (z <= -entry), -1 short fly when rich
    pos, state = pd.Series(index = z.index, dtype = float), 0.0 # position series initialise
    for t, val in z.items():
        if state == 0.0:
            if val <= -entry: state = + 1.0
            elif val >= entry: state = -1.0
        elif state == +1.0 and val >= -exit: state = 0.0
        elif state == -1.0 and val <= exit: state = 0.0
        pos.loc[t] = state
    return pos.fillna(0.0) # add FFILL if vectorising

y = pd.read_csv("yields.csv", parse_dates=["date"], index_col="date")
fly = 2*y["y5"] - y["y2"] - y["y10"]
z = zscore(fly)
pos = band_signal(z)
out = pd.concat({"fly": fly, "z": z, "pos": pos}, axis=1)
out.to_csv("signal.csv")
print("Wrote signal.csv w columns fly, z, pos")

# rolling stats for context lines
mu = fly.rolling(LB, min_periods=LB//2).mean()
sd = fly.rolling(LB, min_periods=LB//2).std(ddof=1)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 1) Fly + rolling mean
axes[0].plot(fly.index, fly, label="Fly = 2*y5 − y2 − y10")
axes[0].plot(mu.index, mu, label=f"Rolling mean ({LB}d)")
axes[0].set_title("Fly and rolling mean")
axes[0].set_ylabel("Level (decimal)")
axes[0].legend(loc="best")

# 2) Z-score with bands and shaded positions
axes[1].plot(z.index, z, label="z-score")
axes[1].axhline(+ENTRY_Z, linestyle="--", linewidth=1)
axes[1].axhline(-ENTRY_Z, linestyle="--", linewidth=1)
axes[1].axhline(+EXIT_Z,  linestyle=":",  linewidth=1)
axes[1].axhline(-EXIT_Z,  linestyle=":",  linewidth=1)
axes[1].set_title("Z-score with entry/exit bands")
axes[1].set_ylabel("z")

# Shade long/short regimes (after lines so we know the y-limits)
ymin, ymax = axes[1].get_ylim()
axes[1].fill_between(z.index, ymin, ymax, where=(pos == 1.0), alpha=0.12, step="pre", label="LONG")
axes[1].fill_between(z.index, ymin, ymax, where=(pos == -1.0), alpha=0.12, step="pre", label="SHORT")
axes[1].legend(loc="best")

plt.tight_layout()
# Optional: save to file
# plt.savefig("step2_signal.png", dpi=150)
plt.show()


