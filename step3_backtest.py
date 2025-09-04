# step3_backtest.py
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from step1_fred import *
from step2_signal import *

# --- constants (rough; replace later when you add CTD/CF) ---
DV01 = {"TU": 20.0, "FV": 45.0, "TY": 85.0}  # $ per bp per contract
DV01_TARGET_GROSS = 10_000.0                 # target $/bp across legs

ENTRY_Z, EXIT_Z, LB = 2.4, 0.5, 252

def dv01_neutral_weights(dv2, dv5, dv10):
    # Solve: w2 + w5 + w10 = 0, dv2*w2 + dv5*w5 + dv10*w10 = 0, set w5=+1; L1-normalize
    A = np.array([[1.0, 1.0], [dv2, dv10]])
    b = np.array([-1.0, -dv5])
    w2, w10 = np.linalg.solve(A, b)
    w = np.array([w2, 1.0, w10])
    return w / np.sum(np.abs(w))  # L1 norm = 1

y = pd.read_csv("yields.csv", parse_dates=["date"], index_col="date")
fly = 2*y["y5"] - y["y2"] - y["y10"]
z = zscore(fly)
side = band_signal(z)

# constant DV01s → same weights every day (fine for now)
w2, w5, w10 = dv01_neutral_weights(DV01["TU"], DV01["FV"], DV01["TY"])
W = pd.DataFrame({"TU": w2, "FV": w5, "TY": w10}, index=y.index)

# contracts per leg (integer), signed by side and weight, scaled by DV01 target
dv = pd.DataFrame({k:[v]*len(y) for k,v in DV01.items()}, index=y.index)
n = {}
for leg in ["TU","FV","TY"]:
    n[leg] = np.round(side * W[leg] * DV01_TARGET_GROSS / dv[leg]).astype(int)
contracts = pd.DataFrame(n, index=y.index)

# map cash yields to contract legs for Δy
dy = y.diff().rename(columns={"y2":"TU","y5":"FV","y10":"TY"}).fillna(0.0).mul(10_000)

# daily P&L (no costs yet):  dPnL ≈ − Σ (n_leg × DV01_leg × Δy_leg)
pnl = -(contracts * dv * dy).sum(axis=1)

# basic stats
ann = 252
mu = pnl.mean() * ann
sd = pnl.std(ddof=1) * (ann**0.5)
sharpe = mu / sd if sd > 0 else float("nan")

out = pd.concat({"side": side, "TU": contracts["TU"], "FV": contracts["FV"], "TY": contracts["TY"], "pnl": pnl}, axis=1)
out.to_csv("backtest.csv")
print(f"Days={len(pnl)}  CumPnL=${pnl.cumsum().iloc[-1]:.0f}  AnnSharpe={sharpe:.2f}")
print("Wrote backtest.csv (side, TU/FV/TY contracts, pnl).")

# Equity, drawdown, rolling Sharpe helpers
equity = pnl.cumsum()

def drawdown(pnl_series: pd.Series) -> pd.Series:
    eq = pnl_series.cumsum()
    peak = eq.cummax()
    return eq - peak  # $ drawdown

dd = drawdown(pnl)

# 63 trading days ~ 3 months
win = 252
roll_mu = pnl.rolling(win).mean() * 252
roll_sd = pnl.rolling(win).std(ddof=1) * np.sqrt(252)
roll_sr = roll_mu / roll_sd

# 1) Daily P&L
plt.figure(figsize=(12, 3.5))
plt.plot(pnl.index, pnl.values)
plt.title("Daily P&L ($)")
plt.xlabel("Date"); plt.ylabel("$")
plt.tight_layout()
# plt.savefig("pnl_daily.png", dpi=150)
plt.show()

# 2) Cumulative P&L (equity curve)
plt.figure(figsize=(12, 3.5))
plt.plot(equity.index, equity.values)
plt.title("Cumulative P&L ($)")
plt.xlabel("Date"); plt.ylabel("$")
plt.tight_layout()
# plt.savefig("pnl_cumulative.png", dpi=150)
plt.show()

# 3) Drawdown (usd)
plt.figure(figsize=(12, 3.5))
plt.plot(dd.index, dd.values)
plt.title("Drawdown ($)")
plt.xlabel("Date"); plt.ylabel("$")
plt.tight_layout()
# plt.savefig("drawdown.png", dpi=150)
plt.show()

# 4) Rolling 3-month Sharpe
plt.figure(figsize=(12, 3.5))
plt.plot(roll_sr.index, roll_sr.values)
plt.title("Rolling Sharpe (" + str(win) + "d window, annualized)")
plt.xlabel("Date"); plt.ylabel("Sharpe")
plt.tight_layout()
# plt.savefig("rolling_sharpe.png", dpi=150)
plt.show()