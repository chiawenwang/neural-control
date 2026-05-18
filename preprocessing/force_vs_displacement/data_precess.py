import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV  = os.path.join(BASE_DIR, "ee_pose_force.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "ee_pose_force_filtered.csv")

# ── 1. load & extract columns ──────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df_slim = df[["time", "y", "Fz"]].copy()

# ── 2. remove static periods (start & end) ────────────────────────────────────
# "static" = y barely changes over a sliding window
WINDOW   = 10        # samples to look at together
Y_THRESH = 1e-4      # max abs change in y within window to be called "static"

# rolling std of y — low std means the robot isn't moving
rolling_std = df_slim["y"].rolling(window=WINDOW, center=True).std().fillna(0)
moving_mask = rolling_std > Y_THRESH

# keep only the contiguous moving block in the middle
# find first and last index where motion is detected
moving_indices = np.where(moving_mask)[0]
if len(moving_indices) == 0:
    raise RuntimeError("No motion detected — check Y_THRESH")

first_moving = moving_indices[0]
last_moving  = moving_indices[-1]

df_filtered = df_slim.iloc[first_moving : last_moving + 1].reset_index(drop=True)

# ── 3. keep only y < -0.1075 ──────────────────────────────────────────────────
# df_filtered = df_filtered[df_filtered["y"] < -0.1075].reset_index(drop=True)

print(f"Total rows       : {len(df_slim)}")
print(f"Rows after filter: {len(df_filtered)}")
print(f"Trimmed start    : {first_moving} rows")
print(f"Trimmed end      : {len(df_slim) - last_moving - 1} rows")

# ── 4. save filtered CSV ───────────────────────────────────────────────────────
df_filtered.to_csv(OUTPUT_CSV, index=False)
print(f"Saved → {OUTPUT_CSV}")

# ── 5. scatter plot: Fz vs y ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df_filtered["y"], df_filtered["Fz"], s=10, alpha=0.6, color="steelblue")
ax.set_xlabel("y  (m)")
ax.set_ylabel("Fz  (N)")
ax.set_title("Force Z vs Y displacement  (static periods removed)")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "Fz_vs_y.png"), dpi=150)
plt.show()
print("Plot saved → Fz_vs_y.png")
