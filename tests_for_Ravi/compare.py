import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# 1) Sensor locations (meters)
# ==================================================
SENSORS = [

    
    (0.21, 0.19),
    (0.23, 0.19),
    (0.25, 0.19),
    (0.27, 0.19),
    (0.29, 0.19),
    (0.31, 0.19),
    (0.33, 0.19),
    (0.35, 0.19),
    (0.37, 0.19),
    (0.39, 0.19)
]

# SENSORS = [
#     (0.19, 0.21),
#     (0.19, 0.23),
#     (0.19, 0.25),
#     (0.19, 0.27),
#     (0.19, 0.29),
#     (0.19, 0.31),
#     (0.19, 0.33),
#     (0.19, 0.35),
#     (0.19, 0.37),
#     (0.19, 0.39)
# ]

# ==================================================
# 2) CSV files and time-column names
# ==================================================
csv_ref = "T_Tamb_293d15_Tinit_300.csv"
csv_res = "field_full.csv"

time_col_ref = "time"   # column name in reference CSV
time_col_res = "t_h"      # CHANGE if needed (e.g. "time")

# ==================================================
# 3) Read CSV files
# ==================================================
df_ref = pd.read_csv(csv_ref)
df_res = pd.read_csv(csv_res)

# avoid float-matching problems
for df in (df_ref, df_res):
    df["x_r"] = df["x"].round(6)
    df["y_r"] = df["y"].round(6)

# ==================================================
# 4) Extract temperature histories at sensor locations
# ==================================================
data_ref = {}
data_res = {}

for i, (xs, ys) in enumerate(SENSORS, start=1):
    xs_r = round(xs, 6)
    ys_r = round(ys, 6)

    g_ref = df_ref[
        (df_ref["x_r"] == xs_r) & (df_ref["y_r"] == ys_r)
    ].sort_values(time_col_ref)

    g_res = df_res[
        (df_res["x_r"] == xs_r) & (df_res["y_r"] == ys_r)
    ].sort_values(time_col_res)

    data_ref[f"S{i}"] = {
        "t_h": g_ref[time_col_ref].to_numpy() / 3600.0,
        "T": g_ref["T"].to_numpy(),
    }

    data_res[f"S{i}"] = {
        "t_h": g_res[time_col_res].to_numpy(),
        "T_K": g_res["T_K"].to_numpy(),
    }

# ==================================================
# 5) Plot comparison (single figure)
# ==================================================
fig, ax = plt.subplots(figsize=(9, 5))

for sid in data_ref.keys():
    ax.plot(
        data_ref[sid]["t_h"],
        data_ref[sid]["T"],
        "-*",
        linewidth=0.8,
        label=f"{sid} reference",
    )

    ax.plot(
        data_res[sid]["t_h"],
        data_res[sid]["T_K"],
        "-o",
        linewidth=0.8,
        label=f"{sid} results",
    )

ax.set_xlabel("Time [h]")
ax.set_ylabel("Temperature [K]")
ax.set_title("Temperature history at sensor locations")
ax.grid(True, alpha=0.3)

ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

plt.tight_layout()
plt.savefig("sensor_temperature_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
