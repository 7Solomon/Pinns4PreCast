import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("field_full.csv")

# target locations (what you want)
LOCATIONS = [
    (0.2, 0.0, 0.2),
    (0.4, 0.0, 0.0),
    (0.2, 0.1, 0.2),
    (0.2, 0.2, 0.2),
    (0.2, 0.4, 0.0),
    (0.2, 0.4, 0.1),
    (0.2, 0.4, 0.2),
    (0.2, 0.4, 0.3),
    (0.2, 0.4, 0.4),
    (0.4, 0.4, 0.2)
]

# ---------------- Temperature figure ----------------
plt.figure()

for i, (x0, y0, z0) in enumerate(LOCATIONS, start=1):
    d2 = (df["x"] - x0)**2 + (df["y"] - y0)**2 + (df["z"] - z0)**2
    i0 = d2.idxmin()

    x_closest = float(df.loc[i0, "x"])
    y_closest = float(df.loc[i0, "y"])
    z_closest = float(df.loc[i0, "z"])

    print(f"Requested: ({x0}, {y0}, {z0})")
    print(f"Using closest point in CSV: ({x_closest}, {y_closest}, {z_closest})")

    sub = df[(df["x"] == x_closest) & (df["y"] == y_closest) & (df["z"] == z_closest)].copy()
    sub = sub.sort_values("t_h")

    t = sub["t_h"].to_numpy()
    T = sub["T_K"].to_numpy()

    plt.plot(t, T, label=f"S{i}")

plt.xlabel("t_h [h]")
plt.ylabel("T_K [K]")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig("temperature_locations.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------- Alpha figure ----------------
plt.figure()

for i, (x0, y0, z0) in enumerate(LOCATIONS, start=1):
    d2 = (df["x"] - x0)**2 + (df["y"] - y0)**2 + (df["z"] - z0)**2
    i0 = d2.idxmin()

    x_closest = float(df.loc[i0, "x"])
    y_closest = float(df.loc[i0, "y"])
    z_closest = float(df.loc[i0, "z"])

    sub = df[(df["x"] == x_closest) & (df["y"] == y_closest) & (df["z"] == z_closest)].copy()
    sub = sub.sort_values("t_h")

    t = sub["t_h"].to_numpy()
    alpha = sub["alpha"].to_numpy()

    plt.plot(t, alpha, label=f"S{i}")

plt.xlabel("t_h [h]")
plt.ylabel("α [-]")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig("alpha_locations.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------- q_Wm3 figure (ADDED BLOCK ONLY) ----------------
plt.figure()

for i, (x0, y0, z0) in enumerate(LOCATIONS, start=1):
    d2 = (df["x"] - x0)**2 + (df["y"] - y0)**2 + (df["z"] - z0)**2
    i0 = d2.idxmin()

    x_closest = float(df.loc[i0, "x"])
    y_closest = float(df.loc[i0, "y"])
    z_closest = float(df.loc[i0, "z"])

    sub = df[(df["x"] == x_closest) & (df["y"] == y_closest) & (df["z"] == z_closest)].copy()
    sub = sub.sort_values("t_h")

    t = sub["t_h"].to_numpy()
    q = sub["q_Wm3"].to_numpy()

    plt.plot(t, q, label=f"S{i}")

plt.xlabel("t_h [h]")
plt.ylabel("q_Wm3 [W/m³]")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig("q_Wm3_locations.png", dpi=200, bbox_inches="tight")
plt.close()
