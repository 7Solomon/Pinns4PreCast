""" This code would be the same as 01_Omid_simulation_data_Gereon.py but for a 2d case. Here the NN is shared for both T and alpha and instead of Dirichlet boundary 
we use convective boundary consition at 4 sides. Also the old code is 3d slab and here we use a square domain."""

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


################ Inject Sparse Data ##############

# ---- 1) Define your 5 sensor locations (in meters) ----
# SENSORS = [
#     (0.01, 0.19),
#     (0.03, 0.19),
#     (0.05, 0.19),
#     (0.07, 0.19),
#     (0.09, 0.19),
#     (0.11, 0.19),
#     (0.13, 0.19),
#     (0.15, 0.19),
#     (0.17, 0.19),
#     (0.19, 0.19),
#     (0.21, 0.19),
#     (0.23, 0.19),
#     (0.25, 0.19),
#     (0.27, 0.19),
#     (0.29, 0.19),
#     (0.31, 0.19),
#     (0.33, 0.19),
#     (0.35, 0.19),
#     (0.37, 0.19),
#     (0.39, 0.19),

# ]
SENSORS = [

    (0.19, 0.01),
    (0.19, 0.11),
    (0.19, 0.19),
    (0.19, 0.27),
    (0.19, 0.39)
]


# ---- 2) Read the CSV (your attached file) ----
df = pd.read_csv("T_Tamb_293d15_Tinit_300.csv")   # columns: x, y, time, T

# =========================
# (A) Plot temperature maps from the INPUT CSV (all locations, all times)
# Produces one PNG per unique time in df["time"] (e.g. 40 frames)
# =========================
import os

def plot_temperature_maps_from_df(
    df_field: pd.DataFrame,
    x_col="x", y_col="y", t_col="time", T_col="T",
    out_dir="temp_maps_input_csv",
    max_frames=None,
):
    os.makedirs(out_dir, exist_ok=True)

    # unique sorted times (seconds)
    times = np.sort(df_field[t_col].unique())
    if max_frames is not None:
        times = times[:max_frames]

    # grid coordinates (assumes structured grid in x and y)
    xs = np.sort(df_field[x_col].unique())
    ys = np.sort(df_field[y_col].unique())

    # Precompute mesh for pcolormesh (note: pcolormesh wants 2D arrays for X,Y or 1D bin edges;
    # here we use 2D mesh of centers and shading='auto' which works for centers)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    for k, t_s in enumerate(times):
        g = df_field[df_field[t_col] == t_s]

        # pivot to 2D array Z(y,x)
        Z = g.pivot(index=y_col, columns=x_col, values=T_col).reindex(index=ys, columns=xs).to_numpy()

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.pcolormesh(X, Y, Z, shading="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Temperature [K]")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")

        t_h = float(t_s) / 3600.0 if t_col == "time" else float(t_s)

        ax.set_title(f"Temperature map at t = {t_h:.3f} h")

        plt.tight_layout()
        fname = os.path.join(out_dir, f"Tmap_{k:03d}_t{t_h:.3f}h.png")
        plt.savefig(fname, dpi=200)
        plt.close(fig)

    print(f"[saved] {len(times)} temperature maps into: {out_dir}")

# call it once (will typically create 40 maps for your file)
plot_temperature_maps_from_df(df, out_dir="temp_maps_input_csv")


# Optional but recommended: rounding avoids float-matching issues
df["x_r"] = df["x"].round(6)
df["y_r"] = df["y"].round(6)

# ---- 3) Extract time histories for each sensor ----
sensor_histories = {}  # dict: "S1" -> {"t_s":..., "t_h":..., "T_K":..., "x":..., "y":...}

for i, (xs, ys) in enumerate(SENSORS, start=1):
    xs_r = round(xs, 6)
    ys_r = round(ys, 6)

    g = df[(df["x_r"] == xs_r) & (df["y_r"] == ys_r)].sort_values("time")

    if g.empty:
        raise ValueError(
            f"Sensor S{i} at (x={xs}, y={ys}) not found in CSV. "
            f"Check that the point matches the grid."
        )

    t_s = g["time"].to_numpy(dtype=np.float64)          # seconds
    t_h = t_s / 3600.0                                   # hours
    T_K = g["T"].to_numpy(dtype=np.float64)              # Kelvin

    sensor_histories[f"S{i}"] = {"x": xs, "y": ys, "t_s": t_s, "t_h": t_h, "T_K": T_K}

# ---- 4) (For training) Flatten ALL sensor observations into one list of points ----
# This is what you need later for data loss: x_obs, y_obs, t_obs_s, T_obs_K
x_obs = np.concatenate([np.full_like(v["t_s"], v["x"], dtype=np.float64) for v in sensor_histories.values()])
y_obs = np.concatenate([np.full_like(v["t_s"], v["y"], dtype=np.float64) for v in sensor_histories.values()])
t_obs_s = np.concatenate([v["t_s"] for v in sensor_histories.values()])
T_obs_K = np.concatenate([v["T_K"] for v in sensor_histories.values()])



# Plot and save temperature data for multiple sensors
fig, ax = plt.subplots(figsize=(8, 5))

for sid, data in sensor_histories.items():
    t_h = data["t_h"]
    T_K = data["T_K"]
    ax.plot(t_h, T_K, marker="*", linewidth=0.5, label=sid)

ax.set_xlabel("Time [h]")
ax.set_ylabel("Temperature [K]")
ax.set_title("Injected Temperature Data (multiple sensors)")
ax.grid(True, alpha=0.3)

# Put legend outside so it won't cover curves
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

plt.tight_layout()
plt.savefig("temperature_data_multi_sensors.png", dpi=200, bbox_inches="tight")
plt.close(fig)



# ---- Plot helpers (fixed to match training scaling + single model) ----

def make_t_plot(dom, n=300, device="cpu"):
    """
    Create a 1D time vector for post-training evaluation/plotting.

    Parameters
    ----------
    dom : Domain
        Domain object containing physical time bounds [s] as dom.t0, dom.t1.
    n : int
        Number of time points.
    device : str or torch.device
        Torch device on which to allocate the tensor.

    Returns
    -------
    t_plot : torch.Tensor, shape (n, 1)
        Time vector in PHYSICAL seconds.
    """
    return torch.linspace(dom.t0, dom.t1, n, device=device).view(-1, 1)




# ---------------------------
# Utilities: scaling
# ---------------------------
@dataclass
class Scales:
    T_c: float = 50.0
    L_c: float = 0.4
    t_c: float = 86400.0  # 1 day [s]


@dataclass
class Domain:
    """
    2D space-time domain bounds in PHYSICAL units.

    Spatial coordinates: (x, y) in meters
    Time coordinate: t in seconds
    """
    x0: float = 0.0
    x1: float = 0.4
    y0: float = 0.0
    y1: float = 0.4
    t0: float = 0.0
    t1: float = 86400.0  # seconds


@dataclass
class Material:
    cp: float = 1025
    rho: float = 2350.0
    k: float = 2.6
    Q_pot: float = 500000.0
    B1: float = 0.0002916
    B2: float = 0.0024229
    deg_hydr_max: float = 0.875
    eta: float = 5.554
    cem: float = 300.0
    Temp_ref: float = 298.15
    R: float = 8.31446261815324
    Ea: float = 47000.0  # <-- IMPORTANT: ensure this matches your repo / calibration
    h = 10.0  # Convective heat transfer coefficient [W/(m^2 K)]


def scale_domain(xyt: torch.Tensor, dom: Domain) -> torch.Tensor:
    """
    Scale PHYSICAL space–time coordinates to the unit cube [0,1].

    In 2D, the input coordinates are (x, y, t).

    Parameters
    ----------
    xyt : torch.Tensor, shape (N, 3)
        Physical coordinates [x, y, t], where t is in seconds.
    dom : Domain
        Domain object containing physical bounds.

    Returns
    -------
    xyt_scaled : torch.Tensor, shape (N, 3)
        Scaled coordinates in [0,1] suitable for neural network input.
    """
    x, y, t = xyt[:, 0], xyt[:, 1], xyt[:, 2]
    xs = (x - dom.x0) / (dom.x1 - dom.x0)
    ys = (y - dom.y0) / (dom.y1 - dom.y0)
    ts = (t - dom.t0) / (dom.t1 - dom.t0)
    return torch.stack([xs, ys, ts], dim=1)


def scale_T(T: torch.Tensor, scales: Scales) -> torch.Tensor:
    """
    Scale temperature from Kelvin to the network output scale.

    Parameters
    ----------
    T : torch.Tensor
        Temperature in Kelvin.
    scales : Scales
        Scaling parameters.

    Returns
    -------
    T_scaled : torch.Tensor
        Scaled temperature used as neural network output/target.
    """
    return T / scales.T_c


def unscale_T(Ts: torch.Tensor, scales: Scales) -> torch.Tensor:
    """
    Convert scaled network temperature output back to Kelvin.

    Parameters
    ----------
    Ts : torch.Tensor
        Scaled temperature.
    scales : Scales
        Scaling parameters.

    Returns
    -------
    T : torch.Tensor
        Temperature in Kelvin.
    """
    return Ts * scales.T_c


# ---------------------------
# Physics: affinity and kinetics
# ---------------------------
def chem_affinity_ref(alpha: torch.Tensor, mat: Material) -> torch.Tensor:
    """
    Reference affinity \tilde{A}_{ref}(alpha) as in the provided slide:
      B1 * (B2/alpha_max + alpha) * (alpha_max - alpha) * exp(-eta * alpha/alpha_max)
    """
    amax = mat.deg_hydr_max
    a = torch.clamp(alpha, 0.0, amax)
    eps = 1e-12
    return (
        mat.B1
        * (mat.B2 / (amax + eps) + a)
        * (amax - a)
        * torch.exp(-mat.eta * a / (amax + eps))
    )

def hydration_rate(alpha: torch.Tensor, T: torch.Tensor, mat: Material) -> torch.Tensor:
    """
    d alpha / dt = A_ref(alpha) * exp(-Ea/R (1/T - 1/Tref))
    """
    a = torch.clamp(alpha, 0.0, mat.deg_hydr_max)
    # Ensure absolute temperature in Kelvin
    T_safe = torch.clamp(T, 200.0, 4000.0)
    exponent = -mat.Ea / mat.R * (1.0 / T_safe - 1.0 / mat.Temp_ref)
    return chem_affinity_ref(a, mat) * torch.exp(exponent)


# ---------------------------
# PINN model
# ---------------------------

class PINN_T(nn.Module):
    """
    Temperature network: predicts only T_scaled.
    """
    def __init__(self, in_dim=3, hidden=256, depth=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, 1))  # output: T_scaled
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [N,3] scaled coords in [0,1]
        T_scaled = self.net(x)
        return T_scaled


class PINN_alpha_sigmoid(nn.Module):
    """
    Alpha network (NON-monotone in time):
    alpha = alpha_max * sigmoid(raw_alpha)  ->  0 <= alpha <= alpha_max
    """
    def __init__(self, in_dim=3, hidden=256, depth=4, alpha_max=0.875):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, 1))  # output: raw_alpha
        self.net = nn.Sequential(*layers)
        self.alpha_max = float(alpha_max)

    def forward(self, x):  # x: [N,3] scaled coords
        raw_alpha = self.net(x)
        alpha = self.alpha_max * torch.sigmoid(raw_alpha)  # enforce 0<=alpha<=alpha_max
        return alpha


class PINN_alpha_monotone(nn.Module):
    """
    Alpha network (monotone in time), your current design:
    raw_rate -> rate = softplus(raw_rate) >= 0
    s(ts) = rate * ts  (s(0)=0, nondecreasing in ts)
    alpha = alpha_max * (1 - exp(-s))
    """
    def __init__(self, in_dim=3, hidden=256, depth=4, alpha_max=0.875):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, 1))  # output: raw_rate
        self.net = nn.Sequential(*layers)
        self.alpha_max = float(alpha_max)

    def forward(self, x):  # x: [N,3] scaled coords in [0,1]
        raw_rate = self.net(x)

        ts = x[:, 2:3]              # scaled time in [0,1]
        rate = F.softplus(raw_rate) # >= 0
        s = rate * ts               # >= 0 and nondecreasing in ts

        alpha = self.alpha_max * (1.0 - torch.exp(-s))
        return alpha

# # ---------------------------
# # Sampling Monte Carlo points
# # ---------------------------
# def sample_uniform(N, dom: Domain, device):
#     """
#     Uniformly sample N space–time points in the 2D domain (x, y, t).
#     """
#     x = torch.rand(N, device=device) * (dom.x1 - dom.x0) + dom.x0
#     y = torch.rand(N, device=device) * (dom.y1 - dom.y0) + dom.y0
#     t = torch.rand(N, device=device) * (dom.t1 - dom.t0) + dom.t0
#     return torch.stack([x, y, t], dim=1)


# def sample_initial(N, dom: Domain, device):
#     """
#     Sample N points at the initial time t = t0 over the spatial domain.
#     """
#     pts = sample_uniform(N, dom, device)
#     pts[:, 2] = dom.t0  # time index in 2D input (x, y, t)
#     return pts


# def sample_boundary_dirichlet_T(N, dom: Domain, device):
#     """
#     Sample N space–time points on the spatial boundary of the 2D domain
#     (x = x0/x1 or y = y0/y1) for Dirichlet temperature conditions.
#     """
#     pts = sample_uniform(N, dom, device)

#     # In 2D there are 4 boundary edges
#     face = torch.randint(0, 4, (N,), device=device)

#     # Set one spatial coordinate to its boundary value
#     pts[face == 0, 0] = dom.x0
#     pts[face == 1, 0] = dom.x1
#     pts[face == 2, 1] = dom.y0
#     pts[face == 3, 1] = dom.y1

#     return pts
# def sample_boundary_convective(N, dom: Domain, device, return_face=False):
#     """
#     Sample N space–time points on the spatial boundary of the 2D domain
#     for convective (Robin/Newton) temperature boundary condition.

#     Returns
#     -------
#     pts : torch.Tensor, shape (N, 3)
#         Physical boundary points [x, y, t] with t in seconds.
#     n : torch.Tensor, shape (N, 2)
#         Outward unit normals [nx, ny] at each boundary point.
#     face : torch.Tensor, shape (N,)
#         Optional integer face id in {0,1,2,3}:
#           0: x=x0 (left), 1: x=x1 (right), 2: y=y0 (bottom), 3: y=y1 (top)
#     """
#     pts = sample_uniform(N, dom, device)

#     # choose boundary edge for each point
#     face = torch.randint(0, 4, (N,), device=device)

#     # initialize normals
#     n = torch.zeros((N, 2), device=device, dtype=pts.dtype)

#     # left: x = x0, n = (-1, 0)
#     m = (face == 0)
#     pts[m, 0] = dom.x0
#     n[m, 0] = -1.0

#     # right: x = x1, n = (+1, 0)
#     m = (face == 1)
#     pts[m, 0] = dom.x1
#     n[m, 0] = +1.0

#     # bottom: y = y0, n = (0, -1)
#     m = (face == 2)
#     pts[m, 1] = dom.y0
#     n[m, 1] = -1.0

#     # top: y = y1, n = (0, +1)
#     m = (face == 3)
#     pts[m, 1] = dom.y1
#     n[m, 1] = +1.0

#     if return_face:
#         return pts, n, face
#     return pts, n


###### End of Sampling Monte Carlo points ######

# # ---------------------------
# # Sampling non-Monte Carlo points
# # ---------------------------
def sample_uniform(N, dom: Domain, device):
    """
    Deterministic (grid-based) sampling of N space–time points
    in the 2D domain (x, y, t).
    """
    # approximate grid resolution
    n = int(round(N ** (1/3)))
    n = max(n, 2)

    x = torch.linspace(dom.x0, dom.x1, n, device=device)
    y = torch.linspace(dom.y0, dom.y1, n, device=device)
    t = torch.linspace(dom.t0, dom.t1, n, device=device)

    X, Y, T = torch.meshgrid(x, y, t, indexing="ij")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1), T.reshape(-1)], dim=1)

    # trim or pad to exactly N points
    if pts.shape[0] > N:
        pts = pts[:N]
    return pts

def sample_initial(N, dom: Domain, device):
    """
    Deterministic sampling at initial time t = t0.
    """
    pts = sample_uniform(N, dom, device)
    pts[:, 2] = dom.t0
    return pts


def sample_boundary_dirichlet_T(N, dom: Domain, device):
    """
    Deterministic sampling of N space–time points on all 4 boundaries,
    with equal points per side.
    """
    # points per face
    Nf = N // 4
    Nt = int(round((Nf) ** 0.5))
    Nt = max(Nt, 2)

    y = torch.linspace(dom.y0, dom.y1, Nt, device=device)
    x = torch.linspace(dom.x0, dom.x1, Nt, device=device)
    t = torch.linspace(dom.t0, dom.t1, Nt, device=device)

    # left (x = x0)
    Y, T = torch.meshgrid(y, t, indexing="ij")
    left = torch.stack([
        torch.full_like(Y.reshape(-1), dom.x0),
        Y.reshape(-1),
        T.reshape(-1)
    ], dim=1)

    # right (x = x1)
    right = left.clone()
    right[:, 0] = dom.x1

    # bottom (y = y0)
    X, T = torch.meshgrid(x, t, indexing="ij")
    bottom = torch.stack([
        X.reshape(-1),
        torch.full_like(X.reshape(-1), dom.y0),
        T.reshape(-1)
    ], dim=1)

    # top (y = y1)
    top = bottom.clone()
    top[:, 1] = dom.y1

    pts = torch.cat([left, right, bottom, top], dim=0)

    if pts.shape[0] > N:
        pts = pts[:N]

    return pts


def sample_boundary_convective_fixed(N, dom: Domain, device, return_face=False):
    """
    Deterministic (grid-based) sampling of N space–time points on the
    spatial boundary of the 2D domain for convective (Robin/Newton)
    temperature boundary conditions.

    Returns
    -------
    pts : torch.Tensor, shape (N, 3)
        Physical boundary points [x, y, t].
    n : torch.Tensor, shape (N, 2)
        Outward unit normals [nx, ny].
    face : torch.Tensor, shape (N,), optional
        Face id in {0,1,2,3}:
          0: x=x0 (left)
          1: x=x1 (right)
          2: y=y0 (bottom)
          3: y=y1 (top)
    """

    # points per face
    Nf = N // 4
    Nt = int(round((Nf) ** 0.5))
    Nt = max(Nt, 2)

    # grids
    x = torch.linspace(dom.x0, dom.x1, Nt, device=device)
    y = torch.linspace(dom.y0, dom.y1, Nt, device=device)
    t = torch.linspace(dom.t0, dom.t1, Nt, device=device)

    # ---- left boundary (x = x0) ----
    Y, T = torch.meshgrid(y, t, indexing="ij")
    left = torch.stack([
        torch.full_like(Y.reshape(-1), dom.x0),
        Y.reshape(-1),
        T.reshape(-1)
    ], dim=1)
    n_left = torch.tensor([-1.0, 0.0], device=device).repeat(left.shape[0], 1)
    f_left = torch.zeros(left.shape[0], device=device, dtype=torch.long)

    # ---- right boundary (x = x1) ----
    right = left.clone()
    right[:, 0] = dom.x1
    n_right = torch.tensor([+1.0, 0.0], device=device).repeat(right.shape[0], 1)
    f_right = torch.ones(right.shape[0], device=device, dtype=torch.long)

    # ---- bottom boundary (y = y0) ----
    X, T = torch.meshgrid(x, t, indexing="ij")
    bottom = torch.stack([
        X.reshape(-1),
        torch.full_like(X.reshape(-1), dom.y0),
        T.reshape(-1)
    ], dim=1)
    n_bottom = torch.tensor([0.0, -1.0], device=device).repeat(bottom.shape[0], 1)
    f_bottom = torch.full((bottom.shape[0],), 2, device=device, dtype=torch.long)

    # ---- top boundary (y = y1) ----
    top = bottom.clone()
    top[:, 1] = dom.y1
    n_top = torch.tensor([0.0, +1.0], device=device).repeat(top.shape[0], 1)
    f_top = torch.full((top.shape[0],), 3, device=device, dtype=torch.long)

    # concatenate
    pts = torch.cat([left, right, bottom, top], dim=0)
    n = torch.cat([n_left, n_right, n_bottom, n_top], dim=0)
    face = torch.cat([f_left, f_right, f_bottom, f_top], dim=0)

    # trim if needed
    if pts.shape[0] > N:
        pts = pts[:N]
        n = n[:N]
        face = face[:N]

    if return_face:
        return pts, n, face
    return pts, n
################# End of Sampling non-Monte Carlo points ##############



# ---------------------------
# Autograd helpers
# ---------------------------
def grad(outputs, inputs, idx):
    """
    Compute ∂outputs / ∂inputs[:, idx] using autograd.

    Parameters
    ----------
    outputs : torch.Tensor, shape (N, 1)
        Scalar network output (e.g. T or alpha).
    inputs : torch.Tensor, shape (N, D)
        Input tensor (D = 3 for 2D: x, y, t).
    idx : int
        Index of the input dimension to differentiate with respect to.

    Returns
    -------
    grad_i : torch.Tensor, shape (N, 1)
        Partial derivative with respect to inputs[:, idx].
    """
    g = torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        retain_graph=True,
        create_graph=True
    )[0]
    return g[:, idx:idx+1]

def laplacian(u, x):
    """
    Compute the 2D Laplacian ∂²u/∂x² + ∂²u/∂y².
    """
    ux = grad(u, x, 0)
    uy = grad(u, x, 1)
    uxx = grad(ux, x, 0)
    uyy = grad(uy, x, 1)
    return uxx + uyy


def save_full_field_on_grid(
    model_T, model_a, dom, scales, mat,
    dx=0.04, dy=0.04,
    x_vals=None, y_vals=None,   # <-- ADD THIS LINE
    t_hours=None,
    out_csv="field_full.csv",
    device="cpu",
    batch_size=200_000,
):

    """
    Save T [K], alpha [-], q [W/m^3] on a structured 2D space–time grid to CSV.

    - Spatial grid includes endpoints (x0..x1, y0..y1).
    - Time grid taken from t_hours (in hours). If None, uses 0..24 with 25 steps.
    - q is computed via autograd: q = Q_pot * cem * d(alpha)/dt  (t in seconds).
    """
    import numpy as np
    import pandas as pd
    import torch

    model_T.eval()
    model_a.eval()

    # ---- Time grid ----
    if t_hours is None:
        t_hours = np.arange(0.0, 25.0, 1.0)  # 0..24 inclusive (25 points)
    t_s = torch.tensor(t_hours, device=device, dtype=torch.float32).view(-1, 1) * 3600.0
    nt = t_s.shape[0]

    # ---- Spatial grids (include endpoints) ----
    # ---- Spatial grids ----
    if x_vals is None or y_vals is None:
        # fallback: uniform grid from dx, dy
        nx = int(round((dom.x1 - dom.x0) / dx)) + 1
        ny = int(round((dom.y1 - dom.y0) / dy)) + 1
        x = torch.linspace(dom.x0, dom.x1, nx, device=device)
        y = torch.linspace(dom.y0, dom.y1, ny, device=device)
    else:
        # exact dataset grid
        x = torch.tensor(np.asarray(x_vals), device=device, dtype=torch.float32)
        y = torch.tensor(np.asarray(y_vals), device=device, dtype=torch.float32)
        nx, ny = x.numel(), y.numel()

    # Full 3D mesh (flattened): (x, y, t)
    X, Y, Tt = torch.meshgrid(x, y, t_s.view(-1), indexing="ij")
    Xf = X.reshape(-1, 1)
    Yf = Y.reshape(-1, 1)
    Tf = Tt.reshape(-1, 1)

    N = Xf.shape[0]

    # Output buffers on CPU
    x_out = np.empty((N,), dtype=np.float32)
    y_out = np.empty((N,), dtype=np.float32)
    t_out = np.empty((N,), dtype=np.float32)

    T_out = np.empty((N,), dtype=np.float32)
    a_out = np.empty((N,), dtype=np.float32)
    q_out = np.empty((N,), dtype=np.float32)

    # Batch evaluation
    for i0 in range(0, N, batch_size):
        i1 = min(i0 + batch_size, N)

        xyt_phys = torch.cat([Xf[i0:i1], Yf[i0:i1], Tf[i0:i1]], dim=1).requires_grad_(True)

        xyt_scaled = scale_domain(xyt_phys, dom)

        T_s_pred = model_T(xyt_scaled)
        alpha = model_a(xyt_scaled)
        T_K = unscale_T(T_s_pred, scales)

        # d alpha / dt (seconds): time index is 2 in 2D input (x, y, t)
        dalpha_dt = torch.autograd.grad(
            alpha, xyt_phys,
            grad_outputs=torch.ones_like(alpha),
            retain_graph=False,
            create_graph=False
        )[0][:, 2:3]

        q = mat.Q_pot * mat.cem * dalpha_dt  # [W/m^3]

        # Store coords + fields
        x_out[i0:i1] = xyt_phys[:, 0].detach().cpu().numpy()
        y_out[i0:i1] = xyt_phys[:, 1].detach().cpu().numpy()
        t_out[i0:i1] = (xyt_phys[:, 2].detach().cpu().numpy() / 3600.0)  # hours

        T_out[i0:i1] = T_K.detach().cpu().numpy().reshape(-1)
        a_out[i0:i1] = alpha.detach().cpu().numpy().reshape(-1)
        q_out[i0:i1] = q.detach().cpu().numpy().reshape(-1)

    df = pd.DataFrame({
        "x": x_out,
        "y": y_out,
        "t_h": t_out,
        "T_K": T_out,
        "alpha": a_out,
        "q_Wm3": q_out,
    })

    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}  (rows={len(df)}, nx={nx}, ny={ny}, nt={nt})")

# ---------------------------
# Main training
# ---------------------------
def main(
    T_ic_K=298.15,
    T_bc_K=298.15,
    T_amb_K=298.15,
    alpha_ic=1e-6,
    steps=1000,
    lr=1e-4,
    N_pde=5000,
    N_ic=2000,
    N_bc=2000,
    BC = 'Dirichlet'  # 'Dirichlet' or 'Convective'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"  # For debugging on CPU


    # Load inputs (optional)
    dom = Domain()
    scales = Scales()
    mat = Material()

    # --- center point for injected temperature data (2D) ---
    # --- temperature observation data (multiple sensors, scattered in space-time) ---
    x_data = torch.tensor(x_obs, device=device, dtype=torch.float32).view(-1, 1)
    y_data = torch.tensor(y_obs, device=device, dtype=torch.float32).view(-1, 1)
    t_data = torch.tensor(t_obs_s, device=device, dtype=torch.float32).view(-1, 1)
    T_data = torch.tensor(T_obs_K, device=device, dtype=torch.float32).view(-1, 1)


    xyt_data_phys = torch.cat([x_data, y_data, t_data], dim=1)
    xyt_data_in   = scale_domain(xyt_data_phys, dom)
    T_data_s      = scale_T(T_data, scales)


    # --- two separate networks (parallel) ---
    model_T = PINN_T().to(device)

    # pick ONE of the alpha models:
    model_a = PINN_alpha_monotone(alpha_max=mat.deg_hydr_max).to(device)
    # model_a = PINN_alpha_sigmoid(alpha_max=mat.deg_hydr_max).to(device)

    opt = torch.optim.Adam(list(model_T.parameters()) + list(model_a.parameters()), lr=lr)

    # nondimensional coefficients consistent with slide form:
    # cp*rho*dT/dt = k*ΔT + Q_pot*cem*dalpha/dt
    # => dT/dt - (k/(cp*rho))*ΔT - (Q_pot*cem/(cp*rho))*dalpha/dt = 0
    kappa = mat.k / (mat.cp * mat.rho)
    beta = (mat.Q_pot * mat.cem) / (mat.cp * mat.rho)

    # fixed Dirichlet values (in Kelvin)
    T_ic = torch.tensor([[T_ic_K]], device=device, dtype=torch.float32)
    T_bc = torch.tensor([[T_bc_K]], device=device, dtype=torch.float32)
    T_inf = torch.tensor([[T_amb_K]], device=device, dtype=torch.float32)  # ambient [K]
    alpha0 = torch.tensor([[alpha_ic]], device=device, dtype=torch.float32)

    # scale T targets to network output scale
    T_ic_s = scale_T(T_ic, scales)
    T_bc_s = scale_T(T_bc, scales)

    # =========================
    # (B) Loss tracking buffers
    # =========================
    it_hist = []
    loss_hist = []
    phys_a_hist = []
    phys_T_hist = []
    ic_hist = []
    bc_hist = []
    data_hist = []

    for it in range(1, steps + 1):
        # ----- sample points
        pde_pts = sample_uniform(N_pde, dom, device).requires_grad_(True)
        ic_pts = sample_initial(N_ic, dom, device)
        if BC == 'Dirichlet':
            bc_pts = sample_boundary_dirichlet_T(N_bc, dom, device)
        elif BC == 'Convective':
            bc_pts, n_bc = sample_boundary_convective_fixed(N_bc, dom, device)  # n_bc: (N,2)
            bc_pts = bc_pts.requires_grad_(True)

        # scale inputs for NN
        pde_in = scale_domain(pde_pts, dom)
        ic_in = scale_domain(ic_pts, dom)
        bc_in = scale_domain(bc_pts, dom)

        # ----- predictions (two separate networks)
        T_s = model_T(pde_in)
        alpha = model_a(pde_in)

        # unscale T for kinetics and PDE physical terms
        T_K = unscale_T(T_s, scales)

        # ----- compute residuals
        # alpha equation residual: d alpha/dt - rate(alpha,T) = 0
        # NOTE: t is the 3rd input coordinate in pde_pts for 2D (x,y,t)
        dalpha_dt = grad(alpha, pde_pts, 2)  # derivative wrt physical t (seconds)
        rate = hydration_rate(alpha, T_K, mat)
        r_alpha = dalpha_dt - rate

        # heat equation residual:
        dTdt = grad(T_K, pde_pts, 2)
        lapT = laplacian(T_K, pde_pts)
        r_T = dTdt - kappa * lapT - beta * dalpha_dt

        loss_phys_T = (r_T.pow(2).mean()) 
        loss_phys_a = (r_alpha.pow(2).mean()) 

        # ----- IC loss (T and alpha at t=0)
        T_ic_pred_s = model_T(ic_in)
        alpha_ic_pred = model_a(ic_in)
        loss_ic_T = F.mse_loss(T_ic_pred_s, T_ic_s.expand_as(T_ic_pred_s))
        loss_ic_a = F.mse_loss(alpha_ic_pred, alpha0.expand_as(alpha_ic_pred))
        loss_ic = loss_ic_T + loss_ic_a

        # ----- BC loss (Dirichlet T on boundary)
        T_bc_pred_s = model_T(bc_in)
        if BC == 'Dirichlet':
            loss_bc = F.mse_loss(T_bc_pred_s, T_bc_s.expand_as(T_bc_pred_s))
        elif BC == 'Convective':
            # IMPORTANT: use physical temperature for BC
            Tbc_K = unscale_T(T_bc_pred_s, scales)

            # compute grad T wrt physical x,y
            Tx = grad(Tbc_K, bc_pts, 0)
            Ty = grad(Tbc_K, bc_pts, 1)

            # normal derivative dT/dn = gradT · n
            dTdn = Tx * n_bc[:, 0:1] + Ty * n_bc[:, 1:2]

            # ambient temperature at boundary (can be constant or time-dependent)
            Tinf = T_inf.expand_as(Tbc_K)

            # residual: -k dT/dn - h (T - Tinf) = 0
            r_bc = (-mat.k) * dTdn - mat.h * (Tbc_K - Tinf)

            loss_bc = (r_bc.pow(2)).mean()

        # --- temperature data loss ---
        T_data_pred_s = model_T(xyt_data_in)
        loss_data_T = F.mse_loss(T_data_pred_s, T_data_s)

        # weights (tune these!)
        w_a_phys, w_T_phys, w_ic, w_bc, w_data = 1.0, 1.0, 1.0, 1.0, 1.0
        loss = w_a_phys * loss_phys_a + w_T_phys * loss_phys_T + w_ic * loss_ic + w_bc * loss_bc + w_data * loss_data_T

        # =========================
        # (B) Record losses
        # =========================
        it_hist.append(it)
        phys_a_hist.append(float(loss_phys_a.detach().cpu()))
        phys_T_hist.append(float(loss_phys_T.detach().cpu()))
        loss_hist.append(float(loss.detach().cpu()))
        ic_hist.append(float(loss_ic.detach().cpu()))
        bc_hist.append(float(loss_bc.detach().cpu()))
        data_hist.append(float(loss_data_T.detach().cpu()))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 500 == 0:
            with torch.no_grad():
                a_min = float(alpha.min().cpu())
                a_max = float(alpha.max().cpu())
                print(
                    f"iter {it:6d} | loss {loss.item():.3e} "
                    f"| phys_a {loss_phys_a.item():.3e} phys_T {loss_phys_T.item():.3e} ic {loss_ic.item():.3e} bc {loss_bc.item():.3e} "
                    f"| alpha[pde] in [{a_min:.3e},{a_max:.3e}]"
                )

    # =========================
    # (B) Plot and save loss curves
    # =========================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(it_hist, loss_hist, label="total")
    ax.semilogy(it_hist, phys_a_hist, label="phys_a")
    ax.semilogy(it_hist, phys_T_hist, label="phys_T")
    ax.semilogy(it_hist, ic_hist, label="ic")
    ax.semilogy(it_hist, bc_hist, label="bc")
    ax.semilogy(it_hist, data_hist, label="data")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training losses (log scale)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("[saved] loss_curves.png")


    # After training you can probe sensor curves by evaluating model at sensor coords over time.
    print("Done.")

    t_hours_data = np.sort(df["time"].unique()) / 3600.0
    x_vals_data = np.sort(df["x"].unique())
    y_vals_data = np.sort(df["y"].unique())



    # ---- Save full space-time fields on a 2D grid and 25 hourly steps (0..24 h) ----
    save_full_field_on_grid(
        model_T=model_T, model_a=model_a, dom=dom, scales=scales, mat=mat,
        x_vals=x_vals_data, y_vals=y_vals_data,   # <-- ADD THIS
        t_hours=t_hours_data,                     # already correct
        out_csv="field_full.csv",
        device=device,
        batch_size=200_000,
    )


    df_pred = pd.read_csv("field_full.csv")
    plot_temperature_maps_from_df(df_pred, t_col="t_h", T_col="T_K", out_dir="temp_maps_predicted")

        # =========================
    # (C) Error maps: compare predicted field_full.csv vs input CSV df
    # Produces one error map per matched time
    # =========================
    import os

    def plot_error_maps_pred_vs_data(
        df_data: pd.DataFrame,            # input CSV already loaded above as `df`
        pred_csv="field_full.csv",        # produced by save_full_field_on_grid
        out_dir="temp_error_maps",
        data_x="x", data_y="y", data_t="time", data_T="T",
        pred_x="x", pred_y="y", pred_t="t_h", pred_T="T_K",
        time_tolerance_s=120.0,           # allowable mismatch in seconds (e.g., 2 minutes)
        mode="abs",                       # "abs" -> |pred-data|, "signed" -> pred-data
        max_frames=None,
    ):
        os.makedirs(out_dir, exist_ok=True)

        df_pred = pd.read_csv(pred_csv)

        # Data times are in seconds; prediction times are in hours
        data_times_s = np.sort(df_data[data_t].unique())
        pred_times_h = np.sort(df_pred[pred_t].unique())

        # We will match each pred time to nearest data time
        matched = []
        for th in pred_times_h:
            ts_pred = float(th) * 3600.0
            j = int(np.argmin(np.abs(data_times_s - ts_pred)))
            ts_data = float(data_times_s[j])
            if abs(ts_data - ts_pred) <= time_tolerance_s:
                matched.append((th, ts_data))

        if not matched:
            print(
                "[warn] No matching times found for error maps. "
                "Your prediction times (t_h) likely do not coincide with data times. "
                "Increase tolerance or save predictions at data times."
            )
            return

        if max_frames is not None:
            matched = matched[:max_frames]

        # shared grids (assumes structured grid)
        xs = np.sort(df_pred[pred_x].unique())
        ys = np.sort(df_pred[pred_y].unique())
        X, Y = np.meshgrid(xs, ys, indexing="xy")

        for k, (th, ts_data) in enumerate(matched):
            # predicted slice at t_h=th
            gp = df_pred[df_pred[pred_t] == th]
            Tp = gp.pivot(index=pred_y, columns=pred_x, values=pred_T).reindex(index=ys, columns=xs).to_numpy()

            # data slice at time=ts_data (seconds)
            gd = df_data[df_data[data_t] == ts_data]
            Td = gd.pivot(index=data_y, columns=data_x, values=data_T).reindex(index=ys, columns=xs).to_numpy()

            # error
            if mode == "abs":
                E = np.abs(Tp - Td)
                label = "|T_pred - T_data| [K]"
                title_mode = "Absolute error"
            else:
                E = Tp - Td
                label = "T_pred - T_data [K]"
                title_mode = "Signed error"

            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.pcolormesh(X, Y, E, shading="auto")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(label)

            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal", adjustable="box")

            t_h_data = float(ts_data) / 3600.0
            ax.set_title(f"{title_mode} map at t ≈ {float(th):.3f} h (data {t_h_data:.3f} h)")

            plt.tight_layout()
            fname = os.path.join(out_dir, f"Err_{k:03d}_t{float(th):.3f}h.png")
            plt.savefig(fname, dpi=200)
            plt.close(fig)

        print(f"[saved] {len(matched)} error maps into: {out_dir}")

    # call error-map plotting
    plot_error_maps_pred_vs_data(
        df_data=df,
        pred_csv="field_full.csv",
        out_dir="temp_error_maps",
        time_tolerance_s=120.0,   # adjust if needed
        mode="abs",
    )



    


if __name__ == "__main__":
    # Set BC/IC in Kelvin for this debug run:
    # e.g., to see a peak more easily: IC warmer than ambient BC
    main(
        T_ic_K=300,                 # 25C
        T_bc_K=293.15,                 # 20C (recommended debug)
        T_amb_K=293.15,               # 20C
        alpha_ic=1e-6,
        steps=40000,
        lr=1e-4,
        N_pde=15000,
        N_ic=6000,
        N_bc=6000,
        BC = 'Dirichlet'  # 'Dirichlet' or 'Convective'
    )
