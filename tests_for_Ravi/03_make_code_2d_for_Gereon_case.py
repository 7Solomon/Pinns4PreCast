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
# t_data_h = [
#     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
#     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
# ]

# T_data_K = [
#     293.15, 296.65, 303.15, 311.15, 319.15, 323.15,
#     321.15, 318.15, 315.15, 312.15, 309.65, 307.15,
#     305.15, 303.65, 302.15, 300.95, 299.95, 298.95,
#     297.95, 297.15, 296.45, 295.85, 295.35, 294.95, 294.65
# ]

t_data_s = [
    0.0, 1846.15384615, 3692.30769231, 5538.46153846,
    7384.61538462, 9230.76923077, 11076.92307692, 12923.07692308,
    14769.23076923, 16615.38461538, 18461.53846154, 20307.69230769,
    22153.84615385, 24000.0, 25846.15384615, 27692.30769231,
    29538.46153846, 31384.61538462, 33230.76923077, 35076.92307692,
    36923.07692308, 38769.23076923, 40615.38461538, 42461.53846154,
    44307.69230769, 46153.84615385, 48000.0, 49846.15384615,
    51692.30769231, 53538.46153846, 55384.61538462, 57230.76923077,
    59076.92307692, 60923.07692308, 62769.23076923, 64615.38461538,
    66461.53846154, 68307.69230769, 70153.84615385, 72000.0
]


t_data_h = [ts/3600.0 for ts in t_data_s]

T_data_K = [290.0, 290.10466405, 290.26967412, 290.51753733, 290.86822953,
 291.34334103, 291.97137136, 292.79153112, 293.85423536, 295.21608649,
 296.92670008, 299.00665858, 301.42248345, 304.07311848, 306.80219264,
 309.43454128, 311.81756784, 313.84733401, 315.47355819, 316.69019488,
 317.52102613, 318.00644482, 318.19374299, 318.13092351, 317.86320671,
 317.43133442, 316.87096937, 316.21270858, 315.48241561, 314.70169438,
 313.88840585, 313.05717351, 312.2198524, 311.38595019, 310.56299772,
 309.75687032, 308.97206474, 308.21193505, 307.4788932, 306.77457858]

# Plot and save temperature data
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_data_h, T_data_K, 'o-', label='Temperature data')
ax.set_xlabel('Time [h]')
ax.set_ylabel('Temperature [K]')
ax.set_title('Injected Temperature Data')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('temperature_data.png', dpi=200)
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


def center_physical_point(dom: "Domain", device):
    """
    Compute the geometric center of the 2D spatial domain in PHYSICAL coordinates.


    Parameters
    ----------
    dom : Domain
        Domain object containing x- and y-bounds (dom.x0, dom.x1, dom.y0, dom.y1).
    device : str or torch.device
        Torch device on which to allocate the tensors.

    Returns
    -------
    x : torch.Tensor, shape (1, 1)
        Center x-coordinate (physical units).
    y : torch.Tensor, shape (1, 1)
        Center y-coordinate (physical units).
    """
    xc = 0.5 * (dom.x0 + dom.x1)
    yc = 0.5 * (dom.y0 + dom.y1)
    x = torch.tensor([[xc]], device=device)
    y = torch.tensor([[yc]], device=device)
    return x, y


def evaluate_center_curves(model, dom, scales, mat, t_plot):
    """
    Evaluate model predictions at the geometric center over a provided time vector,
    and compute the volumetric heat source term q from d(alpha)/dt.

    This returns curves at the geometric center in PHYSICAL coordinates:
      t_h [h], T [K], alpha [-], q [W/m^3]

    Notes
    -----
    - The network is evaluated on SCALED coordinates, but derivatives for q are taken
      with respect to PHYSICAL time (seconds) by differentiating alpha w.r.t. xy_t_phys.
    - In 2D, the model input is (x, y, t). Therefore, time is the 3rd component.

    Parameters
    ----------
    model : torch.nn.Module
        PINN model mapping scaled (x,y,t) -> (T_scaled, alpha).
    dom : Domain
        Domain with physical bounds.
    scales : Scales
        Scaling parameters used to convert between T_scaled and Kelvin.
    mat : Material
        Material parameters; q uses mat.Q_pot and mat.cem.
    t_plot : torch.Tensor, shape (N, 1)
        Time vector in PHYSICAL seconds.

    Returns
    -------
    t_h : np.ndarray, shape (N, 1)
        Time in hours.
    T_K : np.ndarray, shape (N, 1)
        Temperature in Kelvin.
    alpha : np.ndarray, shape (N, 1)
        Degree of hydration (dimensionless).
    q : np.ndarray, shape (N, 1)
        Volumetric heat source in W/m^3 computed as Q_pot * cem * d(alpha)/dt.
    """
    device = t_plot.device

    # Center point (2D)
    x, y = center_physical_point(dom, device)
    x = x.expand_as(t_plot)
    y = y.expand_as(t_plot)

    # Build PHYSICAL input for time-derivative wrt seconds: [x, y, t]
    xyt_phys = torch.cat([x, y, t_plot], dim=1).requires_grad_(True)

    # Network takes SCALED coords
    xyt_scaled = scale_domain(xyt_phys, dom)
    pred = model(xyt_scaled)

    T_s = pred[:, 0:1]
    alpha = pred[:, 1:2]
    T_K = unscale_T(T_s, scales)

    # d alpha / dt (physical seconds): time is index 2 in 2D input (x,y,t)
    dalpha_dt = torch.autograd.grad(
        alpha, xyt_phys,
        grad_outputs=torch.ones_like(alpha),
        retain_graph=False,
        create_graph=False
    )[0][:, 2:3]

    # q = Q_pot * cem * d alpha/dt   [W/m^3] if Q_pot[J/kg_cem], cem[kg_cem/m^3]
    q = mat.Q_pot * mat.cem * dalpha_dt

    return (
        (t_plot.detach().cpu().numpy() / 3600.0),  # hours
        T_K.detach().cpu().numpy(),
        alpha.detach().cpu().numpy(),
        q.detach().cpu().numpy(),
    )


# ---------------------------
# Utilities: scaling
# ---------------------------
@dataclass
class Scales:
    T_c: float = 50.0
    L_c: float = 0.8
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
    y1: float = 0.8
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
    cem: float = 430.0
    Temp_ref: float = 298.15
    R: float = 8.31446261815324
    Ea: float = 47000.0  # <-- IMPORTANT: ensure this matches your repo / calibration

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
# class PINN(nn.Module):
#     def __init__(self, in_dim=4, hidden=256, depth=4, alpha_max=0.875):
#         super().__init__()
#         layers = []
#         layers.append(nn.Linear(in_dim, hidden))
#         layers.append(nn.Tanh())
#         for _ in range(depth - 1):
#             layers.append(nn.Linear(hidden, hidden))
#             layers.append(nn.Tanh())
#         layers.append(nn.Linear(hidden, 2))  # outputs: T_scaled, raw_alpha
#         self.net = nn.Sequential(*layers)
#         self.alpha_max = float(alpha_max)

#     def forward(self, x):  # x: [N,4] scaled coords
#         out = self.net(x)
#         T_scaled = out[:, 0:1]
#         raw_alpha = out[:, 1:2]
#         alpha = self.alpha_max * torch.sigmoid(raw_alpha)  # enforce 0<=alpha<=alpha_max
#         return torch.cat([T_scaled, alpha], dim=1)

class PINN(nn.Module):
    def __init__(self, in_dim=3, hidden=256, depth=4, alpha_max=0.875):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, 2))  # outputs: T_scaled, raw_s
        self.net = nn.Sequential(*layers)
        self.alpha_max = float(alpha_max)

    def forward(self, x):  # x: [N,4] scaled coords in [0,1]
        out = self.net(x)
        T_scaled = out[:, 0:1]

        # --- Monotone hydration via s-field ---
        # Use a nonnegative "rate" in scaled time and integrate analytically:
        # s(ts) = softplus(raw_rate) * ts  ->  ds/dts >= 0  and s(0)=0
        ts = x[:, 2:3]                    # scaled time in [0,1]
        raw_rate = out[:, 1:2]
        rate = F.softplus(raw_rate)       # >= 0
        s = rate * ts                     # >= 0 and nondecreasing in ts

        # Map s -> alpha in [0, alpha_max), monotone in s (hence in time)
        alpha = self.alpha_max * (1.0 - torch.exp(-s))

        return torch.cat([T_scaled, alpha], dim=1)



# ---------------------------
# Sampling
# ---------------------------
def sample_uniform(N, dom: Domain, device):
    """
    Uniformly sample N space–time points in the 2D domain (x, y, t).
    """
    x = torch.rand(N, device=device) * (dom.x1 - dom.x0) + dom.x0
    y = torch.rand(N, device=device) * (dom.y1 - dom.y0) + dom.y0
    t = torch.rand(N, device=device) * (dom.t1 - dom.t0) + dom.t0
    return torch.stack([x, y, t], dim=1)


def sample_initial(N, dom: Domain, device):
    """
    Sample N points at the initial time t = t0 over the spatial domain.
    """
    pts = sample_uniform(N, dom, device)
    pts[:, 2] = dom.t0  # time index in 2D input (x, y, t)
    return pts


def sample_boundary_dirichlet_T(N, dom: Domain, device):
    """
    Sample N space–time points on the spatial boundary of the 2D domain
    (x = x0/x1 or y = y0/y1) for Dirichlet temperature conditions.
    """
    pts = sample_uniform(N, dom, device)

    # In 2D there are 4 boundary edges
    face = torch.randint(0, 4, (N,), device=device)

    # Set one spatial coordinate to its boundary value
    pts[face == 0, 0] = dom.x0
    pts[face == 1, 0] = dom.x1
    pts[face == 2, 1] = dom.y0
    pts[face == 3, 1] = dom.y1

    return pts


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
    model, dom, scales, mat,
    dx=0.04, dy=0.04,
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

    model.eval()

    # ---- Time grid ----
    if t_hours is None:
        t_hours = np.arange(0.0, 25.0, 1.0)  # 0..24 inclusive (25 points)
    t_s = torch.tensor(t_hours, device=device, dtype=torch.float32).view(-1, 1) * 3600.0
    nt = t_s.shape[0]

    # ---- Spatial grids (include endpoints) ----
    nx = int(round((dom.x1 - dom.x0) / dx)) + 1
    ny = int(round((dom.y1 - dom.y0) / dy)) + 1

    x = torch.linspace(dom.x0, dom.x1, nx, device=device)
    y = torch.linspace(dom.y0, dom.y1, ny, device=device)

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
        pred = model(xyt_scaled)

        T_s_pred = pred[:, 0:1]
        alpha = pred[:, 1:2]
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
    domain_json="domain.json",
    material_json="material.json",
    T_ic_K=298.15,
    T_bc_K=298.15,
    alpha_ic=1e-6,
    steps=1000,
    lr=1e-4,
    N_pde=5000,
    N_ic=2000,
    N_bc=2000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"  # For debugging on CPU

    # --- temperature data (center point) ---
    t_data = torch.tensor(t_data_h, device=device).view(-1, 1) * 3600.0
    T_data = torch.tensor(T_data_K, device=device).view(-1, 1)

    # Load inputs (optional)
    dom = Domain()
    scales = Scales()
    mat = Material()

    # --- center point for injected temperature data (2D) ---
    xc = 0.5 * (dom.x0 + dom.x1)
    yc = 0.5 * (dom.y0 + dom.y1)

    x_data = torch.full_like(t_data, xc)
    y_data = torch.full_like(t_data, yc)

    xyt_data_phys = torch.cat([x_data, y_data, t_data], dim=1)
    xyt_data_in   = scale_domain(xyt_data_phys, dom)

    T_data_s = scale_T(T_data, scales)

    model = PINN(alpha_max=mat.deg_hydr_max).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # nondimensional coefficients consistent with slide form:
    # cp*rho*dT/dt = k*ΔT + Q_pot*cem*dalpha/dt
    # => dT/dt - (k/(cp*rho))*ΔT - (Q_pot*cem/(cp*rho))*dalpha/dt = 0
    kappa = mat.k / (mat.cp * mat.rho)
    beta = (mat.Q_pot * mat.cem) / (mat.cp * mat.rho)

    # fixed Dirichlet values (in Kelvin)
    T_ic = torch.tensor([[T_ic_K]], device=device, dtype=torch.float32)
    T_bc = torch.tensor([[T_bc_K]], device=device, dtype=torch.float32)
    alpha0 = torch.tensor([[alpha_ic]], device=device, dtype=torch.float32)

    # scale T targets to network output scale
    T_ic_s = scale_T(T_ic, scales)
    T_bc_s = scale_T(T_bc, scales)

    for it in range(1, steps + 1):
        # ----- sample points
        pde_pts = sample_uniform(N_pde, dom, device).requires_grad_(True)
        ic_pts = sample_initial(N_ic, dom, device)
        bc_pts = sample_boundary_dirichlet_T(N_bc, dom, device)

        # scale inputs for NN
        pde_in = scale_domain(pde_pts, dom)
        ic_in = scale_domain(ic_pts, dom)
        bc_in = scale_domain(bc_pts, dom)

        # ----- predictions
        pred_pde = model(pde_in)
        T_s = pred_pde[:, 0:1]
        alpha = pred_pde[:, 1:2]

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

        loss_phys = (r_T.pow(2).mean() + r_alpha.pow(2).mean())

        # ----- IC loss (T and alpha at t=0)
        pred_ic = model(ic_in)
        loss_ic_T = F.mse_loss(pred_ic[:, 0:1], T_ic_s.expand_as(pred_ic[:, 0:1]))
        loss_ic_a = F.mse_loss(pred_ic[:, 1:2], alpha0.expand_as(pred_ic[:, 1:2]))
        loss_ic = loss_ic_T + loss_ic_a

        # ----- BC loss (Dirichlet T on boundary)
        pred_bc = model(bc_in)
        loss_bc = F.mse_loss(pred_bc[:, 0:1], T_bc_s.expand_as(pred_bc[:, 0:1]))

        # --- temperature data loss ---
        pred_data = model(xyt_data_in)
        loss_data_T = F.mse_loss(pred_data[:, 0:1], T_data_s)

        # weights (tune these!)
        w_phys, w_ic, w_bc, w_data = 1.0, 10.0, 1.0, 1.0
        loss = w_phys * loss_phys + w_ic * loss_ic + w_bc * loss_bc + w_data * loss_data_T

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 500 == 0:
            with torch.no_grad():
                a_min = float(alpha.min().cpu())
                a_max = float(alpha.max().cpu())
                print(
                    f"iter {it:6d} | loss {loss.item():.3e} "
                    f"| phys {loss_phys.item():.3e} ic {loss_ic.item():.3e} bc {loss_bc.item():.3e} "
                    f"| alpha[pde] in [{a_min:.3e},{a_max:.3e}]"
                )

        if it % 1000 == 0:
            # Make time vector on the same device as the model
            t_plot = make_t_plot(dom, n=300, device=device)

            # Evaluate center curves (needs grad for dalpha/dt, so no torch.no_grad here)
            t_h, T_c, alpha_c, q_c = evaluate_center_curves(model, dom, scales, mat, t_plot)

            fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

            axs[0].plot(t_h, T_c, label='Predicted T [K]')
            axs[0].plot(t_data_h, T_data_K, 'r--', label='Injected T [K]')  # Dotted line for injected data
            axs[0].set_ylabel("T [K]")
            axs[0].set_title(f"Center evolution (iter {it})")
            axs[0].legend()

            axs[1].plot(t_h, alpha_c)
            axs[1].set_ylabel(r"$\alpha$")

            axs[2].plot(t_h, q_c)
            axs[2].set_ylabel(r"$q$ [W/m$^3$]")
            axs[2].set_xlabel("Time [h]")

            plt.tight_layout()
            plt.savefig("center_evolution.png", dpi=200)  # overwrites each time
            plt.close(fig)

    # After training you can probe sensor curves by evaluating model at sensor coords over time.
    print("Done.")

    # ---- Save full space-time fields on a 2D grid and 25 hourly steps (0..24 h) ----
    save_full_field_on_grid(
        model=model, dom=dom, scales=scales, mat=mat,
        dx=0.04, dy=0.04,
        t_hours=list(range(0, 25)),          # 25 steps: 0..24 h
        out_csv="field_full.csv",
        device=device,
        batch_size=200_000,
    )

    # Save results to CSV (from the last evaluation at it % 1000 == 0)
    results = pd.DataFrame({
        "t_h": t_h.squeeze(),
        "T_c": T_c.squeeze(),
        "alpha_c": alpha_c.squeeze(),
        "q_c": q_c.squeeze(),
    })
    results.to_csv('center_evolution_results.csv', index=False)


if __name__ == "__main__":
    # Set BC/IC in Kelvin for this debug run:
    # e.g., to see a peak more easily: IC warmer than ambient BC
    main(
        domain_json="domain.json",     # point to your actual domain file
        material_json="material.json", # point to your actual material file
        T_ic_K=290,                 # 25C
        T_bc_K=293.15,                 # 20C (recommended debug)
        alpha_ic=1e-6,
        steps=30000,
        lr=1e-4,
        N_pde=500,
        N_ic=200,
        N_bc=200,
    )
