# pinn_single_case.py
# Single-case PINN for coupled heat + hydration (no DeepONet, no Flask).
# (x,y,z,t) -> (T, alpha), with alpha bounded to [0, alpha_max].

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

################ Inject Sparse Data ##############
t_data_h = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
]

T_data_K = [
    293.15, 296.65, 303.15, 311.15, 319.15, 323.15,
    321.15, 318.15, 315.15, 312.15, 309.65, 307.15,
    305.15, 303.65, 302.15, 300.95, 299.95, 298.95,
    297.95, 297.15, 296.45, 295.85, 295.35, 294.95, 294.65
]


# ---- Plot helpers (fixed to match training scaling + single model) ----
def make_t_plot(dom, n=300, device="cpu"):
    return torch.linspace(dom.t0, dom.t1, n, device=device).view(-1, 1)

def center_physical_point(dom: "Domain", device):
    xc = 0.5 * (dom.x0 + dom.x1)
    yc = 0.5 * (dom.y0 + dom.y1)
    zc = 0.5 * (dom.z0 + dom.z1)
    x = torch.tensor([[xc]], device=device)
    y = torch.tensor([[yc]], device=device)
    z = torch.tensor([[zc]], device=device)
    return x, y, z

def evaluate_center_curves(model, dom, scales, mat, t_plot):
    """
    Returns curves at the geometric center in PHYSICAL coordinates:
      t_h [h], T [K], alpha [-], q [W/m^3]
    """
    device = t_plot.device
    x, y, z = center_physical_point(dom, device)
    x = x.expand_as(t_plot)
    y = y.expand_as(t_plot)
    z = z.expand_as(t_plot)

    # Build PHYSICAL input for time-derivative wrt seconds
    xyzt_phys = torch.cat([x, y, z, t_plot], dim=1).requires_grad_(True)

    # Network takes SCALED coords
    xyzt_scaled = scale_domain(xyzt_phys, dom)
    pred = model(xyzt_scaled)

    T_s = pred[:, 0:1]
    alpha = pred[:, 1:2]
    T_K = unscale_T(T_s, scales)

    # d alpha / dt (physical seconds)
    dalpha_dt = torch.autograd.grad(
        alpha, xyzt_phys,
        grad_outputs=torch.ones_like(alpha),
        retain_graph=False,
        create_graph=False
    )[0][:, 3:4]

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
    x0: float = 0.0
    x1: float = 0.4
    y0: float = 0.0
    y1: float = 0.8
    z0: float = 0.0
    z1: float = 0.4
    t0: float = 0.0
    t1: float = 86400.0  # seconds


@dataclass
class Material:
    cp: float = 850.0
    rho: float = 2400.0
    k: float = 1.4
    Q_pot: float = 500000.0
    B1: float = 0.0002916
    B2: float = 0.0024229
    deg_hydr_max: float = 0.875
    eta: float = 5.554
    cem: float = 300.0
    Temp_ref: float = 298.15
    R: float = 8.31446261815324
    Ea: float = 40000.0  # <-- IMPORTANT: ensure this matches your repo / calibration


def load_json_if_exists(path: str):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return None


def scale_domain(xyzt: torch.Tensor, dom: Domain) -> torch.Tensor:
    # map to [0,1] in each dimension to help conditioning
    x, y, z, t = xyzt[:, 0], xyzt[:, 1], xyzt[:, 2], xyzt[:, 3]
    xs = (x - dom.x0) / (dom.x1 - dom.x0)
    ys = (y - dom.y0) / (dom.y1 - dom.y0)
    zs = (z - dom.z0) / (dom.z1 - dom.z0)
    ts = (t - dom.t0) / (dom.t1 - dom.t0)
    return torch.stack([xs, ys, zs, ts], dim=1)


def scale_T(T: torch.Tensor, scales: Scales) -> torch.Tensor:
    # simple scaling (you can match your repo's exact scaling if different)
    return T / scales.T_c


def unscale_T(Ts: torch.Tensor, scales: Scales) -> torch.Tensor:
    return Ts * scales.T_c


# ---------------------------
# Physics: affinity and kinetics
# ---------------------------
def chem_affinity_ref(alpha: torch.Tensor, mat: Material) -> torch.Tensor:
    """
    A_ref(alpha). You should copy the exact form from your repo for perfect match.
    Here is a reasonable placeholder consistent with your parameters B1,B2,eta.
    """
    # Common form: A_ref = B1 * (alpha_max - alpha) * exp(-B2 * alpha) * alpha^eta
    # (Exact form may differ in your repo.)
    a = torch.clamp(alpha, 0.0, mat.deg_hydr_max)
    eps = 1e-12
    return mat.B1 * torch.pow(a + eps, mat.eta) * torch.exp(-mat.B2 * a) * (mat.deg_hydr_max - a)


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
    def __init__(self, in_dim=4, hidden=256, depth=4, alpha_max=0.875):
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
        ts = x[:, 3:4]                    # scaled time in [0,1]
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
    x = torch.rand(N, device=device) * (dom.x1 - dom.x0) + dom.x0
    y = torch.rand(N, device=device) * (dom.y1 - dom.y0) + dom.y0
    z = torch.rand(N, device=device) * (dom.z1 - dom.z0) + dom.z0
    t = torch.rand(N, device=device) * (dom.t1 - dom.t0) + dom.t0
    return torch.stack([x, y, z, t], dim=1)


def sample_initial(N, dom: Domain, device):
    pts = sample_uniform(N, dom, device)
    pts[:, 3] = dom.t0
    return pts


def sample_boundary_dirichlet_T(N, dom: Domain, device):
    """
    Sample points on the boundary surfaces (x=x0/x1, y=y0/y1, z=z0/z1).
    """
    pts = sample_uniform(N, dom, device)
    face = torch.randint(0, 6, (N,), device=device)
    # set one coordinate to boundary based on face index
    pts[face == 0, 0] = dom.x0
    pts[face == 1, 0] = dom.x1
    pts[face == 2, 1] = dom.y0
    pts[face == 3, 1] = dom.y1
    pts[face == 4, 2] = dom.z0
    pts[face == 5, 2] = dom.z1
    return pts


# ---------------------------
# Autograd helpers
# ---------------------------
def grad(outputs, inputs, idx):
    # outputs: [N,1], inputs: [N,4]
    g = torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs),
        retain_graph=True, create_graph=True
    )[0]
    return g[:, idx:idx+1]


def laplacian(u, x):
    # u: [N,1], x: [N,4] (only x,y,z used)
    ux = grad(u, x, 0)
    uy = grad(u, x, 1)
    uz = grad(u, x, 2)
    uxx = grad(ux, x, 0)
    uyy = grad(uy, x, 1)
    uzz = grad(uz, x, 2)
    return uxx + uyy + uzz


# ---------------------------
# Main training
# ---------------------------
def main(
    domain_json="domain.json",
    material_json="material.json",
    T_ic_K=298.15,
    T_bc_K=298.15,
    alpha_ic=1e-6,
    steps=100000,
    lr=1e-4,
    N_pde=5000,
    N_ic=2000,
    N_bc=2000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- temperature data (center point) ---
    t_data = torch.tensor(t_data_h, device=device).view(-1, 1) * 3600.0
    T_data = torch.tensor(T_data_K, device=device).view(-1, 1)

    



    # Load inputs (optional)
    dom = Domain()
    scales = Scales()
    mat = Material()

    xc = 0.5 * (dom.x0 + dom.x1)
    yc = 0.5 * (dom.y0 + dom.y1)
    zc = 0.5 * (dom.z0 + dom.z1)

    x_data = torch.full_like(t_data, xc)
    y_data = torch.full_like(t_data, yc)
    z_data = torch.full_like(t_data, zc)

    xyzt_data_phys = torch.cat([x_data, y_data, z_data, t_data], dim=1)
    xyzt_data_in   = scale_domain(xyzt_data_phys, dom)

    T_data_s = scale_T(T_data, scales)

    dj = load_json_if_exists(domain_json)
    if dj:
        dom.x0, dom.x1 = dj["x"]
        dom.y0, dom.y1 = dj["y"]
        dom.z0, dom.z1 = dj["z"]
        dom.t0, dom.t1 = dj["t"]
        scales.T_c = float(dj.get("T_c", scales.T_c))
        scales.L_c = float(dj.get("L_c", scales.L_c))
        scales.t_c = float(dj.get("t_c", scales.t_c))

    mj = load_json_if_exists(material_json)
    if mj:
        for k, v in mj.items():
            if hasattr(mat, k):
                setattr(mat, k, float(v))

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
        # NOTE: t is the 4th input coordinate in pde_pts
        dalpha_dt = grad(alpha, pde_pts, 3)  # derivative wrt physical t (seconds)
        rate = hydration_rate(alpha, T_K, mat)
        r_alpha = dalpha_dt - rate

        # heat equation residual:
        dTdt = grad(T_K, pde_pts, 3)
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
        pred_data = model(xyzt_data_in)
        loss_data_T = F.mse_loss(pred_data[:, 0:1], T_data_s)

        # weights (tune these!)
        w_phys, w_ic, w_bc, w_data = 1.0, 10.0, 1.0, 1.0
        loss = w_phys * loss_phys + w_ic * loss_ic + w_bc * loss_bc + w_data * loss_data_T

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 200 == 0:
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

            axs[0].plot(t_h, T_c)
            axs[0].set_ylabel("T [K]")
            axs[0].set_title(f"Center evolution (iter {it})")

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


if __name__ == "__main__":
    # Set BC/IC in Kelvin for this debug run:
    # e.g., to see a peak more easily: IC warmer than ambient BC
    main(
        domain_json="domain.json",     # point to your actual domain file
        material_json="material.json", # point to your actual material file
        T_ic_K=298.15,                 # 25C
        T_bc_K=293.15,                 # 20C (recommended debug)
        alpha_ic=1e-6,
        steps=100000,
        lr=1e-4,
        N_pde=5000,
        N_ic=2000,
        N_bc=2000,
    )
