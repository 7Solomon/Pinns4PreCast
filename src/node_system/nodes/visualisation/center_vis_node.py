from lightning.pytorch.callbacks import Callback
from src.node_system.core import Node, NodeMetadata, Port, PortType, register_node
from src.node_system.event_bus import get_event_bus, EventType, Event
import torch
import numpy as np

def compute_center_curves(model, dom, mat, device="cpu", n_points=100):
    """
    Computes evolution of T, Alpha, and Q at the geometric center of the domain.
    """
    # 1. Create time vector
    t_plot = torch.linspace(dom.t0, dom.t1, n_points, device=device).view(-1, 1)
    
    # 2. Get Center Coordinates
    xc = 0.5 * (dom.x0 + dom.x1)
    yc = 0.5 * (dom.y0 + dom.y1)
    zc = 0.5 * (dom.z0 + dom.z1)
    
    # 3. Create Batch (Space fixed, Time varies)
    x = torch.tensor([[xc]], device=device).expand_as(t_plot)
    y = torch.tensor([[yc]], device=device).expand_as(t_plot)
    z = torch.tensor([[zc]], device=device).expand_as(t_plot)
    
    # 4. Prepare Input for Autograd (requires_grad=True for time derivative)
    xyzt_phys = torch.cat([x, y, z, t_plot], dim=1).requires_grad_(True)
    
    # 5. Scale Inputs (Model expects 0-1 range)
    # Inline scaling logic for portability
    xyzt_scaled = torch.zeros_like(xyzt_phys)
    xyzt_scaled[:, 0] = (xyzt_phys[:, 0] - dom.x0) / (dom.x1 - dom.x0)
    xyzt_scaled[:, 1] = (xyzt_phys[:, 1] - dom.y0) / (dom.y1 - dom.y0)
    xyzt_scaled[:, 2] = (xyzt_phys[:, 2] - dom.z0) / (dom.z1 - dom.z0)
    xyzt_scaled[:, 3] = (xyzt_phys[:, 3] - dom.t0) / (dom.t1 - dom.t0)
    
    # 6. Model Prediction
    pred = model(xyzt_scaled)
    T_s = pred[:, 0:1]
    alpha = pred[:, 1:2]
    
    # 7. Unscale Temperature
    T_K = T_s * dom.T_c
    
    # 8. Compute Time Derivative (d_alpha / d_t)
    dalpha_dt = torch.autograd.grad(
        alpha, xyzt_phys,
        grad_outputs=torch.ones_like(alpha),
        retain_graph=False,
        create_graph=False
    )[0][:, 3:4]
    
    # 9. Compute Heat Generation (q)
    # q = Q_pot * cem * d_alpha/dt
    q = mat.Q_pot * mat.cem * dalpha_dt
    
    return {
        "time": (t_plot.detach().cpu().numpy().flatten() / 3600.0).tolist(), # Hours
        "temperature": T_K.detach().cpu().numpy().flatten().tolist(),        # Kelvin
        "alpha": alpha.detach().cpu().numpy().flatten().tolist(),            # 0-1
        "q": q.detach().cpu().numpy().flatten().tolist()                     # W/m^3
    }


class CenterProbeCallback(Callback):
    def __init__(self, run_id, material, domain):
        self.run_id = run_id
        self.material = material
        self.domain = domain
        self.event_bus = get_event_bus()

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of every epoch"""

        # 2. Compute Physics
        data = compute_center_curves(
            model=pl_module, 
            dom=self.domain, 
            mat=self.material, 
            device=pl_module.device
        )
        
        # 3. Add metadata
        data['epoch'] = trainer.current_epoch
        data['run_id'] = self.run_id

        self.event_bus.publish_sync(Event(
            type=EventType.CENTER_PROBE_DATA, 
            run_id=self.run_id, 
            data=data
        ))

@register_node("center_probe_vis")
class CenterProbeNode(Node):
    @classmethod
    def get_metadata(cls):
        return NodeMetadata(
            category="Visualization",
            display_name="Center Point Probe",
            description="Tracks T, Alpha, and Q at the domain center over time.",
            icon="activity"
        )

    @classmethod
    def get_input_ports(cls):
        return [
            Port("domain", PortType.CONFIG),
            Port("material", PortType.CONFIG),
            Port("scales_config", PortType.CONFIG),
        ]

    @classmethod
    def get_output_ports(cls):
        return [
            Port("callback", PortType.CALLBACK) # Outputs a Lightning Callback
        ]

    def execute(self):
        run_id = self.context.get("run_id", "default")

        domain=  self.inputs.get("domain"),
        material =  self.inputs.get("material"),
        
        callback = CenterProbeCallback(run_id, material, domain)
        
        return {
            "callback": callback
        }
