import torch
from pina.problem import TimeDependentProblem, SpatialProblem
from pina.domain import CartesianDomain
from pina.equation import Equation
from pina import Condition
from pina.operator import grad, laplacian

from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node


class ConcretePhysicsEquations:
    """
    Holds the material and domain constants required for the PDE equations.
    This acts as a 'context' for the equation functions.
    """
    def __init__(self, material, domain):
        self.mat = material
        self.dom = domain
        
    def save_inverse_T(self, T, eps=1e-4):
        return 1.0 / torch.clamp(T, min=eps)

    def chem_affinity_ref(self, deg_hydration):
        """Calculates the chemical affinity at reference temperature"""
        return (self.mat.B1 * ((self.mat.B2 / self.mat.deg_hydr_max) + deg_hydration) * 
                (self.mat.deg_hydr_max - deg_hydration) * 
                torch.exp(-self.mat.eta * deg_hydration / self.mat.deg_hydr_max))

    def hydration_rate(self, input_, output_):
        T_unscaled = self.dom.unscale_T(output_.extract(["T"]), self.mat.Temp_ref)
        alpha_unscaled = output_.extract(["alpha"])
        alpha_safe = torch.clamp(alpha_unscaled, 0.0, self.mat.deg_hydr_max)

        exponent = -self.mat.Ea * (self.save_inverse_T(T_unscaled) - (1/self.mat.Temp_ref)) / self.mat.R
        exponent = torch.clamp(exponent, min=-150, max=75)
        return self.chem_affinity_ref(alpha_safe) * torch.exp(exponent)

    def alpha_pde(self, input_, output_):
        hydration_rate_val = self.hydration_rate(input_, output_)
        alpha = output_.extract(["alpha"])
        d_alpha_dt = grad(alpha, input_, components=["alpha"], d=["t"])

        d_alpha_dt = torch.clamp(d_alpha_dt, min=0.0) 

        return hydration_rate_val * self.dom.t_c - d_alpha_dt

    def heat_generation_through_hydration(self, input_, output_):
        return self.mat.Q_pot * self.hydration_rate(input_, output_) * self.mat.cem

    def heat_pde(self, input_, output_):
        u_t = grad(output_, input_, components=["T"], d=["t"])
        laplacian_u = laplacian(output_, input_, components=["T"], d=["x", "y", "z"])

        pi_one = (self.mat.k * self.dom.t_c) / (self.dom.L_c**2 * self.mat.cp * self.mat.rho)
        pi_two = self.dom.t_c / (self.mat.cp * self.mat.rho * self.dom.T_c)

        heat_source = self.heat_generation_through_hydration(input_, output_)
        return u_t - pi_one * laplacian_u - pi_two * heat_source


class HeatODE(TimeDependentProblem, SpatialProblem):
    output_variables = ["T", "alpha"]

    spatial_domain = None
    temporal_domain = None
    conditions = None
    
    
    def __init__(self, material, domain):
        """
        Args:
            material: ConcreteData object (from Node input)
            domain: DomainVariables object (from Node input)
        """
        start_x, end_x = domain.x
        start_y, end_y = domain.y
        start_z, end_z = domain.z
        start_t, end_t = domain.t

        self.spatial_domain = CartesianDomain({"x": [start_x, end_x], "y": [start_y, end_y], "z": [start_z, end_z]})
        self.temporal_domain = CartesianDomain({"t": [start_t, end_t]})

        #self.input_variables = ["x", "y", "z", "t"] DONT NEEDEd

        self.domain = {
            "D": CartesianDomain({"x": [start_x, end_x], "y": [start_y, end_y], "z": [start_z, end_z], "t": [start_t, end_t]}),
            "left": CartesianDomain({"x": start_x, "y": [start_y, end_y], "z": [start_z, end_z], "t": [start_t, end_t]}),
            "right": CartesianDomain({"x": end_x, "y": [start_y, end_y], "z": [start_z, end_z], "t": [start_t, end_t]}),
            "bottom": CartesianDomain({"x": [start_x, end_x], "y": start_y, "z": [start_z, end_z], "t": [start_t, end_t]}),
            "top": CartesianDomain({"x": [start_x, end_x], "y": end_y, "z": [start_z, end_z], "t": [start_t, end_t]}),
            "front": CartesianDomain({"x": [start_x, end_x], "y": [start_y, end_y], "z": start_z, "t": [start_t, end_t]}),
            "back": CartesianDomain({"x": [start_x, end_x], "y": [start_y, end_y], "z": end_z, "t": [start_t, end_t]}),
            "initial": CartesianDomain({"x": [start_x, end_x], "y": [start_y, end_y], "z": [start_z, end_z], "t": start_t}),
        }
        
        self.physics = ConcretePhysicsEquations(material, domain)  # THIS IS SO TAHT IT NICELY WORKS WITH NODES

        self.conditions = {
            "physi_T": Condition(domain="D", equation=Equation(self.physics.heat_pde)),
            "physi_alpha": Condition(domain="D", equation=Equation(self.physics.alpha_pde)),        
        }
        
        super().__init__()


@register_node("heat_pde")
class HeatPDENode(Node):
    @classmethod
    def get_input_ports(cls):
        return [
            Port("material", PortType.MATERIAL),
            Port("domain", PortType.DOMAIN)
        ]

    @classmethod
    def get_output_ports(cls):
        return [Port("problem_instance", PortType.PROBLEM)]

    @classmethod
    def get_metadata(cls):
        return NodeMetadata(
            category="Problem",
            display_name="Concrete Heat Hydration",
            description="Assembles Material+Domain into the HeatODE",
            icon="fire"
        )
    
    @classmethod
    def get_config_schema(cls):
        return None 

    def execute(self):
        # 1. Get Inputs
        material_obj = self.inputs["material"]
        domain_obj = self.inputs["domain"]
        
        # 2. Instantiate Problem directly (Dependency Injection)
        problem = HeatODE(material=material_obj, domain=domain_obj)
        
        return {"problem_instance": problem}