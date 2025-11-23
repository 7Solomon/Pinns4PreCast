from pina.problem import TimeDependentProblem, SpatialProblem
from pina.domain import CartesianDomain
from pina.equation import Equation, FixedValue
from pina import Condition
from pina.condition import InputTargetCondition
from pina.operator import grad, laplacian
import torch

#from material import ConcreteData
#from domain import DomainVariables
from src.utils import unscale_T, unscale_alpha
from src.state_management.state import State
#material_data = ConcreteData()
#domain_vars = DomainVariables()


def save_inverse_T(T, eps=1e-4):
    return 1.0 / torch.clamp(T, min=eps)
def chem_affinity_ref(deg_hydration):
        """
        calculates the chemical affinity at reference temperature
        """
        return (State().material.B1 * ((State().material.B2 / State().material.deg_hydr_max) + deg_hydration) * (State().material.deg_hydr_max - deg_hydration) 
                * torch.exp(-State().material.eta * deg_hydration / State().material.deg_hydr_max))
def hydration_rate(input_, output_):
    T_unscaled = unscale_T(output_.extract(["T"]))
    #alpha_unscaled = unscale_alpha(output_.extract(["alpha"]))
    alpha_unscaled = output_.extract(["alpha"])
    alpha_safe = torch.clamp(alpha_unscaled, 0.0, State().material.deg_hydr_max)

    exponent = -State().material.Ea * (save_inverse_T(T_unscaled) - (1/State().material.Temp_ref)) / State().material.R
    exponent = torch.clamp(exponent, min=-150, max=75)
    return chem_affinity_ref(alpha_safe) * torch.exp(exponent)

def alpha_pde(input_, output_):
    hydration_rate_val = hydration_rate(input_, output_)
    alpha = output_.extract(["alpha"])
    d_alpha_dt = grad(alpha, input_, components=["alpha"], d=["t"])
    
    return  hydration_rate_val * State().domain.t_c - d_alpha_dt  # NON DIM ???, da beide mal 1/tc

def heat_generation_through_hydration(input_, output_):
    return State().material.Q_pot * hydration_rate(input_, output_) * State().material.cem

def heat_pde(input_, output_):
    #print('input_.requires_grad:', input_.requires_grad)
    #print('input_.is_leaf:', input_.is_leaf)
    #print('output_.requires_grad:', output_.requires_grad)


    u_t = grad(output_, input_, components=["T"], d=["t"])
    laplacian_u = laplacian(output_, input_, components=["T"], d=["x", "y", "z"])

    pi_one = (State().material.k * State().domain.t_c) / (State().domain.L_c**2 * State().material.cp * State().material.rho)
    pi_two = State().domain.t_c / (State().material.cp * State().material.rho * State().domain.T_c)

    heat_source = heat_generation_through_hydration(input_, output_)
    return u_t - pi_one * laplacian_u - pi_two * heat_source

    
class HeatODE(TimeDependentProblem, SpatialProblem):
    output_variables = ["T", "alpha"]
    

    spatial_domain = None
    temporal_domain = None
    conditions = None
    
    def __init__(self):
        start_x = State().domain.x[0]
        end_x = State().domain.x[1]

        start_y = State().domain.y[0]
        end_y = State().domain.y[1]

        start_z = State().domain.z[0]
        end_z = State().domain.z[1]

        start_t = State().domain.t[0]
        end_t = State().domain.t[1]

        self.spatial_domain = CartesianDomain({"x": [start_x, end_x], "y": [start_y, end_y], "z": [start_z, end_z]})
        self.temporal_domain = CartesianDomain({"t": [start_t, end_t]})

        #self.input_variables = ["x", "y", "z", "t"] # Ensure input variables are defined
        
        # Define sub-domains
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
        
        self.conditions = {
            "physi_T": Condition(domain="D", equation=Equation(heat_pde)),
            "physi_alpha": Condition(domain="D", equation=Equation(alpha_pde)),        
        }
        
        super().__init__()

