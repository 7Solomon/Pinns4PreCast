from pina.problem import TimeDependentProblem, SpatialProblem
from pina.domain import CartesianDomain
from pina.equation import Equation, FixedValue
from pina import Condition
from pina.operator import grad, laplacian
import torch

from material import ConcreteData
from domain import DomainVariables
material_data = ConcreteData()
domain_vars = DomainVariables()


def save_inverse_T(T, eps=1e-4):
    return 1.0 / torch.clamp(T, min=eps)
def chem_affinity_ref(deg_hydration):
        """
        calculates the chemical affinity at reference temperature
        """
        return (material_data.B1 * ((material_data.B2 / material_data.deg_hydr_max) + deg_hydration) * (material_data.deg_hydr_max - deg_hydration) 
                * torch.exp(-material_data.eta * deg_hydration / material_data.deg_hydr_max))

def heat_generation_through_hydration(alpha, T):
    exponent = -material_data.Ea * (save_inverse_T(T)-(1/material_data.Temp_ref)) / material_data.R
    #print(f"Exponent min: {torch.min(exponent.tensor)}, max: {torch.max(exponent.tensor)}")
    exponent = torch.clamp(exponent, min=-700, max=50)
    return material_data.Q_pot * chem_affinity_ref(alpha) * torch.exp(exponent)

def heat_equation(input_, output_):
    T = output_.extract(["T"])
    alpha = output_.extract(["alpha"])

    u_t = grad(output_, input_, components=["T"], d=["t"])
    laplacian_u = laplacian(output_, input_, components=["T"], d=["x", "y", "z"])

    pi_one = (material_data.k * domain_vars.t_c) / (domain_vars.L_c**2 * material_data.cp * material_data.rho)
    pi_two = domain_vars.t_c / (material_data.cp * material_data.rho * domain_vars.T_c)

    heat_source = heat_generation_through_hydration(alpha, T)
    return u_t - pi_one * laplacian_u - pi_two * heat_source


def alpha_ic(input_, output_):
    return output_.extract(["alpha"])
    
class HeatODE(TimeDependentProblem, SpatialProblem):
    output_variables = ["T", "alpha"]

    start_x = domain_vars.x[0] / domain_vars.L_c
    end_x = domain_vars.x[1] / domain_vars.L_c

    start_y = domain_vars.y[0] / domain_vars.L_c
    end_y = domain_vars.y[1] / domain_vars.L_c

    start_z = domain_vars.z[0] / domain_vars.L_c
    end_z = domain_vars.z[1] / domain_vars.L_c

    start_t = domain_vars.t[0] / domain_vars.t_c
    end_t = domain_vars.t[1] / domain_vars.t_c

    spatial_domain = CartesianDomain({"x": [start_x, end_x], "y": [start_y, end_y], "z": [start_z, end_z]})
    temporal_domain = CartesianDomain({"t": [start_t, end_t]})

    domain = {
        "D": CartesianDomain({"x": [start_x, end_x], "y": [start_y, end_y], "z": [start_z, end_z], "t": [start_t, end_t]}),
        "left": CartesianDomain({"x": start_x, "y": [start_y, end_y], "z": [start_z, end_z], "t": [start_t, end_t]}),
        "right": CartesianDomain({"x": end_x, "y": [start_y, end_y], "z": [start_z, end_z], "t": [start_t, end_t]}),
        "bottom": CartesianDomain({"x": [start_x, end_x], "y": start_y, "z": [start_z, end_z], "t": [start_t, end_t]}),
        "top": CartesianDomain({"x": [start_x, end_x], "y": end_y, "z": [start_z, end_z], "t": [start_t, end_t]}),
        "front": CartesianDomain({"x": [start_x, end_x], "y": [start_y, end_y], "z": start_z, "t": [start_t, end_t]}),
        "back": CartesianDomain({"x": [start_x, end_x], "y": [start_y, end_y], "z": end_z, "t": [start_t, end_t]}),
        "initial": CartesianDomain({"x": [start_x, end_x], "y": [start_y, end_y], "z": [start_z, end_z], "t": start_t}),
    }
    conditions = {
        "physi": Condition(
            domain="D", equation=Equation(heat_equation)),
        "alpha_ic": Condition(domain="initial", equation=Equation(alpha_ic)),
    }

