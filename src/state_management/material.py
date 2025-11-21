from dataclasses import dataclass

from src.class_definition.base_state import BaseState
    

@dataclass
class BasePhysics(BaseState):
    g:float = 9.81  # [m/s^2]
    Temp_ref: float = 298.15  # [K] (25°C)
    R: float = 8.31446261815324  # [J/(mol*K)]

@dataclass
class ConcreteData(BasePhysics):
    name: str = "Normaler Beton"

    cp: float = 8.5e2 # [J/(kg*K)]
    rho: float = 2.4e3  # [kg/m^3]
    k: float = 1.4 # [W/(m*K)]

    Q_pot: float = 500e3  # [J/kg]
    B1: float = 0.0002916  # [1/s]
    B2: float = 0.0024229  # [1/s]
    deg_hydr_max: float = 0.875  # [-]
    eta: float = 5.554  # [-]

    cem: float = 300.0  # [kg/m^3]

    @property
    def Ea(self) -> float:
        return 5.653 * self.R  # [J/mol]

