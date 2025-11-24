from dataclasses import dataclass, field

from src.class_definition.base_state import BaseState
    

@dataclass
class BasePhysics(BaseState):
    g:float = 9.81  # [m/s^2]
    Temp_ref: float = 298.15  # [K] (25°C)
    R: float = 8.31446261815324  # [J/(mol*K)]

@dataclass
class ConcreteData(BasePhysics):
    name: str = field(default="Normaler Beton", metadata={"label": "Material Name", "type": "text"})

    cp: float = field(default=8.5e2, metadata={"label": "Specific Heat", "unit": "J/(kg·K)", "type": "number"})
    rho: float = field(default=2.4e3, metadata={"label": "Density", "unit": "kg/m³", "type": "number"})
    k: float = field(default=1.4, metadata={"label": "Thermal Conductivity", "unit": "W/(m·K)", "type": "number"})

    Q_pot: float = field(default=500e3, metadata={"label": "Potential Heat", "unit": "J/kg", "type": "number"})
    B1: float = field(default=0.0002916, metadata={"label": "B1 Reaction Rate", "unit": "1/s", "type": "number"})
    B2: float = field(default=0.0024229, metadata={"label": "B2 Reaction Rate", "unit": "1/s", "type": "number"})
    deg_hydr_max: float = field(default=0.875, metadata={"label": "Max Degree Hydration", "type": "number"})
    eta: float = field(default=5.554, metadata={"label": "Eta", "type": "number"})

    cem: float = field(default=300.0, metadata={"label": "Cement Content", "unit": "kg/m³", "type": "number"})

    @property
    def Ea(self) -> float:
        return 5.653 * self.R  # [J/mol]

