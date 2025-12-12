from pydantic import BaseModel, Field
from src.class_definition.base_state import BaseState

class BasePhysics(BaseModel, BaseState):
    g: float = Field(default=9.81, description="Gravity [m/s^2]")
    Temp_ref: float = Field(default=298.15, description="Reference Temp [K]")
    R: float = Field(default=8.31446261815324, description="Gas Constant [J/(mol*K)]")



class ConcreteData(BasePhysics):
    name: str = Field(
        default="Normaler Beton", 
        title="Material Name"
    )

    cp: float = Field(
        default=8.5e2, 
        title="Specific Heat", 
        json_schema_extra={"unit": "J/(kg·K)"}
    )
    
    rho: float = Field(
        default=2.4e3, 
        title="Density", 
        json_schema_extra={"unit": "kg/m³"}
    )
    
    k: float = Field(
        default=1.4, 
        title="Thermal Conductivity", 
        json_schema_extra={"unit": "W/(m·K)"}
    )

    Q_pot: float = Field(
        default=500e3, 
        title="Potential Heat", 
        json_schema_extra={"unit": "J/kg"}
    )
    
    B1: float = Field(
        default=0.0002916, 
        title="B1 Reaction Rate", 
        json_schema_extra={"unit": "1/s"}
    )
    
    B2: float = Field(
        default=0.0024229, 
        title="B2 Reaction Rate", 
        json_schema_extra={"unit": "1/s"}
    )
    
    deg_hydr_max: float = Field(
        default=0.875, 
        title="Max Degree Hydration"
    )
    
    eta: float = Field(
        default=5.554, 
        title="Eta"
    )

    cem: float = Field(
        default=300.0, 
        title="Cement Content", 
        json_schema_extra={"unit": "kg/m³"}
    )

    @property
    def Ea(self) -> float:
        return 5.653 * self.R  # [J/mol]

    @classmethod
    def load(cls, path: str):
        import os
        if not os.path.exists(path):
            return cls.create_default(path)
        with open(path, 'r') as f:
            return cls.model_validate_json(f.read())
