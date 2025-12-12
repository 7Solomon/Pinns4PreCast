from typing import List, Dict
from pydantic import BaseModel, Field

from src.class_definition.base_state import BaseState

class DomainVariables(BaseModel, BaseState):
    x: List[float] = Field(
        default_factory=lambda: [0.0, 0.4],
        title="X Range",
        json_schema_extra={"unit": "m", "type": "range"}
    )
    
    y: List[float] = Field(
        default_factory=lambda: [0.0, 0.8],
        title="Y Range",
        json_schema_extra={"unit": "m", "type": "range"}
    )
    
    z: List[float] = Field(
        default_factory=lambda: [0.0, 0.4],
        title="Z Range",
        json_schema_extra={"unit": "m", "type": "range"}
    )
    
    t: List[float] = Field(
        default_factory=lambda: [0.0, 86400.0],
        title="Time Range",
        json_schema_extra={"unit": "s", "type": "range"}
    )  # time in seconds

    T_c: float = Field(
        default=50.0,
        title="Characteristic Temperature",
        json_schema_extra={"unit": "°C"}
    )
    
    L_c: float = Field(
        default=0.8,
        title="Characteristic Length",
        json_schema_extra={"unit": "m"}
    )
    
    t_c: float = Field(
        default=86400.0,
        title="Characteristic Time",
        json_schema_extra={"unit": "s"}
    )
    
    ### Temporary sensor locations [x,y,z]
    TEMP_SENS_POINTS: Dict[str, List[float]] = Field(
        default_factory=lambda: {
            'T1': [0.2, 0, 0.2],
            'T2': [0.4, 0, 0],
            'T3': [0.2, 0.1, 0.2],
            'T4': [0.2, 0.2, 0.2],
            'T5': [0.2, 0.4, 0],
            'T6': [0.2, 0.4, 0.1],
            'T7': [0.2, 0.4, 0.2],
            'T8': [0.2, 0.4, 0.3],
            'T9': [0.2, 0.4, 0.4],
            'T10': [0.4, 0.4, 0.2]
        },
        title="Temperature Sensors",
        description="Dictionary of sensor names and their [x,y,z] coordinates"
    )

    @classmethod
    def load(cls, path: str):
        """Loads the domain config from a JSON file."""
        import os
        if not os.path.exists(path):
            return cls.create_default(path)
        with open(path, 'r') as f:
            return cls.model_validate_json(f.read())
