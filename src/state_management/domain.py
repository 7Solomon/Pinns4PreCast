from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


from src.class_definition.base_state import BaseState


@dataclass
class DomainVariables(BaseState):

    x: List[float] = field(
        default_factory=lambda: [0.0, 0.4],
        metadata={"label": "X Range", "unit": "m", "type": "range"},
    )
    y: List[float] = field(
        default_factory=lambda: [0.0, 0.8],
        metadata={"label": "Y Range", "unit": "m", "type": "range"},
    )
    z: List[float] = field(
        default_factory=lambda: [0.0, 0.4],
        metadata={"label": "Z Range", "unit": "m", "type": "range"},
    )
    t: List[float] = field(
        default_factory=lambda: [0.0, 86400.0],
        metadata={"label": "Time Range", "unit": "s", "type": "range"},
    )  # time in seconds

    T_c: float = field(
        default=50.0,
        metadata={"label": "Characteristic Temperature", "unit": "°C", "type": "number"},
    )
    L_c: float = field(
        default=0.8,
        metadata={"label": "Characteristic Length", "unit": "m", "type": "number"},
    )
    t_c: float = field(
        default=86400.0,
        metadata={"label": "Characteristic Time", "unit": "s", "type": "number"},
    )
    
    ### Temporary sensor locations [x,y,z]
    TEMP_SENS_POINTS: Dict[str, List[float]] = field(
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
        metadata={"label": "Temperature Sensors", "type": "sensor_dict"},
    )