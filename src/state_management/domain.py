from dataclasses import dataclass, field, asdict
from typing import List, Dict


from src.class_definition.base_state import BaseState



@dataclass
class DomainVariables(BaseState):
    x: List[float] = field(default_factory=lambda: [0.0, 0.4])
    y: List[float] = field(default_factory=lambda: [0.0, 0.8])
    z: List[float] = field(default_factory=lambda: [0.0, 0.4])
    t: List[float] = field(default_factory=lambda: [0.0, 86400.0])  # time in seconds

    T_c: float = 50.0
    L_c: float = 0.8
    t_c: float = 86400.0
    
    ### Temporary sensor locations [x,y,z]
    TEMP_SENS_POINTS: Dict[str,List[float]] = field(default_factory=lambda: {
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
    })
