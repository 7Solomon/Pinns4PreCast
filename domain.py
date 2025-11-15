from dataclasses import dataclass, field
from typing import List

@dataclass
class DomainVariables:
    x: List[float] = field(default_factory=lambda: [0.0, 1.0])
    y: List[float] = field(default_factory=lambda: [0.0, 1.0])
    z: List[float] = field(default_factory=lambda: [0.0, 1.0])
    t: List[float] = field(default_factory=lambda: [0.0, (86400.0 * 24)])  # time in seconds

    T_c: float = 50.0
    L_c: float = 1.0
    t_c: float = (86400.0 * 24)