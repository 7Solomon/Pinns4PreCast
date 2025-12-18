

import time
from src.node_system.core import Node, NodeMetadata, Port, PortType


class FlexibleWidgetAggregatorNode(Node):
    """
    Version mit beliebig vielen Inputs.
    Besser für große Dashboards.
    """
    
    @classmethod
    def get_input_ports(cls):
        # Erstellt 10 optionale Ports
        return {
            f"widget_{i}": Port(f"widget_{i}", PortType.SPEC, required=False)
            for i in range(1, 11)
        }

    @classmethod
    def get_output_ports(cls):
        return {
            "widgets": Port("widgets", PortType.ANY),
            "count": Port("count", PortType.CONFIG)  # Bonus: Anzahl der Widgets
        }

    @classmethod
    def get_metadata(cls):
        return NodeMetadata(
            category="Visualization",
            display_name="Widget Dashboard",
            description="Sammelt bis zu 10 Visualisierungen",
            icon="layout-dashboard"
        )

    def execute(self):
        widgets = []
        
        for i in range(1, 11):
            key = f"widget_{i}"
            widget = self.inputs.get(key)
            if widget is not None:
                widget_with_meta = {
                    **widget,
                    "position": i,  # Reihenfolge merken
                    "timestamp": time.time()
                }
                widgets.append(widget_with_meta)
        
        return {
            "widgets": widgets,
            "count": len(widgets)
        }

