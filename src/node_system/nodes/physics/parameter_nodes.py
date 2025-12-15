from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node
from src.state_management.domain import DomainVariables
from src.state_management.material import ConcreteData

from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node
from src.state_management.config import DatasetConfig, ModelConfig



@register_node("concrete_material")
class ConcreteMaterialNode(Node):
    @classmethod
    def get_input_ports(cls):
        return []

    @classmethod
    def get_output_ports(cls):
        return [Port("material", PortType.MATERIAL)]

    @classmethod
    def get_metadata(cls):
        return NodeMetadata("Parameters", "Concrete Material", "Defines density, conductivity, etc.", icon="cube")

    @classmethod
    def get_config_schema(cls):
        return ConcreteData

    def execute(self):
        # The config IS the ConcreteData object because of Pydantic magic
        return {"material": self.config}


@register_node("spatial_domain")
class DomainNode(Node):
    @classmethod
    def get_input_ports(cls):
        return []

    @classmethod
    def get_output_ports(cls):
        return [Port("domain", PortType.DOMAIN)]

    @classmethod
    def get_metadata(cls):
        return NodeMetadata("Parameters", "Space/Time Domain", "Defines x,y,z,t bounds", icon="globe")

    @classmethod
    def get_config_schema(cls):
        return DomainVariables

    def execute(self):
        return {"domain": self.config}
