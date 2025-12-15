from typing import Type, Dict, Any
from pydantic import BaseModel
from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node, NodeRegistry

def create_config_node(
    config_model: Type[BaseModel], 
    node_type_id: str, 
    display_name: str, 
    description: str = "Configuration Node"
):
    """
    Dynamically creates and registers a Node class for a given Pydantic config model.
    """
    
    class ConfigNode(Node):
        """Auto-generated Config Node"""
        
        @classmethod
        def get_input_ports(cls) -> Dict[str, Port]:
            return {}  # Config nodes usually have no inputs

        @classmethod
        def get_output_ports(cls) -> Dict[str, Port]:
            # Always outputs the config object
            return {
                "config": Port("config", PortType.CONFIG, description=f"Instance of {config_model.__name__}")
            }

        @classmethod
        def get_metadata(cls) -> NodeMetadata:
            return NodeMetadata(
                category="Configs",
                display_name=display_name,
                description=description,
                icon="settings"  # Default icon
            )

        @classmethod
        def get_config_schema(cls):
            return config_model

        def execute(self) -> Dict[str, Any]:
            return {"config": self.config}

    # Rename the class for debugging clarity
    ConfigNode.__name__ = f"{config_model.__name__}Node"
    

    NodeRegistry.register(ConfigNode, node_type=node_type_id)
    
    return ConfigNode
