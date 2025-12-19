from typing import Any, Dict, List

def port_to_dict(ports: Any) -> Dict[str, Any]:
    """
    Robust converter that handles both List[Port] and Dict[str, Port].
    Returns rich metadata { "type": str, "required": bool }.
    """
    result = {}
    
    # Helper to format a single port object
    def format_port(port_obj):
        port_type = "any"
        required = True  # Default to True
        
        if hasattr(port_obj, "port_type"):
            port_type = port_obj.port_type.value
            
        if hasattr(port_obj, "required"):
            required = port_obj.required
            
        return {"type": port_type, "required": required}

    # Case 1: Dict input (e.g. {"material": Port(...), ...})
    if isinstance(ports, dict):
        for name, port_obj in ports.items():
            result[name] = format_port(port_obj)

    # Case 2: List input (e.g. [Port(...), ...])
    elif isinstance(ports, list):
        for i, p in enumerate(ports):
            # Determine name
            name = getattr(p, "name", f"port_{i}")
            result[name] = format_port(p)
                
    return result

def get_config_schema_json(node_cls):
    """Extracts the JSON Schema from the Pydantic config model."""
    schema_model = node_cls.get_config_schema()
    if schema_model:
        try:
            if hasattr(schema_model, "model_json_schema"):
                return schema_model.model_json_schema(mode='serialization')
            elif hasattr(schema_model, "schema"):
                return schema_model.schema()
        except Exception as e:
            print(f"Error generating schema for {node_cls.__name__}: {e}")
            return None
            
    return None
