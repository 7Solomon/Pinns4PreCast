class NodeExecutionError(Exception):
    """Raised when a specific node fails."""
    def __init__(self, node_id: str, node_type: str, original_error: Exception):
        self.node_id = node_id
        self.node_type = node_type
        self.original_error = original_error
        super().__init__(f"Node '{node_id}' ({node_type}) failed: {str(original_error)}")

class MissingInputError(Exception):
    """Raised when a required input is missing."""
    pass