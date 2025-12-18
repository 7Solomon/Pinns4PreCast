# ==============================================================================
# Example: Complete Training Pipeline with Loss Visualization
# ==============================================================================

"""
This example shows how to construct a graph that:
1. Sets up training components
2. Captures the run_id from the logger
3. Connects it to visualization nodes
4. Returns widget specs to the frontend
"""

from src.node_system.core import Node, NodeGraph

# Create graph
graph = NodeGraph()

# === PARAMETERS ===
domain_node = graph.add_node("spatial_domain", "domain_1")
material_node = graph.add_node("concrete_material", "material_1")

# === PROBLEM ===
problem_node = graph.add_node("heat_pde", "problem_1")
graph.connect("domain_1", "domain", "problem_1", "domain")
graph.connect("material_1", "material", "problem_1", "material")

# === CONFIGS ===
dataset_config = graph.add_node("composite_dataset_gen_config", "ds_config")
model_config = graph.add_node("composite_model_config", "model_config")
training_config = graph.add_node("training_config", "train_config")
dataloader_config = graph.add_node("dataloader_config", "dl_config")

# === DATASET & LOADER ===
dataset = graph.add_node("deeponet_dataset", "dataset_1")
graph.connect("problem_1", "problem", "dataset_1", "problem")
graph.connect("material_1", "material", "dataset_1", "material")
graph.connect("domain_1", "domain", "dataset_1", "domain")
graph.connect("ds_config", "config", "dataset_1", "composite_dataset_config")

dataloader = graph.add_node("deeponet_dataloader", "dataloader_1")
graph.connect("dataset_1", "dataset", "dataloader_1", "dataset")
graph.connect("dl_config", "config", "dataloader_1", "data_loadere_config")

# === MODEL ===
model = graph.add_node("flex_deeponet", "model_1")
graph.connect("model_config", "config", "model_1", "model_config")

# === SOLVER ===
solver = graph.add_node("deeponet_solver", "solver_1")
graph.connect("model_1", "model", "solver_1", "model")
graph.connect("problem_1", "problem", "solver_1", "problem")
graph.connect("train_config", "config", "solver_1", "training_config")

# === LOGGER (This generates the run_id) ===
logger = graph.add_node("dashboard_logger", "logger_1", config={
    "save_dir": "content/runs",
    "save_graph": True
})
# Note: The logger needs access to the graph itself for saving
# This is a special case - we'll handle it in the API

# === VISUALIZATION (Connects to run_id from logger) ===
loss_viz = graph.add_node("loss_curve", "loss_viz_1", config={
    "title": "Training Losses",
    "metrics": ["loss", "loss_physics", "loss_bc", "loss_ic"],
    "refresh_rate": 2000
})
# KEY CONNECTION: run_id flows from logger to visualization
graph.connect("logger_1", "run_id", "loss_viz_1", "run_id")

# === TRAINER (The execution target) ===
trainer = graph.add_node("lightning_trainer", "trainer_1")
graph.connect("solver_1", "solver", "trainer_1", "solver")
graph.connect("dataloader_1", "dataloader", "trainer_1", "dataloader")
graph.connect("train_config", "config", "trainer_1", "training_config")
graph.connect("logger_1", "logger", "trainer_1", "logger")

# When this graph executes:
# 1. Logger creates run_id → outputs it
# 2. Loss visualization node receives run_id → creates widget spec
# 3. Trainer runs with logger attached
# 4. Frontend polls the run_id for real-time updates

# Execute and get results
result = graph.execute(output_node="trainer_1")

# Access the widget spec
widget_spec = graph.nodes["loss_viz_1"].get_output("widget_spec")
print("Widget Spec:", widget_spec)
# Output: {
#   "type": "loss_curve",
#   "node_id": "loss_viz_1", 
#   "run_id": "2025-12-17_14-23-45",
#   "title": "Training Losses",
#   "metrics": ["loss", "loss_physics", "loss_bc", "loss_ic"],
#   "refresh_rate": 2000,
#   "data_endpoint": "/monitor/metrics/2025-12-17_14-23-45"
# }


# ==============================================================================
# Alternative: Pre-specify run_id (for reproducible experiments)
# ==============================================================================

logger_with_id = graph.add_node("dashboard_logger", "logger_2", config={
    "save_dir": "content/runs",
    "version": "experiment_001",  # Fixed run ID
    "save_graph": True
})

# Now the run_id is deterministic: "experiment_001"
# Useful for:
# - Resuming training
# - Comparing multiple runs
# - Debugging specific experiments


# ==============================================================================
# Pattern: Multiple Visualizations
# ==============================================================================

# You can have multiple visualization nodes connected to the same run_id

loss_viz_main = graph.add_node("loss_curve", "loss_main", config={
    "title": "Main Losses",
    "metrics": ["loss", "loss_epoch"]
})

loss_viz_components = graph.add_node("loss_curve", "loss_components", config={
    "title": "Loss Components", 
    "metrics": ["loss_physics", "loss_bc", "loss_ic"]
})

loss_viz_granular = graph.add_node("loss_curve", "loss_granular", config={
    "title": "Detailed Physics",
    "metrics": ["loss_phys_temperature", "loss_phys_alpha"]
})

# All receive the same run_id
graph.connect("logger_1", "run_id", "loss_main", "run_id")
graph.connect("logger_1", "run_id", "loss_components", "run_id")  
graph.connect("logger_1", "run_id", "loss_granular", "run_id")

# Frontend will render 3 separate charts, all polling the same metrics file


# ==============================================================================
# Error Handling: What if logger fails?
# ==============================================================================

class SafeLossCurveNode(Node):
    """Enhanced version with error handling"""
    
    def execute(self):
        run_id = self.inputs.get("run_id")
        cfg = self.config
        
        if not run_id:
            # Fallback: create a "waiting" widget spec
            return {
                "widget_spec": {
                    "type": "loss_curve",
                    "node_id": self.node_id,
                    "status": "waiting_for_run_id",
                    "message": "Training not started yet"
                }
            }
        
        # Normal flow
        widget_spec = {
            "type": "loss_curve",
            "node_id": self.node_id,
            "run_id": run_id,
            "title": cfg.title,
            "metrics": cfg.metrics,
            "refresh_rate": cfg.refresh_rate,
            "data_endpoint": f"/monitor/metrics/{run_id}"
        }
        
        return {"widget_spec": widget_spec}