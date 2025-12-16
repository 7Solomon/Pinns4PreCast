from src.node_system.core import NodeGraph

# --- IMPORT ALL NODES TO REGISTER THEM ---
import src.node_system.nodes.deepONet_definitions.model_nodes
import src.node_system.nodes.physics.problem_nodes
import src.node_system.nodes.data.dataloader_nodes
import src.node_system.nodes.data.dataset_nodes
import src.node_system.nodes.deepONet_definitions.solver
import src.node_system.nodes.training.trainer
import src.node_system.nodes.physics.parameter_nodes

def test_full_pipeline():
    graph = NodeGraph()
    
    graph.add_node("concrete_material", "mat", {
        "name": "Test Concrete", "rho": 2400.0, "cem": 350.0, "k": 1.8
    })
    graph.add_node("spatial_domain", "dom", {
        "x": [0.0, 0.5], "y": [0.0, 0.5], "z": [0.0, 0.5], "t": [0.0, 100.0]
    })
    
    # Create a Shared Config Node (Best Practice Fix)
    # Assuming you have a generic config node, or just pass dicts to both
    
    graph.add_node("heat_pde", "pde")
    graph.add_node("deeponet_dataset", "data", {})
    graph.add_node("deeponet_dataloader", "loader", {"batch_size": 4}) # Small batch for testing
    graph.add_node("flex_deeponet", "net", {})
    graph.add_node("deeponet_solver", "solver", {})    
    graph.add_node("lightning_trainer", "trainer", {"max_epochs": 1, "accelerator": "cpu"})

    # --- 2. CONNECT (FIXED) ---
    graph.connect("mat", "material", "pde", "material")
    graph.connect("dom", "domain",   "pde", "domain")
    
    graph.connect("pde", "problem_instance", "data", "problem")
    graph.connect("mat", "material", "data", "material")
    graph.connect("dom", "domain",   "data", "domain")
    
    graph.connect("data", "dataset", "loader", "dataset")
    
    graph.connect("net",  "model_instance",   "solver", "model")
    graph.connect("pde",  "problem_instance", "solver", "problem")
    
    graph.connect("loader", "dataloader", "trainer", "dataloader")
    
    # --- FIX: Connect Solver to Trainer ---
    # The Trainer NEEDS the solver instance to run .fit()
    graph.connect("solver", "solver", "trainer", "solver")
        
    
    result = graph.execute("trainer", "trained_solver")
    
    print("\nSUCCESS!")
    print(f"Final Metrics: {result.trainer.callback_metrics}")

if __name__ == "__main__":
    test_full_pipeline()
