from src.node_system.core import NodeGraph

# --- IMPORT ALL NODES TO REGISTER THEM ---
import src.node_system.nodes.model_nodes
import src.node_system.nodes.problem_nodes
import src.node_system.nodes.data_nodes
import src.node_system.nodes.training_nodes
import src.node_system.nodes.parameter_nodes

def test_full_pipeline():
    graph = NodeGraph()
    
    print("\n--- 1. Creating Nodes ---")
    
    # --- A. PARAMETERS (Data) ---
    # Material
    mat_node = graph.add_node("concrete_material", "mat", {
        "name": "Test Concrete",
        "rho": 2400.0,
        "cem": 350.0,
        "k": 1.8
    })
    
    # Domain
    dom_node = graph.add_node("spatial_domain", "dom", {
        "x": [0.0, 0.5],
        "y": [0.0, 0.5],
        "z": [0.0, 0.5],
        "t": [0.0, 3600.0 * 24] 
    })

    # Optional: Training Config (To test Optimizer switching)
    # train_cfg_node = graph.add_node("training_config", "t_cfg", {
    #     "max_epochs": 2,
    #     "optimizer_type": "AdamW", 
    #     "optimizer_learning_rate": 5e-4
    # })
    
    # --- B. LOGIC (Process) ---
    
    # Physics (Assembler)
    pde_node = graph.add_node("heat_pde", "pde")
    
    # Dataset (Logic)
    # Pass {} to use internal defaults
    data_node = graph.add_node("deeponet_dataset", "data", {})
    
    # --- ADD THIS: Data Loader (Batches Samples) -> Outputs 'dataloader' ---
    loader_node = graph.add_node("deeponet_dataloader", "loader", {})
    # -----------------------------------------------------------------------
    
    # Model (Logic)
    model_node = graph.add_node("flex_deeponet", "net", {})
    
    # Solver (Logic)
    # Pass {} to use internal defaults (Adam, 1e-4)
    solver_node = graph.add_node("deeponet_solver", "trainer", {})
    
    print("--- 2. Connecting Graph ---")
    
    # 1. Physics Assembly
    graph.connect("mat", "material", "pde", "material")
    graph.connect("dom", "domain",   "pde", "domain")
    
    graph.connect("pde", "problem_instance", "data", "problem")
    graph.connect("mat", "material", "data", "material")
    graph.connect("dom", "domain",   "data", "domain")
    graph.connect("data", "dataset", "loader", "dataset")
    graph.connect("net",  "model_instance",   "trainer", "model")
    graph.connect("pde",  "problem_instance", "trainer", "problem")
    
    # --- CHANGE THIS: Connect Loader -> Trainer ---
    # Was: graph.connect("data", "dataloader", ...) -> WRONG (data outputs dataset)
    graph.connect("loader", "dataloader", "trainer", "dataloader") 
    
    # Optional: Connect Training Config
    # graph.connect("t_cfg", "config", "trainer", "training_config")
    
    print("--- 3. Executing Pipeline ---")
    
    result = graph.execute("trainer", "trained_solver")
    
    print("\nSUCCESS!")
    print(f"Final Metrics: {result.trainer.callback_metrics}")

if __name__ == "__main__":
    test_full_pipeline()
