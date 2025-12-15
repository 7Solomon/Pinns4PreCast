from src.node_system.factories import create_config_node
from src.node_system.configs import (
    DataLoaderConfig, 
    InputConfig, 
    DatasetConfig, 
    CompositeDatasetConfig, 
    ModelConfig, 
    CompositeModelConfig, 
    TrainingConfig
)

create_config_node(
    config_model=DataLoaderConfig,
    node_type_id="dataloader_config",
    display_name="DataLoader Settings",
    description="Batch size, shuffle, workers"
)

create_config_node(
    config_model=InputConfig,
    node_type_id="input_config",
    display_name="Sensor Config",
    description="Number of sensors for BC/IC"
)

create_config_node(
    config_model=DatasetConfig,
    node_type_id="dataset_gen_config",
    display_name="Generation Params",
    description="Sample counts for PDE/BC/IC"
)

create_config_node(
    config_model=CompositeDatasetConfig,
    node_type_id="composite_dataset_gen_config",
    display_name="Composite Generation Params",
    description="Sample counts for PDE/BC/IC"
)

create_config_node(
    config_model=TrainingConfig,
    node_type_id="training_config",
    display_name="Training Hyperparams",
    description="Epochs, LR, Optimizer, Scheduler"
)

create_config_node(
    config_model=ModelConfig,
    node_type_id="model_config",
    display_name="DeepONet Architecture",
    description="Layers, activations, Fourier features"
)

create_config_node(
    config_model=CompositeModelConfig,
    node_type_id="composite_model_config",
    display_name="Composite DeepONet Architecture",
    description="Layers, activations, Fourier features"
)

