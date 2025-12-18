import os
import json
import time
import csv
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from src.node_system.core import Node, NodeMetadata, Port, PortType, register_node



def get_new_run(save_dir: str, status_file_name: str = "status.json") -> str:
    """Generates a timestamp ID and initializes the run directory."""
    now = datetime.now()
    timestamp_id = now.strftime("%Y-%m-%d_%H-%M-%S")
    pretty_date = now.strftime("%Y-%m-%d %H:%M:%S")

    run_path = os.path.join(save_dir, timestamp_id)

    os.makedirs(run_path, exist_ok=True)
    #os.makedirs(os.path.join(run_path, 'checkpoints'), exist_ok=True)
    #os.makedirs(os.path.join(run_path, 'vtk'), exist_ok=True)

    # 2. Initialize Metadata
    initial_status = {
        "id": timestamp_id,
        "status": "initializing",
        "start_time": pretty_date,
        "epoch": 0,
        "loss": None
    }
        
    with open(os.path.join(run_path, status_file_name), 'w') as f:
        json.dump(initial_status, f, indent=4)

    return timestamp_id

class LoggerConfig(BaseModel):
    save_dir: str = Field(default="content/runs", title="Runs Directory")
    version: Optional[str] = Field(default=None, title="Run Name (auto-generate if empty)")
    save_graph: bool = Field(default=True, title="Save graph.json to run directory")

@register_node("dashboard_logger")
class DashboardLoggerNode(Node):
    @classmethod
    def get_input_ports(cls):
        return [
            Port("graph", PortType.ANY, required=False, description="NodeGraph instance to save")
        ]

    @classmethod
    def get_output_ports(cls):
        return [
            Port("logger", PortType.LOGGER),
            Port("run_id", PortType.RUN_ID, description="Unique run identifier for monitoring")
        ]
    
    @classmethod
    def get_metadata(cls):
        return NodeMetadata(
            category="Logger",
            display_name="Dashboard Logger",
            description="This crestaes the run ID",
            icon="network-wired"
        )


    @classmethod
    def get_config_schema(cls):
        return LoggerConfig

    def execute(self):
        cfg = self.config
        
        # 1. Create/Get Run Directory
        version_name = cfg.version or get_new_run(cfg.save_dir)
        
        run_path = os.path.join(cfg.save_dir, version_name)
        
        # 2. Save Graph if provided
        if cfg.save_graph and "graph" in self.inputs:
            graph = self.inputs["graph"]
            if graph:
                graph_path = os.path.join(run_path, "graph.json")
                config_path = os.path.join(run_path, "graph_config.json")
                
                graph.save_to_file(graph_path, metadata={
                    "run_id": version_name,
                    "purpose": "training"
                })
                
                config_snapshot = graph.get_config_snapshot()
                with open(config_path, 'w') as f:
                    json.dump(config_snapshot, f, indent=2)
        
        # 3. Create Logger
        logger = DashboardLogger(save_dir=cfg.save_dir, version=version_name)
        
        return {
            "logger": logger,
            "run_id": version_name
        }

class DashboardLogger(Logger):
    """"
        This is a Custome Logger that writes to a CSV file and a status JSON file for the HTML page to read from
    """
    def __init__(self, save_dir, version=None):
        super().__init__()
        self._save_dir = save_dir
        self._version = version or "0"
        self._experiment_name = "dashboard_logs"
        
        # Create Directory
        os.makedirs(self.log_dir, exist_ok=True)
        self.status_file = os.path.join(self.log_dir, 'status.json')
        self.metrics_file = os.path.join(self.log_dir, 'metrics.csv')
        
        self.fieldnames = [
            'step', 'epoch', 'timestamp',
            
            # Main Losses
            'loss_physics', 'loss_bc', 'loss_ic',
            'loss', 'loss_step', 'loss_epoch',
            
            # NEW: Granular Physics Losses
            'loss_phys_temperature', 'loss_phys_alpha',
            
            # NEW: Granular BC/IC Losses
            'loss_bc_temperature', 
            'loss_ic_temperature', 'loss_ic_alpha',
            
            # Validation Metrics
            'val_loss_physics', 'val_loss_bc', 'val_loss_ic',
            'val_loss' 
        ]
        
        self._update_status("initialized")

    @property
    def name(self):
        return self._experiment_name

    @property
    def version(self):
        return self._version

    @property
    def log_dir(self):
        return os.path.join(self._save_dir, self.version)

    @rank_zero_only
    def log_hyperparams(self, params):
        params_file = os.path.join(self.log_dir, 'hparams.json')
        with open(params_file, 'w') as f:
            json.dump(params, f, default=str, indent=4)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """
        Robustly logs metrics to CSV, handling missing keys (e.g. val during train)
        and ensuring column order is always deterministic.
        """
        # Filter for scalars only
        row = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        row['step'] = step
        row['timestamp'] = time.time()

        current_epoch = metrics.get('epoch', None)        
        current_loss = metrics.get('loss_step', metrics.get('loss_epoch', metrics.get('loss', None)))

        # Determine file mode
        file_exists = os.path.isfile(self.metrics_file)
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=self.fieldnames, 
                extrasaction='ignore', 
                restval=''
            )
            
            # Write header only once at the start
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)

        # Update status file for the frontend polling
        self._update_status("running", epoch=current_epoch, loss=current_loss)

    @rank_zero_only
    def finalize(self, status):
        self._update_status(status)

    def _update_status(self, status, epoch=None, loss=None):

        data = {}
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
            except:
                pass # If file is corrupt, we start fresh, which is GUUD

        #Update fields
        data['id'] = self.version
        data['status'] = status
        data['last_update'] = time.time()
        
        if epoch is not None:
            data['epoch'] = int(epoch)
        if loss is not None:
            data['loss'] = float(loss)

        # Atomic write
        temp = self.status_file + '.tmp'
        try:
            with open(temp, 'w') as f:
                json.dump(data, f, indent=4)
            os.replace(temp, self.status_file)
        except Exception as e:
            print(f"Status update failed: {e}")
