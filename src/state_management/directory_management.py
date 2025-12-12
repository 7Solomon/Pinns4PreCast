from datetime import datetime
import json
import os
import time
from typing import Dict, Optional
from pydantic import BaseModel, Field



class DirectoryManager(BaseModel):
    content_path: str = os.path.join('content')
    runs_path: str = os.path.join('content', 'runs')
    
    materials_path: str = os.path.join('content', 'states', 'materials')
    domains_path: str = os.path.join('content', 'states', 'domains')
    configs_path: str = os.path.join('content', 'states', 'configs')

    states_file: str = '.current_state.json'
    run_idx_path: Optional[str] = None
    
    # Use Field for mutable defaults
    state_paths: Dict[str, str] = Field(default_factory=dict)

    @classmethod
    def create(cls):
        """Factory method to create and ensure directories exist"""
        instance = cls()
        os.makedirs(instance.content_path, exist_ok=True)
        os.makedirs(instance.runs_path, exist_ok=True)
        os.makedirs(instance.materials_path, exist_ok=True)
        os.makedirs(instance.domains_path, exist_ok=True)
        os.makedirs(instance.configs_path, exist_ok=True)
        return instance

    def list_state_directories(self):
        return {
            'material': self.materials_path, 
            'domain': self.domains_path, 
            'config': self.configs_path
        }
    
    def set_directories(self, directories: dict):
        self.state_paths = {**self.state_paths, **directories}

    # --- Properties for Dynamic Paths ---
    @property
    def current_run_path(self):
        return os.path.join(self.runs_path, self.run_idx_path) if self.run_idx_path else None
    
    @property
    def checkpoint_path(self):
        return os.path.join(self.runs_path, self.run_idx_path, 'checkpoints')
    
    @property
    def log_path(self):
        return os.path.join(self.runs_path, self.run_idx_path, 'metrics.csv')

    def get_log_path(self, idx):
        return os.path.join(self.runs_path, idx, 'metrics.csv')
    
    @property
    def status_path(self):
        return os.path.join(self.runs_path, self.run_idx_path, 'status.json')

    def get_status_path(self, idx):
        return os.path.join(self.runs_path, idx, 'status.json')

    @property
    def vtk_path(self):
        path = os.path.join(self.runs_path, self.run_idx_path, 'vtk')
        os.makedirs(path, exist_ok=True)
        return path

    def get_vtk_path(self, idx):
        path = os.path.join(self.runs_path, idx, 'vtk')
        os.makedirs(path, exist_ok=True)
        return path
    
    @property
    def sensor_alpha_path(self):
        path = os.path.join(self.runs_path, self.run_idx_path, 'sensor_alpha')
        os.makedirs(path, exist_ok=True)
        return path

    def get_sensor_alpha_path(self, idx):
        path = os.path.join(self.runs_path, idx, 'sensor_alpha')
        os.makedirs(path, exist_ok=True)
        return path
    
    @property
    def sensor_temp_path(self):
        path = os.path.join(self.runs_path, self.run_idx_path, 'sensor_temperature')
        os.makedirs(path, exist_ok=True)
        return path

    def get_sensor_temp_path(self, idx):
        path = os.path.join(self.runs_path, idx, 'sensor_temperature')
        os.makedirs(path, exist_ok=True)
        return path


    ####
    def create_new_run(self) -> str:
        """
        Generates a unique timestamped run ID, creates directories,
        and initializes the status file.
        """
        now = datetime.now()
        timestamp_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        pretty_date = now.strftime("%Y-%m-%d %H:%M:%S")

        # Safety retry logic
        self.run_idx_path = timestamp_id
        if os.path.exists(self.current_run_path):
            time.sleep(1.1)
            now = datetime.now()
            self.run_idx_path = now.strftime("%Y-%m-%d_%H-%M-%S")
            pretty_date = now.strftime("%Y-%m-%d %H:%M:%S")

        # 2. Create directories
        os.makedirs(self.current_run_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.vtk_path, exist_ok=True)
        os.makedirs(self.sensor_alpha_path, exist_ok=True)
        os.makedirs(self.sensor_temp_path, exist_ok=True)

        # 3. Initialize Metadata (status.json)
        initial_status = {
            "id": self.run_idx_path,
            "status": "initializing",
            "start_time": pretty_date,
            "epoch": 0,
            "loss": None
        }
        
        with open(self.status_path, 'w') as f:
            json.dump(initial_status, f, indent=4)

        return self.run_idx_path
