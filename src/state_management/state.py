import json
import os
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

# Import your new Pydantic models
from src.state_management.config import Config
from src.state_management.material import ConcreteData
from src.state_management.domain import DomainVariables

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


class State:
    """
    Singleton class to manage global state including CONFIGS, MATERIAL properties, and DOMAIN variables.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(State, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Initialize Directory Manager
        self.directory_manager = DirectoryManager.create()
        
        # State Containers (Pydantic Models)
        self.config: Optional[Config] = None
        self.material: Optional[ConcreteData] = None
        self.domain: Optional[DomainVariables] = None
        
        # RAM Objects (Runtime)
        self.model = None
        self.problem = None
        self.solver = None
        self.dataset = None
        self.dataloader = None
        self.trainer = None 

        self._initialized = True
        
        self._auto_load_state()
        self._cleanup_stale_runs()

    def _cleanup_stale_runs(self):
        """Marks 'running' runs as 'aborted' if server restarts."""
        runs_path = self.directory_manager.runs_path
        if not os.path.exists(runs_path):
            return

        for dirname in os.listdir(runs_path):
            status_path = os.path.join(runs_path, dirname, 'status.json')
            if os.path.exists(status_path):
                try:
                    with open(status_path, 'r') as f:
                        data = json.load(f)
                    if data.get('status') == 'running':
                        data['status'] = 'aborted'
                        data['message'] = 'Server restarted while training'
                        with open(status_path, 'w') as f:
                            json.dump(data, f, indent=4)
                except Exception as e:
                    print(f"Error cleaning up run {dirname}: {e}")

    def _auto_load_state(self):
        """Automatically load state from persisted file or defaults."""
        if os.path.exists(self.directory_manager.states_file):
            try:
                with open(self.directory_manager.states_file, 'r') as f:
                    state_info = json.load(f)
                    config_file = state_info.get('config')
                    material_file = state_info.get('material')
                    domain_file = state_info.get('domain')
                    print()
                    
                    if config_file and material_file and domain_file:
                        self.load_state(config_file, material_file, domain_file)
                        return
            except Exception as e:
                print(f"Could not load persisted state: {e}")
        self._load_defaults()
    
    def _load_defaults(self):
        try:
            self.load_state('default.json', 'default.json', 'default.json')
        except Exception as e:
            print(f"Could not load default state: {e}")
    
    def load_state(self, config_file: str, material_file: str, domain_file: str):
        """Loads the state from the specified files."""
        config_path = os.path.join(self.directory_manager.configs_path, config_file)
        material_path = os.path.join(self.directory_manager.materials_path, material_file)
        domain_path = os.path.join(self.directory_manager.domains_path, domain_file)
        
        self.directory_manager.set_directories({
            'config': config_path,
            'material': material_path,
            'domain': domain_path
        })

        # Load using the Pydantic .load() methods we defined
        self.config = Config.load(config_path)
        self.material = ConcreteData.load(material_path)
        self.domain = DomainVariables.load(domain_path)
        
        self._persist_state(config_file, material_file, domain_file)
    
    def _persist_state(self, config_file, material_file, domain_file):
        state_info = {
            'config': config_file,
            'material': material_file,
            'domain': domain_file
        }
        try:
            with open(self.directory_manager.states_file, 'w') as f:
                json.dump(state_info, f, indent=2)
        except Exception as e:
            print(f"Could not persist state: {e}")

    def get_all_options_data(self):
        """Returns a dictionary with filenames and their content for each category."""
        data = {}
        dirs = self.directory_manager.list_state_directories()
        
        class_mapping = {
            'config': Config,
            'material': ConcreteData,
            'domain': DomainVariables
        }

        for key, path in dirs.items():
            data[key] = {}
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            # Generate default if empty
            if len(os.listdir(path)) == 0:
                default_path = os.path.join(path, 'default.json')
                if key in class_mapping:
                    # Create default instance and save
                    instance = class_mapping[key]()
                    with open(default_path, 'w') as f:
                        f.write(instance.model_dump_json(indent=4))

            for f in os.listdir(path):
                if f.endswith('.json'):
                    full_path = os.path.join(path, f)
                    try:
                        with open(full_path, 'r') as file:
                            data[key][f] = json.load(file)
                    except Exception as e:
                        data[key][f] = {"error": str(e)}
        return data

    def get_current_state_info(self):
        if os.path.exists(self.directory_manager.states_file):
            try:
                with open(self.directory_manager.states_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'config': 'default.json',
            'material': 'default.json',
            'domain': 'default.json'
        }

    def update_attributes(self, map: dict):
        for attr, value in map.items():
            setattr(self, attr, value)
