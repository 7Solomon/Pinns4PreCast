import json
import os
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

# Import your new Pydantic models
from src.state_management.config import Config
from src.state_management.material import ConcreteData
from src.state_management.domain import DomainVariables
from src.state_management.directory_management import DirectoryManager

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
        self.run_manager = None
        
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
