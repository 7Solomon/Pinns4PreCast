from dataclasses import field, fields
from typing import Any, Dict, List, get_origin
from flask import Blueprint, jsonify, request
import os
import json
import pandas as pd

from src.state_management.state import State
from src.utils import read_files_to_map
from src.state_management.config import Config
from src.state_management.domain import DomainVariables
from src.state_management.material import ConcreteData


info_bp = Blueprint('info', __name__, url_prefix='/info')


@info_bp.route('/runs/data', methods=['GET'])
def get_all_runs():
    """Scans the runs folder and returns a list of all runs with basic status"""
    runs_path = State().directory_manager.runs_path
    if not os.path.exists(runs_path):
        return jsonify([])
        
    runs = []
    all_entries = os.listdir(runs_path)
    
    sorted_entries = sorted(all_entries, reverse=True)

    for dirname in sorted_entries:
        dir_full = os.path.join(runs_path, dirname)
        
        # Filter for directories only
        if os.path.isdir(dir_full):
            status_file = os.path.join(dir_full, 'status.json')
            
            # Default values
            run_info = {
                "id": dirname,
                "status": "unknown",
                "start_time": "Unknown",
                "epoch": 0,
                "loss": None
            }
            
            # Try to read metadata from status.json
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        info = json.load(f)
                        run_info.update(info) # Merge file data into defaults
                except:
                    pass
            
            runs.append(run_info)
            
    return jsonify(runs)


@info_bp.route('/run/<run_id>/log', methods=['GET'])
def get_run_data(run_id):
    """Returns parsed CSV data for the chart"""
    csv_path = State().directory_manager.get_log_path(run_id)
    status_path = State().directory_manager.get_status_path(run_id)
    
    data = {"status": "unknown", "history": []}
    
    # Get Status
    if os.path.exists(status_path):
        try:
            with open(status_path, 'r') as f:
                data.update(json.load(f))
        except: pass
            
    # Get CSV Data
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df = df.replace({float('nan'): None})
            
            data["history"] = df.to_dict(orient='records')
        except Exception as e:
            print(f"Error reading CSV: {e}")
            
    return jsonify(data)


@info_bp.route('/run/<run_id>/vis/sensor', methods=['GET'])
def get_run_sensor_vis(run_id):
    temp_path = State().directory_manager.get_sensor_temp_path(run_id)
    alpha_path = State().directory_manager.get_sensor_alpha_path(run_id)

    return jsonify({
        "temperature": read_files_to_map(temp_path),
        "alpha": read_files_to_map(alpha_path)
    })



@info_bp.route('/save_config', methods=['POST'])
def save_config():
    """
    Saves a JSON configuration file to disk.
    Expects JSON: { "type": "config|material|domain", "filename": "name.json", "content": {...} }
    """
    try:
        data = request.get_json()
        file_type = data.get('type')
        filename = data.get('filename')
        content = data.get('content')

        if not all([file_type, filename, content]):
            return jsonify({"error": "Missing required fields"}), 400

        # Ensure filename ends with .json
        if not filename.endswith('.json'):
            filename += '.json'

        # Determine path based on type
        dm = State().directory_manager
        if file_type == 'config':
            base_path = dm.configs_path
        elif file_type == 'material':
            base_path = dm.materials_path
        elif file_type == 'domain':
            base_path = dm.domains_path
        else:
            return jsonify({"error": "Invalid type"}), 400

        # Validate JSON content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON syntax"}), 400

        # Save to file
        full_path = os.path.join(base_path, filename)
        with open(full_path, 'w') as f:
            json.dump(content, f, indent=4)

        return jsonify({"message": f"Successfully saved {filename}", "filename": filename})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@info_bp.route('/defaults/<config_type>', methods=['GET'])
def get_defaults(config_type):
    """Returns default values from Python dataclasses"""
    from dataclasses import asdict
    
    class_mapping = {
        'config': Config,
        'material': ConcreteData,
        'domain': DomainVariables
    }
    
    if config_type not in class_mapping:
        return jsonify({"error": "Invalid type"}), 400
    
    default_instance = class_mapping[config_type]()
    return jsonify(asdict(default_instance))

@info_bp.route('/load_state', methods=['POST'])
def load_state():
    """Updates current_state.json to point to the new active files."""
    try:
        data = request.get_json()
        
        states_file = State().directory_manager.states_file

        # 1. Read existing state
        current_state = {}
        if os.path.exists(states_file):
            with open(states_file, 'r') as f:
                current_state = json.load(f)

        current_state.update(data)

        with open(states_file, 'w') as f:
            json.dump(current_state, f, indent=4)


        return jsonify({"status": "success", "active_state": current_state})

    except Exception as e:
        print(f"Error loading state: {e}")
        return jsonify({"error": str(e)}), 500
