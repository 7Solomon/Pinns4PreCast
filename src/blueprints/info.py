from flask import Blueprint, jsonify
import os
import json
import pandas as pd

from src.state_management.state import State
from src.utils import read_files_to_map

info_bp = Blueprint('info', __name__, url_prefix='/info')


@info_bp.route('/runs', methods=['GET'])
def get_all_runs():
    """Scans the runs folder and returns a list of all runs with basic status"""
    runs_path = State().directory_manager.runs_path
    if not os.path.exists(runs_path):
        return jsonify([])
        
    runs = []
    # Iterate over directories in runs folder
    for dirname in sorted(os.listdir(runs_path), key=lambda x: int(x) if x.isdigit() else 0, reverse=True):
        dir_full = os.path.join(runs_path, dirname)
        if os.path.isdir(dir_full):
            status_file = os.path.join(dir_full, 'status.json')
            status = "unknown"
            
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        info = json.load(f)
                        status = info.get('status', 'unknown')
                except:
                    pass
            
            runs.append({
                "id": dirname,
                "status": status
            })
    return jsonify(runs)

@info_bp.route('/run/<run_id>/log', methods=['GET'])
def get_run_data(run_id):
    """Returns parsed CSV data for the chart"""
    run_path = os.path.join(State().directory_manager.runs_path, run_id)
    csv_path = os.path.join(run_path, 'metrics.csv')
    status_path = os.path.join(run_path, 'status.json')
    
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