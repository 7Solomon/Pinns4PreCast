import os
import json
import re
import shutil
import glob
import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/monitor", tags=["monitoring"])

RUNS_DIR = "content/runs"

@router.get("/status/{run_id}")
def get_training_status(run_id: str):
    status_path = f"{RUNS_DIR}/{run_id}/status.json"
    if not os.path.exists(status_path):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    
    try:
        with open(status_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/{run_id}")
def get_training_metrics(run_id: str, limit: int = 100):
    metrics_path = f"content/runs/{run_id}/metrics.csv"
    if not os.path.exists(metrics_path):
        return {"metrics": [], "total_epochs": 0, "latest_epoch": 0, "message": "No metrics available"}
    
    try:
        df = pd.read_csv(metrics_path)
        if len(df) == 0:
            return {"metrics": [], "total_epochs": 0, "latest_epoch": 0}
        recent = df.tail(limit)
        metrics = recent.fillna('').to_dict('records')
        latest_epoch = int(df['epoch'].max()) if 'epoch' in df else 0
        
        return {
            "metrics": metrics,
            "total_records": len(df),
            "latest_epoch": latest_epoch,
            "columns": list(df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {str(e)}")

def extract_epoch(filename):
    """Helper to get 10 from 'epoch_10.csv' for sorting"""
    match = re.search(r'epoch_(\d+)', filename)
    return int(match.group(1)) if match else -1

@router.get("/monitor/sensor/{run_id}")
def get_sensor_history(run_id: str):
    """
    Parses all CSVs on the server and returns a single merged time-series JSON.
    Returns: { "data": [ { "step": 0, "temperature": 25.5, "alpha": 0.1 }, ... ] }
    """
    run_path = os.path.join(RUNS_DIR, run_id) # Ensure this matches your RUNS_DIR
    print(run_path)
    if not os.path.exists(run_path):
        raise HTTPException(status_code=404, detail="Run not found")

    # Master dictionary: { epoch_number: { "step": 10, "temp": ... } }
    history = {}

    # --- 1. Process Temperature Files ---
    temp_path = os.path.join(run_path, "sensors_temp")
    if os.path.exists(temp_path):
        files = glob.glob(os.path.join(temp_path, "*.csv"))
        for f in files:
            epoch = extract_epoch(os.path.basename(f))
            try:
                df = pd.read_csv(f)                
                val = df.iloc[:, 0].mean() if not df.empty else 0
                
                if epoch not in history: history[epoch] = {"step": epoch}
                history[epoch]["temperature"] = val
            except Exception as e:
                print(f"Error reading {f}: {e}")

    # --- 2. Process Alpha Files ---
    alpha_path = os.path.join(run_path, "sensors_alpha")
    if os.path.exists(alpha_path):
        files = glob.glob(os.path.join(alpha_path, "*.csv"))
        for f in files:
            epoch = extract_epoch(os.path.basename(f))
            try:
                df = pd.read_csv(f)
                val = df.iloc[:, 0].mean() if not df.empty else 0
                
                if epoch not in history: history[epoch] = {"step": epoch}
                history[epoch]["alpha"] = val
            except Exception as e:
                print(f"Error reading {f}: {e}")

    # --- 3. Convert to List and Sort ---
    data_list = list(history.values())
    data_list.sort(key=lambda x: x["step"])

    return {"data": data_list}


@router.get("/runs")
def list_active_runs():
    runs_dir = "content/runs"
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir, exist_ok=True)
        return {"runs": [], "total": 0}
    
    runs = []
    for run_id in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, run_id)
        if not os.path.isdir(run_path):
            continue
            
        status_path = os.path.join(run_path, "status.json")
        if os.path.exists(status_path):
            try:
                with open(status_path, 'r') as f:
                    status = json.load(f)
                    status['run_id'] = run_id
                    runs.append(status)
            except Exception as e:
                runs.append({"run_id": run_id, "status": "unknown", "error": str(e)})
    
    runs.sort(key=lambda x: x.get('start_time', ''), reverse=True)
    return {"runs": runs, "total": len(runs)}

@router.delete("/runs/{run_id}")
def delete_run(run_id: str):
    run_path = f"content/runs/{run_id}"
    if not os.path.exists(run_path):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    
    try:
        shutil.rmtree(run_path)
        return {"message": f"Run '{run_id}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete run: {str(e)}")
