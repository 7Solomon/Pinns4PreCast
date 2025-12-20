import os
import json
import shutil
import glob
import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/monitor", tags=["monitoring"])

RUNS_DIR = "content/runs"

@router.get("/status/{run_id}")
def get_training_status(run_id: str):
    status_path = f"content/runs/{run_id}/status.json"
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

@router.get("/vis/sensor/{run_id}")
def get_sensor_data(run_id: str):
    """
    Returns a dictionary of all available CSV files for a run.
    Structure:
    {
       "temperature": { "epoch_0.csv": "raw_csv_string", ... },
       "alpha": { "epoch_0.csv": "raw_csv_string", ... }
    }
    """
    run_path = os.path.join(RUNS_DIR, run_id)
    if not os.path.exists(run_path):
        raise HTTPException(status_code=404, detail="Run not found")

    result = {"temperature": {}, "alpha": {}}
    
    # 1. Load Temperature CSVs
    temp_path = os.path.join(run_path, "sensors_temp")
    if os.path.exists(temp_path):
        for f in glob.glob(os.path.join(temp_path, "*.csv")):
            fname = os.path.basename(f)
            with open(f, "r") as file:
                result["temperature"][fname] = file.read()

    # 2. Load Alpha CSVs
    alpha_path = os.path.join(run_path, "sensors_alpha")
    if os.path.exists(alpha_path):
        for f in glob.glob(os.path.join(alpha_path, "*.csv")):
            fname = os.path.basename(f)
            with open(f, "r") as file:
                result["alpha"][fname] = file.read()

    return result


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
