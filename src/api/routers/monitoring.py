import os
import json
import re
import shutil
import glob
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/monitor", tags=["monitoring"])

RUNS_DIR = "content/runs"

#@router.get("/status/{run_id}")
#def get_training_status(run_id: str):
#    status_path = f"{RUNS_DIR}/{run_id}/status.json"
#    if not os.path.exists(status_path):
#        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
#    
#    try:
#        with open(status_path, 'r') as f:
#            return json.load(f)
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{run_id}")
async def get_run_history(run_id: str):
    """Fetch full metrics history via HTTP for completed runs"""
    metrics_path = f"content/runs/{run_id}/metrics.csv"
    
    if not os.path.exists(metrics_path):
        return [] # Return empty list if no metrics yet

    try:
        df = pd.read_csv(metrics_path)
        metrics = df.replace({np.nan: None}).to_dict('records')
        return metrics
    except Exception as e:
        print(f"Error reading metrics: {e}")
        raise HTTPException(status_code=500, detail="Could not read metrics file")


@router.get("/runs")
def list_active_runs():
    runs_dir = "content/runs"
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir, exist_ok=True)
        return {"runs": [], "total": 0}

    runs = {}
    for run_id in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, run_id)
        if not os.path.isdir(run_path):
            continue

        status_path = os.path.join(run_path, "status.json")
        status = {
            "run_id": run_id,
            "status": "unknown",
            "epoch": None,
            "loss": None,
        }

        # 1) Load status.json if present
        if os.path.exists(status_path):
            try:
                with open(status_path, "r") as f:
                    file_status = json.load(f)
                status.update(file_status)
                status["run_id"] = run_id
            except Exception as e:
                status["error"] = str(e)

        # 2) Check for checkpoints
        ckpt_dir = os.path.join(run_path, "checkpoints")
        checkpoints = []
        if os.path.isdir(ckpt_dir):
            for fname in os.listdir(ckpt_dir):
                if fname.endswith(".ckpt"):
                    checkpoints.append({
                        "name": fname,
                        "path": os.path.join(ckpt_dir, fname),
                    })
        status["has_checkpoints"] = len(checkpoints) > 0
        status["checkpoints"] = checkpoints  # keep small; you can also only expose best

        # 3) Check for sensor data
        sensors_dir = os.path.join(run_path, "sensors")
        sensors = []
        if os.path.isdir(sensors_dir):
            for fname in os.listdir(sensors_dir):
                if fname.endswith(".csv"):
                    sensors.append({
                        "name": fname,
                        "path": os.path.join(sensors_dir, fname),
                    })
        status["has_sensors"] = len(sensors) > 0
        status["sensors"] = sensors

        # 4) Check for VTK / visualization
        vtk_dir = os.path.join(run_path, "vtk")
        has_vtk = os.path.isdir(vtk_dir) and any(
            fname.endswith(".vtu") or fname.endswith(".vtk")
            for fname in os.listdir(vtk_dir)
        )
        status["has_vtk"] = has_vtk

        runs[run_id] = status

    # sorted by ID or start tim if present
    runs_list = list(runs.values())
    sorted_runs = sorted(runs_list, key=lambda x: x["run_id"], reverse=True)
    
    return {"runs": sorted_runs, "total": len(sorted_runs)}
