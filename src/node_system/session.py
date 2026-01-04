from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict


ACTIVE_TRAINING_SESSIONS = {}
EXECUTION_LOCK = False

RUN_STATUS: Dict[str, Dict[str, Any]] = {}
STOP_REQUESTED_RUNS = set() # set to track stop requests


@contextmanager
def execution_lock_guard():
    """
    Guarantees that EXECUTION_LOCK is set to True on enter
    and False on exit (even if errors occur).
    """
    global EXECUTION_LOCK
    if EXECUTION_LOCK:
        raise RuntimeError("Global execution lock is already held!")
    
    try:
        print("🔒 Acquiring Global Lock")
        EXECUTION_LOCK = True
        yield
    finally:
        # This block ALWAYS runs, even if code inside crashes
        print("🔓 Releasing Global Lock")
        EXECUTION_LOCK = False

def update_node_status(run_id: str, node_id: str, status: str, error: str = None):
    if run_id not in RUN_STATUS:
        RUN_STATUS[run_id] = {}
    
    RUN_STATUS[run_id][node_id] = {
        "status": status,
        "error": error,
        "timestamp": datetime.now().isoformat()
    }

def register_session(run_id, trainer):
    print(f"🔴 Registering active session: {run_id}")
    ACTIVE_TRAINING_SESSIONS[run_id] = trainer

def unregister_session(run_id):
    if run_id in ACTIVE_TRAINING_SESSIONS:
        #print(f"🟢 Unregistering session: {run_id}")
        del ACTIVE_TRAINING_SESSIONS[run_id]
def request_stop(run_id: str):
    """Mark a run as requested to stop."""
    STOP_REQUESTED_RUNS.add(run_id)

def should_stop(run_id: str) -> bool:
    """Check if run should stop."""
    return run_id in STOP_REQUESTED_RUNS

def clear_stop_request(run_id: str):
    """Cleanup stop flag."""
    if run_id in STOP_REQUESTED_RUNS:
        STOP_REQUESTED_RUNS.remove(run_id)
    unregister_session(run_id) 