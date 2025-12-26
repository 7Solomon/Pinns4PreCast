import threading
from typing import Optional, Dict, Any

class AppState:
    def __init__(self):
        self.current_run_id: Optional[str] = None        
        self._active_sessions: Dict[str, Any] = {}
        
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Read-only property for frontend status"""
        return self.current_run_id is not None

    def register_session(self, run_id: str, trainer: Any):
        """Called when training starts"""
        with self._lock:
            if self.current_run_id is not None:
                raise RuntimeError(f"Run {self.current_run_id} is already active!")
            
            print(f"🔴 Registering active session: {run_id}")
            self.current_run_id = run_id
            self._active_sessions[run_id] = trainer

    def stop_current_session(self):
        """Called when Stop button is clicked"""
        with self._lock:
            run_id = self.current_run_id
            
            if not run_id or run_id not in self._active_sessions:
                print("⚠️ No active session found to stop.")
                return False

            print(f"🛑 Stopping session {run_id}...")
            trainer = self._active_sessions[run_id]

            if hasattr(trainer, "should_stop"):
                trainer.should_stop = True
            else:
                print("⚠️ Trainer object does not have 'should_stop' attribute")

            return True
        
    def update_session_trainer(self, run_id: str, trainer: Any):
        """Updates the trainer object for an existing reserved session"""
        with self._lock:
            if self.current_run_id == run_id:
                self._active_sessions[run_id] = trainer
                print(f"✅ Trainer object attached to session: {run_id}")

    def clear_session(self, run_id: str):
        """Called when training loop officially finishes (cleanup)"""
        with self._lock:
            if self.current_run_id == run_id:
                print(f"🟢 Clearing session: {run_id}")
                self.current_run_id = None
                self._active_sessions.pop(run_id, None)

app_state = AppState()
