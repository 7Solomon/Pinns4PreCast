ACTIVE_TRAINING_SESSIONS = {}
EXECUTION_LOCK = False

def register_session(run_id, trainer):
    print(f"🔴 Registering active session: {run_id}")
    ACTIVE_TRAINING_SESSIONS[run_id] = trainer

def unregister_session(run_id):
    if run_id in ACTIVE_TRAINING_SESSIONS:
        #print(f"🟢 Unregistering session: {run_id}")
        del ACTIVE_TRAINING_SESSIONS[run_id]

def stop_session(run_id):
    print(ACTIVE_TRAINING_SESSIONS)
    if run_id in ACTIVE_TRAINING_SESSIONS:
        print(f"Stopping session {run_id}...")
        trainer = ACTIVE_TRAINING_SESSIONS[run_id]
        trainer.should_stop = True 
        return True
    return False

def kill_session(run_id):
    if run_id in ACTIVE_TRAINING_SESSIONS:
        trainer = ACTIVE_TRAINING_SESSIONS[run_id]


