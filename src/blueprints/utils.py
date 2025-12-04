from flask import Blueprint, jsonify
import os
import signal
import threading

from src.blueprints.api import api_bp


utils_bp = Blueprint('utils', __name__, url_prefix='/utils')

@api_bp.route('/shutdown', methods=['POST'])
def shutdown_server():
    """
    Kills the Flask process. Useful if the port is stuck.
    """
    def kill_me():
        import time
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGINT)

    # Launch the killer in a separate thread so the request can return 200 OK first
    threading.Thread(target=kill_me).start()
    
    return jsonify({"message": "Server is shutting down..."})