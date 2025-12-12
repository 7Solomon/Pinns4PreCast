import os
from flask import Blueprint, render_template, request, redirect, url_for

from src.state_management.state import State
from src.state_management.config import Config
from src.state_management.domain import DomainVariables
from src.state_management.material import ConcreteData
from src.utils import get_pydantic_schema

routes_bp = Blueprint('routes', __name__)


@routes_bp.route('/')
def index():
    """Landing page - always redirect to runs (state is auto-loaded)"""
    print("LANDEDING PAGE")
    return redirect(url_for('routes.runs'))


@routes_bp.route('/runs')
def runs():
    """Display all training runs"""
    # Get all runs from the runs directory
    runs_path = State().directory_manager.runs_path
    runs = []
    if os.path.exists(runs_path):
        runs = sorted(os.listdir(runs_path), reverse=True)
    
    return render_template('runs.html', runs=runs, active_page='runs')

@routes_bp.route('/setup', methods=['GET'])
def setup():
    schemas = {
        'config': Config.model_json_schema(),
        'material': ConcreteData.model_json_schema(),
        'domain': DomainVariables.model_json_schema()
    }

    current_state = {
        'config': State().config.model_dump(),
        'material': State().material.model_dump(),
        'domain': State().domain.model_dump()
    }
    state_info = State().get_current_state_info()


    return render_template(
        'setup.html', 
        schemas=schemas, 
        current_state=current_state, 
        active_filenames=state_info,
        options_data=State().get_all_options_data()
    )