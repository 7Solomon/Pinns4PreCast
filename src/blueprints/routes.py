import os
from flask import Blueprint, render_template, request, redirect, url_for

from src.state_management.state import State
from src.state_management.config import Config
from src.state_management.domain import DomainVariables
from src.state_management.material import ConcreteData
from src.utils import generate_schema

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


@routes_bp.route('/setup', methods=['GET', 'POST'])
def setup():
    """Setup configuration, material, and domain - optional page to change from defaults"""
    if request.method == 'POST':
        config_file = request.form.get('config')
        material_file = request.form.get('material')
        domain_file = request.form.get('domain')
        
        if config_file and material_file and domain_file:
            State().load_state(config_file, material_file, domain_file)
            return redirect(url_for('routes.runs'))
    
    options_data = State().get_all_options_data()
    current_state = State().get_current_state_info()
    return render_template('setup.html', 
                           current_state=current_state,
                           options_data=options_data,
                           config_schema=generate_schema(Config),
                           material_schema=generate_schema(ConcreteData),
                           domain_schema=generate_schema(DomainVariables))