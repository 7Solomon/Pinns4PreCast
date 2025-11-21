import os
from flask import Blueprint, render_template, request, redirect, url_for

from src.state_management.state import State

routes_bp = Blueprint('routes', __name__)


@routes_bp.route('/')
def index():
    """Landing page - always redirect to runs (state is auto-loaded)"""
    return redirect(url_for('routes.index'))


@routes_bp.route('/runs')
def runs():
    """Display all training runs"""
    # Get all runs from the runs directory
    runs_path = State().directory_manager.runs_path
    runs = []
    if os.path.exists(runs_path):
        runs = sorted(os.listdir(runs_path), reverse=True)
    
    return render_template('runs.html', runs=runs, active_page='runs')


@routes_bp.route('/run/<run_id>')
def view_run(run_id):
    """View details of a specific run"""
    # TODO: Implement run details page
    return f"<h1>Run Details: {run_id}</h1><p>Coming soon...</p>"


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
                         options_data=options_data, 
                         current_state=current_state,
                         active_page='setup')
