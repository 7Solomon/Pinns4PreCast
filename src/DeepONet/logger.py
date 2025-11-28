import os
import json
import csv
import time
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only

class DashboardLogger(Logger):
    """"
        This is a Custome Logger that writes to a CSV file and a status JSON file for the HTML page to read from
    """
    def __init__(self, save_dir, version=None):
        super().__init__()
        self._save_dir = save_dir
        self._version = version or "0"
        self._experiment_name = "dashboard_logs"
        
        # Create Directory
        os.makedirs(self.log_dir, exist_ok=True)
        self.status_file = os.path.join(self.log_dir, 'status.json')
        self.metrics_file = os.path.join(self.log_dir, 'metrics.csv')
        
        self.fieldnames = [
            'step', 'epoch', 'timestamp',
            
            # Main Losses
            'loss_physics', 'loss_bc', 'loss_ic',
            'loss', 'loss_step', 'loss_epoch',
            
            # NEW: Granular Physics Losses
            'loss_phys_temperature', 'loss_phys_alpha',
            
            # NEW: Granular BC/IC Losses
            'loss_bc_temperature', 
            'loss_ic_temperature', 'loss_ic_alpha',
            
            # Validation Metrics
            'val_loss_physics', 'val_loss_bc', 'val_loss_ic',
            'val_loss' 
        ]
        
        self._update_status("initialized")

    @property
    def name(self):
        return self._experiment_name

    @property
    def version(self):
        return self._version

    @property
    def log_dir(self):
        return os.path.join(self._save_dir, self.version)

    @rank_zero_only
    def log_hyperparams(self, params):
        params_file = os.path.join(self.log_dir, 'hparams.json')
        with open(params_file, 'w') as f:
            json.dump(params, f, default=str, indent=4)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """
        Robustly logs metrics to CSV, handling missing keys (e.g. val during train)
        and ensuring column order is always deterministic.
        """
        # Filter for scalars only
        row = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        row['step'] = step
        row['timestamp'] = time.time()
        
        # Determine file mode
        file_exists = os.path.isfile(self.metrics_file)
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=self.fieldnames, 
                extrasaction='ignore', 
                restval=''
            )
            
            # Write header only once at the start
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)

        # Update status file for the frontend polling
        self._update_status("running")

    @rank_zero_only
    def finalize(self, status):
        self._update_status(status)

    def _update_status(self, status):
        """Atomic status update"""
        data = {
            "id": self.version,
            "status": status,
            "last_update": time.time()
        }
        temp = self.status_file + '.tmp'
        try:
            with open(temp, 'w') as f:
                json.dump(data, f)
            os.replace(temp, self.status_file)
        except Exception as e:
            print(f"Status update failed: {e}")
