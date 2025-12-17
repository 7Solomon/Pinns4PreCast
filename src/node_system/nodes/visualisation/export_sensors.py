from scipy.interpolate import griddata
import numpy as np

def export_sensors_to_csv(predictions, domain, test_coords, sensor_temp_path, sensor_alpha_path):
    """
    Interpolates data at sensor locations for all time steps and saves two CSVs:
    - sensors_temperature.csv (Time_s, Time_h, <sensor>_Temp...)
    - sensors_alpha.csv       (Time_s, Time_h, <sensor>_Alpha...)
    """
    #if idx_path is None:
    #    content_dir = os.listdir(os.path.join('content'))
    #    if not content_dir:
    #        raise ValueError("No content directory found for exporting VTK files.")
    #    idx_path = os.path.join('content', content_dir[-1])
#
    #output_temp = os.path.join(idx_path, "sensors_temperature.csv")
    #output_alpha = os.path.join(idx_path, "sensors_alpha.csv")

    all_times = np.unique(test_coords[:, 3])
    all_times.sort()

    sensor_ids = list(domain.TEMP_SENS_POINTS.keys())

    header_temp = ["Time_s", "Time_h"] + [f"{sid}_Temp" for sid in sensor_ids]
    header_alpha = ["Time_s", "Time_h"] + [f"{sid}_Alpha" for sid in sensor_ids]

    rows_temp = []
    rows_alpha = []

    #print(f"Interpolating sensors for CSV export ({len(all_times)} timesteps, {len(sensor_ids)} sensors)...")

    for t in all_times:
        mask_t = np.isclose(test_coords[:, 3], t)
        coords_t = test_coords[mask_t, :3]
        pred_t = predictions[mask_t]

        temp_row = [t, t / 3600.0]
        alpha_row = [t, t / 3600.0]

        for sid in sensor_ids:
            point = domain.TEMP_SENS_POINTS[sid]

            temp_val = griddata(coords_t, pred_t[:, 0], point, method='linear')
            alpha_val = griddata(coords_t, pred_t[:, 1], point, method='linear')

            if np.isnan(temp_val):
                temp_val = griddata(coords_t, pred_t[:, 0], point, method='nearest')
            if np.isnan(alpha_val):
                alpha_val = griddata(coords_t, pred_t[:, 1], point, method='nearest')

            temp_row.append(float(temp_val))
            alpha_row.append(float(alpha_val))

        rows_temp.append(temp_row)
        rows_alpha.append(alpha_row)

    # Save CSVs
    np.savetxt(sensor_temp_path, np.array(rows_temp), delimiter=",", header=",".join(header_temp), comments="", fmt="%.6f")
    np.savetxt(sensor_alpha_path, np.array(rows_alpha), delimiter=",", header=",".join(header_alpha), comments="", fmt="%.6f")

    #print(f"Temperature sensor data exported to '{sensor_temp_path}'")
    #print(f"Alpha sensor data exported to '{sensor_alpha_path}'")