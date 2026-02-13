import numpy as np
import torch
import pandas as pd


def calculate_pick_times(n_shots, n_receivers, nx_new, cropped_data_path):
    raw_data = np.fromfile(cropped_data_path, dtype=np.float32)

    # Dynamic reshape
    nt_val = len(raw_data) // (n_shots * n_receivers)
    data = raw_data.reshape(n_shots, n_receivers, nt_val)
    print(f"reshaped to: {data.shape}")

    # adjust dt for picking
    dt_original = 0.004
    total_time = 750 * dt_original
    picking_time = total_time / nt_val
    dx = 4.0

    # These are in METERS for the PINN
    source_depth = 2 * dx
    receiver_depth = 2 * dx
    first_source_x = 10 * dx
    d_source_x = (nx_new - 20) // n_shots * dx 
    first_receiver_x = 0 * dx
    d_receiver_x = 4 * dx


    # 2. The Picker
    training_data = []

    threshold_percent = 0.05 # 5% of max amplitude

    for s in range(n_shots):
        src_x = first_source_x + (s * d_source_x)
        
        for r in range(n_receivers):
            rec_x = first_receiver_x + (r * d_receiver_x)
            
            trace = np.abs(data[s, r, :])
            max_amp = np.max(trace)
            
            # Find first index above threshold
            # We use a 'where' to find indices, then [0][0] for the first one
            arrivals = np.where(trace > (max_amp * threshold_percent))[0]
            
            if len(arrivals) > 0:
                first_arrival_idx = arrivals[0]
                t_obs = first_arrival_idx * picking_time
                
                # Store [src_x, src_z, rec_x, rec_z, time]
                training_data.append([src_x, source_depth, rec_x, receiver_depth, t_obs])

    # 3. Convert to Dataframe and Save
    df = pd.DataFrame(training_data, columns=['sx', 'sz', 'rx', 'rz', 't_obs'])
    df.to_csv("data/marmousi_training_picks.csv", index=False)
    print("Saved 5,000 picked traveltimes to marmousi_training_picks.csv")

if __name__ == "__main__":
    n_shots, n_receivers, nx_new, cropped_data_path = 20, 250, 1000, "data/marmousi_data_cropped.bin"
    calculate_pick_times(n_shots, n_receivers, nx_new, cropped_data_path)