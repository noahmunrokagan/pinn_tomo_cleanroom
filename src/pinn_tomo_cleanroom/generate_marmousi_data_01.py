"""
Marmousi forward modelling using deepwave
This script was based on the example script provided by deepwave found here: https://github.com/ar4/deepwave/blob/master/docs/example_forward_model.py
"""

import matplotlib.pyplot as plt
import torch

import deepwave
from deepwave import scalar

def load_marmousi_model(crop_x, crop_y, path):
    device = torch.device("cpu")
    nx = 2301
    ny = 751
    dx = 4.0

    # Load the velocity model
    v_full = (
        torch.from_file(path, size=ny * nx)
        .reshape(ny, nx)
    )

    # Downsample the velocity model 
    # Cut out rectangle from model and move to device
    v = v_full[0:crop_y, 0:crop_x].to(device)

    ny_new, nx_new = v.shape # adjust nz and nx to new values

    # Adjust geometry
    ## Sources
    n_shots = 20
    first_source = 10 # start 10 pixels in
    source_depth = 2 # 2 pixels deep
    d_source = (nx_new - 20) // n_shots # calculate spacing to fit them evenly in the cropped model
    n_sources_per_shot = 1

    ## Receivers
    d_receiver = 4 # one every 16 meters
    first_receiver = 0
    # Calculate num receivers can fit in new width

    n_receivers_per_shot = (nx_new - first_receiver) // d_receiver
    receiver_depth = 2  # 2 * 4m = 8m

    # debugging step
    print(f"Model size: {v.shape}")
    print(f"Number of shots: {n_shots}")
    print(f"Receivers per shot: {n_receivers_per_shot}")

    freq = 25
    nt = 750
    dt = 0.004
    peak_time = 1.5 / freq

    # source_locations
    source_locations = torch.zeros(
        n_shots,
        n_sources_per_shot,
        2,
        dtype=torch.long,
        device=device,
    )
    source_locations[..., 0] = source_depth
    source_locations[...,0, 1] = torch.arange(n_shots) * d_source + first_source

    # receiver_locations
    receiver_locations = torch.zeros(
        n_shots,
        n_receivers_per_shot,
        2,
        dtype=torch.long,
        device=device,
    )
    receiver_locations[..., 0] = receiver_depth
    receiver_locations[...,:,1] = (
        torch.arange(n_receivers_per_shot) * d_receiver + first_receiver
    ).repeat(n_shots, 1)

    # source_amplitudes
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
        .repeat(n_shots, n_sources_per_shot, 1)
        .to(device)
    )

    # Propagate
    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=8,
        pml_freq=freq,
    )

    # Plot
    receiver_amplitudes = out[-1]
    vmin, vmax = torch.quantile(
        receiver_amplitudes[0],
        torch.tensor([0.05, 0.95]).to(device),
    )   
    return receiver_amplitudes, vmin, vmax

receiver_amplitudes, vmin, vmax = load_marmousi_model(1000, 500, "data/vp.bin")
_, ax = plt.subplots(1, 2, figsize=(10.5, 7), sharey=True)
ax[0].imshow(
    receiver_amplitudes[19].cpu().T,
    aspect="auto",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
)
ax[1].imshow(
    receiver_amplitudes[:, 192].cpu().T,
    aspect="auto",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
)
ax[0].set_xlabel("Channel")
ax[0].set_ylabel("Time Sample")
ax[1].set_xlabel("Shot")
plt.tight_layout()
plt.savefig("example_forward_model.jpg")

receiver_amplitudes.cpu().numpy().tofile("data/marmousi_data_cropped.bin")