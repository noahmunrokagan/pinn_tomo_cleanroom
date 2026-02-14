# Clean room implementation of PINNTOMO: SEISMIC TOMOGRAPHY USING PHYSICS-INFORMED NEURAL NETWORKS
This project was done to practice scientific programming and seismology based PINN designing/training
Additionally, this was done to practice DVC for datasets generated, and ML models with GCloud
## Data
The data is synthetically generated using the Marmousi model downloaded from https://www.geoazur.fr/WIND/bin/view/Main/Data/Marmousi, processed with the `deepwave` library
The marmousi model is cropped to allow for running locally

## Data processing
To prepare the synthetic data for the PINN, we transform raw wavefields into traveltimes:Method: First arrival times are extracted from Deepwave-generated traces using a 5% peak-amplitude threshold.Coordinate Mapping: Pixel indices are converted to physical meters ($dx = 4.0m$) to align with the Eikonal equation's spatial requirements.Output: A dataset of 5,000 source-receiver pairs $(x_s, z_s, x_r, z_r)$ and their corresponding observed traveltimes ($t_{obs}$).

## Model
The goal of this model which follows the theory outlined in Waheed et al 2021, uses the Eikonal equation to approximate the Marmousi model with a PINN

## Key Differences
- I used a batch size of 64, with 5000 samples (due to running the training locally on cpu)

## Reference
Waheed, U. bin, Alkhalifah, T., Haghighat, E., Song, C., & Virieux, J. (2021). PINNtomo: Seismic tomography using physics-informed neural networks (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2104.01588
