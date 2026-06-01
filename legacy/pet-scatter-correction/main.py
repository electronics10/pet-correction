from mcgpu_backend import*
from pathlib import Path
import json
import numpy as np
import gzip


TOTAL_RUN = 1

def run_experiment(total_run):
    for i in range(total_run):
        # experiment storage directory
        run_dir = f"runs/run{i}/"

        # generate JSON configuration
        scanner_config = {
            "acquisition_time": 600.0, # 600 seconds
            "detector_geometry": "0.0  0.0  0.0  12.656  -9.05",  # centered*3, height[cm], -radius[cm]
            "energy_window_low": 350000.0, # kev
            "energy_window_high": 600000.0, # kev

            # [SECTION SINOGRAM PARAMETERS]
            "axial_fov_cm": 12.656,
            "num_rings": 80,
            "total_crystals": 336,
            "num_angular_bins": 168, # total_crystals / 2
            "num_radial_bins": 147,
            "num_z_slices": 159, # num_rings*2 - 1
            "max_ring_difference": 79, # num_rings - 1
            "span": 11,
        }
        json_path = Path(run_dir + "config.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(scanner_config, f, indent=4)
            print(f"Write {json_path}")


        # generate MCGPU-PET.in
        ifg = InFileGenerator()
        ifg.from_json(run_dir)
        ifg.write(run_dir)

        # generate phatom.vox
        pvg = PhantomVoxGenerator()
        pvg.write(run_dir)

        # run mcgpu-pet simulation
        run_mcgpu_pet(run_dir)

for i in range(TOTAL_RUN):
    """ TODO:
    1. Show image_xxx.raw.gz
    2. Show sinogram_xxx.raw.gz (cls: DataLoader from data_loader.py)
    3. Output direct_sinograms (load stack of direct sinograms from rebinning algorithm such as SSRB)
    3. Show reconstructed sinogram_xxx.raw.gz (FBP)
    """
    pass
    