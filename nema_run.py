"""
example_nema_run.py

End-to-end example: build a NEMA IQ phantom, generate the .in file from
config, write the phantom to disk, and launch MCGPU-PET via the Runner.

This mirrors the data/run_1 example in the tutorial but replaces the manual
template.vox copy with a programmatic NEMA phantom.
"""

from pathlib import Path
import shutil

from mcgpu_backend.runner import Runner
from mcgpu_backend.in_generator import InFileGenerator
from mcgpu_backend.vox_generator import VoxFileGenerator
from mcgpu_backend.phantoms import nema_iq_preclinical


# 1. Create the running directory
run_dir = Path("data/run_nema_iq")
run_dir.mkdir(parents=True, exist_ok=True)

# 2. Copy and edit the config (in practice you'd edit fields by hand or in code)
shutil.copy("./mcgpu_backend/templates/template.json", run_dir / "config.json")
# (Manually edit config.json if needed before the next step.)

# 3. Generate the .in file from the config
config = InFileGenerator.load_config(run_dir / "config.json")
gen = InFileGenerator()
gen.from_config(config)
gen.write(run_dir)

# 4. Build the NEMA IQ phantom and write it as phantom.vox
#    Geometry must be consistent with config["geometry"]["img_shape"] and
#    config["geometry"]["voxel_size_mm"] — the phantom IS the simulation source,
#    so its shape determines what MCGPU sees.
geom = config["geometry"]
phantom = nema_iq_preclinical(
    voxel_size_mm=tuple(geom["voxel_size_mm"]),
    fov_shape_xyz=tuple(geom["img_shape"]),
    hot_activity_Bq_per_mL=3700.0,          # 0.1 μCi/mL, typical 18F
    # Default materials: air=(1, 0.0012), water=(2, 1.0), pmma=(2, 1.19).
    # If you add a real PMMA cross-section file as material 3, override here:
    #   materials={"air": (1, 0.0012), "water": (2, 1.0), "pmma": (3, 1.19)}
)
print(f"Phantom: shape={phantom.shape_zyx}, "
      f"total activity={phantom.total_activity_Bq:.0f} Bq")

VoxFileGenerator(phantom).write(run_dir, "phantom.vox", gzipped=False)

# 5. Run the simulation
result = Runner()(run_dir, on_existing="skip")
print(f"Done. Sinogram at {result.sinogram_trues}, wall time {result.wall_time_s:.1f}s")