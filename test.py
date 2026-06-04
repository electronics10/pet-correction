from pathlib import Path
import shutil
from mcgpu_backend.runner import Runner
from mcgpu_backend.in_generator import InFileGenerator
from mcgpu_backend.phantom import Phantom
from mcgpu_backend.vox_generator import VoxFileGenerator

# Create the running directory
run_dir = Path("data/run_2")
run_dir.mkdir(parents=True, exist_ok=True)

# Copy the `.json` file into the running directory and change the name
shutil.copy("./mcgpu_backend/templates/template.json", (run_dir / "config.json"))

# Manually change the specification in config.json as you want

# Generate the `.in` file according to the configuration automatically
config = InFileGenerator.load_config(run_dir / "config.json")
gen = InFileGenerator()
gen.from_config(config)
out = gen.write(run_dir)
print(f"Wrote {out}")

# Copy the `.vox` file into the running directory and change the name
# shutil.copy("./mcgpu_backend/templates/template.vox", (run_dir / "phantom.vox"))
import numpy as np

material = np.ones([9,9,9])
density = np.ones([9,9,9])
activity = np.ones([9,9,9])
voxel = (10,10,10)
phan = Phantom(material, density, activity, voxel)
vfg = VoxFileGenerator(phan)
vfg.write(run_dir)

# Run 
run = Runner()
result = run(run_dir, on_existing="overwrite")
print(result.sinogram_trues, result.wall_time_s)