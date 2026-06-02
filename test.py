from pathlib import Path
import shutil
from mcgpu_backend.runner import Runner
from mcgpu_backend.in_generator import InFileGenerator

# Create the running directory
run_dir = Path("data/run_1")
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
shutil.copy("./mcgpu_backend/templates/template.vox", (run_dir / "phantom.vox"))

# Run 
run = Runner()
result = run(run_dir, on_existing="skip")
print(result.sinogram_trues, result.wall_time_s)