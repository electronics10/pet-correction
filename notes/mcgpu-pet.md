## Prerequisite for MCGPU-PET
[MCGPU-PET](https://github.com/DIDSR/MCGPU-PET.git) is a research code; the authors likely developed it by extending NVIDIA CUDA sample programs. Therefore, CUDA samples dependencies such as `helper_functions.h` are essential but not shipped with the distribution. Moreover, since GPU binaries are architecture-specific (unlike x86-64 CPU binaries), distributing a prebuilt `.x` binary is impractical.

### Installation

Clone the repository:
```bash
git clone https://github.com/DIDSR/MCGPU-PET.git
cd MCGPU-PET
```

The expected environment is Linux. To get the executive file `MCGPU-PET.x`, run
```bash
make
```
However, running `make` directly will likely fail due to the two issues above. If so, fix them as shown in the details.

<details>

1. **CUDA samples dependency**

Generally, find or install `helper_functions.h`:
```bash
# Option A: install via apt (replace version as needed)
sudo apt install cuda-samples-12-0

# Option B: clone NVIDIA's samples repo
git clone https://github.com/NVIDIA/cuda-samples.git
```
Then set `CUDA_SDK_PATH` in the `Makefile` to the `Common/inc` directory of whichever option you used.

Personally, I did
```bash
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Fri_Jan__6_16:45:21_PST_2023
# Cuda compilation tools, release 12.0, V12.0.140
# Build cuda_12.0.r12.0/compiler.32267302_0
```
Since the release is 12.0, I clone the CUDA samples by
```bash
# Clone the matching samples version to ~/cuda-samples
git clone --branch v12.0 https://github.com/NVIDIA/cuda-samples.git ~/cuda-samples
```

Then in `Makefile`, I changed the line:
```makefile
CUDA_SDK_PATH = $(HOME)/cuda-samples/Common/
```

2. **GPU compute capability**

Check your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```
Update `GPU_COMPUTE_CAPABILITY` in the `Makefile` accordingly, e.g. for capability 7.5:
```makefile
GPU_COMPUTE_CAPABILITY = -gencode=arch=compute_75,code=sm_75
```

(Personally, mine is the same as the authors so I don't need to change anything.)

Then compile:
```bash
make
```

(Compilation may succeeds with warnings, which are safe to ignore.)
</details>

### Basic IO

After you get the `MCGPU-PET.x` binary file, it becomes an independent executive binary file (which serves as a MC PET simulator). That is to say, you can deploy the file to any places/projects without any worries, it would work fine on your computer. Personally, I used it as a backend for my Python project. According to the authors, the simulator takes in a `.in` file (`MCGPU-PET.in` in sample example) and is typically used by prompting 
```bash
time ./MCGPU-PET.x MCGPU-PET.in | tee MCGPU-PET.out
```

Note that the `.in` file requires a `.vox` file (simulation object such as a phantom) and some `.gz` files (materials). 


## Python wrapper for MCGPU-PET

To make MCGPU-PET more accessible and flexible, I am developing a Python wrapper for it called mcgpu_backend. First, create a folder `./mcgpu_backend` and a `__init__.py` in it. Also, put the binary `MCGPU-PET.x`, `template.in` and `template.vox` in a `templates` folder, and a folder of `materials` in it, and the basic components are all set.

### A first run (no wrapper)

Before designing anything, run MCGPU-PET and look at what falls out. The wrapper's design decisions all trace back to frictions we hit here.

MCGPU-PET resolves the paths in the input file *relative to the current working directory*, and it writes its outputs into that same directory. The path of least resistance is therefore to make a fresh directory per run and pull the binary, the input file, the phantom, and the materials in by symlink or copy:

```bash
mkdir -p data/raw_run_1
cd data/raw_run_1

ln -s ../../mcgpu_backend/MCGPU-PET.x .
ln -s ../../mcgpu_backend/materials  .
cp ../../mcgpu_backend/templates/template.in   MCGPU-PET.in
cp ../../mcgpu_backend/templates/template.vox  phantom.vox
```

Two conventions are already baked in: the input file is named `MCGPU-PET.in` (the README convention), and it references `phantom.vox` and `materials/...` by those exact names (open `template.in` to confirm). If the names in the directory don't match the strings inside the `.in` file, the binary either crashes or silently uses defaults. This is a constraint from MCGPU-PET.

**To run:**

```bash
time ./MCGPU-PET.x MCGPU-PET.in 2>&1 | tee MCGPU-PET.out
```

The `tee` saves a log alongside the outputs. MCGPU-PET prints the resolved geometry, the counts, and any warnings to stdout.

**What comes out:**

A single run with the default template produces:

| File | Format | Content |
|---|---|---|
| `image_Trues.raw.gz`   | gzipped `int32`, shape $(N_z, N_y, N_x)$ | per-voxel count of *true* coincidences emitted from that voxel |
| `image_Scatter.raw.gz` | same | same, for scattered coincidences |
| `sinogram_Trues.raw.gz`   | gzipped `int32`, flat buffer | trues binned into a 3D PET sinogram (michelogram) |
| `sinogram_Scatter.raw.gz` | same | same, for scatter |
| `Energy_Sinogram_Spectrum.dat` | text | energy spectrum of detected events |
| `MCGPU-PET.out` | text | our tee'd log |


**Emission images** are *not* reconstructions. They are forward tallies — for each voxel, the count of coincidences within the energy window that *originated* there. A real scanner cannot observe this; only the simulator can. They're useful as a sanity check (Trues + Scatter should approximate the input activity map, Poisson-noisy and sensitivity-weighted) and as a ground-truth reference, but not as data we feed into reconstruction.

**Energy spectrum** is a text file we don't use downstream; mentioned for completeness.

**Sinograms** are what reconstruction acts on. They're written as a single flat `int32` stream whose 3D layout is dictated by the `SINOGRAM PARAMETERS` block in the `.in` file — specifically `num_rings`, `MRD`, and `span`. The data format is something worth inspection and deal with in the later section.

**Problems:**

From the above example, it is clear that there are a few problems that need to be deal with:
1. We need a wrapper that can make the call of a run easier for later machine learning data generation pipeline.
2. We need a `.in` file generator based on a specified configuration that can probably be shared with ML and reconstruction (parallelproj) usage.
3. We need a `.vox` generator.
4. We need to know the format of the sinogram output and how to convert them to be suitable for training and reconstruction.

### Runner
Instead of bash, we write a callabe runnner object that take in the path of a directory, run the simulation, and store the output in the directory. Then, create a folder `./data/run_0` and copy `template.in` and `template.vox` inside, change their names to `MCGPU-PET.in` and `phantom.vox`. 

```python
from pathlib import Path
import shutil
from mcgpu_backend.runner import Runner


# Create the running directory
run_dir = Path("data/run_0")
run_dir.mkdir(parents=True, exist_ok=True)

# Copy the `.in` and `.vox` file into the running directory and change their name
shutil.copy("./mcgpu_backend/templates/template.in", (run_dir / "MCGPU-PET.in"))
shutil.copy("./mcgpu_backend/templates/template.vox", (run_dir / "phantom.vox"))

# Run 
run = Runner()
result = run(run_dir, on_existing="skip")
print(result.sinogram_trues, result.wall_time_s)
```

Naturally, the next thing to deal with is how to create the a configuration file (source of truth) that can be shared for reconstruction or ML and generate the `.in` file from it.

### Configuration JSON and .in file generator

The `.in` file for MCGPU-PET is quite static and difficult to both manipulate and access. Moreover, we need a source of truth to record the meta data of each simulation run, and this is where a `.json` file come in play. The design of the `json` file includes the geometry and the mcgpu specification. The idea of the geometry specification is to state the domain and the codomain of the operation (consider PET as an mathematical transformation) while the mcgpu specification complement the geometry by stating other parameters such as acquisition time and energy window, which are more physical-oriented.

Copy the `template.json` file from `./mcgpu_backend/templates/` to the running directory `./data/run_1` as `conifg.json` and manually change the specifications.

```python
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
```

### 