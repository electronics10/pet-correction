## Prerequisite of MCGPU-PET
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

To make MCGPU-PET more accessible and flexible, I am developing a Python wrapper for it called mcgpu_backend. First, create a folder `./mcgpu_backend` and a `__init__.py` in it. Also, put the binary `MCGPU-PET.x`, `template.in`, `template.vox`, and a folder of `materials` in it, and the basic components are all set.

### A first run (no wrapper)

Before designing anything, run MCGPU-PET and look at what falls out. The wrapper's design decisions all trace back to frictions we hit here.

MCGPU-PET resolves the paths in the input file *relative to the current working directory*, and it writes its outputs into that same directory. The path of least resistance is therefore to make a fresh directory per run and pull the binary, the input file, the phantom, and the materials in by symlink or copy:

```bash
mkdir -p data/raw_run_1
cd data/raw_run_1

ln -s ../../mcgpu_backend/MCGPU-PET.x .
ln -s ../../mcgpu_backend/materials  .
cp ../../mcgpu_backend/template.in   MCGPU-PET.in
cp ../../mcgpu_backend/template.vox  phantom.vox
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

**Sinograms** are what reconstruction acts on. They're written as a single flat `int32` stream whose 3D layout is dictated by the `SINOGRAM PARAMETERS` block in the `.in` file — specifically `num_rings`, `MRD`, and `span`.

**Problems:**

From the above example, it is clear that there are a few problems that need to be deal with:
1. We need a wrapper that can make the call of a run easier for later machine learning data generation pipeline.
2. We need a `.in` file generator based on a specified configuration that can probably be shared with ML and reconstruction (parallelproj) usage.
3. We need a `.vox` generator.
4. We need to know the format of the sinogram output and how to convert them to be suitable for training and reconstruction.

### Runner
Instead of bash, we write a callabe runnner object that take in the path of a directory, run the simulation, and store the output in the directory. 

```python
# ./mcgpu_backend/runner.py
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import subprocess
import threading
import time

PKG_DIR = Path(__file__).parent  # where MCGPU-PET.x and materials/ live


@dataclass
class RunResult:
    run_dir: Path
    image_trues: Path
    image_scatter: Path
    sinogram_trues: Path
    sinogram_scatter: Path
    energy_spectrum: Path
    log: Path
    wall_time_s: float
    returncode: int


class Runner:
    def __init__(self, binary=PKG_DIR / "MCGPU-PET.x", materials=PKG_DIR / "materials"):
        self.binary, self.materials = Path(binary), Path(materials)

    def __call__(
        self,
        run_dir,
        on_existing: Literal["error", "overwrite", "skip"] = "error",
        verbose: bool = True,
        timeout_s: Optional[float] = 3600,
    ) -> RunResult:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

        existing = self._handle_existing(run_dir, on_existing)
        if existing is not None:
            return existing

        self._stage(run_dir)
        self._preflight(run_dir)
        rc, dt = self._execute(run_dir, verbose, timeout_s)
        return self._collect(run_dir, rc, dt)

    def _stage(self, d):
        for src, name in [(self.binary, "MCGPU-PET.x"), (self.materials, "materials")]:
            link = d / name
            if not link.exists():
                link.symlink_to(src.resolve())

    def _preflight(self, d):
        required = ["MCGPU-PET.x", "MCGPU-PET.in", "phantom.vox", "materials"]
        missing = [r for r in required if not (d / r).exists()]
        if missing:
            raise FileNotFoundError(f"Missing in {d}: {missing}")

    def _execute(self, d, verbose, timeout_s):
        log = d / "MCGPU-PET.out"
        t0 = time.perf_counter()
        with open(log, "w") as f:
            p = subprocess.Popen(
                ["./MCGPU-PET.x", "MCGPU-PET.in"],
                cwd=d,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            timer = threading.Timer(timeout_s, p.kill) if timeout_s else None
            if timer:
                timer.start()
            try:
                for line in p.stdout:
                    if verbose:
                        print(line, end="")
                    f.write(line)
                p.wait()
            finally:
                if timer:
                    timer.cancel()
        dt = time.perf_counter() - t0
        if p.returncode < 0:  # negative returncode = killed by signal
            raise RuntimeError(
                f"Run exceeded {timeout_s}s, killed (signal {-p.returncode})"
            )
        return p.returncode, dt

    def _collect(self, d, rc, dt):
        expected = [
            "image_Trues.raw.gz",
            "image_Scatter.raw.gz",
            "sinogram_Trues.raw.gz",
            "sinogram_Scatter.raw.gz",
            "Energy_Sinogram_Spectrum.dat",
        ]
        missing = [
            f for f in expected
            if not (d / f).exists() or (d / f).stat().st_size == 0
        ]
        if rc != 0 or missing:
            raise RuntimeError(f"Run failed (rc={rc}); missing/empty: {missing}")
        return RunResult(d, *(d / f for f in expected), d / "MCGPU-PET.out", dt, rc)

    def _handle_existing(self, d, mode):
        outputs = list(d.glob("*.raw.gz"))
        if not outputs:
            return None
        if mode == "error":
            raise FileExistsError(f"Outputs exist in {d}")
        if mode == "skip":
            return self._collect(d, rc=0, dt=0.0)
        if mode == "overwrite":
            for f in outputs + list(d.glob("*.dat")) + [d / "MCGPU-PET.out"]:
                if f.exists():
                    f.unlink()
            return None
        raise ValueError(f"unknown on_existing mode: {mode!r}")
```

Create a folder `./data/run_0` and copy `template.in` and `template.vox` inside, change their names to `MCGPU-PET.in` and `phantom.vox`. 

```python
from mcgpu_backend.runner import Runner

run = Runner()
result = run("data/run_0")
print(result.sinogram_trues, result.wall_time_s)
```

### 