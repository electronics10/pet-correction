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

Then, create a folder `./data/run_0` and copy `template.in` and `template.vox` inside, change their names to `MCGPU-PET.in` and `phantom.vox`. 

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

```json
# ./mcgpu_backend/templates/template.json
{
  "geometry": {
  "scanner_radius_mm": 90.5,
  "num_crystals_per_ring": 336,
  "num_rings": 80,
  "axial_fov_mm": 126.56,
  "radial_trim": 95,
  "num_radial_bins": 147,
  "num_angular_bins": 168,
  "max_ring_difference": 79,
  "span": 11,
  "img_shape": [147, 147, 80],
  "voxel_size_mm": [1.0, 1.0, 1.0]
},
  "mcgpu": {
    "random_seed": 0,
    "gpu_number": 0,
    "gpu_threads_per_block": 32,
    "density_scale_factor": 1.0,

    "acquisition_time_s": 100.0,
    "isotope_mean_life_s": 70000.0,

    "psf_filename": "MCGPU_PET.psf",
    "psf_max_elements": 150000000,
    "report_trues_scatter": 0,
    "report_psf_sinogram": 0,

    "tally_material_dose": "YES",
    "tally_voxel_dose": "NO",
    "dose_filename": "mc-gpu_dose.dat",
    "dose_roi_x": [1, 147],
    "dose_roi_y": [1, 147],
    "dose_roi_z": [1, 159],

    "energy_resolution": 0.12,
    "energy_window_low_eV": 350000.0,
    "energy_window_high_eV": 600000.0,

    "tally_image_resolution": 128,
    "tally_num_z_slices": 159,
    "num_energy_bins": 700,

    "phantom_file": "phantom.vox",
    "materials": [
      "materials/air_5-515keV.mcgpu.gz",
      "materials/water_5-515keV.mcgpu.gz"
    ]
  }
}
```

Then we write a `InFileGenerator` object:

```python
"""
mcgpu_backend/in_generator.py

Generate an MCGPU-PET .in file from a structured config (geometry + mcgpu),
or directly from a flat dict of schema fields.

Pipeline:
    config = {"geometry": {...}, "mcgpu": {...}}
        → _check_consistency  (sanity assertions across fields)
        → _translate          (unit conversion + flatten to schema keys)
        → apply               (mutate self.lines in place via the schema index)
        → write(run_dir)      (run_dir/MCGPU-PET.in)

The SCHEMA maps readable field names → (section, value-line index) in the
template. SCHEMA deals only in MCGPU's native units (cm, eV, etc.); all unit
conversion happens in _translate.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any


# --- Schema: readable name → (section header, value-line index in section) --
# value-line index counts non-empty, non-pure-comment lines, resetting per section.
SCHEMA: dict[str, tuple[str, int]] = {
    # SIMULATION CONFIG
    "random_seed":           ("SIMULATION CONFIG", 0),
    "gpu_number":            ("SIMULATION CONFIG", 1),
    "gpu_threads_per_block": ("SIMULATION CONFIG", 2),
    "density_scale_factor":  ("SIMULATION CONFIG", 3),

    # SOURCE PET SCAN
    "acquisition_time":      ("SOURCE PET SCAN", 0),
    "isotope_mean_life":     ("SOURCE PET SCAN", 1),

    # PHASE SPACE FILE
    "psf_filename":          ("PHASE SPACE FILE", 0),
    "detector_geometry":     ("PHASE SPACE FILE", 1),  # "X Y Z H RADIUS" in cm
    "psf_max_elements":      ("PHASE SPACE FILE", 2),
    "report_trues_scatter":  ("PHASE SPACE FILE", 3),
    "report_psf_sinogram":   ("PHASE SPACE FILE", 4),

    # DOSE DEPOSITION
    "tally_material_dose":   ("DOSE DEPOSITION", 0),
    "tally_voxel_dose":      ("DOSE DEPOSITION", 1),
    "dose_filename":         ("DOSE DEPOSITION", 2),
    "dose_roi_x":            ("DOSE DEPOSITION", 3),   # "min max"
    "dose_roi_y":            ("DOSE DEPOSITION", 4),
    "dose_roi_z":            ("DOSE DEPOSITION", 5),

    # ENERGY PARAMETERS
    "energy_resolution":     ("ENERGY PARAMETERS", 0),
    "energy_window_low":     ("ENERGY PARAMETERS", 1),  # eV (despite template's "keV" comment)
    "energy_window_high":    ("ENERGY PARAMETERS", 2),

    # SINOGRAM PARAMETERS
    "axial_fov_cm":          ("SINOGRAM PARAMETERS", 0),
    "num_rings":             ("SINOGRAM PARAMETERS", 1),
    "total_crystals":        ("SINOGRAM PARAMETERS", 2),
    "num_angular_bins":      ("SINOGRAM PARAMETERS", 3),
    "num_radial_bins":       ("SINOGRAM PARAMETERS", 4),
    "num_z_slices":          ("SINOGRAM PARAMETERS", 5),
    "image_resolution":      ("SINOGRAM PARAMETERS", 6),
    "num_energy_bins":       ("SINOGRAM PARAMETERS", 7),
    "max_ring_difference":   ("SINOGRAM PARAMETERS", 8),
    "span":                  ("SINOGRAM PARAMETERS", 9),

    # VOXELIZED GEOMETRY FILE
    "phantom_file":          ("VOXELIZED GEOMETRY FILE", 0),

    # MATERIAL FILE LIST
    # Positional: template currently has 2 slots. To support N materials,
    # add material{N}_file entries AND add the corresponding lines to template.in.
    "material1_file":        ("MATERIAL FILE LIST", 0),
    "material2_file":        ("MATERIAL FILE LIST", 1),
}

_SECTION_RE = re.compile(r'^\s*#\[SECTION (.+?)\]')
_VALUE_COLUMN = 31  # comment alignment column (cosmetic)


class InFileGenerator:
    """Edit an MCGPU-PET .in file via a structured config or flat schema keys.

    Typical usage:
        cfg = InFileGenerator.load_config("template.json", "data/run_42/delta.json")
        gen = InFileGenerator("mcgpu_backend/templates/template.in")
        gen.from_config(cfg)
        gen.write("data/run_42")          # writes data/run_42/MCGPU-PET.in
    """

    def __init__(self, template_path: str | Path = "mcgpu_backend/templates/template.in"):
        self.lines = Path(template_path).read_text().splitlines()
        self._index = self._build_index()

    # ----- low-level: schema layer ------------------------------------------

    def _build_index(self) -> dict[tuple[str, int], int]:
        """Map (section, value_line_index) → absolute line number in self.lines."""
        index: dict[tuple[str, int], int] = {}
        section: str | None = None
        n = 0
        for i, line in enumerate(self.lines):
            m = _SECTION_RE.match(line)
            if m:
                section = m.group(1).split(' v.')[0].strip()
                n = 0
                continue
            stripped = line.strip()
            if section and stripped and not stripped.startswith('#'):
                index[(section, n)] = i
                n += 1
        return index

    def _set_addr(self, section: str, line_no: int, value: Any) -> None:
        try:
            idx = self._index[(section, line_no)]
        except KeyError:
            raise KeyError(
                f"({section!r}, {line_no}) not in template. "
                f"Check section header text and value-line count."
            )
        old = self.lines[idx]
        cpos = old.find('#')
        comment = old[cpos:] if cpos != -1 else ''
        leading = old[: len(old) - len(old.lstrip())]
        val_str = f"{leading}{value}"
        pad = max(_VALUE_COLUMN - len(val_str), 1)
        self.lines[idx] = f"{val_str}{' ' * pad}{comment}".rstrip()

    def apply(self, flat: dict[str, Any]) -> None:
        """Apply a dict of schema keys → values. Raises KeyError on unknown keys."""
        for key, value in flat.items():
            if key not in SCHEMA:
                raise KeyError(f"Unknown schema field {key!r}.")
            section, line_no = SCHEMA[key]
            self._set_addr(section, line_no, value)

    # ----- high-level: config-driven ----------------------------------------

    def from_config(self, config: dict[str, dict[str, Any]]) -> None:
        """Apply a structured config {"geometry": {...}, "mcgpu": {...}}."""
        self._check_consistency(config)
        self.apply(self._translate(config))

    @staticmethod
    def _check_consistency(config: dict) -> None:
        g = config["geometry"]
        m = config["mcgpu"]

        # Sinogram radial dim ↔ crystal count + radial_trim (hard identity)
        expected_rad = g["num_crystals_per_ring"] + 1 - 2 * g["radial_trim"]
        if g["num_radial_bins"] != expected_rad:
            raise ValueError(
                f"num_radial_bins={g['num_radial_bins']} but expected "
                f"{expected_rad} = num_crystals_per_ring + 1 - 2*radial_trim."
            )

        # Angular bins ↔ crystals/2 (convention, breakable e.g. with mashing)
        if g["num_angular_bins"] != g["num_crystals_per_ring"] // 2:
            print(
                f"WARNING: num_angular_bins={g['num_angular_bins']} "
                f"≠ num_crystals_per_ring/2={g['num_crystals_per_ring']//2}. "
                f"Make sure this is intentional."
            )

        # MRD must fit
        if g["max_ring_difference"] >= g["num_rings"]:
            raise ValueError(
                f"max_ring_difference={g['max_ring_difference']} must be "
                f"< num_rings={g['num_rings']}."
            )

        # Energy window
        if m["energy_window_low_eV"] >= m["energy_window_high_eV"]:
            raise ValueError("energy_window_low_eV must be < energy_window_high_eV.")

        # Time
        if m["acquisition_time_s"] <= 0:
            raise ValueError("acquisition_time_s must be > 0.")

        # Materials: schema currently has exactly 2 slots
        if len(m["materials"]) != 2:
            raise ValueError(
                f"materials has {len(m['materials'])} entries; schema supports 2. "
                f"To extend: add material{{N}}_file to SCHEMA AND template.in."
            )

    @staticmethod
    def _translate(config: dict) -> dict[str, Any]:
        """Config (mm, s, eV) → flat schema dict in MCGPU's native units (cm, s, eV)."""
        g = config["geometry"]
        m = config["mcgpu"]

        scanner_radius_cm = g["scanner_radius_mm"] / 10.0
        axial_fov_cm = g["axial_fov_mm"] / 10.0
        # Detector cylinder: X Y Z H RADIUS in cm; negative RADIUS means
        # "centered on the voxel geometry," which is what we want.
        detector_geom = f"0.0 0.0 0.0 {axial_fov_cm} -{scanner_radius_cm}"

        return {
            # SIMULATION CONFIG
            "random_seed":           m["random_seed"],
            "gpu_number":            m["gpu_number"],
            "gpu_threads_per_block": m["gpu_threads_per_block"],
            "density_scale_factor":  m["density_scale_factor"],

            # SOURCE PET SCAN
            "acquisition_time":      m["acquisition_time_s"],
            "isotope_mean_life":     m["isotope_mean_life_s"],

            # PHASE SPACE FILE
            "psf_filename":          m["psf_filename"],
            "detector_geometry":     detector_geom,
            "psf_max_elements":      m["psf_max_elements"],
            "report_trues_scatter":  m["report_trues_scatter"],
            "report_psf_sinogram":   m["report_psf_sinogram"],

            # DOSE DEPOSITION
            "tally_material_dose":   m["tally_material_dose"],
            "tally_voxel_dose":      m["tally_voxel_dose"],
            "dose_filename":         m["dose_filename"],
            "dose_roi_x":            f"{m['dose_roi_x'][0]} {m['dose_roi_x'][1]}",
            "dose_roi_y":            f"{m['dose_roi_y'][0]} {m['dose_roi_y'][1]}",
            "dose_roi_z":            f"{m['dose_roi_z'][0]} {m['dose_roi_z'][1]}",

            # ENERGY PARAMETERS
            "energy_resolution":     m["energy_resolution"],
            "energy_window_low":     m["energy_window_low_eV"],
            "energy_window_high":    m["energy_window_high_eV"],

            # SINOGRAM PARAMETERS
            "axial_fov_cm":          axial_fov_cm,
            "num_rings":             g["num_rings"],
            "total_crystals":        g["num_crystals_per_ring"],
            "num_angular_bins":      g["num_angular_bins"],
            "num_radial_bins":       g["num_radial_bins"],
            "num_z_slices":          m["tally_num_z_slices"],
            "image_resolution":      m["tally_image_resolution"],
            "num_energy_bins":       m["num_energy_bins"],
            "max_ring_difference":   g["max_ring_difference"],
            "span":                  g["span"],

            # VOXELIZED GEOMETRY FILE
            "phantom_file":          m["phantom_file"],

            # MATERIAL FILE LIST
            "material1_file":        m["materials"][0],
            "material2_file":        m["materials"][1],
        }

    # ----- I/O --------------------------------------------------------------

    def write(self, run_dir: str | Path) -> Path:
        """Write current state to run_dir/MCGPU-PET.in. Returns the path."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "MCGPU-PET.in"
        out_path.write_text('\n'.join(self.lines) + '\n')
        return out_path

    @staticmethod
    def load_config(*paths: str | Path) -> dict:
        """Load and shallow-deep-merge JSON configs. Later files win per-field.

        Pattern: load_config("template.json", "data/run_42/delta.json")
        Merging is per-section: within "geometry" or "mcgpu", later keys
        override earlier ones; sections themselves are merged, not replaced.
        """
        merged: dict = {}
        for p in paths:
            cfg = json.loads(Path(p).read_text())
            for section, fields in cfg.items():
                merged.setdefault(section, {}).update(fields)
        return merged


if __name__ == "__main__":
    config = InFileGenerator.load_config("mcgpu_backend/templates/template.json")
    gen = InFileGenerator()
    gen.from_config(config)
    out = gen.write("data/run_0")
    print(f"Wrote {out}")
```

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

### Vox generator

We want a wrapper that can generate `phantom.vox` from possibly a user defined phantom, a standard NEMA phantom, or even a digital mouse.

The `.vox` file is MCGPU-PET's source description: a voxelized map of material ID, mass density, and activity (Bq per voxel). The format is penEasy 2008 with one PET-specific addition — a third column for activity — and it sits in the first octant with voxel `{1,1,1}` cornered at the origin, x-index running fastest. We could write these files by hand for every experiment, but that doesn't scale: a phantom is a *modeling decision* (what geometry, what materials, what activity distribution), and modeling decisions are worth recording, parameterizing, and reusing. We want a Python-side `Phantom` object that owns the three voxel arrays (material, density, activity) plus the voxel size, and a `VoxFileGenerator` that serializes it into the format MCGPU-PET parses.

#### Phantom Object
```python
# mcgpu_backend/phantom.py
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Phantom:
    """A voxelized phantom: per-voxel material ID, density, and activity.

    Shape convention: all three arrays have shape (Nz, Ny, Nx). Indexing as
    arr[z, y, x] matches the natural reading of axial slices arr[z].

    Material IDs should start from 1 instead of 0 (MCGPU doesn't allow).
    """
    material_id: np.ndarray  # (Nz, Ny, Nx), uint8
    density: np.ndarray  # (Nz, Ny, Nx), float32, g/cm^3
    activity: np.ndarray  # (Nz, Ny, Nx), float32, Bq per voxel
    voxel_size_mm: tuple[float, float, float]  # (dx, dy, dz) mm
    material_names: list[str] = field(default_factory=list)  # bookkeeping
```

Note that density is not determined by the material. This is because, to my understanding, by material, MCGPU-PET actually mean something like $\mu/\rho$ ( is the linear attenuation coefficient and $\rho$ is the density). To be more precise, $\mu/\rho = \frac{Z}{A}N_A\sigma$, which is exactly specified by the material in microscopic scale (and energy).

#### VoxFileGenerator
```python
# mcgpu_backend/vox_generator.py
from mcgpu_backend.phantom import Phantom
from pathlib import Path
import gzip


class VoxFileGenerator:
    
    def __init__(self, phantom: Phantom):
        phantom.validate()
        self.phantom = phantom

    def _build_header(self) -> str:
        nx, ny, nz = self.phantom.shape_xyz
        dx_cm, dy_cm, dz_cm = (d / 10.0 for d in self.phantom.voxel_size_mm)
        # Match the spacing/comment style of the penEasy sample for readability.
        lines = [
            "[SECTION VOXELS HEADER v.2008-04-13]",
            f"{nx} {ny} {nz}       No. OF VOXELS IN X,Y,Z",
            f"{dx_cm:.6g} {dy_cm:.6g} {dz_cm:.6g}       VOXEL SIZE (cm) ALONG X,Y,Z",
            "1                    COLUMN NUMBER WHERE MATERIAL ID IS LOCATED",
            "2                    COLUMN NUMBER WHERE THE MASS DENSITY [g/cm3] IS LOCATED",
            "0                    BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)",
            "[END OF VXH SECTION]",
            "",
        ]
        return "\n".join(lines)

    def _build_body(self) -> str:
        # Array is (Nz, Ny, Nx). ravel(order='C') gives last-axis-fastest = x-fastest.
        mat = self.phantom.material_id.ravel(order="C")
        rho = self.phantom.density.ravel(order="C")
        act = self.phantom.activity.ravel(order="C")

        # Compose lines efficiently. Plain join with newlines is dominant cost.
        # Format: "<mat> <density:.6g> <activity:.6g>"
        # Use numpy formatting only if very large; otherwise a list comp is fine.
        n = mat.size
        # Pre-format with vectorized string operations for speed
        # (savetxt would also work but gives less control over formatting)
        parts = [
            f"{m} {d:.6g} {a:.6g}"
            for m, d, a in zip(mat.tolist(), rho.tolist(), act.tolist())
        ]
        # Append trailing newline for cleanliness
        return "\n".join(parts) + "\n"
    
    def write(
        self,
        run_dir: str | Path,
        filename: str = "phantom.vox",
        gzipped: bool = False,
    ) -> Path:
        """Write to run_dir/filename. Returns the written path.

        If gzipped=True, automatically appends '.gz' to filename if not present.
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        if gzipped and not filename.endswith(".gz"):
            filename = filename + ".gz"
        out_path = run_dir / filename

        header = self._build_header()
        body = self._build_body()
        payload = header + body

        if gzipped:
            with gzip.open(out_path, "wt") as f:
                f.write(payload)
        else: out_path.write_text(payload)

        return out_path
```

Run to check
```python


##### Legacy
For the first phantom, we implement the **NEMA NU 4-2008 Image Quality (IQ) phantom**. The NEMA standards define a small set of physical phantoms with fixed geometry that every preclinical PET scanner is benchmarked against, which is exactly why we want one: it gives us a phantom whose simulated results can be compared directly against published measurements (e.g. Table 1 and Fig. 4 of the MCGPU-PET paper, which validates on this same phantom). The IQ phantom is also geometrically simple — a PMMA cylinder with a uniform hot region, two cold inserts, and five hot rods of decreasing diameter — so it factors cleanly into a few primitive operations (`add_cylinder`, `add_sphere`, `fill_background`) that any future phantom will also need. Concretely, we'll build a `PhantomBuilder` exposing these primitives, a `nema_iq_preclinical(...)` factory that composes them into the standard IQ geometry with user-specified activity concentrations, and the `VoxFileGenerator` that writes the result. Once this works end-to-end, custom phantoms and (later) anatomical phantoms reduce to writing new factories against the same builder — the serialization, the simulation runner, and the configuration system don't change.