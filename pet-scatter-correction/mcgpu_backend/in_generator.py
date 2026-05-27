"""
in_generator.py

Generate an MCGPU-PET .in file from a canonical template plus one or more
JSON override files, addressed by readable names defined in schema.py.

CLI:
    python in_generator.py
.py TEMPLATE.in OUT.in config1.json [config2.json ...]

Later JSONs override earlier ones (useful: a base.json + a per-run delta).

API:
    cfg = InFileGenerator("MCGPU-PET.in")
    cfg.apply({"acquisition_time": 5.0, "energy_window_low": 400000.0})
    cfg.write("MCGPU-PET_run01.in")
"""

import json
import re
import sys
from pathlib import Path

"""
Schema: readable field name -> (section_name, value_line_index_within_section).

section_name is the SECTION header text up to (but not including) " v.<date>".
value_line_index counts only "value lines" (non-empty, not pure-comment),
starting at 0, resetting at each new section.

Add an entry here the moment you want to sweep a new parameter.
"""
SCHEMA = {
    # [SECTION SIMULATION CONFIG]
    "random_seed":              ("SIMULATION CONFIG", 0),
    "gpu_number":               ("SIMULATION CONFIG", 1),
    "gpu_threads_per_block":    ("SIMULATION CONFIG", 2),
    "density_scale_factor":     ("SIMULATION CONFIG", 3),

    # [SECTION SOURCE PET SCAN]
    "acquisition_time":         ("SOURCE PET SCAN", 0),
    "isotope_mean_life":        ("SOURCE PET SCAN", 1),
    # line 2 / 3 are the material-activity table terminators; usually left alone

    # [SECTION PHASE SPACE FILE]
    "psf_filename":             ("PHASE SPACE FILE", 0),
    "detector_geometry":        ("PHASE SPACE FILE", 1),  # "X Y Z H RADIUS" as one string
    "psf_max_elements":         ("PHASE SPACE FILE", 2),
    "report_trues_scatter":     ("PHASE SPACE FILE", 3),   # 0 both, 1 trues, 2 scatter
    "report_psf_sinogram":      ("PHASE SPACE FILE", 4),   # 0 both, 1 psf, 2 sinogram

    # [SECTION DOSE DEPOSITION]
    "tally_material_dose":      ("DOSE DEPOSITION", 0),     # YES/NO
    "tally_voxel_dose":         ("DOSE DEPOSITION", 1),     # YES/NO
    "dose_filename":            ("DOSE DEPOSITION", 2),
    "dose_roi_x":               ("DOSE DEPOSITION", 3),     # "min max"
    "dose_roi_y":               ("DOSE DEPOSITION", 4),
    "dose_roi_z":               ("DOSE DEPOSITION", 5),

    # [SECTION ENERGY PARAMETERS]
    "energy_resolution":        ("ENERGY PARAMETERS", 0),
    "energy_window_low":        ("ENERGY PARAMETERS", 1),
    "energy_window_high":       ("ENERGY PARAMETERS", 2),

    # [SECTION SINOGRAM PARAMETERS]
    "axial_fov_cm":             ("SINOGRAM PARAMETERS", 0),
    "num_rings":                 ("SINOGRAM PARAMETERS", 1),
    "total_crystals":           ("SINOGRAM PARAMETERS", 2),
    "num_angular_bins":         ("SINOGRAM PARAMETERS", 3),
    "num_radial_bins":          ("SINOGRAM PARAMETERS", 4),
    "num_z_slices":             ("SINOGRAM PARAMETERS", 5),
    "image_resolution":         ("SINOGRAM PARAMETERS", 6), # NUMBER OF BINS IN THE IMAGE
    "num_energy_bins":          ("SINOGRAM PARAMETERS", 7),
    "max_ring_difference":      ("SINOGRAM PARAMETERS", 8),
    "span":                     ("SINOGRAM PARAMETERS", 9),

    # [SECTION VOXELIZED GEOMETRY FILE]
    "phantom_file":             ("VOXELIZED GEOMETRY FILE", 0),

    # [SECTION MATERIAL FILE]
    "material1_file":           ("MATERIAL FILE LIST", 0),
    "material2_file":           ("MATERIAL FILE LIST", 1)
}
_SECTION_RE = re.compile(r'^\s*#\[SECTION (.+?)\]')
_VALUE_COLUMN = 31  # comments in the original start ~here; keeps file tidy


class InFileGenerator:
    def __init__(self, template_path = "mcgpu_backend/template.in"):
        self.lines = Path(template_path).read_text().splitlines()
        self._index = self._build_index()

    def _build_index(self):
        index = {}
        section = None
        n = 0
        for i, line in enumerate(self.lines):
            m = _SECTION_RE.match(line)
            if m:
                section = m.group(1).split(' v.')[0].strip()
                n = 0
                continue
            stripped = line.strip()
            if section and stripped and not stripped.startswith('#'):
                index[(section, n)] = i # map schema (section, n) into actual .in line
                n += 1
        return index

    def _set_addr(self, section, line_no, value): # actually change the value in the .in lines
        try:
            idx = self._index[(section, line_no)]
        except KeyError:
            raise KeyError(
                f"({section!r}, {line_no}) not found in template. "
                f"Check the section header text and value-line count."
            )
        old = self.lines[idx]
        cpos = old.find('#')
        comment = old[cpos:] if cpos != -1 else ''
        leading = old[:len(old) - len(old.lstrip())]
        val_str = f"{leading}{value}"
        # pad so the comment lines up; if value is long, just one space
        pad = max(_VALUE_COLUMN - len(val_str), 1)
        self.lines[idx] = f"{val_str}{' ' * pad}{comment}".rstrip()

    def apply(self, config: dict):
        for key, value in config.items():
            if key not in SCHEMA:
                raise KeyError(
                    f"Unknown field {key!r}. Add it to schema.py if it's a "
                    f"real knob, or fix the typo."
                )
            section, line_no = SCHEMA[key]
            self._set_addr(section, line_no, value)

    def write(self, out_dir: str | Path):
        out_path = Path(out_dir)  / "MCGPU-PET.in"
        # Create the parent directories if they don't exist
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text('\n'.join(self.lines) + '\n')
        print(f"Write {out_path}")

    def from_json(self, json_path: str): 
        # pass in running directory with config.json inside
        json_path = json_path + "config.json"
        json_path = Path(json_path)
        config = json.loads(json_path.read_text())
        self.apply(config)


if __name__ == "__main__":
    OUT_PATH = "runs/run1/test.in"
    CONFIG_PATH = "runs/run1/config.json"
    ifg = InFileGenerator()
    # print(ifg._index)
    # input = {"acquisition_time": 600.0}
    # ifg.apply(input)
    ifg.write(OUT_PATH)
    # ifg.fromJSON(CONFIG_PATH)