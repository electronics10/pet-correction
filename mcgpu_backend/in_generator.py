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