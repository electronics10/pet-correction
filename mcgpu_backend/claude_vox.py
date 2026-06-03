"""
mcgpu_backend/vox_generator.py

Phantom representation and .vox serialization for MCGPU-PET.

The Phantom dataclass owns three voxel arrays (material_id, density, activity)
and the voxel size; it is the single source of truth for the simulation source.
The PhantomBuilder exposes painting primitives (background, cylinder, sphere).
The VoxFileGenerator serializes a Phantom into MCGPU-PET's penEasy-2008-style
.vox format (optionally gzipped).

Coordinate convention (matches MCGPU-PET / penEasy):
    - Bounding box sits in the first octant, voxel (1,1,1) cornered at origin.
    - Voxel array shape is (Nz, Ny, Nx) — natural numpy ordering, so arr[z]
      is an axial slice and arr[z, y, x] addresses individual voxels.
    - Center of voxel [k, j, i] is at ((i+0.5)*dx, (j+0.5)*dy, (k+0.5)*dz) mm.
    - .vox serialization uses ravel(order='C'), giving x-fastest order as the
      format requires.

Units:
    - voxel_size_mm: millimeters (matches geometry config)
    - density: g/cm^3 (MCGPU native)
    - activity (stored): Bq per voxel (MCGPU native, .vox third column)
    - activity (user-facing in builder/factories): Bq per mL (physicist-native)
      Conversion: 1 voxel of (dx, dy, dz) mm = dx*dy*dz/1000 mL.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

# ----------------------------------------------------------------------------
# Phantom: the data object
# ----------------------------------------------------------------------------

@dataclass
class Phantom:
    """A voxelized phantom: per-voxel material ID, density, and activity.

    Shape convention: all three arrays have shape (Nz, Ny, Nx). Indexing as
    arr[z, y, x] matches the natural reading of axial slices arr[z].

    Material IDs are 1-indexed (MCGPU does not allow vacuum / material 0).
    """
    material_id: np.ndarray   # (Nz, Ny, Nx), uint8, values in 1..N
    density:     np.ndarray   # (Nz, Ny, Nx), float32, g/cm^3
    activity:    np.ndarray   # (Nz, Ny, Nx), float32, Bq per voxel
    voxel_size_mm: tuple[float, float, float]  # (dx, dy, dz)
    material_names: list[str] = field(default_factory=list)  # bookkeeping

    # ---- derived properties ----

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(self.material_id.shape)

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        nz, ny, nx = self.material_id.shape
        return (nx, ny, nz)

    @property
    def bbox_size_mm(self) -> tuple[float, float, float]:
        nx, ny, nz = self.shape_xyz
        dx, dy, dz = self.voxel_size_mm
        return (nx * dx, ny * dy, nz * dz)

    @property
    def voxel_volume_mL(self) -> float:
        dx, dy, dz = self.voxel_size_mm
        return (dx * dy * dz) / 1000.0  # mm^3 -> mL

    @property
    def total_activity_Bq(self) -> float:
        return float(self.activity.sum())

    # ---- validation ----

    def validate(self) -> None:
        """Raise ValueError if the phantom violates MCGPU-PET invariants."""
        s = self.material_id.shape
        if self.density.shape != s or self.activity.shape != s:
            raise ValueError(
                f"shape mismatch: material_id {s}, density {self.density.shape}, "
                f"activity {self.activity.shape}"
            )
        if self.material_id.min() < 1:
            n_bad = int((self.material_id < 1).sum())
            raise ValueError(
                f"{n_bad} voxels have material_id < 1 (vacuum not allowed in MCGPU)."
            )
        if (self.density < 0).any():
            raise ValueError("Negative densities not allowed.")
        if (self.activity < 0).any():
            raise ValueError("Negative activities not allowed.")
        if any(d <= 0 for d in self.voxel_size_mm):
            raise ValueError(f"voxel_size_mm must be positive: {self.voxel_size_mm}")


# ----------------------------------------------------------------------------
# PhantomBuilder: paint primitives into a Phantom
# ----------------------------------------------------------------------------

class PhantomBuilder:
    """Build a Phantom by painting primitives into an initially-empty volume.

    All physical coordinates are in millimeters, measured from the bounding
    box origin (first octant); the bbox spans [0, Nx*dx] x [0, Ny*dy] x [0, Nz*dz].
    Use `bbox_center_mm` for the natural FOV center.

    Activity is specified by users in Bq/mL (concentration); internally
    converted to Bq/voxel when painting.

    Paint semantics: later calls overwrite earlier ones at any overlapping
    voxel. The order of calls therefore matters — paint the background first,
    then larger compartments, then smaller features.
    """

    def __init__(
        self,
        shape_xyz: tuple[int, int, int],
        voxel_size_mm: tuple[float, float, float],
        material_names: list[str] | None = None,
    ):
        nx, ny, nz = shape_xyz
        if any(n <= 0 for n in (nx, ny, nz)):
            raise ValueError(f"shape must be positive: {shape_xyz}")
        if any(d <= 0 for d in voxel_size_mm):
            raise ValueError(f"voxel_size_mm must be positive: {voxel_size_mm}")

        self.shape_xyz = (nx, ny, nz)
        self.voxel_size_mm = tuple(voxel_size_mm)
        self.material_names = list(material_names) if material_names else []

        # Arrays stored as (Nz, Ny, Nx)
        self.material_id = np.zeros((nz, ny, nx), dtype=np.uint8)
        self.density     = np.zeros((nz, ny, nx), dtype=np.float32)
        self.activity    = np.zeros((nz, ny, nx), dtype=np.float32)

        # Precompute voxel center coordinates (1D each, broadcast on demand)
        dx, dy, dz = self.voxel_size_mm
        self._x = (np.arange(nx, dtype=np.float64) + 0.5) * dx
        self._y = (np.arange(ny, dtype=np.float64) + 0.5) * dy
        self._z = (np.arange(nz, dtype=np.float64) + 0.5) * dz

    # ---- coordinate helpers ----

    @property
    def bbox_size_mm(self) -> tuple[float, float, float]:
        nx, ny, nz = self.shape_xyz
        dx, dy, dz = self.voxel_size_mm
        return (nx * dx, ny * dy, nz * dz)

    @property
    def bbox_center_mm(self) -> tuple[float, float, float]:
        bx, by, bz = self.bbox_size_mm
        return (bx / 2.0, by / 2.0, bz / 2.0)

    # ---- internal painting helper ----

    def _paint_mask(
        self,
        mask: np.ndarray,
        material_id: int,
        density: float,
        activity_Bq_per_mL: float,
    ) -> int:
        """Paint a boolean (Nz,Ny,Nx) mask with the given properties.

        Returns the number of voxels painted.
        """
        if material_id < 1 or material_id > 255:
            raise ValueError(f"material_id must be in 1..255, got {material_id}")
        if density < 0:
            raise ValueError(f"density must be >= 0, got {density}")
        if activity_Bq_per_mL < 0:
            raise ValueError(f"activity_Bq_per_mL must be >= 0, got {activity_Bq_per_mL}")

        activity_per_voxel = activity_Bq_per_mL * self._voxel_volume_mL()
        self.material_id[mask] = material_id
        self.density[mask] = density
        self.activity[mask] = activity_per_voxel
        return int(mask.sum())

    def _voxel_volume_mL(self) -> float:
        dx, dy, dz = self.voxel_size_mm
        return (dx * dy * dz) / 1000.0

    # ---- primitives ----

    def fill_background(
        self,
        material_id: int,
        density: float,
        activity_Bq_per_mL: float = 0.0,
    ) -> int:
        """Fill the entire volume. Typically called first (e.g., with air)."""
        mask = np.ones_like(self.material_id, dtype=bool)
        return self._paint_mask(mask, material_id, density, activity_Bq_per_mL)

    def add_cylinder(
        self,
        center_mm: tuple[float, float, float],
        radius_mm: float,
        height_mm: float,
        axis: Literal["x", "y", "z"],
        material_id: int,
        density: float,
        activity_Bq_per_mL: float = 0.0,
    ) -> int:
        """Paint an axis-aligned cylinder.

        center_mm  : (x, y, z) center of the cylinder in bbox coords (mm).
        radius_mm  : cylinder cross-section radius (mm).
        height_mm  : full length along `axis` (mm).
        axis       : 'x', 'y', or 'z' — the cylinder's axis.
        """
        if radius_mm <= 0 or height_mm <= 0:
            raise ValueError(f"radius_mm and height_mm must be > 0")

        cx, cy, cz = center_mm
        # Build broadcasted coordinate arrays only over the bounding box
        # of the cylinder to keep this fast for small features in big volumes.
        half = height_mm / 2.0

        if axis == "z":
            # cross section in (x,y), height along z
            z_lo, z_hi = cz - half, cz + half
            x_lo, x_hi = cx - radius_mm, cx + radius_mm
            y_lo, y_hi = cy - radius_mm, cy + radius_mm
            k_slice, j_slice, i_slice = self._bbox_slices(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
            X = self._x[i_slice][None, None, :]
            Y = self._y[j_slice][None, :, None]
            Z = self._z[k_slice][:, None, None]
            cross = (X - cx) ** 2 + (Y - cy) ** 2 <= radius_mm ** 2
            within_height = np.abs(Z - cz) <= half
        elif axis == "x":
            x_lo, x_hi = cx - half, cx + half
            y_lo, y_hi = cy - radius_mm, cy + radius_mm
            z_lo, z_hi = cz - radius_mm, cz + radius_mm
            k_slice, j_slice, i_slice = self._bbox_slices(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
            X = self._x[i_slice][None, None, :]
            Y = self._y[j_slice][None, :, None]
            Z = self._z[k_slice][:, None, None]
            cross = (Y - cy) ** 2 + (Z - cz) ** 2 <= radius_mm ** 2
            within_height = np.abs(X - cx) <= half
        elif axis == "y":
            x_lo, x_hi = cx - radius_mm, cx + radius_mm
            y_lo, y_hi = cy - half, cy + half
            z_lo, z_hi = cz - radius_mm, cz + radius_mm
            k_slice, j_slice, i_slice = self._bbox_slices(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
            X = self._x[i_slice][None, None, :]
            Y = self._y[j_slice][None, :, None]
            Z = self._z[k_slice][:, None, None]
            cross = (X - cx) ** 2 + (Z - cz) ** 2 <= radius_mm ** 2
            within_height = np.abs(Y - cy) <= half
        else:
            raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")

        local_mask = cross & within_height
        if not local_mask.any():
            return 0

        # Build full-shape mask only at painting time
        full_mask = np.zeros_like(self.material_id, dtype=bool)
        full_mask[k_slice, j_slice, i_slice] = local_mask
        return self._paint_mask(full_mask, material_id, density, activity_Bq_per_mL)

    def add_sphere(
        self,
        center_mm: tuple[float, float, float],
        radius_mm: float,
        material_id: int,
        density: float,
        activity_Bq_per_mL: float = 0.0,
    ) -> int:
        """Paint a sphere centered at center_mm with the given radius."""
        if radius_mm <= 0:
            raise ValueError("radius_mm must be > 0")

        cx, cy, cz = center_mm
        k_slice, j_slice, i_slice = self._bbox_slices(
            cx - radius_mm, cx + radius_mm,
            cy - radius_mm, cy + radius_mm,
            cz - radius_mm, cz + radius_mm,
        )
        X = self._x[i_slice][None, None, :]
        Y = self._y[j_slice][None, :, None]
        Z = self._z[k_slice][:, None, None]
        local_mask = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= radius_mm ** 2

        if not local_mask.any():
            return 0
        full_mask = np.zeros_like(self.material_id, dtype=bool)
        full_mask[k_slice, j_slice, i_slice] = local_mask
        return self._paint_mask(full_mask, material_id, density, activity_Bq_per_mL)

    def _bbox_slices(
        self,
        x_lo: float, x_hi: float,
        y_lo: float, y_hi: float,
        z_lo: float, z_hi: float,
    ) -> tuple[slice, slice, slice]:
        """Voxel-index slices covering the given mm bounding box, clamped to volume.

        Returns slices in (k, j, i) = (z, y, x) order.
        """
        nx, ny, nz = self.shape_xyz
        dx, dy, dz = self.voxel_size_mm
        # voxel center is at (i+0.5)*dx, so a voxel intersects [x_lo, x_hi]
        # when (i+0.5)*dx >= x_lo and (i+0.5)*dx <= x_hi  -> we expand by half a voxel
        # but use a conservative ceil/floor pattern to be safe.
        i_min = max(0, int(np.floor(x_lo / dx)))
        i_max = min(nx, int(np.ceil(x_hi / dx)) + 1)
        j_min = max(0, int(np.floor(y_lo / dy)))
        j_max = min(ny, int(np.ceil(y_hi / dy)) + 1)
        k_min = max(0, int(np.floor(z_lo / dz)))
        k_max = min(nz, int(np.ceil(z_hi / dz)) + 1)
        return slice(k_min, k_max), slice(j_min, j_max), slice(i_min, i_max)

    # ---- build ----

    def build(self) -> Phantom:
        """Finalize and return the Phantom. Validates before returning."""
        ph = Phantom(
            material_id=self.material_id.copy(),
            density=self.density.copy(),
            activity=self.activity.copy(),
            voxel_size_mm=self.voxel_size_mm,
            material_names=list(self.material_names),
        )
        ph.validate()
        return ph


# ----------------------------------------------------------------------------
# VoxFileGenerator: serialize Phantom -> .vox
# ----------------------------------------------------------------------------

class VoxFileGenerator:
    """Write a Phantom to MCGPU-PET's .vox format.

    The format is penEasy 2008 + a third column for activity (Bq/voxel).
    Header + body, x-fastest order. Blank lines between x/y cycles are
    optional; we omit them (BLANK LINES flag = 0) for a smaller file.
    """

    def __init__(self, phantom: Phantom):
        phantom.validate()
        self.phantom = phantom

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
        else:
            out_path.write_text(payload)
        return out_path

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