# mcgpu_backend/phantom.py
from dataclasses import dataclass, field
import numpy as np

from pathlib import Path
import gzip


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

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(self.material_id.shape)
    
    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        nz, ny, nx = self.material_id.shape
        return (nx, ny, nz)
    
    @property
    def bbox_size_mm(self) -> tuple[float, float, float]: # bounding box
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
    
    @property
    def total_volume_ml(self) -> float:
        nx, ny, nz = self.shape_xyz
        return  (nx * ny * nz) * self.voxel_volume_mL
        
    @property
    def total_mass_gram(self) -> float:
        return float(self.density.sum()) * self.voxel_volume_mL

    # ---- validation ------
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

# ------ Phantom -> phantom.vox

class VoxFileGenerator:
    """Write a Phantom to MCGPU-PET's .vox format.

    The format is penEasy 2008 + a third column for activity (Bq/voxel).
    Header + body, x-fastest order. Blank lines between x/y cycles are
    optional; we omit them (BLANK LINES flag = 0) for a smaller file.
    """
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
    

if __name__ == "__main__":
    import numpy as np

    material = np.ones([9,9,9])
    density = np.ones([9,9,9])
    activity = np.ones([9,9,9])
    voxel = (10,10,10)
    phan = Phantom(material, density, activity, voxel)
    vfg = VoxFileGenerator(phan)