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
    
    @property
    def total_volume_ml(self) -> float:
        nx, ny, nz = self.shape_xyz
        return  (nx * ny * nz) * self.voxel_volume_mL
        
    @property
    def total_mass_gram(self) -> float:
        density_avg = np.mean(self.density)
        return  density_avg * self.total_volume_ml

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

    


if __name__ == "__main__":
    m = np.ones([5,5,5])
    d = np.ones([5,5,5])
    a = np.ones([5,5,5])
    v = (2, 1, 1)
    phh = Phantom(m, d, a, v)
    print(m)