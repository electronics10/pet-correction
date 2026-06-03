# mcgpu_backend/vox_generator.py
from mcgpu_backend.phantom import Phantom
from pathlib import Path


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
