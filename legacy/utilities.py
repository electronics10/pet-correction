"""
image_Trues.raw.gz / image_Scatter.raw.gz are emission images, not the sinograms. 
They're a 3D voxel array of int32, length NVOX_SIM = num_voxels.x * num_voxels.y * num_voxels.z, 
written in raw row-major order (x fastest, then y, then z). 
The 729 = 9x9x9 confirms your phantom is voxelized at 9³, 
and int32 is correct (gzwrite(file3, Imagen_T, NVOX_SIM*sizeof(int))).

Each voxel counts coincidence events that originated (were emitted) from that voxel 
and passed the energy window, split by coincidence type:

image_Trues: the emitting voxel got +1 every time it produced a coincidence where neither photon scattered (true coincidence).
image_Scatter: +1 when the coincidence was detected but at least one photon Compton/Rayleigh scattered.

So these are not reconstructed images and not the activity map you fed in directly, 
they're a forward-tally of how many accepted coincidences each source voxel actually contributed. 

image_Trues + image_Scatter should roughly reproduce your input activity phantom shape (Poisson-noisy, sensitivity-weighted). 
A quick way to confirm the simulation emitted from the right places.

Scatter fraction by location. image_Scatter / (image_Trues + image_Scatter) per voxel 
tells you where in the phantom scatter is being generated,
occasionally useful for debugging geometry or for understanding why a region is hard.

Ground-truth-ish reference if you ever wanted to evaluate a reconstruction.
"""

from pathlib import Path
import gzip

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon

# --- filtered backprojection ---

def fbp_stack(sinogram_stack: np.ndarray, output_size: int | None = None,
              filter_name: str = "ramp") -> np.ndarray:
    """FBP a 2D sinogram stack shaped (n_planes, nrad, nang).

    Each plane is passed to skimage.transform.iradon independently.
    Angles are uniform over [0, 180) since a half-turn covers all PET views.
    Returns float32 (n_planes, output_size, output_size).
    """
    n_planes, nrad, nang = sinogram_stack.shape
    if output_size is None:
        output_size = nrad
    angles = np.linspace(0, 180, nang, endpoint=False)
    print("Filtered Backprojection...")
    slices = [
        iradon(sinogram_stack[i], theta=angles, filter_name=filter_name,
               output_size=output_size, circle=True)
        for i in range(n_planes)
    ]
    return np.stack(slices, axis=0).astype(np.float32)


# ------------------------------------------------------------------ display

def show_2dimage(
        image: np.ndarray,
        fig_title: str,
        save_path: str | Path) -> None:

    plt.imshow(image, cmap="gray", origin="lower")
    plt.title(fig_title)
    plt.colorbar()
    save_path = Path(save_path)
    plt.savefig(save_path)
    plt.close()
    print(f"figure saved as {save_path}")


def show_3dimage(
        image: np.ndarray,
        fig_title: str,
        save_path: str | Path,
        position_factor: tuple[float, float, float] = (0.5, 0.5, 0.5),
        position: tuple[int, int, int] | None = None) -> None:
    """
    Plots axial, coronal, and sagittal cuts of a 3D volume.

    position_factor: (z, y, x) each in [0, 1]. Used if position is None.
    position:        (z, y, x) absolute voxel indices. Overrides position_factor.
    """
    Z, Y, X = image.shape

    if position is not None:
        iz, iy, ix = position
        if not (0 <= iz < Z and 0 <= iy < Y and 0 <= ix < X):
            raise ValueError("position out of bounds")
    else:
        fz, fy, fx = position_factor
        if not all(0.0 <= f <= 1.0 for f in (fz, fy, fx)):
            raise ValueError("position_factor values must be in [0, 1]")
        iz = int(fz * (Z - 1))
        iy = int(fy * (Y - 1))
        ix = int(fx * (X - 1))

    cuts = [
        (image[iz, :, :], f"Axial  z={iz}"),
        (image[:, iy, :], f"Coronal  y={iy}"),
        (image[:, :, ix], f"Sagittal  x={ix}"),
    ]

    vmin, vmax = image.min(), image.max()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (img2d, title) in zip(axes, cuts):
        im = ax.imshow(img2d, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    fig.suptitle(fig_title)
    plt.tight_layout()
    save_path = Path(save_path)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"figure saved as {save_path}")


# ------------------------------------------------------------------ emission images

def load_emission_image(image_path: str | Path, vox: tuple[int, int, int]) -> np.ndarray:
    """
    Read a gzipped int32 voxel array written by MCGPU-PET (x fastest).
    vox: (NX, NY, NZ). Returns array shaped (NZ, NY, NX).
    """
    with gzip.open(Path(image_path), "rb") as f:
        flat = np.frombuffer(f.read(), dtype=np.int32)
    return flat.reshape(vox[2], vox[1], vox[0])


def load_emission_images(
        directory: str | Path,
        vox: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (trues, scatter), each shaped (NZ, NY, NX)."""
    d = Path(directory)
    return (
        load_emission_image(d / "image_Trues.raw.gz",   vox),
        load_emission_image(d / "image_Scatter.raw.gz", vox),
    )


def show_emission_images(
        directory: str | Path,
        vox: tuple[int, int, int],
        position_factor: tuple[float, float, float] = (0.5, 0.4, 0.7),
        position: tuple[int, int, int] | None = None) -> None:
    """Load and save orthogonal-cut figures for trues and scatter emission images."""
    d = Path(directory)
    trues, scatter = load_emission_images(d, vox)
    show_3dimage(trues,   "Emission Image (Trues)",   d / "emission_image_trues.png",   position_factor, position)
    show_3dimage(scatter, "Emission Image (Scatter)", d / "emission_image_scatter.png", position_factor, position)


if __name__ == "__main__":
    NX = 9
    NY = 9
    NZ = 9 
    VOX = (NX, NY, NZ)
    run_dir = "runs/run0/"
    show_emission_images(run_dir, VOX)

