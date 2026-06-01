import numpy as np
from parallelproj.pet_scanners import RegularPolygonPETScannerGeometry
import parallelproj
import matplotlib.pyplot as plt

scanner = RegularPolygonPETScannerGeometry(
    xp=np,
    dev="cpu",
    radius=90.5, # mm
    num_sides=336, # crystals per ring (sides because polygon)
    num_lor_endpoints_per_side=1,# sub-crystals per "side" (usually 1)
    lor_spacing=1.0, # spacing between sub-crystals (mm); irrelevant when endpoints_per_side=1
    ring_positions=np.linspace(0, 126.56, 80),  # z-coordinate of each ring; 80 rings spanning 126.56 mm
    symmetry_axis=2, # which axis is the ring axis (0=x, 1=y, 2=z)
)

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner=scanner,
    radial_trim=95, # how many radial bins to drop from the edges (where data is unreliable)
)

img_shape = (147, 147, 80) # voxels (x, y, z)
voxel_size = (1, 1, 1) # mm per voxel

proj = parallelproj.RegularPolygonPETProjector(
    lor_descriptor=lor_desc,
    img_shape=img_shape,
    voxel_size=voxel_size,
)

# Make a fake phantom: cube of activity in the center
phantom = np.zeros(img_shape, dtype=np.float32)
phantom[50:78, 50:78, 30:50] = 1.0

sinogram = proj(phantom)


# ------ Reconstruction ------
def ramp_filter_sinogram(sino: np.ndarray, axis: int = 0) -> np.ndarray:
    """Apply ramp filter |k| along the radial axis of a sinogram.
    
    sino shape: (radial, angular, plane)
    """
    print("filtering...")
    n_radial = sino.shape[axis]
    # Build the ramp filter in frequency space
    freqs = np.fft.fftfreq(n_radial)
    ramp = np.abs(freqs).astype(np.float32)
    
    # FFT along radial axis, multiply by ramp, inverse FFT
    sino_fft = np.fft.fft(sino, axis=axis)
    # Reshape ramp to broadcast across (radial, view, plane)
    ramp_shape = [1] * sino.ndim
    ramp_shape[axis] = n_radial
    ramp = ramp.reshape(ramp_shape)
    sino_filtered = np.fft.ifft(sino_fft * ramp, axis=axis).real.astype(np.float32)
    print("filtered")
    return sino_filtered

sino_filtered = ramp_filter_sinogram(sinogram, axis=0)
recon_fbp = proj.adjoint(sino_filtered)
# Optional: clamp negatives (FBP can produce them due to filter)
recon_fbp_clamped = np.maximum(recon_fbp, 0)
print(f"Done. Image min/max: {recon_fbp.min():.2f}/{recon_fbp.max():.2f}\n")

def mlem(proj, y, n_iter=20, x0=None, eps=1e-9):
    """Maximum Likelihood Expectation Maximization.

    proj : parallelproj projector (callable = forward A, .adjoint = A^T)
    y    : measured sinogram, shape (radial, angular, plane)
    """
    print("MLEM...")
    img_shape = proj.in_shape  # voxel grid shape

    # Sensitivity image s = A^T 1 — constant, compute once
    ones_sino = np.ones_like(y)
    sens = proj.adjoint(ones_sino)
    sens = np.maximum(sens, eps)  # guard against divide-by-zero outside FOV

    # Initialize with a positive uniform image (MUST be > 0)
    x = np.ones(img_shape, dtype=np.float32) if x0 is None else x0.copy()

    print(f"Total {n_iter} iterations")
    for k in range(n_iter):
        print(f"iteration {k}...")
        ybar = proj(x)                      # forward project: A x
        ybar = np.maximum(ybar, eps)        # guard divide-by-zero
        ratio = y / ybar                    # measured / predicted, in sinogram space
        correction = proj.adjoint(ratio)    # back project the ratio: A^T (y / A x)
        x = x * correction / sens           # multiplicative update
    print("MLEM done.")
    return x

recon_mlem = mlem(proj, sinogram, n_iter=20)
print(f"MLEM done. min/max: {recon_mlem.min():.2f}/{recon_mlem.max():.2f}")

def make_view_subsets(num_views, n_subsets):
    """Partition angular views into interleaved subsets.

    Subset m gets views m, m+M, m+2M, ...  (spread across all angles,
    so each subset still 'sees' the whole object — subset balance).
    Returns a list of index arrays.
    """
    print(f"{n_subsets} made")
    return [np.arange(m, num_views, n_subsets) for m in range(n_subsets)]


def osem(proj, y, n_subsets=5, n_iter=4, x0=None, eps=1e-9):
    """Ordered Subset Expectation Maximization.

    proj : parallelproj projector; sinogram axis order (radial, angular, plane)
    y    : measured sinogram
    n_iter : full passes through all subsets (each pass = M updates)
    """
    print("OSEM...")
    img_shape = proj.in_shape
    num_views = y.shape[1]  # angular axis
    subsets = make_view_subsets(num_views, n_subsets)

    # Per-subset sensitivity images s_m = A_m^T 1.
    # Realize the subset by masking the sinogram to only its views,
    # then back projecting — equivalent to a view-restricted A_m^T.
    sens = []
    for m, views in enumerate(subsets):
        print(f"calculating adjoint for subset {m}")
        ones_masked = np.zeros_like(y)
        ones_masked[:, views, :] = 1.0
        s_m = proj.adjoint(ones_masked)
        sens.append(np.maximum(s_m, eps))  # guard divide-by-zero

    x = np.ones(img_shape, dtype=np.float32) if x0 is None else x0.copy()

    for k in range(n_iter):
        for m, views in enumerate(subsets):
            print(f"iteration {k} for subset {m}")
            ybar = proj(x)                      # A x  (full forward)
            ybar = np.maximum(ybar, eps)
            ratio = np.zeros_like(y)            # keep only subset m's LORs
            ratio[:, views, :] = y[:, views, :] / ybar[:, views, :]
            correction = proj.adjoint(ratio)    # A_m^T (y_m / A_m x)
            x = x * correction / sens[m]        # multiplicative subset update
    print("OSEM done")
    return x


# 5 subsets x 4 iterations = 20 image updates
recon_osem = osem(proj, sinogram, n_subsets=5, n_iter=4)

# Compare FBP vs MLEM vs OSEM on the same z-slice
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, img, title in zip(
    axes,
    [phantom, recon_fbp_clamped, recon_mlem, recon_osem],
    ["Phantom (z=40)", "FBP", "MLEM 20 iter", "OSEM 5x4"],
):
    ax.imshow(img[:, :, 40], cmap="gray")
    ax.set_title(title)
plt.tight_layout()
plt.savefig("osem_compare.png")
print("Saved osem_compare.png")