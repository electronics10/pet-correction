"""
FBP and MLEM reconstruction comparison using parallelproj.

We build the same phantom as your previous test, forward project it,
then reconstruct three ways:
  1. Naive back-projection (the blurry one you saw)
  2. FBP (filtered back-projection)
  3. MLEM (iterative)

All on the same data so you can compare side by side.
"""

import numpy as np
import parallelproj
from parallelproj.pet_scanners import RegularPolygonPETScannerGeometry
import matplotlib.pyplot as plt

# ── Setup: same scanner as before ──────────────────────────────────────────
scanner = RegularPolygonPETScannerGeometry(
    xp=np, dev="cpu",
    radius=90.5,
    num_sides=336,
    num_lor_endpoints_per_side=1,
    lor_spacing=1.0,
    ring_positions=np.linspace(0, 126.56, 80),
    symmetry_axis=2,
)

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner=scanner,
    radial_trim=95,         # matches MCGPU's 147 radial bins
    max_ring_difference=79, # all ring pairs
)

img_shape = (128, 128, 80)
voxel_size = (1.5, 1.5, 1.60)
proj = parallelproj.RegularPolygonPETProjector(
    lor_descriptor=lor_desc, img_shape=img_shape, voxel_size=voxel_size,
)

print(f"Image shape:    {proj.in_shape}")
print(f"Sinogram shape: {proj.out_shape}\n")

# ── Phantom and synthetic measurement ──────────────────────────────────────
phantom = np.zeros(img_shape, dtype=np.float32)
phantom[50:78, 50:78, 30:50] = 1.0

print("Forward projecting phantom...")
sinogram = proj(phantom)
print(f"Done. Sinogram min/max: {sinogram.min():.2f}/{sinogram.max():.2f}\n")

# ── Reconstruction 1: Naive back-projection ────────────────────────────────
print("Reconstruction 1: Naive back-projection (A^T y)...")
recon_bp = proj.adjoint(sinogram)
print(f"Done. Image min/max: {recon_bp.min():.2f}/{recon_bp.max():.2f}\n")

# ── Reconstruction 2: FBP (Filtered Back-Projection) ───────────────────────
# Apply a ramp filter to each (radial, view, plane) slice along the radial axis,
# then back-project. The ramp filter cancels the 1/|ω| blur of A^T A.
print("Reconstruction 2: FBP (ramp-filtered back-projection)...")

def ramp_filter_sinogram(sino: np.ndarray, axis: int = 0) -> np.ndarray:
    """Apply ramp filter |omega| along the radial axis of a sinogram.
    
    sino shape: (radial, view, plane)
    """
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
    return sino_filtered

sino_filtered = ramp_filter_sinogram(sinogram, axis=0)
recon_fbp = proj.adjoint(sino_filtered)
# Optional: clamp negatives (FBP can produce them due to filter)
recon_fbp_clamped = np.maximum(recon_fbp, 0)
print(f"Done. Image min/max: {recon_fbp.min():.2f}/{recon_fbp.max():.2f}\n")

# ── Reconstruction 3: MLEM ─────────────────────────────────────────────────
# Maximum Likelihood Expectation Maximization.
# Update rule:
#   x^(k+1) = x^(k) / (A^T 1) * A^T (y / (A x^(k)))
print("Reconstruction 3: MLEM (20 iterations)...")

# Sensitivity image: back-projection of all-ones sinogram
# This corrects for the fact that different voxels are "seen" by different
# numbers of LORs (edge voxels are seen less than central ones).
ones_sino = np.ones_like(sinogram)
sensitivity = proj.adjoint(ones_sino)
sensitivity_safe = np.where(sensitivity > 0, sensitivity, 1.0)

# Initialize estimate to uniform
x_mlem = np.ones(img_shape, dtype=np.float32)

num_iterations = 20
for k in range(num_iterations):
    # Forward project current estimate
    predicted = proj(x_mlem)
    
    # Ratio of measured to predicted (handle division by zero)
    ratio = sinogram / (predicted + 1e-6)
    
    # Back project the ratio
    correction = proj.adjoint(ratio)
    
    # Multiplicative update, normalized by sensitivity
    x_mlem = x_mlem * correction / sensitivity_safe
    
    # Diagnostic
    if (k + 1) % 5 == 0 or k == 0:
        # Compare predicted sinogram to measured
        rmse = np.sqrt(np.mean((proj(x_mlem) - sinogram)**2))
        print(f"  Iteration {k+1:2d}: RMSE = {rmse:8.2f}, "
              f"max image value = {x_mlem.max():.3f}")

print(f"Done. Final image min/max: {x_mlem.min():.3f}/{x_mlem.max():.3f}\n")

# ── Visualization ──────────────────────────────────────────────────────────
print("Generating comparison figure...")

z_slice = 40  # center slice
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Top row: the reconstructions
images = [
    (phantom[:, :, z_slice], "Original phantom"),
    (recon_bp[:, :, z_slice], "Naive back-projection"),
    (sinogram[:, :, 1000], "Sinogram (plane 1000)"),
]
for ax, (img, title) in zip(axes[0], images):
    im = ax.imshow(img, cmap="gray")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)

# Bottom row: phantom, FBP, MLEM
recons = [
    (phantom[:, :, z_slice], "Original phantom"),
    (recon_fbp[:, :, z_slice], "FBP"),
    (x_mlem[:, :, z_slice], f"MLEM ({num_iterations} iters)"),
]
for ax, (img, title) in zip(axes[1], recons):
    im = ax.imshow(img, cmap="gray")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle(f"Reconstruction comparison (z={z_slice})", fontsize=14)
plt.tight_layout()
plt.savefig("reconstruction_comparison.png", dpi=120)
plt.close()
print("Saved reconstruction_comparison.png")

# Also save a profile through the center to compare resolution
print("\nGenerating intensity profile through phantom center...")
center_y = 64
fig, ax = plt.subplots(figsize=(10, 6))
x_axis = np.arange(128)
ax.plot(x_axis, phantom[center_y, :, z_slice], 'k-', label='Phantom', linewidth=2)
ax.plot(x_axis, recon_bp[center_y, :, z_slice] / recon_bp.max(),
        'b--', label='Back-projection (normalized)', alpha=0.7)
ax.plot(x_axis, recon_fbp[center_y, :, z_slice] / max(recon_fbp.max(), 1e-9),
        'g-', label='FBP (normalized)', alpha=0.7)
ax.plot(x_axis, x_mlem[center_y, :, z_slice] / max(x_mlem.max(), 1e-9),
        'r-', label='MLEM (normalized)', alpha=0.7)
ax.set_xlabel("Voxel index (x)")
ax.set_ylabel("Intensity (normalized)")
ax.set_title(f"Horizontal profile through y={center_y}, z={z_slice}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reconstruction_profile.png", dpi=120)
plt.close()
print("Saved reconstruction_profile.png")