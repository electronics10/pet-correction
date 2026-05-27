"""
Minimal OSEM (Ordered-Subsets Expectation Maximization) reconstruction.

This script demonstrates the core OSEM algorithm:
  x^(k+1) = x^(k) * (A^T (y / (A x^(k) + s)) / A^T 1) ^ (1/num_subsets)

where:
  - x = image estimate
  - A = forward projector
  - A^T = back projector
  - y = measured sinogram
  - s = scatter estimate (zero for this demo)
"""

import numpy as np
import parallelproj
from parallelproj.pet_scanners import RegularPolygonPETScannerGeometry
import matplotlib.pyplot as plt

print("\n" + "="*70)
print("MINIMAL OSEM RECONSTRUCTION")
print("="*70 + "\n")

# ── Setup ──────────────────────────────────────────────────────────────────
xp = np  # numpy for CPU (swap to cupy for GPU)

# Scanner geometry (same as test_parallelproj.py)
scanner = RegularPolygonPETScannerGeometry(
    xp=xp, dev="cpu",
    radius=79.0, num_sides=80, num_lor_endpoints_per_side=1,
    lor_spacing=2.0,
    ring_positions=xp.linspace(0, 4.77, 4),
    symmetry_axis=2,
)

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner=scanner, radial_trim=3, max_ring_difference=3,
)

img_shape = (80, 80, 4)
voxel_size = (1.0, 1.0, 1.59)

proj = parallelproj.RegularPolygonPETProjector(
    lor_descriptor=lor_desc, img_shape=img_shape, voxel_size=voxel_size,
)

print(f"Image shape  : {proj.in_shape}")
print(f"Sino  shape  : {proj.out_shape}\n")

# ── Create synthetic data ──────────────────────────────────────────────────
# True image: small cylinder in center
true_image = xp.zeros(img_shape, dtype=xp.float32)
true_image[30:50, 30:50, :] = 1.0  # activity in center

# Forward project to get "measured" sinogram (no noise for this demo)
measured_sino = proj(true_image)
print(f"True image max: {true_image.max():.3f}")
print(f"Measured sino max: {measured_sino.max():.3f}\n")

# ── Normalization (sensitivity image) ──────────────────────────────────────
# sensitivity = A^T 1  (backproject all-ones sinogram)
# Used to correct for uneven detector response
ones_sino = xp.ones_like(measured_sino)
sensitivity = proj.adjoint(ones_sino)
sensitivity_safe = xp.where(sensitivity > 0, sensitivity, 1.0)  # avoid division by zero

print(f"Sensitivity image min/max: {sensitivity.min():.3f} / {sensitivity.max():.3f}\n")

# ── OSEM parameters ────────────────────────────────────────────────────────
num_subsets = 8          # order subsets (typical: 4-16)
num_iterations = 10      # total iterations (typical: 3-5 for clinical)
num_updates_per_iter = num_subsets  # one update per subset

# ── Initialize estimate ────────────────────────────────────────────────────
x_estimate = xp.ones(img_shape, dtype=xp.float32)  # start uniform

print(f"OSEM parameters:")
print(f"  Subsets: {num_subsets}")
print(f"  Iterations: {num_iterations}")
print(f"  Total updates: {num_iterations * num_updates_per_iter}\n")

# ── OSEM loop ──────────────────────────────────────────────────────────────
for iteration in range(num_iterations):
    for subset in range(num_subsets):
        # 1. Forward project current estimate
        estimate_sino = proj(x_estimate)
        
        # 2. Compute ratio: y / (A*x + s)
        # In real data: add randoms/scatter estimate as contamination term
        ratio = measured_sino / (estimate_sino + 1e-6)  # avoid division by zero
        
        # 3. Back project the ratio
        back_projected = proj.adjoint(ratio)
        
        # 4. Multiplicative update (normalized by sensitivity)
        # Exponent 1/num_subsets spreads the update over the subsets
        exponent = 1.0 / num_subsets
        x_estimate *= xp.power(back_projected / sensitivity_safe + 1e-6, exponent)
        
        # Optional: enforce non-negativity (should be implicit but be safe)
        x_estimate = xp.maximum(x_estimate, 0.0)
    
    # Print progress
    estimate_sino_final = proj(x_estimate)
    rmse = xp.sqrt(xp.mean((estimate_sino_final - measured_sino)**2))
    print(f"Iteration {iteration+1:2d}: RMSE = {rmse:.4f}, "
          f"image max = {x_estimate.max():.4f}")

print("\n" + "="*70)
print("RECONSTRUCTION COMPLETE")
print("="*70 + "\n")

# ── Simple comparison ──────────────────────────────────────────────────────
# Compute reconstructed sinogram
final_sino = proj(x_estimate)

# Back project the measured data directly (FBP-like baseline)
fbp_like = proj.adjoint(measured_sino)
fbp_like = fbp_like / sensitivity_safe  # normalize

print(f"Final image max: {x_estimate.max():.4f}")
print(f"True  image max: {true_image.max():.4f}")
print(f"FBP-like max:    {fbp_like.max():.4f}\n")

# Central slice comparison
print("Central z-slice statistics:")
z_slice = img_shape[2] // 2
print(f"  OSEM   : mean={x_estimate[:,:,z_slice].mean():.4f}, "
      f"std={x_estimate[:,:,z_slice].std():.4f}")
print(f"  True   : mean={true_image[:,:,z_slice].mean():.4f}, "
      f"std={true_image[:,:,z_slice].std():.4f}")
print(f"  FBP-like: mean={fbp_like[:,:,z_slice].mean():.4f}, "
      f"std={fbp_like[:,:,z_slice].std():.4f}\n")

print("✓ Done. Next step: integrate your scatter correction!")