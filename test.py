import numpy as np
import parallelproj
from parallelproj.pet_scanners import RegularPolygonPETScannerGeometry
import matplotlib.pyplot as plt

# Scanner geometry
scanner = RegularPolygonPETScannerGeometry(
    xp=np,
    dev="cpu",
    radius=90.5,                   # mm
    num_sides=336,                 # crystals per ring
    num_lor_endpoints_per_side=1,
    lor_spacing=1.0,               # irrelevant when endpoints_per_side=1
    ring_positions=np.linspace(0, 126.56, 80),  # 80 rings
    symmetry_axis=2,
)

# LOR descriptors
lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner=scanner,
    radial_trim=95, # how many radial bins to drop from the edges (where data is unreliable)
)

print("Number of radial bins: ", lor_desc.num_rad, 
      "\nNumber of angular views: ", lor_desc.num_views,
      "\nTotal ring-pair planes: ", lor_desc.num_planes)

# Voxel grids
img_shape = (147, 147, 80) # voxels (x, y, z)
voxel_size = (1, 1, 1) # mm per voxel

# Projector
proj = parallelproj.RegularPolygonPETProjector(
    lor_descriptor=lor_desc,
    img_shape=img_shape,
    voxel_size=voxel_size,
)

# ── Use the projector ──────────────────────────────────────
# Make a fake phantom: cube of activity in the center
phantom = np.zeros(img_shape, dtype=np.float32)
phantom[50:78, 50:78, 30:50] = 1.0

# Forward project: this is your "synthetic measurement"
sinogram = proj(phantom)

# Back project (simplest possible "reconstruction" — not useful, just demo)
back_proj = proj.adjoint(sinogram)
print("Back projection shape:", back_proj.shape)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(phantom[:, :, 40], cmap="gray")
axes[0].set_title("Original phantom (z=40)")
axes[1].imshow(sinogram[:, :, 1000], cmap="gray")
axes[1].set_title("Sinogram (plane 1000)")
axes[2].imshow(back_proj[:, :, 40], cmap="gray")
axes[2].set_title("Back projection (z=40)")
plt.tight_layout()
plt.savefig("parallelproj_intro.png")
print("Saved parallelproj_intro.png")