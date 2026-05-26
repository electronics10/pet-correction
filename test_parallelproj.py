import numpy as np
import parallelproj
from parallelproj.pet_scanners import RegularPolygonPETScannerGeometry

print("\n" + "="*60)
print("Testing parallelproj installation")
print("="*60 + "\n")

xp = np  # use numpy (CPU); swap to cupy for GPU

# ── Step 1: Scanner geometry ──────────────────────────────────
# Rough preclinical geometry (e.g. Siemens Inveon-like)
num_rings   = 4        # axial rings
ring_spacing = 1.59   # mm between rings
scanner = RegularPolygonPETScannerGeometry(
    xp                       = xp,
    dev                       = "cpu",
    radius                    = 79.0,           # mm, transaxial ring radius
    num_sides                 = 80,             # detector crystals per ring
    num_lor_endpoints_per_side= 1,              # 1 crystal per "side"
    lor_spacing               = 2.0,            # mm between LOR endpoints
    ring_positions            = xp.linspace(0, (num_rings-1)*ring_spacing, num_rings),
    symmetry_axis             = 2,              # z-axis
)
print(f"✓ Scanner geometry: 80 crystals/ring, {num_rings} rings")

# ── Step 2: LOR descriptor (defines sinogram structure) ───────
lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner            = scanner,
    radial_trim        = 3,
    max_ring_difference= num_rings - 1,         # include all cross-plane LORs
)
print(f"✓ LOR descriptor created")
print(f"  - Sinogram shape: {lor_desc.spatial_sinogram_shape}")

# ── Step 3: Projector ─────────────────────────────────────────
img_shape  = (80, 80, num_rings)   # voxels (x, y, z)
voxel_size = (1.0, 1.0, ring_spacing)  # mm per voxel

proj = parallelproj.RegularPolygonPETProjector(
    lor_descriptor = lor_desc,
    img_shape      = img_shape,
    voxel_size     = voxel_size,
)
print(f"✓ Projector created")
print(f"  - Image shape : {proj.in_shape}")
print(f"  - Sino  shape : {proj.out_shape}\n")

# ── Step 4: Forward / back projection ────────────────────────
x  = xp.ones(img_shape, dtype=xp.float32)
y  = proj(x)
bp = proj.adjoint(y)

print(f"✓ Forward projection: {x.shape} → {y.shape}")
print(f"✓ Back   projection: {y.shape} → {bp.shape}")
print(f"\n{'='*60}")
print(f"✓ parallelproj is working correctly!")
print(f"{'='*60}\n")