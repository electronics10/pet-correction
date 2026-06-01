import numpy as np
from mcgpu_backend.sinogram import PETGeometry, Sinogram

geom = PETGeometry(num_radial_bins=147, num_angular_bins=168,
                   num_rings=80, span=1)
sino = Sinogram("data/raw_run_1/sinogram_Trues.raw.gz", geom)

for seg in [0, +1, -1, +2]:
    counts = sino.segments[seg].sum(axis=(1, 2))  # per-plane counts
    nonzero = np.flatnonzero(counts)
    print(f"seg {seg:+d}: {len(nonzero)} non-empty planes / {len(counts)} total")
    print(f"  first 10 non-empty indices: {nonzero[:10].tolist()}")
    print(f"  first 12 counts: {counts[:12].tolist()}")
    print()