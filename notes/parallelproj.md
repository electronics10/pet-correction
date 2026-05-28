## Installation of parallelproj
[Parallelproj](https://github.com/KUL-recon-lab/parallelproj) is an open-source project for tomographic reconstruction. 

To install the package
```bash
mamba install parallelproj
```
or 
```bash
conda install -c conda-forge parallelproj
```
(Mamba is a reimplementation of the conda package manager in C++, which is more efficient.)

---

## Introduction to parallelproj
Essentially, parallelproj is a library that gives you two operations:
1. Forward projection $A$: takes an image, returns a sinogram
2. Back projection $A^T$: takes a sinogram, return an image.

(For people who are not familiar with linear algebra, transpose (adjoint) is the natural way to send information backward through a linear map — it's what powers least squares, gradient computations, and MLEM, etc., which are just clever combinations of the forward and the adjoint operations.)

To make $A$ (and $A^T$) works, parallelproj needs to know three things:
- Scanner geometry (entries)
- LOR (column space dimension)
- Voxel grid (row space dimension)

In practice, the size of $A$ is too large and won't be stored exactly. Parallelproj implements the operator as a matrix-free linear operator (Joseph's method). 

### Scanner Geometry

```python
import numpy as np
from parallelproj.pet_scanners import RegularPolygonPETScannerGeometry

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

print(scanner)
```

### LOR Descriptor
Handle which pairs of detectors form valid lines of response, and how to bin them (organize them into a sinogram).

```python
import parallelproj

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner=scanner,
    radial_trim=95, # how many radial bins to drop from the edges (where data is unreliable)
)

print("Number of radial bins: ", lor_desc.num_rad, 
      "\nNumber of angular views: ", lor_desc.num_views,
      "\nTotal ring-pair planes: ", lor_desc.num_planes)
```

The backend calculation is as `self._num_rad = (scanner.num_lor_endpoints_per_ring + 1) - 2 * self._radial_trim`. The principle: keep enough bins to cover the transaxial FOV of your reconstruction grid, but no more.

### Voxel grid
```python
img_shape = (147, 147, 80) # voxels (x, y, z)
voxel_size = (1, 1, 1) # mm per voxel
```

### Projector $A$
Finally, we can acquire $A$ (and the corresponding $A^T$).
```python
proj = parallelproj.RegularPolygonPETProjector(
    lor_descriptor=lor_desc,
    img_shape=img_shape,
    voxel_size=voxel_size,
)
```

Now `proj` is a callable object.
```python
# Forward project: image → sinogram
y = proj(x) # y is (radial, angular, plane)

# Back project: sinogram → image  
x_back = proj.adjoint(y) 
```

For example:
```python
# Make a fake phantom: cube of activity in the center
phantom = np.zeros(img_shape, dtype=np.float32)
phantom[50:78, 50:78, 30:50] = 1.0

# Forward project: this is your "synthetic measurement"
sinogram = proj(phantom)

# Back project (simplest possible "reconstruction" — not useful, just demo)
back_proj = proj.adjoint(sinogram)
print("Back projection shape:", back_proj.shape)
```

For visualization:
```python
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
```

---

## Reconstruction
Now we know the basic of prallelproj, we want to implement reconstruction algorithms rather than naive back projection.

### Filtered back projection (FBP)

#### Theory

See [filtered_backprojection.md](filtered_backprojection.md)

#### Practice
```python
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
```

#### Degradation

The continuous FBP formula is mathematically exact, but we can see from the experiment that reconstruction degrade. The reasons are that:

1. **Finitely many angles.** The proof swept $\theta$ continuously over $[0,\pi)$; reality gives $N$ discrete spokes in Fourier space. The spokes diverge at high $|\mathbf k|$, leaving angular gaps where fine detail lives → **streak artifacts**.

2. **Ramp amplifies noise.** The ramp $|\sigma|$ is *necessary* (cancels the $1/|\mathbf k|$ blur) but grows unbounded, so it boosts white noise linearly with frequency. The fix — a *windowed* ramp — suppresses noise but discards real high-frequency signal → **blur**. Sharpness vs. noise is an unavoidable trade-off; they're the same operator.

3. **Finite detector resolution.** Real bin width $\Delta s$ caps frequencies at Nyquist $\sigma_{\max}=\pi/\Delta s$ → resolution limit, **aliasing**, and interpolation error during backprojection.

4. **Imperfect forward model.** The proof assumed $A$ = ideal line integrals; physics (beam hardening, scatter, motion) means $g \approx Af + \text{error}$, and the exact inverse faithfully reconstructs the error too → **cupping, ghosting, bands**.

FBP is the exact inverse of an $A$ that no real scanner implements. The ramp makes that inverse **ill-conditioned at high frequency** (small input error → large output error), so discrete/noisy data reconstructs poorly. This isn't universal — with many angles, low noise, and a good model (e.g. high-dose industrial CT), FBP is excellent. Poorness dominates the **low-dose / few-angle / strong-physics** regime. Because the failures are gaps in an exact inverse, the modern fix is to stop inverting and instead solve $\min_f \|Af-g\|^2 + \lambda R(f)$ (iterative / model-based reconstruction) — which reuses the same $A$, $A^T$ adjoint pair inside the iterations.