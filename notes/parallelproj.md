## Installation of parallelproj

[Parallelproj](https://github.com/KUL-recon-lab/parallelproj) is an open-source project for tomographic reconstruction. It is distributed only via conda-forge (not PyPI), because it ships compiled C and CUDA libraries that don't fit PyPI's wheel model well.

Two install routes, depending on the personal preferences:

**Pixi (project-local, recommended for new projects):**
```bash
# pixi add parallelproj                   # CPU build
pixi add "python=3.12" parallelproj cupy "cuda-version=12.*"   # GPU build
```
For GPU, also add `cuda = "12"` under `[system-requirements]` in `pixi.toml` so the solver picks the CUDA variant.

**Mamba/conda (global or named env):**
```bash
mamba install -c conda-forge parallelproj
```
Mamba is a faster C++ reimplementation of conda; they're interchangeable for install commands.

**uv (does not work):** parallelproj is not on PyPI, so uv cannot install it.

## GPU notes
- The solver picks the CPU build by default. To get GPU support you must signal it explicitly (via `cupy` + `cuda-version`, or `system-requirements.cuda`).
- For cupy, device is an **integer index** (`dev=0`), not the string `"cuda"`.
- Use `array_api_compat.cupy` rather than raw `cupy` as the array namespace passed to parallelproj's `xp=` argument — it adds array-API conformance that parallelproj relies on (e.g. `device=` on `xp.eye`).


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

**Fourier Slice Theorem**

To my understanding, given a 2D image $f(x,y)=f(\mathbf{x})$, the Radon transform $f(x,y) \mapsto g(\theta, s)$ is basically the collection of a wheel of 1D projection $p_\theta(s)=\int_{l_{(\theta, s)}}f\,dl$ where $p_\theta(s)=g(\theta, s)$ and $l$ is a line passing through a point of radial distance $s$ and with direction parallel to $\hat{\theta}$.

  

We denote the Fourier transform of $f(\mathbf{x})$ as $\hat{f}(\mathbf{k})$ and of $p_\theta(s)$ as $\hat{p}_\theta(\sigma)$. The Fouriere slice theorem states that the 1D Fourier transform of a projection equals a central slice of the 2D Fourier transform of the image take along the same direction. Precisely, $$\hat{p}_\theta(\sigma) = \hat{f}(\sigma\cos\theta,\ \sigma\sin\theta) = \hat{f}(\sigma\,\hat{\mathbf{n}}_\theta)$$where $\hat{\mathbf{n}}_\theta = (\cos\theta,\sin\theta)$ is the unit vector along the projection direction, and $\sigma$ is the 1D frequency conjugate to $s$.

Proof:
1. $p_\theta(s)=\int_{l_{(\theta, s)}}f\,dl = \iint f(\mathbf{x})\delta(\mathbf{x}\cdot\hat{\mathbf{n}}_\theta - s)\,d\mathbf{x}$

2. Take the 1D FT in $s$: $$\hat{p}_\theta(\sigma)=\int p_\theta(s)\,e^{-i\sigma s}\,ds = \iint f(\mathbf{x})\left[\int \delta(\mathbf{x}\cdot\hat{\mathbf{n}}_\theta - s)\,e^{-i\sigma s}\,ds\right]d\mathbf{x}.$$
3. The inner integral collapses the delta: $\int\delta(\cdots - s)e^{-i\sigma s}ds = e^{-i\sigma(\mathbf{x}\cdot\hat{\mathbf{n}}_\theta)}$.

4. So $\hat{p}_\theta(\sigma)=\iint f(\mathbf{x})\,e^{-i(\sigma\hat{\mathbf{n}}_\theta)\cdot\mathbf{x}}\,d\mathbf{x} = \hat{f}(\sigma\hat{\mathbf{n}}_\theta)$



**Backprojection**

If we denote the Radon transform operator as $A$, it is clear that $$Af=\iint f(\mathbf{x})\delta(\mathbf{x}\cdot\hat{\mathbf{n}}_\theta - s)\,d\mathbf{x}=g_0(\theta, s).$$ We can see that $\forall g$,  $$\langle Af, g\rangle = \int_0^{\pi}\int_{\mathbb R} \left[\iint f(\mathbf x)\,\delta(\mathbf x\cdot\hat{\mathbf n}_\theta - s)\,d\mathbf x\right] g(\theta,s)\, ds\, d\theta = \iint f(\mathbf x)\left[\int_0^{\pi} g(\theta,\ \mathbf x\cdot\hat{\mathbf n}_\theta)\, d\theta\right] d\mathbf x.$$
Observe that $\iint f(\mathbf x)\left[\int_0^{\pi} g(\theta,\ \mathbf x\cdot\hat{\mathbf n}_\theta)\, d\theta\right] d\mathbf x=\langle f, A^Tg\rangle$. Therefore, the adjoint $A^T$ satisfies:
$$\boxed{(A^T g)(\mathbf x) = \int_0^\pi g\big(\theta,\ \mathbf x\cdot\hat{\mathbf n}_\theta\big)\, d\theta}$$

Geometrically, to find the value of $A^T g$ at a point $\mathbf x$, for each angle $\theta$ pick out the projection value of the line that passes through $\mathbf x$, then sum (integrate) over all angles. That is exactly backprojection: smear each projection back along its lines and add up the angles.

**Filtered backprojection**

Leaving the proof aside, we get a clean result: $$(A^T A f)(\mathbf{x}) = (f * h)(\mathbf{x}), \qquad \hat{h}(\mathbf{k}) = \frac{2\pi}{|\mathbf{k}|}$$
$A^T A$ is a convolution whose Fourier multiplier is $1/|\mathbf{k}|$ (up to the $2\pi$ from my unnormalized FT convention). It is a radially-symmetric low-pass-ish smear: it over-weights low frequencies by exactly $1/|\mathbf{k}|$. Reconstructing the 2D inverse FT requires the area element $d\mathbf{k} = |\sigma|,d\sigma,d\theta$, but backprojection supplies only $d\sigma,d\theta$. The missing Jacobian factor $|\sigma| = |\mathbf{k}|$ is exactly what's _under_-counted, so $A^TA$ acquires the multiplier $1/|\mathbf{k}|$.

To invert, premultiply by the ramp filter $|\mathbf{k}|$ in Fourier space before backprojection. That "filter then backproject" is filtered backprojection

Intuitively, backprojection smears each line's value across the whole plane, dumping energy near the origin in Fourier space (every slice passes through $\mathbf{k}=0$). Low frequencies get counted by every angle; high frequencies by few. The ramp reweights to undo this $1/|\mathbf{k}|$ angular crowding.

#### Practice
```python
def ramp_filter_sinogram(sino: np.ndarray, axis: int = 0) -> np.ndarray:
    """Apply ramp filter |k| along the radial axis of a sinogram.
    
    sino shape: (radial, angular, plane)
    """
    print("ramp filtering...")
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
```

#### Degradation

The continuous FBP formula is mathematically exact, but we can see from the experiment that reconstruction degrade. The reasons are that:

1. **Finitely many angles.** The proof swept $\theta$ continuously over $[0,\pi)$; reality gives $N$ discrete spokes in Fourier space. The spokes diverge at high $|\mathbf k|$, leaving angular gaps where fine detail lives → **streak artifacts**.

2. **Ramp amplifies noise.** The ramp $|\sigma|$ is *necessary* (cancels the $1/|\mathbf k|$ blur) but grows unbounded, so it boosts white noise linearly with frequency. The fix — a *windowed* ramp — suppresses noise but discards real high-frequency signal → **blur**. Sharpness vs. noise is an unavoidable trade-off; they're the same operator.

3. **Finite detector resolution.** Real bin width $\Delta s$ caps frequencies at Nyquist $\sigma_{\max}=\pi/\Delta s$ → resolution limit, **aliasing**, and interpolation error during backprojection.

4. **Imperfect forward model.** The proof assumed $A$ = ideal line integrals; physics (beam hardening, scatter, motion) means $g \approx Af + \text{error}$, and the exact inverse faithfully reconstructs the error too → **cupping, ghosting, bands**.

FBP is the exact inverse of an $A$ that no real scanner implements. The ramp makes that inverse **ill-conditioned at high frequency** (small input error → large output error), so discrete/noisy data reconstructs poorly. This isn't universal — with many angles, low noise, and a good model (e.g. high-dose industrial CT), FBP is excellent. Poorness dominates the **low-dose / few-angle / strong-physics** regime. Because the failures are gaps in an exact inverse, the modern fix is to stop inverting and instead solve $\min_f \|Af-g\|^2 + \lambda R(f)$ (iterative / model-based reconstruction) — which reuses the same $A$, $A^T$ adjoint pair inside the iterations.


### Maximum Likelihood Expectation Maximization (MLEM)

#### Theory

**Why a statistical model at all.**

FBP failed in the low-dose regime because it faithfully inverts noise. The fix flagged at the end of the last section: stop inverting, start *fitting*. To fit, we need a noise model — a statement of what "the data is random" actually means.

**The Poisson model (the modeling hypothesis).**

In PET, the measurement $y_i$ in LOR $i$ is a *count* of detected coincidence events. Counts of independent rare events are Poisson. So we posit:
$$y_i \sim \text{Poisson}(\bar{y}_i), \qquad \bar{y}_i = (Ax)_i = \sum_j A_{ij}\, x_j$$
where $x_j \ge 0$ is the activity in voxel $j$, $A_{ij}$ is the probability that an emission in voxel $j$ is detected in LOR $i$ (the same forward operator as before), and $\bar{y}_i$ is the *expected* count. The LORs are assumed independent.

Logical status: this is a **modeling assumption**, not a theorem. Everything below is exact *given* this model.

**The likelihood and the objective.**

The Poisson PMF is $P(y_i \mid \bar y_i) = \bar y_i^{\,y_i} e^{-\bar y_i}/y_i!$. Independence makes the joint likelihood a product; take the log and drop the $x$-independent $\log y_i!$ term:
$$L(x) = \sum_i \big[\, y_i \log (Ax)_i - (Ax)_i \,\big]$$
We want $\boxed{\hat x = \arg\max_{x \ge 0} L(x)}$ — the **maximum likelihood** estimate. Note $L$ is concave (each term is concave in $(Ax)_i$, composed with linear $A$), so a maximizer exists; the nonnegativity constraint $x\ge 0$ is what makes it interesting.

**Why not just take the gradient.**

$\partial L/\partial x_j = \sum_i A_{ij}\big(\frac{y_i}{(Ax)_i} - 1\big)$. Setting it to zero is coupled across all voxels through $(Ax)_i$, and the constraint $x \ge 0$ blocks a clean closed form. EM gives an iteration that respects $x\ge 0$ automatically and never decreases $L$.

**EM (the algorithm, and its guarantee).**

Expectation–Maximization is a general recipe for ML estimation with *latent variables*. The latent quantity here: $z_{ij}$ = number of events emitted in voxel $j$ *and* detected in LOR $i$. We never observe $z_{ij}$; we only observe the row sums $y_i = \sum_j z_{ij}$. If we *could* see $z_{ij}$, the ML estimate would be trivial. EM alternates:

- **E-step**: given the current guess $x^{(k)}$, compute the expected latent counts. For Poisson, $\mathbb E[z_{ij} \mid y_i, x^{(k)}] = y_i \cdot \dfrac{A_{ij} x_j^{(k)}}{(Ax^{(k)})_i}$ — i.e. split the observed $y_i$ among voxels in proportion to their current predicted contribution.
- **M-step**: maximize the expected complete-data log-likelihood. This decouples per voxel and gives a closed form.

Carrying out the algebra collapses both steps into one update:
$$\boxed{\; x_j^{(k+1)} = \frac{x_j^{(k)}}{\sum_i A_{ij}} \sum_i A_{ij}\, \frac{y_i}{(Ax^{(k)})_i} \;}$$

The EM guarantee (a **theorem**, given the model): $L(x^{(k+1)}) \ge L(x^{(k)})$ at every step — monotone ascent — and the iterates converge to a maximizer of $L$.

**Reading the update as operators (the part that connects to parallelproj).**

Define everything you need in terms of $A$ and $A^T$:

- $(Ax^{(k)})_i$ — **forward project** the current image → predicted sinogram.
- $r_i = y_i / (Ax^{(k)})_i$ — elementwise **ratio** of measured to predicted, in sinogram space. (Where you over-predict, $r<1$; under-predict, $r>1$.)
- $(A^T r)_j = \sum_i A_{ij} r_i$ — **back project** the ratio → correction image.
- $s_j = \sum_i A_{ij} = (A^T \mathbf 1)_j$ — **sensitivity image**, the back projection of an all-ones sinogram. Constant across iterations, so compute once.

Then the whole iteration is, in words: *forward project, take the ratio against the data, back project that ratio, and use it to multiplicatively rescale the current image, normalized by sensitivity.*
$$x^{(k+1)} = \frac{x^{(k)}}{A^T \mathbf 1} \odot A^T\!\left(\frac{y}{A x^{(k)}}\right)$$
where $\odot$ and the division are elementwise.

**Three structural facts worth flagging.**

1. **Multiplicative, hence nonnegativity-preserving.** Start with $x^{(0)} > 0$; since $A_{ij}\ge 0$ and $y_i \ge 0$, every factor is $\ge 0$, so $x^{(k)} \ge 0$ for free. Contrast with FBP, which happily returns negatives (you clamped them). This is the constraint $x\ge0$ being *built into the geometry of the update* rather than imposed afterward.

2. **Self-consistency / fixed point.** At convergence the multiplicative factor is $1$, i.e. $A^T(y/A\hat x) = A^T\mathbf 1$. Compare the gradient condition $A^T(y/A x - \mathbf 1)=0$ — they're the same equation. The fixed point of MLEM *is* the stationary point of $L$. (Slogan: "MLEM is gradient ascent wearing multiplicative clothing." Stress-test: it's a *preconditioned, multiplicative* ascent, not literally $x + \eta\nabla L$ — the step size is effectively $x_j^{(k)}/s_j$, so the analogy holds in direction and fixed point, not in the literal increment.)

3. **The cross-field analogy.** This multiplicative-update + nonnegativity structure is the *same* as the multiplicative update rules in **nonnegative matrix factorization (NMF)**, and the ratio-backprojection form recurs in the **Richardson–Lucy deconvolution** used in optics/astronomy. RL is literally MLEM with $A$ = a blur kernel instead of a projection. Hypothesis under which they coincide: Poisson data + nonnegative linear forward model. Change the noise model (e.g. Gaussian) and you get a different, *additive* algorithm (then it really is least squares / gradient steps).

**The catch (sets up the next section).**

MLEM maximizes $L$ — but the ML solution of a noisy inverse problem is itself noisy. Unregularized MLEM, run to convergence, reconstructs noise just like FBP did; the early iterations look good and late iterations degrade. People exploit this by **stopping early** (a crude implicit regularizer) or by adding a penalty $R(x)$ — which lands you back at the $\min_x \|Ax-y\|^2_{\text{(Poisson sense)}} + \lambda R(x)$ framing, now solved with MLEM-style updates (MAP-EM / OSEM).

#### Practice

The update is four operator calls per iteration. The sensitivity image is computed once outside the loop.

```python
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
    print("MLEM done.")          # multiplicative update
    return x

recon_mlem = mlem(proj, sinogram, n_iter=20)
```

```python
import matplotlib.pyplot as plt

# Compare FBP vs MLEM on the same z-slice
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(phantom[:, :, 40], cmap="gray")
axes[0].set_title("Original phantom (z=40)")
axes[1].imshow(recon_fbp_clamped[:, :, 40], cmap="gray")
axes[1].set_title("FBP (z=40)")
axes[2].imshow(recon_mlem[:, :, 40], cmap="gray")
axes[2].set_title("MLEM, 20 iter (z=40)")
plt.tight_layout()
plt.savefig("mlem_vs_fbp.png")
print("Saved mlem_vs_fbp.png")
```

Two practical notes that follow directly from the theory: the `eps` guards exist because the update divides by $(Ax)_i$ and by $s_j$, both of which vanish outside the field of view; and watch the iteration count — because unregularized MLEM eventually reconstructs noise, the "best" image is often at some intermediate `n_iter`, not the final one. Try plotting the result at 10, 30, and 100 iterations to *see* the noise grow.

Continuing in your voice.

---

### Ordered Subset Expectation Maximization (OSEM)

#### Theory

**The problem OSEM solves.**

MLEM works but is slow. Each iteration touches *every* LOR once (one full forward + one full back projection) to make *one* update to the image. In PET the sinogram has millions of LORs, and you need tens of iterations → expensive. The question: can we get a useful image update without using all the data every time?

**The core idea (the modeling move).**

Partition the LORs into $M$ disjoint **subsets** $S_1, \dots, S_M$. Instead of one update per full sweep, do $M$ updates per sweep — one per subset — each using only the LORs in that subset. One pass through all $M$ subsets is called a **sub-iteration** or one "OSEM iteration," and it costs the same as a *single* MLEM iteration but produces $M$ image updates.

Define the per-subset operators. For subset $S_m$, let $A_m$ be the forward projector restricted to those LORs (i.e. only the rows $i \in S_m$ of $A$). Then within subset $m$ the update is exactly the MLEM update but summed only over $i \in S_m$:
$$\boxed{\; x_j \leftarrow \frac{x_j}{\sum_{i \in S_m} A_{ij}} \sum_{i \in S_m} A_{ij}\, \frac{y_i}{(A_m x)_i} \;}$$
In operator form, identical to MLEM but with $A_m, A_m^T$ in place of $A, A^T$:
$$x \leftarrow \frac{x}{A_m^T \mathbf 1} \odot A_m^T\!\left(\frac{y_m}{A_m x}\right)$$
where $y_m$ is the data on subset $m$. Crucially the **sensitivity image is now per-subset**: $s_m = A_m^T \mathbf 1$, the back projection of an all-ones sinogram *restricted to $S_m$*. You precompute $M$ of them.

The algorithm: cycle $m = 1, 2, \dots, M$, feeding each update into the next. After the last subset, that's one full OSEM iteration; repeat for several iterations.

**Why this gives a speedup (the heuristic).**

Each subset is a *coarse, noisy estimate* of the full gradient direction — it points roughly the same way as the full MLEM update but is computed from $1/M$ of the data. If the subsets are **balanced** (each subset's projections, summed over angles, see the whole object roughly equally — the *subset balance* condition), then $M$ cheap approximate steps move you about as far as one expensive exact step, but in the time of one. Empirically you get roughly an $M\times$ acceleration in early iterations: OSEM with $M=16$ subsets at 2 iterations $\approx$ MLEM at $\sim$32 iterations, at $1/16$ the per-iteration cost — net $\sim 16\times$ faster to a comparable image.

Logical status: this is a **heuristic with strong empirical support**, *not* a theorem. Here is the honest accounting:

1. **No monotone-ascent guarantee.** MLEM's clean $L(x^{(k+1)}) \ge L(x^{(k)})$ is *lost*. Each subset increases its *own* sub-likelihood, not the global $L$.

2. **Does not converge to the ML solution.** OSEM does not have a fixed point at $\hat x_{\text{ML}}$. Instead, once it gets close it enters a **limit cycle**: it cycles among $M$ slightly different images, one per subset, never settling. The "answer" is wherever you happen to stop within the cycle.

3. **The fix exists but costs the speed.** Variants like **RAMLA**, **BSREM**, and **COSEM** introduce relaxation (a decaying step size) to kill the limit cycle and restore convergence to the true maximizer. They trade some of OSEM's raw speed for a real convergence guarantee.

**The structural picture (cross-field connection).**

This is *exactly* the relationship between **gradient descent and stochastic gradient descent (SGD)** in machine learning. MLEM = full-batch (use all data, exact step, slow, convergent). A subset = a mini-batch. OSEM = SGD with a *fixed* learning rate: fast early progress, then it bounces around the optimum in a noise ball instead of converging — the limit cycle *is* SGD's "the loss plateaus and jitters." The relaxation in RAMLA/BSREM is precisely SGD's **decaying learning rate schedule** ($\eta_k \to 0$) that turns the noise ball into genuine convergence (Robbins–Monro conditions: $\sum \eta_k = \infty, \sum \eta_k^2 < \infty$).

Slogan: **"OSEM is SGD for tomography."** Stress-test / hypothesis: the analogy is structural, not literal — OSEM's update is *multiplicative* (preserving $x \ge 0$, as in MLEM) while textbook SGD is *additive*; and OSEM cycles subsets in a *fixed deterministic order* rather than sampling mini-batches randomly. The shared essence is "approximate the full data term with a cheap subset to take more, noisier steps." Where the analogy holds: convergence behavior (limit cycle ↔ noise ball, relaxation ↔ LR decay). Where it breaks: the algebraic form of the step.

**How many subsets — the trade-off.**

More subsets $M$ → faster early acceleration, but each subset is noisier and the limit cycle is larger (worse final image if you don't relax). Fewer subsets → slower, but more stable and closer to MLEM. A common practical sweet spot is $M$ in the range where each subset still has enough angular coverage to "see" the object — too few angles per subset and the subset balance condition fails, producing artifacts. Typical PET practice: a handful to a few dozen subsets, often chosen so the number of angular views divides evenly.

#### Practice

The only new ingredients over MLEM: a rule for partitioning LORs into subsets, and a list of per-subset sensitivity images. The standard PET partition is **by angular view** — subset $m$ takes every $M$-th view (interleaved, not contiguous blocks), which keeps each subset angularly spread out and roughly satisfies subset balance.

A clean way to realize "$A_m$" with parallelproj is to build one projector per subset, each restricted to that subset's views, by slicing the LOR descriptor's view indices. To keep this self-contained and backend-agnostic, the version below instead applies the subset as a **mask on the sinogram**: zero out all LORs not in $S_m$ before back projecting. This computes the *same* $A_m^T(\cdot)$ as a restricted projector — back projecting a sinogram that is zero outside $S_m$ is identical to back projecting only the $S_m$ rows — at the cost of still running the full projector (so you lose the FLOP savings, but the *math and convergence behavior are identical*). For real speed you slice the projector; for learning the algorithm, the mask makes the logic transparent.

```python
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

    print(f"Total {n_iter} iterations")
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
```

```python
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
```

Two practical notes that fall straight out of the theory. First, the **product $n_{\text{subsets}} \times n_{\text{iter}}$** is the meaningful quantity — it's the total number of image updates, the rough equivalent of MLEM's iteration count; OSEM $5 \times 4$ and MLEM $20$ should look similar, but OSEM gets there in $\sim 4$ full data sweeps instead of $20$. Second, because of the **limit cycle**, running OSEM "longer" past convergence does not refine the image — it just cycles, and with many subsets and no relaxation the cycled images visibly degrade. Stop when the image stabilizes, or move to a relaxed variant (BSREM) if you need a guaranteed-convergent answer.

The natural next step closes the loop back to the regularized objective $\min_x \,\|Ax - y\|^2_{\text{Poisson}} + \lambda R(x)\$.

TODO:
1. scatter contamination