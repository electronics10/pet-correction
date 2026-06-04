from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mcgpu_backend.phantom import Phantom


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Cylinder envelope (mm). Increased max sizes to push integrated electron
# density (and hence scatter fraction) higher.
RX_MIN, RX_MAX = 10.0, 45.0     # transverse half-axis x
RY_MIN, RY_MAX = 10.0, 45.0     # transverse half-axis y
LZ_MIN, LZ_MAX = 80.0, 120.0    # axial length (full)

# Background medium
RHO_BG_MIN, RHO_BG_MAX = 0.8, 1.2  # g/cm^3
A_BG = 1.0                          # reference activity unit

# Balls
N_BALLS = 6                         # max balls per phantom
P_BALL_PRESENT = 0.85               # prob. each slot is filled
R_BALL_MIN, R_BALL_MAX = 2.0, 8.0   # mm
MAX_PLACEMENT_TRIES = 100           # rejection-sampling cap

# (rho, activity) joint categories: (weight, rho_range, a/A_BG range, label)
CATEGORIES = [
    (0.35, (0.9, 1.1), (10.0, 50.0), "soft/hot"),
    (0.15, (0.9, 1.1), (1.0,   5.0), "soft/warm"),
    (0.20, (0.9, 1.1), (0.0,   0.0), "soft/cold"),
    (0.15, (1.4, 1.9), (0.0,   0.0), "bone/cold"),
    (0.15, (0.2, 0.4), (0.0,   2.0), "lung/low"),
]

# Output
N_PHANTOMS  = 10
MASTER_SEED = 42
CSV_PATH    = "phantoms.csv"


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_category(rng):
    weights = np.array([c[0] for c in CATEGORIES])
    idx = rng.choice(len(CATEGORIES), p=weights / weights.sum())
    _, rho_range, a_range, label = CATEGORIES[idx]
    rho = rng.uniform(*rho_range)
    a   = rng.uniform(*a_range) * A_BG
    return rho, a, label


def fits_in_cylinder(x, y, z, r, rx, ry, lz):
    """Ball (center, radius) lies entirely inside cylinder centered at origin."""
    if abs(z) + r > lz / 2:
        return False
    if rx <= r or ry <= r:
        return False
    return (x / (rx - r))**2 + (y / (ry - r))**2 <= 1.0


def overlaps_any(x, y, z, r, placed):
    for (xp, yp, zp, rp) in placed:
        if (x - xp)**2 + (y - yp)**2 + (z - zp)**2 < (r + rp)**2:
            return True
    return False


def sample_ball(rng, rx, ry, lz, placed):
    r = rng.uniform(R_BALL_MIN, R_BALL_MAX)
    for _ in range(MAX_PLACEMENT_TRIES):
        x = rng.uniform(-rx, rx)
        y = rng.uniform(-ry, ry)
        z = rng.uniform(-lz / 2, lz / 2)
        if not fits_in_cylinder(x, y, z, r, rx, ry, lz):
            continue
        if overlaps_any(x, y, z, r, placed):
            continue
        rho, a, label = sample_category(rng)
        return dict(present=True, r=r, x=x, y=y, z=z, rho=rho, a=a, category=label)
    return absent_ball()


def absent_ball():
    return dict(present=False, r=np.nan, x=np.nan, y=np.nan, z=np.nan,
                rho=np.nan, a=np.nan, category=None)


def sample_phantom(rng):
    rx     = rng.uniform(RX_MIN, RX_MAX)
    ry     = rng.uniform(RY_MIN, RY_MAX)
    lz     = rng.uniform(LZ_MIN, LZ_MAX)
    rho_bg = rng.uniform(RHO_BG_MIN, RHO_BG_MAX)

    balls, placed = [], []
    for _ in range(N_BALLS):
        if rng.random() > P_BALL_PRESENT:
            balls.append(absent_ball())
            continue
        b = sample_ball(rng, rx, ry, lz, placed)
        if b["present"]:
            placed.append((b["x"], b["y"], b["z"], b["r"]))
        balls.append(b)

    return dict(cyl_rx=rx, cyl_ry=ry, cyl_lz=lz,
                cyl_rho=rho_bg, cyl_a=A_BG, balls=balls)


# ---------------------------------------------------------------------------
# Flatten + write
# ---------------------------------------------------------------------------

def to_row(run_id, seed, phantom):
    row = dict(run_id=run_id, seed=seed,
               cyl_rx=phantom["cyl_rx"], cyl_ry=phantom["cyl_ry"],
               cyl_lz=phantom["cyl_lz"],
               cyl_rho=phantom["cyl_rho"], cyl_a=phantom["cyl_a"])
    for i, b in enumerate(phantom["balls"], start=1):
        row[f"ball{i}_present"]  = b["present"]
        row[f"ball{i}_r"]        = b["r"]
        row[f"ball{i}_x"]        = b["x"]
        row[f"ball{i}_y"]        = b["y"]
        row[f"ball{i}_z"]        = b["z"]
        row[f"ball{i}_rho"]      = b["rho"]
        row[f"ball{i}_a"]        = b["a"]
        row[f"ball{i}_category"] = b["category"]
    return row


def generate_phantoms_csv():
    rows = []
    for i in range(N_PHANTOMS):
        seed = MASTER_SEED + i
        rng  = np.random.default_rng(seed)
        rows.append(to_row(run_id=i, seed=seed, phantom=sample_phantom(rng)))

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Wrote {len(df)} phantoms to {CSV_PATH}\n")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df)


"""
Phantom voxelizer: turn one row of phantoms.csv into a voxelized Phantom
suitable for MCGPU-PET.

Conventions
-----------
- Arrays are shaped (Nz, Ny, Nx); arr[z, y, x].
- The grid is centered on the origin and tightly fits the cylinder plus padding.
- Materials: 1 = air (outside the cylinder), 2 = water (inside the cylinder
  and inside the balls, at scaled density). At 511 keV soft tissue, bone-like
  and lung-like inclusions are all represented as water at varying density;
  the residual Z/A error for bone is a few percent and acceptable for Stage 1.
- The CSV activity is treated as a concentration (arbitrary units per mL);
  the builder converts to per-voxel activity by multiplying by voxel volume.
  Scale the whole array later to convert into Bq.
"""


MATERIAL_AIR   = 1
MATERIAL_WATER = 2
DENSITY_AIR    = 0.0012  # g/cm^3 at STP


def build_phantom(
    row,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    padding_mm: float = 5.0,
) -> Phantom:
    """Voxelize one phantom-spec row (pandas Series or dict) into a Phantom."""
    dx, dy, dz = voxel_size_mm
    rx = float(row["cyl_rx"])
    ry = float(row["cyl_ry"])
    lz = float(row["cyl_lz"])

    # Grid extent: cylinder bounding box + padding, snapped to integer voxels.
    x_half = rx + padding_mm
    y_half = ry + padding_mm
    z_half = lz / 2 + padding_mm
    Nx = int(np.ceil(2 * x_half / dx))
    Ny = int(np.ceil(2 * y_half / dy))
    Nz = int(np.ceil(2 * z_half / dz))
    x_half = Nx * dx / 2
    y_half = Ny * dy / 2
    z_half = Nz * dz / 2

    # Voxel-center coordinates (1-D), reshaped to broadcast as (Nz, Ny, Nx).
    x = (-x_half + (np.arange(Nx) + 0.5) * dx).astype(np.float32)
    y = (-y_half + (np.arange(Ny) + 0.5) * dy).astype(np.float32)
    z = (-z_half + (np.arange(Nz) + 0.5) * dz).astype(np.float32)
    X = x[None, None, :]
    Y = y[None, :, None]
    Z = z[:, None, None]

    # Initialize: air everywhere.
    material_id = np.full((Nz, Ny, Nx), MATERIAL_AIR, dtype=np.uint8)
    density     = np.full((Nz, Ny, Nx), DENSITY_AIR, dtype=np.float32)
    activity    = np.zeros((Nz, Ny, Nx), dtype=np.float32)

    voxel_vol_mL = (dx * dy * dz) / 1000.0

    # Cylinder: elliptical in xy, finite in z.
    in_cyl = ((X / rx) ** 2 + (Y / ry) ** 2 <= 1.0) & (np.abs(Z) <= lz / 2)
    material_id[in_cyl] = MATERIAL_WATER
    density[in_cyl]     = float(row["cyl_rho"])
    activity[in_cyl]    = float(row["cyl_a"]) * voxel_vol_mL

    # Balls: later balls overwrite earlier ones if they overlap (with proper
    # rejection sampling upstream they don't).
    for i in (1, 2, 3):
        r = row.get(f"ball{i}_r", np.nan)
        if pd.isna(r):
            continue
        bx, by, bz = (float(row[f"ball{i}_x"]),
                      float(row[f"ball{i}_y"]),
                      float(row[f"ball{i}_z"]))
        br   = float(r)
        brho = float(row[f"ball{i}_rho"])
        ba   = float(row[f"ball{i}_a"])
        in_ball = (X - bx) ** 2 + (Y - by) ** 2 + (Z - bz) ** 2 <= br ** 2
        material_id[in_ball] = MATERIAL_WATER
        density[in_ball]     = brho
        activity[in_ball]    = ba * voxel_vol_mL

    return Phantom(
        material_id=material_id,
        density=density,
        activity=activity,
        voxel_size_mm=voxel_size_mm,
        material_names=["air", "water"],
    )


def iter_phantoms_from_csv(csv_path: str, **kwargs):
    """Yield (run_id, Phantom) pairs, one per row of the CSV."""
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        yield int(row["run_id"]), build_phantom(row, **kwargs)


# ---------------------------------------------------------------------------
# Demo: build the first phantom and save preview slices.
# ---------------------------------------------------------------------------

def show_phantom(row):
    phantom = build_phantom(row, voxel_size_mm=(1.0, 1.0, 1.0), padding_mm=5.0)
    phantom.validate()

    print(f"Phantom for run_id {int(row['run_id'])} (seed {int(row['seed'])}):")
    print(f"  grid shape (Nz, Ny, Nx) = {phantom.shape_zyx}")
    print(f"  voxel size              = {phantom.voxel_size_mm} mm")
    print(f"  total volume            = {phantom.total_volume_ml:.2f} mL")
    print(f"  total mass              = {phantom.total_mass_gram:.2f} g")
    print(f"  total activity          = {phantom.total_activity_Bq:.4f} (arb units)")
    print(f"  density range           = [{phantom.density.min():.4f}, "
          f"{phantom.density.max():.4f}] g/cm^3")
    print(f"  activity range          = [{phantom.activity.min():.4f}, "
          f"{phantom.activity.max():.4f}]")

    # Three orthogonal central slices for each of (material, density, activity).
    Nz, Ny, Nx = phantom.shape_zyx
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fields = [("material_id", phantom.material_id),
              ("density",     phantom.density),
              ("activity",    phantom.activity)]
    for col, (label, arr) in enumerate(fields):
        axes[0, col].imshow(arr[Nz // 2], cmap="gray")
        axes[0, col].set_title(f"{label} (axial, z={Nz // 2})")
        axes[1, col].imshow(arr[:, Ny // 2, :], cmap="gray")
        axes[1, col].set_title(f"{label} (coronal, y={Ny // 2})")
        axes[2, col].imshow(arr[:, :, Nx // 2], cmap="gray")
        axes[2, col].set_title(f"{label} (sagittal, x={Nx // 2})")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("phantom_preview.png", dpi=110)
    print("\nSaved slice preview to phantom_preview.png")

if __name__ == "__main__":
    # generate_phantoms_csv()
    df  = pd.read_csv("phantoms.csv")
    row = df.iloc[8]
    show_phantom(row)

    