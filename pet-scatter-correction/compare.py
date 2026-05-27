"""
compare.py — visualize a saved scatter-correction run.

Loads a directory written by `simple_2D_train.save_run` and produces side-by-side
plots of total / true scatter / predicted scatter / corrected-trues for chosen
axial slices. Highlights which slices were in the validation set so you can read
honest-vs-leaky performance at a glance.

Usage
-----
CLI:
    python compare.py runs/run0/ml_unet_poisson_nll
    python compare.py runs/run0/ml_unet_poisson_nll --slices 40 79 120
    python compare.py runs/run0/ml_unet_poisson_nll --n-val 4 --n-train 2

Programmatic:
    from compare import plot_slices, plot_loss_curve, summarize
    arrays, history, cfg = load_artifacts("runs/run0/ml_unet_poisson")
    plot_slices(arrays, slices=[79], save_path="cmp.png")
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt


# ============================================================== Loading

def load_artifacts(run_dir: str | Path) -> tuple[dict, dict, dict]:
    """Load arrays.npz, history.json, config.json. No torch dependency."""
    run_dir = Path(run_dir)
    arrays = dict(np.load(run_dir / "arrays.npz"))
    with open(run_dir / "history.json") as f:
        history = json.load(f)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    return arrays, history, cfg


# ============================================================== Metrics

def per_slice_metrics(scatter_true: np.ndarray, scatter_pred: np.ndarray) -> dict:
    """Per-slice scalar metrics; arrays shaped (n_direct, nrad, nang)."""
    diff = scatter_pred - scatter_true
    mae = np.mean(np.abs(diff), axis=(1, 2))
    rmse = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))
    # Relative bias on total scatter counts, guarding against empty slices.
    true_sum = scatter_true.sum(axis=(1, 2))
    pred_sum = scatter_pred.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_bias = np.where(true_sum > 0, (pred_sum - true_sum) / true_sum, 0.0)
    return {"mae": mae, "rmse": rmse, "rel_bias": rel_bias,
            "true_sum": true_sum, "pred_sum": pred_sum}


def summarize(arrays: dict, history: dict, cfg: dict) -> None:
    """Print a one-screen summary of a run."""
    scatter_true = arrays["scatter_true"]
    scatter_pred = arrays["scatter_pred"]
    val_idx = arrays["val_indices"]
    n_direct = scatter_true.shape[0]
    train_idx = np.setdiff1d(np.arange(n_direct), val_idx)

    m = per_slice_metrics(scatter_true, scatter_pred)

    print(f"model={cfg['model']}  loss={cfg['loss']}  "
          f"epochs={cfg['n_epochs']}  lr={cfg['lr']}  seed={cfg['seed']}")
    print(f"input_scale={cfg['input_scale']:.4g}")
    print(f"best epoch = {history.get('best_epoch', '?')}  "
          f"best val loss = {history.get('best_val_loss', float('nan')):.4f}")
    print(f"n_slices: total={n_direct}  train={len(train_idx)}  val={len(val_idx)}")

    def _row(label, idx):
        if len(idx) == 0:
            return
        print(f"  {label:5s}  MAE={m['mae'][idx].mean():8.3f}  "
              f"RMSE={m['rmse'][idx].mean():8.3f}  "
              f"rel_bias={m['rel_bias'][idx].mean():+.3%}")
    _row("train", train_idx)
    _row("val",   val_idx)


# ============================================================== Plotting

def plot_loss_curve(history: dict, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"],   label="val")
    best = history.get("best_epoch")
    if best is not None and best >= 0:
        ax.axvline(best, color="k", linestyle="--", alpha=0.4,
                   label=f"best epoch ({best+1})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Training history")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"saved {save_path}")


def plot_slices(
    arrays: dict,
    slices: list[int],
    save_path: str | Path,
    val_set: set[int] | None = None,
) -> None:
    """One row per slice: total | scatter_true | scatter_pred | trues_estimated.

    `val_set`: indices that were in val. If a row's slice is in val, the row
    label says "[val]" so you can read leakage at a glance.
    """
    total        = arrays["total"]
    scatter_true = arrays["scatter_true"]
    scatter_pred = arrays["scatter_pred"]
    trues_est    = total - scatter_pred  # corrected estimate

    n_rows = len(slices)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.2 * n_rows), squeeze=False)

    col_titles = ["total (input)", "scatter (true)", "scatter (pred)", "trues = total − pred"]

    for r, z in enumerate(slices):
        if not (0 <= z < total.shape[0]):
            raise IndexError(f"slice {z} out of range [0, {total.shape[0]})")

        # Use a common scale across scatter true/pred for fair visual comparison.
        s_vmax = max(scatter_true[z].max(), scatter_pred[z].max(), 1e-12)
        panels = [
            (total[z],        None,  None,    "hot"),
            (scatter_true[z], 0.0,   s_vmax,  "hot"),
            (scatter_pred[z], 0.0,   s_vmax,  "hot"),
            (trues_est[z],    None,  None,    "hot"),
        ]
        for c, (img, vmin, vmax, cmap) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(img, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if r == 0:
                ax.set_title(col_titles[c])
            ax.set_xticks([]); ax.set_yticks([])

        tag = " [val]" if (val_set is not None and z in val_set) else " [train]"
        axes[r, 0].set_ylabel(f"z={z}{tag}", fontsize=11)

    fig.suptitle("Scatter prediction — per-slice comparison", y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")


# ============================================================== Slice selection

def pick_slices(
    arrays: dict,
    n_val: int,
    n_train: int,
    seed: int = 0,
) -> list[int]:
    """Pick a few val slices and a few train slices, spaced across the volume."""
    n_direct = arrays["scatter_true"].shape[0]
    val_idx = np.asarray(arrays["val_indices"], dtype=int)
    train_idx = np.setdiff1d(np.arange(n_direct), val_idx)

    def _spread(idx_pool: np.ndarray, k: int) -> list[int]:
        if len(idx_pool) == 0 or k == 0:
            return []
        # Evenly spaced picks across the sorted pool — covers the volume.
        positions = np.linspace(0, len(idx_pool) - 1, num=min(k, len(idx_pool)))
        return sorted(int(idx_pool[int(round(p))]) for p in positions)

    return _spread(np.sort(val_idx), n_val) + _spread(np.sort(train_idx), n_train)


# ============================================================== CLI

def main():
    p = argparse.ArgumentParser(description="Compare saved scatter-correction run.")
    p.add_argument("run_dir", type=str, help="directory written by save_run()")
    p.add_argument("--slices", type=int, nargs="+", default=None,
                   help="explicit slice indices; overrides --n-val / --n-train")
    p.add_argument("--n-val",   type=int, default=3, help="# val slices to plot")
    p.add_argument("--n-train", type=int, default=2, help="# train slices to plot")
    p.add_argument("--out", type=str, default=None,
                   help="output prefix; default: <run_dir>/compare")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_prefix = Path(args.out) if args.out else run_dir / "compare"

    arrays, history, cfg = load_artifacts(run_dir)

    print("=" * 60)
    summarize(arrays, history, cfg)
    print("=" * 60)

    slices = args.slices if args.slices is not None else pick_slices(arrays, args.n_val, args.n_train)
    val_set = set(int(i) for i in arrays["val_indices"])
    plot_slices(arrays, slices, save_path=f"{out_prefix}_slices.png", val_set=val_set)
    plot_loss_curve(history, save_path=f"{out_prefix}_loss.png")


if __name__ == "__main__":
    main()