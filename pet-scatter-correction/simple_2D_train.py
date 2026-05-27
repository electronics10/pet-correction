"""
simple_2D_train.py — 2D scatter sinogram prediction for PET (pipeline check).

Treats a single 3D direct-sinogram stack of shape (n_direct, nrad, nang) as a
dataset of n_direct independent 2D samples. Input is the *total* sinogram,
target is the *scatter* sinogram. This is a pipeline sanity check, not a real
evaluation: slices from one volume are highly correlated, so the random
slice-level val split is optimistic. When multi-scan data arrives, split by
scan instead.

Loss / output convention
------------------------
Two loss modes are supported and they imply *different* meanings for the raw
network output:

  loss = "poisson_nll" : raw output is log(rate). Loss is
                         F.poisson_nll_loss(..., log_input=True), i.e.
                         exp(pred) - target * pred. To recover predicted
                         counts, exp() the raw output.
  loss = "mse"         : raw output is counts directly. No transform needed
                         (negative outputs are clipped at 0 at inference).

`to_counts()` hides this asymmetry — always call it (directly or via
`predict()`) before comparing predictions to ground truth, plotting, or
saving arrays.

Run artifacts
-------------
`save_run(out_dir, ...)` writes four files into `out_dir`:
  model.pt       — torch state_dict of the best-val-loss model
  config.json    — TrainConfig (model name, loss, lr, input_scale, seed, ...)
  history.json   — per-epoch train/val loss, best_val_loss, val_indices
  arrays.npz     — total, scatter_true, scatter_pred, val_indices

`load_run(out_dir)` reconstructs (model, cfg, history, arrays) from the same.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Type
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================== Config

@dataclass
class TrainConfig:
    model: str = "unet"           # "unet" | "convnet"
    loss: str = "poisson_nll"     # "poisson_nll" | "mse"
    n_epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-3
    val_fraction: float = 0.2
    seed: int = 0
    device: str = "cuda"
    input_scale: float | None = None   # set from train data if None

    def model_cls(self) -> Type[nn.Module]:
        return {"unet": UNet, "convnet": SimpleConvNet}[self.model]


# ============================================================== Dataset

class SinogramDataset(Dataset):
    """One 2D (1, nrad, nang) sample per direct plane.

    `input_scale` divides the input only. Target stays in raw counts so the
    Poisson statistics of the target are preserved when loss="poisson_nll".
    """

    def __init__(
        self,
        total_stack: np.ndarray,
        scatter_stack: np.ndarray,
        input_scale: float = 1.0,
    ):
        self.X = torch.tensor(total_stack[:, None] / input_scale, dtype=torch.float32)
        self.Y = torch.tensor(scatter_stack[:, None], dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


# ============================================================== Models

class SimpleConvNet(nn.Module):
    """Three-layer conv net. Single channel in, single channel out, same H×W."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Tiny U-Net: two downsampling levels, skip connections, 1×1 head."""

    def __init__(self):
        super().__init__()
        self.enc1 = self._block(1, 16)
        self.enc2 = self._block(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(32, 64)
        self.dec2 = self._block(64 + 32, 32)
        self.dec1 = self._block(32 + 16, 16)
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), nn.ReLU(),
        )

    @staticmethod
    def _up(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Bilinear upsample to exactly match target's H×W (handles odd dims).
        return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b  = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self._up(b,  e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        return self.final(d1)


# ============================================================== Loss & output transform

def compute_loss(pred_raw: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    """Loss in the natural parameterization for each mode."""
    if loss_name == "poisson_nll":
        # log_input=True: loss = exp(pred) - target*pred. Minimum at pred = log(target).
        return F.poisson_nll_loss(pred_raw, target, log_input=True, full=False, reduction="mean")
    if loss_name == "mse":
        return F.mse_loss(pred_raw, target)
    raise ValueError(f"Unknown loss: {loss_name!r}")


def to_counts(pred_raw: torch.Tensor, loss_name: str) -> torch.Tensor:
    """Map raw network output to physical counts (non-negative)."""
    if loss_name == "poisson_nll":
        return torch.exp(pred_raw)
    if loss_name == "mse":
        return pred_raw.clamp_min(0.0)   # negative scatter counts are unphysical
    raise ValueError(f"Unknown loss: {loss_name!r}")


# ============================================================== Training

def train(
    total_stack: np.ndarray,
    scatter_stack: np.ndarray,
    cfg: TrainConfig | None = None,
) -> tuple[nn.Module, TrainConfig, dict]:
    """Train and return (best-val model, populated cfg, history).

    The returned model has the weights from the epoch with lowest val loss.
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Compute input scale once from full input stack. This is target-independent,
    # so it does not leak validation information.
    if cfg.input_scale is None:
        mean = float(total_stack.mean())
        cfg.input_scale = mean if mean > 0 else 1.0

    dataset = SinogramDataset(total_stack, scatter_stack, input_scale=cfg.input_scale)
    n_val = int(len(dataset) * cfg.val_fraction)
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size)

    model = cfg.model_cls()().to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: dict = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state: dict | None = None
    best_epoch = -1

    for epoch in range(cfg.n_epochs):
        # ----- train -----
        model.train()
        tr_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            pred = model(x)
            loss = compute_loss(pred, y, cfg.loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_sum += loss.item()
        tr_avg = tr_sum / max(1, len(train_loader))

        # ----- validate -----
        model.eval()
        vl_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(cfg.device), y.to(cfg.device)
                vl_sum += compute_loss(model(x), y, cfg.loss).item()
        vl_avg = vl_sum / max(1, len(val_loader))

        history["train_loss"].append(tr_avg)
        history["val_loss"].append(vl_avg)

        improved = vl_avg < best_val
        if improved:
            best_val = vl_avg
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        marker = "  *" if improved else ""
        print(f"epoch {epoch+1:3d} | train {tr_avg:.4f} | val {vl_avg:.4f}{marker}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # `random_split` returns `Subset` instances; `.indices` gives the original
    # dataset positions, i.e. which axial planes are in val.
    history["val_indices"] = list(map(int, val_set.indices)) if hasattr(val_set, "indices") else []
    history["best_val_loss"] = best_val
    history["best_epoch"] = best_epoch
    return model, cfg, history


# ============================================================== Inference

def predict(model: nn.Module, total_stack: np.ndarray, cfg: TrainConfig) -> np.ndarray:
    """Predict scatter counts. Returns (n_direct, nrad, nang) float32 in count units.

    Applies the same input scaling used at training (cfg.input_scale) and the
    loss-appropriate output transform (`to_counts`).
    """
    model.eval()
    x = torch.tensor(total_stack[:, None] / cfg.input_scale, dtype=torch.float32).to(cfg.device)
    with torch.no_grad():
        out = to_counts(model(x), cfg.loss)
    return out.squeeze(1).cpu().numpy()


# ============================================================== Save / load

def save_run(
    out_dir: str | Path,
    model: nn.Module,
    cfg: TrainConfig,
    history: dict,
    total_stack: np.ndarray,
    scatter_stack: np.ndarray,
    scatter_pred: np.ndarray,
) -> Path:
    """Write model.pt, config.json, history.json, arrays.npz into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    np.savez_compressed(
        out_dir / "arrays.npz",
        total=total_stack.astype(np.float32),
        scatter_true=scatter_stack.astype(np.float32),
        scatter_pred=scatter_pred.astype(np.float32),
        val_indices=np.asarray(history.get("val_indices", []), dtype=np.int64),
    )
    print(f"saved run to {out_dir}")
    return out_dir


def load_run(
    run_dir: str | Path,
    device: str | None = None,
) -> tuple[nn.Module, TrainConfig, dict, dict]:
    """Reconstruct (model, cfg, history, arrays) saved by `save_run`."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        cfg = TrainConfig(**json.load(f))
    if device is not None:
        cfg.device = device
    model = cfg.model_cls()().to(cfg.device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=cfg.device))
    model.eval()
    with open(run_dir / "history.json") as f:
        history = json.load(f)
    arrays = dict(np.load(run_dir / "arrays.npz"))
    return model, cfg, history, arrays


# ============================================================== Main

if __name__ == "__main__":
    from mcgpu_backend import PETGeometry, Sinogram

    data_dir = Path("runs/run0")
    geom = PETGeometry.from_json(data_dir / "config.json")

    true_sino    = Sinogram(data_dir / "sinogram_Trues.raw.gz",   geom)
    scatter_sino = Sinogram(data_dir / "sinogram_Scatter.raw.gz", geom)
    total_sino   = true_sino + scatter_sino

    total_stack   = total_sino.ssrb()       # (n_direct, nrad, nang)
    scatter_stack = scatter_sino.ssrb()
    print(f"data shapes: total={total_stack.shape}, scatter={scatter_stack.shape}")

    cfg = TrainConfig(
        model="unet",
        loss="poisson_nll",
        n_epochs=30,
        batch_size=8,
        lr=1e-3,
        seed=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model, cfg, history = train(total_stack, scatter_stack, cfg)
    scatter_pred = predict(model, total_stack, cfg)

    out_dir = data_dir / f"ml_{cfg.model}_{cfg.loss}"
    save_run(out_dir, model, cfg, history,
             total_stack, scatter_stack, scatter_pred)