# ./mcgpu_backend/runner.py
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import subprocess
import threading
import time

PKG_DIR = Path(__file__).parent  # where MCGPU-PET.x and materials/ live


@dataclass
class RunResult:
    run_dir: Path
    image_trues: Path
    image_scatter: Path
    sinogram_trues: Path
    sinogram_scatter: Path
    energy_spectrum: Path
    log: Path
    wall_time_s: float
    returncode: int


class Runner:
    def __init__(self, binary=PKG_DIR / "MCGPU-PET.x", materials=PKG_DIR / "materials"):
        self.binary, self.materials = Path(binary), Path(materials)

    def __call__(
        self,
        run_dir,
        on_existing: Literal["error", "overwrite", "skip"] = "error",
        verbose: bool = True,
        timeout_s: Optional[float] = 3600,
    ) -> RunResult:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

        existing = self._handle_existing(run_dir, on_existing)
        if existing is not None:
            return existing

        self._stage(run_dir)
        self._preflight(run_dir)
        rc, dt = self._execute(run_dir, verbose, timeout_s)
        return self._collect(run_dir, rc, dt)

    def _stage(self, d):
        for src, name in [(self.binary, "MCGPU-PET.x"), (self.materials, "materials")]:
            link = d / name
            if not link.exists():
                link.symlink_to(src.resolve())

    def _preflight(self, d):
        required = ["MCGPU-PET.x", "MCGPU-PET.in", "phantom.vox", "materials"]
        missing = [r for r in required if not (d / r).exists()]
        if missing:
            raise FileNotFoundError(f"Missing in {d}: {missing}")

    def _execute(self, d, verbose, timeout_s):
        log = d / "MCGPU-PET.out"
        t0 = time.perf_counter()
        with open(log, "w") as f:
            p = subprocess.Popen(
                ["./MCGPU-PET.x", "MCGPU-PET.in"],
                cwd=d,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            timer = threading.Timer(timeout_s, p.kill) if timeout_s else None
            if timer:
                timer.start()
            try:
                for line in p.stdout:
                    if verbose:
                        print(line, end="")
                    f.write(line)
                p.wait()
            finally:
                if timer:
                    timer.cancel()
        dt = time.perf_counter() - t0
        if p.returncode < 0:  # negative returncode = killed by signal
            raise RuntimeError(
                f"Run exceeded {timeout_s}s, killed (signal {-p.returncode})"
            )
        return p.returncode, dt

    def _collect(self, d, rc, dt):
        expected = [
            "image_Trues.raw.gz",
            "image_Scatter.raw.gz",
            "sinogram_Trues.raw.gz",
            "sinogram_Scatter.raw.gz",
            "Energy_Sinogram_Spectrum.dat",
        ]
        missing = [
            f for f in expected
            if not (d / f).exists() or (d / f).stat().st_size == 0
        ]
        if rc != 0 or missing:
            raise RuntimeError(f"Run failed (rc={rc}); missing/empty: {missing}")
        return RunResult(d, *(d / f for f in expected), d / "MCGPU-PET.out", dt, rc)

    def _handle_existing(self, d, mode):
        outputs = list(d.glob("*.raw.gz"))
        if not outputs:
            return None
        if mode == "error":
            raise FileExistsError(f"Outputs exist in {d}")
        if mode == "skip":
            return self._collect(d, rc=0, dt=0.0)
        if mode == "overwrite":
            for f in outputs + list(d.glob("*.dat")) + [d / "MCGPU-PET.out"]:
                if f.exists():
                    f.unlink()
            return None
        raise ValueError(f"unknown on_existing mode: {mode!r}")