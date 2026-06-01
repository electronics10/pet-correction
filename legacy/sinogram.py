from dataclasses import dataclass, fields
from pathlib import Path
import gzip
import json

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PETGeometry:
    """Scanner + histogramming parameters shared across all measurements in a run.

    Frozen so two Sinograms can safely share one instance and equality is
    well-defined for add/compatibility checks.
    """
    num_radial_bins: int
    num_angular_bins: int
    num_rings: int
    span: int

    @property
    def mrd(self) -> int:
        """Maximum ring difference."""
        return self.num_rings - 1

    @property
    def n_direct_planes(self) -> int:
        """Number of direct (segment-0) planes after SSRB."""
        return 2 * self.num_rings - 1

    @classmethod
    def from_json(cls, file_path: str | Path) -> "PETGeometry":
        with open(file_path, "r") as f:
            config = json.load(f)
        keep = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in keep})

    @property
    def segment_table(self) -> list[tuple[int, int, int, int]]:
        """Michelogram segment ordering used by MCGPU-PET.

        Returns list of (segment, n_planes, min_ring_diff, max_ring_diff).
        Order: seg 0, seg -1, seg +1, seg -2, seg +2, ...

        Spanning conflates `span` adjacent ring differences into one segment;
        it is lossy (uninvertible) but reduces storage.
        """
        max_rd = (self.span - 1) // 2
        table = [(0, self.n_direct_planes, -max_rd, max_rd)]
        k = 1
        while True:
            min_rd = max_rd + 1
            if min_rd > self.mrd:
                break
            n_planes = self.n_direct_planes - 2 * min_rd
            if n_planes <= 0:
                break
            max_rd = min(min_rd + self.span - 1, self.mrd)
            table.append((-k, n_planes, -max_rd, -min_rd))
            table.append((+k, n_planes, +min_rd, +max_rd))
            k += 1
        return table

    @property
    def total_planes(self) -> int:
        return sum(n for _, n, *_ in self.segment_table)


class Sinogram:
    """One loaded PET measurement: counts indexed by segment, view, radial bin.

    Internal representation is `segments`: {segment_index: (n_planes, phi, s)}.
    The flat 1D buffer on disk and the 3D michelogram are reshapes of the
    same data — we keep the dict-of-stacks form because every downstream op
    (SSRB, FORE, plotting) is naturally per-segment.
    """

    def __init__(
        self,
        source: str | Path | dict[int, np.ndarray],
        geom: PETGeometry,
    ) -> None:
        self.geom = geom
        if isinstance(source, (str, Path)):
            self.segments = self._load_from_disk(source)
        elif isinstance(source, dict):
            self.segments = source
        else:
            raise TypeError(f"source must be path or dict, got {type(source)}")

    # ------------------------------------------------------------------ I/O

    def _load_from_disk(self, path: str | Path) -> dict[int, np.ndarray]:
        """Read gzipped int32 buffer and split into per-segment stacks."""
        with gzip.open(path, "rb") as f:
            flat = np.frombuffer(f.read(), dtype=np.int32)

        expected = (
            self.geom.total_planes
            * self.geom.num_angular_bins
            * self.geom.num_radial_bins
        )
        if flat.size != expected:
            raise ValueError(
                f"{path}: expected {expected} int32 values "
                f"({self.geom.total_planes} planes), got {flat.size}"
            )

        michelogram = flat.reshape(
            self.geom.total_planes,
            self.geom.num_angular_bins,
            self.geom.num_radial_bins,
        )

        segments: dict[int, np.ndarray] = {}
        offset = 0
        for seg, n_planes, *_ in self.geom.segment_table:
            segments[seg] = michelogram[offset : offset + n_planes]
            offset += n_planes
        return segments

    # ------------------------------------------------------------------ algebra

    def __add__(self, other: "Sinogram") -> "Sinogram":
        if self.geom != other.geom:
            raise ValueError("Cannot add Sinograms with different geometries")
        new_segs = {k: self.segments[k] + other.segments[k] for k in self.segments}
        return Sinogram(new_segs, self.geom)

    def __mul__(self, scalar: float) -> "Sinogram":
        new_segs = {k: v * scalar for k, v in self.segments.items()}
        return Sinogram(new_segs, self.geom)

    __rmul__ = __mul__

    # ------------------------------------------------------------------ rebinning

    def ssrb(self) -> np.ndarray:
        """Single-Slice Rebinning: assign each oblique LOR to its axial midpoint.

        Output shape: (n_direct, num_radial_bins, num_angular_bins), float32.
        Layout matches skimage.transform.iradon's expected (detector, angle) per slice.
        Counts are summed (not averaged) to pool statistics across segments.
        """
        n_direct = self.geom.n_direct_planes
        out = np.zeros(
            (n_direct, self.geom.num_angular_bins, self.geom.num_radial_bins),
            dtype=np.float32,
        )
        for stack in self.segments.values():
            n_planes = stack.shape[0]
            # n_direct - n_planes is always even (= 2 * min_rd of the segment),
            # so integer division is exact: center the oblique stack on the
            # direct-plane axis.
            shift = (n_direct - n_planes) // 2
            out[shift : shift + n_planes] += stack
        return out.transpose(0, 2, 1)  # -> (plane, s, phi) for skimage convention

    def fore(self) -> np.ndarray:
        """Fourier Rebinning. Not yet implemented."""
        raise NotImplementedError


if __name__ == "__main__":
    run_dir = Path("runs/run0/")
    geom = PETGeometry.from_json(run_dir / "config.json")

    true    = Sinogram(run_dir / "sinogram_Trues.raw.gz",   geom)
    scatter = Sinogram(run_dir / "sinogram_Scatter.raw.gz", geom)
    total = true + scatter

    true_stack = true.ssrb()
    total_stack = total.ssrb()
    scatter_stack = scatter.ssrb()
    print(true_stack, total_stack, scatter_stack)