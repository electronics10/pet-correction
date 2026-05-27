import shutil, subprocess, tempfile
from pathlib import Path

BACKEND = Path("mcgpu_backend").resolve()  # holds MCGPU-PET.x + materials/

def run_mcgpu_pet(run_dir):
    run_dir = Path(run_dir).resolve()  # has MCGPU-PET.in, phantom.vox

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # symlink backend pieces in (no copy, no edits to his code)
        (tmp / "MCGPU-PET.x").symlink_to(BACKEND / "MCGPU-PET.x")
        (tmp / "materials").symlink_to(BACKEND / "materials")
        # symlink this run's inputs
        (tmp / "MCGPU-PET.in").symlink_to(run_dir / "MCGPU-PET.in")
        (tmp / "phantom.vox").symlink_to(run_dir / "phantom.vox")

        subprocess.run(["./MCGPU-PET.x", "MCGPU-PET.in"], cwd=tmp, check=True)

        # harvest outputs back into the run dir
        for f in tmp.iterdir():
            if f.name not in {"MCGPU-PET.x", "materials", "MCGPU-PET.in", "phantom.vox"}:
                shutil.move(str(f), run_dir / f.name)