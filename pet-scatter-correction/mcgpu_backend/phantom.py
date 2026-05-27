from pathlib import Path

class PhantomVoxGenerator:
    def __init__(self, template_path = "mcgpu_backend/template.vox"):
        self.lines = Path(template_path).read_text().splitlines()

    def write(self, out_dir: str | Path):
        out_path = Path(out_dir) / "phantom.vox"
        # Create the parent directories if they don't exist
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text('\n'.join(self.lines) + '\n')
        print(f"Write {out_path}")


if __name__ == "__main__":
    OUT_DIR = "runs/run2"
    pvg = PhantomVoxGenerator()
    pvg.write(OUT_DIR)