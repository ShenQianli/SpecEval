"""Locate bundled resources or shared source-checkout inputs."""

from pathlib import Path


def resource_dir(name):
    package = Path(__file__).resolve().parent
    packaged = package / name
    if packaged.is_dir():
        return packaged
    for root in package.parents:
        if (root / "data/profiles/manifest.json").is_file() and (root / "results/validation/manifest.json").is_file():
            return root / name
    raise FileNotFoundError(f"Cannot locate {name} resources for async_hbn")
