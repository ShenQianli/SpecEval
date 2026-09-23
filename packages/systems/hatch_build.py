"""Include shared reference inputs in system distributions."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        package = Path(self.root)
        # Source distributions already carry the resources under src/async_hbn.
        if (package / "src/async_hbn/data/profiles/manifest.json").is_file():
            return
        root = package.parent.parent
        resources = (
            "data/profiles",
            "data/benchmarks/manifest.json",
            "results/configurations",
            "results/validation",
            "results/systems",
            "results/replay",
        )
        prefix = "src/async_hbn" if self.target_name == "sdist" else "async_hbn"
        build_data["force_include"] = {
            str(root / name): f"{prefix}/{name}" for name in resources
        }
