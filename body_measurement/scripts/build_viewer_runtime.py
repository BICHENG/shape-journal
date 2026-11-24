"""Build the offline viewer runtime bundle from pinned npm packages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from body_measurement.viewer import VIEWER_RUNTIME_SOURCES, viewer_source_stamp

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
VIEWER_ASSETS_DIR = PACKAGE_ROOT / "viewer_assets"
ENTRY = VIEWER_ASSETS_DIR / "src" / "viewer_runtime.js"
OUTPUT = VIEWER_ASSETS_DIR / "viewer_runtime.bundle.js"
NPM_CANDIDATES = (
    "npm.cmd",
    "npm.exe",
    "npm",
    r"C:\Program Files\nodejs\npm.cmd",
)
TECH_STACK_PATH = VIEWER_ASSETS_DIR / "tech_stack.json"


def build_viewer_runtime(node_modules: Path | None = None) -> Path:
    stack = json.loads(TECH_STACK_PATH.read_text(encoding="utf-8"))
    specs = stack["runtime_policy"]["build_packages"]
    dependencies = dict(spec.rsplit("@", 1) for spec in specs)
    with tempfile.TemporaryDirectory(prefix="body_measurement_viewer_build_") as temp_dir:
        build_dir = Path(temp_dir)
        build_src = build_dir / "src"
        build_src.mkdir()
        for name in VIEWER_RUNTIME_SOURCES:
            shutil.copyfile(ENTRY.parent / name, build_src / name)
        if node_modules is None:
            subprocess.run(
                [
                    _find_npm(), "install", "--prefer-offline", "--no-audit", "--no-fund",
                    "--save-exact", *specs,
                ],
                cwd=build_dir,
                check=True,
            )
            node_modules = build_dir / "node_modules"
        node_modules = node_modules.resolve(strict=True)
        for name, version in dependencies.items():
            package = json.loads((node_modules / name / "package.json").read_text(encoding="utf-8"))
            if package["name"] != name or package["version"] != version:
                raise ValueError(f"Viewer runtime requires {name}@{version} in {node_modules}")

        subprocess.run(
            [
                "node", str(node_modules / "esbuild/bin/esbuild"), str(build_src / ENTRY.name),
                "--bundle", "--format=iife", "--global-name=BodyMeasurementViewer",
                f"--outfile={OUTPUT}", "--log-level=warning", "--legal-comments=none",
                f"--banner:js={viewer_source_stamp(build_src)}",
            ],
            cwd=build_dir,
            env={**os.environ, "NODE_PATH": str(node_modules)},
            check=True,
        )
    return OUTPUT


def _find_npm() -> str:
    for candidate in NPM_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        path = Path(candidate)
        if path.exists():
            return str(path)
    raise RuntimeError("npm is required to rebuild viewer_runtime.bundle.js")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-modules", type=Path, help="Use an existing installation of the pinned packages")
    args = parser.parse_args()
    print(build_viewer_runtime(args.node_modules))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
