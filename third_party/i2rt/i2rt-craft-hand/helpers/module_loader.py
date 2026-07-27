"""Import sibling experiment helper packages under stable aliases."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_helper_package(alias: str, package_dir: Path):
    """Load a helper package whose parent folder is not importable by name.

    The active experiment folders are named with hyphens (`i2rt-hand`,
    `craft-hand`), and several contain a top-level `helpers` package.
    Loading them under aliases keeps their relative imports working without
    colliding on `helpers.*`.
    """

    if alias in sys.modules:
        return sys.modules[alias]

    package_dir = package_dir.resolve()
    init_path = package_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        alias,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper package {alias} from {package_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module
