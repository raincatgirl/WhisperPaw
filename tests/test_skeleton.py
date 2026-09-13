"""Sanity tests for the package skeleton — these do not exercise real behavior yet."""
from __future__ import annotations

import importlib


def test_package_imports() -> None:
    """The top-level package should import without side effects."""
    mod = importlib.import_module("whisperpaw")
    assert mod.__version__ == "0.0.1"
    for name in ("read", "sound", "watch", "zoom"):
        assert hasattr(mod, name), f"missing submodule: {name}"


def test_subcommand_entrypoints_exist() -> None:
    """Every subcommand module should expose a main() returning an int."""
    for name in ("read", "sound", "watch", "zoom"):
        mod = importlib.import_module(f"whisperpaw.{name}")
        assert callable(getattr(mod, "main", None)), f"{name} missing main()"
        assert isinstance(mod.main(), int), f"{name}.main() must return int"
