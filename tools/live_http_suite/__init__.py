"""Opt-in live HTTP suite against a running omni instance (issue #15).

Targets a process you already started. Does not spawn omni. Does not run
under default cargo test or CI. See docs/README.md for how to run.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
