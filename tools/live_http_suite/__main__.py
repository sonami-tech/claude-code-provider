"""CLI entry: python3 -m tools.live_http_suite

Targets a *running* omni process. Does not spawn omni. Opt-in only; not
invoked by cargo test or CI.

Examples:
  # Default base http://127.0.0.1:18321
  python3 -m tools.live_http_suite

  # Ephemeral verification instance
  python3 -m tools.live_http_suite --base-url http://127.0.0.1:19001

  # Skip dual-mode Anthropic edge (flag must be set before that test runs)
  python3 -m tools.live_http_suite --dual-mode-off
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python3 -m tools.live_http_suite` from repo root without install.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.live_http_suite.config import SuiteConfig  # noqa: E402
from tools.live_http_suite.suite import run_suite, summarize  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m tools.live_http_suite",
        description=(
            "Live HTTP suite (issue #15) against a running omni instance. "
            "Does not spawn omni. Does not replace OMNI_LIVE_TESTS cargo tests."
        ),
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="Omni base URL (default OMNI_BASE_URL or http://127.0.0.1:18321)",
    )
    p.add_argument(
        "--dual-mode-off",
        action="store_true",
        help=(
            "Skip dual-mode Anthropic edge test. Must be set before that test; "
            "equivalent to OMNI_TEST_DUAL_MODE_OFF=1. Never inferred from 4xx."
        ),
    )
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Run only these test names (use --list). Empty selection is an error.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List test names and exit",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        from tools.live_http_suite.suite import TEST_ORDER

        for name, _ in TEST_ORDER:
            print(name)
        return 0

    cfg = SuiteConfig.from_env(
        base_url=args.base_url,
        dual_mode_off=True if args.dual_mode_off else None,
    )
    # Distinguish "flag absent" (None → run all) from "flag present, empty" (error).
    only: set[str] | None
    if args.only is None:
        only = None
    else:
        only = set(args.only)
    print(f"Live HTTP suite → {cfg.base_url_normalized}")
    print(
        f"pins: claude={cfg.claude_model} dual={cfg.dual_mode_model} "
        f"responses={cfg.responses_model} dual_mode_off={cfg.dual_mode_off}"
    )
    outcomes = run_suite(cfg, only=only)
    return summarize(outcomes)


if __name__ == "__main__":
    raise SystemExit(main())
