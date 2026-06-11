#!/usr/bin/env python3
"""Async entry-point for JSONL curation (drop-in for curate_jsonl.py).

Differences from the synchronous ``curate_jsonl.py``:
- Uses aiohttp for non-blocking HTTP → LLM reviews run concurrently.
- Adds --curation-concurrency (default 16) and --curation-batch-size (default 200).
- Runs deterministic checks up-front so only samples requiring LLM review
  consume concurrency slots.
- Progress is reported per batch, not per row.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the repo src is on sys.path before importing gsdg modules.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    # Reuse the full sync-curation parser as our base so we automatically
    # pick up any new CLI knobs added to curate_jsonl.py.
    from gsdg.curate_jsonl import build_parser as _build_base_parser

    parser = _build_base_parser()

    # -- async-specific knobs ------------------------------------------------
    parser.add_argument(
        "--curation-concurrency",
        type=int,
        default=16,
        help="Maximum simultaneous in-flight LLM review requests (default: 16)",
    )
    parser.add_argument(
        "--curation-batch-size",
        type=int,
        default=200,
        help="Rows to collect before reporting progress (default: 200)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    from gsdg.curate_jsonl import configure_logging, validate_args

    configure_logging(args.log_level)
    validate_args(args)

    from gsdg.async_curation import run_async

    return asyncio.run(run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
