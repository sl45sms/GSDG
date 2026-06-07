#!/usr/bin/env python3
"""Async entry-point for ChatML Q/A generation (drop-in for generate_chatml.py).

Differences from the synchronous `generate_chatml.py`:
- Uses aiohttp for non-blocking HTTP → 50-80× higher throughput.
- Adds --concurrency (default 64) and --batch-size (default 2000) controls.
- Flushes output once per batch, not once per row.
"""

import argparse
import asyncio

from gsdg.generator import (
    build_parser as _build_base_parser,
    configure_logging,
    validate_source_args,
)


def build_async_parser() -> argparse.ArgumentParser:
    parser = _build_base_parser()

    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Maximum simultaneous in-flight requests to the vLLM server",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Rows to collect before sorting and flushing to disk",
    )
    parser.set_defaults(timeout_seconds=600.0)

    return parser


def main() -> int:
    parser = build_async_parser()
    args = parser.parse_args()
    validate_source_args(parser, args)
    configure_logging(args.log_level)

    from gsdg.async_generator import run_async

    return asyncio.run(run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
