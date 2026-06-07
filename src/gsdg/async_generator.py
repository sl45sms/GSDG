"""Async batch generator for ChatML Q/A pairs.

Processes a dataset in batches of `--batch-size` rows.  Within each batch
every row is processed concurrently (throttled by `--concurrency`) via an
async HTTP client, keeping the vLLM server's continuous-batching pipeline
saturated.

Output is written in row-index order within each batch so resumability
semantics are preserved.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp

from gsdg.async_client import AsyncOpenAICompatibleClient
from gsdg.openai_client import InferenceError, parse_qa_json
from gsdg.prompting import SYSTEM_PROMPT, build_chatml_record, build_user_prompt
from gsdg.text_extraction import extract_best_text, infer_row_id

LOGGER = logging.getLogger("gsdg.async")

# ---------------------------------------------------------------------------
# Fast JSON – use orjson when available (3-5× faster than stdlib json)
# ---------------------------------------------------------------------------
try:
    import orjson as _orjson

    def _json_dumps(obj: object) -> str:
        return _orjson.dumps(obj).decode("utf-8")

    LOGGER.debug("Using orjson for JSON serialization.")
except ImportError:
    import json as _json

    def _json_dumps(obj: object) -> str:
        return _json.dumps(obj, ensure_ascii=False)

    LOGGER.debug("orjson not available; using stdlib json.")


# ---------------------------------------------------------------------------
# Per-row async worker
# ---------------------------------------------------------------------------

async def _process_one_row(
    row_index: int,
    row: dict,
    client: AsyncOpenAICompatibleClient,
    semaphore: asyncio.Semaphore,
    source_label: str,
    max_source_chars: int,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
) -> Tuple[int, Optional[dict]]:
    """Extract text, call the API, parse, and return a ChatML record (or None).

    The *semaphore* bounds the number of simultaneous in-flight requests so
    we do not overwhelm the server or local networking.
    """
    async with semaphore:
        # -- text extraction (cpu-only, very fast) --------------------------
        try:
            source_fields, source_text = extract_best_text(row, max_source_chars)
        except ValueError:
            LOGGER.warning("Skipping row %s: no usable text fields", row_index)
            return (row_index, None)

        row_id = infer_row_id(row, row_index)
        user_prompt = build_user_prompt(source_text)

        # -- inference ------------------------------------------------------
        try:
            raw = await client.create_chat_completion(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            qa_pair = parse_qa_json(raw)
        except InferenceError as exc:
            LOGGER.warning(
                "Skipping row %s after inference error: %s", row_index, exc
            )
            return (row_index, None)
        except Exception:
            LOGGER.exception(
                "Skipping row %s after unexpected inference error", row_index
            )
            return (row_index, None)

        # -- assemble record ------------------------------------------------
        record = build_chatml_record(
            user_prompt=user_prompt,
            question=qa_pair["question"],
            answer=qa_pair["answer"],
            dataset_name=source_label,
            split_name="train",  # parquet inputs always use "train" split
            row_id=row_id,
            source_row_index=row_index,
            source_fields=source_fields,
        )
        return (row_index, record)


# ---------------------------------------------------------------------------
# Main async driver
# ---------------------------------------------------------------------------

async def run_async(args) -> int:  # args: argparse.Namespace
    # -- output file --------------------------------------------------------
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -- aiohttp session (connection pooling) -------------------------------
    connector = aiohttp.TCPConnector(
        limit=args.concurrency + 16,
        limit_per_host=args.concurrency + 16,
        ttl_dns_cache=300,
        keepalive_timeout=60,
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
    ) as session:
        client = AsyncOpenAICompatibleClient(
            session=session,
            api_base=args.api_base,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            api_key=args.api_key,
        )

        # -- healthcheck ----------------------------------------------------
        if not args.skip_healthcheck:
            LOGGER.info("Running inference server health check")
            await client.healthcheck()

        # -- load dataset ---------------------------------------------------
        from gsdg.generator import load_input_rows

        source_label, dataset = load_input_rows(args)
        total_rows = len(dataset)
        LOGGER.info(
            "Loaded %s rows  columns=%s",
            total_rows,
            list(dataset.column_names),
        )

        # -- effective row window -------------------------------------------
        effective_end = (
            total_rows
            if args.max_rows is None
            else min(args.start_row + args.max_rows, total_rows)
        )
        effective_total = effective_end - args.start_row
        LOGGER.info(
            "Window: rows [%s, %s) = %s  concurrency=%s  batch_size=%s",
            args.start_row,
            effective_end,
            effective_total,
            args.concurrency,
            args.batch_size,
        )

        semaphore = asyncio.Semaphore(args.concurrency)

        # -- process in batches (ordered output) ----------------------------
        written = 0
        skipped = 0
        total_start = time.monotonic()
        batch_start = total_start

        with output_path.open("a", encoding="utf-8") as fh:
            batch_rows: List[Tuple[int, dict]] = []

            for row_index, row in enumerate(dataset):
                if row_index < args.start_row:
                    continue
                if row_index >= effective_end:
                    break

                batch_rows.append((row_index, row))

                if len(batch_rows) >= args.batch_size:
                    w, s = await _flush_batch(
                        batch_rows,
                        client,
                        semaphore,
                        source_label,
                        args,
                        fh,
                    )
                    written += w
                    skipped += s

                    # -- progress -------------------------------------------
                    now = time.monotonic()
                    batch_elapsed = now - batch_start
                    total_elapsed = now - total_start
                    done = written + skipped
                    rate = done / total_elapsed if total_elapsed > 0 else 0.0
                    eta_h = (
                        (effective_total - done) / rate / 3600.0
                        if rate > 0
                        else float("inf")
                    )

                    LOGGER.info(
                        "batch %s-%s  wrote=%s  skip=%s  "
                        "batch_t=%.1fs  total=%s/%s  rate=%.1f row/s  ETA=%.1fh",
                        batch_rows[0][0],
                        batch_rows[-1][0],
                        w,
                        s,
                        batch_elapsed,
                        done,
                        effective_total,
                        rate,
                        eta_h,
                    )
                    batch_rows.clear()
                    batch_start = time.monotonic()

            # -- final (partial) batch --------------------------------------
            if batch_rows:
                w, s = await _flush_batch(
                    batch_rows,
                    client,
                    semaphore,
                    source_label,
                    args,
                    fh,
                )
                written += w
                skipped += s

        # -- summary --------------------------------------------------------
        total_elapsed = time.monotonic() - total_start
        done = written + skipped
        rate = done / total_elapsed if total_elapsed > 0 else 0.0
        LOGGER.info(
            "DONE  written=%s  skipped=%s  elapsed=%.1fs  rate=%.1f row/s  → %s",
            written,
            skipped,
            total_elapsed,
            rate,
            output_path,
        )

    return 0


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

async def _flush_batch(
    batch_rows: List[Tuple[int, dict]],
    client: AsyncOpenAICompatibleClient,
    semaphore: asyncio.Semaphore,
    source_label: str,
    args,
    fh,
) -> Tuple[int, int]:
    """Fire every row in *batch_rows* concurrently, sort by row_index, write."""
    tasks = [
        _process_one_row(
            row_index=idx,
            row=row,
            client=client,
            semaphore=semaphore,
            source_label=source_label,
            max_source_chars=args.max_source_chars,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
        )
        for idx, row in batch_rows
    ]

    results = await asyncio.gather(*tasks)

    # Sort so the output file stays ordered by row_index.
    results.sort(key=lambda r: r[0])

    written = 0
    skipped = 0
    for _, record in results:
        if record is None:
            skipped += 1
            continue
        fh.write(_json_dumps(record) + "\n")
        written += 1

    # Flush once per batch (not per row!).
    fh.flush()

    return written, skipped
