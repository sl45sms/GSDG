"""Async batch curation for ChatML JSONL records.

The LLM review step is the only network call, so we batch rows together
and run reviews concurrently via aiohttp, keeping the vLLM server's
continuous-batching pipeline saturated.

All deterministic checks, dedup, and I/O operations remain fast synchronous
per-sample work.  The SQLite dedup store is protected by an asyncio lock so
concurrent review completions are serialised safely.

The pipeline is **streaming**: rows are read one at a time from the input
JSONL, deterministic checks run inline, and only ``batch_size`` rows are
held in memory at any point.  This means memory usage is O(batch_size +
concurrency), not O(total rows), and very large curation files are handled
without issue.

**Dedup runs before the LLM review** so exact- and near-duplicate samples
are skipped without spending GPU compute.  A second dedup check after the
LLM review catches any sample that became a near-duplicate while the review
was in-flight (rare).

Differences from the synchronous ``curate_jsonl.py``:
- Reads the JSONL once, streaming; never loads the full file into a list.
- Deterministic checks run inline per-row; rejects are written immediately.
- Exact- and near-duplicate checks run *before* LLM review, saving GPU time.
- Rows that pass dedup are queued and, once a batch fills, reviewed
  concurrently under an asyncio.Semaphore.
- Dedup recheck + write + checkpoint are serialised behind a single
  asyncio.Lock.
- Progress is reported per batch with a breakdown of skip reasons
  (exact dup, near dup, error).
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp

from gsdg.async_client import AsyncOpenAICompatibleClient
from gsdg.curate_jsonl import (
    CATEGORY_NAMES,
    DEFAULT_REVIEW_SYSTEM_PROMPT,
    DeduplicationStore,
    ParsedSample,
    ReviewDecision,
    TokenCounter,
    append_jsonl_line,
    atomic_write_json,
    build_dedupe_text,
    build_review_prompt,
    build_sample,
    deterministic_reject_labels,
    ensure_parent_directory,
    extract_json_object,
    heuristic_category,
    iter_jsonl_records,
    maybe_bootstrap_existing_outputs,
    parse_review_decision,
    sha256_text,
    should_skip_sample,
    summarize_rationale,
    write_reject_log,
)
from gsdg.openai_client import InferenceError

LOGGER = logging.getLogger("gsdg.async_curation")


# ---------------------------------------------------------------------------
# Transient-error detection (for retry logic)
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds, doubled each attempt


def _is_transient_error(exc: BaseException) -> bool:
    """Return True for network errors worth retrying."""
    if isinstance(
        exc,
        (
            aiohttp.ServerDisconnectedError,
            aiohttp.ClientConnectorError,
            aiohttp.ClientOSError,
            asyncio.TimeoutError,
        ),
    ):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return 500 <= exc.status < 600
    return False


# ---------------------------------------------------------------------------
# Async LLM review (with retries)
# ---------------------------------------------------------------------------


async def _run_llm_review_async(
    client: AsyncOpenAICompatibleClient,
    question: str,
    answer: str,
    review_max_tokens: int,
    review_temperature: float,
    semaphore: asyncio.Semaphore,
) -> ReviewDecision:
    """Async version of ``run_llm_review`` with transient-error retries.

    The *semaphore* is held for the entire review (prompt retries are fast
    enough that releasing between attempts would not meaningfully increase
    throughput).
    """
    prompts = [
        build_review_prompt(question, answer, strict=False),
        build_review_prompt(question, answer, strict=True),
    ]
    last_error: Optional[BaseException] = None

    async with semaphore:
        for user_prompt in prompts:
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    raw_response = await client.create_chat_completion(
                        system_prompt=DEFAULT_REVIEW_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        temperature=review_temperature,
                        max_tokens=review_max_tokens,
                        enable_thinking=False,
                    )
                    parsed = extract_json_object(raw_response)
                    review = parse_review_decision(parsed, question, answer)
                    return ReviewDecision(
                        accept=review.accept,
                        category=review.category,
                        reject_labels=review.reject_labels,
                        rationale=review.rationale,
                        used_llm=True,
                    )
                except InferenceError:
                    # Non-retryable – malformed model output; try the
                    # alternative prompt before giving up.
                    break
                except Exception as exc:
                    last_error = exc
                    if not _is_transient_error(exc) or attempt == _MAX_RETRIES:
                        break
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    LOGGER.debug(
                        "Review retry %s/%s in %.1fs: %s",
                        attempt,
                        _MAX_RETRIES,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    break  # success

    if last_error is None:
        raise InferenceError("LLM review failed without an explicit error")
    if isinstance(last_error, InferenceError):
        raise last_error
    raise InferenceError(f"LLM review failed: {last_error}") from last_error


# ---------------------------------------------------------------------------
# Per-sample processor (called concurrently within a batch)
# ---------------------------------------------------------------------------


async def _process_one_sample(
    sample: ParsedSample,
    client: Optional[AsyncOpenAICompatibleClient],
    semaphore: asyncio.Semaphore,
    args: object,
    dedup_lock: asyncio.Lock,
    store: DeduplicationStore,
    reject_log_path: Optional[Path],
    output_directory: Path,
) -> Tuple[str, Optional[str]]:
    """Dedup → LLM review → dedup-recheck → write for a single sample.

    Exact- and near-duplicate checks run **before** the LLM review so
    duplicate samples are skipped without wasting GPU compute.  A second
    dedup pass after the LLM review catches any sample that became a
    near-duplicate while the review was in-flight (rare).

    Returns ``(status, None)`` where *status* is one of
    ``"accepted"``, ``"rejected"``, ``"skipped_exact"``, ``"skipped_near"``,
    or ``"skipped_error"``.
    """
    input_path = Path(getattr(args, "input", ""))

    # ------------------------------------------------------------------
    # Phase A — fast dedup (exact + near) BEFORE any network call
    # ------------------------------------------------------------------
    normalized_text = build_dedupe_text(sample.question, sample.answer)
    text_hash = sha256_text(normalized_text)

    async with dedup_lock:
        if store.has_exact_duplicate(text_hash):
            store.update_checkpoint(input_path, sample.source_row_index, sample.line_number)
            store.connection.commit()
            LOGGER.debug(
                "Line %s (src_row=%s): exact duplicate — skipped",
                sample.line_number,
                sample.source_row_index,
            )
            return ("skipped_exact", None)

        near_duplicate = store.find_near_duplicate(normalized_text)
        if near_duplicate is not None:
            store.update_checkpoint(input_path, sample.source_row_index, sample.line_number)
            store.connection.commit()
            LOGGER.debug(
                "Line %s (src_row=%s): near-duplicate of sample_id %s — skipped",
                sample.line_number,
                sample.source_row_index,
                near_duplicate["id"],
            )
            return ("skipped_near", None)

    # ------------------------------------------------------------------
    # Phase B — LLM review (only for genuinely new content)
    # ------------------------------------------------------------------
    if client is not None:
        try:
            review = await _run_llm_review_async(
                client=client,
                question=sample.question,
                answer=sample.answer,
                review_max_tokens=getattr(args, "review_max_tokens", 256),
                review_temperature=getattr(args, "review_temperature", 0.0),
                semaphore=semaphore,
            )
        except InferenceError as exc:
            LOGGER.warning(
                "LLM review failed at line %s (source_row_index=%s); "
                "falling back to heuristic classification: %s",
                sample.line_number,
                sample.source_row_index,
                exc,
            )
            review = ReviewDecision(
                accept=True,
                category=heuristic_category(sample.question, sample.answer),
                reject_labels=[],
                rationale="LLM review fallback after malformed output",
                used_llm=False,
            )
    else:
        review = ReviewDecision(
            accept=True,
            category=heuristic_category(sample.question, sample.answer),
            reject_labels=[],
            rationale="Deterministic-only mode",
            used_llm=False,
        )

    # ------------------------------------------------------------------
    # Phase C — commit (reject log, dedup recheck, write output)
    # ------------------------------------------------------------------
    async with dedup_lock:
        if not review.accept:
            write_reject_log(
                reject_log_path=reject_log_path,
                sample=sample,
                reject_labels=review.reject_labels or ["REJECT_LOW_QUALITY"],
                rationale=review.rationale,
                used_llm=review.used_llm,
            )
            store.update_checkpoint(input_path, sample.source_row_index, sample.line_number)
            store.connection.commit()
            return ("rejected", None)

        # Re-check dedup — another coroutine may have added a near-duplicate
        # while this sample's LLM review was in-flight.
        if store.has_exact_duplicate(text_hash):
            store.update_checkpoint(input_path, sample.source_row_index, sample.line_number)
            store.connection.commit()
            return ("skipped_exact", None)

        near_duplicate = store.find_near_duplicate(normalized_text)
        if near_duplicate is not None:
            store.update_checkpoint(input_path, sample.source_row_index, sample.line_number)
            store.connection.commit()
            LOGGER.debug(
                "Line %s (src_row=%s): became near-duplicate of sample_id %s "
                "during LLM review — skipped",
                sample.line_number,
                sample.source_row_index,
                near_duplicate["id"],
            )
            return ("skipped_near", None)

        output_path = output_directory / f"{review.category}.jsonl"
        append_jsonl_line(output_path, sample.record)
        store.add_sample(
            normalized_text=normalized_text,
            text_hash=text_hash,
            category=review.category,
            source_row_index=sample.source_row_index,
            input_path=input_path,
            output_path=output_path,
        )
        store.update_checkpoint(input_path, sample.source_row_index, sample.line_number)
        store.connection.commit()

    return ("accepted", None)


# ---------------------------------------------------------------------------
# Batch flush helper
# ---------------------------------------------------------------------------


async def _flush_curation_batch(
    batch: List[ParsedSample],
    client: Optional[AsyncOpenAICompatibleClient],
    semaphore: asyncio.Semaphore,
    dedup_lock: asyncio.Lock,
    store: DeduplicationStore,
    reject_log_path: Optional[Path],
    output_directory: Path,
    input_path: Path,
    args: object,
) -> Tuple[int, int, int, int, int]:
    """Run concurrent processing for *batch*.

    Returns ``(accepted, rejected, skipped_exact, skipped_near, skipped_error)``.
    """
    tasks = [
        _process_one_sample(
            sample=sample,
            client=client,
            semaphore=semaphore,
            args=args,
            dedup_lock=dedup_lock,
            store=store,
            reject_log_path=reject_log_path,
            output_directory=output_directory,
        )
        for sample in batch
    ]

    # return_exceptions=True so one crashing sample doesn't kill the batch.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    accepted = rejected = 0
    skipped_exact = skipped_near = skipped_error = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            LOGGER.error(
                "Sample line %s failed: %s",
                batch[i].line_number,
                result,
            )
            skipped_error += 1
            store.update_checkpoint(
                input_path,
                batch[i].source_row_index,
                batch[i].line_number,
            )
            store.connection.commit()
        else:
            status, _ = result
            if status == "accepted":
                accepted += 1
            elif status == "rejected":
                rejected += 1
            elif status == "skipped_exact":
                skipped_exact += 1
            elif status == "skipped_near":
                skipped_near += 1
            else:
                skipped_error += 1

    return accepted, rejected, skipped_exact, skipped_near, skipped_error


# ---------------------------------------------------------------------------
# Main async driver (streaming — never loads the full JSONL into memory)
# ---------------------------------------------------------------------------


async def run_async(args) -> int:  # args: argparse.Namespace
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    state_db_path = (
        Path(args.state_db) if args.state_db else output_directory / ".curation_state.sqlite3"
    )
    ensure_parent_directory(state_db_path)

    reject_log_path = Path(args.reject_log) if args.reject_log else None
    if reject_log_path is not None:
        ensure_parent_directory(reject_log_path)

    token_counter = TokenCounter(args.tokenizer_model, args.hf_token)

    # ------------------------------------------------------------------
    # Phase 1 — init dedup store & checkpoint
    # ------------------------------------------------------------------
    store = DeduplicationStore(
        db_path=state_db_path,
        num_perm=args.num_perm,
        bands=args.bands,
        threshold=args.near_duplicate_threshold,
    )
    maybe_bootstrap_existing_outputs(store, output_directory)

    last_source_row_index, last_line_number = store.load_checkpoint(input_path)
    LOGGER.info(
        "Resuming from source_row_index=%s line=%s",
        last_source_row_index,
        last_line_number,
    )

    # ------------------------------------------------------------------
    # Phase 2 — session (aiohttp, client, concurrency controls)
    # ------------------------------------------------------------------
    concurrency = getattr(args, "curation_concurrency", 16)
    batch_size = getattr(args, "curation_batch_size", 200)

    connector = aiohttp.TCPConnector(
        limit=concurrency + 16,
        limit_per_host=concurrency + 16,
        ttl_dns_cache=300,
        keepalive_timeout=60,
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        client = None
        if not args.disable_llm_review:
            client = AsyncOpenAICompatibleClient(
                session=session,
                api_base=args.api_base,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                api_key=args.api_key,
            )
            if not args.skip_healthcheck:
                LOGGER.info("Running inference server health check")
                await client.healthcheck()

        semaphore = asyncio.Semaphore(concurrency)
        dedup_lock = asyncio.Lock()

        # ------------------------------------------------------------------
        # Phase 3 — streaming loop: read → deterministic filter → async
        #           review batch → repeat.  Memory proportional to
        #           batch_size + concurrency, not to total file size.
        # ------------------------------------------------------------------
        pending_batch: List[ParsedSample] = []
        accepted = 0
        rejected_deterministic = 0
        rejected_llm = 0
        skipped_exact = 0
        skipped_near = 0
        skipped_error = 0
        skipped_no_qa = 0
        rows_seen = 0
        rows_skipped_checkpoint = 0
        total_start = time.monotonic()
        batch_start = total_start

        for line_number, record in iter_jsonl_records(input_path):
            rows_seen += 1

            sample = build_sample(record, line_number)
            if sample is None:
                skipped_no_qa += 1
                store.update_checkpoint(input_path, None, line_number)
                store.connection.commit()
                LOGGER.warning(
                    "Skipping line %s because no question/answer pair was found",
                    line_number,
                )
                continue

            # -- honour checkpoint (skip already-processed rows) ----------
            if should_skip_sample(sample, last_source_row_index, last_line_number):
                rows_skipped_checkpoint += 1
                continue

            # -- deterministic checks (fast, sync, no network) ------------
            reject_labels = deterministic_reject_labels(
                question=sample.question,
                answer=sample.answer,
                token_counter=token_counter,
                min_answer_words=args.min_answer_words,
                max_total_tokens=args.max_total_tokens,
                max_word_ratio=args.max_word_ratio,
            )

            if reject_labels:
                write_reject_log(
                    reject_log_path=reject_log_path,
                    sample=sample,
                    reject_labels=reject_labels,
                    rationale=summarize_rationale(reject_labels),
                    used_llm=False,
                )
                store.update_checkpoint(input_path, sample.source_row_index, line_number)
                store.connection.commit()
                rejected_deterministic += 1
                continue

            # -- queue for concurrent processing --------------------------
            pending_batch.append(sample)

            if len(pending_batch) >= batch_size:
                a, r, se, sn, serr = await _flush_curation_batch(
                    batch=pending_batch,
                    client=client,
                    semaphore=semaphore,
                    dedup_lock=dedup_lock,
                    store=store,
                    reject_log_path=reject_log_path,
                    output_directory=output_directory,
                    input_path=input_path,
                    args=args,
                )
                accepted += a
                rejected_llm += r
                skipped_exact += se
                skipped_near += sn
                skipped_error += serr

                # -- progress ---------------------------------------------
                now = time.monotonic()
                batch_elapsed = now - batch_start
                total_elapsed = now - total_start
                total_skipped = skipped_exact + skipped_near + skipped_error + skipped_no_qa
                done = accepted + rejected_llm + rejected_deterministic + total_skipped
                rate = done / total_elapsed if total_elapsed > 0 else 0.0
                LOGGER.info(
                    "rows %s-%s  accepted=%s  rej_det=%s  rej_llm=%s  "
                    "skip_exact=%s  skip_near=%s  skip_err=%s  "
                    "batch_t=%.1fs  total_done=%s  rate=%.1f row/s",
                    pending_batch[0].line_number,
                    pending_batch[-1].line_number,
                    a,
                    rejected_deterministic,
                    r,
                    se,
                    sn,
                    serr,
                    batch_elapsed,
                    done,
                    rate,
                )
                pending_batch.clear()
                batch_start = time.monotonic()

        # -- final (partial) batch ------------------------------------------
        if pending_batch:
            a, r, se, sn, serr = await _flush_curation_batch(
                batch=pending_batch,
                client=client,
                semaphore=semaphore,
                dedup_lock=dedup_lock,
                store=store,
                reject_log_path=reject_log_path,
                output_directory=output_directory,
                input_path=input_path,
                args=args,
            )
            accepted += a
            rejected_llm += r
            skipped_exact += se
            skipped_near += sn
            skipped_error += serr

    # ------------------------------------------------------------------
    # Phase 4 — summary
    # ------------------------------------------------------------------
    total_rejected = rejected_deterministic + rejected_llm
    total_skipped = skipped_exact + skipped_near + skipped_error + skipped_no_qa
    summary_path = output_directory / ".curation_summary.json"
    last_src, last_ln = store.load_checkpoint(input_path)
    atomic_write_json(
        summary_path,
        {
            "accepted": accepted,
            "rejected": total_rejected,
            "rejected_deterministic": rejected_deterministic,
            "rejected_llm": rejected_llm,
            "skipped_exact": skipped_exact,
            "skipped_near": skipped_near,
            "skipped_error": skipped_error,
            "skipped_no_qa": skipped_no_qa,
            "rows_seen": rows_seen,
            "rows_skipped_checkpoint": rows_skipped_checkpoint,
            "input": str(input_path.resolve()),
            "last_checkpoint": {
                "source_row_index": last_src,
                "line_number": last_ln,
            },
            "state_db": str(state_db_path.resolve()),
        },
    )
    store.close()

    total_elapsed = time.monotonic() - total_start
    LOGGER.info(
        "Finished async curation: accepted=%s rejected=%s (deterministic=%s llm=%s) "
        "skipped=%s (exact=%s near=%s err=%s no_qa=%s) "
        "rows_seen=%s elapsed=%.1fs out_dir=%s",
        accepted,
        total_rejected,
        rejected_deterministic,
        rejected_llm,
        total_skipped,
        skipped_exact,
        skipped_near,
        skipped_error,
        skipped_no_qa,
        rows_seen,
        total_elapsed,
        output_directory,
    )
    return 0
