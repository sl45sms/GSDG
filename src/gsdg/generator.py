import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from gsdg.openai_client import InferenceError, OpenAICompatibleClient, parse_qa_json
from gsdg.prompting import SYSTEM_PROMPT, build_chatml_record, build_user_prompt
from gsdg.text_extraction import extract_best_text, infer_row_id


LOGGER = logging.getLogger("gsdg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Greek ChatML Q/A pairs from a HuggingFace dataset.",
    )
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--split", default="train", help="Dataset split to read")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-source-chars", type=int, default=4000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Allow the model to emit thinking traces instead of forcing non-thinking mode",
    )
    parser.add_argument(
        "--skip-healthcheck",
        action="store_true",
        help="Skip the pre-run API health check",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_selected_rows(dataset, start_row: int, max_rows: Optional[int]):
    stop_row = None if max_rows is None else start_row + max_rows
    for row_index, row in enumerate(dataset):
        if row_index < start_row:
            continue
        if stop_row is not None and row_index >= stop_row:
            break
        yield row_index, row


def run(args: argparse.Namespace) -> int:
    output_path = Path(args.out)
    ensure_parent_directory(output_path)

    client = OpenAICompatibleClient(
        api_base=args.api_base,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        api_key=args.api_key,
    )
    if not args.skip_healthcheck:
        LOGGER.info("Running inference server health check")
        client.healthcheck()

    LOGGER.info("Loading dataset %s split=%s", args.dataset, args.split)
    dataset = load_dataset(args.dataset, split=args.split)
    total_rows = len(dataset)
    LOGGER.info("Loaded %s rows with columns: %s", total_rows, list(dataset.column_names))

    written = 0
    skipped = 0

    with output_path.open("a", encoding="utf-8") as handle:
        progress = tqdm(iter_selected_rows(dataset, args.start_row, args.max_rows), total=args.max_rows)
        for row_index, row in progress:
            progress.set_description(f"row {row_index}")
            try:
                source_fields, source_text = extract_best_text(row, args.max_source_chars)
            except ValueError as exc:
                skipped += 1
                LOGGER.warning("Skipping row %s: %s", row_index, exc)
                continue

            row_id = infer_row_id(row, row_index)
            user_prompt = build_user_prompt(source_text)

            try:
                raw_response = client.create_chat_completion(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    enable_thinking=args.enable_thinking,
                )
                qa_pair = parse_qa_json(raw_response)
            except InferenceError as exc:
                skipped += 1
                LOGGER.warning("Skipping row %s after inference error: %s", row_index, exc)
                continue
            except Exception as exc:  # pragma: no cover - defensive fallback for batch jobs
                skipped += 1
                LOGGER.exception("Skipping row %s after unexpected error: %s", row_index, exc)
                continue

            record = build_chatml_record(
                user_prompt=user_prompt,
                question=qa_pair["question"],
                answer=qa_pair["answer"],
                dataset_name=args.dataset,
                split_name=args.split,
                row_id=row_id,
                source_fields=source_fields,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    LOGGER.info("Finished generation: written=%s skipped=%s out=%s", written, skipped, output_path)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
