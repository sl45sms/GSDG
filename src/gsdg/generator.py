import argparse
import fnmatch
import json
import logging
import os
from glob import glob
from pathlib import Path
from typing import Optional, Sequence

from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

from gsdg.openai_client import InferenceError, OpenAICompatibleClient, parse_qa_json
from gsdg.prompting import SYSTEM_PROMPT, build_chatml_record, build_user_prompt
from gsdg.text_extraction import extract_best_text, infer_row_id


LOGGER = logging.getLogger("gsdg")


def _trimmed_join(values: Sequence[str], max_items: int = 5) -> str:
    if len(values) <= max_items:
        return ", ".join(values)
    head = ", ".join(values[:max_items])
    return f"{head}, ... ({len(values)} total)"


def _dedupe_preserving_order(values: Sequence[str]) -> list[str]:
    unique_values = []
    seen_values = set()

    for value in values:
        if value in seen_values:
            continue
        unique_values.append(value)
        seen_values.add(value)

    return unique_values


def _match_hf_repo_files(repo_files: Sequence[str], pattern: str) -> list[str]:
    normalized_pattern = pattern.lstrip("/")
    if "/" in normalized_pattern:
        return sorted(
            repo_file
            for repo_file in repo_files
            if fnmatch.fnmatch(repo_file, normalized_pattern)
        )

    return sorted(
        repo_file
        for repo_file in repo_files
        if fnmatch.fnmatch(Path(repo_file).name, normalized_pattern)
    )


def resolve_local_parquet_files(patterns: Sequence[str]) -> list[str]:
    resolved_files = []

    for pattern in patterns:
        expanded_pattern = os.path.expanduser(pattern)
        matches = [match for match in glob(expanded_pattern, recursive=True) if Path(match).is_file()]
        if not matches:
            raise FileNotFoundError(f"No local parquet files matched: {pattern}")
        resolved_files.extend(sorted(matches))

    return _dedupe_preserving_order(resolved_files)


def resolve_hf_parquet_files(repo_id: str, patterns: Sequence[str], token: Optional[str]) -> list[str]:
    repo_files = HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquet_repo_files = [repo_file for repo_file in repo_files if repo_file.endswith(".parquet")]
    if not parquet_repo_files:
        raise FileNotFoundError(f"Dataset repo {repo_id} does not contain any parquet files")

    resolved_files = []
    for pattern in patterns:
        matches = _match_hf_repo_files(parquet_repo_files, pattern)
        if not matches:
            raise FileNotFoundError(
                f"No parquet files in dataset repo {repo_id} matched: {pattern}"
            )
        resolved_files.extend(matches)

    unique_files = _dedupe_preserving_order(resolved_files)
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=list(unique_files),
        token=token,
        resume_download=True,
    )
    return [str(Path(snapshot_path, repo_file)) for repo_file in unique_files]


def build_source_label(args: argparse.Namespace) -> str:
    if args.dataset:
        return args.dataset

    parquet_selector = ",".join(args.parquet_files)
    if args.hf_parquet_repo:
        return f"{args.hf_parquet_repo}::{parquet_selector}"
    return f"local::{parquet_selector}"


def load_input_rows(args: argparse.Namespace):
    if args.dataset:
        LOGGER.info("Loading dataset %s split=%s", args.dataset, args.split)
        dataset = load_dataset(args.dataset, split=args.split, token=args.hf_token)
        return build_source_label(args), dataset

    if args.hf_parquet_repo:
        parquet_files = resolve_hf_parquet_files(
            args.hf_parquet_repo,
            args.parquet_files,
            args.hf_token,
        )
    else:
        parquet_files = resolve_local_parquet_files(args.parquet_files)

    LOGGER.info(
        "Loading %s parquet file(s) for split=%s: %s",
        len(parquet_files),
        args.split,
        _trimmed_join(parquet_files),
    )
    dataset = load_dataset(
        "parquet",
        data_files={args.split: parquet_files},
        split=args.split,
        token=args.hf_token,
    )
    return build_source_label(args), dataset


def validate_source_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    has_dataset = bool(args.dataset)
    has_parquet_files = bool(args.parquet_files)

    if has_dataset == has_parquet_files:
        parser.error("Provide exactly one of --dataset or --parquet-file")
    if args.hf_parquet_repo and not has_parquet_files:
        parser.error("--hf-parquet-repo requires at least one --parquet-file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Greek ChatML Q/A pairs from a Hugging Face dataset or parquet input.",
    )
    parser.add_argument("--dataset", help="Hugging Face dataset name")
    parser.add_argument(
        "--hf-parquet-repo",
        help="Hugging Face dataset repo containing parquet files selected via --parquet-file",
    )
    parser.add_argument(
        "--parquet-file",
        dest="parquet_files",
        action="append",
        default=[],
        help=(
            "Parquet file path or glob. Repeat to combine multiple selections. "
            "With --hf-parquet-repo, patterns match repo-relative paths or basenames."
        ),
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to read, or logical split name to assign to parquet input",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token for gated or private dataset access",
    )
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

    source_label, dataset = load_input_rows(args)
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
                dataset_name=source_label,
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
    validate_source_args(parser, args)
    configure_logging(args.log_level)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
