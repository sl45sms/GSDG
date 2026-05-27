import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple


LOGGER = logging.getLogger("gsdg.combine_jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine two or more JSONL files into a single JSONL output.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input JSONL files to combine in the given order",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Skip duplicate records that have the same question/answer pair",
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


def validate_paths(input_paths: List[Path], output_path: Path) -> None:
    if len(input_paths) < 2:
        raise ValueError("Provide at least two input JSONL files.")

    missing_paths = [str(path) for path in input_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Input file(s) not found: {', '.join(missing_paths)}")

    resolved_output = output_path.resolve()
    conflicting_inputs = [str(path) for path in input_paths if path.resolve() == resolved_output]
    if conflicting_inputs:
        raise ValueError(
            "Output path must be different from every input path: "
            f"{', '.join(conflicting_inputs)}"
        )


def normalize_text(value: Any) -> str:
    return str(value).strip()


def extract_question_answer_pair(record: Any) -> Optional[Tuple[str, str]]:
    if not isinstance(record, dict):
        return None

    question = record.get("question")
    answer = record.get("answer")
    if question is not None and answer is not None:
        return normalize_text(question), normalize_text(answer)

    messages = record.get("messages")
    if not isinstance(messages, list):
        return None

    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue

        content = message.get("content")
        if isinstance(content, dict):
            payload = content
        elif isinstance(content, str):
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
        else:
            continue

        if not isinstance(payload, dict):
            continue

        question = payload.get("question")
        answer = payload.get("answer")
        if question is not None and answer is not None:
            return normalize_text(question), normalize_text(answer)

    return None


def build_dedupe_key(record: Any) -> Optional[str]:
    qa_pair = extract_question_answer_pair(record)
    if qa_pair is None:
        return None

    question, answer = qa_pair
    return "qa::" + json.dumps(
        {"question": question, "answer": answer},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def renumber_row_id(record: Any, row_index: int) -> Any:
    if not isinstance(record, dict):
        return record

    meta = record.get("meta")
    if not isinstance(meta, dict):
        return record

    meta["row_id"] = str(row_index)
    return record


def combine_files(input_paths: List[Path], output_path: Path, dedupe: bool) -> Tuple[int, int]:
    ensure_parent_directory(output_path)

    seen_keys: Set[str] = set()
    written = 0
    skipped = 0

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f"{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp_handle:
        temp_path = Path(tmp_handle.name)
        try:
            for input_path in input_paths:
                LOGGER.info("Reading %s", input_path)
                with input_path.open("r", encoding="utf-8") as source_handle:
                    for line_number, raw_line in enumerate(source_handle, start=1):
                        line = raw_line.strip()
                        if not line:
                            continue

                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Invalid JSON in {input_path} at line {line_number}: {exc.msg}"
                            ) from exc

                        if dedupe:
                            dedupe_key = build_dedupe_key(record)
                            if dedupe_key is not None:
                                if dedupe_key in seen_keys:
                                    skipped += 1
                                    continue
                                seen_keys.add(dedupe_key)

                        record = renumber_row_id(record, written)
                        tmp_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    temp_path.replace(output_path)
    return written, skipped


def run(args: argparse.Namespace) -> int:
    input_paths = [Path(path) for path in args.inputs]
    output_path = Path(args.out)

    validate_paths(input_paths, output_path)
    written, skipped = combine_files(input_paths, output_path, args.dedupe)

    LOGGER.info(
        "Finished combining %s file(s): written=%s skipped=%s out=%s",
        len(input_paths),
        written,
        skipped,
        output_path,
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())