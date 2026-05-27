import argparse
import fnmatch
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

from datasets import load_dataset
from datasets.exceptions import DataFilesNotFoundError
from huggingface_hub import HfApi, snapshot_download


LOGGER = logging.getLogger("gsdg.prefetch")


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


def resolve_hf_parquet_repo_files(repo_id: str, patterns: Sequence[str], token: Optional[str]) -> list[str]:
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

    unique_files = []
    seen_files = set()
    for repo_file in resolved_files:
        if repo_file in seen_files:
            continue
        unique_files.append(repo_file)
        seen_files.add(repo_file)

    return unique_files


def default_parquet_out_dir(repo_id: str) -> str:
    scratch_root = os.environ.get("SCRATCH")
    base_dir = Path(scratch_root) if scratch_root else Path.cwd()
    return str(base_dir / "gsdg_parquet_prefetch" / repo_id.replace("/", "__"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prefetch HuggingFace model and dataset assets into persistent caches.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-397B-A17B",
        help="Model repository to prefetch",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset to prefetch. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to materialize for each dataset argument",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF token; defaults to HF_TOKEN from the environment",
    )
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
            "Parquet file path or glob inside --hf-parquet-repo. Repeat to combine multiple selections. "
            "Patterns match repo-relative paths or basenames."
        ),
    )
    parser.add_argument(
        "--parquet-out-dir",
        default=None,
        help="Target directory for prefetched parquet files from --hf-parquet-repo",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model prefetching",
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip dataset prefetching",
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


def prefetch_model(model: str, revision: str, token: str) -> None:
    LOGGER.info("Prefetching model %s", model)
    snapshot_path = snapshot_download(
        repo_id=model,
        revision=revision,
        token=token,
        repo_type="model",
    )
    LOGGER.info("Model %s cached at %s", model, snapshot_path)


def prefetch_dataset(dataset_name: str, split_name: str, token: str) -> None:
    LOGGER.info("Prefetching dataset %s split=%s", dataset_name, split_name)
    dataset = load_dataset(dataset_name, split=split_name, token=token)
    LOGGER.info(
        "Dataset %s cached with %s rows and columns=%s",
        dataset_name,
        len(dataset),
        list(dataset.column_names),
    )


def prefetch_parquet_repo(
    repo_id: str,
    parquet_files: Sequence[str],
    token: Optional[str],
    out_dir: Optional[str],
) -> None:
    resolved_files = resolve_hf_parquet_repo_files(repo_id, parquet_files, token)
    target_dir = out_dir or default_parquet_out_dir(repo_id)
    LOGGER.info(
        "Prefetching %s parquet file(s) from %s into %s",
        len(resolved_files),
        repo_id,
        target_dir,
    )
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=resolved_files,
        token=token,
        local_dir=target_dir,
    )
    LOGGER.info("Parquet prefetch completed: %s", target_dir)


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.hf_parquet_repo and not args.parquet_files:
        parser.error("--hf-parquet-repo requires at least one --parquet-file")
    if args.parquet_files and not args.hf_parquet_repo:
        parser.error("--parquet-file requires --hf-parquet-repo")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    configure_logging(args.log_level)

    token = args.token or os.environ.get("HF_TOKEN")
    LOGGER.info(
        "Prefetch config: skip_model=%s skip_datasets=%s hf_home=%s hf_datasets_cache=%s hf_hub_disable_xet=%s",
        args.skip_model,
        args.skip_datasets,
        os.environ.get("HF_HOME"),
        os.environ.get("HF_DATASETS_CACHE"),
        os.environ.get("HF_HUB_DISABLE_XET", "0"),
    )

    if not args.skip_model:
        prefetch_model(args.model, args.revision, token)

    if not args.skip_datasets:
        if not args.dataset:
            LOGGER.warning("No datasets requested; skipping dataset prefetch")
        for dataset_name in args.dataset:
            try:
                prefetch_dataset(dataset_name, args.split, token)
            except DataFilesNotFoundError as exc:
                # Some repos exist on the Hub but only contain documentation,
                # not actual dataset files (e.g. just README/DATASET_ACCESS).
                # Treat as a warning so other datasets still get prefetched.
                LOGGER.warning(
                    "Skipping dataset %s: no supported data files found (%s)",
                    dataset_name,
                    exc,
                )
            except Exception:
                # Keep prefetch best-effort across multiple datasets.
                LOGGER.exception("Failed to prefetch dataset %s", dataset_name)

    if args.hf_parquet_repo:
        prefetch_parquet_repo(
            repo_id=args.hf_parquet_repo,
            parquet_files=args.parquet_files,
            token=token,
            out_dir=args.parquet_out_dir,
        )

    LOGGER.info("Prefetch completed successfully")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
