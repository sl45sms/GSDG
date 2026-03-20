import argparse
import logging
import os

from datasets import load_dataset
from huggingface_hub import snapshot_download


LOGGER = logging.getLogger("gsdg.prefetch")


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
        resume_download=True,
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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
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
            prefetch_dataset(dataset_name, args.split, token)

    LOGGER.info("Prefetch completed successfully")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
