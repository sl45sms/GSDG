#!/usr/bin/env python3

import argparse
import os
import sys
from importlib.util import find_spec

import ray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--reasoning-parser", default="")
    parser.add_argument("--language-model-only", action="store_true")
    return parser.parse_args()


def maybe_patch_qwen35_gdn_prefill(model_name: str) -> None:
    if "Qwen3.5" not in model_name:
        return
    if find_spec("flashinfer") is not None:
        return

    import vllm.model_executor.models.qwen3_next as qwen3_next

    def patched_init(self) -> None:
        qwen3_next.CustomOp.__init__(self)
        qwen3_next.logger.warning(
            "flashinfer is not installed; forcing native GDN prefill kernel."
        )
        self._forward_method = self.forward_native

    qwen3_next.ChunkGatedDeltaRule.__init__ = patched_init


def main() -> None:
    args = parse_args()

    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    ray.init(address=args.ray_address, logging_level="ERROR")
    maybe_patch_qwen35_gdn_prefill(args.model)

    cli_args = [
        "vllm",
        "serve",
        args.model,
        "--distributed-executor-backend",
        "ray",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(args.pipeline_parallel_size),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    if args.reasoning_parser:
        cli_args.extend(["--reasoning-parser", args.reasoning_parser])
    if args.language_model_only:
        cli_args.append("--language-model-only")

    sys.argv = cli_args
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()