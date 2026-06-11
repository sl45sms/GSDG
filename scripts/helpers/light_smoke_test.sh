#!/bin/bash
sbatch --export=ALL,MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct,TENSOR_PARALLEL_SIZE=1,MAX_MODEL_LEN=2048,REASONING_PARSER= scripts/smoke_test_vllm_qwen3.sh