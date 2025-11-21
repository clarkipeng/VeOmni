#!/bin/bash
set -x

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# Add CUDA libraries to LD_LIBRARY_PATH
for cuda_dir in /usr/local/cuda-12* /usr/local/cuda; do
    if [ -d "$cuda_dir/targets/x86_64-linux/lib" ]; then
        export LD_LIBRARY_PATH="$cuda_dir/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
    fi
    if [ -d "$cuda_dir/lib64" ]; then
        export LD_LIBRARY_PATH="$cuda_dir/lib64:$LD_LIBRARY_PATH"
    fi
done

# Add BlobLearn's bundled cuSPARSELt library to LD_LIBRARY_PATH
# BlobLearn's PyTorch includes this library in its venv
BLOBLEARN_CUSPARSELT="/home/ubuntu/BlobLearn/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib"
if [ -d "$BLOBLEARN_CUSPARSELT" ]; then
    export LD_LIBRARY_PATH="$BLOBLEARN_CUSPARSELT:$LD_LIBRARY_PATH"
fi

# Load .env file if it exists and export all variables
# Check multiple locations for .env file
ENV_FILE=""
for path in "/home/ubuntu/.env" "/home/ubuntu/VeOmni/.env"; do
    if [ -f "$path" ]; then
        ENV_FILE="$path"
        break
    fi
done

if [ -n "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    # Export AWS credentials explicitly for child processes
    export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-}
    export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-}
    export AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN:-}
    export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
    echo "Loaded .env file from $ENV_FILE and exported AWS credentials"
else
    echo "Warning: No .env file found. AWS credentials may not be available."
fi

# Use BlobLearn's venv which has torch installed
if [ -f "/home/ubuntu/VeOmni/.venv/bin/activate" ]; then
    source /home/ubuntu/VeOmni/.venv/bin/activate
fi

# Add VeOmni to Python path
export PYTHONPATH="/home/ubuntu/VeOmni:$PYTHONPATH"

NNODES=${NNODES:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=4}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}
export CUDA_LAUNCH_BLOCKING=0

if [[ "$NNODES" == "1" ]]; then
  additional_args="--standalone"
else
  additional_args="--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi
uv sync --extra gpu --extra audio

# Use venv's Python directly to ensure flash_attn is available
$VIRTUAL_ENV/bin/python -m torch.distributed.run \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$NODE_RANK \
  $additional_args \
  tasks/omni/train_qwen2_vl.py \
  configs/multimodal/qwen3_vl/qwen3_vl_2b_compare.yaml \
  2>&1 | tee veomni_compare.log
