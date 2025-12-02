#!/bin/bash
set -x

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# --- FIX 1: Correct Library Path Precedence ---
# We build the path string first, then export it once.
# We explicitly look for the symlink 'cuda' FIRST so specific versions override it later if prepended,
# OR we just add the specific version we want. Best practice: explicitly pick ONE.
# Current logic: Add newest detected version to the FRONT.

NEW_LD_PATH=""
# Find all cuda directories, sort them reverse version order (newest first)
CUDA_DIRS=$(ls -d /usr/local/cuda-12* 2>/dev/null | sort -r)

# Prepend newest CUDA 12 libs
for cuda_dir in $CUDA_DIRS; do
    if [ -d "$cuda_dir/lib64" ]; then
        NEW_LD_PATH="$cuda_dir/lib64:$NEW_LD_PATH"
    fi
done

# Add generic /usr/local/cuda last (lowest priority) if needed
if [ -d "/usr/local/cuda/lib64" ]; then
    NEW_LD_PATH="${NEW_LD_PATH}:/usr/local/cuda/lib64"
fi

# BlobLearn cuSPARSELt (Keep this if you are 100% sure VeOmni needs specifically THIS copy)
BLOBLEARN_CUSPARSELT="/home/ubuntu/BlobLearn/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib"
if [ -d "$BLOBLEARN_CUSPARSELT" ]; then
    NEW_LD_PATH="$BLOBLEARN_CUSPARSELT:$NEW_LD_PATH"
fi

export LD_LIBRARY_PATH="$NEW_LD_PATH:$LD_LIBRARY_PATH"


# --- Load .env ---
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
    export AWS_ACCESS_KEY_ID
    export AWS_SECRET_ACCESS_KEY
    export AWS_REGION
    export AWS_DEFAULT_REGION
else
    echo "Warning: No .env file found."
fi

# --- FIX 2: Activate Venv BEFORE uv sync ---
if [ -f "/home/ubuntu/VeOmni/.venv/bin/activate" ]; then
    source /home/ubuntu/VeOmni/.venv/bin/activate
else
    echo "Error: VeOmni virtualenv not found!"
    exit 1
fi

export PYTHONPATH="/home/ubuntu/VeOmni:$PYTHONPATH"

# --- FIX 3: Safer uv sync ---
# uv sync will PRUNE packages not in uv.lock. 
# If you modify the env manually, use 'uv pip install' instead.
# Assuming you want strict sync:
uv sync --frozen --extra gpu --extra audio
# If you DO NOT want strict sync (preserve manual installs), comment above and use:
# uv pip install -e .[gpu,audio]

NNODES=${NNODES:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=8}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}
export CUDA_LAUNCH_BLOCKING=0

if [[ "$NNODES" == "1" ]]; then
  additional_args="--standalone"
else
  additional_args="--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

# # Use the python explicitly from the currently active VIRTUAL_ENV
# uv run torchrun \
#   --nnodes=$NNODES \
#   --nproc-per-node=$NPROC_PER_NODE \
#   --node-rank=$NODE_RANK \
#   $additional_args \
#   tasks/omni/train_qwen2_vl.py \
#   configs/multimodal/qwen3_vl/qwen3_vl_8b.yaml \
#   2>&1 | tee veomni_compare.log
# # Use the python explicitly from the currently active VIRTUAL_ENV
uv run torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$NODE_RANK \
  $additional_args \
  tasks/omni/train_qwen2_vl.py \
  configs/multimodal/qwen3_vl/qwen3_vl_8b_sft.yaml \
  2>&1 | tee veomni_compare.log