#!/bin/bash
set -e

curl -LsSf https://astral.sh/uv/0.9.8/install.sh | sh
source $HOME/.local/bin/env
uv sync --extra gpu

# Source .env for AWS credentials
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Install unzip if not present
if ! command -v unzip &> /dev/null; then
    echo "Installing unzip..."
    sudo apt-get update && sudo apt-get install -y unzip
fi

echo "Setting up datasets..."

wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json -P sharegpt4v_cap_100k/

# 1. COCO
if [ ! -d "coco/train2017" ]; then
    echo "Processing COCO..."
    if [ ! -f "coco/train2017.zip" ]; then
        echo "Downloading COCO train2017 (18GB)..."
        mkdir -p coco
        wget -c http://images.cocodataset.org/zips/train2017.zip -O coco/train2017.zip
    fi
    echo "Extracting COCO..."
    unzip -q coco/train2017.zip -d coco/
    echo "Cleaning up..."
    rm coco/train2017.zip
else
    echo "COCO train2017 already exists."
fi

# 2. SAM
if [ ! -d "sam/images" ]; then
    echo "Processing SAM..."
    mkdir -p sam
    if [ ! -f "sam/sam_images.zip" ]; then
        echo "Downloading SAM images from Google Drive..."
        # Use gdown to download from Google Drive
        uv pip install -q gdown
        uv run gdown "https://drive.google.com/uc?id=1dKumdOKSXtV7lIXdrG7jsIK_z2vZv2gs" -O sam/sam_images.zip
    fi
    echo "Extracting SAM..."
    unzip -q sam/sam_images.zip -d sam/
    
    # Check if we need to restructure
    if [ -d "sam/sam_images_share-sft" ]; then
        mv sam/sam_images_share-sft sam/images
    elif [ ! -d "sam/images" ]; then
        # If images are directly in sam/, move them to sam/images
        mkdir -p sam/images
        mv sam/*.jpg sam/images/ 2>/dev/null || true
    fi
    echo "Cleaning up..."
    rm sam/sam_images.zip
else
    echo "SAM images already exist."
fi

# 3. LLaVA
if [ ! -d "llava/llava_pretrain/images" ]; then
    echo "Processing LLaVA..."
    mkdir -p llava/llava_pretrain
    if [ ! -f "llava/llava_pretrain/images.zip" ]; then
        echo "Downloading LLaVA pretrain images (25GB)..."
        wget -c https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip -O llava/llava_pretrain/images.zip
    fi
    echo "Extracting LLaVA..."
    unzip -q llava/llava_pretrain/images.zip -d llava/llava_pretrain/images
    echo "Cleaning up..."
    rm llava/llava_pretrain/images.zip
else
    echo "LLaVA images already exist."
fi

# 4. ShareGPT4V Additional Data
if [ ! -d "sharegpt4v_additional" ]; then
    echo "Processing ShareGPT4V additional data..."
    mkdir -p sharegpt4v_additional
    echo "Downloading ShareGPT4V additional data from Google Drive..."
    uv pip install -q gdown 2>/dev/null || true
    uv run gdown --folder "https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax?usp=sharing" -O sharegpt4v_additional --remaining-ok
    
    # Extract any zip files in the downloaded folder
    echo "Extracting zip files..."
    for zipfile in sharegpt4v_additional/*.zip; do
        if [ -f "$zipfile" ]; then
            # Get filename without extension
            dirname=$(basename "$zipfile" .zip)
            echo "Extracting $dirname..."
            mkdir -p "$dirname"
            unzip -q "$zipfile" -d "$dirname/"
            
            # Flatten nested structure if exists (e.g., wikiart/data/wikiart/images -> wikiart/images)
            # Check if there's a nested directory with the same name
            if [ -d "$dirname/data/$dirname" ]; then
                echo "Flattening nested structure for $dirname..."
                mv "$dirname/data/$dirname"/* "$dirname/"
                rm -rf "$dirname/data"
            fi
            
            rm "$zipfile"
        fi
    done
    
    echo "ShareGPT4V additional data downloaded and extracted."
else
    echo "ShareGPT4V additional data already exists."
fi

echo "Dataset setup complete!"
echo "Total disk usage:"
du -sh coco sam llava 2>/dev/null || true
