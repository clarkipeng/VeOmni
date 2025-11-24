#!/bin/bash
set -e

# Create directories
mkdir -p sharegpt4v_cap_100k
mkdir -p coco

echo "Downloading ShareGPT4V dataset (100k subset)..."
cd sharegpt4v_cap_100k
if [ ! -f "sharegpt4v_instruct_gpt4-vision_cap100k.json" ]; then
    wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json
else
    echo "ShareGPT4V dataset already exists."
fi
cd ..

echo "Downloading COCO train2017 images (18GB)..."
if [ ! -d "coco/train2017" ]; then
    wget -c http://images.cocodataset.org/zips/train2017.zip -O coco/train2017.zip
    echo "Extracting COCO images..."
    unzip -q coco/train2017.zip -d coco/
    rm coco/train2017.zip
else
    echo "COCO images already exist."
fi

echo "Dataset download complete!"
