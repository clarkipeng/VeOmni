
# VeOmni: Vision-Language Model Training

This repository contains the code for training VeOmni, a vision-language model based on Qwen2-VL.

## Dataset Setup

To train the model with the ShareGPT4V dataset, you need to download the dataset JSON files and the COCO images.

### 1. Download ShareGPT4V Datasets

Run the following commands to download the ShareGPT4V datasets into the `sharegpt4v_cap_100k` directory:

```bash
mkdir -p sharegpt4v_cap_100k
cd sharegpt4v_cap_100k

# Download the 100k subset (Recommended for this project)
wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json

# Optional: Download other subsets if needed
# wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json
# wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/share-captioner_coco_lcs_sam_1246k_1107.json

cd ..
```

### 2. Download COCO Images

The ShareGPT4V dataset references images from the COCO 2017 Train dataset. You need to download and extract them:

```bash
# Create coco directory
mkdir -p coco

# Download COCO train2017 images (18GB)
wget http://images.cocodataset.org/zips/train2017.zip -O coco/train2017.zip

# Extract images
unzip -q coco/train2017.zip -d coco/

# Clean up zip file
rm coco/train2017.zip
```

After these steps, your directory structure should look like this:

```
VeOmni/
├── sharegpt4v_cap_100k/
│   └── sharegpt4v_instruct_gpt4-vision_cap100k.json
├── coco/
│   └── train2017/
│       ├── 000000000009.jpg
│       └── ...
├── tasks/
├── veomni/
└── ...
```

## Training

To start training, run:

```bash
./run_veomni_compare.sh
```
