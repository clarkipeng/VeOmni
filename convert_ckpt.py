
import os
import shutil
import torch
from veomni.models import build_foundation_model
from veomni.models.module_utils import save_model_weights

# Paths
CHECKPOINT_PATH = "./qwen3_vl_moe_sft_2e-5/checkpoints/global_step_465/hf_ckpt"
CONFIG_PATH = "./qwen3_vl_moe_sft_2e-5/checkpoints/global_step_465/hf_ckpt/config.json"
NEW_CHECKPOINT_PATH = "./qwen3_vl_moe_sft_2e-5/checkpoints/global_step_465/hf_ckpt_converted"

def convert_checkpoint():
    print(f"Loading state dict from {CHECKPOINT_PATH}...")
    state_dict = {}
    
    # Manually load all safetensors files
    from safetensors.torch import load_file
    files = [f for f in os.listdir(CHECKPOINT_PATH) if f.endswith(".safetensors")]
    for f in sorted(files):
        path = os.path.join(CHECKPOINT_PATH, f)
        print(f"Loading {f}...")
        # Load to CPU to avoid OOM and meta tensors
        shard = load_file(path, device="cpu")
        state_dict.update(shard)
        
    print(f"Loaded {len(state_dict)} keys.")

    print(f"Saving converted model to {NEW_CHECKPOINT_PATH}...")
    if os.path.exists(NEW_CHECKPOINT_PATH):
        shutil.rmtree(NEW_CHECKPOINT_PATH)
        
    # This call triggers the auto-conversion hook I added to save_model_weights
    save_model_weights(
        output_dir=NEW_CHECKPOINT_PATH,
        state_dict=state_dict,
        safe_serialization=True 
    )
    
    # Copy config and other assets (tokenizer, etc)
    for filename in os.listdir(CHECKPOINT_PATH):
        if not filename.endswith(".safetensors") and not filename.endswith(".index.json"):
            src = os.path.join(CHECKPOINT_PATH, filename)
            dst = os.path.join(NEW_CHECKPOINT_PATH, filename)
            if os.path.isfile(src):
                shutil.copy(src, dst)
                
    print("Conversion complete.")

if __name__ == "__main__":
    convert_checkpoint()
