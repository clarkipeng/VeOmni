#!/usr/bin/env python3
"""
Enhanced inference script for Qwen3-VL with image support.

Usage:
  # Text only
  python tasks/infer_vl.py --infer.model_path /path/to/model --infer.tokenizer_path /path/to/tokenizer
  
  # With image
  python tasks/infer_vl.py --infer.model_path /path/to/model --infer.tokenizer_path /path/to/tokenizer
  Then type: /image /path/to/image.jpg
  Then type: Describe this image
"""
import json
import readline  # noqa: F401
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, TextStreamer

from veomni.data.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.models import build_foundation_model
from veomni.utils import helper
from veomni.utils.arguments import InferArguments, parse_args


logger = helper.create_logger(__name__)


@dataclass
class Arguments:
    infer: "InferArguments" = field(default_factory=InferArguments)


def main() -> None:
    args = parse_args(Arguments)
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    helper.set_seed(args.infer.seed)
    helper.enable_third_party_logging()
    
    model = build_foundation_model(config_path=args.infer.model_path, weights_path=args.infer.model_path)
    processor = AutoProcessor.from_pretrained(args.infer.tokenizer_path)
    tokenizer = processor.tokenizer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Patch the model to add image_mask and video_mask in prepare_inputs_for_generation
    original_prepare_inputs = model.prepare_inputs_for_generation
    
    def patched_prepare_inputs(input_ids, **kwargs):
        model_inputs = original_prepare_inputs(input_ids, **kwargs)
        # Add image_mask and video_mask if not present
        if "image_mask" not in model_inputs:
            model_inputs["image_mask"] = input_ids == IMAGE_INPUT_INDEX
        if "video_mask" not in model_inputs:
            model_inputs["video_mask"] = input_ids == VIDEO_INPUT_INDEX
        return model_inputs
    
    model.prepare_inputs_for_generation = patched_prepare_inputs
    
    # Load custom chat template if provided
    if args.infer.template_path:
        from pathlib import Path
        template_path = Path(args.infer.template_path)
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        with open(template_path, 'r') as f:
            custom_template = f.read()
        tokenizer.chat_template = custom_template
        logger.info(f"Loaded custom chat template from: {args.infer.template_path}")
    
    logger.info("Tips:")
    logger.info("  - Type '/image <path>' to load an image")
    logger.info("  - Type 'clear' to remove the history")
    logger.info("  - Type 'exit' to exit the conversation")

    messages = []
    current_images = []
    
    while True:
        query = input("\nUser: ")

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            current_images = []
            print("History has been removed.")
            continue
        
        # Handle image loading
        if query.strip().startswith("/image "):
            image_path = query.strip()[7:].strip()
            try:
                img = Image.open(image_path).convert("RGB")
                current_images.append(img)
                print(f"Loaded image: {image_path} ({len(current_images)} total)")
                continue
            except Exception as e:
                print(f"Error loading image: {e}")
                continue

        # Build message content
        if current_images:
            # For vision-language models, construct multimodal content
            content = []
            for img in current_images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": query})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": query})

        # Process with processor
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Process images if present
        if current_images:
            inputs = processor(
                text=[text],
                images=current_images,
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = processor(
                text=[text],
                return_tensors="pt",
                padding=True,
            )
        
        # Move to device
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        gen_kwargs = {
            "do_sample": args.infer.do_sample,
            "temperature": args.infer.temperature,
            "top_p": args.infer.top_p,
            "max_new_tokens": args.infer.max_tokens,
            "repetition_penalty": args.infer.repetition_penalty,
            "streamer": streamer,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        print("Assistant: ", end="", flush=True)
        with torch.no_grad():
            generated_tokens = model.generate(**inputs, **gen_kwargs)
        
        response = tokenizer.decode(generated_tokens[0, len(input_ids[0]):], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})
        
        # Clear images after use (or keep them depending on your use case)
        # current_images = []


if __name__ == "__main__":
    main()
