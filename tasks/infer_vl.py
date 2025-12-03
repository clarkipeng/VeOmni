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
    model.eval()
    processor = AutoProcessor.from_pretrained(args.infer.tokenizer_path)
    tokenizer = processor.tokenizer
    
    if args.infer.template_path:
        logger.info(f"Loading chat template from {args.infer.template_path}")
        with open(args.infer.template_path, "r") as f:
            tokenizer.chat_template = f.read()
    else:
        # Try to load chat_template.jinja from model_path if it exists
        possible_template = Path(args.infer.model_path) / "chat_template.jinja"
        if possible_template.exists():
            logger.info(f"Found chat_template.jinja in model path: {possible_template}")
            with open(possible_template, "r") as f:
                tokenizer.chat_template = f.read()
        else:
            logger.warning("No chat template provided and none found in model path. Using default.")

    print(f"Current chat template (first 100 chars): {tokenizer.chat_template[:100] if tokenizer.chat_template else 'None'}")
            
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Patch the model to add image_mask and video_mask in prepare_inputs_for_generation
    original_prepare_inputs = model.prepare_inputs_for_generation
    
    # Get token IDs for masks
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    
    def patched_prepare_inputs(input_ids, **kwargs):
        model_inputs = original_prepare_inputs(input_ids, **kwargs)
        # Add image_mask and video_mask if not present
        if "image_mask" not in model_inputs:
            # Check for both the constant index (if preprocessed) and the actual token ID (if from AutoProcessor)
            model_inputs["image_mask"] = (input_ids == IMAGE_INPUT_INDEX) | (input_ids == image_token_id)
        if "video_mask" not in model_inputs:
            model_inputs["video_mask"] = (input_ids == VIDEO_INPUT_INDEX) | (input_ids == video_token_id)
        return model_inputs
    
    model.prepare_inputs_for_generation = patched_prepare_inputs
    
    logger.info("Tips:")
    logger.info("  - Type '/image <path>' to load an image")
    logger.info("  - Type 'clear' to remove the history")
    logger.info("  - Type 'exit' to exit the conversation")

    messages = []
    current_images = []

    if args.infer.prompt_file:
        with open(args.infer.prompt_file, "r") as f:
            loaded_messages = json.load(f)
            # Validate and add to messages
            for msg in loaded_messages:
                messages.append(msg)
            print(f"Loaded {len(messages)} messages from {args.infer.prompt_file}")
            
            # If the last message is from user, generate a response immediately
            if messages and messages[-1]["role"] == "user":
                print("Last message is from user, generating response...")
                
                # Prepare text for generation
                text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                print("FORMATTED TEXT:", text)
                
                inputs = processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True,
                )
                
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                gen_kwargs = {
                    "do_sample": args.infer.do_sample,
                    "temperature": args.infer.temperature,
                    "top_p": args.infer.top_p,
                    "max_new_tokens": args.infer.max_tokens,
                    "streamer": streamer,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                
                print("Assistant: ", end="", flush=True)
                with torch.no_grad():
                    generated_tokens = model.generate(**inputs, **gen_kwargs)
                
                response = tokenizer.decode(generated_tokens[0, len(inputs["input_ids"][0]):], skip_special_tokens=True)
                messages.append({"role": "assistant", "content": response})
                print("\n")
    
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
        
        print("FORMATTED TEXT:", text)
        if "<|image_pad|>" not in text:
            print("WARNING: <|image_pad|> not found in formatted text!")
        else:
            print("INFO: <|image_pad|> found in formatted text.")
        
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
        
        print("Input keys:", inputs.keys())
        if "image_grid_thw" in inputs:
            print("image_grid_thw:", inputs["image_grid_thw"])
        if "input_ids" in inputs:
            print("input_ids shape:", inputs["input_ids"].shape)
            # Check for image tokens
            image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            print(f"Image token ID: {image_token_id}")
            print(f"Number of image tokens in input_ids: {(inputs['input_ids'] == image_token_id).sum().item()}")
            
            # Debug: Print decoded input
            print("DECODED INPUT:", tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))
        
        # Move to device
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        gen_kwargs = {
            "do_sample": args.infer.do_sample,
            "temperature": args.infer.temperature,
            "top_p": args.infer.top_p,
            "max_new_tokens": args.infer.max_tokens,
            "streamer": streamer,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        print("Assistant: ", end="", flush=True)
        with torch.no_grad():
            generated_tokens = model.generate(**inputs, **gen_kwargs)
        
        response = tokenizer.decode(generated_tokens[0, len(inputs["input_ids"][0]):], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})
        
        # Clear images after use (or keep them depending on your use case)
        # current_images = []


if __name__ == "__main__":
    main()
