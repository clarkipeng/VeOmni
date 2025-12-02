import datetime
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import wandb
from PIL import Image
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict, dcp_to_torch_state_dict
from veomni.data import (
    OmniDataCollatorWithPacking,
    OmniDataCollatorWithPadding,
    OmniSequenceShardCollator,
    build_dataloader,
    build_dataset,
    build_multimodal_chat_template,
)
from veomni.data.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.data.multimodal import conv_preprocess
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_processor, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from veomni.data.chat_template import ChatTemplate


logger = helper.create_logger(__name__)


def upload_checkpoint_to_s3(local_path: str, s3_path: str) -> None:
    """
    Upload a checkpoint directory to S3.
    
    Args:
        local_path: Local path to the checkpoint directory
        s3_path: S3 path in format s3://bucket/key
    """
    if not s3_path.startswith("s3://"):
        logger.warning(f"S3 path must start with 's3://', got {s3_path}. Skipping upload.")
        return
    
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        logger.warning("boto3 is not installed. Skipping S3 upload. Install with: pip install boto3")
        return
    
    try:
        # Parse s3://bucket/key
        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        base_key = parts[1] if len(parts) > 1 else ""
        
        s3_client = boto3.client("s3")
        local_path_obj = Path(local_path)
        
        if not local_path_obj.exists():
            logger.warning(f"Local checkpoint path does not exist: {local_path}. Skipping upload.")
            return
        
        # Upload all files in the checkpoint directory
        uploaded_count = 0
        for file_path in local_path_obj.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path_obj)
                s3_key = f"{base_key}/{relative_path}".replace("\\", "/") if base_key else str(relative_path).replace("\\", "/")
                
                s3_client.upload_file(str(file_path), bucket, s3_key)
                uploaded_count += 1
        
        logger.info_rank0(f"Uploaded {uploaded_count} files from {local_path} to {s3_path}")
    except ClientError as e:
        logger.error(f"Failed to upload checkpoint to S3: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during S3 upload: {e}")


def save_and_upload_checkpoint(
    Checkpointer,
    save_checkpoint_path: str,
    state: dict,
    global_step: int,
    args,
    model_config,
    processor,
) -> str:
    """
    Save DCP checkpoint, convert to HF format, and upload to S3.
    
    Args:
        Checkpointer: Checkpoint manager
        save_checkpoint_path: Base path for checkpoints
        state: State dict containing model, optimizer, extra_state
        global_step: Current global step
        args: Training arguments
        model_config: Model configuration
        processor: Model processor
    
    Returns:
        checkpoint_path: Full path to the saved checkpoint
    """
    checkpoint_path = os.path.join(save_checkpoint_path, f"global_step_{global_step}")
    Checkpointer.save(save_checkpoint_path, state, global_steps=global_step)
    dist.barrier()
    logger.info_rank0(f"Distributed checkpoint saved at {checkpoint_path} successfully!")
    
    # Convert to HF and upload to S3 if configured
    if args.train.checkpoint_upload_path and args.train.global_rank == 0:
        logger.info_rank0("Converting checkpoint to HuggingFace format for S3 upload...")
        hf_checkpoint_path = f"{checkpoint_path}_hf"
        state_dict = dcp_to_torch_state_dict(save_checkpoint_path=checkpoint_path)
        save_model_weights(hf_checkpoint_path, state_dict, model_assets=[model_config, processor])
        
        # Explicitly save chat template to ensure it's correct
        if hasattr(processor.tokenizer, "chat_template") and processor.tokenizer.chat_template:
            template_file = os.path.join(hf_checkpoint_path, "chat_template.jinja")
            with open(template_file, "w") as f:
                f.write(processor.tokenizer.chat_template)
            logger.info_rank0(f"Explicitly saved chat template to {template_file}")

        logger.info_rank0(f"HF checkpoint saved at {hf_checkpoint_path}")
        
        s3_upload_path = f"{args.train.checkpoint_upload_path}/global_step_{global_step}"
        upload_checkpoint_to_s3(hf_checkpoint_path, s3_upload_path)
    
    return checkpoint_path


MAX_PIXELS = 128 * 28 * 28
ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
}


def process_sample(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    max_seq_len: int = 32768,
    **kwargs,
):
    """
    Processes multimodal example with qwen2vl's pre-processor.
    """
    source_name = sample.get("source_name") or kwargs.get("source_name")
    # Handle different data formats: "text" for fineweb, "conversations" or "messages" for conversation data
    if source_name == "fineweb_100BT":
        conversations = sample["text"]
    elif "conversations" in sample and sample["conversations"]:
        conversations = sample["conversations"]
    elif "messages" in sample and sample["messages"]:
        conversations = sample["messages"]
    else:
        raise KeyError(f"Sample must have one of 'text', 'conversations', or 'messages' keys (with non-None values). Found keys: {list(sample.keys())}")
    
    # Skip preprocessing if source_name is None or empty (data is already in correct format)
    if source_name:
        conversations = conv_preprocess(source_name, conversations, **kwargs)
    elif conversations:
        converted_conversations = []
        for msg in conversations:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role in ["user", "assistant"]:
                converted_conversations.append([role, ("text", content)])
        conversations = converted_conversations

    token_num_inputs, image_inputs = {}, {}
    image_grid_thw = None
    if "images" in sample and sample["images"]:
        images = []
        for image in sample["images"]:
            images.append(Image.open(BytesIO(image)).convert("RGB"))

        image_inputs = processor.image_processor(images=images, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = processor.image_processor.merge_size**2
        image_token_num = image_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["image"] = image_token_num

    tokenized_example = chat_template.encode_messages(conversations, token_num_inputs)
    tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}
    input_ids = tokenized_example["input_ids"]

    position_ids = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=image_grid_thw,
        attention_mask=tokenized_example["attention_mask"].unsqueeze(0),
    )["position_ids"]

    tokenized_example["position_ids"] = position_ids.squeeze().clone()  # (dim, l)
    # clone here as text_only data is (1, l).expand(dim, -1),

    # [SEQ_OVERFLOW] Log warning if sequence exceeds max_seq_len
    seq_len = position_ids.shape[-1]
    if seq_len > max_seq_len:
        logger.warning(
            f"INPUT [SEQ_OVERFLOW] Sequence length {seq_len} exceeds max_seq_len {max_seq_len} "
            f"(overflow: {seq_len - max_seq_len} tokens, "
            f"{100 * (seq_len - max_seq_len) / max_seq_len:.1f}%) - DROPPING SAMPLE")
        return []
    # if seq_len > 32768:
    #     logger.warning(
    #         f"INPUT [SEQ_OVERFLOW] Sequence length {seq_len} exceeds max_seq_len 32768 "
    #         f"(overflow: {seq_len - 32768} tokens, "
    #         f"{100 * (seq_len - 32768) / 32768:.1f}%)")
    #     if image_grid_thw is not None:
    #         logger.warning(f"INPUT [SEQ_OVERFLOW] Sample has {image_grid_thw.shape[0]} images: {image_grid_thw.tolist()}")

    tokenized_example["image_mask"] = tokenized_example["input_ids"] == IMAGE_INPUT_INDEX
    tokenized_example["video_mask"] = tokenized_example["input_ids"] == VIDEO_INPUT_INDEX
    tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
    tokenized_example["input_ids"][tokenized_example["video_mask"]] = 0
    tokenized_example.update(image_inputs)
    return [tokenized_example]


def get_param_groups(model: "torch.nn.Module", default_lr: float, vit_lr: float):
    vit_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "visual" in name:
                vit_params.append(param)
            else:
                other_params.append(param)

    return [{"params": vit_params, "lr": vit_lr}, {"params": other_params, "lr": default_lr}]


@dataclass
class MyDataArguments(DataArguments):
    val_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to validation dataset. Can be a single path or multisource YAML config."},
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vit parameters."},
    )
    vit_lr: float = field(
        default=1e-6,
        metadata={"help": "Maximum learning rate for vit parameters."},
    )
    eval_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Run evaluation every X steps. If None, only evaluate at end of epoch."},
    )


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    dist.init_process_group(backend=get_dist_comm_backend())
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        config_kwargs = {"attn_implementation":args.model.attn_implementation},
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
    )
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    logger.info_rank0("Prepare data")
    processor = build_processor(args.model.tokenizer_path)
    processor.image_processor.max_pixels = MAX_PIXELS
    position_id_func = model.get_position_id_func()
    chat_template = build_multimodal_chat_template(
        args.data.chat_template, 
        processor.tokenizer, 
        template_path=args.data.template_path
    )
    processor.tokenizer.chat_template = chat_template.chat_template
    if hasattr(processor.tokenizer, "init_kwargs"):
        processor.tokenizer.init_kwargs["chat_template"] = chat_template.chat_template
    print("SET CHAT TEMPLATE", chat_template.chat_template)
    transform = partial(
        process_sample,
        processor=processor,
        chat_template=chat_template,
        position_id_func=position_id_func,
        max_seq_len=args.data.max_seq_len,
    )

    if args.train.rmpad:
        raise ValueError("Qwen2-VL does not support rmpad. Use `rmpad_with_pos_ids` instead.")

    data_collate_fn = []
    if args.train.rmpad_with_pos_ids:
        data_collate_fn.append(OmniDataCollatorWithPacking())
    else:
        data_collate_fn.append(OmniDataCollatorWithPadding())
    if get_parallel_state().sp_enabled:
        data_collate_fn.append(
            OmniSequenceShardCollator(
                padding_scale={
                    "pixel_values": processor.image_processor.merge_size**2,
                },
                rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            )
        )

    train_dataset = build_dataset(
        dataset_name=args.data.dataset_name,
        transform=transform,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        **asdict(args.data),
    )
    dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
    if args.data.datasets_type == "mapping":
        dataset_length = dataset_length / args.train.data_parallel_size
    args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        collate_fn=data_collate_fn,
        max_seq_len=args.data.max_seq_len,
        train_steps=args.train.train_steps,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        dyn_bsz_margin=args.train.dyn_bsz_margin,
        dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
    )

    # Build validation dataset if val_path is provided
    val_dataloader = None
    if args.data.val_path:
        logger.info_rank0("Start building validation dataset")
        val_enable_multisource = args.data.val_path.endswith(".yaml")
        val_dataset = None
        if val_enable_multisource:
            val_dataset = build_interleave_dataset(
                args.data.val_path, args.data.datasets_type, transform=transform, seed=args.train.seed
            )
        elif args.data.datasets_type == "iterable":
            val_dataset = build_iterative_dataset(
                args.data.val_path, transform=transform, seed=args.train.seed, source_name=args.data.source_name
            )
        elif args.data.datasets_type == "mapping":
            val_dataset = build_mapping_dataset(
                args.data.val_path, transform=transform, source_name=args.data.source_name
            )
        
        if val_dataset is not None:
            # Calculate approximate validation steps
            val_steps = 100  # Default
            if hasattr(val_dataset, "__len__"):
                val_steps = max(1, len(val_dataset) // args.train.global_batch_size)
            
            val_dataloader = build_dataloader(
                dataset=val_dataset,
                micro_batch_size=args.train.micro_batch_size,
                global_batch_size=args.train.global_batch_size,
                dataloader_batch_size=args.train.dataloader_batch_size,
                seed=args.train.seed,
                collate_fn=data_collate_fn,
                max_seq_len=args.data.max_seq_len,
                train_steps=val_steps,
                rmpad=args.train.rmpad,
                rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
                bsz_warmup_ratio=0,  # No warmup for validation
                dyn_bsz_margin=args.train.dyn_bsz_margin,
                dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
                num_workers=args.data.num_workers,
                prefetch_factor=args.data.prefetch_factor,
                pin_memory=args.data.pin_memory,
                drop_last=False,  # Don't drop last batch in validation
            )
            logger.info_rank0(f"Validation dataset built with {len(val_dataset) if hasattr(val_dataset, '__len__') else 'unknown'} samples")
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    fsdp_kwargs = {}
    if args.train.freeze_vit:
        model.visual.requires_grad_(False)
        if args.train.data_parallel_mode == "fsdp1":
            fsdp_kwargs["use_orig_params"] = True

    model = build_parallelize_model(
        model,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        init_device=args.train.init_device,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        fsdp_kwargs=fsdp_kwargs,
        basic_modules=model._no_split_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )
    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=False,
        optimizer_type=args.train.optimizer,
        param_groups=get_param_groups(model, args.train.lr, args.train.vit_lr),
    )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        model_assets = [model_config, processor]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )

    if args.train.enable_compile:
        logger.info_rank0("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1
            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1 and args.train.local_rank == 0:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)
                
                # Dump first batch to file for debugging
                dump_file = Path(args.train.output_dir) / "log.txt"
                dump_file.parent.mkdir(parents=True, exist_ok=True)
                tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                
                with open(dump_file, "w", encoding="utf-8") as f:
                    f.write("=" * 80 + "\n")
                    f.write("VeOmni First Batch Sequence Dump\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for mb_idx, micro_batch in enumerate(micro_batches):
                        f.write(f"\n--- Micro Batch {mb_idx} ---\n\n")
                        input_ids = micro_batch.get("input_ids")
                        labels = micro_batch.get("labels")
                        
                        if input_ids is not None:
                            f.write(f"input_ids shape: {input_ids.shape}\n")
                            try:
                                # Handle negative IDs for decoding
                                clean_input_ids = []
                                for tid in input_ids[0].tolist():
                                    if tid < 0:
                                        clean_input_ids.append(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
                                    else:
                                        clean_input_ids.append(tid)
                                
                                decoded = tokenizer.decode(clean_input_ids, skip_special_tokens=False)
                                f.write(f"Decoded : {decoded}...\n\n")
                                logger.info_rank0(f"Decoded Input (MB {mb_idx}):\n{decoded}...")
                            except Exception as e:
                                f.write(f"Could not decode: {e}\n\n")
                                logger.warning(f"Could not decode input: {e}")
                        
                        if labels is not None:
                            num_valid = (labels != -100).sum().item()
                            num_total = labels.numel()
                            num_masked = (labels == -100).sum().item()
                            f.write(f"Labels: {num_valid}/{num_total} valid ({100*num_valid/num_total:.2f}%), {num_masked} masked\n")
                            
                            # Decode targets (valid labels only)
                            try:
                                clean_labels = []
                                for tid in labels[0].tolist():
                                    if tid == -100:
                                        continue # Skip ignored
                                    if tid < 0:
                                        clean_labels.append(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
                                    else:
                                        clean_labels.append(tid)
                                decoded_labels = tokenizer.decode(clean_labels, skip_special_tokens=False)
                                f.write(f"Decoded Targets (first 1000 chars): {decoded_labels[:1000]}...\n\n")
                                logger.info_rank0(f"Decoded Targets (MB {mb_idx}):\n{decoded_labels[:1000]}...")
                            except Exception as e:
                                f.write(f"Could not decode labels: {e}\n\n")

                            # Check if image tokens are properly masked
                            if input_ids is not None:
                                image_mask = (input_ids == IMAGE_INPUT_INDEX) | (input_ids == 0)
                                image_positions = image_mask[0].nonzero(as_tuple=True)[0].tolist()
                                if len(image_positions) > 0:
                                    image_labels = labels[0, image_positions[:min(50, len(image_positions))]]
                                    num_image_masked = (image_labels == -100).sum().item()
                                    f.write(f"Image tokens: {len(image_positions)} found, {num_image_masked}/{len(image_labels)} masked in labels\n")
                                    if num_image_masked < len(image_labels):
                                        f.write(f"WARNING: Some image tokens are NOT masked in labels!\n")
                
                logger.info_rank0(f"Dumped first batch to {dump_file}")

            total_loss = 0
            synchronize()
            start_time = time.time()
            for micro_batch_idx, micro_batch in enumerate(micro_batches):
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("cur_token_num", None)
                    micro_batch.pop("source_name", None)

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                with model_fwd_context:
                    loss: "torch.Tensor" = model(**micro_batch, use_cache=False).loss / len(micro_batches)
                # loss: "torch.Tensor" = model(**micro_batch, use_cache=False).loss / len(micro_batches)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = model.clip_grad_norm_(args.train.max_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm, foreach=True)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            # Extract tok/s from metrics (it's in millions, convert to regular tok/s)
            tok_per_sec = train_metrics.get("tokens_per_second(M)", 0) * 1e6
            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}, tok/s: {tok_per_sec:.0f}"
            )
            data_loader_tqdm.update(1)
            data_loader_tqdm.refresh()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()
                    print(profiler.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
            # Run validation at specified steps if eval_steps is set
            if val_dataloader is not None and args.train.eval_steps and global_step % args.train.eval_steps == 0:
                logger.info_rank0(f"Running validation at step {global_step}...")
                model.eval()
                val_loss = 0.0
                val_num_batches = 0
                
                with torch.no_grad():
                    val_iterator = iter(val_dataloader)
                    try:
                        # Limit validation to a reasonable number of batches
                        max_val_batches = 50
                        for _ in range(max_val_batches):
                            try:
                                micro_batches: List[Dict[str, Any]] = next(val_iterator)
                            except StopIteration:
                                break
                            
                            for micro_batch in micro_batches:
                                if args.data.enable_multisource:
                                    micro_batch.pop("ds_idx", None)
                                    micro_batch.pop("source_name", None)
                                
                                micro_batch = {
                                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                                    for k, v in micro_batch.items()
                                }
                                
                                with model_fwd_context:
                                    loss: "torch.Tensor" = model(**micro_batch, use_cache=False).loss
                                
                                val_loss += loss.item()
                                val_num_batches += 1
                                del micro_batch
                    except Exception as e:
                        logger.warning(f"Error during validation: {e}")
                
                # Aggregate validation loss across all processes
                if val_num_batches > 0:
                    val_loss, val_num_batches = all_reduce((val_loss, val_num_batches), group=get_parallel_state().fsdp_group)
                    val_loss = val_loss / val_num_batches
                    logger.info_rank0(f"Validation loss at step {global_step}: {val_loss:.4f} (over {val_num_batches} batches)")
                    
                    if args.train.global_rank == 0 and args.train.use_wandb:
                        wandb.log({"validation/loss": val_loss}, step=global_step)
                
                model.train()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                save_checkpoint_path = save_and_upload_checkpoint(
                    Checkpointer, args.train.save_checkpoint_path, state, global_step, args, model_config, processor
                )

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        
        # Run validation at end of epoch if val_dataloader is available
        if val_dataloader is not None:
            logger.info_rank0("Running validation...")
            model.eval()
            val_loss = 0.0
            val_num_batches = 0
            val_num_samples = 0
            
            with torch.no_grad():
                val_iterator = iter(val_dataloader)
                try:
                    while True:
                        try:
                            micro_batches: List[Dict[str, Any]] = next(val_iterator)
                        except StopIteration:
                            break
                        
                        for micro_batch in micro_batches:
                            if args.data.enable_multisource:
                                micro_batch.pop("ds_idx", None)
                                micro_batch.pop("source_name", None)
                            
                            micro_batch = {
                                k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                                for k, v in micro_batch.items()
                            }
                            
                            with model_fwd_context:
                                loss: "torch.Tensor" = model(**micro_batch, use_cache=False).loss
                            
                            val_loss += loss.item()
                            val_num_batches += 1
                            val_num_samples += micro_batch.get("input_ids", torch.tensor([])).shape[0] if isinstance(micro_batch.get("input_ids"), torch.Tensor) else 1
                            
                            del micro_batch
                except Exception as e:
                    logger.warning(f"Error during validation: {e}")
            
            # Aggregate validation loss across all processes
            val_loss, val_num_batches = all_reduce((val_loss, val_num_batches), group=get_parallel_state().fsdp_group)
            if val_num_batches > 0:
                val_loss = val_loss / val_num_batches
                logger.info_rank0(f"Validation loss at epoch {epoch + 1}: {val_loss:.4f} (over {val_num_batches} batches)")
                
                if args.train.global_rank == 0 and args.train.use_wandb:
                    wandb.log({"validation/loss": val_loss, "validation/epoch": epoch + 1}, step=global_step)
            
            model.train()
        
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            save_checkpoint_path = save_and_upload_checkpoint(
                Checkpointer, args.train.save_checkpoint_path, state, global_step, args, model_config, processor
            )

    synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0:
        if args.train.save_hf_weights and save_checkpoint_path is not None:
            hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
            model_state_dict = ckpt_to_state_dict(
                save_checkpoint_path=save_checkpoint_path,
                output_dir=args.train.output_dir,
                ckpt_manager=args.train.ckpt_manager,
            )
            save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
            
            # Explicitly save chat template to ensure it's correct
            # Note: model_assets[1] is processor
            proc = model_assets[1]
            if hasattr(proc.tokenizer, "chat_template") and proc.tokenizer.chat_template:
                template_file = os.path.join(hf_weights_path, "chat_template.jinja")
                with open(template_file, "w") as f:
                    f.write(proc.tokenizer.chat_template)
                logger.info_rank0(f"Explicitly saved chat template to {template_file}")

            logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
            
            # Upload final checkpoint to S3 if configured
            if args.train.checkpoint_upload_path:
                final_global_step = save_checkpoint_path.split("_")[-1]
                s3_upload_path = f"{args.train.checkpoint_upload_path}/global_step_{final_global_step}"
                upload_checkpoint_to_s3(hf_weights_path, s3_upload_path)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
