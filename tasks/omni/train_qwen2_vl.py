import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import torch
import torch.distributed as dist
import wandb
from PIL import Image
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
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


MAX_PIXELS = 256 * 28 * 28
ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
}


def process_sample(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    **kwargs,
):
    """
    Processes multimodal example with qwen2vl's pre-processor.
    """
    source_name = sample.get("source_name") or kwargs.get("source_name")
    # Handle different data formats: "text" for fineweb, "conversations" or "messages" for conversation data
    if source_name == "fineweb_100BT":
        conversations = sample["text"]
    elif "conversations" in sample:
        conversations = sample["conversations"]
    elif "messages" in sample:
        conversations = sample["messages"]
    else:
        raise KeyError(f"Sample must have one of 'text', 'conversations', or 'messages' keys. Found keys: {list(sample.keys())}")
    
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
class MyTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vit parameters."},
    )
    vit_lr: float = field(
        default=1e-6,
        metadata={"help": "Maximum learning rate for vit parameters."},
    )


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
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
    chat_template = build_multimodal_chat_template(args.data.chat_template, processor.tokenizer)
    transform = partial(
        process_sample,
        processor=processor,
        chat_template=chat_template,
        position_id_func=position_id_func,
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
        dataloader_type=args.data.dataloader_type,
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

    fsdp_kwargs = {}
    if args.train.freeze_vit:
        model.visual.requires_grad_(False)
        if args.train.data_parallel_mode == "fsdp1":
            fsdp_kwargs["use_orig_params"] = True

    model = build_parallelize_model(
        model,
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
                
                # Dump first batch sequence to file for debugging
                dump_file = Path(args.train.output_dir) / "veomni_first_batch_dump.txt"
                dump_file.parent.mkdir(parents=True, exist_ok=True)
                with open(dump_file, "w", encoding="utf-8") as f:
                    f.write("=" * 80 + "\n")
                    f.write("VeOmni First Batch Sequence Dump\n")
                    f.write("=" * 80 + "\n\n")
                    for mb_idx, micro_batch in enumerate(micro_batches):
                        f.write(f"\n--- Micro Batch {mb_idx} ---\n\n")
                        input_ids = micro_batch.get("input_ids")
                        labels = micro_batch.get("labels")
                        attention_mask = micro_batch.get("attention_mask")
                        
                        if input_ids is not None:
                            f.write(f"input_ids shape: {input_ids.shape}\n")
                            f.write(f"input_ids (first 200 tokens): {input_ids[0, :200].tolist()}\n")
                            # Decode tokens
                            try:
                                # Use the processor that was already built earlier
                                tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                                decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                                f.write(f"\nDecoded input_ids:\n{decoded}\n")
                            except Exception as e:
                                f.write(f"\nCould not decode tokens: {e}\n")
                        
                        if labels is not None:
                            f.write(f"\nlabels shape: {labels.shape}\n")
                            num_valid = (labels != -100).sum().item()
                            num_total = labels.numel()
                            f.write(f"valid labels: {num_valid}/{num_total} ({100*num_valid/num_total:.2f}%)\n")
                            f.write(f"labels (first 200 tokens): {labels[0, :200].tolist()}\n")
                            # Show which positions are valid
                            valid_positions = (labels[0] != -100).nonzero(as_tuple=True)[0].tolist()
                            f.write(f"valid label positions (first 100): {valid_positions[:100]}\n")
                            
                            # Show tokens INCLUDED in loss, in segments
                            if input_ids is not None:
                                f.write(f"\n--- Tokens INCLUDED in Loss (shown in segments) ---\n")
                                f.write(f"Total valid positions: {len(valid_positions)}\n\n")
                                
                                if len(valid_positions) > 0:
                                    try:
                                        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                                        
                                        # Group consecutive positions into segments
                                        segments = []
                                        current_segment = [valid_positions[0]]
                                        for i in range(1, len(valid_positions)):
                                            if valid_positions[i] == valid_positions[i-1] + 1:
                                                current_segment.append(valid_positions[i])
                                            else:
                                                segments.append(current_segment)
                                                current_segment = [valid_positions[i]]
                                        segments.append(current_segment)
                                        
                                        f.write(f"Found {len(segments)} continuous segments of valid tokens\n\n")
                                        
                                        # Show each segment
                                        for seg_idx, segment in enumerate(segments):
                                            start_pos = segment[0]
                                            end_pos = segment[-1]
                                            segment_token_ids = input_ids[0, start_pos:end_pos+1].tolist()
                                            segment_labels = labels[0, start_pos:end_pos+1].tolist()
                                            
                                            f.write(f"--- Segment {seg_idx + 1}/{len(segments)}: positions {start_pos}-{end_pos} ({len(segment)} tokens) ---\n")
                                            f.write(f"Token IDs: {segment_token_ids[:50]}{'...' if len(segment_token_ids) > 50 else ''}\n")
                                            f.write(f"Labels: {segment_labels[:50]}{'...' if len(segment_labels) > 50 else ''}\n")
                                            
                                            # Decode the segment
                                            try:
                                                decoded_segment = tokenizer.decode(segment_token_ids, skip_special_tokens=False)
                                                f.write(f"Decoded text:\n{decoded_segment}\n")
                                            except Exception as e:
                                                f.write(f"Could not decode segment: {e}\n")
                                            f.write("\n")
                                            
                                            # Limit to first 20 segments to avoid huge files
                                            if seg_idx >= 19:
                                                remaining = len(segments) - 20
                                                if remaining > 0:
                                                    f.write(f"... ({remaining} more segments omitted)\n")
                                                break
                                    except Exception as e:
                                        f.write(f"\nCould not process valid tokens: {e}\n")
                        
                        if attention_mask is not None:
                            f.write(f"\nattention_mask shape: {attention_mask.shape}\n")
                            num_attn = attention_mask.sum().item()
                            f.write(f"attention_mask sum: {num_attn}/{attention_mask.numel()}\n")
                        
                        # Check for any mask fields
                        for key in micro_batch.keys():
                            if "mask" in key.lower() and key not in ["attention_mask", "labels"]:
                                mask_val = micro_batch[key]
                                if isinstance(mask_val, torch.Tensor):
                                    f.write(f"\n{key} shape: {mask_val.shape}\n")
                                    if mask_val.numel() < 500:
                                        f.write(f"{key} values: {mask_val.tolist()}\n")
                                    else:
                                        f.write(f"{key} (first 200): {mask_val.flatten()[:200].tolist()}\n")
                logger.warning(f"Dumped first batch to {dump_file}")

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
                    helper.upload_trace(args.train.wandb_project, args.train.wandb_name, args.train.profile_trace_dir)

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
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
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
                
                # Upload to S3 if configured
                if args.train.checkpoint_upload_path and args.train.global_rank == 0:
                    s3_upload_path = f"{args.train.checkpoint_upload_path}/global_step_{global_step}"
                    upload_checkpoint_to_s3(save_checkpoint_path, s3_upload_path)

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
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
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

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
            logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
