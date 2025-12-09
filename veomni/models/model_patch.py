"""Model patching utilities."""

import re
import torch
import torch.nn as nn
from loguru import logger

from torchtitan.models.moe import MoE, MoEArgs

from veomni.models.transformers.qwen3_moe import modeling_qwen3_moe
from veomni.models.transformers.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from veomni.models.transformers.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer


from veomni.models.transformers.qwen3_vl_moe import modeling_qwen3_vl_moe
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeTextConfig


# ref: https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/blob/main/config.json
def get_qwen3_moe_args(config, use_grouped_mm: bool = True) -> MoEArgs:
    return MoEArgs(
        num_experts=config.num_experts,
        num_shared_experts=0,
        top_k=config.num_experts_per_tok,
        score_func="softmax",
        route_norm=config.norm_topk_prob,
        route_scale=1.0,
        score_before_experts=False,
        use_grouped_mm=use_grouped_mm,
    )


# ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
class Qwen3MoeTitanDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        moe_args = get_qwen3_moe_args(config)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)


class Qwen3VLMoeTextTitanDecoderLayer(modeling_qwen3_vl_moe.Qwen3VLMoeTextDecoderLayer):
    def __init__(self, config: Qwen3VLMoeTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        moe_args = get_qwen3_moe_args(config)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)


def apply_tt_moe(model_type: str) -> None:
    """Patch model MoE layer with torchtitan implementation."""
    logger.info(f"Applying torchtitan MoE patch for model_type: {model_type}")
    if model_type == "qwen3_moe":
        logger.warning(f"Patching Qwen3MoeDecoderLayer with Qwen3MoeTitanDecoderLayer for {model_type}")
        modeling_qwen3_moe.Qwen3MoeDecoderLayer = Qwen3MoeTitanDecoderLayer
    elif model_type == "qwen3_vl_moe":
        logger.warning(f"Patching Qwen3VLMoeTextDecoderLayer with Qwen3VLMoeTextTitanDecoderLayer for {model_type}")
        modeling_qwen3_vl_moe.Qwen3VLMoeTextDecoderLayer = Qwen3VLMoeTextTitanDecoderLayer


def convert_hf_moe_key_to_tt(key: str) -> tuple[str, bool]:
    """
    Convert a HuggingFace MoE weight key to TorchTitan format.
    Returns (converted_key, needs_tensor_conversion).
    
    HF format: layers.{i}.mlp.gate.weight, layers.{i}.mlp.experts.gate_up_proj, layers.{i}.mlp.experts.down_proj
    TT format: layers.{i}.mlp.router.gate.weight, layers.{i}.mlp.experts.w1/w2/w3
    """
    # Router gate: mlp.gate.weight -> mlp.router.gate.weight
    if ".mlp.gate.weight" in key:
        return key.replace(".mlp.gate.weight", ".mlp.router.gate.weight"), False
    
    # Experts gate_up_proj -> needs special handling (split into w1 and w3)
    if ".mlp.experts.gate_up_proj" in key:
        # This weight needs to be split, we'll handle it specially
        return key, True  # Mark for special conversion
    
    # Experts down_proj -> w2
    if ".mlp.experts.down_proj" in key:
        return key.replace(".mlp.experts.down_proj", ".mlp.experts.w2"), True
    
    return key, False


def convert_hf_moe_tensor_to_tt(key: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """
    Convert a HuggingFace MoE weight tensor to TorchTitan format.
    May return multiple tensors if the input needs to be split.
    
    HF gate_up_proj shape: (num_experts, hidden_size, 2 * intermediate_size)
    TT expects: w1 (num_experts, intermediate_size, hidden_size), w3 (num_experts, intermediate_size, hidden_size)
    
    HF down_proj shape: (num_experts, intermediate_size, hidden_size)
    TT w2 expects: (num_experts, hidden_size, intermediate_size)
    """
    if ".mlp.experts.gate_up_proj" in key:
        # Split gate_up_proj into w1 (gate) and w3 (up)
        # HF shape: (num_experts, hidden_size, 2 * intermediate_size)
        # Split along last dim, then transpose
        num_experts, hidden_size, double_intermediate = tensor.shape
        intermediate_size = double_intermediate // 2
        
        # Split into gate and up
        gate_proj = tensor[:, :, :intermediate_size]  # (num_experts, hidden_size, intermediate_size)
        up_proj = tensor[:, :, intermediate_size:]    # (num_experts, hidden_size, intermediate_size)
        
        # TorchTitan expects (num_experts, intermediate_size, hidden_size)
        w1 = gate_proj.transpose(1, 2).contiguous()  # w1 = gate_proj
        w3 = up_proj.transpose(1, 2).contiguous()    # w3 = up_proj
        
        base_key = key.replace(".mlp.experts.gate_up_proj", ".mlp.experts")
        return [(f"{base_key}.w1", w1), (f"{base_key}.w3", w3)]
    
    if ".mlp.experts.down_proj" in key or ".mlp.experts.w2" in key:
        # HF down_proj shape: (num_experts, intermediate_size, hidden_size)
        # TT w2 expects: (num_experts, hidden_size, intermediate_size)
        w2 = tensor.transpose(1, 2).contiguous()
        new_key = key.replace(".mlp.experts.down_proj", ".mlp.experts.w2")
        return [(new_key, w2)]
    
    # For other keys, no conversion needed
    return [(key, tensor)]


def is_moe_layer_key(key: str) -> bool:
    """Check if a key belongs to an MoE layer that needs conversion."""
    return (".mlp.gate.weight" in key or 
            ".mlp.experts.gate_up_proj" in key or 
            ".mlp.experts.down_proj" in key)


def apply_compile(model: nn.Module) -> None:
    """Apply torch.compile to model layers."""
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.capture_scalar_outputs = True

    layers = None
    # Handle different model structures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "active_layers"): # Qwen2-VL specific?
         # Need to check where layers are in Qwen3-VL
         # In Qwen2VL: model.visual (Vit) + model.model (Text)
         # In Qwen3VL Moe: model.language_model.model.layers ?
         pass 
         
    # Fallback/General search
    if layers is None or not isinstance(layers, (list, nn.ModuleList)):
        # Try to find layers in common locations
        if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
             layers = model.language_model.model.layers
    
    if layers is None:
        logger.warning("Could not find layers to compile. Skipping per-layer compilation.")
        return

    for layer in layers:
        layer.compile()

    logger.info("Applied torch.compile to model layers")


def initialize_moe_buffers(model: nn.Module) -> None:
    """
    Initialize TorchTitan MoE buffers after to_empty().
    
    TorchTitan MoE creates tokens_per_expert and expert_bias buffers that
    contain uninitialized memory after model.to_empty(). This function zeros
    them out to prevent NaN/garbage values during training.
    
    This should be called AFTER weight loading is complete.
    """
    initialized_count = 0
    for name, buf in model.named_buffers():
        if "mlp.tokens_per_expert" in name or "mlp.expert_bias" in name:
            buf.zero_()
            initialized_count += 1
            logger.debug(f"Zeroed MoE buffer: {name}")
    
    if initialized_count > 0:
        logger.info(f"Initialized {initialized_count} TorchTitan MoE buffers")


