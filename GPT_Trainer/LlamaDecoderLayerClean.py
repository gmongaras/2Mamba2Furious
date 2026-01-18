from typing import Callable, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    is_torch_flex_attn_available,
    logging,
)
from transformers.models.llama.configuration_llama import LlamaConfig
import math
from torch.utils.checkpoint import checkpoint
# from transformers.models.mamba2.modeling_mamba2 import Mamba2Block as Mamba2Block_SM
# from transformers.models.mamba2.modeling_mamba2 import Mamba2Config

from einops import rearrange
from causal_conv1d import causal_conv1d_fn
from kernel._2Mamba2Furious_square import _attention as _2Mamba2Furious_square
from kernel._2Mamba2Furious_exp import _attention as _2Mamba2Furious_exp
from kernel.LinearKernel import _attention as LinearKernel
from kernel.LinearKernelAMask import _attention as LinearKernelAMask
from kernel.LinearKernelSMNorm import _attention as LinearKernelSMNorm
from kernel.SquaredKernelAMask import _attention as SquaredKernelAMask
from Triton_Efficient_Kronecker_Product.kron import kron

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from rotary_embedding_torch import RotaryEmbedding






class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        """
        Absolute Positional Encoding module.

        Args:
            d_model (int): The dimension of the embeddings.
            max_len (int): Maximum sequence length for which to compute positional encodings.
            dropout (float): Dropout rate applied to positional encodings.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Not a parameter but persistent with the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Input tensor with positional encoding added, same shape as input
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)






logger = logging.get_logger(__name__)
elu = torch.nn.functional.elu
relu = torch.nn.functional.relu


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()

        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    

class LlamaRotaryEmbeddingPrecompute(nn.Module):
    def __init__(self, config: LlamaConfig, dim=None, device=None):
        super().__init__()

        if dim is not None:
            from copy import deepcopy
            config = deepcopy(config)
            config.hidden_size = config.head_dim = dim
            config.num_attention_heads = 1

        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.float(), sin.float()


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights








# Configs for various linear attention types
configs = {
    # (1) Full softmax attention
    "softmax": {
        "full_softmax_attention": True
    },
    
    # (2) Full mamba
    "mamba": {
        "full_mamba": True
    },
    
    # (3) Just normal linear attention
    "linear": {
        "use_kernel": True, 
        "power": "1",
        "qk_activation_type": "relu",
        "use_in_conv": False,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "sm_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "none",
        "precompute_attn_mask": True,
    },
    
    # (4) Changes from linear:
    # - use output norm instead of sm norm
    # - do not use ReLU
    "linear__output_norm": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": False,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "none",
        "precompute_attn_mask": True,
    },
    
    # (5) Changes from linear:
    # - use output norm instead of sm norm
    "linear__output_norm__ReLU": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "relu",
        "use_in_conv": False,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "none",
        "precompute_attn_mask": True,
    },
    
    # (6) + in conv (ws = 2)
    # Changes from linear:
    # - use output norm instead of sm norm
    # - in conv (window size of 2)
    "linear__output_norm__in_conv_k_2": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": True,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "none",
        "precompute_attn_mask": True,
    },
    
    # (7) + in conv (ws = 3)
    # Changes from linear:
    # - use output norm instead of sm norm
    # - in conv (window size of 3)
    "linear__output_norm__in_conv_k_3": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": True,
        "in_conv_dim": 3,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "none",
        "precompute_attn_mask": True,
    },
    # w=4
    "linear__output_norm__in_conv_k_4": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": True,
        "in_conv_dim": 4,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "none",
        "precompute_attn_mask": True,
    },
    
    # (8) + in conv (ws = 2) + silu
    # Changes from linear:
    # - use output norm instead of sm norm
    # - in conv (window size of 3)
    # - silu activation function
    "linear__output_norm__in_conv_k_2__silu": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "silu",
        "use_in_conv": True,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "none",
        "precompute_attn_mask": True,
    },
    # w=3
    "linear__output_norm__in_conv_k_3__silu": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "silu",
        "use_in_conv": True,
        "in_conv_dim": 3,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "none",
        "precompute_attn_mask": True,
    },
    # w=4
    "linear__output_norm__in_conv_k_4__silu": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "silu",
        "use_in_conv": True,
        "in_conv_dim": 4,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "none",
        "precompute_attn_mask": True,
    },
    
    # (9) + D res
    # Changes from linear:
    # - use output norm instead of sm norm
    # - D residual
    "linear__output_norm__D_res": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": False,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": True,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "none",
        "precompute_attn_mask": True,
    },
    
    # (10) + Z gate
    # Changes from linear:
    # - use output norm instead of sm norm
    # - Z output gate
    "linear__output_norm__Z_out_gate": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": False,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": True,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "none",
        "precompute_attn_mask": True,
    },
    
    # (11) + dt on values
    # Changes from linear:
    # - use output norm instead of sm norm
    # - dt on values (dt)
    "linear__output_norm__dt_on_values": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": False,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "dt",
        "A_mask_type": "none",
        "precompute_attn_mask": True,
    },
    
    # (12) + A mask (dt method from mamba 2)
    # Changes from linear:
    # - use output norm instead of sm norm
    # - use A mask (Mamba 2 discretize method)
    "linear__output_norm__A_mask_type_discretize": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": False,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "discretize",
        "precompute_attn_mask": True,
    },
    
    # (13) + A mask (neg_softplus method)
    # Changes from linear:
    # - use output norm instead of sm norm
    # - use A mask (neg_softplus method)
    "linear__output_norm__A_mask_type_neg_softplus": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": False,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "neg_softplus",
        "precompute_attn_mask": True,
    },
    
    
    
    
    
    
    
    # (14) + in conv + A mask (method 12, with dt)
    # Changes from linear:
    # - use output norm instead of sm norm
    # - use A mask (Mamba 2 discretize method)
    # - in conv (window size of 2)
    "linear__output_norm__A_mask_type_discretize__in_conv_k_2": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": True,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "discretize",
        "precompute_attn_mask": True,
    },
    
    # (15) + in conv + A mask (method 13, with A proj)
    # Changes from linear:
    # - use output norm instead of sm norm
    # - use A mask (neg_softplus method)
    # - in conv (window size of 2)
    "linear__output_norm__A_mask_type_neg_softplus__in_conv_k_2": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": True,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "neg_softplus",
        "precompute_attn_mask": True,
    },
    
    # (16) + in conv + A mask (method 13, with A proj) + discretize values 
    # Changes from linear:
    # - use output norm instead of sm norm
    # - use A mask (neg_softplus method)
    # - in conv (window size of 2)
    # - dt on values (dt)
    "linear__output_norm__A_mask_type_neg_softplus__in_conv_k_2__dt_on_values": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": True,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "dt",
        "A_mask_type": "neg_softplus",
        "precompute_attn_mask": True,
    },
    
    # (17) + in conv + A mask (method 13, with A proj) + discretize values + conv activation
    # Changes from linear:
    # - use output norm instead of sm norm
    # - use A mask (neg_softplus method)
    # - in conv (window size of 2)
    # - dt on values (dt)
    # - silu activation function
    "linear__output_norm__A_mask_type_neg_softplus__in_conv_k_2__dt_on_values__silu": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "silu",
        "use_in_conv": True,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "dt",
        "A_mask_type": "neg_softplus",
        "precompute_attn_mask": True,
    },
    
    # (18) + in conv + A mask (method 13, with A proj) + discretize values + D res
    # Changes from linear:
    # - use output norm instead of sm norm
    # - use A mask (neg_softplus method)
    # - in conv (window size of 2)
    # - dt on values (dt)
    # - D res
    "linear__output_norm__A_mask_type_neg_softplus__in_conv_k_2__dt_on_values__D_res": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": True,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": True,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "dt",
        "A_mask_type": "neg_softplus",
        "precompute_attn_mask": True,
    },
    
    # (19) + in conv + A mask (method 13, with A proj) + discretize values + Z gate
    # Changes from linear:
    # - use output norm instead of sm norm
    # - use A mask (neg_softplus method)
    # - in conv (window size of 2)
    # - dt on values (dt)
    # - Z output gate
    "linear__output_norm__A_mask_type_neg_softplus__in_conv_k_2__dt_on_values__Z_out_gate": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": True,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": True,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "dt",
        "A_mask_type": "neg_softplus",
        "precompute_attn_mask": True,
    },
    
    
    
    
    
    
    
    
    
    ### Squared kernels
    
    # (20) - (method 16 with A mask, in conv, and dt) + squared + output norm
    # Changes from linear:
    # - qk squared
    # - use output norm
    # - use A mask (neg_softplus method)
    # - in conv (window size of 2)
    # - dt on values (dt)
    "squared__output_norm__A_mask_type_neg_softplus__in_conv_k_2__dt_on_values": {
        "use_kernel": True,
        "power": "2",
        "qk_activation_type": "none",
        "use_in_conv": True,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "dt",
        "A_mask_type": "neg_softplus",
        "precompute_attn_mask": True,
    },
    
    # (21) - (method 16 with A mask, in conv, and dt) + squared + sm norm
    # Changes from linear:
    # - qk squared
    # - use sm norm
    # - use A mask (neg_softplus method)
    # - in conv (window size of 2)
    # - dt on values (dt)
    "squared__sm_norm__A_mask_type_neg_softplus__in_conv_k_2__dt_on_values": {
        "use_kernel": True,
        "power": "2",
        "qk_activation_type": "none",
        "use_in_conv": True,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "sm_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "dt",
        "A_mask_type": "neg_softplus",
        "precompute_attn_mask": True,
    },
    
    # basically 21 without dt
    # (22) - (method 16 with A mask and in conv) + squared + sm norm
    # Changes from linear:
    # - qk squared
    # - use sm norm
    # - use A mask (neg_softplus method)
    # - in conv (window size of 2)
    "squared__sm_norm__A_mask_type_neg_softplus__in_conv_k_2": {
        "use_kernel": True,
        "power": "2",
        "qk_activation_type": "none",
        "use_in_conv": True,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "sm_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "neg_softplus",
        "precompute_attn_mask": True,
    },
    
    
    # basically 22, but linear for experiments with the hidden state
    # (23) - (method 22 with A mask and in conv) + linear + out norm
    # Changes from 22:
    # - qk linear
    # - use out norm
    # - use A mask (neg_softplus method)
    # - in conv (window size of 2)
    "linear__out_norm__A_mask_type_neg_softplus__in_conv_k_2": {
        "use_kernel": True,
        "power": "1",
        "qk_activation_type": "none",
        "use_in_conv": True,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "output_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "neg_softplus",
        "precompute_attn_mask": True,
    },
    
    
    
    
    
    # (24) - (method 21 with A mask, in conv, and dt) + exp + sm norm
    # Changes from linear:
    # - qk exp
    # - use sm norm
    # - use A mask (neg_softplus method)
    # - in conv (window size of 2)
    # - dt on values (dt)
    "exp__sm_norm__A_mask_type_neg_softplus__in_conv_k_2__dt_on_values": {
        "use_kernel": True,
        "power": "exp",
        "qk_activation_type": "none",
        "use_in_conv": True,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "sm_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "dt",
        "A_mask_type": "neg_softplus",
        "precompute_attn_mask": True,
    },
    
    # (25) - (method 22 with A mask, in conv) + exp + sm norm
    # Changes from linear:
    # - qk exp
    # - use sm norm
    # - use A mask (neg_softplus method)
    # - in conv (window size of 2)
    "exp__sm_norm__A_mask_type_neg_softplus__in_conv_k_2": {
        "use_kernel": True,
        "power": "exp",
        "qk_activation_type": "none",
        "use_in_conv": True,
        "in_conv_dim": 2,
        "use_NoPE": True,
        "use_D_res": False,
        "use_z_out_gate": False,
        "norm_type": "sm_norm",
        "use_dt_bias": False,
        "value_disc_dtype": "none",
        "A_mask_type": "neg_softplus",
        "precompute_attn_mask": True,
    },
    
}








class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int, get_taylor_terms=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        
        
        # Bunch o stuff >w<
        self.head_dim = config.hidden_size // config.num_attention_heads
        assert self.head_dim == config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.is_causal = True
        self.attention_type = config.attention_type
        self.do_softmax_attention = False
        self.do_mamba = False

        
        
        
        
        # Params from config -->
        layer_config = configs[config.attention_type]
        
        
        # For inference time
        self.use_efficient = False
        self.hidden_conv = None
        self.hidden_num = None
        self.hidden_denom = None
        self.is_inference = False
                
        
        
        # Fast path - 
        # Full softmax attention will use RoPE and just have
        # the normal qkv projection + output projection
        if "full_softmax_attention" in layer_config.keys() and layer_config["full_softmax_attention"] == True:
            self.do_softmax_attention = True
            
            # RoPE
            self.rotary_emb = RotaryEmbedding(dim = self.head_dim)
            
            # Input and output projections
            qkv_dim = config.num_attention_heads * self.head_dim + 2 * config.num_key_value_heads * self.head_dim
            self.q_size = config.num_attention_heads * self.head_dim
            self.kv_size = config.num_key_value_heads * self.head_dim
            self.qkv_proj = nn.Linear(
                config.hidden_size, qkv_dim, bias=config.attention_bias
            )
            self.o_proj = nn.Linear(
                config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
            )
            
            # Softmax inference params
            self.k_cache = None
            self.v_cache = None
            
            return
            
        
        # Fast path - 
        # Full mamba will just use the mamba kernel from the official repo
        if "full_mamba" in layer_config.keys() and layer_config["full_mamba"] == True:
            from mamba_ssm import Mamba2
            
            self.do_mamba = True
            
            self.mamba_layer = Mamba2(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=config.hidden_size,
                d_conv=2,
                conv_init=None,
                expand=1,
                headdim=self.head_dim,
                d_ssm=self.head_dim,
                ngroups=1,
                A_init_range=(1, 16),
                D_has_hdim=True,
                rmsnorm=True,
                norm_before_gate=False,
                dt_min=0.001,
                dt_max=0.1,
                dt_init_floor=0.0001,
                dt_limit=(0, float("inf")),
                bias=False,
                conv_bias=False,
                chunk_size=256,
                use_mem_eff_path=True,
                layer_idx=layer_idx,
            )
            
            return
        
        
        
        # True to use a kernel. False to use torch ops
        self.use_kernel = layer_config["use_kernel"]
        
        # Power applied on q and k inner product
        self.power = layer_config["power"] # Can be one of "1", "2", or "exp"
        assert self.power in ["1", "2", "exp"]
        
        # Activation applied on q and k (after the input convolution)
        self.qk_activation_type = layer_config["qk_activation_type"]
        assert self.qk_activation_type in ["relu", "silu", "none"]
        
        # True to use the input time convolution on QKV, False to have no input conv
        self.use_in_conv = layer_config["use_in_conv"]
        self.in_conv_dim = layer_config["in_conv_dim"]
        
        # Use No poisitional encodings? If False, RoPE is used.
        self.NoPE = layer_config["use_NoPE"]
        if not self.NoPE:
            self.rotary_emb = RotaryEmbedding(dim = self.head_dim)
        
        # Use D residual or not?
        self.use_D_res = layer_config["use_D_res"]
        
        # Add the z output gate or not?
        self.use_z_out_gate = layer_config["use_z_out_gate"]
        
        """
        Norm type can be one of
        - "sm_norm" - Applies normalization like softmax, directly on q and k
        - "output_norm" - Applies the norm after all the operations, right before the output projection
        """
        self.norm_type = layer_config["norm_type"]
        assert self.norm_type in ["sm_norm", "output_norm"]
        
        # dt bias
        self.dt_bias = layer_config["use_dt_bias"] # Adds a bias to the dt value, pre softplus
        """
        Discretizes the values. 
        "dt" is what mamba uses
        Can take on one of:
        - "dt" - values = values * dt = values * softplus(dt)
        - "sigmoid" - values = values * sigmoid(dt)
        - "silu" - values = silu(values)
        - "softplus" - values = softplus(values)
        - "softplus2" - values = values*softplus(values)
        - "none" - No discretization
        """
        self.value_disc_dtype = layer_config["value_disc_dtype"]
        # self.value_disc_dtype = "dt"
        assert self.value_disc_dtype in ["dt", "sigmoid", "silu", "softplus", "softplus2", "none"]
        self.clamp_dt = False
        """
        A mask types can be one of (only if A mask is used)
        "discretize" is what mamba uses
        - "discretize" - Normal A mask: A_mask = -exp(A_log) * dt
        - "neg_softplus" - A_mask = -softplus(A)
        - "neg_softplus_dt" - A_mask = -softplus(A) * dt
        - "neg_softplus2" - A_mask = -A*softplus(A)
        - "neg_silu" - A_mask = -silu(A)
        - "none" - No A mask
        """
        self.A_mask_type = layer_config["A_mask_type"]
        # self.A_mask_type = "neg_softplus"
        assert self.A_mask_type in ["none", "discretize", "neg_softplus", "neg_softplus_dt", "neg_softplus2", "neg_silu"]





        # Combine the QKV projections
        qkv_dim = config.num_attention_heads * self.head_dim + 2 * config.num_key_value_heads * self.head_dim
        self.q_size = config.num_attention_heads * self.head_dim
        self.kv_size = config.num_key_value_heads * self.head_dim
        
        # Input and output projections
        self.qkv_proj = nn.Linear(
            config.hidden_size, qkv_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        
        # A mask projection
        if self.A_mask_type != "none":
            # We only have A_log if we are using
            # "discretize" for A_mask_type
            # This will create A mask via the dt projection
            if self.A_mask_type == "discretize":
                self.A_log = nn.Parameter(
                    torch.empty(config.num_attention_heads, dtype=torch.float32).uniform_(1, 16).log()
                )
            # Otherwise, we have a projection for the A mask
            # by itself (adding H more params)
            else:
                self.A_mask_proj = nn.Linear(
                    config.hidden_size, config.num_attention_heads, bias=config.attention_bias
                )

        # Input convolution
        if self.use_in_conv:
            d_conv = int(self.in_conv_dim)
            self.conv1d = nn.Conv1d(
                in_channels=qkv_dim,
                out_channels=qkv_dim,
                bias=True,
                kernel_size=d_conv,
                groups=qkv_dim,
                padding=d_conv - 1,
            )

            global causal_conv1d_fn
            from causal_conv1d import causal_conv1d_fn
            
           
        # qk activation function
        if self.qk_activation_type == "silu":
            self.qk_act = torch.nn.functional.silu
        elif self.qk_activation_type == "relu":
            self.qk_act = torch.nn.functional.relu
        elif self.qk_activation_type == "none":
            self.qk_act = torch.nn.Identity()
        else:
            assert False
            
        # dt projection
        if self.value_disc_dtype == "dt" \
            or self.value_disc_dtype == "sigmoid" \
            or self.A_mask_type == "discretize" \
            or self.A_mask_type == "neg_softplus_dt":
            
            self.dt_proj = nn.Linear(
                config.hidden_size, config.num_attention_heads, bias=config.attention_bias
            )
        
        # dt bias
        if self.dt_bias:
            assert self.value_disc_dtype == "dt" \
                or self.value_disc_dtype == "sigmoid" \
                or self.A_mask_type == "discretize" \
                or self.A_mask_type == "neg_softplus_dt", \
                "dt bias can only be added when dt is used"
            
            # Initialize log dt bias
            dt_max = 0.1
            dt_min = 0.001
            dt = torch.exp(
                torch.rand(config.num_attention_heads) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=1e-4)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias_value = nn.Parameter(inv_dt[None, None, :])
            # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            self.dt_bias_value._no_weight_decay = True
        else:
            self.dt_bias_value = 0
            
        # D residual
        if self.use_D_res:
            self.D = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        
        
        # z output gate requires another projection of the input
        if self.use_z_out_gate:
            self.z_gate_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
            
            
        # Ouptput norm
        if self.norm_type == "output_norm":
            self.out_norm = nn.RMSNorm(self.num_heads * self.head_dim)
            
        
        # Precompute attention mask if we are not using a kernel
        if not self.use_kernel or True:
            max_seq_len = config.max_position_embeddings
            self.register_buffer(
                "attn_mask",
                ~torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()[None, None, :, :]
            )



    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        
        
        
        
        
        # Fast path - full softmax attention
        if self.do_softmax_attention:
            # Combined QKV proj
            QKV = self.qkv_proj(hidden_states)

            # Get QKV tensors
            query_states = QKV[:, :, :self.q_size]
            key_states = QKV[:, :, self.q_size:self.q_size+self.kv_size]
            value_states = QKV[:, :, self.q_size+self.kv_size:]
            
            # Add heads
            query_states = query_states.view(hidden_shape).transpose(1, 2)
            key_states = key_states.view(hidden_shape).transpose(1, 2)
            value_states = value_states.view(hidden_shape).transpose(1, 2)
            
            # Efficient path will use the KV cache
            if self.is_inference and self.use_efficient and self.k_cache is not None:
                # Concat past keys and values
                key_states = torch.cat((self.k_cache, key_states), dim=-2)
                value_states = torch.cat((self.v_cache, value_states), dim=-2)
                
                # Offset for queries
                RoPE_offset = key_states.shape[-2]-1
                causal_ = False
            else:
                RoPE_offset = 0
                causal_ = True
                
            # Store KV cache
            if self.is_inference and self.use_efficient:
                # Store KV cache
                self.k_cache = key_states.clone()
                self.v_cache = value_states.clone()
                
            # RoPE
            query_states = self.rotary_emb.rotate_queries_or_keys(query_states, offset=RoPE_offset)
            key_states = self.rotary_emb.rotate_queries_or_keys(key_states)
                
            # sdpa because I don't feel like writing a kernel
            def forward_(query, key, value, scale):
                return torch.nn.functional.scaled_dot_product_attention(
                                query,
                                key, 
                                value, 
                                attn_mask=None, 
                                dropout_p=0.0, 
                                is_causal=causal_,  
                                scale=scale, 
                                enable_gqa=False
                )
            attn_output = checkpoint(
                forward_, query_states, key_states, value_states, (1/math.sqrt(key_states.shape[-1])),
            )
                
            # Remove heads
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()

            # Output projection
            return self.o_proj(attn_output), None
        
        # Fast path - Full mamba
        if self.do_mamba:
            def forward_(hidden_states):
                return self.mamba_layer(hidden_states)
            
            attn_output = checkpoint(
                forward_, hidden_states,
            )
            return attn_output, None
        
        
        
        


        # Combined QKV proj
        QKV = self.qkv_proj(hidden_states)

        # # Apply input convolution
        # if self.use_in_conv:
        #     QKV = causal_conv1d_fn(
        #         x=QKV.transpose(1, 2).contiguous(),
        #         weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
        #         bias=self.conv1d.bias,
        #         activation=None,
        #     ).transpose(1, 2)
        # Apply input convolution
        if self.use_efficient:
            if self.use_in_conv:
                # Append the previous part of the sequence
                assert self.conv1d.weight.shape[-1] == 2, "conv1d dimensions larger than 2 are not supported, but can be easily lol"
                h_is_none = self.hidden_conv is None
                if not h_is_none:
                    QKV = torch.cat([self.hidden_conv, QKV], dim=-2)
                    
                # Save the last token
                self.hidden_conv = QKV[:, -1:]
                    
                # Do the conv
                QKV = causal_conv1d_fn(
                    x=QKV.transpose(1, 2).contiguous(),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=None,
                ).transpose(1, 2)
                
                # Get the last token in the sequence if it's not the first pass
                if not h_is_none:
                    QKV = QKV[:, -1:]
        else:
            if self.use_in_conv:
                QKV = causal_conv1d_fn(
                    x=QKV.transpose(1, 2).contiguous(),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=None,
                ).transpose(1, 2)

        # Get QKV tensors
        query_states = QKV[:, :, :self.q_size]
        key_states = QKV[:, :, self.q_size:self.q_size+self.kv_size]
        value_states = QKV[:, :, self.q_size+self.kv_size:]
        
        # Add heads
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # Get dt
        if self.value_disc_dtype == "dt" \
            or self.value_disc_dtype == "sigmoid" \
            or self.A_mask_type == "discretize" \
            or self.A_mask_type == "neg_softplus_dt":
            
            # dt projection
            dt = self.dt_proj(hidden_states)
                
            # transform dt
            if self.value_disc_dtype == "sigmoid":
                dt = torch.nn.functional.sigmoid(dt + self.dt_bias_value)
            else:
                dt = torch.nn.functional.softplus(dt + self.dt_bias_value)
            
            # Clamping dt
            if self.clamp_dt:
                dt = dt.clamp(max=10)
                
        # Get A mask
        if self.A_mask_type != "none":
            # If the mask type is "discretize", we calculate the
            # mask like normal
            # A_mask = -exp(A)*dt
            if self.A_mask_type == "discretize":
                A = -torch.exp(self.A_log)[None, None, :]*dt
            # Otherwise, we get the additional A projection
            # and calculate the A mask with that
            else:
                # Get A mask projection
                A = self.A_mask_proj(hidden_states)
                
                # A_mask = -softplus(A)
                if self.A_mask_type == "neg_softplus":
                    A = -torch.nn.functional.softplus(A)
                # A_mask = -softplus(A)
                elif self.A_mask_type == "neg_softplus_dt":
                    A = -torch.nn.functional.softplus(A) * dt
                # A_mask = -A*softplus(A)
                elif self.A_mask_type == "neg_softplus2":
                    A = -A*torch.nn.functional.softplus(A)
                # A_mask = -silu(A) = -A*sigmoid(A)
                elif self.A_mask_type == "neg_silu":
                    A = -torch.nn.functional.silu(A)
                else:
                    assert False
        else:
            A = None

        # D res comes from multiplying D with the values before transforming the values at all
        if self.use_D_res:
            D_res = value_states * self.D[None, :, None, :,]

        # Discretization for value heads as done in Mamba
        if self.value_disc_dtype != "none":
            assert self.value_disc_dtype in ["dt", "silu", "softplus", "softplus2", "sigmoid"]
            if self.value_disc_dtype == "dt" or self.value_disc_dtype == "sigmoid":
                value_states = value_states * dt.mT[..., None]
            elif self.value_disc_dtype == "silu":
                value_states = torch.nn.functional.silu(value_states)
            elif self.value_disc_dtype == "softplus":
                value_states = torch.nn.functional.softplus(value_states)
            elif self.value_disc_dtype == "softplus2":
                value_states = value_states*torch.nn.functional.softplus(value_states)
            else:
                assert False

        # Apply activation function
        query_states = self.qk_act(query_states)
        key_states = self.qk_act(key_states)

        # RoPE
        if not self.NoPE:
            query_states = self.rotary_emb.rotate_queries_or_keys(query_states)
            key_states = self.rotary_emb.rotate_queries_or_keys(key_states)
            
        # Create z output gate
        if self.use_z_out_gate:
            z_out_gate = torch.nn.functional.silu(self.z_gate_proj(hidden_states)).view(hidden_shape).transpose(1, 2)


        
        def forwrd_gated(query_states, key_states, value_states, attention_mask, A):
            # return norm((1/2) * ((query_states @ key_states.mT * (1/math.sqrt(key_states.shape[-1]))).masked_fill(attention_mask!=0, torch.tensor(0.0, device=query_states.device)) @ value_states.float())**2)
            
            if self.power == "exp":
                attn_weights = (query_states @ key_states.mT * (1/math.sqrt(key_states.shape[-1]))).exp()
            else:
                attn_weights = (query_states @ key_states.mT * (1/math.sqrt(key_states.shape[-1])))**int(self.power)
            
            # Apply A mask
            if self.A_mask_type != "none":
                A_cumsum = torch.cumsum(A.float(), dim=-2).mT
                A_mask = (((A_cumsum[:, :, :, None] - A_cumsum[:, :, None, :]))).masked_fill(attention_mask.bool(), -torch.inf).exp().to(query_states.dtype)
                attn_weights = attn_weights * A_mask
            else:
                attn_weights = attn_weights.masked_fill(attention_mask[:, :, :query_states.shape[2], :key_states.shape[2]].clone().cuda(), torch.tensor(0.0, device=query_states.device))
                # attn_weights = attn_weights.masked_fill(attention_mask!=0, torch.tensor(0.0, device=query_states.device))

            # Denominator
            if self.norm_type == "sm_norm":
                denom = attn_weights.sum(dim=-1, keepdim=True)
                assert torch.all(denom >= 0), "ayo wtf why are there negatives in the denom?"
                
                # Denominators that are zero will be turned into ones as the output
                # No gradient but whatever. This should be rare enough it doesn't matter.
                # Better than arbitrary clamping the denom which could cause large values if
                # an outlier occurs.
                attn_weights = torch.where(
                    denom > 0,
                    attn_weights / denom,
                    # Trick since attn mask in binary
                    (~attention_mask).float()
                )

            # Denominator
            # attn_weights = attn_weights / (attn_weights.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            
            # Output
            return attn_weights @ value_states.float()
            
            
            
        # Inference 
        if self.is_inference:
            if self.use_efficient:
                # If the numerator and denominator are None, we need to compute the hidden state manually
                if self.hidden_num is None:
                    # Compute outputs like normal cause it's easy this way
                    attention_mask = ~torch.tril(torch.ones(query_states.shape[2], query_states.shape[2])).bool().repeat(query_states.shape[0], query_states.shape[1], 1, 1).to(query_states.device)
                    attn_output = forwrd_gated(query_states, key_states, value_states, attention_mask, A)
                    
                    # Multiply keys by scale factor
                    key_states = key_states * (1/math.sqrt(key_states.shape[-1]))
                    
                    # self-kronecker for queries and key
                    query_states = kron(query_states.float())
                    key_states = kron(key_states.float())
                    # query_states = (query_states[..., :, None] * query_states[..., None, :]).flatten(-2, -1)
                    # key_states = (key_states[..., :, None] * key_states[..., None, :]).flatten(-2, -1)
                    
                    # Compute A values
                    A = A.mT[..., None].exp()
                    
                    # Iterate over sequence and produce hidden state
                    self.hidden_num = self.hidden_denom = 0
                    for t in range(0, key_states.shape[-2]):
                        # Compute next hidden states
                        A_ = A[:, :, t:t+1]
                        K = key_states[:, :, t:t+1].mT
                        V = value_states[:, :, t:t+1]
                        self.hidden_num = self.hidden_num * A_ + (K @ V)
                        self.hidden_denom = self.hidden_denom * A_ + K
                    
                # If previous hidden states exist, we can reuse them
                else:
                    # Multiply keys by scale factor
                    key_states = key_states * (1/math.sqrt(key_states.shape[-1]))
                    
                    # self-kronecker for queries and key
                    query_states = kron(query_states, 1)
                    key_states = kron(key_states, 1)
                    
                    # Compute alpha value at current timestep
                    A = A.mT[..., None].exp()
                    
                    # Update hidden states
                    self.hidden_num = self.hidden_num * A + (key_states.mT @ value_states)
                    self.hidden_denom = self.hidden_denom * A + key_states.mT
                    
                    # Calculate output
                    attn_output = (query_states @ self.hidden_num) / (query_states @ self.hidden_denom)
            else:
                attention_mask = ~torch.tril(torch.ones(query_states.shape[2], query_states.shape[2])).bool().repeat(query_states.shape[0], query_states.shape[1], 1, 1).to(query_states.device)
                attn_output = forwrd_gated(query_states, key_states, value_states, attention_mask, A)
        
        # Not inference
        else:
            assert self.power in ["1", "2", "exp"]
            # No kernel used
            if not self.use_kernel:
                attn_output = torch.utils.checkpoint.checkpoint(
                    forwrd_gated,
                    query_states,
                    key_states,
                    value_states,
                    self.attn_mask[:, :, :query_states.shape[2], :query_states.shape[2]].clone(),
                    A
                )
            # Kernel used
            else:
                if self.power == "1":
                    # # Mamba kernel equivalence assuming use_sm_norm is set to False
                    # out_mamba = mamba_chunk_scan_combined(
                    #     (value_states / dt.mT[..., None]).transpose(1, 2),
                    #     dt,
                    #     -torch.exp(self.A_log),
                    #     key_states.transpose(1, 2) * (1/math.sqrt(key_states.shape[-1])),
                    #     query_states.transpose(1, 2),
                    #     chunk_size=32,
                    #     D=self.D if self.use_D_res else None,
                    #     z=self.z_gate_proj(hidden_states).view(hidden_shape) if self.use_z_out_gate else None,
                    # ).transpose(1, 2)
                    # attention_mask = ~torch.tril(torch.ones(query_states.shape[2], query_states.shape[2])).bool().repeat(query_states.shape[0], query_states.shape[1], 1, 1).to(query_states.device)
                    
                    # No normalization in the kernel
                    if self.norm_type == "output_norm":
                        # Kernel, no A mask
                        if A is None:
                            attn_output = LinearKernel.apply(
                                query_states.float(), 
                                key_states.float(), 
                                value_states.float(), 
                                True, 
                                (1/math.sqrt(key_states.shape[-1])), 
                                False
                            )
                        # Kernel with A mask
                        else:
                            attn_output = LinearKernelAMask.apply(
                                query_states.float(), 
                                key_states.float(), 
                                value_states.float(), 
                                A.mT.float().cumsum(-1),
                                True, 
                                (1/math.sqrt(key_states.shape[-1])), 
                                False
                            )
                    # sm normalization in the kernel
                    elif self.norm_type == "sm_norm":
                        # Kernel, no A mask
                        if A is None:
                            attn_output = LinearKernelSMNorm.apply(
                                query_states.float(), 
                                key_states.float(), 
                                value_states.float(), 
                                True, 
                                (1/math.sqrt(key_states.shape[-1])), 
                                False
                            )
                        # Kernel with A mask
                        else:
                            assert False
                elif self.power == "2":
                    # I only have kernels for A masks
                    if A is None:
                        assert False
                        
                    # No normalization in the kernel
                    if self.norm_type == "output_norm":
                        attn_output = SquaredKernelAMask.apply(
                            query_states.float(), 
                            key_states.float(), 
                            value_states.float(), 
                            A.mT.float().cumsum(-1),
                            True, 
                            (1/math.sqrt(key_states.shape[-1])), 
                            False
                        )
                    # Online sm norm in the kernel
                    elif self.norm_type == "sm_norm":
                        attn_output = _2Mamba2Furious_square.apply(
                            query_states.float(), 
                            key_states.float(), 
                            value_states.float(), 
                            A.mT.float().cumsum(-1),
                            True, 
                            1/math.sqrt(key_states.shape[-1]), 
                            False
                        )
                    else:
                        assert False
                elif self.power == "exp":
                    # No normalization in the kernel
                    if self.norm_type == "output_norm":
                        assert False
                    # online sm norm in the kernel
                    elif self.norm_type == "sm_norm":
                        if A is None:
                            # exp with no A mask is just softmax
                            assert False
                        else:
                            attn_output = _2Mamba2Furious_exp.apply(
                                query_states.float(), 
                                key_states.float(), 
                                value_states.float(), 
                                A.mT.float().cumsum(-1),
                                True, 
                                1/math.sqrt(key_states.shape[-1]), 
                                False
                            )
                    else:
                        assert False
                else:
                    assert False
            # attention_mask = ~torch.tril(torch.ones(query_states.shape[2], query_states.shape[2])).bool().repeat(query_states.shape[0], query_states.shape[1], 1, 1).to(query_states.device)
            # attn_output = forwrd_gated(query_states, key_states, value_states, attention_mask, A)


        # torch.save(query_states, "debug_output/query_states")
        # torch.save(key_states, "debug_output/key_states")
        # torch.save(value_states, "debug_output/value_states")
        # torch.save(A.mT.float().cumsum(-1), "debug_output/A_cumsum")
        
        
        # Apply the D residual
        if self.use_D_res:
            attn_output = attn_output + D_res
            
        # Apply the z output gate
        if self.use_z_out_gate:
            attn_output = attn_output * z_out_gate
        
        # Remove heads
        #### NOTE: For some reason huggingface decided to put the transpose in
        ####       the function (I think due to flash attn). I put it back here
        ####       but this means normal huggingface functions will have to be transposed :/
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        
        # Output norm
        if self.norm_type == "output_norm":
            attn_output = self.out_norm(attn_output)

        # Output projection
        attn_output = self.o_proj(attn_output)


        return attn_output, None
    














class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, get_taylor_terms=False):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx, get_taylor_terms=get_taylor_terms)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = checkpoint(
            self.mlp, hidden_states, use_reentrant=True
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

if __name__ == "__main__":
    B, L, d = 16, 1024, 1024+512
    
    import transformers
    config = transformers.LlamaConfig.from_dict({
            "_name_or_path": "meta-llama/Llama-2-7b-hf",
            "architectures": [
                "LlamaForCausalLM"
            ],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": d,
            "initializer_range": 0.02,
            "intermediate_size": d*2,
            "max_position_embeddings": L*2,
            "model_type": "llama",
            "num_attention_heads": 24,
            "num_hidden_layers": 27,
            "num_key_value_heads": 24,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "use_cache": True,
            "vocab_size": 1024,
            "attention_type": "exp__sm_norm__A_mask_type_neg_softplus__in_conv_k_2",
        })
    layer = LlamaDecoderLayer(config=config, layer_idx=0).cuda()
    hidden_states = torch.randn(B, L, d).cuda()
    position_embeddings = torch.arange(0, L)[None, :].repeat(B, 1).cuda()
    # attention_mask = torch.ones(L)[None, :].repeat(B, 1).cuda()
    attention_mask = None
    out = layer.forward(
        hidden_states,
    )
    # out.sum().backward()
    
    
    with torch.autograd.profiler.profile(use_device="cuda", with_stack=True, with_flops=True) as prof:
        with torch.autograd.profiler.record_function("fw_pass"):
            loss = layer.forward(hidden_states,)[0].sum()
        with torch.autograd.profiler.record_function("bw_pass"):
            loss.backward() 
    
    print(prof.key_averages().table(sort_by="cuda_time_total"))