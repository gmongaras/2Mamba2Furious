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
from Triton_Efficient_Kronecker_Product.kron import kron






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



class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int, get_taylor_terms=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx








        self.use_exp = False

        # Params
        self.powers = [2]
        self.learnable_coefficients = False
        self.gamma_domain = False
        self.use_norm = False
        self.in_conv = True
        self.low_rank_heads = False
        self.NoPE = True
        
        # Only for A masking
        self.A_mask = True
        self.dt_bias = False # Adds a bias to the dt value, pre softplus
        """
        Discretizes the values. 
        "dt" is what mamba uses
        Can take on one of:
        - "dt" - values = values * dt
        - "silu" - values = silu(values)
        - "softplus" - values = softplus(values)
        - "softplus2" - values = values*softplus(values)
        - "none" - No discretization
        """
        self.A_mask_value_dist_type = "dt"
        """
        A mask types can be one of
        "discretize" is what mamba uses
        - "discretize" - Normal A mask: A_mask = -exp(A_log) * dt
        - "neg_softplus" - A_mask = -softplus(A)
        - "neg_softplus2" - A_mask = -A*softplus(A)
        - "neg_silu" - A_mask = -silu(A)
        """
        self.A_mask_type = "neg_softplus"









        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        if self.low_rank_heads:
            # Setting number of heads to 1
            self.num_heads = config.num_attention_heads
            self.config.num_attention_heads = self.config.num_key_value_heads = config.num_attention_heads = config.num_key_value_heads = 1
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.attention_type = config.attention_type





        # Combine the QKV projections
        all_dim = config.num_attention_heads * self.head_dim + 2 * config.num_key_value_heads * self.head_dim
        if self.low_rank_heads:
            # Adding low rank heads
            self.all_dim_no_heads = all_dim
            self.all_dim_heads = all_dim+self.num_heads*3
            all_dim = self.all_dim_heads
        if self.A_mask:
            all_dim_ = all_dim
            all_dim = all_dim + config.num_attention_heads
            
            # We only have A_log if we are using
            # "discretize" for A_mask_type
            if self.A_mask_type == "discretize":
                self.A_log = nn.Parameter(
                    torch.empty(config.num_attention_heads, dtype=torch.float32).uniform_(1, 16).log()
                )
            # Otherwise, we increase the all dimension dimension
            # by adding H more params
            else:
                all_dim = all_dim + config.num_attention_heads
            
            # Bias
            if self.dt_bias:
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
        self.q_size = config.num_attention_heads * self.head_dim
        self.kv_size = config.num_key_value_heads * self.head_dim
        self.qkv_proj = nn.Linear(
            config.hidden_size, all_dim, bias=config.attention_bias
        )
        if self.low_rank_heads:
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
            )
        else:
            self.o_proj = nn.Linear(
                config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
            )


        if self.in_conv:
            d_conv = 2
            if self.low_rank_heads:
                self.conv1d = nn.Conv1d(
                    in_channels=self.all_dim_no_heads,
                    out_channels=self.all_dim_no_heads,
                    bias=True,
                    kernel_size=d_conv,
                    groups=self.all_dim_no_heads,
                    padding=d_conv - 1,
                )
            elif self.A_mask:
                self.conv1d = nn.Conv1d(
                    in_channels=all_dim_,
                    out_channels=all_dim_,
                    bias=True,
                    kernel_size=d_conv,
                    groups=all_dim_,
                    padding=d_conv - 1,
                )
            else:
                self.conv1d = nn.Conv1d(
                    in_channels=all_dim,
                    out_channels=all_dim,
                    bias=True,
                    kernel_size=d_conv,
                    groups=all_dim,
                    padding=d_conv - 1,
                )

            global causal_conv1d_fn
            from causal_conv1d import causal_conv1d_fn


        # Coefficients for all orders (initialized to 1/n!)
        if not self.use_exp:
            if self.gamma_domain:
                order_coefficients = torch.tensor(self.powers, dtype=torch.float)[:, None, None, None].repeat(1, config.num_attention_heads, 1, 1).float()
                # self.register_buffer("order_coefficients", torch.tensor([i+0.5 for i in range(0, self.order+1)], dtype=torch.float)[:, None, None, None].repeat(1, config.num_attention_heads, 1, 1).float())
            else:
                # self.order_coefficients = nn.Parameter(torch.tensor([1/math.factorial(i) for i in range(0, self.order+1)])[:, None, None, None].repeat(1, config.num_attention_heads, 1, 1).float())
                order_coefficients = torch.tensor([0.0 for i in self.powers])[:, None, None, None].repeat(1, config.num_attention_heads, 1, 1).float()
            
            if self.learnable_coefficients:
                self.order_coefficients = nn.Parameter(order_coefficients)
            else:
                self.register_buffer("order_coefficients", order_coefficients)
        else:
            self.order_coefficients = None
        # self.order_coefficients = nn.Parameter((1/torch.arange(1, self.order+1).cumprod(-1))[:, None, None, None].repeat(1, config.num_attention_heads, 1, 1).float())
        # self.register_buffer("order_coefficients", (1/torch.arange(1, self.order+1).cumprod(-1))[:, None, None, None].repeat(1, config.num_attention_heads, 1, 1).float())

        # # Matrices for each order
        # min_val = 4
        # sizes = [self.head_dim for i in range(1, self.order+1)]
        # self.A_projs = nn.ParameterList([
        #     torch.eye(self.head_dim) for i in range(1, self.order+1)
        # ])
        # self.B_projs = nn.ParameterList([
        #     torch.eye(self.head_dim) for i in range(1, self.order+1)
        # ])
        
        # position_ids = torch.arange(0, 1024)
        # self.rope_embeddings = [
        #     torch.stack(LlamaRotaryEmbeddingPrecompute(config, dim=sizes[i-1])(position_ids[None, :])) \
        #         for i in range(1, self.order+1)
        # ]
        

        # Output Norm
        if self.use_norm:
            self.out_norm = nn.RMSNorm(config.head_dim)
        else:
            self.out_norm = nn.Identity()
            
        # For inference time
        self.use_efficient = False
        self.hidden_conv = None
        self.hidden_num = None
        self.hidden_denom = None
        self.is_inference = False



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

        # Add positional encodings
        # hidden_states = self.abs_pos_enc(hidden_states)





        # Combined QKV
        QKV = self.qkv_proj(hidden_states)
        if self.low_rank_heads:
            # Get heads
            heads_ = torch.nn.functional.softplus(QKV[:, :, self.all_dim_no_heads:])
            QKV = QKV[:, :, :self.all_dim_no_heads]
            query_heads, key_heads, value_heads = heads_.split(self.num_heads, dim=-1)
        if self.A_mask:
            # Get dt projection
            dt = torch.nn.functional.softplus(QKV[:, :, -self.config.num_attention_heads:] + self.dt_bias_value)
            QKV = QKV[:, :, :-self.config.num_attention_heads]
            
            # If the mask type is "discretize", we calculate the
            # mask like normal
            # A_mask = -exp(A)*dt
            if self.A_mask_type == "discretize":
                A = -torch.exp(self.A_log)[None, None, :]*dt
            # Otherwise, we get the additional A projection
            # and calculate the A mask with that
            else:
                # Get A mask projection
                A = QKV[:, :, -self.config.num_attention_heads:]
                QKV = QKV[:, :, :-self.config.num_attention_heads]
                
                # A_mask = -softplus(A)
                if self.A_mask_type == "neg_softplus":
                    A = -torch.nn.functional.softplus(A)
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

        # Apply input convolution
        if self.use_efficient:
            if self.in_conv:
                # Append the previous part of the sequence
                assert self.conv1d.weight.shape[-1] == 2, "conv1d dimensions larger than 2 are not supported, but can be easily lol"
                h_is_none = self.hidden_conv is None
                if not h_is_none:
                    QKV = torch.cat([self.hidden_conv, QKV], dim=-2)
                    
                # Save the last token
                self.hidden_conv = QKV[:, -1:].clone()
                    
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
            if self.in_conv:
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
        if self.low_rank_heads:
            query_states = query_states * query_heads.mT[:, :, :, None]
            key_states = key_states * key_heads.mT[:, :, :, None]
            value_states = value_states * value_heads.mT[:, :, :, None]

        # Discretization for value heads as done in Mamba
        if self.A_mask and self.A_mask_value_dist_type != "none":
            assert self.A_mask_value_dist_type in ["dt", "silu", "softplus", "softplus2"]
            if self.A_mask_value_dist_type == "dt":
                value_states = value_states * dt.mT[..., None]
            elif self.A_mask_value_dist_type == "silu":
                value_states = torch.nn.functional.silu(value_states)
            elif self.A_mask_value_dist_type == "softplus":
                value_states = torch.nn.functional.softplus(value_states)
            elif self.A_mask_value_dist_type == "softplus2":
                value_states = value_states*torch.nn.functional.softplus(value_states)
            else:
                assert False

        # RoPE
        if not self.NoPE:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)





        
        def forwrd_gated(query_states, key_states, value_states, attention_mask, order_coefficients, norm, A):
            # return norm((1/2) * ((query_states @ key_states.mT * (1/math.sqrt(key_states.shape[-1]))).masked_fill(attention_mask!=0, torch.tensor(0.0, device=query_states.device)) @ value_states.float())**2)
            
            if self.use_exp:
                attn_weights = (query_states @ key_states.mT * (1/math.sqrt(key_states.shape[-1]))).exp()
            else:
                # Convert if gamma domain
                if self.gamma_domain:
                    # if self.cumulative:
                    powers = order_coefficients.clamp(min=0)
                    coefs = 1/torch.lgamma(powers+1).exp().cfloat()
                    attn_mat = (query_states.cfloat() @ key_states.cfloat().mT * (1/math.sqrt(key_states.shape[-1])))
                    attn_weights = 0
                    for iter_num, i in enumerate(self.powers):
                        attn_weights = attn_weights + coefs[iter_num:iter_num+1] * attn_mat**powers[iter_num:iter_num+1]
                    # else:
                    #     powers = order_coefficients.cfloat().clamp(min=0)
                    #     coefs = 1/torch.lgamma(powers+1).exp()
                    #     attn_weights = (query_states.cfloat() @ key_states.cfloat().mT * (1/math.sqrt(key_states.shape[-1])))
                    #     attn_weights = coefs * attn_mat**powers

                    # Only teh real part matters
                    attn_weights = attn_weights.real
                
                else:
                    # Accumulate over orders
                    factor_ = 1/math.sqrt(key_states.shape[-1])

                    # Convert to a value between 0 and 2
                    order_coefficients = 2*order_coefficients.sigmoid()

                    # if self.cumulative:
                    attn_weights = 0
                    for iter_num, _i in enumerate(self.powers):
                        weight_ = order_coefficients[iter_num:iter_num+1] * (1/math.factorial(_i))
                        if _i == 0:
                            attn_weights = attn_weights + weight_
                        else:
                            attn_weights = attn_weights + (weight_ * (query_states.float() @ key_states.float().mT * factor_)**_i)
                    # else:
                    #     attn_weights = order_coefficients * (1/math.factorial(self.order)) * ((query_states.float() @ key_states.float().mT * factor_)**self.order)
            
            # Apply A mask
            if self.A_mask:
                A_cumsum = torch.cumsum(A.float(), dim=-2).mT
                A_mask = (((A_cumsum[:, :, :, None] - A_cumsum[:, :, None, :]))).masked_fill(attention_mask.bool(), -torch.inf).exp().to(query_states.dtype)
                attn_weights = attn_weights * A_mask
            else:
                attn_weights = attn_weights.masked_fill(attention_mask!=0, torch.tensor(0.0, device=query_states.device))

            # Denominator
            if not self.use_norm:
                attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

            # Denominator
            # attn_weights = attn_weights / (attn_weights.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            
            # Output
            return norm(attn_weights @ value_states.float())
            # return (query_states @ key_states.mT / math.sqrt(key_states.shape[-1]) + attention_mask).softmax(dim=-1) @ value_states
            
            
        # attn_output_ = checkpoint(
        #     forwrd_gated, query_states.clone().half(), key_states.clone().half(), value_states.clone().half(), attention_mask, self.order_coefficients, self.out_norm, A, use_reentrant=False
        # )
        
        is_inference = self.is_inference
        if is_inference:
            if self.use_efficient:
                # If the numerator and denominator are None, we need to compute the hidden state manually
                if self.hidden_num is None:
                    # Compute outputs like normal cause it's easy this way
                    attention_mask = ~torch.tril(torch.ones(query_states.shape[2], query_states.shape[2])).bool().repeat(query_states.shape[0], query_states.shape[1], 1, 1).to(query_states.device)
                    attn_output = forwrd_gated(query_states, key_states, value_states, attention_mask, self.order_coefficients, self.out_norm, A)
                    
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
                    # query_states = (query_states[..., :, None] * query_states[..., None, :]).flatten(-2, -1)
                    # key_states = (key_states[..., :, None] * key_states[..., None, :]).flatten(-2, -1)
                    
                    # Compute alpha value at current timestep
                    A = A.mT[..., None].exp()
                    
                    # Update hidden states
                    self.hidden_num = self.hidden_num * A + (key_states.mT @ value_states)
                    self.hidden_denom = self.hidden_denom * A + key_states.mT
                    
                    # Calculate output
                    attn_output = (query_states @ self.hidden_num) / (query_states @ self.hidden_denom)
            else:
                attention_mask = ~torch.tril(torch.ones(query_states.shape[2], query_states.shape[2])).bool().repeat(query_states.shape[0], query_states.shape[1], 1, 1).to(query_states.device)
                attn_output = forwrd_gated(query_states, key_states, value_states, attention_mask, self.order_coefficients, self.out_norm, A)
        else:
            A_cumsum = A.mT.float().cumsum(-1)
            attn_output = _2Mamba2Furious_square.apply(query_states.half(), key_states.half(), value_states.half(), A_cumsum.float(), True, 1/math.sqrt(key_states.shape[-1]), False)








        # Remove heads, output projection
        #### NOTE: For some reason huggingface decided to put the transpose in
        ####       the function (I think due to flash attn). I put it back here
        ####       but this means normal huggingface functions will have to be transposed :/
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
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
