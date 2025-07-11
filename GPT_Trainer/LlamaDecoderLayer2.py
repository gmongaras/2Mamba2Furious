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






class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
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
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.attention_type = config.attention_type





        # Params
        self.no_rope = True         # True
        self.no_A_mask = False      # False
        out_norm = True             # True
        q_conv = True               # True
        k_conv = True               # True
        v_conv = True               # True
        self.order = 1              # 1
        no_A = False                # False
        self.no_value_dt = False    # False
        self.act = None             # None
        self.no_bias = False        # False
        d_conv = 2                  # 2






        # Combine the QKV projections
        all_dim = config.num_attention_heads * self.head_dim + 2 * config.num_key_value_heads * self.head_dim
        self.q_size = config.num_attention_heads * self.head_dim
        self.kv_size = config.num_key_value_heads * self.head_dim
        self.qkv_proj = nn.Linear(
            config.hidden_size, all_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )



        # Coefficients for all orders (initialized to 1/n!)
        self.order_coefficients = nn.Parameter((1/torch.arange(1, self.order+1).cumprod(-1))[:, None, None, None].repeat(1, config.num_attention_heads, 1, 1).float())

        # Input convolution
        self.q_conv = nn.Conv1d(
            in_channels=config.num_attention_heads * self.head_dim,
            out_channels=config.num_attention_heads * self.head_dim,
            bias=True,
            kernel_size=d_conv,
            groups=config.num_attention_heads * self.head_dim,
            padding=d_conv - 1,
        ) if q_conv else None
        self.k_conv = nn.Conv1d(
            in_channels=config.num_key_value_heads * self.head_dim,
            out_channels=config.num_key_value_heads * self.head_dim,
            bias=True,
            kernel_size=d_conv,
            groups=config.num_key_value_heads * self.head_dim,
            padding=d_conv - 1,
        ) if k_conv else None
        self.v_conv = nn.Conv1d(
            in_channels=config.num_key_value_heads * self.head_dim,
            out_channels=config.num_key_value_heads * self.head_dim,
            bias=True,
            kernel_size=d_conv,
            groups=config.num_key_value_heads * self.head_dim,
            padding=d_conv - 1,
        ) if v_conv else None


        # Norm
        self.norm = nn.RMSNorm(config.head_dim) if out_norm else nn.Identity()


        # A param
        if no_A:
            self.A_log = None
        else:
            self.A_log = nn.Parameter(torch.log(torch.empty(config.num_attention_heads, dtype=torch.float32).uniform_(1, 16)))
            self.A_log._no_weight_decay = True
        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(config.num_attention_heads) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ) # value between [0.001, 0.1]
        dt = torch.clamp(dt, min=1e-4)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # self.dt_bias = nn.Parameter(-torch.rand(config.num_attention_heads))
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True
        # dt projeciton
        self.dt_proj = nn.Linear(config.hidden_size, config.num_attention_heads)



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





        # Combined QKV
        QKV = self.qkv_proj(hidden_states)

        # Get QKV tensors
        query_states = QKV[:, :, :self.q_size]
        key_states = QKV[:, :, self.q_size:self.q_size+self.kv_size]
        value_states = QKV[:, :, self.q_size+self.kv_size:]

        # Apply convolution
        if self.q_conv is not None:
            query_states = causal_conv1d_fn(
                x=query_states.transpose(1, 2),
                weight=rearrange(self.q_conv.weight, "d 1 w -> d w"),
                bias=self.q_conv.bias,
                activation=self.act,
            ).transpose(1, 2)
        if self.k_conv is not None:
            key_states = causal_conv1d_fn(
                x=key_states.transpose(1, 2),
                weight=rearrange(self.k_conv.weight, "d 1 w -> d w"),
                bias=self.k_conv.bias,
                activation=self.act,
            ).transpose(1, 2)
        if self.v_conv is not None:
            value_states = causal_conv1d_fn(
                x=value_states.transpose(1, 2),
                weight=rearrange(self.v_conv.weight, "d 1 w -> d w"),
                bias=self.v_conv.bias,
                activation=self.act,
            ).transpose(1, 2)
        
        # Add heads
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # RoPE
        if self.no_rope == False:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)





        
        # Get dt
        if self.no_bias:
            dt = nn.functional.softplus(self.dt_proj(hidden_states) + self.dt_bias*0)
        else:
            dt = nn.functional.softplus(self.dt_proj(hidden_states) + self.dt_bias)

        def forwrd_gated(query_states, key_states, value_states, attention_mask, A_log, dt):
            # Discretize x and A
            if not self.no_value_dt:
                value_states = value_states * dt.mT[..., None]
            if A_log is not None:
                # Convert from log space
                A = -torch.exp(A_log.float())
                A = A[None, None, :].to(value_states.dtype) * dt
            else:
                A = -dt

            # Mask
            A_cumsum = torch.cumsum(A, dim=-2).mT
            if self.no_A_mask:
                A_mask = ((((A_cumsum[:, :, :, None] - A_cumsum[:, :, None, :])))*0).masked_fill(attention_mask!=0, -torch.inf).exp()
            else:
                A_mask = (((A_cumsum[:, :, :, None] - A_cumsum[:, :, None, :]))).masked_fill(attention_mask!=0, -torch.inf).exp()

            # Inner product
            attn_weights_ = query_states @ key_states.mT# * (1/math.sqrt(self.head_dim))

            # Mask
            # if self.training:
            #     causal_mask = (attention_mask==0)
            # else:
            #     causal_mask = torch.triu(torch.ones(1, 1, input_shape[-1], input_shape[-1])).mT.to(query_states.device).to(query_states.dtype)
            attn_weights_ = attn_weights_ * A_mask

            attn_weights = attn_weights_.clone() * 1/self.order_coefficients[:1]
            for _i in range(2, self.order+1):
                attn_weights = attn_weights + (attn_weights_**_i) * (torch.nn.functional.softplus(self.order_coefficients[_i-1:_i]))

            # Denominator
            # attn_weights = attn_weights / (attn_weights.norm(p=2, dim=-1, keepdim=True) + 1e-8)

            # Output
            return attn_weights @ value_states

        attn_output = checkpoint(
            forwrd_gated, query_states, key_states, value_states, attention_mask, self.A_log, dt
        )

        # Output norm
        attn_output = self.norm(attn_output)








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
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs