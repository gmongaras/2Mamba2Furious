import math
import torch
from torch import nn
from mamba_ssm import Mamba2
from causal_conv1d import causal_conv1d_fn




class Mamba2Custom(nn.Module):
    def __init__(
            self,
            hidden_size,
            conv_dim,
            expand,
            head_dim,
            d_ssm,
            A_init_range,
            dt_min,
            dt_max,
            dt_init_floor,
            dt_limit,
            seq_len,
            use_kernel,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.conv_dim = conv_dim
        self.expand = expand
        self.head_dim = head_dim
        self.d_ssm = d_ssm
        self.A_init_range = A_init_range
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.dt_limit = dt_limit
        self.use_kernel = use_kernel

        
        
        
        
        
        

        
        
        
        
        
        # Activation applied on q and k after the input convolution is silu
        self.qk_activation_type = "silu"





        # Combine the QKV projections
        qkv_dim = 3*d_ssm
        self.q_size = d_ssm
        self.kv_size = d_ssm
        
        # Input and output projections
        self.qkv_proj = nn.Linear(
            hidden_size, qkv_dim, bias=False
        )
        self.o_proj = nn.Linear(
            d_ssm, hidden_size, bias=False
        )
        
        # A mask projection
        if self.A_mask_type != "none":
            # We only have A_log if we are using
            # "discretize" for A_mask_type
            # This will create A mask via the dt projection
            if self.A_mask_type == "discretize":
                self.A_log = nn.Parameter(
                    torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16).log()
                )
            # Otherwise, we have a projection for the A mask
            # by itself (adding H more params)
            else:
                self.A_mask_proj = nn.Linear(
                    hidden_size, num_heads, bias=False
                )

        # Input convolution
        d_conv = int(self.in_conv_dim)
        self.conv1d = nn.Conv1d(
            in_channels=qkv_dim,
            out_channels=qkv_dim,
            bias=False,
            kernel_size=d_conv,
            groups=qkv_dim,
            padding=d_conv - 1,
        )
            
           
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
        self.dt_proj = nn.Linear(
            hidden_size, num_heads, bias=False
        )
        
        # Initialize log dt bias
        dt_max = 0.1
        dt_min = 0.001
        dt = torch.exp(
            torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=1e-4)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias_value = nn.Parameter(inv_dt[None, None, :])
        self.dt_bias_value._no_weight_decay = True
            
        # D residual
        self.D = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        
        
        # z output gate requires another projection of the input
        self.z_gate_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim)
            
            
        # Ouptput norm
        self.out_norm = nn.RMSNorm(self.num_heads * self.head_dim)
            
        
        # Precompute attention mask if we are not using a kernel
        max_seq_len = seq_len
        self.register_buffer(
            "attn_mask",
            ~torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()[None, None, :, :]
        )




# Mamba layer
batch_size = 16
seq_len = 1024
hidden_size = 1024+512
num_heads = 24
head_dim = hidden_size // num_heads
conv_dim = 2
expand_factor = 1
ngroups = 1
A_init_range = (1, 16)
dt_range = (0.001, 0.1)
dt_init_floor = 0.0001
mamba_layer = Mamba2(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=hidden_size,
    d_conv=conv_dim,
    conv_init=None,
    expand=expand_factor,
    headdim=head_dim,
    d_ssm=head_dim,
    ngroups=ngroups,
    A_init_range=A_init_range,
    D_has_hdim=True,
    rmsnorm=True,
    norm_before_gate=False,
    dt_min=dt_range[0],
    dt_max=dt_range[1],
    dt_init_floor=dt_init_floor,
    dt_limit=(0, float("inf")),
    bias=False,
    conv_bias=False,
    chunk_size=256,
    use_mem_eff_path=True,
    layer_idx=0,
).cuda()
custom_layer = Mamba2Custom(
    hidden_size=hidden_size,
    conv_dim=conv_dim,
    expand=expand_factor,
    head_dim=head_dim,
    d_ssm=head_dim,
    A_init_range=A_init_range,
    dt_min=dt_range[0],
    dt_max=dt_range[1],
    dt_init_floor=dt_init_floor,
    dt_limit=(0, float("inf")),
    seq_len=seq_len,
    use_kernel=False
).cuda()






# Mamba output and custom output should be close
hidden_states = torch.randn(batch_size, seq_len, hidden_size).cuda()
mamba_output = mamba_layer(hidden_states)
custom_output = custom_layer(hidden_states)
torch.allclose(mamba_output, custom_output)