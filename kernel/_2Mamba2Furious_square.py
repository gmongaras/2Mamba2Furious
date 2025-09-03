import pytest
import torch
import os

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, A_q,  #
                    desc_k, desc_v, desc_A_k, #
                    qo_offset_y, offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        
        # Compute A mask
        A_k = desc_A_k.load([offsetk_y])
        A_mask = (A_q[:, None] - A_k[None, :]).to(tl.float32)
        
        # Squared inner product on q and k
        qk = qk.to(tl.float32) * qk_scale
        qk2 = qk * qk 
        
        # Masking diag and off diag
        if STAGE == 2:
            # Get mask
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            
            # Get max
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            
            # Subtract min from A mask and calculate exponentate the A mask
            A_mask_m = tl.exp2(A_mask - m_ij[:, None])
            
            # Compute qkA and mask
            qkA = qk2 * A_mask_m
            p = tl.where(mask, qkA, 0)
        else:
            # Get max
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            
            # Subtract min from A mask and calculate exponentate the A mask
            A_mask_m = tl.exp2(A_mask - m_ij[:, None])
            
            # Compute qkA
            p = qk2 * A_mask_m
        
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v.to(tl.float32), acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in NUM_STAGES_OPTIONS \
    for w in [4, 8]\
]
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = [
        triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook),
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]

    # Filter out configs where BLOCK_M > N_CTX
    return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M, #
              Z, H, desc_q, desc_k, desc_v, desc_o, desc_A, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_A_q = _maybe_make_tensor_desc(desc_A, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_M])
    desc_A_k = _maybe_make_tensor_desc(desc_A, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_N])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    # qk_scale *= 1.44269504  # 1/ln(2)
    # Note that this converts a base 2 exp via: e^A = 2^(A*log_2(e)) = 2^(A*(1/ln(2)))
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # Load A_Q
    A_q = desc_A_q.load([qo_offset_y])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, A_q,  #
                                        desc_k, desc_v, desc_A_k, #
                                        qo_offset_y, offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, A_q,  #
                                        desc_k, desc_v, desc_A_k, #
                                        qo_offset_y, offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # epilogue
    
    # m_i during the loop is the max value across rows.
    # Here, the log2 of the sum of the row (denominator) is
    # added. This way when we do 2^{... - m_i}, we are doing two things.
    # This first is subtracting the min for stability. The second is
    # 2^{-log2(li)} = 1/l_i effectively does the denominator.
    m_i += tl.math.log2(l_i)
    
    # Divide by the denominator
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))
    
    
    
    
    
    
    
    
    
@triton.jit
def _attn_bwd_preprocess_inner(acc, do, q, A_q, m, #
                    desc_k, desc_v, desc_A_k, #
                    qo_offset_y, offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        
        # Compute A mask
        A_k = desc_A_k.load([offsetk_y])
        A_mask = (A_q[:, None] - A_k[None, :]).to(tl.float32)
        
        # Squared inner product on q and k
        qk = qk.to(tl.float32) * qk_scale
        qk2 = qk * qk 
        
        # Subtractax from A mask and calculate exponentate the A mask
        A_mask_m = tl.exp2(A_mask - m[:, None])
        
        # Compute qkA
        p = qk2 * A_mask_m
        
        # Masking diag and off diag
        if STAGE == 2:
            # Get mask
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            
            # Mask the attention scores
            p = tl.where(mask, p, 0)
            
        # Compute dp = do vT
        v = desc_v.load([offsetk_y, 0]).T
        dp = tl.dot(do, v)
        
        # scalar-wise multiply dp and normalized p
        grad = dp * p
        
        # sum and accumulate
        acc += tl.sum(grad, axis=1)
        
        # Step offsets
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
        
    return acc
    
@triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_bwd_preprocess_(sm_scale, S, #
              Z, H, desc_q, desc_k, desc_v, desc_do, desc_A, desc_m, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_do = _maybe_make_tensor_desc(desc_do, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_A_q = _maybe_make_tensor_desc(desc_A, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_M])
    desc_A_k = _maybe_make_tensor_desc(desc_A, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_N])
    desc_m = _maybe_make_tensor_desc(desc_m, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_M])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Initialize output
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    # qk_scale *= 1.44269504  # 1/ln(2)
    # Note that this converts a base 2 exp via: e^A = 2^(A*log_2(e)) = 2^(A*(1/ln(2)))
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # Load in do
    do = desc_do.load([qo_offset_y, 0])
    # Load in the max values
    m = desc_m.load([qo_offset_y])
    # Load A_Q
    A_q = desc_A_q.load([qo_offset_y])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc = _attn_bwd_preprocess_inner(acc, do, q, A_q, m,  #
                                        desc_k, desc_v, desc_A_k, #
                                        qo_offset_y, offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # stage 2: on-band
    if STAGE & 2:
        acc = _attn_bwd_preprocess_inner(acc, do, q, A_q, m,  #
                                        desc_k, desc_v, desc_A_k, #
                                        qo_offset_y, offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # epilogue
    
    # Divide by the denominator
    s_ptrs = S + off_hz * N_CTX + offs_m
    tl.store(s_ptrs, acc)


# Computes the summation part cause it needs to go over the entire sequence
# \sum do * o 
@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv, da_k,  #
                   Q, k, v, A, Ak, sm_scale,  #
                   DO,  #
                   M, S, #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        s = tl.load(S + offs_m)
        
        # Compute qk**2
        qkT = tl.dot(k, qT)
        qkT2 = qkT * qkT
        
        # Compute the A mask and exponentiate the negative max
        # which also gives the denominator
        Aq = tl.load(A + offs_m)
        A_maskTm = tl.math.exp2(Aq.to(tl.float32)[None, :] - Ak.to(tl.float32)[:, None] - m[None, :])
        
        # Multiply the squared component and the A mask
        pT = qkT2 * A_maskTm
        
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
            
        # Compute dp
        do = tl.load(do_ptrs)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
            
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        
        # squaremax derivative
        # dsT = A_maskTm * (dpT - tl.sum(dpT * ppT, 0, keep_dims=True))
        dsT = A_maskTm * (dpT - s[None, :])
        dsT = (dsT).to(tl.float16)
        if MASK:
            dsT = tl.where(mask, dsT, 0.0)
        
        # We need to multiply by 2qk and the A mask to handle the square term
        dqkT = 2 * qkT.to(tl.float16) * dsT
        # Accumulate dk grads
        dk += tl.dot(dqkT, tl.trans(qT)).to(tl.float32)
        
        # Multiply by qk2 to get the da grad
        daT = dsT * qkT2
        # Accumulate dA grads
        da_k += -tl.sum(daT, axis=1)
        
        # Add row
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv, da_k


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dqda(dq, da_q, q, K, V, A, Aq,  #
                 do, m, s,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    # Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        # Inner product
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        
        # Compute qk**2
        qk = tl.dot(q, kT)
        qk2 = qk * qk
        
        # Compute the A mask and exponentiate the negative max
        # which also gives the denominator
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        Ak = tl.load(A + offs_n)
        A_mask = Aq.to(tl.float32)[:, None] - Ak.to(tl.float32)[None, :]
        A_maskm = tl.exp2(A_mask - m)
        
        # Multiply the squared component and A mask
        p = qk2 * A_maskm
        
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
            
        # Compute dp
        dp = tl.dot(do, vT).to(tl.float32)
            
        # squaremax derivative.
        # ds = A_maskm * (dp - tl.sum(dp * p, 1, keep_dims=True))
        ds = A_maskm * (dp - s[:, None])
        if MASK:
            ds = tl.where(mask, ds, 0.0)
            
        # We need to multiply by 2qk and the A mask to handle the square term
        dqk = 2 * qk.to(tl.float32) * ds.to(tl.float32)
        # Accumulate dq grads
        dq = tl.dot(dqk.to(tl.float32), tl.trans(kT).to(tl.float32), dq)
        
        # Multiply by qk2 to get the da grad
        da = ds * qk2
        # Accumulate dA grad
        da_q += tl.sum(da, axis=1)
        
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq, da_q


@triton.jit
def _attn_bwd(Q, K, V, A, sm_scale,  #
              DO,  #
              DQ, DK, DV, DA_Q, DA_K,  #
              M, S,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              A_stride_z, A_stride_h, #
              H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    adj_A = (A_stride_h * (bhid % H) + A_stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    A += adj_A
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    DA_Q += adj_A
    DA_K += adj_A
    M += off_chz
    S += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    da_k = tl.zeros([BLOCK_N1], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    
    # Load in Ak. It will stay in SRAM throughout the inner loop
    Ak = tl.load(A + offs_n)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv, da_k = _attn_bwd_dkdv(dk, dv, da_k, #
                            Q, k, v, A, Ak, sm_scale,  #
                            DO,  #
                            M, S, #
                            stride_tok, stride_d,  #
                            H, N_CTX,  #
                            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                            start_n, start_m, num_steps,  #
                            MASK=True  #
                            )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv, da_k = _attn_bwd_dkdv(  #
        dk, dv, da_k,  #
        Q, k, v, A, Ak, sm_scale,  #
        DO,  #
        M, S, #
        stride_tok, stride_d,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=False  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    da_q = tl.zeros([BLOCK_M2], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]
    
    s = tl.load(S + offs_m)
    
    # Load the Q part of the A mask which stays in memory
    Aq = tl.load(A + offs_m)

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq, da_q = _attn_bwd_dqda(dq, da_q, q, K, V, A, Aq,  #
                      do, m, s, #
                      stride_tok, stride_d,  #
                      H, N_CTX,   #
                      BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                      MASK=True  #
                      )
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = end_n // BLOCK_N2
    dq, da_q = _attn_bwd_dqda(dq, da_q, q, K, V, A, Aq,  #
                      do, m, s, #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * BLOCK_N2, num_steps,  #
                      MASK=False  #
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # dq *= LN2
    tl.store(dq_ptrs, dq)
    
    # Di is divided by the sm_scale. We need to undo that
    # da_k *= 1
    # da_q *= LN2
    
    # Write back da
    da_q_ptrs = DA_Q + offs_m
    tl.store(da_q_ptrs, da_q)
    da_k_ptrs = DA_K + offs_n
    tl.store(da_k_ptrs, da_k)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, A_cumsum, causal, sm_scale, warp_specialize=True):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        A_cumsum = A_cumsum.contiguous()
        
        # Divide the A mask by the sm_scale. It will be multiplied later and we
        # don't want this to have any effect on the A .
        # Since 1/ln(2) will be combined with sm_scale, don't scale here
        A_cumsum = A_cumsum * 1.44269504  # 1/ln(2)
        
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Use device_descriptor for Hopper + warpspec.
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]

            dummy_block = [1, 1]
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1],
                                          block_shape=dummy_block)
            else:
                desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1],
                                          block_shape=dummy_block)
            desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_A = TensorDescriptor(A_cumsum, shape=[y_dim], strides=[1], block_shape=[1])
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o
            desc_A = A_cumsum

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80
        _attn_fwd[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o, desc_A, #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            STAGE=stage,  #
            warp_specialize=warp_specialize,  #
            IS_HOPPER=is_hopper(),  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M, A_cumsum)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.warp_specialize = warp_specialize
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M, A_cumsum = ctx.saved_tensors
        do = do.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = o.contiguous()
        M = M.contiguous()
        A_cumsum = A_cumsum.contiguous()
        # assert do.is_contiguous()
        # assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dA_q = torch.empty_like(A_cumsum)
        dA_k = torch.empty_like(A_cumsum)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        
        # Merge the softmax scale into k
        arg_k = k
        arg_k = arg_k * ctx.sm_scale
        # Scale A cumsum just like K. This is what we did in the forward pass
        # so we also need to do it here.
        arg_A_cumsum = A_cumsum
        arg_A_cumsum = arg_A_cumsum
        
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        # _attn_bwd_preprocess[pre_grid](
        #     o, do,  #
        #     delta,  #
        #     BATCH, N_HEAD, N_CTX,  #
        #     BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        # )
        HEAD_DIM_K = k.shape[-1]
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}
        if is_blackwell() and ctx.warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80
        S = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_bwd_preprocess_[ctx.grid](
            ctx.sm_scale, S,  #
            q.shape[0], q.shape[1],  #
            q, k, v, do, A_cumsum, M, #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            STAGE=3 if ctx.causal else 1,  #
            warp_specialize=ctx.warp_specialize,  #
            IS_HOPPER=is_hopper(),  #
            **extra_kern_args)
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, arg_k, v, arg_A_cumsum, ctx.sm_scale, do, dq, dk, dv, dA_q, dA_k,  #
            M, S, #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            A_cumsum.stride(0), A_cumsum.stride(1),
            N_HEAD, N_CTX,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )
        
        dA = dA_q + dA_k

        return dq, dk, dv, dA, None, None, None, None


attention = _attention.apply

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')


@pytest.mark.parametrize("Z", [1, 4])
@pytest.mark.parametrize("H", [2, 48])
@pytest.mark.parametrize("N_CTX", [128, 1024, (2 if is_hip() else 4) * 1024])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [True])  # FIXME: Non-causal tests do not pass at the moment.
@pytest.mark.parametrize("warp_specialize", [False, True] if is_blackwell() else [False])
@pytest.mark.parametrize("mode", ["fwd", "bwd"])
@pytest.mark.parametrize("provider", ["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []))
def test_op(Z, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, dtype=torch.float16):
    # if mode == "fwd" and "fp16" in provider:
    #     pytest.skip("Avoid running the forward computation twice.")
    # if mode == "bwd" and "fp8" in provider:
    #     pytest.skip("Backward pass with FP8 is not supported.")
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    # reference implementation
    ref_dtype = dtype
    if mode == "fwd" and "fp8" in provider:
        ref_dtype = torch.float32
    # q = q.to(ref_dtype)
    # k = k.to(ref_dtype)
    # v = v.to(ref_dtype)
    
    # Create A mask
    A = -torch.nn.functional.softplus(torch.empty(Z, H, N_CTX, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    attention_mask = torch.tril(torch.ones((Z, H, N_CTX, N_CTX), device=DEVICE))
    A_cumsum = torch.cumsum(A.float(), dim=-1).detach().requires_grad_()
    A_mask = (((A_cumsum[:, :, :, None] - A_cumsum[:, :, None, :]))).masked_fill(~attention_mask.bool(), -torch.inf).exp().contiguous()#.to(ref_dtype)
    
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    # if causal:
    #     p[:, :, M == 0] = float("-inf")
    p = p**2
    p = p.masked_fill(~attention_mask.bool(), 0)
    p = p * A_mask
    # p = torch.softmax(p.float(), dim=-1)
    p = p / p.sum(dim=-1, keepdim=True)
    p = p.to(ref_dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v).half()
    if mode == "bwd":
        dout = torch.randn_like(q)
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
        ref_da, A_cumsum.grad = A_cumsum.grad.clone(), None
    # triton implementation
    if mode == "fwd" and "fp8" in provider:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    tri_out = attention(q, k, v, A_cumsum, causal, sm_scale, warp_specialize).half()
    if mode == "fwd":
        atol = 3 if "fp8" in provider else 1e-2
        # torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    tri_da, A_cumsum.grad = A_cumsum.grad.clone(), None
    # compare
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
    rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of CDNA2 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 5e-2
    torch.testing.assert_close(tri_dv, ref_dv, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dk, ref_dk, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dq, ref_dq, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_da, ref_da, atol=1e-2, rtol=rtol)
    

if __name__ == "__main__":
        
    
    # test_op(
    #     16,
    #     16,
    #     2048,
    #     64,
    #     causal=True,
    #     warp_specialize=False,
    #     mode="fwd",
    #     provider="triton-fp16",
    # )

    test_op(
        16,
        16,
        2048,
        64,
        causal=True,
        warp_specialize=False,
        mode="bwd",
        provider="triton-fp16",
    )

    exit()


    try:
        from flash_attn.flash_attn_interface import \
            flash_attn_qkvpacked_func as flash_attn_func
        HAS_FLASH = True
    except BaseException:
        HAS_FLASH = False

    TORCH_HAS_FP8 = False #hasattr(torch, 'float8_e5m2')
    BATCH, N_HEADS = 4, 32
    # vary seq length for fixed head and batch=4
    configs = []
    for HEAD_DIM in [64, 128]:
        for mode in ["fwd", "bwd"]:
            for causal in [True, False]:
                # Enable warpspec for causal fwd on Hopper
                enable_ws = mode == "fwd" and (is_blackwell() or (is_hopper() and not causal))
                for warp_specialize in [False, True] if enable_ws else [False]:
                    configs.append(
                        triton.testing.Benchmark(
                            x_names=["N_CTX"],
                            x_vals=[2**i for i in range(10, 15)],
                            line_arg="provider",
                            line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                            (["flash"] if HAS_FLASH else []),
                            line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                            (["Flash-2"] if HAS_FLASH else []),
                            styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                            ylabel="TFLOPS",
                            plot_name=
                            f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-warp_specialize={warp_specialize}",
                            args={
                                "H": N_HEADS,
                                "BATCH": BATCH,
                                "HEAD_DIM": HEAD_DIM,
                                "mode": mode,
                                "causal": causal,
                                "warp_specialize": warp_specialize,
                            },
                        ))


    @triton.testing.perf_report(configs)
    def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, device=DEVICE):
        assert mode in ["fwd", "bwd"]
        dtype = torch.float16
        if "triton" in provider:
            q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
            k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
            v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
            
            A = -torch.nn.functional.softplus(torch.randn(BATCH, H, N_CTX, device=torch.device("cuda:0")))
            A_cumsum = torch.cumsum(A.float(), dim=-1).to(dtype)
            
            if mode == "fwd" and "fp8" in provider:
                q = q.to(torch.float8_e5m2)
                k = k.to(torch.float8_e5m2)
                v = v.permute(0, 1, 3, 2).contiguous()
                v = v.permute(0, 1, 3, 2)
                v = v.to(torch.float8_e5m2)
            sm_scale = 1.3
            fn = lambda: attention(q, k, v, A_cumsum, causal, sm_scale, warp_specialize)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn)

        if provider == "flash":
            qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
            fn = lambda: flash_attn_func(qkv, causal=causal)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn)
        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
        total_flops = 2 * flops_per_matmul
        if causal:
            total_flops *= 0.5
        if mode == "bwd":
            total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        return total_flops * 1e-12 / (ms * 1e-3)


    
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)