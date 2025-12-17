import pytest
import torch
import os

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()

if __name__ == "__main__":
    DEBUG=True
else:
    DEBUG=False
# DEBUG=False

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


@triton.jit(debug=DEBUG)
def _attn_fwd_inner(acc, l_i, m_i, q, A_q,  #
                    desc_k, desc_v, desc_A_k, #
                    qo_offset_y, offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE_CAUSAL: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if not STAGE_CAUSAL: # non causal stage (off diag)
        # Handle all blocks up to the last, which is the one before the on-diag block
        lo, hi = 0, start_m * BLOCK_M
    else: # causal stage (on diag)
        # Handle just the last block
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        
    # Block offset along the row
    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    
    # loop over k, v (in blocks of BLOCK_N) and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T.to(tl.float32)
        qk = tl.dot(q, k)
        
        # Compute A mask
        A_k = desc_A_k.load([offsetk_y]).to(tl.float32)
        A_mask = (A_q[:, None] - A_k[None, :])
        
        # Squared inner product on q and k
        qk = qk * qk_scale
        
        # Mask qk inner product if causal
        if STAGE_CAUSAL:
            # Get mask if we are on-diag
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            # Mask upper triangle
            qk = tl.where(mask, qk, 0)
        
        # Get qk max
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij = tl.where(m_ij > 32, 32, m_ij) # Large max values can be problematic
        
        # Subtract max from A mask and calculate exponentate the A mask
        A_mask_m = (A_mask - m_ij[:, None])
        A_mask_m = tl.where(A_mask_m < -1024, 0.0, tl.exp2(A_mask_m)) # Mask where pre exp was less than -1024 to zero
        if STAGE_CAUSAL:
            # Mask upper triangle
            A_mask_m = tl.where(mask, A_mask_m, 0)
        
        # A_mask * qk**2
        p = qk * qk * A_mask_m
        
        # Compute the correction factor. This will change things from using
        # a max of m_i (previous blocks) to m_ij (current block).
        # It will be 1 if m_i = m_ij meaning the max didn't change
        alpha = tl.math.exp2(m_i - m_ij)
        
        # Update output accumulator with the correct max value
        acc = acc * alpha[:, None]
        
        # prepare p and v for the dot
        v = desc_v.load([offsetv_y, 0]).to(tl.float32)
        
        # Inner product with v to get output (without denominator. That will be applied
        # after the entire block sum is computed via the l sum tensor)
        # This operation is actually more precise in float16, but
        # the A mask values get really small off diag, so it's probably
        # better to keep it in FP32. On diag, keeping this in FP16
        # causes large errors while FP32 causes more small errors.
        acc = tl.dot(p, v, acc)
        
        # Compute denominator value
        l_ij = tl.sum(p, 1)
        # Update the denominator by adding all past values with this block
        # and updating the denominator with the correct max value
        l_i = l_i * alpha + l_ij
        
        # Update the max value along blocks
        m_i = m_ij
        
        # Update pointers
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


@triton.jit(debug=DEBUG)
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit(debug=DEBUG)
def _attn_fwd(sm_scale, M, #
              Z, H, desc_q, desc_k, desc_v, desc_o, desc_A, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              ):
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Initialize tensor blocks
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
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

    # initialize offsets
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    # load q: it will stay in SRAM throughout iterations along the row
    q = desc_q.load([qo_offset_y, 0]).to(tl.float32)
    # Load A_Q, it will also stay in SRAM through the loop
    A_q = desc_A_q.load([qo_offset_y]).to(tl.float32)
    # stage 1: causal - on-band - on diag
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, A_q,  #
                                desc_k, desc_v, desc_A_k, #
                                qo_offset_y, offset_y, dtype, start_m, qk_scale,  #
                                BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                True, offs_m, offs_n, N_CTX,  #
                                warp_specialize, IS_HOPPER)
    # stage 2: non causal - off-band - off diag
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, A_q,  #
                                    desc_k, desc_v, desc_A_k, #
                                    qo_offset_y, offset_y, dtype, start_m, qk_scale,  #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    False, offs_m, offs_n, N_CTX,  #
                                    warp_specialize, IS_HOPPER)
    
    # m_i during the loop is the max value across rows.
    # Here, the log2 of the sum of the row (denominator) is
    # added. This way when we do 2^{... - m_i}, we are doing two things.
    # This first is subtracting the min for stability. The second is
    # 2^{-log2(li)} = 1/l_i effectively does the denominator.
    m_i += tl.math.log2(l_i)
    # m_i = tl.math.log2(l_i * tl.exp2(m_i))
    
    # Divide by the denominator
    acc = acc / l_i[:, None]
    
    # Store max values (combined with the denominator) and output
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(tl.float32))
    
    

    
    
    
@triton.jit(debug=DEBUG)
def _attn_bwd_preprocess_inner(acc, do, q, A_q, m_i, l_i, #
                    desc_k, desc_v, desc_A_k, #
                    qo_offset_y, offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE_CAUSAL: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if not STAGE_CAUSAL: # non causal stage (off diag)
        # Handle all blocks up to the last, which is the one before the on-diag block
        lo, hi = 0, start_m * BLOCK_M
    else: # causal stage (on diag)
        # Handle just the last block
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # Block offset along the row
    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    
    # loop over k, v (in blocks of BLOCK_N) and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        kT = desc_k.load([offsetk_y, 0]).T.to(tl.float32)
        qk = tl.dot(q, kT)
        
        # Compute A mask
        A_k = desc_A_k.load([offsetk_y]).to(tl.float32)
        A_mask = (A_q[:, None] - A_k[None, :])
        
        # Inner product on q and k
        qk = qk * qk_scale
        
        # Mask qk inner product if causal
        if STAGE_CAUSAL:
            # Get mask if we are on-diag
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            # Mask upper triangle
            qk = tl.where(mask, qk, 0)
        
        # Get qk max
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij = tl.where(m_ij > 32, 32, m_ij) # Large max values can be problematic
        
        # Subtractax from A mask and calculate exponentate the A mask
        # This also applies the denominator
        A_mask_m = (A_mask - m_ij[:, None])
        A_mask_m = tl.where(A_mask_m < -1024, 0.0, tl.exp2(A_mask_m)) # Mask where pre exp was less than -1024 to zero
        if STAGE_CAUSAL:
            # Mask upper triangle
            A_mask_m = tl.where(mask, A_mask_m, 0)
        
        # A_mask * qk**2
        p = qk * qk * A_mask_m
        
        # Compute the correction factor. This will change things from using
        # a max of m_i (previous blocks) to m_ij (current block).
        # It will be 1 if m_i = m_ij meaning the max didn't change
        alpha = tl.math.exp2(m_i - m_ij)
        
        # Update output accumulator with the correct max value
        acc = acc * alpha
            
        # Compute dp = do vT
        vT = desc_v.load([offsetv_y, 0]).T.to(tl.float32)
        dp = tl.dot(do, vT)
        
        # scalar-wise multiply dp and normalized p
        grad = dp * p
        
        # sum and accumulate
        acc += tl.sum(grad, axis=1)
        
        # Compute denominator value
        l_ij = tl.sum(p, 1)
        # Update the denominator by adding all past values with this block
        # and updating the denominator with the correct max value
        l_i = l_i * alpha + l_ij
        
        # Update the max value along blocks
        m_i = m_ij
        
        # Step offsets
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
        
    return acc, m_i, l_i
    
@triton.jit(debug=DEBUG)
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
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
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

    # initialize offsets
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Initialize output
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0]).to(tl.float32)
    # Load in do
    do = desc_do.load([qo_offset_y, 0]).to(tl.float32)
    # Load A_Q
    A_q = desc_A_q.load([qo_offset_y]).to(tl.float32)
    # stage 1: causal - on-band - on diag
    acc, m_i, l_i = _attn_bwd_preprocess_inner(acc, do, q, A_q, m_i, l_i,  #
                                    desc_k, desc_v, desc_A_k, #
                                    qo_offset_y, offset_y, dtype, start_m, qk_scale,  #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    True, offs_m, offs_n, N_CTX,  #
                                    warp_specialize, IS_HOPPER)
    # stage 2: non causal - off-band - off diag
    acc, m_i, l_i = _attn_bwd_preprocess_inner(acc, do, q, A_q, m_i, l_i,  #
                                    desc_k, desc_v, desc_A_k, #
                                    qo_offset_y, offset_y, dtype, start_m, qk_scale,  #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    False, offs_m, offs_n, N_CTX,  #
                                    warp_specialize, IS_HOPPER)
    
    # Divide the accumulation tensor by the denominator
    acc = acc / l_i
    
    # Store output
    s_ptrs = S + off_hz * N_CTX + offs_m
    tl.store(s_ptrs, acc.to(tl.float32))
    
    
    
# the main inner-loop logic for computing dQ
@triton.jit(debug=DEBUG)
def _attn_bwd_dkdvdak(
                 k, v, A_k, # pre loaded
                 desc_q, desc_do, desc_A_q, desc_m, desc_s, # To load when iterating
                 dk, dv, dA_k, # To write to
                 kv_offset_y, offset_y, dtype: tl.constexpr, start_n, qk_scale,  #
                 BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                 STAGE_CAUSAL: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                 N_CTX: tl.constexpr):
    assert BLOCK_M <= BLOCK_N
    
    # range of values handled by this stage
    # Note that this iterates from the causal part to the end of the sequence
    # whereas for q this iterates from the beginning of the block to the
    # causal part.
    if not STAGE_CAUSAL: # non causal stage (off diag)
        # Handle all blocks after the causal block
        lo, hi = start_n + BLOCK_N, N_CTX
    else: # causal stage (on diag)
        # Handle the causal block
        lo, hi = start_n, start_n + BLOCK_N
    
    # Block offset along the row for K and V
    offsetq_y = offset_y + lo
    
    # loop over k, v (in blocks of BLOCK_N) and update accumulator
    for start_m in tl.range(lo, hi, BLOCK_M, warp_specialize=False):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        
        # Load q, do, m, and s
        qT = desc_q.load([offsetq_y, 0]).T.to(tl.float32)
        doT = desc_do.load([offsetq_y, 0]).T.to(tl.float32)
        m = desc_m.load([offsetq_y])[None, :].to(tl.float32)
        s = desc_s.load([offsetq_y])[None, :].to(tl.float32)
        
        # -- compute qkT ----
        qkT = tl.dot(k, qT)
        qkT2 = qkT * qkT
        
        # Compute A mask and subtract max
        A_q = desc_A_q.load([offsetq_y]).to(tl.float32)
        A_mask_m_T = A_q[None, :] - A_k[:, None] - m
        
        # Exponentate the A mask
        A_mask_m_T = tl.where(A_mask_m_T < -1024, 0.0, tl.exp2(A_mask_m_T)) # Mask where pre exp was less than -1024 to zero
        
        # Get causal mask 
        if STAGE_CAUSAL:
            mask = (start_m + offs_m[None, :]) >= offs_n[:, None]
            
        # Multiply squared component and the A mask for the attn scores
        pT = qkT2 * A_mask_m_T
        if STAGE_CAUSAL:
            pT = tl.where(mask, pT, 0.0)
            
        # Compute dp
        dpT = tl.dot(v, doT)
        
        # Value derivative - dv
        dv = tl.dot(pT, tl.trans(doT), dv)
            
        # squaremax derivative.
        dsT = A_mask_m_T * (dpT - s)
            
        # We need to multiply by 2qk and the A mask to handle the square term
        dqkT = 2 * qkT * dsT
        if STAGE_CAUSAL:
            dqkT = tl.where(mask, dqkT, 0.0)
        
        # Accumulate dk grads
        dk = tl.dot(dqkT, tl.trans(qT), dk)
        
        # Multiply by qk2 to get the da grad
        daT = dsT * qkT2
        if STAGE_CAUSAL:
            daT = tl.where(mask, daT, 0.0)
        # Accumulate dA grad
        dA_k -= tl.sum(daT, axis=1)
        
        # Update pointers
        offsetq_y += BLOCK_M
    return dk, dv, dA_k
    
    
    
    
# the main inner-loop logic for computing dQ and DA_Q
@triton.jit(debug=DEBUG)
def _attn_bwd_dqdaq(
                 q, do, m, s, A_q, # pre loaded
                 desc_k, desc_v, desc_A_k, # To load when iterating
                 dq, dA_q, # To write to
                 qo_offset_y, offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                 BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                 STAGE_CAUSAL: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                 N_CTX: tl.constexpr):
    assert BLOCK_N <= BLOCK_M
    
    # range of values handled by this stage
    if not STAGE_CAUSAL: # non causal stage (off diag)
        # Handle all blocks up to the last, which is the one before the on-diag block
        lo, hi = 0, start_m
    else: # causal stage (on diag)
        # Handle just the last block
        lo, hi = start_m, start_m + BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    
    # Block offset along the row for K and V
    offsetk_y = offset_y + lo
    
    # loop over k, v (in blocks of BLOCK_N) and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=False):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load k and v
        kT = desc_k.load([offsetk_y, 0]).T.to(tl.float32)
        vT = desc_v.load([offsetk_y, 0]).T.to(tl.float32)
        
        # -- compute qk ----
        qk = tl.dot(q, kT)
        
        # Squared inner product on q and k
        qk2 = qk * qk
        
        # Compute A mask and subtract max
        A_k = desc_A_k.load([offsetk_y]).to(tl.float32)
        A_mask_m = A_q[:, None] - A_k[None, :] - m
        
        # Exponentate the A mask
        A_mask_m = tl.where(A_mask_m < -1024, 0.0, tl.exp2(A_mask_m)) # Mask where pre exp was less than -1024 to zero
        
        # Ge causal mask 
        if STAGE_CAUSAL:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            
        # Compute dp
        dp = tl.dot(do, vT)
            
        # squaremax derivative.
        ds = A_mask_m * (dp - s)
            
        # We need to multiply by 2qk and the A mask to handle the square term
        dqk = 2 * qk * ds
        if STAGE_CAUSAL:
            dqk = tl.where(mask, dqk, 0.0)
        
        # Accumulate dq grads
        dq = tl.dot(dqk, tl.trans(kT), dq)
        
        # Multiply by qk2 to get the da grad
        da = ds * qk2
        if STAGE_CAUSAL:
            da = tl.where(mask, da, 0.0)
        # Accumulate dA grad
        dA_q += tl.sum(da, axis=1)
        
        # Update pointers
        offsetk_y += BLOCK_N
    return dq, dA_q
    
    
    
@triton.jit(debug=DEBUG)
def _attn_bwd(desc_q, desc_k, desc_v, desc_A, sm_scale,  #
              desc_do,  #
              desc_dq, desc_dk, desc_dv, desc_dA_q, desc_dA_k,  #
              desc_m, desc_s,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              A_stride_z, A_stride_h, #
              Z, H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              HEAD_DIM: tl.constexpr):
    
    
    
    # For K and V, the thread blocks are over N, (the cols)
    # and we iterate over M (the rows) for each thread block
    dtype = tl.float16
    blk_N = tl.program_id(0) # index of the Nth block
    start_n = blk_N * BLOCK_N1 # Starting index within N
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Initialize tensor blocks
    y_dim = Z * H * N_CTX
    # inputs
    desc_q_ = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M1, HEAD_DIM])
    desc_do_ = _maybe_make_tensor_desc(desc_do, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M1, HEAD_DIM])
    desc_v_ = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                        block_shape=[BLOCK_N1, HEAD_DIM])
    desc_k_ = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N1, HEAD_DIM])
    desc_A_q_ = _maybe_make_tensor_desc(desc_A, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_M1])
    desc_A_k_ = _maybe_make_tensor_desc(desc_A, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_N1])
    desc_m_ = _maybe_make_tensor_desc(desc_m, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_M1])
    desc_s_ = _maybe_make_tensor_desc(desc_s, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_M1])
    # outputs
    desc_dv_ = _maybe_make_tensor_desc(desc_dv, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                        block_shape=[BLOCK_N1, HEAD_DIM])
    desc_dk_ = _maybe_make_tensor_desc(desc_dk, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N1, HEAD_DIM])
    desc_dA_k_ = _maybe_make_tensor_desc(desc_dA_k, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_N1])
    
    
    # temp outputs
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dA_k = tl.zeros([BLOCK_N1], dtype=tl.float32)
    
    
    # initialize offsets
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX # Batch, head offset
    kv_offset_y = offset_y + start_n # Batch, head, sequence offset
    offs_n = start_n + tl.arange(0, BLOCK_N1) # Offsets for the Q (row) positions
    offs_m = tl.arange(0, BLOCK_M1) # Offsets for the K/V (col) positions
    # load scales
    qk_scale = sm_scale
    # load k, v, and A_k: it will stay in SRAM throughout iterations along the row
    k = desc_k_.load([kv_offset_y, 0]).to(tl.float32)
    v = desc_v_.load([kv_offset_y, 0]).to(tl.float32)
    A_k = desc_A_k_.load([kv_offset_y]).to(tl.float32)
    # stage 1: causal - on-band - on diag
    dk, dv, dA_k = _attn_bwd_dkdvdak(k, v, A_k, # pre loaded
                                desc_q_, desc_do_, desc_A_q_, desc_m_, desc_s_, # To load when iterating
                                dk, dv, dA_k, # To write to
                                kv_offset_y, offset_y, dtype, start_n, qk_scale,  #
                                BLOCK_M1, HEAD_DIM, BLOCK_N1,  #
                                True, offs_m, offs_n, N_CTX,  #
                                )
    # stage 2: non causal - off-band - off diag
    dk, dv, dA_k = _attn_bwd_dkdvdak(k, v, A_k, # pre loaded
                                desc_q_, desc_do_, desc_A_q_, desc_m_, desc_s_, # To load when iterating
                                dk, dv, dA_k, # To write to
                                kv_offset_y, offset_y, dtype, start_n, qk_scale,  #
                                BLOCK_M1, HEAD_DIM, BLOCK_N1,  #
                                False, offs_m, offs_n, N_CTX,  #
                                )
    
    dk *= sm_scale
    
    # Store the outputs
    desc_dk_.store([kv_offset_y, 0], dk.to(dtype))
    desc_dv_.store([kv_offset_y, 0], dv.to(dtype))
    desc_dA_k_.store([kv_offset_y], dA_k.to(tl.float32))











    
    
    
    

    # For Q, the thread blocks are over M, (the rows)
    # and we iterate over N (the cols) for each thread block
    blk_M = tl.program_id(0) # index of the Mth block
    start_m = blk_M * BLOCK_M2 # Starting index within M

    # Initialize tensor blocks
    y_dim = Z * H * N_CTX
    # inputs
    desc_q_ = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M2, HEAD_DIM])
    desc_do_ = _maybe_make_tensor_desc(desc_do, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M2, HEAD_DIM])
    desc_v_ = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                        block_shape=[BLOCK_N2, HEAD_DIM])
    desc_k_ = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N2, HEAD_DIM])
    desc_A_q_ = _maybe_make_tensor_desc(desc_A, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_M2])
    desc_A_k_ = _maybe_make_tensor_desc(desc_A, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_N2])
    desc_m_ = _maybe_make_tensor_desc(desc_m, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_M2])
    desc_s_ = _maybe_make_tensor_desc(desc_s, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_M2])
    # outputs
    desc_dq_ = _maybe_make_tensor_desc(desc_dq, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M2, HEAD_DIM])
    desc_dA_q_ = _maybe_make_tensor_desc(desc_dA_q, shape=[y_dim], strides=[1],
                                     block_shape=[BLOCK_M2])
    # temp outputs
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    dA_q = tl.zeros([BLOCK_M2], dtype=tl.float32)

    # initialize offsets
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX # Batch, head offset
    qo_offset_y = offset_y + start_m # Batch, head, sequence offset
    offs_m = start_m + tl.arange(0, BLOCK_M2) # Offsets for the Q (row) positions
    offs_n = tl.arange(0, BLOCK_N2) # Offsets for the K/V (col) positions
    # load scales
    qk_scale = sm_scale
    # load q and do: it will stay in SRAM throughout iterations along the row
    q = desc_q_.load([qo_offset_y, 0]).to(tl.float32)
    do = desc_do_.load([qo_offset_y, 0]).to(tl.float32)
    # Load A_Q and S, it will also stay in SRAM through the loop
    A_q = desc_A_q_.load([qo_offset_y]).to(tl.float32)
    M = desc_m_.load([qo_offset_y])[:, None].to(tl.float32)
    S = desc_s_.load([qo_offset_y])[:, None].to(tl.float32)
    # stage 1: causal - on-band - on diag
    dq, dA_q = _attn_bwd_dqdaq(q, do, M, S, A_q, # pre loaded
                                desc_k_, desc_v_, desc_A_k_, # To load when iterating
                                dq, dA_q, # To write to
                                qo_offset_y, offset_y, dtype, start_m, qk_scale,  #
                                BLOCK_M2, HEAD_DIM, BLOCK_N2,  #
                                True, offs_m, offs_n, N_CTX,  #
                                )
    # stage 2: non causal - off-band - off diag
    dq, dA_q = _attn_bwd_dqdaq(q, do, M, S, A_q, # pre loaded
                                desc_k_, desc_v_, desc_A_k_, # To load when iterating
                                dq, dA_q, # To write to
                                qo_offset_y, offset_y, dtype, start_m, qk_scale,  #
                                BLOCK_M2, HEAD_DIM, BLOCK_N2,  #
                                False, offs_m, offs_n, N_CTX,  #
                                )
    
    # Store the outputs
    desc_dq_.store([qo_offset_y, 0], dq.to(dtype))
    desc_dA_q_.store([qo_offset_y], dA_q.to(tl.float32))

    
    
    
    
    
    
    
    
    
    
    
    
    


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, A_cumsum, causal, sm_scale, warp_specialize=True):
        # Convert q, k, and v to contiguous half precision
        # Convert A cumsum to a contiguous float.
        q = q.half().contiguous()
        k = k.half().contiguous()
        v = v.half().contiguous()
        A_cumsum = A_cumsum.float().contiguous()
        
        # Divide the A mask by the sm_scale. It will be multiplied later and we
        # don't want this to have any effect on the A .
        A_cumsum = A_cumsum * 1.4426950408889634074  # 1/ln(2)
        
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q).float()
        stage = 3 if causal else 1
        extra_kern_args = {}
        if is_hip():
            assert False, "AMD skill issue"

        # Hold max values along all rows during the attention computation
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Initialize descriptors
        desc_q = q
        desc_v = v
        desc_k = k
        desc_o = o
        desc_A = A_cumsum

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")
        triton.set_allocator(alloc_fn)

        # Define grid for spinning up kernels
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
        BLOCK_MP, BLOCK_NP = 128, 32
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        
        # Merge the softmax scale into k
        arg_k = k * ctx.sm_scale
        
        assert N_CTX % PRE_BLOCK == 0
        HEAD_DIM_K = k.shape[-1]
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            assert False
        if is_blackwell() and ctx.warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80
                
        # Get S matrix. Need to basically redo the entire forward pass :/
        S = torch.empty_like(M).float()
        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        _attn_bwd_preprocess_[grid](
            ctx.sm_scale, S,  #
            q.shape[0], q.shape[1],  #
            q, k, v, do, A_cumsum, M, #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            STAGE=3 if ctx.causal else 1,  #
            warp_specialize=ctx.warp_specialize,  #
            IS_HOPPER=is_hopper(),  #
            BLOCK_M=BLOCK_MP, BLOCK_N=BLOCK_NP,
            **extra_kern_args)
        grid = (N_CTX // BLOCK_N1, BATCH * N_HEAD, 1)
        _attn_bwd[grid](
            q, arg_k, v, A_cumsum, ctx.sm_scale, do, dq, dk, dv, dA_q, dA_k,  #
            M, S, #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            A_cumsum.stride(0), A_cumsum.stride(1),
            q.shape[0], N_HEAD, N_CTX,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            **extra_kern_args
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
def test_op(Z, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, dtype=torch.float16, q=None, k=None, v=None, A_cumsum=None):
    # if mode == "fwd" and "fp16" in provider:
    #     pytest.skip("Avoid running the forward computation twice.")
    # if mode == "bwd" and "fp8" in provider:
    #     pytest.skip("Backward pass with FP8 is not supported.")
    torch.manual_seed(20)
    if q is None:
        q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).float().requires_grad_())
    else:
        q = q.float().detach().requires_grad_()
    if k is None:
        k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).float().requires_grad_())
    else:
        k = k.float().detach().requires_grad_()
    if v is None:
        v = ((torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)).float().requires_grad_())
        # v_ = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)) * 20
        # v = (v * torch.nn.functional.softplus(v_)).detach().float().requires_grad_()
    else:
        v = v.float().detach().requires_grad_()
    sm_scale = 1/(HEAD_DIM)**0.5
    # reference implementation
    ref_dtype = dtype
    if mode == "fwd" and "fp8" in provider:
        ref_dtype = torch.float32
    # q = q.to(ref_dtype)
    # k = k.to(ref_dtype)
    # v = v.to(ref_dtype)
    
    # Create A mask
    attention_mask = torch.tril(torch.ones((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=DEVICE))
    def make_blockwise_mask(q_len, k_len, B_M, B_N, mode="causal", device=None):
        """
        Creates a blockwise attention mask tensor with options for causal, noncausal, or both.
        
        Args:
            q_len (int): Length of query sequence.
            k_len (int): Length of key sequence.
            B_M (int): Block size along the query dimension.
            B_N (int): Block size along the key dimension.
            mode (str): 'causal', 'noncausal', or 'both'.
            device (torch.device, optional): Device to create the mask on.
        
        Returns:
            torch.Tensor: A [q_len, k_len] mask tensor of 0s and 1s.
                        (1 = allowed attention, 0 = masked out)
        """
        assert mode in ("causal", "noncausal", "both"), "mode must be 'causal', 'noncausal', or 'both'"
        # number of blocks
        num_blocks_m = (q_len + B_M - 1) // B_M
        num_blocks_n = (k_len + B_N - 1) // B_N
        # start with block-level grid
        block_mask = torch.zeros((num_blocks_m, num_blocks_n), device=device)
        for i in range(num_blocks_m):
            for j in range(num_blocks_n):
                if mode == "both":
                    block_mask[i, j] = 1
                elif mode == "causal" and i == j:
                    block_mask[i, j] = 1
                elif mode == "noncausal" and i < j:
                    block_mask[i, j] = 1
        # expand to full token-level mask
        mask = block_mask.repeat_interleave(B_M, dim=0).repeat_interleave(B_N, dim=1)
        mask = mask[:q_len, :k_len]
        return mask.cuda()
    # attention_mask = make_blockwise_mask( # Block-size only on diag
    #     q.shape[2],
    #     q.shape[2],
    #     64,
    #     64,
    #     "causal"
    # ).bool().repeat(q.shape[0], q.shape[1], 1, 1) & attention_mask.bool()
    # attention_mask = ~make_blockwise_mask( # Block-size only off diag
    #     q.shape[2],
    #     q.shape[2],
    #     64,
    #     64,
    #     "causal"
    # ).bool().repeat(q.shape[0], q.shape[1], 1, 1) & attention_mask.bool()
    #torch.tril(torch.ones((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=DEVICE))

    if A_cumsum is None:
        A = -torch.nn.functional.softplus(torch.empty(Z, H, N_CTX, device=DEVICE).normal_(mean=0.0, std=0.5).float().requires_grad_())
        A_cumsum = torch.cumsum(A.float(), dim=-1).detach().requires_grad_()
    else:
        A_cumsum = A_cumsum.float().detach().requires_grad_()
    A_mask = (((A_cumsum[:, :, :, None] - A_cumsum[:, :, None, :]))).masked_fill(~attention_mask.bool(), -torch.inf).exp().contiguous()#.to(ref_dtype)
    
    p = torch.matmul(q.float(), k.float().transpose(2, 3)) * sm_scale
    # if causal:
    #     p[:, :, M == 0] = float("-inf")
    p = p**2
    p = p.masked_fill(~attention_mask.bool(), 0)
    p = p * A_mask
    # p = torch.softmax(p.float(), dim=-1)
    p = p / p.sum(dim=-1, keepdim=True)
    # p = p.to(ref_dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p.float(), v.float())#.to(ref_dtype)
    if mode == "bwd":
        dout = torch.randn_like(q)
        # dout = v.clone()
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
    tri_out = attention(q.half(), k.half(), v.half(), A_cumsum.float(), causal, sm_scale, warp_specialize)
    if mode == "fwd":
        atol = 3 if "fp8" in provider else 1e-2
        torch.testing.assert_close(tri_out.float(), ref_out.float(), atol=atol, rtol=0)
        return
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    tri_da, A_cumsum.grad = A_cumsum.grad.clone(), None
    # compare
    torch.testing.assert_close(tri_out.float(), ref_out.float(), atol=1e-2, rtol=0)
    rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of CDNA2 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 5e-2
    torch.testing.assert_close(tri_dv.float(), ref_dv.float(), atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dk.float(), ref_dk.float(), atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dq.float(), ref_dq.float(), atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_da.float(), ref_da.float(), atol=1e-2, rtol=rtol)
    

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
    
    """
    b_idx = 0
    q = torch.load("debug_output/query_states").cuda().half().detach().requires_grad_()
    seq_len = (~(q[b_idx,0,:,0] == q[b_idx,0,-1,0])).sum()
    seq_len = (seq_len//128)*128
    q = q[b_idx:b_idx+1,:,:seq_len].contiguous().detach().requires_grad_()
    k = torch.load("debug_output/key_states").cuda().half()[b_idx:b_idx+1,:,:seq_len].contiguous().detach().requires_grad_()
    v = torch.load("debug_output/value_states").cuda().half()[b_idx:b_idx+1,:,:seq_len].contiguous().detach().requires_grad_()
    A_cumsum = torch.load("debug_output/A_cumsum").cuda().half()[b_idx:b_idx+1,:,:seq_len].contiguous().detach().requires_grad_()
    out = attention(q, k, v, A_cumsum, True, 0.125, False)
    out.sum().backward()
    """
    """
    test_op(
        q.shape[0],
        q.shape[1],
        q.shape[2],
        q.shape[3],
        causal=True,
        warp_specialize=False,
        mode="bwd",
        provider="triton-fp16",
        q=q,
        k=k,
        v=v,
        A_cumsum=A_cumsum
    )
    """

    test_op(
        8,
        32,
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