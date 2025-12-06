import torch




# Function used to test if the custom forward pass
# and grads are correct fo a given forward function
#   forward - Function to test and do a backward pass on. Takes in args.
#   backward_cusom - Function to manally calculate all grads. Takes in args
#   *args - Tesnor inputs into forward and backward
def test_grads(forward, backward_custom, *args, manual=False):
    # Define custom PyTorch function to test
    class Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            ctx.save_for_backward(*args)
            return forward(*args)

        @staticmethod
        def backward(ctx, prev_grad):
            args = ctx.saved_tensors
            
            # Get gradient for each input
            grads = backward_custom(prev_grad, *args)
            
            return grads
        
        
    if manual:
        # Which inputs require grads?
        requires_grads_ = [arg.requires_grad for arg in args]
        
        # Clone all input twice for each forward/backward pass
        args_ = [args[i].clone().detach().requires_grad_(requires_grads_[i]) for i in range(0, len(args))]
        args_custom = [args[i].clone().detach().requires_grad_(requires_grads_[i]) for i in range(0, len(args))]
        
        # Send through the forward function and backward pass
        forward(*args_).sum().backward()
        Function.apply(*args_custom).sum().backward()
        
        # Compare all gradients
        for i in range(0, len(args)):
            if not requires_grads_[i]:
                continue
            
            adiff = torch.abs(args_[i].grad - args_custom[i].grad)
            rdiff_ = torch.abs(args_[i].grad / args_custom[i].grad)
            rdiff = 1-torch.where(rdiff_ > 1, 1/rdiff_, rdiff_)
            max_adiff = adiff.max()
            max_rdiff = rdiff.max()
            assert torch.allclose(args_[i].grad, args_custom[i].grad)
    else:
        torch.autograd.gradcheck(Function.apply, args, eps=1e-4)
    
    
    
    
# Test tensors I will be using
B = 2
H = 2
N = 1024
d = 64
Q = torch.randn(B, H, N, d).cuda().detach().requires_grad_(True)
K = torch.randn(B, H, N, d).cuda().detach().requires_grad_(True)
V = torch.randn(B, H, N, d).cuda().detach().requires_grad_(True)
A = torch.randn(B, H, N).cuda().detach().requires_grad_(True)
mask = torch.tril(torch.ones(N, N)).bool()[None, None, ...].cuda().detach()
    


# Check forward and backward for plain linear attention
def forward_lin(Q, K, V, M):
    return (Q @ K.mT * M) @ V
def backward_lin(prev_grad, Q, K, V, M):
    dq = (prev_grad @ V.mT * M) @ K
    dk = (V @ prev_grad.mT * M.mT) @ Q
    dv = (Q @ K.mT * M).mT @ prev_grad
    return dq, dk, dv, None
# test_grads(
#     forward_lin,
#     backward_lin,
#     Q, K, V, mask, manual=True
# )


# Check forward and backward for plain linear attention with sm norm
def forward_lin_sm_norm(Q, K, V, M):
    Y = (Q @ K.mT * M)
    Y_N = Y / Y.sum(-1, keepdims=True)
    return Y_N @ V
def backward_lin_sm_norm(prev_grad, Q, K, V, M):
    Y = (Q @ K.mT * M)
    Y_N = Y / Y.sum(-1, keepdims=True)
    X = prev_grad @ V.mT
    S = (X * Y_N).sum(-1, keepdims=True)
    d_inner = (X - S) / Y.sum(-1, keepdims=True)
    dq = (d_inner * M) @ K
    dk = (d_inner * M).mT @ Q
    dv = Y_N.mT @ prev_grad
    return dq, dk, dv, None
# test_grads(
#     forward_lin_sm_norm,
#     backward_lin_sm_norm,
#     Q, K, V, mask, manual=True
# )
    




# Check forward and backward for plain linear attention
# with an A mask
def forward_lin_Amask(Q, K, V, A, M):
    A_M = (A[..., :, None] - A[..., None, :]).exp()
    return (Q @ K.mT * M * A_M) @ V
def backward_lin_Amask(prev_grad, Q, K, V, A, M):
    A_M = (A[..., :, None] - A[..., None, :]).exp()
    GV = prev_grad @ V.mT
    QK = Q @ K.mT
    GVAM = GV * A_M * M
    QKAM = QK * A_M * M
    dq = GVAM @ K
    dk = GVAM.mT @ Q
    dv = QKAM.mT @ prev_grad
    da_n = (QK * GVAM).sum(-1)
    da_m = -(QK * GVAM).sum(-2)
    da = da_n + da_m
    return dq, dk, dv, da, None
# test_grads(
#     forward_lin_Amask,
#     backward_lin_Amask,
#     Q, K, V, A, mask, manual=True
# )







# Check forward and backward for squared linear attention
# with an A mask
def forward_square_Amask(Q, K, V, A, M):
    A_M = (A[..., :, None] - A[..., None, :]).exp()
    return ((Q @ K.mT)**2 * M * A_M) @ V
def backward_square_Amask(prev_grad, Q, K, V, A, M):
    A_M = (A[..., :, None] - A[..., None, :]).exp()
    GV = prev_grad @ V.mT
    QK = Q @ K.mT
    QK2 = (QK)**2
    GVAM = GV * A_M * M
    _2QKGVAM = 2 * QK * GVAM
    QK2AM = QK2 * A_M * M
    dq = _2QKGVAM @ K
    dk = _2QKGVAM.mT @ Q
    dv = QK2AM.mT @ prev_grad
    da_n = (QK2 * GVAM).sum(-1)
    da_m = -(QK2 * GVAM).sum(-2)
    da = da_n + da_m
    return dq, dk, dv, da, None
test_grads(
    forward_square_Amask,
    backward_square_Amask,
    Q, K, V, A, mask, manual=True
)


print()



exit()

    




def forward(X, A, V, mask):
    M = (A[..., :, None] - A[..., None, :]).masked_fill(~mask, -torch.inf).exp()
    O = M * X
    return O @ V

def manual_grad(X, A, V, mask):
    X = X.clone().detach()
    A = A.clone().detach().requires_grad_()
    V = V.clone().detach()
    mask = mask.clone().detach()
    out = forward(X, A, V, mask)
    out.sum().backward()
    return A.grad

class Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, A, V, mask):
        ctx.save_for_backward(X, A, V, mask)
        return forward(X, A, V, mask)

    @staticmethod
    def backward(ctx, prev_grad):
        X, A, V, mask = ctx.saved_tensors
        
        M = (A[..., :, None] - A[..., None, :]).masked_fill(~mask, -torch.inf).exp()
        da = prev_grad @ V.mT
        vals = da * M * X
        
        A_grad = vals.sum(-1) - vals.sum(-2)
        
        return None, A_grad, None, None
    
    
    
    
    
    
    
    
    
N = 2
H = 2
S = 1024
d = 64

X = torch.randn(N, H, S, S).cuda().detach()
A = (-torch.nn.functional.softplus(torch.randn(N, H, S) * 0.1).cuda().cumsum(-1)).detach().requires_grad_()
V = torch.randn(N, H, S, d).cuda().detach()
mask = torch.tril(torch.ones(S, S)).bool()[None, None, ...].cuda().detach()




A_grad = manual_grad(X, A, V, mask)
A_grad

out = Function.apply(X, A, V, mask)
out.sum().backward()
A_grad_ = A.grad
A_grad_
(A_grad - A_grad_).abs().max()
torch.autograd.gradcheck(Function.apply, (X, A, V, mask), eps=1e-4)