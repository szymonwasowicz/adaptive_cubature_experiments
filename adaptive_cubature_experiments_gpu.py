import torch
import time
from math import factorial
from itertools import permutations
import numpy as np

# Check for CUDA availability to leverage GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU unavailable, using CPU.")

'''
GPU-accelerated version of the n-dimensional adaptive cubature experiment.
This function implements the same logic as experiment_ndim but uses PyTorch 
tensors and CUDA kernels to parallelize the subdivision and evaluation of thousands
of simplices simultaneously.
'''
def experiment_gpu(
    f = lambda x: 1 / (1 + torch.sum(x, dim=-1)),
    S = np.vstack([np.zeros(2), np.eye(2)]),
    epsilon=0.0001,
    exact_integral=None,
):
    start_time = time.perf_counter()
    
    # Convert S to a GPU tensor
    S = torch.as_tensor(S, dtype=torch.float64, device=device)
    n_pts, dim = S.shape
    n = dim
    
    simplices = S.unsqueeze(0) # (1, n+1, n)
    level = -1
    total_Q, total_error = 0.0, 0.0
    total_good_simplices = 0
    
    # Calculate initial volume
    v_diff = S[1:] - S[0]
    current_vol = (n + 1) * torch.abs(torch.linalg.det(v_diff))
    factor = factorial(n + 1)
    
    # Prepare permutation indices for GPU
    perms = torch.tensor(list(permutations(range(n + 1))), device=device)

    while simplices.shape[0] > 0:
        level += 1
        current_vol /= factor
        
        # 1. Centroids and function values
        b = torch.mean(simplices, dim=1)
        f_verts = f(simplices) # Assuming f handles tensors (N, n+1, dim)
        f_center = f(b.unsqueeze(1)).squeeze(1)
        
        M = current_vol * f_center
        T = current_vol * torch.mean(f_verts, dim=1)
        
        # Cubature formulas
        Q = ((n + 2) * M + n * T) / (2 * n + 2)
        E = n * torch.abs(T - M) / (2 * n + 2)
        
        # Logical masking
        keep_mask = E > (epsilon / (factor**level))
        done_mask = ~keep_mask
        
        total_good_simplices += torch.sum(done_mask).item()
        total_Q += torch.sum(Q[done_mask]).item()
        total_error += torch.sum(E[done_mask]).item()
        
        simplices = simplices[keep_mask]
        
        if simplices.shape[0] == 0 or level >= 15:
            break
            
        # Barycentric Subdivision (GPU-optimized)
        # Create all new simplices at once
        ordered_verts = simplices[:, perms, :] # (N, (n+1)!, n+1, dim)
        prefix_sums = torch.cumsum(ordered_verts, dim=2)
        denominators = torch.arange(1, n + 2, device=device, dtype=torch.float64).view(1, 1, n + 1, 1)
        
        new_simplices = prefix_sums / denominators
        simplices = new_simplices.view(-1, n + 1, n)

    full_count = factor**level
    print("_"*55)
    print(f"FINAL STATISTICS (GPU)")
    if level >= 15:
        print("_"*55)
        print(f"Reached maximal 15th level.\nExecution halted to prevent memory exhaustion.")
    print("_"*55)
    print(f"Dimension:                  {n}")
    if exact_integral == None:
        print(f"Exact integral:             N/A")
    else:
        print(f"Exact integral:             {exact_integral:.8f}")
    print(f"Cubature value (Q):         {total_Q:.8f}")
    print(f"Total Error:                {total_error:.8f}")
    print(f"Max Depth Level:            {level}")
    print(f"Total 'good' simplices:     {total_good_simplices}")
    print(f"Full partition:             {full_count}")
    print(f"Efficiency (% of Full):     {(total_good_simplices/full_count)*100:.4f}%")
    print("_"*55)
    print(f"Execution time: {time.perf_counter() - start_time:.4f} s")
    print("_"*55)