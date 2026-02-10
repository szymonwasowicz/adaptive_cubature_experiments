import numpy as np
import time
from math import factorial
from itertools import permutations
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

'''
General n-dimensional adaptive cubature experiment.
Uses barycentric subdivision to recursively partition simplices where the 
estimated error exceeds the tolerance threshold.
'''
def experiment_ndim(
    f = lambda x: 1 / (1 + np.sum(x, axis=-1)),
    S = np.vstack([np.zeros(2), np.eye(2)]),
    epsilon = 0.0001,
    exact_integral = None,
):

    start_time = time.perf_counter()
    
    # S has shape (n+1, n)
    n_pts, dim = S.shape
    n = dim
    
    # Data initialization: (number_of_simplices, vertices, coordinates)
    simplices = S.astype(float).reshape(1, n + 1, dim)
    level = -1
    total_Q, total_error = 0.0, 0.0
    total_good_simplices = 0
    
    # Volume of n-dimensional simplex: |det(V_1-V_0, ..., V_n-V_0)| / n!
    # Your code used (n+1)*det, which is specific to your constant. 
    # Standard measure:
    current_vol = (n + 1) * np.abs(np.linalg.det(S[1:] - S[0]))
    
    factor = int(factorial(n + 1))
    
    # Pre-generating permutations for barycentric subdivision
    # Each new simplex is created by the means of vertex subsets
    perms = list(permutations(range(n + 1)))

    while len(simplices) > 0:
        level += 1
        current_vol /= factor
        
        # 1. Calculating centroids (barycenters) for each simplex
        # b has shape (N_simplices, dim)
        b = np.mean(simplices, axis=1)
        
        # 2. Function values at vertices and center
        # f_verts: (N_simplices, n+1)
        f_verts = np.apply_along_axis(f, -1, simplices)
        f_center = f(b)
        
        # M = mean at center * volume, T = mean at vertices * volume
        M = current_vol * f_center
        T = current_vol * np.mean(f_verts, axis=1)
        
        # Cubature formulas (generalized)
        Q = ((n + 2) * M + n * T) / (2 * n + 2)
        E = n * np.abs(T - M) / (2 * n + 2)
        
        # Logical masking: identify which simplices need further refinement
        keep_mask = E > (epsilon / (factor**level))
        done_mask = ~keep_mask
        
        # Accumulate results and count "good" (terminal) simplices
        num_done = np.sum(done_mask)
        total_good_simplices += num_done
        total_Q += np.sum(Q[done_mask])
        total_error += np.sum(E[done_mask])
        
        # Filter out terminal simplices
        simplices = simplices[keep_mask]
        
        if len(simplices) == 0 or level >= 15: # Safety limit for high dimensions
            break
            
        # --- BARYCENTRIC SUBDIVISION (Permutation Method) ---
        # For each simplex, we create (n+1)! new simplices.
        # New vertices are cumulative means (prefix sums) of the original vertices.
        
        num_active = len(simplices)
        new_simplices = []
        
        # Calculating "barycentric coordinates" of the new sub-simplices' vertices
        # For efficiency, we do this per simplex in a loop (or further vectorize)
        # Here: hybrid approach for readability and avoiding memory errors
        
        sub_simplices = []
        for p in perms:
            # Select vertices in specific order and calculate cumulative means
            # This generates sub-simplex vertices: V0, (V0+V1)/2, (V0+V1+V2)/3...
            ordered_verts = simplices[:, p, :]
            prefix_sums = np.cumsum(ordered_verts, axis=1)
            denominators = np.arange(1, n + 2).reshape(1, n + 1, 1)
            sub_simplices.append(prefix_sums / denominators)
            
        simplices = np.concatenate(sub_simplices, axis=0)

    # # --- STATISTICS ---
    # full_partition_count = factor**level
    # efficiency = (total_good_simplices / full_partition_count) * 100 if full_partition_count > 0 else 0
    
    # print(f"Dimension: {n} | Level: {level}")
    # print(f"Q: {total_Q:.8f} | Error: {total_error:.8f}")
    # print(f"Simplices: {total_good_simplices} | Efficiency: {efficiency:.4f}%")

    # --- FINAL STATISTICS ---
    # Total number of simplices in a full (non-adaptive) barycentric subdivision
    full_partition_count = factor**level
    percent_of_full = (total_good_simplices / full_partition_count) * 100
    
    print("_"*55)
    print(f"FINAL STATISTICS")
    if level >= 15:
        print("_"*55)
        print(f"Reached maximal 15th level.\nExecution halted to prevent memory exhaustion.")
    print("_"*55)
    print(f"Dimension:                  {dim}")
    if exact_integral == None:
        print(f"Exact integral:             N/A")
    else:
        print(f"Exact integral:             {exact_integral:.8f}")
    print(f"Cubature value (Q):         {total_Q:.8f}")
    print(f"Total Error:                {total_error:.8f}")
    print(f"Max Depth Level:            {level}")
    print(f"Total 'good' simplices:     {total_good_simplices}")
    print(f"Full partition:             {full_partition_count}")
    print(f"Efficiency (% of Full):     {percent_of_full:.4f}%")
    print("_"*55)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} s")
    print("_"*55)

'''
2-dimensional version of the above function
Permutations are made by hand, not generated
in 2 dimensions it is considerably faster
(for epsilon = 1e-7 about 100 times)
'''
def experiment_2dim(
    f = lambda x: 1 / (2 * (x[:, 1] + np.sqrt(3))),
    S = np.array([[-1, np.sqrt(3)], [0, 0], [1, np.sqrt(3)]]),
    epsilon = 0.0001,
    exact_integral = None,
):

    start_time = time.perf_counter()
    dimension = S.shape[1]
    
    # Data initialization
    simplices = S.astype(float).reshape(1, dimension + 1, dimension)
    level = -1
    total_Q, total_error = 0, 0
    total_good_simplices = 0  # Total count of "good" simplices found during the process
    volume = (dimension + 1) * np.linalg.det(S[1:] - S[0])
    # In each level of partitioning the number of simplices is increased ,,factor'' times
    factor = factorial(dimension + 1) 
    
    # --- MAIN ALGORITHM LOOP ---
    while len(simplices) > 0:
        level += 1
        volume /= factor
        
        # Vectorized calculations for all simplices at the current level
        A, B, C = simplices[:, 0, :], simplices[:, 1, :], simplices[:, 2, :]
        b = (A + B + C) / 3
        
        M, T = volume * f(b), volume * (f(A) + f(B) + f(C)) / 3
        Q, E = ((dimension + 2)*M + dimension*T) / (2*dimension + 2), dimension * np.abs(T - M) / (2*dimension + 2)
        
        # Logical masking: identify which simplices need further refinement
        keep_mask = E > (epsilon / factor**level)
        done_mask = ~keep_mask
        
        # Accumulate results and count "good" (terminal) simplices
        num_done = np.sum(done_mask)
        total_good_simplices += num_done
        
        total_Q += np.sum(Q[done_mask])
        total_error += np.sum(E[done_mask])
        
        # Filter out terminal simplices
        simplices = simplices[keep_mask]
        
        if len(simplices) == 0 or level >= 15: # Safety break for depth
            break
            
        # Vectorized barycentric subdivision into 6 smaller triangles
        A, B, C = simplices[:, 0, :], simplices[:, 1, :], simplices[:, 2, :]
        b = (A + B + C) / 3
        AB2, AC2, CB2 = (A+B)/2, (A+C)/2, (C+B)/2
        
        # Constructing new sets of vertices without for-loops
        s1 = np.stack([A, AB2, b], axis=1); s2 = np.stack([A, AC2, b], axis=1)
        s3 = np.stack([B, AB2, b], axis=1); s4 = np.stack([B, CB2, b], axis=1)
        s5 = np.stack([C, AC2, b], axis=1); s6 = np.stack([C, CB2, b], axis=1)
        
        # Combine into new simplices matrix
        simplices = np.vstack([s1, s2, s3, s4, s5, s6])
    
    # --- FINAL STATISTICS ---
    # Total number of simplices in a full (non-adaptive) barycentric subdivision
    full_partition_count = factor**level
    percent_of_full = (total_good_simplices / full_partition_count) * 100
    
    print("_"*55)
    print(f"FINAL STATISTICS")
    if level >= 15:
        print("_"*55)
        print(f"Reached 15th level.\nExecution halted to prevent memory exhaustion.")
    print("_"*55)
    print("_"*55)
    print(f"Dimension:                  {dimension}")
    if exact_integral == None:
        print(f"Exact integral:             {exact_integral}")
    else:
        print(f"Exact integral:             {exact_integral:.8f}")
    print(f"Cubature value (Q):         {total_Q:.8f}")
    print(f"Total Error:                {total_error:.8f}")
    print(f"Max Depth Level:            {level}")
    print(f"Total 'good' simplices:     {total_good_simplices}")
    print(f"Full partition:             {full_partition_count}")
    print(f"Efficiency (% of Full):     {percent_of_full:.4f}%")
    print("_"*55)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} s")
    print("_"*55)

'''
2-dimensional experiment with plotting triangles
'''
def experiment_2dim_plot(
    f = lambda x: 1 / (2 * (x[:, 1] + np.sqrt(3))),
    S = np.array([[-1, np.sqrt(3)], [0, 0], [1, np.sqrt(3)]]),
    epsilon = 0.0001,
    exact_integral = None,
    picname = 'Cubature_Q.png'
):
    # --- CONFIGURATION ---
    # Initial triangle vertices
    max_display_levels = 15
    
    # Colormap setup - 'gist_ncar' offers high contrast for deep levels
    cmap_base = plt.get_cmap('gist_ncar')
    # Factor 0.7 means colors will be 30% darker
    darken_factor = 0.8
    
    colors_list = [
        (r * darken_factor, g * darken_factor, b * darken_factor, a) 
        for r, g, b, a in [cmap_base(i) for i in np.linspace(0, 0.9, max_display_levels)]
    ]
    custom_cmap = ListedColormap(colors_list)
    
    # Plot initialization
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_aspect('equal')
    
    start_time = time.perf_counter()
    
    # Data initialization
    simplices = S.astype(float).reshape(1, 3, 2)
    level = -1
    total_Q, total_error = 0, 0
    total_good_simplices = 0  # Total count of "good" simplices found during the process
    volume = 3 * np.linalg.det(S[1:] - S[0])
    
    # --- MAIN ALGORITHM LOOP ---
    while len(simplices) > 0:
        level += 1
        volume /= 6
        
        # Vectorized calculations for all simplices at the current level
        A, B, C = simplices[:, 0, :], simplices[:, 1, :], simplices[:, 2, :]
        b = (A + B + C) / 3
        
        M = volume * f(b)
        T = volume * (f(A) + f(B) + f(C)) / 3
        Q, E = (2*M + T)/3, np.abs(T - M)/3
        
        # Efficient drawing using PolyCollection
        coll = PolyCollection(simplices, 
                              edgecolors=colors_list[level % max_display_levels], 
                              facecolors='lightgray', linewidths=1, alpha=1)
        ax.add_collection(coll)
        
        # Logical masking: identify which simplices need further refinement
        keep_mask = E > (epsilon / 6**level)
        done_mask = ~keep_mask
        
        # Accumulate results and count "good" (terminal) simplices
        num_done = np.sum(done_mask)
        total_good_simplices += num_done
        
        total_Q += np.sum(Q[done_mask])
        total_error += np.sum(E[done_mask])
        
        # Filter out terminal simplices
        simplices = simplices[keep_mask]
        
        if len(simplices) == 0 or level >= max_display_levels: # Safety break for depth
            break
            
        # Vectorized barycentric subdivision into 6 smaller triangles
        A, B, C = simplices[:, 0, :], simplices[:, 1, :], simplices[:, 2, :]
        b = (A + B + C) / 3
        AB2, AC2, CB2 = (A+B)/2, (A+C)/2, (C+B)/2
        
        # Constructing new sets of vertices without for-loops
        s1 = np.stack([A, AB2, b], axis=1); s2 = np.stack([A, AC2, b], axis=1)
        s3 = np.stack([B, AB2, b], axis=1); s4 = np.stack([B, CB2, b], axis=1)
        s5 = np.stack([C, AC2, b], axis=1); s6 = np.stack([C, CB2, b], axis=1)
        
        # Combine into new simplices matrix
        simplices = np.vstack([s1, s2, s3, s4, s5, s6])
    
    # --- FINAL STATISTICS ---
    # Total number of simplices in a full (non-adaptive) barycentric subdivision
    full_partition_count = 6**level 
    percent_of_full = (total_good_simplices / full_partition_count) * 100
    
    print("_"*55)
    print(f"FINAL STATISTICS")
    if level >= max_display_levels:
        print("_"*55)
        print(f"Reached 15th level.\nExecution halted to prevent memory exhaustion.")
    print("_"*55)
    if exact_integral == None:
        print(f"Exact integral:             {exact_integral}")
    else:
        print(f"Exact integral:             {exact_integral:.8f}")
    print(f"Cubature value (Q):         {total_Q:.8f}")
    print(f"Total Error:                {total_error:.8f}")
    print(f"Max Depth Level:            {level}")
    print(f"Total 'good' simplices:     {total_good_simplices}")
    print(f"Full partition:             {full_partition_count}")
    print(f"Efficiency (% of Full):     {percent_of_full:.4f}%")
    print("_"*55)
    
    # --- LEGEND AND FINISHING ---
    # Discrete Colorbar matching the levels
    bounds = np.arange(level + 2)
    norm = BoundaryNorm(bounds, custom_cmap.N)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, ticks=bounds[:-1] + 0.5, fraction=0.046, pad=0.04)
    cbar.set_ticklabels([f'Lvl {i}' for i in range(level + 1)])
    cbar.set_label('Barycentric subdivision depth', fontsize=16)
    
    ax.autoscale_view()
    ax.set_title(f'Adaptive cubature', fontsize=22)
    
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} s")
    print("_"*55)
    
    # Save high-quality output for scientific publication
    plt.savefig(picname, bbox_inches='tight')
    plt.show()