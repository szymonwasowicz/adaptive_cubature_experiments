\# Adaptive Cubature on Simplices



This repository contains the numerical experiment software for the paper **"Adaptive integration of convex functions of multiple variables"** by **Andrzej Komisarski** and **Szymon Wąsowicz**. This is the research code for computing multiple integrals over n-dimensional simplices using \*\*Adaptive Cubature\*\* algorithms based on barycentric subdivision.



\## Project Overview

The project explores an adaptive strategy for numerical integration. Instead of a uniform partition, the algorithm recursively divides simplices only in regions where the function variation (and thus the estimation error) is high.



\### Key Features

\* \*\*n-Dimensional Support\*\*: Functions to integrate over any $n$-simplex.

\* \*\*Barycentric Subdivision\*\*: A robust method for partitioning simplices into $(n+1)!$ sub-simplices.

\* \*\*CPU \& GPU Implementations\*\*: Includes a standard NumPy version and a high-performance PyTorch/CUDA version for high-dimensional problems.

\* \*\*Visualization\*\*: 2D plotting tools to visualize the adaptive mesh refinement.



\## Files Description



\### 1. Core Functions

\* \*\*`adaptive\_cubature\_experiments.py`\*\*: The main library containing:

&nbsp;   \* `experiment\_ndim`: General adaptive cubature logic using NumPy.

&nbsp;   \* `experiment\_2dim`: Optimized 2D version.

&nbsp;   \* `experiment\_2dim\_plot`: Version with Matplotlib integration for mesh visualization.

\* \*\*`adaptive\_cubature\_experiments\_gpu.py`\*\*: A GPU-accelerated version of the n-dimensional algorithm using \*\*PyTorch\*\*. This version significantly speeds up computation for deep subdivision levels or high dimensions.



\### 2. Jupyter Notebooks (Frontend)

\* \*\*`adaptive\_cubature\_experiments.ipynb`\*\*: Demonstrates the CPU algorithms, shows convergence, and generates plots for 2D and 3D cases.

\* \*\*`adaptive\_cubature\_experiments\_gpu.ipynb`\*\*: Demonstrates the performance gains using GPU acceleration for 4D and higher-dimensional simplices.



\## Requirements

\* Python 3.x

\* NumPy, Matplotlib

\* PyTorch (for GPU acceleration)



\## Usage

Simply run the provided notebooks to replicate the experiments or import the functions into your own research scripts.



```python

from adaptive\_cubature\_experiments import experiment\_ndim

import numpy as np



\# Define a function and a simplex

f = lambda x: np.exp(-np.sum(x\*\*2, axis=-1))

S = np.array(\[\[0,0], \[1,0], \[0,1]]) # 2D Unit Simplex



experiment\_ndim(f, S, epsilon=1e-5)

