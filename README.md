# TerMinator 🤖
> **Ter**nary **M**atrix **I**terative **N**umerical **A**pproxima**tor**

[![NLA Project](https://img.shields.io/badge/Subject-Numerical_Linear_Algebra-purple)]()
[![Python](https://img.shields.io/badge/Python-3.x-blue)]()

## 🎯 Project Goal
This project investigates the "Rounding Problem" in 1.58-bit LLM quantization. We explore and benchmark different numerical strategies to compress high-precision matrices into ternary states $\lbrace-1, 0, 1\rbrace$ by solving the minimization problem:

$$ \min_{\alpha, Q} ||W - \alpha Q||_F^2 $$

Our goal is to find the optimal trade-off between **reconstruction accuracy** (Frobenius/Spectral norms) and **computational complexity**.

## 🧪 Methods & Experiments
We implement and compare a hierarchy of approximation strategies:

1. **Baseline (Naive):** One-pass "AbsMean" quantization (reproducing *BitNet b1.58*).
2. **ALS Optimization:** Iterative Coordinate Descent to strictly minimize error.
3. **Granularity:** Comparing Global Scaling ($\alpha$) vs. Row-wise Scaling ($\Lambda$).
4. **Spectral Initialization:** Using Power Method / SVD to de-noise $W$ before quantization.
5. **Outlier-Aware:** "Sparse + Quantized" decomposition for heavy-tailed distributions (Laplacian).

## 📂 Repository Structure
- `base.py` — **Interface Definitions.** Defines the abstract `BaseQuantizer` class to ensure all solvers follow a strict input/output contract.
- `solvers.py` — **Algorithms.** Implementations of the quantization strategies:
    - `NaiveQuantizer` (AbsMean / BitNet)
    - `ALSQuantizer` (Iterative solver with SVD init & row-wise support)
- `data.py` — **Data Pipeline.** Utilities to generate synthetic matrices (Gaussian/Laplacian) and load real-world weights from HuggingFace models.
- `metrics.py` — **NLA Analysis.** Mathematical functions to compute Relative Frobenius Error, Spectral Gap, and other precision metrics.
- `main.py` — **Benchmarks.** The main entry point to run experiments, compare solvers, and generate convergence reports.

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies (mainly `numpy`, `scipy`, `matplotlib`):
```bash
git clone https://github.com/your-team/TerMinator.git
cd TerMinator
pip install -r requirements.txt
```

### 2. Run Benchmarks
To reproduce our experiments and see the comparison between Naive and ALS methods:
```bash
python main.py
```
*This will generate synthetic data, run all solvers, and print the Frobenius error report.*

### 3. Usage API
You can use `TerMinator` components in your own scripts:

```python
import numpy as np
from solvers import ALSQuantizer

# 1. Create a high-precision matrix
W = np.random.randn(1024, 1024)

# 2. Initialize the solver (e.g., Row-wise ALS with SVD initialization)
solver = ALSQuantizer(max_iter=10, row_wise=True, init_svd=True)

# 3. Compress
Q, alpha, stats = solver.quantize(W)

print(f"Reconstruction Error: {stats['final_error']:.2%}")
```


> *"Hasta la vista, floating point."*
