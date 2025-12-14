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

1.  **Baseline (Naive):** One-pass "AbsMean" quantization (reproducing *BitNet b1.58*).
2.  **ALS Optimization:** Iterative Coordinate Descent to strictly minimize error.
3.  **Granularity:** Comparing Global Scaling ($\alpha$) vs. Row-wise Scaling ($\Lambda$).
4.  **Spectral Initialization:** Using Power Method / SVD to de-noise $W$ before quantization.
5.  **Outlier-Aware:** "Sparse + Quantized" decomposition for heavy-tailed distributions (Laplacian).

## 📂 Repository Structure
*   `solvers.py` — Implementation of Quantizers (Naive, ALS, Row-wise).
*   `data_loader.py` — Synthetic generator (Gaussian/Laplacian) and Real LLM weights loader.
*   `analysis.py` — NLA metrics: Frobenius error, Singular Value/Spectral analysis.
*   `experiments.py` — Main benchmarking loop.

## 🚀 Quick Start
```bash
# Run the comparative benchmark
python experiments.py
```

> *"Hasta la vista, floating point."*
