# Quantum-Annealing Enhanced Machine Learning for Interpretable Phase Classification of High-Entropy Alloys

[![arXiv](https://img.shields.io/badge/arXiv-2507.10237v1-b31b1b.svg)](https://arxiv.org/abs/2507.10237)

This repository contains the code and data accompanying the paper:

> **Quantum-Annealing Enhanced Machine Learning for Interpretable Phase Classification of High-Entropy Alloys**  
> Diego Ibarra Hoyos, Gia-Wei Chern, Israel Klich, Joseph Poon  
> [arXiv:2507.10237](https://arxiv.org/abs/2507.10237)

Our work demonstrates a **quantum-enhanced machine learning (QML) framework** that integrates:

- **Quantum Boosting (QBoost)** for interpretable feature selection and phase classification.
- **Quantum Support Vector Machines (QSVM)** with quantum-enhanced kernels for capturing nonlinear structure–property relationships.

The pipeline is fully reproducible, hardware-agnostic (classical/quantum backends), and designed to accelerate **High-Entropy Alloy (HEA)** discovery.

> **Note:** Several scripts rely on execution on a **D-Wave quantum annealer** (Advantage or hybrid solvers).  
> You must have a valid **D-Wave Leap account and purchased compute time** for these parts of the code to run.  
> Classical fallback versions are included for reproducibility but will not reproduce the quantum-specific speedups.

---

## 📖 Overview

Crystallographic phase prediction is a key challenge in HEA design due to:
- **Small, imbalanced datasets**  
- **Nonlinear, high-dimensional feature spaces**
- **Computational bottlenecks** in iterative retraining (e.g., active learning)

Our solution formulates feature selection and classification as **Quadratic Unconstrained Binary Optimization (QUBO)** problems, solved using **D-Wave’s quantum annealer** (Advantage system, Pegasus topology) and hybrid solvers. We show:

- **10,000× runtime speedups** over simulated annealing baselines.
- **Improved or matched generalization** compared to classical models on experimentally validated HEAs.
- **Physically interpretable feature sets** mapping directly to phase-stability descriptors (VEC, Ω, η, etc.).

---

## 🛠️ Repository Structure

