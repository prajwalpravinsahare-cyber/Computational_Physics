# Computational Physics Laboratory

This repository contains Python implementations of numerical algorithms, laboratory assignments, and computational models developed for the undergraduate Computational Physics Laboratory at the National Institute of Science Education and Research (NISER), Bhubaneswar. 

All core algorithms are centralized in a standalone library (`pj_Library.py`), which is imported across assignments to solve linear systems, integrate differential equations, find roots, and execute Monte Carlo simulations.

The listing below organizes the completed lab exercises, algorithms implemented in the library, and associated solution scripts.

---

## Numerical Linear Algebra

* Gaussian Elimination & Back Substitution 
* LU Decomposition (Doolittle / Crout algorithms)
* Cholesky Decomposition for Symmetric Positive-Definite Matrices 
* Jacobi & Gauss-Seidel Iterative Solvers 
* Power Iteration (Dominant Eigenvalues & Eigenvectors) 

---

## Nonlinear Equations & Root-Finding

* Bisection Method 
* Regula-Falsi (False Position) Method 
* Newton-Raphson Method 
* Fixed-Point Iteration 

---

## Numerical Integration & Quadrature

* Trapezoidal Rule 
* Simpson's 1/3 and 3/8 Rules 
* Linear Congruential Generator (LCG) Pseudo-Random Sampling 
* Monte Carlo Integration & Variance Analysis 

---

## Ordinary Differential Equations

* Euler and Modified Euler Methods 
* 4th-Order Runge-Kutta (RK4) Method 
* Coupled First-Order ODE Systems 
* Shooting Method for Boundary Value Problems 

---

## Data Fitting & Analysis

* Linear Regression and Error Estimation 
* Polynomial Least-Squares Fitting 
* Chi-Square ($\chi^2$) Minimization 

---

## Repository Structure

```text
├── pj_Library.py              # Central module containing all numerical routines
├── assignments/               # Weekly lab assignments and execution drivers
│   ├── week1_assignment_01/
│   │   ├── main.py
│   │   └── data.txt
│   └── ...
├── plots/                     # Output figures, phase space plots, error curves
├── requirements.txt           # Python dependencies
└── README.md
