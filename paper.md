---
title: 'nnde: A Python package for solving differential equations using neural networks'
tags:
  - Python
  - neural networks
  - differential equations
authors:
  - name: Eric Winter
    orcid: 0000-0001-5226-2107
    affiliation: George Mason University
  - name: Bob Weigel
    orcid: XXXX
    affiliation: George Mason University
date: 7 February 2021
bibliography: paper.bib
---

# Summary

Neural networks have been shown to have the ability to solve differential equations [@Yadav:2015]. The `nnde` software provides a pure-Python package for the solution of ordinary and partial differential equations of up to second order. We present results of sample runs showing the effectiveness of the software in solving the two-dimensional diffusion problem.

# Statement of need

The most commonly used methods for solving differential equations are the Finite Element Method (FEM) and Finite Difference Method (FDM). However, these methods can be complicated to parallelize and can involve large storage requirements for model outputs. The neural network method provides straightforward parallelization due to the independent characteristics of the computational nodes in each network layer. Additionally, the trained network solution is more compact than an FDM or FEM solution because the storage of only the network weights and biases are required. Additionally, the neural network solution is mesh-free and does not require interpolation to retrieve the solution at a non-grid point, as is the case with FDM or FEM. Once the network is trained, computing a solution requires only a series of simple matrix multiplications, one per network layer. The trained solution is a sum of arbitrary differentiable basis functions, and therefore the trained solution is also differentiable, which is particularly useful when computing derived quantities such as gradients and fluxes.

The `nnde` package provides a pure-Python implementation of one of the earliest approaches to using neural networks to solve differential equations - the trial function method [@Lagaris:1998]. It was initially developed primarily as a vehicle for understanding the internal workings of feedforward neural networks. It has since been enhanced to provide the capabilitiy to solve differential equations of scientific interest, such as the diffusion equation described here. The ultimate goal of the package is to provide the capability to solve systems of coupled partial differential equations, such as the equations of magnetohydrodynamics.

# Description

`nnde` is a pure-Python package that implements a version of the trial function algorithm described by [@Lagaris:1998]. This software also incorporates a modification of the trial function algorithm to automatically incorporate arbitrary Dirichlet boundary conditions of the problem directly into the neural network solution.

This software was developed as part of a project with the objective of demonstrating that the coupled partial differential equations of magnetohydrodynamics (MHD) [@Chen:1984] can be solved using the neural network method. As part of this ongoing project, we implemented many of the test problems in the literature and the result is a package can be used for any ordinary or partial differential equations of up to second order. `nnde` was initially developed entirely in Python to understand the workings of the neural network algorithm. Planned future work will incorporate the high-performance TensorFlow library to improve the speed of solutions.

onsider the diffusion equation in two dimensions as an example of the type of problem that can be solved with `nnde`. The PDE is

$$\frac {\partial G_i} {\partial p} = \frac {\partial G_i} {\partial \psi_{ti}} \frac {\partial \psi_{ti}} {\partial p} + \sum_{j=1}^m \frac {\partial G_i} {\partial \nabla_{ij} \psi_{ti}} \frac {\partial \nabla_{ij} \psi_{ti}} {\partial p} + \sum_{j=1}^m \frac {\partial G_i} {\partial \nabla_{ij}^2 \psi_{ti}} \frac {\partial \nabla_{ij}^2 \psi_{ti}} {\partial p}
  \label{diffusion2D}$$

If all boundaries are fixed at $0$ and with an initial condition of

$$\psi(x,y,0) = \sin(\pi x) \sin(\pi y)$$

the analytical solution is

$$\psi_a(\mathbf x) = e^{-2\pi^2 D t} \sin(\pi x) \sin(\pi y)$$

The `nnde` package was used to create a neural network with a single hidden layer and 10 hidden nodes and trained to solve this problem. The error in the trained solution for the case of $D=0.1$ is shown in \autoref{fig:diff2d_error}.

![Computed error in solution of 2-D diffusion problem using `nnde` with 10 nodes.\label{fig:diff2d_error}](figures/diff2d_error_heatmaps.png)

# Software repository

The `nnde` software is available on GitHub at https://github.com/elwinter/nnde.

A collection of example python scripts using `nnde`  is available on GitHub at https://github.com/elwinter/nnde_demos.

A collection of example Jupyter notebooks using `nnde` is available on GitHub at https://github.com/elwinter/nnde_notebooks.

# References
