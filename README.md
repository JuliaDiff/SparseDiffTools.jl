# SparseDiffTools.jl

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/JuliaDiffEq/SparseDiffTools.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/SparseDiffTools.jl)

A Julia package to exploit the sparsity found in derivative matrices to enhance and speed up their computation with the help of matrix coloring. Jacobians of large dimensions frequently have a lot of elements equal to zero, and this fact can be utilised to speed up the process of computing such matrices. This package is fundamentally based and works on this observation.

This package comprises of three separate and independent modules: `Automatic Sparsity Detection`, `Matrix Partitioning`, and `Automatic Differentiation and Recovery`.

Under automatic sparsity detection, there are 2 main methods: 

i. Partial Computation of Jacobians

ii. Analytical Method


Under matrix partitioning, we have 2 sub categories:

i. distance-1-coloring, under which we have
  1. Recursive-Largest First Algorithm (`contraction_algo.jl`)
  2. Backtracking Sequential Algorithm (`BSC_algo.jl`)
  3. Largest-First Algorithm
  4. DSATUR Algorithm
  
ii. distance-2-coloring
  1. greedy distance-2-coloring algorithm
  2. greedy star coloring algorithm
  
 Under automatic recovery module, I plan on implementing 2 direct recovery methods:
 
 i. Automatic Differentiation Recovery
 
 ii. Forward Differencing Recovery
