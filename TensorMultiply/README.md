# Overview
Project to implement tensor-tensor multiplication for a square rank-3 tensor.

Matrix (rank-2 tensor) multiplication can be defined in exactly one way, but for a rank-3 tensor there is some ambiguity as to how the process should be done, i.e., which indices should be contracted. Regardless of the choice made, it will be true that the components of the output tensor, C_{ijk}, will be produced by taking an inner product between a linear dimension of one tensor, and a diagonal dimension of a sub-matrix of the other. 

For example, C_{ijk} = A_{ijl}B_{llk}, is one way to contract (multiply) the two tensors together. The Einstein summation convention applied here sums over repeated indices. What this means in the previous equation, is that we are contracting over the layer (3rd dimension) of A, and the (row/column)diagonal of the kth layer of B, in order to produce the ijk-th element of C.
  
# Current Tasks
(1) Implement code and timing in `./src/main.cu`

# Project Status
(1) Algorithm successfully implemented, and validated in `./test/validate.cu`

# Directory Structure