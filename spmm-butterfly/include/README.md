# Current Tasks
(1) Complete implementation of `spGEMM` kernel in `spGEMM.cu`
- `spGEMM` only needs to work on butterfly matrices
-- [Efficient GPU Memory Management for butterfly-spmm](https://arxiv.org/pdf/2405.15013v1)
-- [Primer on Butterfly Matrices](https://dawn.cs.stanford.edu/2019/06/13/butterfly/)

(2) Implement `initializeButterFlyMatrix` in `initCSR.cu`
- Validate creation for appropriate number of stages

(3) Implement pipeline for spmm chain which is `e = log_{2}(L)` stages deep
- Rough sketch:
-- Start with `B_{0}`, allocate `B_{1}`, and `C_{01}` in memory
-- Perform `C_{01} = B_{0} * B_{1}`
-- De-allocate `B_{0}`, and `B_{1}` (de-allocating `B_{0}` might need to wait until the end)
-- Allocate `B_{2}` and `C_{12}`
-- Perform `C_{12} = C_{01} * B_{2}`
-- De-allocate `B_{2}` and `C_{01}`
-- So-on, and so-forth, until the required depth
-- Write out `C_{e-2,e-1}`, then de-allocate 
-- Clean up all left-over memory

