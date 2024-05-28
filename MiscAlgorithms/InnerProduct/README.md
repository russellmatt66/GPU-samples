# Summary
Explore the difference in performance between a GPU and CPU using the calculation of the inner product of a vector.

# Run Instructions
inner.cu
- Compile with: nvcc -o innerCU inner.cu 
- Run on command-line with: $./innerCU lshift num_threads_per_block
-- 'lshift' = log2(N) - 1, where N is the length of the arrays whose inner product is to be calculated 

- Profile with: nsys profile --stats=true ./innerCU lshift num_threads_per_block

