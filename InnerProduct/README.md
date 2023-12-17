# Summary
Explore the difference in performance between a GPU and CPU using the calculation of the inner product of a vector.

# Run Instructions
inner.cu
- Compile with: 
- Run on command-line with: $./innerCU lshift num_blocks num_threads_per_block
-- 'lshift' is log2(N) - 1, where N is the length of the array whose inner product is to be calculated 
