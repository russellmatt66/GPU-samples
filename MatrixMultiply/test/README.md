# Directory Structure
problem_size.py
- Determines bounds for problem size on which RTX 2060 can perform matrix multiply

validate_matmul.cu
- CUDA code that validates `matmul` routine
- `nvcc -o val_mm validate_matmul.cu`

val_mm
- Binary executable compiled from `validate_matmul.cu`
- `./val_mm N SM_multiplier_x SM_multiplier_y num_threads_per_blk_x num_threads_per_blk_y`