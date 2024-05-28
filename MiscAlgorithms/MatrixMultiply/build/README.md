# Directory Structure
matmul
- Binary for `../src/main.cu`
- `./matmul N SM_x SM_y ntpb_x ntpb_y`
    - N: dimension of square matrices
    - SM_x: multiplies number of device SMs to get num_blocks in the x-direction of the grid
    - SM_y: multiplies number of device SMs to get num_blocks in the y-direction " " "
    - ntpb_x: number of threads per block in the x-direction of the block
    - ntpb_y: " " " " " " " y-direction " " "