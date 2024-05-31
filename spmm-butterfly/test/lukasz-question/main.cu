// Kernel for multiplying layer of N inputs with 2N weights

__global__ void firstSliceMultiply(float* output, const float* layer, const float* weights, const int N){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int xthreads = blockDim.x * gridDim.x;

    int first = tidx - (tidx % 2);
    int second = first + 1;

    /* Compute fidx, sidx, widx_one, widx_two */
    if (tidx < N){
        output[tidx] = layer[first] * weights[tidx] + layer[second] * weights[tidx + N]; 
    }

    return;
}