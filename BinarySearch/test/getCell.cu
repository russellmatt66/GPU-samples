#include <string>
#include <math.h>

/*
Test the 'getCell()' function - it doesn't work
*/

__device__ int getNumIter(int i){
    int num_iter = 1;
    int k = 0;
    while (k < i){
        k += pow(2, num_iter);
        num_iter++;
    }
    return num_iter;
}

// With these, it is possible to write 'getCell()' and 'connectNodes()'
__device__ int getLeftChildInd(int i, int num_iter, int Nx){
    if (num_iter > (int)log2f(Nx)){
        return -2;
    }
    return i + pow(2, num_iter - 1);
}

__device__ int getRightChildInd(int i, int num_iter, int Nx){
    return getLeftChildInd(i, num_iter, Nx) + 1;
}

// I don't know how to write this function
__device__ int getCell(int i, int Nx, int num_iter, int k, int low, int high){
    // Calculate the grid-cell that is associated with the ith node
    int guess = (low + high) / 2;
    if (i == k){
        return guess;
    }
    else if (k == -1 || k == -2){
        return -1;
    }
    int left_high = guess;
    int right_low = guess;
    int left_guess = getCell(i, Nx, num_iter, getLeftChildInd(k, num_iter, Nx), low, left_high);
    int right_guess = getCell(i, Nx, num_iter, getRightChildInd(k, num_iter, Nx), right_low, high);
    if (left_guess != -1){
        guess = left_guess;
    }
    else if (right_guess != -1){
        guess = right_guess;
    }
    return guess;
}

__global__ void testGetCell(int* cell_values, int Nx){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    int cell = 0;
    int num_iter = 0;
    int low = 0, high = Nx - 1, k = 0;
    for (int i = tidx; i < Nx - 1; i += nthreads){ // number of nodes = Nx-1 = number of cells 
        num_iter = getNumIter(i);
        cell = getCell(i, Nx, num_iter, k, low, high); // I don't know how to write this
        cell_values[i] = cell;
    }
    return;
}

__global__ void initCellValues(int* cell_values, const int Nx){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    for (int j = tidx; j < Nx; j += nthreads){
        cell_values[j] = -10;
    }
}

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char* argv[]){
    int Nx = std::stoi(argv[1]);

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    int* cell_values;

    checkCuda(cudaMallocManaged(&cell_values, Nx*sizeof(int)));

    int num_blocks = numberOfSMs;
    int num_threads_per_block = 32;

    initCellValues<<<num_blocks, num_threads_per_block>>>(cell_values, Nx);
    checkCuda(cudaDeviceSynchronize());
    testGetCell<<<num_blocks, num_threads_per_block>>>(cell_values, Nx);
    checkCuda(cudaDeviceSynchronize());

    for (int j = 0; j < Nx; j++){
        printf("Node %d represents looking in cell %d\n", j, cell_values[j]);
    }

    cudaFree(cell_values);
    return 0;
}
