#include <cmath>

// Locates all the items in *item_positions that are within the grid, using binary search.
__global__ void BinarySearchGPU(const float *grid, const int Nx, const float *item_positions, const int Ni, int *item_indices){
    /* Grid-Stride through array of item positions and call bs() for each element */
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    for (int i = tidx; i < Ni; i += nthreads){
        BinarySearchDevice(grid, Nx, item_positions[i], item_indices, i);
    }
}

// Implementation of Binary Search for use on the device
/* Do __device__ specified functions need to return void? */
__device__ void BinarySearchDevice(const float *grid, const int Nx, const float item_position, int *item_indices, const int i){
    /*
    N - size of grid 
    item_position - exactly where the object is located in the grid
    item_indices - the cells where the objects are in the grid
    i - the location in item_indices where the found index (j) should be placed
    */
    int low = 0;
    int high = Nx-1;
    int j = 0, counter = 0;

    while (low <= high){
        j = floor((low + high) / 2);
        counter++;
        if (grid[j] <= item_position && grid[j+1] > item_position){ // inside cell j
            item_indices[i] = j;
        }
        else if (item_position > grid[j]){ // item is to the right of cell j
            low = j+1;
        }
        else if (item_position < grid[j]){ // item is to the left of cell j
            high = j;
        }
        else if (counter >= sqrtf32(Nx)){ // It's not in the grid
            item_indices[i] = -1; 
        }
    }
}

// CPU wrapper for locating all the objects that are within the grid, using binary search. 
void BinarySearchCPU(const float *grid, const int Nx, const float *item_positions, const int Ni, int *item_indices){
    for (int i = 0; i < Ni; i++){
        item_indices[i] = BinarySearchHost(grid, Nx, item_positions[i]);
    }
}

// Host implementation of binary search
int BinarySearchHost(const float *grid, const int Nx, const float item_position){
    int low = 0;
    int high = Nx-1;
    int j = 0, counter = 0;

    while (low <= high){
        j = floor((low + high) / 2);
        counter++;
        if (grid[j] <= item_position && grid[j+1] > item_position){ // inside cell j
            return j;
        }
        else if (item_position > grid[j]){ // item is to the right of cell j
            low = j+1;
        }
        else if (item_position < grid[j]){ // item is to the left of cell j
            high = j;
        }
        else if (counter >= sqrt(N)){ // It's not in the grid
            return -1; 
        }
    }

    return -1; // Not found
}


int main(int argc, char* argv[]){
    /* Initialize grid, and objects inside grid */

    /* Set execution configuration and call CUDA kernel to find objects */

    /* Call CPU code */

    /* Free grid, and the other objects */
}