/*
Implement binary search for benchmarking
*/

// Binary search
int binarySearch(int* particles, int* grid, const int i, const int N, const int Nx){
    int low = 0, high = Nx-1;
    int guess;

    int particle_pos = particles[i];
    while (low <= high){
        guess = (low + high) / 2;
        if (particle_pos >= grid[guess] && particle_pos < grid[guess + 1]){
            return guess;
        }
        else if (particle_pos < grid[guess]){ // to left of guess
            high = guess;
        }
        else { // to right of guess
            low = guess;
        }
    }
    return -1;
}

int main(int argc, char* argv[]){
    int N = atoi(argv[1]);
    int Nx = atoi(argv[2]);

    int *particles, *grid, *found;

    particles = (int*)malloc(N*sizeof(int));
    grid = (int*)malloc(Nx*sizeof(Nx));
    found = (int*)malloc(N*sizeof(int));

    // Initialize particles: uniform, random

    // Initialize grid: uniform, cartesian

    // Initialize found: all -1

    // Find the particles
    for (int i = 0; i < N; i++){
        found[i] = binarySearch(particles, grid, i, N, Nx);
    }

    free(particles);
    free(grid);
    free(found);

    return 0;
}